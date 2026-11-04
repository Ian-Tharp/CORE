"""Verify an agent's action claims against the tool calls it actually made.

An LLM step can say "I wrote config.yaml" or "I committed the fix" without any
tool having run. This module pulls first-person, completed-action claims out of
text and checks each one against a tool-call trace (``ToolCallRecord`` list).
A claim is *supported* only when a successful trace entry of the right tool,
action and (when the claim names one) target exists.

Deterministic regexes, no LLM, no IO. Recall is deliberately modest: phrasings
it does not recognise are not checked at all, so a clean report means "no
recognised claim is unsupported", not "every sentence is true". Plans and
intentions ("I will write ...") are not claims.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence

from app.evaluation.models import ToolCallRecord

ClaimKind = Literal[
    "file_write",  # wrote / created / saved / deleted a file
    "file_inspect",  # read / opened / listed / searched files
    "git_read",  # ran git status / log / diff ...
    "git_write",  # committed / pushed / merged ... (no tool can do this today)
    "web_fetch",  # fetched / visited / downloaded a URL
    "database",  # queried / updated the database
    "command",  # ran tests / a script / a shell command (no tool can do this)
]

_FILE_INSPECT_ACTIONS = {"read", "list", "search"}
_GIT_READ_ACTIONS = {"status", "diff", "log", "show", "branch", "ls-files"}

# First-person subject, optionally perfect tense: "I", "I've", "I have".
_I = r"\bI(?:'ve|\s+have)?\s+(?:also\s+|now\s+|successfully\s+|just\s+)?"
# Clause text after the verb. A dot only ends it when followed by whitespace or
# end of text, so "config/app.yaml" and "example.com/a" stay whole.
_CLAUSE = r"(?:[^.\n!?;]|\.(?=[^\s.]))"
_REST = r"(?P<rest>" + _CLAUSE + r"{0,160})"

_PATH_RE = re.compile(r"[`'\"]?((?:[\w.-]+/)*[\w-]+\.[A-Za-z0-9]{1,8})\b[`'\"]?")
_URL_RE = re.compile(r"https?://[^\s`'\")>\]]+")
_FILEISH_RE = re.compile(r"\b(?:file|files|folder|directory|directories)\b", re.I)
_WEBISH_RE = re.compile(r"\b(?:url|page|website|site|web|online|internet)\b", re.I)
_DBISH_RE = re.compile(r"\b(?:database|db|table|tables|records?|rows?)\b", re.I)
_CMDISH_RE = re.compile(
    r"`|\b(?:tests?|test\s+suite|pytest|npm|script|command|shell|build|linter)\b",
    re.I,
)
_GIT_SUB_RE = re.compile(r"\bgit\s+([a-z-]+)", re.I)

# (kind, pattern). Order matters only for readability; every match is kept.
_PATTERNS = [
    (
        "file_write",
        re.compile(
            _I + r"(?:wrote|written|created|saved|updated|modified|edited|"
            r"overwrote|overwritten|deleted|removed|appended(?:\s+to)?)\b" + _REST,
            re.I,
        ),
    ),
    (
        "file_write",
        re.compile(
            r"\bsuccessfully\s+(?:wrote|written|saved|created)\s+"
            r"(?:to\s+)?(?P<rest>" + _CLAUSE + r"{0,120})",
            re.I,
        ),
    ),
    (
        "file_write",
        re.compile(
            r"(?P<rest>" + _CLAUSE + r"{0,80})\b(?:has|have|was|were)\s+(?:been\s+)?"
            r"(?:written|created|saved|updated|modified|deleted)\b",
            re.I,
        ),
    ),
    (
        "file_inspect",
        re.compile(
            _I + r"(?:read|opened|reviewed|inspected|examined|listed|searched|"
            r"scanned|looked\s+(?:at|through))\b" + _REST,
            re.I,
        ),
    ),
    (
        "git_write",
        re.compile(
            _I + r"(?:committed|pushed|merged|rebased|reverted|tagged|staged|"
            r"cherry-picked|checked\s+out|created\s+(?:a\s+|the\s+|new\s+)*branch)\b"
            + _REST,
            re.I,
        ),
    ),
    (
        "web_fetch",
        re.compile(
            _I + r"(?:fetched|downloaded|visited|browsed|retrieved|scraped|"
            r"accessed|searched\s+the\s+web|looked\s+up)\b" + _REST,
            re.I,
        ),
    ),
    (
        "database",
        re.compile(
            _I + r"(?:queried|inserted|updated|deleted\s+from|wrote\s+to)\b" + _REST,
            re.I,
        ),
    ),
    (
        "command",
        re.compile(
            _I + r"(?:ran|executed|(?:(?<=have\s)|(?<='ve\s))run)\b" + _REST, re.I
        ),
    ),
]

_CONDITIONAL_RE = re.compile(r"\b(?:if|once|when|after|unless|until)\s*$", re.I)


@dataclass
class ClaimCheck:
    kind: str
    text: str
    target: Optional[str] = None
    detail: Optional[str] = None  # e.g. git subcommand
    supported: bool = False
    evidence: Optional[str] = None


@dataclass
class ClaimReport:
    checks: List[ClaimCheck] = field(default_factory=list)

    @property
    def unsupported(self) -> List[ClaimCheck]:
        return [c for c in self.checks if not c.supported]

    def is_clean(self) -> bool:
        return not self.unsupported

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checked": len(self.checks),
            "unsupported": [asdict(c) for c in self.unsupported],
            "supported": [asdict(c) for c in self.checks if c.supported],
        }


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def _first_path(rest: str) -> Optional[str]:
    if _URL_RE.search(rest):
        rest = _URL_RE.sub(" ", rest)
    m = _PATH_RE.search(rest)
    return m.group(1) if m else None


def _classify(kind: str, rest: str) -> Optional[ClaimCheck]:
    """Turn a raw match into a claim, or None if the object isn't checkable."""
    if kind in ("file_write", "file_inspect"):
        # "I updated the database" is a db claim, not a file claim.
        if (
            kind == "file_write"
            and _DBISH_RE.search(rest)
            and not _PATH_RE.search(rest)
        ):
            return ClaimCheck(kind="database", text="")
        url = _URL_RE.search(rest)
        if url and kind == "file_inspect":
            return ClaimCheck(kind="web_fetch", text="", target=url.group(0))
        path = _first_path(rest)
        if path is None and not _FILEISH_RE.search(rest):
            if kind == "file_inspect" and _GIT_SUB_RE.search(rest):
                return _classify("command", rest)
            return None  # "I created a plan", "I reviewed the options"
        return ClaimCheck(kind=kind, text="", target=path)
    if kind == "git_write":
        return ClaimCheck(kind=kind, text="")
    if kind == "web_fetch":
        url = _URL_RE.search(rest)
        if url:
            return ClaimCheck(kind=kind, text="", target=url.group(0))
        return ClaimCheck(kind=kind, text="") if _WEBISH_RE.search(rest) else None
    if kind == "database":
        return ClaimCheck(kind=kind, text="") if _DBISH_RE.search(rest) else None
    if kind == "command":
        git = _GIT_SUB_RE.search(rest)
        if git:
            sub = git.group(1).lower()
            if sub in _GIT_READ_ACTIONS:
                return ClaimCheck(kind="git_read", text="", detail=sub)
            return ClaimCheck(kind="git_write", text="", detail=sub)
        return ClaimCheck(kind=kind, text="") if _CMDISH_RE.search(rest) else None
    return None


def extract_claims(text: Optional[str]) -> List[ClaimCheck]:
    """Find completed-action claims in ``text`` (unverified, ``supported=False``)."""
    if not text:
        return []
    norm = str(text).replace("’", "'").replace("‘", "'")
    out: List[ClaimCheck] = []
    seen = set()
    for kind, pat in _PATTERNS:
        for m in pat.finditer(norm):
            if _CONDITIONAL_RE.search(norm[max(0, m.start() - 8) : m.start()]):
                continue  # "if I wrote ...", "once I've pushed ..."
            claim = _classify(kind, m.group("rest") or "")
            if claim is None:
                continue
            claim.text = m.group(0).strip()
            key = (claim.kind, claim.target, claim.detail, m.start() // 200)
            if key in seen:
                continue
            seen.add(key)
            out.append(claim)
    return out


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def _succeeded(call: ToolCallRecord) -> bool:
    if call.error:
        return False
    res = call.result
    if isinstance(res, dict) and res.get("status") == "error":
        return False
    return True


def _norm_path(p: str) -> str:
    p = p.strip().strip("`'\"").replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p.casefold()


def _paths_of(call: ToolCallRecord) -> List[str]:
    paths: List[str] = []
    for src in (
        call.params or {},
        call.result if isinstance(call.result, dict) else {},
    ):
        v = src.get("path")
        if isinstance(v, str) and v:
            paths.append(v)
    if isinstance(call.result, dict):
        for key in ("entries", "matches"):
            vals = call.result.get(key)
            if isinstance(vals, list):
                paths.extend(str(x) for x in vals)
    return paths


def _path_matches(claimed: str, actual: str) -> bool:
    a, b = _norm_path(claimed), _norm_path(actual).rstrip("/")
    return a == b or b.endswith("/" + a) or a.endswith("/" + b)


def _action(call: ToolCallRecord) -> Optional[str]:
    v = (call.params or {}).get("action")
    return str(v) if v is not None else None


def _supports(claim: ClaimCheck, call: ToolCallRecord) -> bool:
    if not _succeeded(call):
        return False
    tool, action = call.name, _action(call)
    if claim.kind == "file_write":
        if tool != "file_operations" or action != "write":
            return False
    elif claim.kind == "file_inspect":
        if tool != "file_operations" or (action or "list") not in _FILE_INSPECT_ACTIONS:
            return False
    elif claim.kind == "git_read":
        if tool != "git":
            return False
        return claim.detail is None or (action or "status") == claim.detail
    elif claim.kind == "git_write":
        # The builtin git tool is read-only, so this only passes for a future
        # (e.g. MCP) git tool that actually ran the write action.
        return tool == "git" and action is not None and action not in _GIT_READ_ACTIONS
    elif claim.kind == "web_fetch":
        if tool != "web_research":
            return False
        if claim.target:
            url = str((call.params or {}).get("url", ""))
            final = (
                str(call.result.get("url", "")) if isinstance(call.result, dict) else ""
            )
            want = claim.target.rstrip("/").casefold()
            return any(
                u.rstrip("/").casefold().startswith(want) for u in (url, final) if u
            )
        return True
    elif claim.kind == "database":
        return tool == "database"
    elif claim.kind == "command":
        return False  # no shell/test-runner tool exists in the dispatcher
    else:
        return False

    if claim.target:
        return any(_path_matches(claim.target, p) for p in _paths_of(call))
    return True


def _describe(call: ToolCallRecord) -> str:
    action = _action(call)
    target = (call.params or {}).get("path") or (call.params or {}).get("url") or ""
    return " ".join(x for x in (call.name, action or "", str(target)) if x)


def verify_claims(text: Optional[str], trace: Sequence[ToolCallRecord]) -> ClaimReport:
    """Check every recognised action claim in ``text`` against ``trace``."""
    checks = extract_claims(text)
    for claim in checks:
        for call in trace:
            if _supports(claim, call):
                claim.supported = True
                claim.evidence = _describe(call)
                break
    return ClaimReport(checks=checks)


def verify_texts(
    texts: Iterable[Optional[str]], trace: Sequence[ToolCallRecord]
) -> ClaimReport:
    report = ClaimReport()
    for t in texts:
        report.checks.extend(verify_claims(t, trace).checks)
    return report


# --------------------------------------------------------------------------
# COREState adapters (duck-typed, like capture.py)
# --------------------------------------------------------------------------


def trace_from_state(state: Any) -> List[ToolCallRecord]:
    """Tool calls the dispatcher actually ran (``StepResult.dispatched_tool``).

    Planned-but-not-dispatched steps (LLM-executed, dry-run, blocked) are not in
    the trace: a plan saying "use file_operations" is not evidence it ran.
    """
    plan = getattr(state, "plan", None)
    params_by_id = {
        getattr(s, "id", None): dict(getattr(s, "params", None) or {})
        for s in (getattr(plan, "steps", None) or [])
    }
    trace: List[ToolCallRecord] = []
    for r in getattr(state, "step_results", None) or []:
        tool = getattr(r, "dispatched_tool", None)
        if not tool:
            continue
        trace.append(
            ToolCallRecord(
                name=tool,
                params=params_by_id.get(getattr(r, "step_id", None), {}),
                result=dict(r.outputs) if getattr(r, "outputs", None) else None,
                error=getattr(r, "error", None),
            )
        )
    return trace


def agent_texts_from_state(state: Any) -> List[str]:
    """LLM-authored step outputs: the text that can carry claims.

    Dispatched tool output is excluded, since a file or web page saying "I wrote
    X" is data the agent read, not a claim the agent made.
    """
    texts: List[str] = []
    for r in getattr(state, "step_results", None) or []:
        if getattr(r, "dispatched_tool", None):
            continue
        out = getattr(r, "outputs", None) or {}
        for key in ("result", "output", "content"):
            v = out.get(key)
            if isinstance(v, str) and v.strip():
                texts.append(v)
    return texts


def verify_state_claims(state: Any) -> ClaimReport:
    return verify_texts(agent_texts_from_state(state), trace_from_state(state))
