"""Run the trace-vs-claim verifier (``claims.py``) as a Claude Code ``Stop`` hook.

Claude Code calls this when the agent is about to end its turn. Stdin is the
hook payload (``transcript_path``, ``session_id``, ``stop_hook_active``, ...).
The hook reads the session transcript (JSONL), turns every tool call into a
``ToolCallRecord``, pulls the agent's own text for the current turn, and runs
``verify_texts``. If a completed-action claim ("I committed and pushed",
"I ran the tests", "I wrote config.yaml") has no matching successful tool call,
it prints ``{"decision": "block", "reason": ...}`` so the agent must either do
the work or correct its message before stopping.

Mapping Claude Code tools onto the names ``claims.py`` understands:

- ``Write`` / ``Edit`` / ``MultiEdit`` / ``NotebookEdit`` -> ``file_operations`` write
- ``Read`` / ``Glob`` / ``Grep`` / ``LS`` -> ``file_operations`` read / list / search
- ``WebFetch`` / ``WebSearch`` -> ``web_research``
- ``Bash`` -> one ``bash`` record, plus derived ``git`` (per subcommand),
  ``file_operations`` (rm, mv, cp, touch, ``>`` ...) and ``database`` records
- GitHub MCP writes (push_files, create_or_update_file, ...) -> ``git`` commit/push

``claims.py`` never supports ``command`` claims (CORE has no shell tool), and
accepts any git write for "committed and pushed". Here a shell exists, so two
re-checks run after ``verify_texts``: command claims are matched against Bash
calls (a test claim needs a successful test-runner command), and git-write
claims need every verb they name (commit *and* push).

The hook fails open: any internal error exits 0 without blocking. It blocks at
most once per stop (``stop_hook_active`` is respected), so it cannot loop.

Environment:
    CORE_CLAIM_HOOK_MODE      ``block`` (default), ``warn`` (show a message to
                              the user, don't block) or ``off``.
    CORE_EVAL_CAPTURE_FAULTS  When on, an unsupported claim is also appended to
                              ``faults.jsonl`` as a ``FaultReport`` (see
                              ``app/config/evaluation.py``).

Usage (hook): ``python -m app.evaluation.claude_code_hook`` with the payload on
stdin. Usage (audit): ``python -m app.evaluation.claude_code_hook --transcript
path/to/session.jsonl`` prints the claim report as JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from app.evaluation.claims import ClaimCheck, ClaimReport, verify_texts
from app.evaluation.models import FaultReport, StepRecord, ToolCallRecord

logger = logging.getLogger(__name__)

HOOK_SOURCE = "claude_code_stop_hook"
_MODES = ("block", "warn", "off")

# --------------------------------------------------------------------------
# Transcript reading
# --------------------------------------------------------------------------


def load_transcript(path: str | Path) -> List[Dict[str, Any]]:
    """Parse a Claude Code session transcript, skipping unparseable lines."""
    entries: List[Dict[str, Any]] = []
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            entries.append(obj)
    return entries


def _blocks(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    content = (entry.get("message") or {}).get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def _is_human_prompt(entry: Dict[str, Any]) -> bool:
    """A user entry typed by a person (not a tool result, meta note or summary)."""
    if entry.get("type") != "user" or entry.get("isSidechain"):
        return False
    if entry.get("isMeta") or entry.get("isCompactSummary"):
        return False
    if "toolUseResult" in entry:
        return False
    blocks = _blocks(entry)
    if any(b.get("type") == "tool_result" for b in blocks):
        return False
    return any(
        b.get("type") == "text" and str(b.get("text", "")).strip() for b in blocks
    )


def _current_turn_start(entries: Sequence[Dict[str, Any]]) -> int:
    for i in range(len(entries) - 1, -1, -1):
        if _is_human_prompt(entries[i]):
            return i + 1
    return 0


def last_user_prompt(entries: Sequence[Dict[str, Any]]) -> str:
    start = _current_turn_start(entries)
    if start == 0:
        return ""
    return "\n".join(
        str(b.get("text", ""))
        for b in _blocks(entries[start - 1])
        if b.get("type") == "text"
    ).strip()


_FENCE_RE = re.compile(r"```.*?(?:```|\Z)", re.S)


def turn_texts(entries: Sequence[Dict[str, Any]]) -> List[str]:
    """The agent's own text since the last human prompt (main thread only).

    Fenced code blocks are dropped: a quoted log saying "I committed" is data,
    not the agent's claim.
    """
    texts: List[str] = []
    for entry in entries[_current_turn_start(entries) :]:
        if entry.get("type") != "assistant" or entry.get("isSidechain"):
            continue
        for b in _blocks(entry):
            if b.get("type") != "text":
                continue
            t = _FENCE_RE.sub(" ", str(b.get("text") or "")).strip()
            if t:
                texts.append(t)
    return texts


# --------------------------------------------------------------------------
# Tool calls -> ToolCallRecord
# --------------------------------------------------------------------------


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(b.get("text", ""))
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return "" if content is None else str(content)


_EXIT_FAIL_RE = re.compile(r"^\s*(?:Error:\s*)?Exit code [1-9]\d*", re.I)


def _rec(
    name: str,
    params: Dict[str, Any],
    error: Optional[str],
    result: Optional[Dict[str, Any]] = None,
) -> ToolCallRecord:
    res = dict(result or {})
    res.setdefault("status", "error" if error else "success")
    if "path" in params and "path" not in res:
        res["path"] = params["path"]
    return ToolCallRecord(name=name, params=params, result=res, error=error)


def _lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _grep_paths(text: str) -> List[str]:
    out = []
    for ln in _lines(text):
        cand = ln.split(":", 1)[0].strip()
        if ("/" in cand or "." in cand) and " " not in cand:
            out.append(cand)
    return out


# Bash ------------------------------------------------------------------------

_SEPARATORS = {";", "&&", "||", "|", "&", "\n", "(", ")", "|&", ";;"}
_REDIRECTS = {">", ">>", "&>", "&>>", ">|"}
_PREFIXES = {"sudo", "env", "command", "time", "nohup", "exec", "xargs"}
_WRITE_CMDS = {
    "rm",
    "rmdir",
    "mv",
    "cp",
    "touch",
    "mkdir",
    "tee",
    "truncate",
    "install",
    "unlink",
    "ln",
}
_INPLACE_CMDS = {"sed", "perl"}  # write only with -i
_READ_CMDS = {"cat", "head", "tail", "less", "more", "wc", "stat", "file", "diff"}
_LIST_CMDS = {"ls", "find", "tree"}
_SEARCH_CMDS = {"grep", "rg", "ag", "egrep", "fgrep"}
_DB_CMDS = {"psql", "sqlite3", "mysql", "mongosh", "redis-cli", "alembic"}
_GIT_OPTS_WITH_ARG = {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}


def _split_commands(command: str) -> List[List[str]]:
    """Split a shell command into simple commands (token lists).

    Quoted text stays inside one token, so a heredoc commit message containing
    "git push" is not mistaken for a push.
    """
    try:
        lex = shlex.shlex(command, posix=True, punctuation_chars="();<>|&\n")
        lex.whitespace = " \t\r"
        lex.whitespace_split = True
        tokens = list(lex)
    except ValueError:
        tokens = []
        for part in re.split(r"&&|\|\||[;|\n]", command):
            tokens.extend(part.split())
            tokens.append(";")
    cmds: List[List[str]] = []
    cur: List[str] = []
    for tok in tokens:
        if tok in _SEPARATORS:
            if cur:
                cmds.append(cur)
            cur = []
        else:
            cur.append(tok)
    if cur:
        cmds.append(cur)
    return cmds


def _strip_prefix(tokens: List[str]) -> List[str]:
    i = 0
    while i < len(tokens) and (
        tokens[i] in _PREFIXES or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[i])
    ):
        i += 1
    rest = tokens[i:]
    # "uv run X", "poetry run X", "npx X", "python -m X" -> X
    if len(rest) >= 2 and rest[0] in ("uv", "poetry", "pipenv") and rest[1] == "run":
        rest = rest[2:]
        while rest and rest[0].startswith("-"):
            rest = rest[1:]
    if rest and rest[0] in ("npx", "bunx", "pnpx"):
        rest = rest[1:]
    if len(rest) >= 3 and re.match(r"^python[\d.]*$", rest[0]) and rest[1] == "-m":
        rest = rest[2:]
    return rest


def _split_redirects(tokens: List[str]) -> Tuple[List[str], List[str]]:
    """(tokens without redirections, files written by > / >>)."""
    args: List[str] = []
    written: List[str] = []
    skip = False
    for i, tok in enumerate(tokens):
        if skip:
            skip = False
            continue
        if tok in _REDIRECTS or tok == "<" or tok == ">&":
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            if tok in _REDIRECTS and nxt and nxt != "/dev/null" and not nxt.isdigit():
                written.append(nxt)
            skip = True
            continue
        if (
            tok.isdigit()
            and i + 1 < len(tokens)
            and tokens[i + 1] in _REDIRECTS | {">&"}
        ):
            continue  # the "2" in "2>&1" / "2> err.log"
        args.append(tok)
    return args, written


def _positional(args: Iterable[str]) -> List[str]:
    return [a for a in args if a and not a.startswith("-")]


def _git_records(args: List[str], error: Optional[str]) -> List[ToolCallRecord]:
    i = 1
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i] in _GIT_OPTS_WITH_ARG else 1
    if i >= len(args):
        return []
    sub, rest = args[i].lower(), args[i + 1 :]
    if sub == "branch":
        if any(r in ("-d", "-D", "--delete") for r in rest):
            sub = "branch-delete"
        elif any(r in ("-m", "-M", "--move") for r in rest):
            sub = "branch-rename"
        elif _positional(rest) and not any(
            r in ("--list", "-l", "-a", "-r", "-v", "-vv") for r in rest
        ):
            sub = "branch-create"
    return [_rec("git", {"action": sub, "args": rest}, error)]


def bash_records(command: str, error: Optional[str] = None) -> List[ToolCallRecord]:
    """``bash`` record for the whole command plus records derived from it."""
    records = [_rec("bash", {"action": "run", "command": command}, error)]
    for raw in _split_commands(command):
        args, redirected = _split_redirects(_strip_prefix(raw))
        for path in redirected:
            records.append(
                _rec("file_operations", {"action": "write", "path": path}, error)
            )
        if not args:
            continue
        prog = os.path.basename(args[0])
        pos = _positional(args[1:])
        if prog == "git":
            records.extend(_git_records(args, error))
        elif prog in _WRITE_CMDS or (
            prog in _INPLACE_CMDS
            and any(a.startswith("-i") or a == "--in-place" for a in args[1:])
        ):
            targets = pos[1:] if prog in _INPLACE_CMDS else pos
            for path in targets:
                records.append(
                    _rec("file_operations", {"action": "write", "path": path}, error)
                )
        elif prog in _READ_CMDS:
            for path in pos:
                records.append(
                    _rec("file_operations", {"action": "read", "path": path}, error)
                )
        elif prog in _LIST_CMDS or prog in _SEARCH_CMDS:
            action = "list" if prog in _LIST_CMDS else "search"
            targets = pos[1:] if prog in _SEARCH_CMDS else pos
            for path in targets or ["."]:
                records.append(
                    _rec("file_operations", {"action": action, "path": path}, error)
                )
        elif prog in _DB_CMDS:
            records.append(_rec("database", {"action": "query", "tool": prog}, error))
    return records


# Tool mapping ------------------------------------------------------------------

_FILE_WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
_GH_COMMIT_TOOLS = ("push_files", "create_or_update_file", "delete_file")
_DB_TOOL_RE = re.compile(r"postgres|sqlite|mysql|database|\bsql\b|_sql_|query", re.I)


def records_for_tool_use(
    name: str,
    tool_input: Dict[str, Any],
    result_text: str = "",
    error: Optional[str] = None,
) -> List[ToolCallRecord]:
    """Map one Claude Code tool call to the ``ToolCallRecord``s it evidences."""
    inp = tool_input or {}
    path = inp.get("file_path") or inp.get("notebook_path") or inp.get("path")
    if name in _FILE_WRITE_TOOLS:
        return [_rec("file_operations", {"action": "write", "path": path}, error)]
    if name in ("Read", "NotebookRead"):
        return [_rec("file_operations", {"action": "read", "path": path}, error)]
    if name in ("Glob", "LS"):
        entries = [] if error else _lines(result_text)
        return [
            _rec(
                "file_operations",
                {"action": "list", "path": path or "."},
                error,
                {"entries": entries},
            )
        ]
    if name == "Grep":
        entries = [] if error else _grep_paths(result_text)
        return [
            _rec(
                "file_operations",
                {"action": "search", "path": path or "."},
                error,
                {"entries": entries},
            )
        ]
    if name == "WebFetch":
        url = str(inp.get("url") or "")
        return [_rec("web_research", {"url": url}, error, {"url": url})]
    if name == "WebSearch":
        return [_rec("web_research", {"query": inp.get("query")}, error)]
    if name == "Bash":
        return bash_records(str(inp.get("command") or ""), error)
    if name.startswith("mcp__"):
        short = name.rsplit("__", 1)[-1]
        if "github" in name:
            if short in _GH_COMMIT_TOOLS:
                files = inp.get("files") or (
                    [{"path": inp["path"]}] if inp.get("path") else []
                )
                out = [
                    _rec("git", {"action": "commit", "via": name}, error),
                    _rec("git", {"action": "push", "via": name}, error),
                ]
                for f in files:
                    if isinstance(f, dict) and f.get("path"):
                        out.append(
                            _rec(
                                "file_operations",
                                {"action": "write", "path": f["path"]},
                                error,
                            )
                        )
                return out
            if short == "create_branch":
                return [_rec("git", {"action": "branch-create", "via": name}, error)]
            if short == "merge_pull_request":
                return [_rec("git", {"action": "merge", "via": name}, error)]
            return []
        if _DB_TOOL_RE.search(name):
            return [_rec("database", {"action": short, "via": name}, error)]
    return []


def trace_from_transcript(entries: Sequence[Dict[str, Any]]) -> List[ToolCallRecord]:
    """Every tool call in the session, with its outcome, as ``ToolCallRecord``s.

    The whole session counts (not only the current turn), so "I committed it"
    about earlier work is not flagged. Sidechain (subagent) calls are included:
    they really ran. A call with no ``tool_result`` (interrupted) counts as failed.
    """
    results: Dict[str, Tuple[str, bool]] = {}
    uses: List[Tuple[str, str, Dict[str, Any]]] = []
    for entry in entries:
        for b in _blocks(entry):
            t = b.get("type")
            if t == "tool_use" and entry.get("type") == "assistant":
                uses.append(
                    (str(b.get("id")), str(b.get("name") or ""), b.get("input") or {})
                )
            elif t == "tool_result":
                text = _result_text(b.get("content"))
                results[str(b.get("tool_use_id"))] = (text, bool(b.get("is_error")))
    trace: List[ToolCallRecord] = []
    for use_id, name, inp in uses:
        if use_id in results:
            text, is_error = results[use_id]
            failed = is_error or (name == "Bash" and bool(_EXIT_FAIL_RE.match(text)))
            error = (text.strip()[:200] or "tool error") if failed else None
        else:
            text, error = "", "no tool_result (interrupted or still running)"
        trace.extend(
            records_for_tool_use(
                name, inp if isinstance(inp, dict) else {}, text, error
            )
        )
    return trace


# --------------------------------------------------------------------------
# Re-checks where a shell changes the answer
# --------------------------------------------------------------------------

_TEST_CLAIM_RE = re.compile(r"\btests?\b|test\s+suite|\bpytest\b|\bspecs?\b", re.I)
_TEST_RUNNER_RE = re.compile(
    r"\b(?:pytest|py\.test|unittest|tox|nox|jest|vitest|mocha|karma|rspec|phpunit|"
    r"(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?test|ng\s+test|go\s+test|cargo\s+test|"
    r"make\s+(?:test|check)|dotnet\s+test|mvn\s+test|gradle\w*\s+test)\b",
    re.I,
)
_LINT_CLAIM_RE = re.compile(r"\b(?:lint(?:er|ed|ing)?|format(?:ter|ted)?)\b", re.I)
_LINT_RUNNER_RE = re.compile(
    r"\b(?:lint|ruff|flake8|pylint|eslint|black|prettier|mypy|isort|tsc)\b", re.I
)
_BUILD_CLAIM_RE = re.compile(r"\bbuild\b", re.I)
_BUILD_RUNNER_RE = re.compile(r"\b(?:build|compile|tsc|make)\b", re.I)
_BACKTICK_RE = re.compile(r"`([^`\n]+)`")

# git verb in the claim text -> git subcommands that make it true
_GIT_VERBS: List[Tuple[re.Pattern, set]] = [
    (re.compile(r"\bcommitted\b", re.I), {"commit"}),
    (re.compile(r"\bpushed\b", re.I), {"push"}),
    (re.compile(r"\bmerged\b", re.I), {"merge", "pull"}),
    (re.compile(r"\brebased\b", re.I), {"rebase", "pull"}),
    (re.compile(r"\breverted\b", re.I), {"revert", "reset", "restore", "checkout"}),
    (re.compile(r"\btagged\b", re.I), {"tag"}),
    (re.compile(r"\bstaged\b", re.I), {"add", "stage"}),
    (re.compile(r"\bcherry-picked\b", re.I), {"cherry-pick"}),
    (
        re.compile(
            r"\bchecked\s+out\b|\bcreated\s+(?:a\s+|the\s+|new\s+)*branch", re.I
        ),
        {"checkout", "switch", "branch-create", "worktree"},
    ),
]


def _ok(call: ToolCallRecord) -> bool:
    return not call.error


def _recheck_command(claim: ClaimCheck, trace: Sequence[ToolCallRecord]) -> None:
    runs = [c for c in trace if c.name == "bash"]
    text = claim.text
    if _TEST_CLAIM_RE.search(text):
        pattern, what = _TEST_RUNNER_RE, "test run"
    elif _LINT_CLAIM_RE.search(text):
        pattern, what = _LINT_RUNNER_RE, "lint/format run"
    elif _BUILD_CLAIM_RE.search(text):
        pattern, what = _BUILD_RUNNER_RE, "build"
    else:
        tick = _BACKTICK_RE.search(text)
        words = _strip_prefix(tick.group(1).split()) if tick else []
        if words:
            prog = re.escape(os.path.basename(words[0]))
            pattern, what = re.compile(rf"(?:^|[\s/;&|(]){prog}\b"), words[0]
        else:
            pattern, what = None, "shell command"
    matching = [
        c
        for c in runs
        if pattern is None or pattern.search(str(c.params.get("command", "")))
    ]
    good = [c for c in matching if _ok(c)]
    if good:
        claim.supported = True
        claim.evidence = f"Bash: {str(good[-1].params.get('command', ''))[:120]}"
    else:
        claim.supported = False
        claim.evidence = (
            f"no successful {what}; last attempt failed: "
            f"{str(matching[-1].params.get('command', ''))[:120]}"
            if matching
            else None
        )


def _recheck_git_write(claim: ClaimCheck, trace: Sequence[ToolCallRecord]) -> None:
    if claim.detail:
        groups = [{claim.detail}]
    else:
        groups = [acts for rx, acts in _GIT_VERBS if rx.search(claim.text)]
    if not groups:
        return  # unknown verb: keep claims.py's verdict
    done = [c for c in trace if c.name == "git" and _ok(c)]
    evidence, missing = [], []
    for acts in groups:
        hit = next((c for c in done if c.params.get("action") in acts), None)
        if hit is None:
            missing.append("/".join(sorted(acts)))
        else:
            evidence.append(f"git {hit.params.get('action')}")
    claim.supported = not missing
    claim.evidence = (
        ", ".join(evidence)
        if not missing
        else f"no successful git {' or '.join(missing)}"
    )


def verify_transcript_texts(
    texts: Sequence[str], trace: Sequence[ToolCallRecord]
) -> ClaimReport:
    """``verify_texts`` plus the shell-aware command and git re-checks."""
    report = verify_texts(texts, trace)
    for claim in report.checks:
        if claim.kind == "command":
            _recheck_command(claim, trace)
        elif claim.kind == "git_write":
            _recheck_git_write(claim, trace)
    return report


@dataclass
class TranscriptCheck:
    report: ClaimReport
    trace: List[ToolCallRecord] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    prompt: str = ""


def check_transcript(
    entries: Sequence[Dict[str, Any]], last_message: Optional[str] = None
) -> TranscriptCheck:
    """Verify the current turn's claims against the session's tool calls.

    ``last_message`` (the hook payload's ``last_assistant_message``, when the
    Claude Code version sends it) is added if the transcript does not yet hold
    it, since the final entry can land after the hook fires.
    """
    texts = turn_texts(entries)
    if last_message and last_message.strip():
        lm = _FENCE_RE.sub(" ", last_message).strip()
        if lm and not any(lm in t or t in lm for t in texts):
            texts.append(lm)
    trace = trace_from_transcript(entries)
    return TranscriptCheck(
        report=verify_transcript_texts(texts, trace),
        trace=trace,
        texts=texts,
        prompt=last_user_prompt(entries),
    )


# --------------------------------------------------------------------------
# Hook protocol
# --------------------------------------------------------------------------


def hook_mode() -> str:
    mode = os.getenv("CORE_CLAIM_HOOK_MODE", "block").strip().lower() or "block"
    return mode if mode in _MODES else "block"


def _describe_claim(c: ClaimCheck) -> str:
    target = f" [{c.target}]" if c.target else ""
    note = f" ({c.evidence})" if c.evidence else ""
    return f'- {c.kind}{target}: "{c.text}"{note}'


def format_reason(report: ClaimReport) -> str:
    lines = [
        "Claim check: these statements in your reply have no matching successful "
        "tool call in this session:",
        *(_describe_claim(c) for c in report.unsupported),
        "",
        "If it did not happen, do it now or correct the reply to say it was not "
        "done. If it happened in a way this check cannot see (a subagent, the "
        "user, an earlier session), say so plainly in the reply.",
    ]
    return "\n".join(lines)


def build_fault_report(check: TranscriptCheck, payload: Dict[str, Any]) -> FaultReport:
    return FaultReport(
        run_id=payload.get("session_id"),
        input=check.prompt,
        steps=[
            StepRecord(
                name=f"{c.name}:{c.params.get('action', '')}",
                status="failure" if c.error else "success",
                error=c.error,
            )
            for c in check.trace
        ],
        tool_calls=check.trace,
        actual="\n\n".join(check.texts) or None,
        metadata={
            "source": HOOK_SOURCE,
            "fault_reasons": ["unsupported_claims"],
            "unsupported_claims": check.report.to_dict()["unsupported"],
            "session_id": payload.get("session_id"),
            "transcript_path": payload.get("transcript_path"),
            "cwd": payload.get("cwd"),
            "hook_event_name": payload.get("hook_event_name"),
            "stop_hook_active": bool(payload.get("stop_hook_active")),
        },
    )


def _maybe_capture(check: TranscriptCheck, payload: Dict[str, Any]) -> None:
    try:
        from app.config.evaluation import get_fault_capture_config
        from app.evaluation.capture import capture_fault

        cfg = get_fault_capture_config()
        if cfg.enabled:
            capture_fault(build_fault_report(check, payload), cfg.directory)
    except Exception as exc:  # never let capture break the hook
        logger.warning("claim hook: fault capture failed: %s", exc)


def run_hook(payload: Dict[str, Any], mode: Optional[str] = None) -> Dict[str, Any]:
    """Decide the hook output for one ``Stop`` payload. ``{}`` means allow."""
    mode = mode or hook_mode()
    if mode == "off":
        return {}
    path = payload.get("transcript_path")
    if not path or not Path(path).is_file():
        return {}
    check = check_transcript(
        load_transcript(path), payload.get("last_assistant_message")
    )
    if check.report.is_clean():
        return {}
    _maybe_capture(check, payload)
    reason = format_reason(check.report)
    if mode == "warn" or payload.get("stop_hook_active"):
        # Already blocked once for this stop: tell the user, don't loop.
        return {"systemMessage": reason}
    return {"decision": "block", "reason": reason}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--transcript", help="audit a transcript file and print the report"
    )
    args = parser.parse_args(argv)
    if args.transcript:
        check = check_transcript(load_transcript(args.transcript))
        print(json.dumps(check.report.to_dict(), indent=2))
        return 0 if check.report.is_clean() else 1
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        out = run_hook(payload if isinstance(payload, dict) else {})
    except Exception as exc:  # fail open: a broken check must not trap the agent
        print(f"claim hook error (ignored): {exc}", file=sys.stderr)
        return 0
    if out:
        print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
