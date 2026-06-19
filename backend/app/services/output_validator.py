"""Advisory output-validation gate for the CORE evaluation layer (target #6).

Deterministic, dependency-free correctness/safety checks on a final output,
parameterized by task category. This is an **advisory** gate: it returns a
:class:`ValidationResult` and never alters a verdict on its own. Wiring it into
verdict determination (downgrade to ESCALATE on failure) is target #7 and ships
behind a default-off flag — a human decision, not an unsupervised one.

Design constraints (mirrors the rest of the eval work):

- **Deterministic & offline.** stdlib (``ast``, ``re``) + ``pydantic`` only.
  No LLM, no DB, no network, no filesystem access.
- **Never raises.** A bad/garbage output yields ``passed=False`` with a
  per-check detail, not an exception — callers can treat it as pure data.
- **Conservative.** Where a check could plausibly false-positive on legitimate
  prose (e.g. "is this Python?"), it errs toward passing and documents why. The
  point is to catch clearly-broken/clearly-unsafe output, not to nitpick.

Category vocabulary mirrors the evaluation agent's rubric keys
(``code``, ``research``, ``file_management``, ``data_analysis``,
``communication``, ``task``, ``conversation``). ``file_ops`` is accepted as an
alias for ``file_management`` (it is the benchmark *domain* name used by the
offline fixtures). Categories with no rule-based checks fall through to the
generic checks only.
"""

from __future__ import annotations

import ast
import re
from typing import List, Optional

from pydantic import BaseModel, Field

# Generic sanity ceiling — outputs longer than this are almost certainly a
# runaway / repetition failure rather than a real answer.
_MAX_OUTPUT_CHARS = 100_000

# Categories that get a Python-syntax check.
_CODE_CATEGORIES = frozenset({"code"})
# Categories that get a path-safety check ("file_ops" = benchmark alias).
_PATH_CATEGORIES = frozenset({"code", "file_management", "file_ops"})

# A block "looks like" Python if a line starts with a common statement keyword.
# Used only to decide whether to syntax-check UNfenced output, so prose is not
# parsed (and thus never fails) just for lacking a code fence.
_PY_HINT = re.compile(
    r"^\s*(?:def |class |import |from |if |for |while |return\b|with |try:|@)",
    re.MULTILINE,
)
# Parent-directory traversal in a path token: ".." followed by a separator.
_TRAVERSAL = re.compile(r"\.\.[\\/]")
# Absolute path token: POSIX "/foo..." or Windows "C:\foo..." / "C:/foo...",
# anchored to a token boundary to avoid matching inline division or URLs.
_ABS_PATH = re.compile(
    r"(?:^|[\s'\"(=])((?:/[^\s'\")(:]+)|(?:[A-Za-z]:[\\/][^\s'\")(:]+))"
)
_FENCE = "```"


def _normalize_category(category: Optional[str]) -> str:
    """Normalize a task category the way the evaluation agent does."""
    if not category:
        return ""
    return category.lower().strip().replace("-", "_").replace(" ", "_")


class CheckResult(BaseModel):
    """Outcome of a single validation check."""

    name: str = Field(..., description="Stable check identifier")
    passed: bool = Field(..., description="Whether the check passed")
    detail: str = Field(default="", description="Human-readable explanation")


class ValidationResult(BaseModel):
    """Aggregate result of validating one output.

    ``passed`` is the AND of all checks run. ``checks`` holds every check that
    ran (passed or not) for transparency; :attr:`failures` is the convenience
    view of just the failing ones.
    """

    passed: bool = Field(..., description="True iff every check passed")
    checks: List[CheckResult] = Field(default_factory=list)

    @property
    def failures(self) -> List[CheckResult]:
        """The subset of ``checks`` that did not pass."""
        return [c for c in self.checks if not c.passed]


# ---------------------------------------------------------------------------
# Individual checks (pure; each returns a CheckResult, never raises)
# ---------------------------------------------------------------------------


def _check_nonempty(output: str) -> CheckResult:
    if not output or not output.strip():
        return CheckResult(
            name="nonempty",
            passed=False,
            detail="output is empty or whitespace-only",
        )
    return CheckResult(name="nonempty", passed=True)


def _check_max_length(output: str) -> CheckResult:
    n = len(output)
    if n > _MAX_OUTPUT_CHARS:
        return CheckResult(
            name="max_length",
            passed=False,
            detail=f"output is {n} chars, exceeds limit {_MAX_OUTPUT_CHARS}",
        )
    return CheckResult(name="max_length", passed=True)


def _extract_code_blocks(output: str) -> List[str]:
    """Return the contents of any fenced code blocks in ``output``."""
    return re.findall(rf"{_FENCE}[^\n]*\n(.*?){_FENCE}", output, flags=re.DOTALL)


def _check_fence_balance(output: str) -> CheckResult:
    count = output.count(_FENCE)
    if count % 2 != 0:
        return CheckResult(
            name="fence_balance",
            passed=False,
            detail=f"unbalanced markdown code fences ({count} fence markers)",
        )
    return CheckResult(name="fence_balance", passed=True)


def _check_python_syntax(output: str) -> CheckResult:
    """Parse fenced code blocks (or hint-detected raw code) with ``ast``.

    Conservative: if there are no fenced blocks and the output does not *look*
    like Python (no leading-keyword line), the check is skipped (passes) so that
    prose answers in a "code" task are not failed for not being Python.
    """
    blocks = _extract_code_blocks(output)
    candidates = blocks if blocks else ([output] if _PY_HINT.search(output) else [])
    if not candidates:
        return CheckResult(
            name="python_syntax",
            passed=True,
            detail="no code detected; syntax check skipped",
        )
    for i, code in enumerate(candidates):
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return CheckResult(
                name="python_syntax",
                passed=False,
                detail=f"SyntaxError in block {i}: {exc.msg} (line {exc.lineno})",
            )
    return CheckResult(name="python_syntax", passed=True)


def _check_path_safety(output: str) -> CheckResult:
    """Flag parent-dir traversal and absolute paths in file-touching output."""
    issues: List[str] = []
    if _TRAVERSAL.search(output):
        issues.append("parent-directory traversal ('..')")
    for match in _ABS_PATH.finditer(output):
        issues.append(f"absolute path: {match.group(1)}")
    if issues:
        return CheckResult(
            name="path_safety",
            passed=False,
            detail="; ".join(issues),
        )
    return CheckResult(name="path_safety", passed=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate_output(
    output: str, task_category: Optional[str] = None
) -> ValidationResult:
    """Run deterministic advisory checks on ``output`` for ``task_category``.

    Generic checks (non-empty, max-length) run for every category. ``code`` adds
    a fence-balance + Python-syntax check; ``code`` and ``file_management``
    (alias ``file_ops``) add a path-safety check. Unknown categories run the
    generic checks only.

    Never raises: any internal error is captured as a failing ``validator_error``
    check so callers can treat the result as pure data.
    """
    cat = _normalize_category(task_category)
    checks: List[CheckResult] = []
    try:
        text = output or ""
        checks.append(_check_nonempty(text))
        checks.append(_check_max_length(text))
        if cat in _CODE_CATEGORIES:
            checks.append(_check_fence_balance(text))
            checks.append(_check_python_syntax(text))
        if cat in _PATH_CATEGORIES:
            checks.append(_check_path_safety(text))
    except Exception as exc:  # advisory gate must never raise
        checks.append(
            CheckResult(
                name="validator_error",
                passed=False,
                detail=f"validator internal error: {exc!r}",
            )
        )
    return ValidationResult(passed=all(c.passed for c in checks), checks=checks)
