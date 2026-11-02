"""Golden Set: JSONL-backed store of replayable evaluation cases."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from app.evaluation.evaluators import EvaluatorSpec, ExactMatch
from app.evaluation.models import FaultReport, utcnow

logger = logging.getLogger(__name__)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def compute_signature(input_text: str, expected: Optional[str]) -> str:
    """Stable identity of a case: normalised input + expected outcome.

    Whitespace and case differences do not create a new case.
    """
    payload = json.dumps([_norm(input_text), _norm(expected or "")], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class GoldenCase(BaseModel):
    id: str
    input: str
    expected: Optional[str] = None
    evaluators: list[EvaluatorSpec] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    source: str = "manual"  # "manual" | "fault"
    source_fault_id: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def signature(self) -> str:
        return compute_signature(self.input, self.expected)


class LoadError(BaseModel):
    line: int
    message: str


class LoadResult(BaseModel):
    cases: list[GoldenCase] = Field(default_factory=list)
    errors: list[LoadError] = Field(default_factory=list)


class GoldenSet:
    """A golden set backed by one JSONL file (one case per line)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> LoadResult:
        """Load and validate. Corrupt/invalid/duplicate lines are reported in
        ``errors`` and skipped; the rest still load."""
        result = LoadResult()
        if not self.path.exists():
            return result
        seen: dict[str, int] = {}
        text = self.path.read_text(encoding="utf-8", errors="replace")
        for n, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                case = GoldenCase.model_validate_json(line)
            except ValidationError as exc:
                result.errors.append(LoadError(line=n, message=_first_error(exc)))
                continue
            sig = case.signature
            if sig in seen:
                result.errors.append(
                    LoadError(line=n, message=f"duplicate of line {seen[sig]}")
                )
                continue
            seen[sig] = n
            result.cases.append(case)
        for err in result.errors:
            logger.warning(
                "golden set %s line %d: %s", self.path, err.line, err.message
            )
        return result

    def add_case(self, case: GoldenCase) -> bool:
        """Append ``case``. Returns False (and writes nothing) if a case with
        the same signature already exists."""
        existing = self.load().cases
        if case.signature in {c.signature for c in existing}:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # If the file lacks a trailing newline (e.g. truncated write), start a
        # fresh line so we never glue onto a corrupt one.
        prefix = ""
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("rb") as fh:
                fh.seek(-1, 2)
                if fh.read(1) != b"\n":
                    prefix = "\n"
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(prefix + case.model_dump_json() + "\n")
        return True

    def add_fault(
        self,
        fault: FaultReport,
        *,
        expected: Optional[str] = None,
        evaluators: Optional[list[EvaluatorSpec]] = None,
        tags: Optional[list[str]] = None,
    ) -> Optional[GoldenCase]:
        """Turn a fault into a new case. Returns the case, or None if it was a
        duplicate. ``expected`` falls back to ``fault.expected``; without any
        evaluators the default is an exact match on the expected output."""
        exp = expected if expected is not None else fault.expected
        if evaluators is None:
            if exp is None:
                raise ValueError("need `expected` or explicit evaluators")
            evaluators = [ExactMatch(expected=exp)]
        case = GoldenCase(
            id="gc-" + compute_signature(fault.input, exp)[:12],
            input=fault.input,
            expected=exp,
            evaluators=evaluators,
            tags=tags or ["fault"],
            source="fault",
            source_fault_id=fault.id,
        )
        return case if self.add_case(case) else None


def _first_error(exc: ValidationError) -> str:
    e = exc.errors()[0]
    loc = ".".join(str(p) for p in e["loc"])
    return f"{loc}: {e['msg']}" if loc else e["msg"]


__all__ = ["GoldenCase", "GoldenSet", "LoadError", "LoadResult", "compute_signature"]
