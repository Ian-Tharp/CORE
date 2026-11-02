"""Golden Set store: one JSON object per line, on disk, no database.

* ``load`` never raises on a bad line; it reports it and keeps the rest.
* ``add`` / ``append_fault`` reject duplicates by :func:`case_signature` (and
  by id) so a recurring fault does not grow the set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from pydantic import ValidationError

from app.evaluation.models import EvaluatorSpec, ExactMatch, FaultReport, GoldenCase
from app.evaluation.signature import case_signature


class DuplicateCaseError(ValueError):
    """Raised when a case with the same signature or id already exists."""


@dataclass(frozen=True)
class SkippedLine:
    line_no: int  # 1-based
    reason: str


@dataclass
class LoadResult:
    cases: List[GoldenCase] = field(default_factory=list)
    skipped: List[SkippedLine] = field(default_factory=list)


class GoldenSet:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    # ---- read -----------------------------------------------------------

    def load(self) -> LoadResult:
        result = LoadResult()
        if not self.path.exists():
            return result
        seen_sigs: Dict[str, int] = {}
        seen_ids: Dict[str, int] = {}
        with self.path.open("r", encoding="utf-8", errors="replace") as fh:
            for line_no, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    case = GoldenCase.model_validate_json(line)
                except ValidationError as exc:
                    first = exc.errors()[0]
                    where = ".".join(str(p) for p in first["loc"])
                    result.skipped.append(
                        SkippedLine(line_no, f"invalid: {where}: {first['msg']}")
                    )
                    continue
                sig = case_signature(case.input, case.evaluators)
                if sig in seen_sigs or case.id in seen_ids:
                    first_line = seen_sigs.get(sig) or seen_ids[case.id]
                    result.skipped.append(
                        SkippedLine(line_no, f"duplicate of line {first_line}")
                    )
                    continue
                seen_sigs[sig] = line_no
                seen_ids[case.id] = line_no
                result.cases.append(case)
        return result

    # ---- write ----------------------------------------------------------

    def add(self, case: GoldenCase) -> GoldenCase:
        existing = self.load().cases
        sig = case_signature(case.input, case.evaluators)
        for other in existing:
            if case_signature(other.input, other.evaluators) == sig:
                raise DuplicateCaseError(f"duplicate of case {other.id!r}")
            if other.id == case.id:
                raise DuplicateCaseError(f"id {case.id!r} already exists")
        self._append_line(case.model_dump_json())
        return case

    def append_fault(
        self,
        fault: FaultReport,
        evaluators: Optional[Sequence[EvaluatorSpec]] = None,
        tags: Sequence[str] = (),
    ) -> GoldenCase:
        """Turn a fault into a new case. Defaults to exact match on ``expected``."""
        checks = list(evaluators) if evaluators else []
        if not checks:
            if not (fault.expected and fault.expected.strip()):
                raise ValueError(
                    "fault has no expected output and no evaluators were given"
                )
            checks = [ExactMatch(value=fault.expected)]
        sig = case_signature(fault.input, checks)
        case = GoldenCase(
            id=f"gc-{sig[:12]}",
            input=fault.input,
            evaluators=checks,
            tags=list(tags),
            source="fault",
            fault_id=fault.fault_id,
        )
        return self.add(case)

    def _append_line(self, line: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        prefix = ""
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("rb") as fh:
                fh.seek(-1, 2)
                if fh.read(1) != b"\n":  # repair a truncated final line
                    prefix = "\n"
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(prefix + line + "\n")
