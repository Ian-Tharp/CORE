"""Replay golden cases through an injected callable and score them."""

from __future__ import annotations

import time
from typing import Callable, Optional, Union

from pydantic import BaseModel, Field

from app.evaluation.evaluators import EvaluatorResult, evaluate
from app.evaluation.golden_set import GoldenCase


class RunOutput(BaseModel):
    """Optional rich return value; a plain ``str`` is also accepted.

    If ``latency_ms`` is omitted the runner measures wall-clock time.
    """

    output: str
    latency_ms: Optional[float] = None


Target = Callable[[str], Union[str, RunOutput]]


class CaseResult(BaseModel):
    case_id: str
    passed: bool
    output: Optional[str] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    evaluations: list[EvaluatorResult] = Field(default_factory=list)


class RunSummary(BaseModel):
    total: int
    passed: int
    failed: int
    errored: int
    pass_rate: float


class RunResult(BaseModel):
    results: list[CaseResult]
    summary: RunSummary

    @property
    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if not r.passed]


def run_cases(cases: list[GoldenCase], target: Target) -> RunResult:
    """Execute every case. A raising target fails that case only."""
    results: list[CaseResult] = []
    for case in cases:
        start = time.perf_counter()
        try:
            raw = target(case.input)
        except Exception as exc:  # noqa: BLE001 - target is untrusted
            results.append(
                CaseResult(
                    case_id=case.id,
                    passed=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        elapsed = (time.perf_counter() - start) * 1000
        if isinstance(raw, RunOutput):
            output, latency = raw.output, raw.latency_ms
        elif isinstance(raw, str):
            output, latency = raw, None
        else:
            results.append(
                CaseResult(
                    case_id=case.id,
                    passed=False,
                    error=f"target returned {type(raw).__name__}, expected str",
                )
            )
            continue
        latency = elapsed if latency is None else latency
        evals = [evaluate(spec, output, latency) for spec in case.evaluators]
        results.append(
            CaseResult(
                case_id=case.id,
                passed=all(e.passed for e in evals),
                output=output,
                latency_ms=latency,
                evaluations=evals,
            )
        )
    total = len(results)
    passed = sum(r.passed for r in results)
    summary = RunSummary(
        total=total,
        passed=passed,
        failed=total - passed,
        errored=sum(r.error is not None for r in results),
        pass_rate=(passed / total) if total else 0.0,
    )
    return RunResult(results=results, summary=summary)
