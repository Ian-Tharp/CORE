"""Replay golden cases through an injected callable and score them.

The callable receives only the case *input* (never the expected values) and
returns either a ``str`` or a :class:`RunOutput`. No LLM or service is ever
called from here; production wiring is the caller's job.
"""

from __future__ import annotations

import inspect
import time
from typing import Awaitable, Callable, Iterable, List, Optional, Union

from app.evaluation.evaluators import evaluate
from app.evaluation.models import (
    CaseResult,
    GoldenCase,
    RunOutput,
    RunResult,
    RunSummary,
    utcnow,
)

CaseCallable = Callable[[str], Union[str, RunOutput]]
AsyncCaseCallable = Callable[[str], Union[str, RunOutput, Awaitable]]


def _score(case: GoldenCase, produced: object, wall_ms: float) -> CaseResult:
    if isinstance(produced, str):
        produced = RunOutput(output=produced)
    if not isinstance(produced, RunOutput):
        return CaseResult(
            case_id=case.id,
            passed=False,
            error=f"callable returned {type(produced).__name__}, expected str|RunOutput",
        )
    latency = produced.latency_ms if produced.latency_ms is not None else wall_ms
    outcomes = [
        evaluate(spec, produced.output, latency, produced.tool_calls)
        for spec in case.evaluators
    ]
    return CaseResult(
        case_id=case.id,
        passed=all(o.passed for o in outcomes),
        output=produced.output,
        latency_ms=latency,
        outcomes=outcomes,
    )


def _error(case: GoldenCase, exc: BaseException) -> CaseResult:
    return CaseResult(
        case_id=case.id, passed=False, error=f"{type(exc).__name__}: {exc}"
    )


def summarize(results: List[CaseResult]) -> RunSummary:
    total = len(results)
    passed = sum(r.passed for r in results)
    by_eval: dict = {}
    for r in results:
        for o in r.outcomes:
            bucket = by_eval.setdefault(o.evaluator, {"passed": 0, "failed": 0})
            bucket["passed" if o.passed else "failed"] += 1
    return RunSummary(
        total=total,
        passed=passed,
        failed=total - passed,
        errored=sum(r.error is not None for r in results),
        pass_rate=(passed / total) if total else 0.0,
        failed_case_ids=[r.case_id for r in results if not r.passed],
        by_evaluator=by_eval,
    )


def run_cases(cases: Iterable[GoldenCase], fn: CaseCallable) -> RunResult:
    started = utcnow()
    results: List[CaseResult] = []
    for case in cases:
        t0 = time.perf_counter()
        try:
            produced = fn(case.input)
        except Exception as exc:  # a crashing callable is a failed case, not a crash
            results.append(_error(case, exc))
            continue
        results.append(_score(case, produced, (time.perf_counter() - t0) * 1000))
    return RunResult(
        started_at=started,
        ended_at=utcnow(),
        results=results,
        summary=summarize(results),
    )


async def arun_cases(cases: Iterable[GoldenCase], fn: AsyncCaseCallable) -> RunResult:
    """Like :func:`run_cases` but accepts sync or async callables."""
    started = utcnow()
    results: List[CaseResult] = []
    for case in cases:
        t0 = time.perf_counter()
        try:
            produced = fn(case.input)
            if inspect.isawaitable(produced):
                produced = await produced
        except Exception as exc:
            results.append(_error(case, exc))
            continue
        results.append(_score(case, produced, (time.perf_counter() - t0) * 1000))
    return RunResult(
        started_at=started,
        ended_at=utcnow(),
        results=results,
        summary=summarize(results),
    )
