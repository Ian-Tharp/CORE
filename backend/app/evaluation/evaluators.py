"""Deterministic, code-based evaluators. No LLM, no IO."""

from __future__ import annotations

import re
from typing import Optional, Sequence

from app.evaluation.models import (
    ClaimsSupported,
    Contains,
    EvaluatorOutcome,
    EvaluatorSpec,
    ExactMatch,
    ForbiddenSubstring,
    MaxLatency,
    Regex,
    ToolCallRecord,
)


def evaluate(
    spec: EvaluatorSpec,
    output: str,
    latency_ms: Optional[float],
    tool_calls: Sequence[ToolCallRecord] = (),
) -> EvaluatorOutcome:
    name = spec.type

    if isinstance(spec, ExactMatch):
        got, want = output, spec.value
        if spec.strip:
            got, want = got.strip(), want.strip()
        if not spec.case_sensitive:
            got, want = got.casefold(), want.casefold()
        ok = got == want
        return EvaluatorOutcome(
            evaluator=name, passed=ok, detail="" if ok else "output != expected"
        )

    if isinstance(spec, Contains):
        got, want = output, spec.value
        if not spec.case_sensitive:
            got, want = got.casefold(), want.casefold()
        ok = want in got
        return EvaluatorOutcome(
            evaluator=name,
            passed=ok,
            detail="" if ok else f"missing {spec.value!r}",
        )

    if isinstance(spec, Regex):
        flags = re.IGNORECASE if spec.ignore_case else 0
        try:
            ok = re.search(spec.pattern, output, flags) is not None
        except re.error as exc:
            return EvaluatorOutcome(
                evaluator=name, passed=False, detail=f"invalid regex: {exc}"
            )
        return EvaluatorOutcome(
            evaluator=name,
            passed=ok,
            detail="" if ok else f"no match for /{spec.pattern}/",
        )

    if isinstance(spec, MaxLatency):
        if latency_ms is None:
            return EvaluatorOutcome(
                evaluator=name, passed=False, detail="latency unavailable"
            )
        ok = latency_ms <= spec.ms
        return EvaluatorOutcome(
            evaluator=name,
            passed=ok,
            detail="" if ok else f"{latency_ms:.1f}ms > {spec.ms:.1f}ms",
        )

    if isinstance(spec, ForbiddenSubstring):
        hay = output if spec.case_sensitive else output.casefold()
        hits = [
            v
            for v in spec.values
            if (v if spec.case_sensitive else v.casefold()) in hay
        ]
        return EvaluatorOutcome(
            evaluator=name,
            passed=not hits,
            detail="" if not hits else f"forbidden: {hits!r}",
        )

    if isinstance(spec, ClaimsSupported):
        from app.evaluation.claims import verify_claims

        bad = verify_claims(output, tool_calls).unsupported
        return EvaluatorOutcome(
            evaluator=name,
            passed=not bad,
            detail="" if not bad else f"unsupported claims: {[c.text for c in bad]!r}",
        )

    raise TypeError(f"unsupported evaluator spec: {spec!r}")  # pragma: no cover
