"""Deterministic, code-based evaluators (no LLM calls)."""

from __future__ import annotations

import re
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter, field_validator


class ExactMatch(BaseModel):
    type: Literal["exact_match"] = "exact_match"
    expected: str
    strip: bool = True
    case_sensitive: bool = True


class Contains(BaseModel):
    type: Literal["contains"] = "contains"
    substring: str
    case_sensitive: bool = False


class RegexMatch(BaseModel):
    type: Literal["regex"] = "regex"
    pattern: str

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, v: str) -> str:
        try:
            re.compile(v)
        except re.error as exc:  # re.error is not a ValueError; wrap it
            raise ValueError(f"invalid regex: {exc}") from exc
        return v


class MaxLatency(BaseModel):
    type: Literal["max_latency"] = "max_latency"
    max_ms: float = Field(gt=0)


class ForbiddenSubstring(BaseModel):
    """Guardrail: output must contain none of these."""

    type: Literal["forbidden_substring"] = "forbidden_substring"
    substrings: list[str] = Field(min_length=1)
    case_sensitive: bool = False


EvaluatorSpec = Annotated[
    Union[ExactMatch, Contains, RegexMatch, MaxLatency, ForbiddenSubstring],
    Field(discriminator="type"),
]
evaluator_adapter: TypeAdapter = TypeAdapter(EvaluatorSpec)


class EvaluatorResult(BaseModel):
    type: str
    passed: bool
    detail: str = ""


def evaluate(
    spec: ExactMatch | Contains | RegexMatch | MaxLatency | ForbiddenSubstring,
    output: str,
    latency_ms: Optional[float] = None,
) -> EvaluatorResult:
    t = spec.type
    if isinstance(spec, ExactMatch):
        a, b = output, spec.expected
        if spec.strip:
            a, b = a.strip(), b.strip()
        if not spec.case_sensitive:
            a, b = a.casefold(), b.casefold()
        ok = a == b
        return EvaluatorResult(
            type=t, passed=ok, detail="" if ok else f"expected {spec.expected!r}"
        )
    if isinstance(spec, Contains):
        hay, needle = output, spec.substring
        if not spec.case_sensitive:
            hay, needle = hay.casefold(), needle.casefold()
        ok = needle in hay
        return EvaluatorResult(
            type=t, passed=ok, detail="" if ok else f"missing {spec.substring!r}"
        )
    if isinstance(spec, RegexMatch):
        ok = re.search(spec.pattern, output) is not None
        return EvaluatorResult(
            type=t, passed=ok, detail="" if ok else f"no match for /{spec.pattern}/"
        )
    if isinstance(spec, MaxLatency):
        if latency_ms is None:
            return EvaluatorResult(type=t, passed=False, detail="latency unknown")
        ok = latency_ms <= spec.max_ms
        return EvaluatorResult(
            type=t,
            passed=ok,
            detail="" if ok else f"{latency_ms:.1f}ms > {spec.max_ms}ms",
        )
    # ForbiddenSubstring
    hay = output if spec.case_sensitive else output.casefold()
    hits = [
        s
        for s in spec.substrings
        if (s if spec.case_sensitive else s.casefold()) in hay
    ]
    return EvaluatorResult(
        type=t, passed=not hits, detail=f"forbidden: {hits}" if hits else ""
    )
