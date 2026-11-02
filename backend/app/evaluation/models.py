"""Typed models for the evaluation layer (Phase I/II slice).

Everything here is plain pydantic v2 and JSON round-trippable: no DB, no
network, no LLM. See ``docs/evaluation_spec.md``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


# --------------------------------------------------------------------------
# Fault capture
# --------------------------------------------------------------------------


class TokenUsage(BaseModel):
    prompt: Optional[int] = Field(default=None, ge=0)
    completion: Optional[int] = Field(default=None, ge=0)

    @property
    def total(self) -> Optional[int]:
        if self.prompt is None and self.completion is None:
            return None
        return (self.prompt or 0) + (self.completion or 0)


class ToolCallRecord(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = Field(default=None, ge=0)


class StepRecord(BaseModel):
    """One step the pipeline took (node, plan step, reasoning step...)."""

    name: str
    status: Optional[str] = None
    detail: Optional[str] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = Field(default=None, ge=0)
    tokens: Optional[TokenUsage] = None


class FaultReport(BaseModel):
    """A captured failed pipeline run (spec section 3.C, "Capture")."""

    fault_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: Optional[str] = None
    input: str
    steps: List[StepRecord] = Field(default_factory=list)
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    expected: Optional[str] = None
    actual: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    captured_at: datetime = Field(default_factory=utcnow)
    latency_ms: Optional[float] = Field(default=None, ge=0)
    tokens: Optional[TokenUsage] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# Evaluator specs (discriminated union on ``type``)
# --------------------------------------------------------------------------


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExactMatch(_Spec):
    type: Literal["exact_match"] = "exact_match"
    value: str
    strip: bool = True
    case_sensitive: bool = True


class Contains(_Spec):
    type: Literal["contains"] = "contains"
    value: str
    case_sensitive: bool = False


class Regex(_Spec):
    type: Literal["regex"] = "regex"
    pattern: str
    ignore_case: bool = False


class MaxLatency(_Spec):
    type: Literal["max_latency_ms"] = "max_latency_ms"
    ms: float = Field(gt=0)


class ForbiddenSubstring(_Spec):
    """Guardrail: output must NOT contain any of ``values``."""

    type: Literal["forbidden_substring"] = "forbidden_substring"
    values: List[str] = Field(min_length=1)
    case_sensitive: bool = False


EvaluatorSpec = Annotated[
    Union[ExactMatch, Contains, Regex, MaxLatency, ForbiddenSubstring],
    Field(discriminator="type"),
]


# --------------------------------------------------------------------------
# Golden set + run results
# --------------------------------------------------------------------------


class GoldenCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    input: str = Field(min_length=1)
    evaluators: List[EvaluatorSpec] = Field(min_length=1)
    tags: List[str] = Field(default_factory=list)
    source: Literal["seed", "fault", "manual"] = "manual"
    fault_id: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)


class RunOutput(BaseModel):
    """What an injected callable may return instead of a bare string."""

    output: str
    latency_ms: Optional[float] = Field(default=None, ge=0)
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    steps: List[StepRecord] = Field(default_factory=list)
    tokens: Optional[TokenUsage] = None


class EvaluatorOutcome(BaseModel):
    evaluator: str
    passed: bool
    detail: str = ""


class CaseResult(BaseModel):
    case_id: str
    passed: bool
    output: Optional[str] = None
    latency_ms: Optional[float] = None
    outcomes: List[EvaluatorOutcome] = Field(default_factory=list)
    error: Optional[str] = None  # callable raised


class RunSummary(BaseModel):
    total: int
    passed: int
    failed: int
    errored: int
    pass_rate: float
    failed_case_ids: List[str] = Field(default_factory=list)
    by_evaluator: Dict[str, Dict[str, int]] = Field(default_factory=dict)


class RunResult(BaseModel):
    started_at: datetime
    ended_at: datetime
    results: List[CaseResult]
    summary: RunSummary
