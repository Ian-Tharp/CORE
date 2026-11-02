"""Fault Report: a structured record of a failed pipeline run."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ToolCallRecord(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = Field(default=None, ge=0)


class StepRecord(BaseModel):
    """One pipeline step (e.g. Comprehension, Reasoning)."""

    name: str
    input: Optional[Any] = None
    output: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    latency_ms: Optional[float] = Field(default=None, ge=0)
    prompt_tokens: Optional[int] = Field(default=None, ge=0)
    completion_tokens: Optional[int] = Field(default=None, ge=0)


class FaultReport(BaseModel):
    """Everything needed to understand (and later replay) a failed run."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    input: str
    steps: list[StepRecord] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    expected: Optional[str] = None
    actual: Optional[str] = None
    error: Optional[str] = None
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: Optional[datetime] = None
    latency_ms: Optional[float] = Field(default=None, ge=0)
    prompt_tokens: Optional[int] = Field(default=None, ge=0)
    completion_tokens: Optional[int] = Field(default=None, ge=0)
    total_tokens: Optional[int] = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _derive(self) -> "FaultReport":
        # Fill gaps from what is known; never overwrite explicit values.
        if self.latency_ms is None and self.finished_at is not None:
            delta = (self.finished_at - self.started_at).total_seconds() * 1000
            self.latency_ms = max(delta, 0.0)
        if self.total_tokens is None and (
            self.prompt_tokens is not None or self.completion_tokens is not None
        ):
            self.total_tokens = (self.prompt_tokens or 0) + (
                self.completion_tokens or 0
            )
        return self
