"""CORE evaluation layer (Phase I/II slice): fault capture + golden set.

See docs/evaluation_spec.md. Distinct from ``app.eval`` (the offline
verdict-drift benchmark). No LLM calls, DB, or network here.
"""

from app.evaluation.capture import capture_fault, is_capture_enabled
from app.evaluation.golden_set import GoldenCase, GoldenSet
from app.evaluation.models import FaultReport, StepRecord, ToolCallRecord
from app.evaluation.runner import RunOutput, RunResult, run_cases

__all__ = [
    "FaultReport",
    "StepRecord",
    "ToolCallRecord",
    "GoldenCase",
    "GoldenSet",
    "RunOutput",
    "RunResult",
    "run_cases",
    "capture_fault",
    "is_capture_enabled",
]
