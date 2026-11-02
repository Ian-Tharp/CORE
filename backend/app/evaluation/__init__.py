"""CORE evaluation layer (Phase I/II slice). See docs/evaluation_spec.md.

Distinct from ``app.eval`` (offline verdict-drift harness for
``evaluation_service``): this package captures faults and replays a Golden Set.
"""

from app.evaluation.capture import (
    capture_fault,
    fault_report_from_state,
    load_faults,
    state_failed,
)
from app.evaluation.golden_set import DuplicateCaseError, GoldenSet, LoadResult
from app.evaluation.models import (
    CaseResult,
    FaultReport,
    GoldenCase,
    RunOutput,
    RunResult,
    RunSummary,
)
from app.evaluation.runner import arun_cases, run_cases

__all__ = [
    "CaseResult",
    "DuplicateCaseError",
    "FaultReport",
    "GoldenCase",
    "GoldenSet",
    "LoadResult",
    "RunOutput",
    "RunResult",
    "RunSummary",
    "arun_cases",
    "capture_fault",
    "fault_report_from_state",
    "load_faults",
    "run_cases",
    "state_failed",
]
