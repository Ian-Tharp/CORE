"""CORE evaluation layer (Phase I/II slice). See docs/evaluation_spec.md.

Distinct from ``app.eval`` (offline verdict-drift harness for
``evaluation_service``): this package captures faults and replays a Golden Set.
"""

from app.evaluation.capture import (
    capture_fault,
    fault_reasons,
    fault_report_from_state,
    load_faults,
    maybe_capture_fault,
    state_failed,
)
from app.evaluation.claims import (
    ClaimCheck,
    ClaimReport,
    extract_claims,
    trace_from_state,
    verify_claims,
    verify_state_claims,
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
    "ClaimCheck",
    "ClaimReport",
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
    "extract_claims",
    "fault_reasons",
    "fault_report_from_state",
    "load_faults",
    "maybe_capture_fault",
    "run_cases",
    "state_failed",
    "trace_from_state",
    "verify_claims",
    "verify_state_claims",
]
