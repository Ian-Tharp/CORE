"""Fault capture: COREState -> FaultReport -> JSONL on disk.

``fault_report_from_state`` is duck-typed (``getattr`` only) so this module
does not import the graph, the agents or any service.

The live pipeline calls ``maybe_capture_fault`` from
``COREGraph.conversation_node`` (core_graph_v2). It is gated by
``CORE_EVAL_CAPTURE_FAULTS`` (default off) and never raises: a capture failure
is logged at warning level and the user's request carries on.
``capture_fault`` itself still raises on I/O errors, for callers that want to
know.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

from app.config.evaluation import FaultCaptureConfig, get_fault_capture_config
from app.evaluation.models import (
    FaultReport,
    StepRecord,
    ToolCallRecord,
)

FAULTS_FILENAME = "faults.jsonl"

logger = logging.getLogger(__name__)

# Evaluator still wanted more work when the run ended: only reachable at the
# end of a run when the revision loop guard forced finalization.
_UNRESOLVED_ACTIONS = ("revise_plan", "retry_step")


def fault_reasons(state: Any) -> List[str]:
    """Why a finished run counts as a fault; empty list means it doesn't.

    - ``errors``: a node hit an exception path (``state.errors``), including
      evaluation crashes that the graph papers over with a default "success".
    - ``eval_failure``: the evaluator judged the run a failure.
    - ``step_failure``: at least one plan step failed.
    - ``unresolved_evaluation``: the run ended while the evaluator still asked
      to revise/retry, i.e. the plan-revision loop guard forced finalization.
    - ``unsupported_claims``: an action claim in the output that no dispatched
      tool call supports (``state.claim_report``, see app/evaluation/claims.py).

    A front-gate refusal or clarification is the gate working as designed, not
    a fault, so it is not a reason on its own. Likewise a write held for the
    user's confirmation (and the steps waiting on it) is not a step failure.
    """
    reasons: List[str] = []
    if getattr(state, "errors", None):
        reasons.append("errors")
    claims = getattr(state, "claim_report", None)
    if isinstance(claims, dict) and claims.get("unsupported"):
        reasons.append("unsupported_claims")
    ev = getattr(state, "eval_result", None)
    if ev is not None and getattr(ev, "overall_status", None) == "failure":
        reasons.append("eval_failure")
    waiting = _awaiting_confirmation_ids(state)
    if any(
        getattr(r, "status", None) == "failure"
        and getattr(r, "step_id", None) not in waiting
        for r in (getattr(state, "step_results", None) or [])
    ):
        reasons.append("step_failure")
    if ev is not None and getattr(ev, "next_action", None) in _UNRESOLVED_ACTIONS:
        reasons.append("unresolved_evaluation")
    return reasons


def _awaiting_confirmation_ids(state: Any) -> set:
    """Steps held for confirmation, or blocked only because they wait on one."""
    ids = {
        getattr(p, "step_id", None)
        for p in (getattr(state, "pending_confirmations", None) or [])
    }
    ids.update(
        getattr(b, "step_id", None)
        for b in (getattr(state, "blocked_actions", None) or [])
        if getattr(b, "kind", None) == "awaiting_confirmation"
    )
    ids.discard(None)
    return ids


def state_failed(state: Any) -> bool:
    """True if the finished run is a fault (see ``fault_reasons``)."""
    return bool(fault_reasons(state))


def fault_report_from_state(state: Any, expected: Optional[str] = None) -> FaultReport:
    results = list(getattr(state, "step_results", None) or [])
    by_id = {getattr(r, "step_id", None): r for r in results}

    steps: List[StepRecord] = [
        StepRecord(name=str(n), status="visited")
        for n in (getattr(state, "execution_history", None) or [])
    ]
    for r in results:
        steps.append(
            StepRecord(
                name=str(r.step_id),
                status=r.status,
                error=r.error,
                latency_ms=max(0.0, float(r.duration_seconds or 0.0)) * 1000,
            )
        )

    tool_calls: List[ToolCallRecord] = []
    plan = getattr(state, "plan", None)
    for ps in getattr(plan, "steps", None) or []:
        if not getattr(ps, "tool", None):
            continue
        r = by_id.get(ps.id)
        # A result that says the dispatcher did NOT run this tool (LLM-executed,
        # dry-run or blocked step) must not appear as a tool call in the trace.
        # Results without the field (older/duck-typed states) keep the plan view.
        if r is not None and hasattr(r, "dispatched_tool") and not r.dispatched_tool:
            continue
        tool_calls.append(
            ToolCallRecord(
                name=ps.tool,
                params=dict(ps.params or {}),
                result=dict(r.outputs) if r is not None and r.outputs else None,
                error=r.error if r is not None else None,
            )
        )

    started = getattr(state, "started_at", None)
    ended = getattr(state, "completed_at", None)
    latency = (
        max(0.0, (ended - started).total_seconds() * 1000)
        if started is not None and ended is not None
        else None
    )
    metadata = {}
    refusal = getattr(state, "refusal", None)
    if refusal is not None:
        metadata["refusal"] = (
            refusal.model_dump() if hasattr(refusal, "model_dump") else refusal
        )
    blocked = getattr(state, "blocked_actions", None) or []
    if blocked:
        metadata["blocked_actions"] = [
            b.model_dump() if hasattr(b, "model_dump") else b for b in blocked
        ]
    pending = getattr(state, "pending_confirmations", None) or []
    if pending:
        metadata["pending_confirmations"] = [
            p.model_dump() if hasattr(p, "model_dump") else p for p in pending
        ]
    claims = getattr(state, "claim_report", None)
    if isinstance(claims, dict) and claims.get("unsupported"):
        metadata["unsupported_claims"] = claims["unsupported"]
    return FaultReport(
        run_id=getattr(state, "run_id", None),
        input=getattr(state, "user_input", "") or "",
        steps=steps,
        tool_calls=tool_calls,
        expected=expected,
        actual=getattr(state, "response", None),
        errors=list(getattr(state, "errors", None) or []),
        started_at=started,
        ended_at=ended,
        latency_ms=latency,
        metadata=metadata,
    )


def capture_fault(report: FaultReport, directory: str | Path) -> Path:
    """Append ``report`` to ``<directory>/faults.jsonl`` and return the path."""
    path = Path(directory) / FAULTS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(report.model_dump_json() + "\n")
    return path


def maybe_capture_fault(
    state: Any, config: Optional[FaultCaptureConfig] = None
) -> Optional[Path]:
    """Capture ``state`` if capture is enabled and the run faulted. Never raises.

    Returns the JSONL path on a write, ``None`` otherwise (disabled, no fault,
    or the capture itself failed, which is logged at warning level).
    """
    try:
        cfg = config if config is not None else get_fault_capture_config()
        if not cfg.enabled:
            return None
        reasons = fault_reasons(state)
        if not reasons:
            return None
        report = fault_report_from_state(state)
        ev = getattr(state, "eval_result", None)
        report.metadata.update(
            {
                "source": "core_graph_v2",
                "fault_reasons": reasons,
                "gate_outcome": getattr(state, "gate_outcome", None),
                "plan_revisions": getattr(state, "plan_revisions", None),
                "eval_status": getattr(ev, "overall_status", None),
                "eval_next_action": getattr(ev, "next_action", None),
            }
        )
        return capture_fault(report, cfg.directory)
    except Exception as exc:
        try:
            run_id = getattr(state, "run_id", None)
        except Exception:
            run_id = None
        logger.warning(
            "Fault capture failed; request unaffected (run_id=%s): %s: %s",
            run_id,
            type(exc).__name__,
            exc,
        )
        return None


def load_faults(directory: str | Path) -> List[FaultReport]:
    """Read back captured faults, skipping unparseable lines."""
    path = Path(directory) / FAULTS_FILENAME
    out: List[FaultReport] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            out.append(FaultReport.model_validate_json(line))
        except ValueError:
            continue
    return out
