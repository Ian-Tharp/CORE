"""Fault capture: COREState -> FaultReport -> JSONL on disk.

``fault_report_from_state`` is duck-typed (``getattr`` only) so this module
does not import the graph, the agents or any service. It is deliberately NOT
wired into ``core_graph_v2`` yet; see docs/evaluation_spec.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from app.evaluation.models import (
    FaultReport,
    StepRecord,
    ToolCallRecord,
)

FAULTS_FILENAME = "faults.jsonl"


def state_failed(state: Any) -> bool:
    """True if the run recorded errors, a failed eval, a failed step, or an
    action claim that no tool call supports."""
    if getattr(state, "errors", None):
        return True
    claims = getattr(state, "claim_report", None)
    if isinstance(claims, dict) and claims.get("unsupported"):
        return True
    ev = getattr(state, "eval_result", None)
    if ev is not None and getattr(ev, "overall_status", None) == "failure":
        return True
    return any(
        getattr(r, "status", None) == "failure"
        for r in (getattr(state, "step_results", None) or [])
    )


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
