"""
Per-step evaluation (comprehension-gate spec Phase 3, §4.4).

A cheap, deterministic check run after each plan step executes. It answers only
"did this step visibly work?", not "did the run achieve the intent?" (the final
evaluation still owns that):

- ``no_result``      the step has no StepResult (never ran)
- ``error``          the result failed or carries an error
- ``partial``        the result reports partial success
- ``no_output``      the result has neither outputs nor artifacts
- ``schema_mismatch`` outputs don't match the step's declared ``output_schema``

No LLM, no IO. Results are recorded on ``COREState.step_checks`` and failing
ones are passed to the final evaluation so they can't be silently finalized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional

from app.core.registry._schema_validate import validate_params
from app.models.core_state import ExecutionPlan, PlanStep, StepResult

StepFailure = Literal["no_result", "error", "partial", "no_output", "schema_mismatch"]


@dataclass(frozen=True)
class StepCheck:
    """Verdict of the deterministic check for one plan step."""

    step_id: str
    step_name: str
    passed: bool
    failures: List[StepFailure] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_name": self.step_name,
            "passed": self.passed,
            "failures": list(self.failures),
            "detail": self.detail,
        }


def check_step(step: PlanStep, result: Optional[StepResult]) -> StepCheck:
    """Check one step against its result. Never raises."""
    if result is None:
        return StepCheck(
            step_id=step.id,
            step_name=step.name,
            passed=False,
            failures=["no_result"],
            detail="step produced no result",
        )

    failures: List[StepFailure] = []
    notes: List[str] = []

    if result.status == "failure" or result.error:
        failures.append("error")
        notes.append(result.error or "status=failure")
    elif result.status == "partial":
        failures.append("partial")
        notes.append("status=partial")

    has_output = any(v not in (None, "", [], {}) for v in result.outputs.values())
    if not has_output and not result.artifacts:
        failures.append("no_output")
        notes.append("no outputs or artifacts")

    schema = step.output_schema
    if schema and result.outputs:
        problem = validate_params(result.outputs, schema)
        if problem:
            failures.append("schema_mismatch")
            notes.append(f"output schema: {problem}")

    return StepCheck(
        step_id=step.id,
        step_name=step.name,
        passed=not failures,
        failures=failures,
        detail="; ".join(notes),
    )


def check_steps(
    plan: Optional[ExecutionPlan], step_results: Iterable[StepResult]
) -> List[StepCheck]:
    """Check every step that ran or should have run.

    Steps still ``pending`` with no result are skipped: they may simply not have
    been reached yet (e.g. a retry started mid-plan). Steps marked ``skipped`` by
    the executor are also skipped. The latest result per step wins.
    """
    if plan is None:
        return []
    latest: Dict[str, StepResult] = {}
    for result in step_results:
        latest[result.step_id] = result

    checks: List[StepCheck] = []
    for step in plan.steps:
        result = latest.get(step.id)
        if result is None and step.status in ("pending", "skipped"):
            continue
        checks.append(check_step(step, result))
    return checks


def failing_checks(checks: Iterable[Any]) -> List[Dict[str, Any]]:
    """Failing checks as dicts, from StepCheck objects or their dict form."""
    out: List[Dict[str, Any]] = []
    for check in checks or []:
        data = check.to_dict() if isinstance(check, StepCheck) else dict(check)
        if not data.get("passed", True):
            out.append(data)
    return out
