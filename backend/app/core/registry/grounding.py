"""
The plan-grounding engine — orchestration's self-grounding back-half (Phase 2).

A draft `ExecutionPlan` (the orchestration agent's LLM output) goes in; a plan
whose steps are all **registry-backed** and **schema-valid** comes out, together
with a `GroundingReport` of what was dropped and why.

This is the orchestration-side analogue of the comprehension front gate: where
the gate checks capability *presence* over the whole intent, grounding checks
that each *concrete step* names a real dispatcher tool, resolves to a real
registry capability, and carries params that satisfy that capability's schema.

Design constraints (mirrors comprehension_gate.py):
- PURE: registry + plan in, grounded plan + report out. No LLM, no IO.
- NEVER raises on a bad step — bad steps become `DroppedStep` entries.
- NEVER mutates the input plan/steps — `ground_plan` returns a NEW ExecutionPlan
  preserving id/goal/revision/reasoning, because the revise_plan loop passes the
  previous plan back into create_plan and re-grounds it.

The dispatcher-root vs registry-action-id wrinkle is reconciled via
`capability_id_for_step` (homed in builtin.py next to the _BUILTIN vocabulary).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from app.models.core_state import ExecutionPlan, PlanStep
from app.core.registry.capability import CapabilityRegistry
from app.core.registry.builtin import capability_id_for_step
from app.core.registry._schema_validate import validate_params

# Re-export so callers can import the reconciliation helper from one place.
__all__ = [
    "capability_id_for_step",
    "DroppedStep",
    "GroundingReport",
    "ground_plan",
]

DropReason = Literal["invented_tool", "unknown_action", "invalid_params"]


@dataclass(frozen=True)
class DroppedStep:
    """One plan step removed during grounding, with the reason it failed."""

    step_id: str
    step_name: str
    tool: Optional[str]  # root tool as emitted by the LLM
    reason: DropReason
    detail: str
    capability_id: Optional[str]  # the f"{root}.{action}" attempted, if any

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_name": self.step_name,
            "tool": self.tool,
            "reason": self.reason,
            "detail": self.detail,
            "capability_id": self.capability_id,
        }


@dataclass(frozen=True)
class GroundingReport:
    """Outcome of grounding a plan: survivors, drops, and orphaned dependents."""

    grounded_step_ids: List[str] = field(default_factory=list)
    dropped: List[DroppedStep] = field(default_factory=list)
    orphaned_dependents: List[str] = field(default_factory=list)

    def is_clean(self) -> bool:
        """True when nothing was dropped."""
        return not self.dropped

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grounded_step_ids": list(self.grounded_step_ids),
            "dropped": [d.to_dict() for d in self.dropped],
            "orphaned_dependents": list(self.orphaned_dependents),
        }


def _params_without_action(params: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the dispatcher 'action' key — it is not part of any params_schema."""
    return {k: v for k, v in params.items() if k != "action"}


def ground_plan(
    plan: ExecutionPlan,
    registry: CapabilityRegistry,
    roots: Set[str],
    *,
    drop: bool = True,
) -> Tuple[ExecutionPlan, GroundingReport]:
    """Ground `plan` against `registry`. PURE; returns (grounded_plan, report).

    Per-step rule:
      1. step.tool is None        -> KEEP (legit LLM-only step; never flagged).
      2. step.tool not in roots   -> DROP, reason="invented_tool".
      3. capability id not in registry -> DROP, reason="unknown_action".
      4. params (minus 'action') fail params_schema -> DROP, reason="invalid_params".
      5. otherwise KEEP.

    With drop=True (default) failing steps are removed from the returned plan.
    With drop=False they are marked status="skipped" and kept (report-only mode).
    Surviving steps whose dependencies reference a dropped id are recorded in
    `report.orphaned_dependents` (report-only; reasoning treats unknown dep ids
    as satisfied, so this never deadlocks).
    """
    survivors: List[PlanStep] = []
    skipped: List[PlanStep] = []
    dropped: List[DroppedStep] = []

    for step in plan.steps:
        # 1) LLM-only step — always legitimate.
        if step.tool is None:
            survivors.append(step)
            continue

        # 2) invented tool — not a registry-backed dispatcher root.
        if step.tool not in roots:
            dropped.append(
                DroppedStep(
                    step_id=step.id,
                    step_name=step.name,
                    tool=step.tool,
                    reason="invented_tool",
                    detail=(
                        f"tool '{step.tool}' is not a registry-backed dispatcher tool"
                    ),
                    capability_id=None,
                )
            )
            skipped.append(step)
            continue

        # 3) real root, but the composed action has no capability entry.
        cap_id = capability_id_for_step(step.tool, step.params)
        entry = registry.get(cap_id)
        if entry is None:
            dropped.append(
                DroppedStep(
                    step_id=step.id,
                    step_name=step.name,
                    tool=step.tool,
                    reason="unknown_action",
                    detail=f"no registry capability '{cap_id}'",
                    capability_id=cap_id,
                )
            )
            skipped.append(step)
            continue

        # 4) param-schema validation (action key excluded; {} accepts).
        err = validate_params(_params_without_action(step.params), entry.params_schema)
        if err is not None:
            dropped.append(
                DroppedStep(
                    step_id=step.id,
                    step_name=step.name,
                    tool=step.tool,
                    reason="invalid_params",
                    detail=err,
                    capability_id=cap_id,
                )
            )
            skipped.append(step)
            continue

        # 5) grounded.
        survivors.append(step)

    dropped_ids = {d.step_id for d in dropped}

    # Build the NEW step list without mutating the input steps.
    if drop:
        kept_steps = [s.model_copy(deep=True) for s in survivors]
    else:
        kept_steps = []
        for s in plan.steps:
            copy = s.model_copy(deep=True)
            if s.id in dropped_ids:
                copy.status = "skipped"
            kept_steps.append(copy)

    # 5b) orphaned dependents: surviving steps that depend on a dropped step.
    surviving_ids = {s.id for s in kept_steps if s.id not in dropped_ids}
    orphaned = [
        s.id
        for s in kept_steps
        if s.id in surviving_ids and any(dep in dropped_ids for dep in s.dependencies)
    ]

    grounded_plan = ExecutionPlan(
        id=plan.id,
        goal=plan.goal,
        steps=kept_steps,
        created_at=plan.created_at,
        updated_at=plan.updated_at,
        revision=plan.revision,
        reasoning=plan.reasoning,
    )

    report = GroundingReport(
        grounded_step_ids=[s.id for s in kept_steps if s.id not in dropped_ids],
        dropped=dropped,
        orphaned_dependents=orphaned,
    )
    return grounded_plan, report
