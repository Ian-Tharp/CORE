"""
Action guard: capability side effects decide what may run without the user.

Two rules, both deterministic (no LLM):

**Confirm tier.** Every dispatchable step is classified by the capability
registry (``side_effects``: none / read / network / write; see
:func:`app.core.registry.builtin.side_effect_for_step`). A ``write`` step, or a
step the plan marked ``requires_hitl``, is *held*: it is never handed to the
dispatcher, comes back as a failed ``StepResult`` (so its dependents wait too)
and is recorded as a :class:`PendingConfirmation`. Reads and network fetches
run as before. The user approves by re-sending the request with the capability
id (or ``"*"``) in ``config["approved_capabilities"]``; that is a new run.

**Dependency latch.** A step whose dependency did not succeed (failed, was
blocked, was held, or never ran because of a dependency cycle) does not run,
on the first pass *and* on a retry. It is recorded as a :class:`BlockedAction`
with ``kind="dependency_failed"``, or ``kind="awaiting_confirmation"`` when the
dependency it waits on is held for confirmation. A retry that makes the
dependency succeed resumes its dependency-blocked dependents.

Steps that the dispatcher would not run (LLM steps, dry runs) are not held:
nothing they do touches the outside world. The claim verifier covers LLM
steps that say they changed something.

``CORE_CONFIRM_WRITES=0`` turns the confirm tier off (for ablation runs). The
dependency latch has no switch.
"""

from __future__ import annotations

import os
import threading
from typing import Iterable, List, Optional, Set

from app.core.registry.builtin import side_effect_for_step
from app.core.registry.capability import CapabilityRegistry
from app.models.core_state import BlockedAction, PendingConfirmation, PlanStep

APPROVE_ALL = "*"


def confirm_writes_enabled() -> bool:
    return os.getenv("CORE_CONFIRM_WRITES", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _target(params: dict) -> str:
    for key in ("path", "url", "endpoint", "pattern"):
        value = (params or {}).get(key)
        if value:
            return str(value)[:120]
    return ""


def mark_confirmation_steps(plan, registry: Optional[CapabilityRegistry]) -> int:
    """Set ``requires_hitl`` on every write step of ``plan``; return how many.

    Keeps the plan honest about its checkpoints (the orchestration cost
    estimate and the UI read the flag). The guard holds write steps whether or
    not the flag is set, so this is a report, not the enforcement.
    """
    if plan is None:
        return 0
    marked = 0
    for step in plan.steps:
        if not step.tool or step.requires_hitl:
            continue
        _, effect = side_effect_for_step(step.tool, step.params, registry)
        if effect == "write":
            step.requires_hitl = True
            marked += 1
    return marked


class ActionGuard:
    """Per-run, thread-safe record of held and dependency-blocked steps.

    One instance per reasoning pass; the agents are shared across runs, so the
    guard must never live on the agent.
    """

    def __init__(
        self,
        registry: Optional[CapabilityRegistry] = None,
        approved: Optional[Iterable[str]] = None,
        confirm: bool = True,
        awaiting: Iterable[str] = (),
    ) -> None:
        self._lock = threading.Lock()
        self._registry = registry
        self._approved: Set[str] = {str(a) for a in (approved or [])}
        self._confirm = confirm
        self._pending: List[PendingConfirmation] = []
        self._blocked: List[BlockedAction] = []
        # Step ids held (or waiting on a held step), seeded from earlier
        # passes of the same run so a retry still knows what is held.
        self._awaiting: Set[str] = set(awaiting)

    @property
    def confirm_enabled(self) -> bool:
        return self._confirm

    @property
    def pending(self) -> List[PendingConfirmation]:
        with self._lock:
            return list(self._pending)

    @property
    def blocked(self) -> List[BlockedAction]:
        with self._lock:
            return list(self._blocked)

    def is_awaiting(self, step_id: str) -> bool:
        """True if ``step_id`` is held, or waits on a held step."""
        with self._lock:
            return step_id in self._awaiting

    def approved(self, capability_id: str) -> bool:
        return APPROVE_ALL in self._approved or capability_id in self._approved

    def check_confirmation(self, step: PlanStep) -> Optional[PendingConfirmation]:
        """Hold ``step`` if it needs the user's go-ahead; returns the record.

        Call only for steps the dispatcher would actually run.
        """
        if not self._confirm or not step.tool:
            return None
        cap_id, effect = side_effect_for_step(step.tool, step.params, self._registry)
        if effect == "write":
            reason = "write"
        elif step.requires_hitl:
            reason = "requires_hitl"
        else:
            return None
        if self.approved(cap_id):
            return None
        held = PendingConfirmation(
            step_id=step.id,
            step_name=step.name,
            tool=step.tool,
            capability_id=cap_id,
            side_effect=effect,
            reason=reason,
            target=_target(step.params),
        )
        with self._lock:
            self._pending.append(held)
            self._awaiting.add(step.id)
        return held

    def block_dependent(
        self, step: PlanStep, cause_step_id: str, detail: str = ""
    ) -> BlockedAction:
        """Record that ``step`` did not run because ``cause_step_id`` didn't succeed."""
        with self._lock:
            awaiting = cause_step_id in self._awaiting
            if awaiting:
                self._awaiting.add(step.id)
            blocked = BlockedAction(
                step_id=step.id,
                step_name=step.name,
                tool=step.tool,
                kind="awaiting_confirmation" if awaiting else "dependency_failed",
                cause_step_id=cause_step_id,
                reason=(
                    f"waits on step {cause_step_id}, held for confirmation"
                    if awaiting
                    else f"dependency {cause_step_id} {detail or 'failed'}"
                ),
            )
            self._blocked.append(blocked)
            return blocked
