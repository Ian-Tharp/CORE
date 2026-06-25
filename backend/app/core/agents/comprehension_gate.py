"""
The comprehension tri-state gate — the system's FRONT verifier.

Given a classified `UserIntent` and the `CapabilityRegistry`, decide one of:

- **proceed** — understood, and the needed capabilities are present → run the loop.
- **clarify** — plausibly understood but ambiguous or low-confidence → ask the user
  (surface as a base-model reply; no O→R→E spend).
- **refuse**  — a required capability is genuinely ABSENT → say so honestly, with
  what the system *can* do, instead of hallucinating a plan.

This is the cheapest, highest-leverage place to run a verifier: it prevents the
entire expensive deliberation loop from running on requests that are ambiguous or
impossible. It is pure (no LLM, no IO) so it is fully deterministic and testable;
the LLM-driven classification happens upstream in the comprehension agent.

Division of labor (kept deliberately narrow): the gate checks capability
*presence* only. Whether a plan can be *composed* is orchestration's job; whether
the result *achieves* the intent is evaluation's. Overloading the gate with those
produces over-conservative refusals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

from app.core.registry.capability import CapabilityRegistry
from app.models.core_state import UserIntent

GateOutcome = Literal["proceed", "clarify", "refuse"]


@dataclass(frozen=True)
class GateThresholds:
    """Confidence band for the gate. Below `tau_low` we ask; above `tau_high` we
    act; in between we clarify. Heuristic defaults; tune on the loop benchmark."""

    tau_low: float = 0.45
    tau_high: float = 0.75


@dataclass
class GateDecision:
    outcome: GateOutcome
    matched: List[str] = field(
        default_factory=list
    )  # capability ids that cover the intent
    missing: List[str] = field(
        default_factory=list
    )  # named tools absent from the registry
    message: str = ""  # user-facing text for clarify/refuse
    rationale: str = ""  # internal note


def _capability_names(
    registry: CapabilityRegistry, query: str, k: int = 4
) -> List[str]:
    hits = registry.search(query, k=k) or registry.all()[:k]
    return [e.name for e in hits]


def decide_gate(
    intent: UserIntent,
    registry: CapabilityRegistry,
    thresholds: GateThresholds | None = None,
) -> GateDecision:
    t = thresholds or GateThresholds()

    # Non-actionable intents (chat / follow-ups) are not gated here; they route
    # through the conversation node, not the O→R→E loop.
    if intent.type not in ("task", "question"):
        return GateDecision(outcome="proceed", rationale=f"intent.type={intent.type}")

    matched = [e.id for e in registry.search(intent.description, k=3)]

    # 1) Too unsure to even judge feasibility → ask.
    if intent.confidence < t.tau_low:
        return GateDecision(
            outcome="clarify",
            matched=matched,
            message=_clarify_message(intent, registry),
            rationale=f"confidence {intent.confidence:.2f} < tau_low {t.tau_low}",
        )

    # 2) Understood, but a NAMED required tool is absent → honest refuse.
    if intent.requires_tools and intent.tools_needed:
        missing = [
            tool for tool in intent.tools_needed if not registry.has_capability(tool)
        ]
        if missing:
            return GateDecision(
                outcome="refuse",
                matched=matched,
                missing=missing,
                message=_refuse_message(missing, registry),
                rationale=f"absent capabilities: {missing}",
            )

    # 3) Understood and feasible, but ambiguous or mid-confidence → clarify.
    if intent.ambiguities or intent.confidence < t.tau_high:
        return GateDecision(
            outcome="clarify",
            matched=matched,
            message=_clarify_message(intent, registry),
            rationale=(
                f"ambiguities={len(intent.ambiguities)}, "
                f"confidence {intent.confidence:.2f} < tau_high {t.tau_high}"
            ),
        )

    # 4) Clear + feasible → run the loop.
    return GateDecision(
        outcome="proceed", matched=matched, rationale="clear and feasible"
    )


def _refuse_message(missing: List[str], registry: CapabilityRegistry) -> str:
    can_do = ", ".join(_capability_names(registry, " ".join(missing))) or "nothing yet"
    return (
        f"I can't do that — I don't have a capability for: {', '.join(missing)}. "
        f"Here's what I can do: {can_do}."
    )


def _clarify_message(intent: UserIntent, registry: CapabilityRegistry) -> str:
    can_do = ", ".join(_capability_names(registry, intent.description))
    if intent.ambiguities:
        asks = " ".join(intent.ambiguities)
        return f"I want to make sure I understand — {asks} (I can: {can_do}.)"
    return (
        f"I'm not fully sure what you mean. Could you clarify? "
        f"For context, I can: {can_do}."
    )
