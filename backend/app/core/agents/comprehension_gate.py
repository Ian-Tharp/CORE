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

Capability presence is checked at the ACTION level, not the root. A named tool
must resolve exactly to a registered root ("git") or capability id ("git.log");
an id the registry lacks ("git.push") is missing, and so is a gated operation
verb in the request ("push", "delete", "drop"; see `app.core.registry.actions`)
that no capability of the named roots serves. Without this, "git" lexically
matches git.status and "push main to origin" runs the loop.

Division of labor (kept deliberately narrow): the gate checks capability
*presence* only. Whether a plan can be *composed* is orchestration's job; whether
the result *achieves* the intent is evaluation's. Overloading the gate with those
produces over-conservative refusals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

from app.core.registry.actions import requested_operations
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


def _normalize_tool(name: str) -> str:
    return "_".join(str(name).strip().lower().replace("-", " ").split())


def resolve_tools(
    tools_needed: List[str], registry: CapabilityRegistry
) -> Tuple[List[str], List[str]]:
    """Split named tools into (present roots, missing names). Exact, not lexical.

    - "git" (a registered root)            -> root present
    - "git.log" (a registered id)          -> root "git" present
    - "git.push" (root present, id absent) -> missing "git.push"
    - "email", "web_search"                -> missing as named
    """
    roots = registry.roots()
    present: List[str] = []
    missing: List[str] = []
    for tool in tools_needed:
        name = _normalize_tool(tool)
        if not name:
            continue
        if name in roots:
            root = name
        elif registry.get(name) is not None:
            root = registry.get(name).root
        else:
            missing.append(name if "." in name else str(tool).strip())
            continue
        if root not in present:
            present.append(root)
    return present, missing


def unsupported_operations(
    roots: List[str], text: str, registry: CapabilityRegistry
) -> List[str]:
    """Gated verbs in ``text`` that no capability of ``roots`` serves, as
    "<root>.<verb>" (first root) so the refusal names the missing action."""
    if not roots:
        return []
    served = {v for r in roots for e in registry.for_root(r) for v in e.action_verbs()}
    return [f"{roots[0]}.{op}" for op in requested_operations(text) if op not in served]


def decide_gate(
    intent: UserIntent,
    registry: CapabilityRegistry,
    thresholds: GateThresholds | None = None,
    *,
    text: Optional[str] = None,
) -> GateDecision:
    """``text`` is the raw user input. When given, its operation verbs are
    checked along with the intent description's (the classifier may paraphrase
    "push main" as "update the remote")."""
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

    # 2) Understood, but a NAMED required tool, or the action asked of a present
    #    tool, is absent → honest refuse.
    if intent.requires_tools and intent.tools_needed:
        roots, missing = resolve_tools(intent.tools_needed, registry)
        if not missing:
            request_text = "\n".join(x for x in (text, intent.description) if x)
            missing = unsupported_operations(roots, request_text, registry)
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
    # For a missing action on a present root, list what that root can do.
    roots = registry.roots()
    present = [m.split(".", 1)[0] for m in missing if m.split(".", 1)[0] in roots]
    if present:
        names = [e.name for r in dict.fromkeys(present) for e in registry.for_root(r)]
    else:
        names = _capability_names(registry, " ".join(missing).replace(".", " "))
    can_do = ", ".join(names) or "nothing yet"
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
