"""
The comprehension tri-state gate: proceed / clarify / refuse.

Pure logic over (UserIntent, CapabilityRegistry, thresholds) — no model, no
network — so the highest-leverage verifier in the system is fully testable.
"""

from __future__ import annotations

from app.core.agents.comprehension_gate import decide_gate, GateThresholds
from app.core.registry.capability import CapabilityEntry, CapabilityRegistry
from app.models.core_state import UserIntent


def _registry() -> CapabilityRegistry:
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(
            id="file_operations.read",
            name="Read file",
            description="Read the contents of a file in the workspace",
            side_effects="read",
            examples=["read a file"],
        )
    )
    return reg


def _intent(**kw) -> UserIntent:
    return UserIntent(
        type=kw.get("type", "task"),
        description=kw.get("description", "do something"),
        confidence=kw.get("confidence", 0.9),
        requires_tools=kw.get("requires_tools", False),
        tools_needed=kw.get("tools_needed", []),
        ambiguities=kw.get("ambiguities", []),
    )


def test_proceed_when_clear_and_capability_present():
    intent = _intent(
        description="read the file config.py",
        confidence=0.9,
        requires_tools=True,
        tools_needed=["file_operations"],
    )
    d = decide_gate(intent, _registry())
    assert d.outcome == "proceed"


def test_proceed_when_no_tools_required():
    intent = _intent(description="what is 2+2", confidence=0.95, requires_tools=False)
    d = decide_gate(intent, _registry())
    assert d.outcome == "proceed"


def test_refuse_when_required_capability_absent():
    intent = _intent(
        description="email my boss the report",
        confidence=0.9,
        requires_tools=True,
        tools_needed=["email"],
    )
    d = decide_gate(intent, _registry())
    assert d.outcome == "refuse"
    assert "email" in d.missing
    assert d.message  # honest, user-facing


def test_clarify_when_ambiguities_present():
    intent = _intent(
        description="read it",
        confidence=0.9,
        requires_tools=True,
        tools_needed=["file_operations"],
        ambiguities=["what does 'it' refer to?"],
    )
    d = decide_gate(intent, _registry())
    assert d.outcome == "clarify"


def test_clarify_when_confidence_in_mid_band():
    intent = _intent(
        description="read the file",
        confidence=0.6,
        requires_tools=True,
        tools_needed=["file_operations"],
    )
    d = decide_gate(intent, _registry())
    assert d.outcome == "clarify"


def test_clarify_when_confidence_below_low_even_if_capability_present():
    intent = _intent(
        description="read the file",
        confidence=0.2,
        requires_tools=True,
        tools_needed=["file_operations"],
    )
    d = decide_gate(intent, _registry())
    assert d.outcome == "clarify"


def test_refuse_precedence_over_clarify_when_understood_but_missing():
    # Understood well enough (conf above low) but the needed tool is absent →
    # refuse, not clarify. Don't ask the user to refine something we can't do.
    intent = _intent(
        description="send a text message",
        confidence=0.85,
        requires_tools=True,
        tools_needed=["sms"],
        ambiguities=["which number?"],
    )
    d = decide_gate(intent, _registry())
    assert d.outcome == "refuse"


def test_conversation_intent_passes_through():
    intent = _intent(type="conversation", description="how are you", confidence=0.9)
    d = decide_gate(intent, _registry())
    assert d.outcome == "proceed"


def test_thresholds_are_configurable():
    intent = _intent(
        description="read the file",
        confidence=0.6,
        requires_tools=True,
        tools_needed=["file_operations"],
    )
    # Lower tau_high below the confidence → now it proceeds.
    d = decide_gate(intent, _registry(), GateThresholds(tau_low=0.3, tau_high=0.5))
    assert d.outcome == "proceed"
