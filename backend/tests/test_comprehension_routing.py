"""
Routing after comprehension: the tri-state gate outcome decides whether the loop
runs. proceed (+ actionable intent) → orchestration; clarify/refuse → conversation
(surface the message, no O→R→E spend).
"""

from __future__ import annotations

from app.core.agents.comprehension_routing import route_after_comprehension
from app.models.core_state import UserIntent


def _intent(type_="task"):
    return UserIntent(type=type_, description="x", confidence=0.9)


def test_proceed_task_goes_to_orchestration():
    assert route_after_comprehension(_intent("task"), "proceed") == "orchestration"


def test_proceed_question_goes_to_orchestration():
    assert route_after_comprehension(_intent("question"), "proceed") == "orchestration"


def test_proceed_conversation_goes_to_conversation():
    assert (
        route_after_comprehension(_intent("conversation"), "proceed") == "conversation"
    )


def test_clarify_goes_to_conversation():
    assert route_after_comprehension(_intent("task"), "clarify") == "conversation"


def test_refuse_goes_to_conversation():
    assert route_after_comprehension(_intent("task"), "refuse") == "conversation"


def test_missing_intent_defaults_to_conversation():
    assert route_after_comprehension(None, None) == "conversation"


def test_missing_gate_falls_back_to_intent_type():
    # If the gate hasn't run (gate_outcome None), route by intent type — don't
    # silently block. The gate only *blocks* on an explicit clarify/refuse.
    assert route_after_comprehension(_intent("task"), None) == "orchestration"
    assert route_after_comprehension(_intent("conversation"), None) == "conversation"
