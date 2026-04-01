"""
Tests for COREGraph routing functions — pure decision logic with no LLM calls.

Sentinel-value contract:
- Remove intent.type check in route_from_comprehension → all inputs go to conversation → test fails
- Remove state.intent None guard → NoneType error on missing intent → test fails
- Remove plan_revisions loop guard → revise_plan loops infinitely → test fails
- Remove action_routes mapping → all eval results default to conversation → test fails
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.models.core_state import COREState, EvaluationResult, UserIntent
from app.core.langgraph.core_graph_v2 import COREGraph, _MAX_PLAN_REVISIONS


# ===========================================================================
# Helpers
# ===========================================================================

def _make_state(**kwargs) -> COREState:
    defaults = {"messages": [], "user_input": "test"}
    defaults.update(kwargs)
    return COREState(**defaults)


def _make_graph() -> COREGraph:
    # We only need the routing methods; avoid __init__ graph building
    graph = object.__new__(COREGraph)
    return graph


def _make_intent(intent_type: str = "task") -> UserIntent:
    """Build a UserIntent. For unknown types, returns a MagicMock."""
    if intent_type not in ("task", "conversation", "question", "clarification"):
        mock = MagicMock(spec=UserIntent)
        mock.type = intent_type
        return mock
    return UserIntent(type=intent_type, description="test intent", confidence=0.9)


def _make_eval_result(next_action: str = "finalize") -> EvaluationResult:
    result = MagicMock(spec=EvaluationResult)
    result.next_action = next_action
    return result


# ===========================================================================
# route_from_comprehension
# ===========================================================================

class TestRouteFromComprehension:
    def test_no_intent_routes_to_conversation(self):
        """
        SENTINEL — missing intent must default to 'conversation'.
        Remove None guard → AttributeError on intent.type → pipeline crashes → test fails.
        """
        graph = _make_graph()
        state = _make_state(intent=None)
        result = graph.route_from_comprehension(state)
        assert result == "conversation"

    def test_task_intent_routes_to_orchestration(self):
        """
        SENTINEL — 'task' intent must route to orchestration for planning.
        Route all intents to conversation → tasks never planned → test fails.
        """
        graph = _make_graph()
        state = _make_state(intent=_make_intent("task"))
        result = graph.route_from_comprehension(state)
        assert result == "orchestration"

    def test_question_intent_routes_to_orchestration(self):
        """
        SENTINEL — 'question' intent must also route to orchestration.
        Only route 'task' → questions never answered by pipeline → test fails.
        """
        graph = _make_graph()
        state = _make_state(intent=_make_intent("question"))
        result = graph.route_from_comprehension(state)
        assert result == "orchestration"

    def test_conversation_intent_routes_to_conversation(self):
        """'conversation' intent must route to conversation node."""
        graph = _make_graph()
        state = _make_state(intent=_make_intent("conversation"))
        result = graph.route_from_comprehension(state)
        assert result == "conversation"

    def test_clarification_intent_routes_to_conversation(self):
        """'clarification' intent must route to conversation node."""
        graph = _make_graph()
        state = _make_state(intent=_make_intent("clarification"))
        result = graph.route_from_comprehension(state)
        assert result == "conversation"

    def test_unknown_intent_type_routes_to_conversation(self):
        """Unknown intent types must default to conversation."""
        graph = _make_graph()
        state = _make_state(intent=_make_intent("SENTINEL_UNKNOWN"))
        result = graph.route_from_comprehension(state)
        assert result == "conversation"


# ===========================================================================
# route_from_evaluation
# ===========================================================================

class TestRouteFromEvaluation:
    def test_no_eval_result_routes_to_conversation(self):
        """
        SENTINEL — missing eval_result must default to 'conversation'.
        Remove None guard → AttributeError on eval_result.next_action → test fails.
        """
        graph = _make_graph()
        state = _make_state(eval_result=None)
        result = graph.route_from_evaluation(state)
        assert result == "conversation"

    def test_finalize_routes_to_conversation(self):
        """
        SENTINEL — 'finalize' action must route to conversation (pipeline done).
        Route finalize to orchestration → extra planning after completion → test fails.
        """
        graph = _make_graph()
        state = _make_state(eval_result=_make_eval_result("finalize"))
        result = graph.route_from_evaluation(state)
        assert result == "conversation"

    def test_revise_plan_routes_to_orchestration(self):
        """
        SENTINEL — 'revise_plan' must route back to orchestration for replanning.
        Route to reasoning → plan revision skips orchestration → test fails.
        """
        graph = _make_graph()
        state = _make_state(eval_result=_make_eval_result("revise_plan"), plan_revisions=0)
        result = graph.route_from_evaluation(state)
        assert result == "orchestration"

    def test_retry_step_routes_to_reasoning(self):
        """
        SENTINEL — 'retry_step' must route to reasoning for re-execution.
        Route to orchestration → retried step never actually executes → test fails.
        """
        graph = _make_graph()
        state = _make_state(eval_result=_make_eval_result("retry_step"))
        result = graph.route_from_evaluation(state)
        assert result == "reasoning"

    def test_ask_user_routes_to_conversation(self):
        """'ask_user' must route to conversation to surface the question."""
        graph = _make_graph()
        state = _make_state(eval_result=_make_eval_result("ask_user"))
        result = graph.route_from_evaluation(state)
        assert result == "conversation"

    def test_unknown_action_defaults_to_conversation(self):
        """Unknown next_action must default to 'conversation' (safe fallback)."""
        graph = _make_graph()
        state = _make_state(eval_result=_make_eval_result("SENTINEL_UNKNOWN_ACTION"))
        result = graph.route_from_evaluation(state)
        assert result == "conversation"

    def test_max_plan_revisions_forces_conversation(self):
        """
        SENTINEL — when plan_revisions >= _MAX_PLAN_REVISIONS, revise_plan must
        be overridden and route to conversation (loop guard).
        Remove loop guard → infinite revise_plan↔orchestration cycle → test fails.
        """
        graph = _make_graph()
        state = _make_state(
            eval_result=_make_eval_result("revise_plan"),
            plan_revisions=_MAX_PLAN_REVISIONS,  # at limit
        )
        result = graph.route_from_evaluation(state)
        assert result == "conversation"

    def test_below_max_revisions_still_routes_to_orchestration(self):
        """
        SENTINEL — plan_revisions < _MAX_PLAN_REVISIONS must still allow revise_plan.
        Apply guard too early → pipeline stops replanning prematurely → test fails.
        """
        graph = _make_graph()
        state = _make_state(
            eval_result=_make_eval_result("revise_plan"),
            plan_revisions=_MAX_PLAN_REVISIONS - 1,  # one below limit
        )
        result = graph.route_from_evaluation(state)
        assert result == "orchestration"
