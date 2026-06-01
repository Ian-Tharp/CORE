"""
Tests for COREGraph routing methods in app/core/langgraph/core_graph_v2.py.

Sentinel-value contract:
- Remove route_from_comprehension "task"->"orchestration" branch -> task intents routed to conversation -> test fails
- Remove route_from_comprehension None-intent guard -> AttributeError instead of "conversation" default -> test fails
- Remove route_from_evaluation "revise_plan"->"orchestration" branch -> plan revision loops to conversation -> test fails
- Remove route_from_evaluation "retry_step"->"reasoning" branch -> step retries skip reasoning -> test fails
- Remove route_from_evaluation max-revisions guard -> revise_plan loops forever -> test fails
- Remove route_from_evaluation None eval_result guard -> AttributeError in production -> test fails
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from app.models.core_state import COREState, UserIntent, EvaluationResult


@pytest.fixture()
def graph():
    with patch.multiple(
        "app.core.langgraph.core_graph_v2",
        ComprehensionAgent=MagicMock(),
        OrchestrationAgent=MagicMock(),
        ReasoningAgent=MagicMock(),
        EvaluationAgent=MagicMock(),
    ):
        from app.core.langgraph.core_graph_v2 import COREGraph

        return COREGraph()


def _state(**overrides) -> COREState:
    base = {"user_input": "SENTINEL_INPUT"}
    base.update(overrides)
    return COREState(**base)


def _intent(type_: str, confidence: float = 0.9) -> UserIntent:
    return UserIntent(type=type_, description="test", confidence=confidence)


def _eval_result(next_action: str) -> EvaluationResult:
    return EvaluationResult(
        overall_status="success",
        confidence=0.9,
        meets_requirements=True,
        quality_score=0.85,
        next_action=next_action,
    )


class TestRouteFromComprehension:
    def test_task_intent_routes_to_orchestration(self, graph):
        state = _state(intent=_intent("task"))
        assert graph.route_from_comprehension(state) == "orchestration"

    def test_question_intent_routes_to_orchestration(self, graph):
        state = _state(intent=_intent("question"))
        assert graph.route_from_comprehension(state) == "orchestration"

    def test_conversation_intent_routes_to_conversation(self, graph):
        state = _state(intent=_intent("conversation"))
        assert graph.route_from_comprehension(state) == "conversation"

    def test_clarification_intent_routes_to_conversation(self, graph):
        state = _state(intent=_intent("clarification"))
        assert graph.route_from_comprehension(state) == "conversation"

    def test_none_intent_defaults_to_conversation(self, graph):
        """
        SENTINEL - when state.intent is None, must return 'conversation' not raise.
        Remove None guard -> AttributeError on comprehension failure -> test fails.
        """
        state = _state()
        assert state.intent is None
        assert graph.route_from_comprehension(state) == "conversation"


class TestRouteFromEvaluation:
    def test_finalize_routes_to_conversation(self, graph):
        """
        SENTINEL - next_action='finalize' must return 'conversation'.
        Remove finalize branch -> successful evaluation has no response -> test fails.
        """
        state = _state(eval_result=_eval_result("finalize"))
        assert graph.route_from_evaluation(state) == "conversation"

    def test_revise_plan_routes_to_orchestration(self, graph):
        """
        SENTINEL - next_action='revise_plan' must return 'orchestration'.
        Remove branch -> plan revision goes to conversation -> no revised plan -> test fails.
        """
        state = _state(eval_result=_eval_result("revise_plan"), plan_revisions=0)
        assert graph.route_from_evaluation(state) == "orchestration"

    def test_retry_step_routes_to_reasoning(self, graph):
        """
        SENTINEL - next_action='retry_step' must return 'reasoning'.
        Remove branch -> step retries skip execution -> failed step never retried -> test fails.
        """
        state = _state(eval_result=_eval_result("retry_step"))
        assert graph.route_from_evaluation(state) == "reasoning"

    def test_ask_user_routes_to_conversation(self, graph):
        state = _state(eval_result=_eval_result("ask_user"))
        assert graph.route_from_evaluation(state) == "conversation"

    def test_none_eval_result_defaults_to_conversation(self, graph):
        """
        SENTINEL - when state.eval_result is None, must return 'conversation' not raise.
        Remove None guard -> AttributeError before evaluation runs -> test fails.
        """
        state = _state()
        assert state.eval_result is None
        assert graph.route_from_evaluation(state) == "conversation"

    def test_revise_plan_capped_at_max_revisions(self, graph):
        """
        SENTINEL - when plan_revisions >= _MAX_PLAN_REVISIONS, revise_plan returns 'conversation'.
        Remove loop guard -> revise_plan -> orchestration -> reasoning -> evaluation loops forever.
        """
        from app.core.langgraph import core_graph_v2

        max_rev = core_graph_v2._MAX_PLAN_REVISIONS
        state = _state(eval_result=_eval_result("revise_plan"), plan_revisions=max_rev)
        assert graph.route_from_evaluation(state) == "conversation"

    def test_revise_plan_allowed_below_max(self, graph):
        """revise_plan is allowed when plan_revisions < _MAX_PLAN_REVISIONS."""
        from app.core.langgraph import core_graph_v2

        max_rev = core_graph_v2._MAX_PLAN_REVISIONS
        state = _state(
            eval_result=_eval_result("revise_plan"), plan_revisions=max_rev - 1
        )
        assert graph.route_from_evaluation(state) == "orchestration"

    def test_unknown_action_defaults_to_conversation(self, graph):
        """
        SENTINEL - unknown next_action must fall back to 'conversation' not raise KeyError.
        Remove .get default -> KeyError on unexpected evaluation outputs -> test fails.
        """
        eval_result = _eval_result("finalize")
        object.__setattr__(eval_result, "next_action", "SENTINEL_UNKNOWN_ACTION")
        state = _state(eval_result=eval_result)
        assert graph.route_from_evaluation(state) == "conversation"

    def test_max_plan_revisions_is_positive_int(self, graph):
        """
        SENTINEL - _MAX_PLAN_REVISIONS must be a positive integer.
        Set to 0 -> first evaluation always forces finalization -> plans never revised -> test fails.
        """
        from app.core.langgraph import core_graph_v2

        assert isinstance(core_graph_v2._MAX_PLAN_REVISIONS, int)
        assert core_graph_v2._MAX_PLAN_REVISIONS > 0
