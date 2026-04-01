"""
Tests for OrchestrationAgent: dynamic tool descriptions, cost estimation,
plan revision feedback, and the loop guard in COREGraph.

Sentinel-value contract:
- Remove _get_available_tools() → tool list hardcoded → test_prompt_includes_dynamic_tools fails
- Remove estimate_plan_cost() → function absent → test_cost_estimation_* fails
- Remove loop guard → infinite revise_plan loop possible → test_loop_guard_caps_revisions fails
- Remove feedback injection → eval feedback absent from prompt → test_revision_includes_feedback fails
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from app.core.agents.orchestration_agent import OrchestrationAgent
from app.models.core_state import (
    COREState,
    ExecutionPlan,
    PlanStep,
    EvaluationResult,
    UserIntent,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_plan(*tools, hitl_indices=None):
    hitl_indices = hitl_indices or set()
    steps = [
        PlanStep(
            name=f"step-{i}",
            description="do something",
            tool=tool,
            requires_hitl=(i in hitl_indices),
        )
        for i, tool in enumerate(tools)
    ]
    return ExecutionPlan(goal="test goal", steps=steps)


def _make_eval(next_action="revise_plan", feedback="SENTINEL_FEEDBACK"):
    return EvaluationResult(
        overall_status="needs_revision",
        confidence=0.4,
        meets_requirements=False,
        quality_score=0.3,
        feedback=feedback,
        next_action=next_action,
    )


# ---------------------------------------------------------------------------
# Dynamic tool description
# ---------------------------------------------------------------------------

class TestDynamicToolDescriptions:
    def test_prompt_includes_dynamic_tools(self):
        """
        SENTINEL TEST — system prompt must list tools from _get_available_tools().
        Hardcode prompt → SENTINEL_TOOL never appears → test fails.
        """
        agent = OrchestrationAgent()
        with patch.object(
            OrchestrationAgent,
            "_get_available_tools",
            return_value=["SENTINEL_TOOL", "other_tool"],
        ):
            agent.system_prompt = agent._build_system_prompt()

        assert "SENTINEL_TOOL" in agent.system_prompt

    def test_tool_descriptions_bullet_format(self):
        """Each tool appears as a '- **name**:' bullet."""
        agent = OrchestrationAgent()
        desc = agent._tool_descriptions(["file_operations", "git"])
        assert "- **file_operations**:" in desc
        assert "- **git**:" in desc

    def test_empty_tool_list_produces_no_tools_line(self):
        agent = OrchestrationAgent()
        desc = agent._tool_descriptions([])
        assert "no tools" in desc.lower()

    def test_get_available_tools_returns_list(self):
        """_get_available_tools() must return a list of at least one tool name."""
        tools = OrchestrationAgent._get_available_tools()
        assert isinstance(tools, list)
        assert len(tools) >= 1

    def test_known_tools_are_in_available_list(self):
        tools = OrchestrationAgent._get_available_tools()
        for expected in ("file_operations", "git", "web_research"):
            assert expected in tools, f"Expected tool '{expected}' in available_tools"


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

class TestEstimatePlanCost:
    def test_step_count_matches_plan(self):
        """
        SENTINEL TEST — step_count must equal number of steps in plan.
        Return 0 always → 3 != 0 → fails.
        """
        agent = OrchestrationAgent()
        plan = _make_plan(None, "git", "file_operations")
        result = agent.estimate_plan_cost(plan)
        assert result["step_count"] == 3

    def test_tool_steps_counted_separately(self):
        """
        SENTINEL TEST — tool_step_count must equal steps with registered tools.
        Remove tool counting → returns 0 → fails.
        """
        agent = OrchestrationAgent()
        plan = _make_plan("git", None, "file_operations")  # 2 tool steps, 1 llm step
        result = agent.estimate_plan_cost(plan)
        assert result["tool_step_count"] == 2
        assert result["llm_step_count"] == 1

    def test_llm_steps_include_null_tool(self):
        agent = OrchestrationAgent()
        plan = _make_plan(None, None)  # both LLM steps
        result = agent.estimate_plan_cost(plan)
        assert result["llm_step_count"] == 2
        assert result["tool_step_count"] == 0

    def test_estimated_tokens_nonzero_for_llm_steps(self):
        """LLM steps contribute significantly more tokens than tool steps."""
        agent = OrchestrationAgent()
        plan = _make_plan(None)  # 1 LLM step
        result = agent.estimate_plan_cost(plan)
        assert result["estimated_tokens"] > 0

    def test_hitl_checkpoints_counted(self):
        """
        SENTINEL TEST — hitl_checkpoints must equal steps with requires_hitl=True.
        Ignore requires_hitl → returns 0 → fails.
        """
        agent = OrchestrationAgent()
        plan = _make_plan("git", "file_operations", None, hitl_indices={0, 2})
        result = agent.estimate_plan_cost(plan)
        assert result["hitl_checkpoints"] == 2

    def test_risk_high_when_high_risk_tool_and_hitl(self):
        """
        SENTINEL TEST — git (high-risk) + HITL = high risk.
        Remove risk logic → returns 'low' always → fails.
        """
        agent = OrchestrationAgent()
        plan = _make_plan("git", hitl_indices={0})
        result = agent.estimate_plan_cost(plan)
        assert result["risk_level"] == "high"

    def test_risk_medium_for_high_risk_tool_without_hitl(self):
        agent = OrchestrationAgent()
        plan = _make_plan("git")  # high-risk tool, no HITL
        result = agent.estimate_plan_cost(plan)
        assert result["risk_level"] == "medium"

    def test_risk_medium_for_hitl_without_high_risk_tool(self):
        agent = OrchestrationAgent()
        plan = _make_plan("file_operations", hitl_indices={0})
        result = agent.estimate_plan_cost(plan)
        assert result["risk_level"] == "medium"

    def test_risk_low_for_simple_plan(self):
        agent = OrchestrationAgent()
        plan = _make_plan(None)  # LLM step, no HITL
        result = agent.estimate_plan_cost(plan)
        assert result["risk_level"] == "low"

    def test_tool_breakdown_groups_by_tool(self):
        """tool_breakdown maps tool_name → count."""
        agent = OrchestrationAgent()
        plan = _make_plan("git", "git", "file_operations")
        result = agent.estimate_plan_cost(plan)
        assert result["tool_breakdown"]["git"] == 2
        assert result["tool_breakdown"]["file_operations"] == 1

    def test_empty_plan_returns_zero_cost(self):
        agent = OrchestrationAgent()
        plan = ExecutionPlan(goal="empty", steps=[])
        result = agent.estimate_plan_cost(plan)
        assert result["step_count"] == 0
        assert result["estimated_tokens"] == 0
        assert result["risk_level"] == "low"


# ---------------------------------------------------------------------------
# Plan revision feedback
# ---------------------------------------------------------------------------

class TestPlanRevisionFeedback:
    def test_revision_includes_feedback_in_llm_prompt(self):
        """
        SENTINEL TEST — evaluation feedback must appear in the LLM user message
        when revising a plan.  Remove feedback injection → SENTINEL_FEEDBACK
        absent from prompt → test fails.
        """
        agent = OrchestrationAgent()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"goal":"g","reasoning":"r","steps":[]}'))]
        )

        prev_plan = _make_plan(None)

        with patch("app.core.agents.orchestration_agent.get_openai_client_sync",
                   return_value=mock_client):
            agent.create_plan(
                user_input="do the thing",
                previous_plan=prev_plan,
                evaluation_feedback="SENTINEL_FEEDBACK",
                revision=2,
            )

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        assert "SENTINEL_FEEDBACK" in user_msg

    def test_revision_number_increments_in_plan(self):
        """The returned plan's revision field matches the revision arg."""
        agent = OrchestrationAgent()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"goal":"g","reasoning":"r","steps":[]}'))]
        )

        with patch("app.core.agents.orchestration_agent.get_openai_client_sync",
                   return_value=mock_client):
            plan = agent.create_plan("do it", revision=5)

        assert plan.revision == 5


# ---------------------------------------------------------------------------
# Loop guard (COREGraph.route_from_evaluation)
# ---------------------------------------------------------------------------

class TestLoopGuard:
    def _make_core_graph(self):
        """Import COREGraph without triggering LangGraph compilation."""
        from app.core.langgraph.core_graph_v2 import COREGraph
        return COREGraph()

    def _make_state(self, plan_revisions: int, next_action: str = "revise_plan"):
        eval_result = _make_eval(next_action=next_action)
        plan = _make_plan(None)
        return COREState(
            user_input="test",
            plan=plan,
            eval_result=eval_result,
            plan_revisions=plan_revisions,
        )

    def test_loop_guard_caps_revisions(self):
        """
        SENTINEL TEST — when plan_revisions >= MAX, route_from_evaluation must
        return 'conversation', NOT 'orchestration'.
        Remove loop guard → returns 'orchestration' forever → test fails.
        """
        from app.core.langgraph import core_graph_v2 as mod

        original_max = mod._MAX_PLAN_REVISIONS
        try:
            mod._MAX_PLAN_REVISIONS = 2  # low cap for test
            graph = self._make_core_graph()
            state = self._make_state(plan_revisions=2, next_action="revise_plan")
            route = graph.route_from_evaluation(state)
            assert route == "conversation", (
                "Expected 'conversation' after hitting max revisions, got: " + route
            )
        finally:
            mod._MAX_PLAN_REVISIONS = original_max

    def test_revise_plan_allowed_below_limit(self):
        """Below the cap, revise_plan should still route to orchestration."""
        from app.core.langgraph import core_graph_v2 as mod

        original_max = mod._MAX_PLAN_REVISIONS
        try:
            mod._MAX_PLAN_REVISIONS = 5
            graph = self._make_core_graph()
            state = self._make_state(plan_revisions=1, next_action="revise_plan")
            route = graph.route_from_evaluation(state)
            assert route == "orchestration"
        finally:
            mod._MAX_PLAN_REVISIONS = original_max

    def test_finalize_always_routes_to_conversation(self):
        graph = self._make_core_graph()
        state = self._make_state(plan_revisions=100, next_action="finalize")
        assert graph.route_from_evaluation(state) == "conversation"

    def test_plan_revisions_field_exists_in_core_state(self):
        """COREState must have plan_revisions field (default 0)."""
        state = COREState(user_input="hi")
        assert hasattr(state, "plan_revisions")
        assert state.plan_revisions == 0

    def test_orchestration_node_increments_plan_revisions(self):
        """
        SENTINEL TEST — orchestration_node must increment plan_revisions when
        revising (plan + eval_result both present).
        Remove increment → plan_revisions stays 0 → loop guard never fires → fails.
        """
        from app.core.langgraph.core_graph_v2 import COREGraph

        graph = COREGraph()
        state = COREState(
            user_input="revise me",
            plan=_make_plan(None),
            eval_result=_make_eval(),
            plan_revisions=2,
        )

        mock_plan = _make_plan(None)
        with patch.object(graph.orchestration_agent, "create_plan", return_value=mock_plan):
            new_state = graph.orchestration_node(state)

        assert new_state.plan_revisions == 3
