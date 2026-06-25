"""
Node-level tests: orchestration_node grounds the draft plan ADDITIVELY against
the Capability Registry (Phase 2 hook), without disturbing the pinned
plan_revisions increment / try-except contract.

These sit beside the sentinel tests in test_orchestration_agent.py: the increment
semantics they pin must still hold while grounding runs on the (possibly revised)
plan.
"""

from __future__ import annotations

from unittest.mock import patch

from app.core.langgraph.core_graph_v2 import COREGraph
from app.models.core_state import (
    COREState,
    ExecutionPlan,
    PlanStep,
    EvaluationResult,
)


def _valid_step():
    return PlanStep(
        name="read",
        description="read a file",
        tool="file_operations",
        params={"action": "read", "path": "main.py"},
    )


def _invented_step():
    return PlanStep(
        name="browse",
        description="open a browser",
        tool="browser",
        params={"url": "https://example.com"},
    )


def _eval(next_action="revise_plan"):
    return EvaluationResult(
        overall_status="needs_revision",
        confidence=0.4,
        meets_requirements=False,
        quality_score=0.3,
        feedback="revise",
        next_action=next_action,
    )


class TestOrchestrationGrounding:
    def test_invented_tool_step_dropped_in_node(self):
        graph = COREGraph()
        valid = _valid_step()
        invented = _invented_step()
        draft = ExecutionPlan(goal="g", steps=[valid, invented])

        state = COREState(user_input="do it")
        with patch.object(graph.orchestration_agent, "create_plan", return_value=draft):
            new_state = graph.orchestration_node(state)

        assert new_state.plan is not None
        surviving = {s.id for s in new_state.plan.steps}
        assert valid.id in surviving
        assert invented.id not in surviving

    def test_grounding_surfaces_warning_and_report(self):
        graph = COREGraph()
        draft = ExecutionPlan(goal="g", steps=[_valid_step(), _invented_step()])

        state = COREState(user_input="do it")
        with patch.object(graph.orchestration_agent, "create_plan", return_value=draft):
            new_state = graph.orchestration_node(state)

        assert any("grounding" in w.lower() for w in new_state.warnings)
        assert new_state.grounding_report is not None
        assert len(new_state.grounding_report["dropped"]) == 1
        assert new_state.grounding_report["dropped"][0]["reason"] == "invented_tool"

    def test_clean_plan_no_warning(self):
        graph = COREGraph()
        draft = ExecutionPlan(goal="g", steps=[_valid_step()])

        state = COREState(user_input="do it")
        with patch.object(graph.orchestration_agent, "create_plan", return_value=draft):
            new_state = graph.orchestration_node(state)

        assert not any("grounding dropped" in w.lower() for w in new_state.warnings)
        assert len(new_state.plan.steps) == 1

    def test_none_plan_is_safe(self):
        """create_plan returning None must not raise in the grounding hook."""
        graph = COREGraph()
        state = COREState(user_input="do it")
        with patch.object(graph.orchestration_agent, "create_plan", return_value=None):
            new_state = graph.orchestration_node(state)
        assert new_state.plan is None

    def test_increment_still_holds_and_grounding_runs_on_revision(self):
        """
        SENTINEL: the pinned plan_revisions increment must still fire on the
        revise_plan path, AND grounding must run on the revised draft plan.
        """
        graph = COREGraph()
        prior_plan = ExecutionPlan(goal="g", steps=[_valid_step()])
        revised_draft = ExecutionPlan(
            goal="g", steps=[_valid_step(), _invented_step()], revision=2
        )

        state = COREState(
            user_input="revise me",
            plan=prior_plan,
            eval_result=_eval(),
            plan_revisions=2,
        )
        with patch.object(
            graph.orchestration_agent, "create_plan", return_value=revised_draft
        ):
            new_state = graph.orchestration_node(state)

        # increment preserved
        assert new_state.plan_revisions == 3
        # grounding ran: invented step dropped from the revised plan
        surviving = {s.id for s in new_state.plan.steps}
        assert len(surviving) == 1
        assert new_state.grounding_report is not None
        assert len(new_state.grounding_report["dropped"]) == 1
