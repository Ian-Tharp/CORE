"""
Tests for the abstention guard (app/core/agents/refusal_guard.py) and its wiring.

Sentinel-value contract:
- Remove the latch check in ReasoningAgent._run_step -> tool steps after a refusal
  are dispatched -> test_steps_after_refusal_are_not_dispatched fails
- Stop scanning LLM output for refusals -> refusing step reads as success
  -> test_refusing_step_fails_and_latches fails
- Scan tool output too -> a fetched page saying "I can't" latches the run
  -> test_tool_output_is_never_treated_as_refusal fails
- Remove the refusal branch in route_from_evaluation -> retry/revise loops around
  a refusal -> TestRouteGuard fails
- Drop gate seeding in comprehension_node -> reasoning after a gate refuse still
  acts -> test_gate_refusal_seeds_latch fails
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.agents.reasoning_agent import ReasoningAgent
from app.core.agents.refusal_guard import (
    RefusalLatch,
    detect_refusal,
    gate_refusal,
    guard_enabled,
)
from app.models.core_state import (
    COREState,
    EvaluationResult,
    ExecutionPlan,
    PlanStep,
    RefusalRecord,
    StepResult,
    UserIntent,
)

# ---------------------------------------------------------------------------
# detect_refusal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "I can't help with that request.",
        "I’m sorry, but I cannot do that.",
        "I won't execute that command.",
        "I'm unable to proceed with this step.",
        "I must decline.",
        "I refuse to delete those files.",
        "That would violate my guidelines.",
        "I am not comfortable doing this.",
    ],
)
def test_detects_refusals(text):
    assert detect_refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        None,
        "Here is the summary you asked for.",
        "I can't find any matching files.",  # a failure, not a refusal
        "I can't help but notice the off-by-one bug.",
        "The user said they cannot attend.",
    ],
)
def test_ignores_non_refusals(text):
    assert detect_refusal(text) is None


# ---------------------------------------------------------------------------
# RefusalLatch
# ---------------------------------------------------------------------------


def test_latch_first_refusal_wins_and_blocks():
    latch = RefusalLatch()
    step = PlanStep(id="s2", name="write", description="w", tool="file_operations")
    assert latch.check(step) is None and not latch.active

    assert latch.trip(RefusalRecord(source="reasoning", reason="a", step_id="s1"))
    assert not latch.trip(RefusalRecord(source="reasoning", reason="b"))
    assert latch.record.reason == "a"

    blocked = latch.check(step)
    assert blocked.step_id == "s2" and blocked.tool == "file_operations"
    assert "s1" in blocked.reason
    assert [b.step_id for b in latch.blocked] == ["s2"]


def test_latch_seeded_from_existing_record_blocks_immediately():
    latch = RefusalLatch(existing=RefusalRecord(source="gate", reason="x"))
    assert latch.check(PlanStep(name="n", description="d")) is not None


def test_gate_refusal_only_for_blocking_outcomes():
    assert gate_refusal("proceed", "m") is None
    assert gate_refusal(None, None) is None
    assert gate_refusal("refuse", "no tool").source == "gate"
    assert "clarification" in gate_refusal("clarify", "?").reason


def test_guard_env_switch(monkeypatch):
    monkeypatch.delenv("CORE_BLOCK_AFTER_REFUSAL", raising=False)
    assert guard_enabled()
    monkeypatch.setenv("CORE_BLOCK_AFTER_REFUSAL", "0")
    assert not guard_enabled()


# ---------------------------------------------------------------------------
# ReasoningAgent integration
# ---------------------------------------------------------------------------


def _agent():
    agent = ReasoningAgent.__new__(ReasoningAgent)
    agent.model = "test"
    agent.tool_dispatcher = MagicMock()
    agent.tool_dispatcher.available_tools = ["file_operations", "web_research"]
    agent.tool_dispatcher.dispatch.return_value = {"result": "ok", "status": "success"}
    agent.tool_dispatcher.get_artifacts.return_value = []
    return agent


def _llm_step(id_, deps=None):
    return PlanStep(id=id_, name=id_, description="think", dependencies=deps or [])


def _tool_step(id_, deps=None, tool="file_operations"):
    return PlanStep(
        id=id_,
        name=id_,
        description="act",
        tool=tool,
        params={"action": "write", "path": "x.txt", "content": "y"},
        dependencies=deps or [],
    )


def _plan(*steps):
    return ExecutionPlan(goal="g", steps=list(steps))


def test_refusing_step_fails_and_latches():
    agent = _agent()
    latch = RefusalLatch()
    with patch.object(
        agent, "_execute_with_llm", return_value={"result": "I can't help with that."}
    ):
        [res] = agent.execute_plan(_plan(_llm_step("s1")), refusal_latch=latch)
    assert res.status == "failure" and res.error.startswith("Refused")
    assert res.outputs["result"] == "I can't help with that."  # model's words kept
    assert latch.record.source == "reasoning" and latch.record.step_id == "s1"


def test_steps_after_refusal_are_not_dispatched():
    agent = _agent()
    latch = RefusalLatch()
    # s2 depends on s1; s3 is independent but in a later wave via s2.
    plan = _plan(
        _llm_step("s1"), _tool_step("s2", deps=["s1"]), _tool_step("s3", deps=["s2"])
    )
    with patch.object(
        agent, "_execute_with_llm", return_value={"result": "I must decline."}
    ):
        results = agent.execute_plan(plan, refusal_latch=latch)

    agent.tool_dispatcher.dispatch.assert_not_called()
    assert [r.status for r in results] == ["failure", "failure", "failure"]
    assert all(r.dispatched_tool is None for r in results)


def test_independent_sibling_step_is_blocked_too():
    """Blocking is not just dependency skipping: a step that does not depend on
    the refusing step must not run either. One worker makes the wave run in
    plan order, so the sibling starts after the refusal deterministically."""
    agent = _agent()
    agent._MAX_PARALLEL_WORKERS = 1
    latch = RefusalLatch()
    s1 = _llm_step("s1")
    sibling = _tool_step("sibling")  # same wave, no dependency on s1
    with patch.object(
        agent, "_execute_with_llm", return_value={"result": "I won't do that."}
    ):
        results = agent.execute_plan(_plan(s1, sibling), refusal_latch=latch)

    agent.tool_dispatcher.dispatch.assert_not_called()
    by_id = {r.step_id: r for r in results}
    assert by_id["sibling"].error.startswith("Blocked")
    assert sibling.status == "skipped"
    assert [b.step_id for b in latch.blocked] == ["sibling"]


def test_dependents_after_refusal_are_recorded_as_blocked():
    agent = _agent()
    latch = RefusalLatch()
    plan = _plan(_llm_step("s1"), _tool_step("s2", deps=["s1"]))
    with patch.object(
        agent, "_execute_with_llm", return_value={"result": "I must decline."}
    ):
        agent.execute_plan(plan, refusal_latch=latch)
    assert [b.step_id for b in latch.blocked] == ["s2"]


def test_tool_output_is_never_treated_as_refusal():
    agent = _agent()
    agent.tool_dispatcher.dispatch.return_value = {
        "result": "Page text: I can't help with that.",
        "status": "success",
    }
    latch = RefusalLatch()
    plan = _plan(_tool_step("s1", tool="web_research"), _tool_step("s2", deps=["s1"]))
    results = agent.execute_plan(plan, refusal_latch=latch)
    assert not latch.active
    assert [r.status for r in results] == ["success", "success"]
    assert results[0].dispatched_tool == "web_research"


def test_retry_path_is_blocked_by_seeded_latch():
    agent = _agent()
    latch = RefusalLatch(existing=RefusalRecord(source="reasoning", reason="r"))
    plan = _plan(_tool_step("s1"))
    [res] = agent.execute_plan(plan, start_from_step="s1", refusal_latch=latch)
    agent.tool_dispatcher.dispatch.assert_not_called()
    assert res.error.startswith("Blocked")


def test_no_latch_keeps_previous_behaviour():
    agent = _agent()
    plan = _plan(_llm_step("s1"), _tool_step("s2"))
    with patch.object(
        agent, "_execute_with_llm", return_value={"result": "I can't help."}
    ):
        results = agent.execute_plan(plan)
    assert [r.status for r in results] == ["success", "success"]
    agent.tool_dispatcher.dispatch.assert_called_once()


def test_dry_run_is_not_scanned():
    agent = _agent()
    latch = RefusalLatch()
    step = PlanStep(id="s1", name="n", description="I can't help with that")
    agent.execute_plan(_plan(step), enable_tools=False, refusal_latch=latch)
    assert not latch.active


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------


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

        yield COREGraph()


def _eval(next_action):
    return EvaluationResult(
        overall_status="needs_retry",
        confidence=0.4,
        meets_requirements=False,
        quality_score=0.3,
        next_action=next_action,
        retry_step_id="s1",
    )


class TestRouteGuard:
    @pytest.mark.parametrize("action", ["retry_step", "revise_plan"])
    def test_refusal_blocks_retry_and_revise(self, graph, action):
        state = COREState(
            user_input="x",
            eval_result=_eval(action),
            refusal=RefusalRecord(source="reasoning", reason="r"),
        )
        assert graph.route_from_evaluation(state) == "conversation"

    def test_without_refusal_routing_unchanged(self, graph):
        state = COREState(user_input="x", eval_result=_eval("retry_step"))
        assert graph.route_from_evaluation(state) == "reasoning"

    def test_guard_off_restores_routing(self, graph, monkeypatch):
        monkeypatch.setenv("CORE_BLOCK_AFTER_REFUSAL", "off")
        state = COREState(
            user_input="x",
            eval_result=_eval("retry_step"),
            refusal=RefusalRecord(source="reasoning", reason="r"),
        )
        assert graph.route_from_evaluation(state) == "reasoning"


def test_gate_refusal_seeds_latch(graph):
    graph.comprehension_agent.analyze_intent.return_value = UserIntent(
        type="task",
        description="send an email",
        confidence=0.95,
        requires_tools=True,
        tools_needed=["zzz_quantum_teleporter"],
    )
    graph.comprehension_agent.detect_ambiguities.return_value = []
    state = graph.comprehension_node(COREState(user_input="send an email"))
    assert state.gate_outcome == "refuse"
    assert state.refusal is not None and state.refusal.source == "gate"

    # If reasoning runs anyway, the latch it hands the agent is already tripped.
    state.plan = _plan(_tool_step("s1"))
    graph.reasoning_agent.execute_plan.return_value = []
    graph.reasoning_node(state)
    latch = graph.reasoning_agent.execute_plan.call_args.kwargs["refusal_latch"]
    assert latch.active and latch.record.source == "gate"


def test_reasoning_node_records_refusal_and_blocked(graph):
    def fake_execute(plan, start_from_step, enable_tools, refusal_latch, **_):
        refusal_latch.trip(RefusalRecord(source="reasoning", reason="r", step_id="s1"))
        refusal_latch.check(plan.steps[1])
        return [
            StepResult(step_id="s1", status="failure"),
            StepResult(step_id="s2", status="failure"),
        ]

    graph.reasoning_agent.execute_plan.side_effect = fake_execute
    state = COREState(
        user_input="x", plan=_plan(_llm_step("s1"), _tool_step("s2", deps=["s1"]))
    )
    state = graph.reasoning_node(state)
    assert state.refusal.step_id == "s1"
    assert [b.step_id for b in state.blocked_actions] == ["s2"]
    assert any("abstention guard" in w for w in state.warnings)


def test_conversation_leads_with_refusal_notice():
    from app.core.langgraph.core_graph_v2 import COREGraph
    from app.models.core_state import BlockedAction

    graph = object.__new__(COREGraph)
    state = COREState(
        user_input="x",
        intent=UserIntent(type="task", description="t", confidence=0.9),
        plan=_plan(_llm_step("s1"), _tool_step("s2")),
        step_results=[
            StepResult(
                step_id="s1",
                status="failure",
                outputs={"result": "I can't help with that."},
            )
        ],
        eval_result=EvaluationResult(
            overall_status="failure",
            confidence=0.9,
            meets_requirements=False,
            quality_score=0.1,
        ),
        refusal=RefusalRecord(source="reasoning", reason="step declined", step_id="s1"),
        blocked_actions=[BlockedAction(step_id="s2", step_name="write file")],
    )
    with patch("app.core.langgraph.core_graph_v2.record_pipeline_run"):
        out = graph.conversation_node(state)
    assert out.response.startswith("I stopped before finishing")
    assert "write file" in out.response
    assert "I can't help with that." in out.response
