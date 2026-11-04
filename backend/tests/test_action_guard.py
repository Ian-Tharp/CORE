"""
Tests for the confirm tier and the dependency latch
(app/core/agents/action_guard.py) and their wiring.

Sentinel-value contract:
- Drop the confirm check in ReasoningAgent._run_step -> a write step is
  dispatched without approval -> test_write_step_is_held_not_dispatched fails
- Ignore config["approved_capabilities"] -> an approved write stays held
  -> test_approved_write_is_dispatched fails
- Ignore requires_hitl -> a planner checkpoint on a read runs anyway
  -> test_requires_hitl_holds_a_read_step fails
- Treat database POST as its registry class (network) -> a data-changing call
  runs unconfirmed -> test_side_effect_classes fails
- Remove the dependency check on retry -> a retried step runs although its
  prerequisite failed -> test_retry_checks_dependencies fails
- Swallow LLM errors as success again -> a write runs on a missing answer
  -> test_llm_error_fails_the_step_and_blocks_dependents fails
- Run a dependency cycle anyway -> test_dependency_cycle_does_not_run fails
- Count a held write as a step failure -> held runs get captured as faults
  -> test_held_write_is_not_a_fault fails
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import app.core.langgraph.core_graph_v2 as cg
from app.core.agents.action_guard import (
    ActionGuard,
    confirm_writes_enabled,
    mark_confirmation_steps,
)
from app.core.agents.comprehension_gate import GateDecision
from app.core.agents.reasoning_agent import ReasoningAgent
from app.core.registry.builtin import (
    build_registry_from_dispatcher,
    side_effect_for_step,
)
from app.evaluation.capture import fault_reasons, fault_report_from_state
from app.models.core_state import (
    BlockedAction,
    COREState,
    EvaluationResult,
    ExecutionPlan,
    PendingConfirmation,
    PlanStep,
    StepResult,
    UserIntent,
)

TOOLS = ["file_operations", "git", "web_research", "database"]


@pytest.fixture(scope="module")
def registry():
    return build_registry_from_dispatcher(MagicMock(available_tools=TOOLS))


def _agent():
    agent = ReasoningAgent.__new__(ReasoningAgent)
    agent.model = "test"
    agent.tool_dispatcher = MagicMock()
    agent.tool_dispatcher.available_tools = TOOLS
    agent.tool_dispatcher.dispatch.return_value = {"result": "ok", "status": "success"}
    agent.tool_dispatcher.get_artifacts.return_value = []
    return agent


def _write(id_, deps=None, path="out.txt"):
    return PlanStep(
        id=id_,
        name=f"write {path}",
        description="write a file",
        tool="file_operations",
        params={"action": "write", "path": path, "content": "x"},
        dependencies=deps or [],
    )


def _read(id_, deps=None, path="in.txt"):
    return PlanStep(
        id=id_,
        name=f"read {path}",
        description="read a file",
        tool="file_operations",
        params={"action": "read", "path": path},
        dependencies=deps or [],
    )


def _llm(id_, deps=None):
    return PlanStep(id=id_, name=id_, description="think", dependencies=deps or [])


def _plan(*steps):
    return ExecutionPlan(goal="g", steps=list(steps))


def _dispatched(agent):
    return [
        c.args[1].get("path") for c in agent.tool_dispatcher.dispatch.call_args_list
    ]


# ---------------------------------------------------------------------------
# Side-effect classes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool,params,expected",
    [
        ("file_operations", {"action": "write", "path": "a"}, "write"),
        ("file_operations", {"action": "read", "path": "a"}, "read"),
        ("file_operations", {}, "read"),  # defaults to list
        ("git", {"action": "log"}, "read"),
        ("web_research", {"url": "https://x"}, "network"),
        ("database", {"endpoint": "conversations"}, "network"),
        ("database", {"endpoint": "conversations", "method": "POST"}, "write"),
        (
            "database",
            {"endpoint": "/knowledgebase/semantic-search", "method": "post"},
            "network",
        ),
        ("file_operations", {"action": "delete", "path": "a"}, "write"),  # unknown
    ],
)
def test_side_effect_classes(registry, tool, params, expected):
    _, effect = side_effect_for_step(tool, params, registry)
    assert effect == expected


def test_unknown_capability_fails_closed():
    assert side_effect_for_step("git", {"action": "push"}, None) == (
        "git.push",
        "write",
    )


def test_mark_confirmation_steps_sets_requires_hitl_on_writes_only(registry):
    plan = _plan(_read("r"), _write("w"), _llm("l"))
    assert mark_confirmation_steps(plan, registry) == 1
    assert [s.requires_hitl for s in plan.steps] == [False, True, False]


def test_confirm_env_switch(monkeypatch):
    monkeypatch.delenv("CORE_CONFIRM_WRITES", raising=False)
    assert confirm_writes_enabled()
    monkeypatch.setenv("CORE_CONFIRM_WRITES", "off")
    assert not confirm_writes_enabled()


# ---------------------------------------------------------------------------
# ActionGuard
# ---------------------------------------------------------------------------


def test_guard_holds_write_and_records_target(registry):
    guard = ActionGuard(registry=registry)
    held = guard.check_confirmation(_write("w", path="config.yaml"))
    assert held.capability_id == "file_operations.write"
    assert held.reason == "write" and held.target == "config.yaml"
    assert guard.is_awaiting("w")
    assert guard.check_confirmation(_read("r")) is None


@pytest.mark.parametrize("approved", [["file_operations.write"], ["*"]])
def test_guard_approval(registry, approved):
    guard = ActionGuard(registry=registry, approved=approved)
    assert guard.check_confirmation(_write("w")) is None


def test_guard_approval_is_per_capability(registry):
    guard = ActionGuard(registry=registry, approved=["git.status"])
    assert guard.check_confirmation(_write("w")) is not None


def test_guard_dependents_of_held_step_await_confirmation(registry):
    guard = ActionGuard(registry=registry)
    guard.check_confirmation(_write("w"))
    b = guard.block_dependent(_read("r", deps=["w"]), "w")
    assert b.kind == "awaiting_confirmation" and b.cause_step_id == "w"
    # ...and that propagates down the chain.
    c = guard.block_dependent(_read("r2", deps=["r"]), "r")
    assert c.kind == "awaiting_confirmation"
    d = guard.block_dependent(_read("x"), "other")
    assert d.kind == "dependency_failed"


# ---------------------------------------------------------------------------
# ReasoningAgent: confirm tier
# ---------------------------------------------------------------------------


def test_write_step_is_held_not_dispatched(registry):
    agent = _agent()
    guard = ActionGuard(registry=registry)
    plan = _plan(_read("r"), _write("w"), _read("after", deps=["w"], path="a.txt"))
    results = agent.execute_plan(plan, action_guard=guard)

    assert _dispatched(agent) == ["in.txt"]  # the independent read still ran
    by_id = {r.step_id: r for r in results}
    assert by_id["w"].status == "failure"
    assert by_id["w"].error.startswith("Awaiting confirmation")
    assert by_id["w"].dispatched_tool is None
    assert plan.steps[1].status == "skipped"
    assert [p.step_id for p in guard.pending] == ["w"]
    assert [(b.step_id, b.kind) for b in guard.blocked] == [
        ("after", "awaiting_confirmation")
    ]


def test_approved_write_is_dispatched(registry):
    agent = _agent()
    guard = ActionGuard(registry=registry, approved=["file_operations.write"])
    [res] = agent.execute_plan(_plan(_write("w")), action_guard=guard)
    assert res.status == "success" and res.dispatched_tool == "file_operations"
    assert guard.pending == []


def test_requires_hitl_holds_a_read_step(registry):
    agent = _agent()
    step = _read("r")
    step.requires_hitl = True
    guard = ActionGuard(registry=registry)
    [res] = agent.execute_plan(_plan(step), action_guard=guard)
    agent.tool_dispatcher.dispatch.assert_not_called()
    assert guard.pending[0].reason == "requires_hitl"
    assert res.error.startswith("Awaiting confirmation")


def test_confirm_off_dispatches_writes(registry):
    agent = _agent()
    guard = ActionGuard(registry=registry, confirm=False)
    [res] = agent.execute_plan(_plan(_write("w")), action_guard=guard)
    assert res.status == "success"


def test_dry_run_and_llm_steps_are_not_held(registry):
    agent = _agent()
    guard = ActionGuard(registry=registry)
    agent.execute_plan(_plan(_write("w")), enable_tools=False, action_guard=guard)
    with patch.object(agent, "_execute_with_llm", return_value={"result": "done"}):
        agent.execute_plan(_plan(_llm("l")), action_guard=guard)
    assert guard.pending == []


def test_no_guard_keeps_previous_behaviour():
    agent = _agent()
    [res] = agent.execute_plan(_plan(_write("w")))
    assert res.status == "success"


# ---------------------------------------------------------------------------
# ReasoningAgent: dependency latch
# ---------------------------------------------------------------------------


def test_failed_dependency_blocks_dependent_write():
    agent = _agent()
    agent.tool_dispatcher.dispatch.return_value = {"result": "nope", "status": "error"}
    guard = ActionGuard(confirm=False)
    results = agent.execute_plan(
        _plan(_read("r"), _write("w", deps=["r"])), action_guard=guard
    )
    assert _dispatched(agent) == ["in.txt"]
    assert results[1].error.startswith("Dependency failed")
    assert [(b.step_id, b.kind, b.cause_step_id) for b in guard.blocked] == [
        ("w", "dependency_failed", "r")
    ]


def test_llm_error_fails_the_step_and_blocks_dependents():
    agent = _agent()
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("model down")
    guard = ActionGuard(confirm=False)
    with patch(
        "app.core.agents.reasoning_agent.get_openai_client_sync", return_value=client
    ), patch.object(agent, "fetch_step_context", return_value=None):
        results = agent.execute_plan(
            _plan(_llm("l"), _write("w", deps=["l"])), action_guard=guard
        )
    assert results[0].status == "failure" and "model down" in results[0].error
    agent.tool_dispatcher.dispatch.assert_not_called()
    assert guard.blocked[0].step_id == "w"


def test_empty_llm_answer_fails_the_step():
    agent = _agent()
    client = MagicMock()
    client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content=""))
    ]
    with patch(
        "app.core.agents.reasoning_agent.get_openai_client_sync", return_value=client
    ), patch.object(agent, "fetch_step_context", return_value=None):
        [res] = agent.execute_plan(_plan(_llm("l")))
    assert res.status == "failure"


def test_dependency_cycle_does_not_run():
    agent = _agent()
    a, b = _write("a", path="a"), _write("b", path="b")
    a.dependencies, b.dependencies = ["b"], ["a"]
    guard = ActionGuard(confirm=False)
    results = agent.execute_plan(_plan(a, b), action_guard=guard)
    agent.tool_dispatcher.dispatch.assert_not_called()
    assert [r.status for r in results] == ["failure", "failure"]
    assert {x.step_id for x in guard.blocked} == {"a", "b"}


def test_unknown_dependency_id_is_still_ignored():
    """Planners emit placeholder ids ("step_1_id"); those stay satisfied."""
    agent = _agent()
    [res] = agent.execute_plan(_plan(_read("r", deps=["step_1_id"])))
    assert res.status == "success"


def test_retry_checks_dependencies():
    agent = _agent()
    plan = _plan(_read("r"), _write("w", deps=["r"]))
    prior = [
        StepResult(step_id="r", status="failure", error="boom"),
        StepResult(step_id="w", status="failure", error="tool error"),
    ]
    guard = ActionGuard(confirm=False)
    [res] = agent.execute_plan(
        plan, start_from_step="w", prior_results=prior, action_guard=guard
    )
    agent.tool_dispatcher.dispatch.assert_not_called()
    assert res.error.startswith("Dependency failed")


def test_retry_resumes_dependency_blocked_dependents():
    agent = _agent()
    plan = _plan(
        _read("r"),
        _read("r2", deps=["r"], path="b"),
        _read("r3", deps=["r2"], path="c"),
    )
    prior = [
        StepResult(step_id="r", status="failure", error="boom"),
        StepResult(step_id="r2", status="failure", error="Dependency failed: r"),
        StepResult(step_id="r3", status="failure", error="Dependency failed: r2"),
    ]
    for s in plan.steps[1:]:
        s.status = "skipped"
    results = agent.execute_plan(plan, start_from_step="r", prior_results=prior)
    assert [r.step_id for r in results] == ["r", "r2", "r3"]
    assert all(r.status == "success" for r in results)
    assert [s.status for s in plan.steps] == ["completed"] * 3


def test_retry_does_not_resume_other_failures():
    agent = _agent()
    plan = _plan(_read("r"), _read("r2", deps=["r"], path="b"))
    prior = [
        StepResult(step_id="r", status="failure", error="boom"),
        StepResult(step_id="r2", status="failure", error="tool error"),
    ]
    results = agent.execute_plan(plan, start_from_step="r", prior_results=prior)
    assert [r.step_id for r in results] == ["r"]


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------


@pytest.fixture()
def graph(monkeypatch):
    monkeypatch.setattr(cg, "record_pipeline_run", MagicMock())
    with patch.multiple(
        "app.core.langgraph.core_graph_v2",
        ComprehensionAgent=MagicMock(),
        OrchestrationAgent=MagicMock(),
        EvaluationAgent=MagicMock(),
    ):
        g = cg.COREGraph()
    g.reasoning_agent = _agent()
    g.comprehension_agent.detect_ambiguities.return_value = []
    return g


def _eval(next_action="finalize", **kw):
    return EvaluationResult(
        overall_status=kw.pop("overall_status", "success"),
        confidence=0.9,
        meets_requirements=True,
        quality_score=0.8,
        next_action=next_action,
        **kw,
    )


def _run(g, **config):
    state = COREState(user_input="update the config file")
    state.config.update(config)
    with patch(
        "app.core.agents.comprehension_gate.decide_gate",
        return_value=GateDecision(outcome="proceed"),
    ):
        out = g.get_graph().invoke(state)
    return COREState(**out) if isinstance(out, dict) else out


def test_full_run_holds_write_and_asks_for_confirmation(graph):
    graph.comprehension_agent.analyze_intent.return_value = UserIntent(
        type="task", description="t", confidence=0.95
    )
    graph.orchestration_agent.create_plan.return_value = _plan(
        _read("r"), _write("w", path="config.yaml")
    )
    # The evaluator wants to retry the "failed" write: that must not loop.
    graph.evaluation_agent.evaluate_execution.return_value = _eval(
        "retry_step", retry_step_id="w", overall_status="needs_retry"
    )

    final = _run(graph)

    assert _dispatched(graph.reasoning_agent) == ["in.txt"]
    assert final.execution_history.count("reasoning") == 1
    assert final.plan.steps[1].requires_hitl is True  # marked by orchestration
    assert [p.capability_id for p in final.pending_confirmations] == [
        "file_operations.write"
    ]
    assert final.eval_result.next_action == "ask_user"
    assert final.response.startswith("I need your go-ahead")
    assert "file_operations.write → config.yaml" in final.response
    assert "approval for: file_operations.write" in final.response
    assert any("confirm tier held 1" in w for w in final.warnings)
    assert "step_failure" not in fault_reasons(final)


def test_full_run_with_approval_dispatches_write(graph):
    graph.comprehension_agent.analyze_intent.return_value = UserIntent(
        type="task", description="t", confidence=0.95
    )
    graph.orchestration_agent.create_plan.return_value = _plan(
        _write("w", path="config.yaml")
    )
    graph.evaluation_agent.evaluate_execution.return_value = _eval()

    final = _run(graph, approved_capabilities=["file_operations.write"])

    assert _dispatched(graph.reasoning_agent) == ["config.yaml"]
    assert final.pending_confirmations == []
    assert not final.response.startswith("I need your go-ahead")


def test_retry_replaces_results_and_block_records(graph):
    agent = graph.reasoning_agent
    plan = _plan(_read("r"), _read("r2", deps=["r"], path="b"))
    agent.tool_dispatcher.dispatch.return_value = {"result": "no", "status": "error"}
    state = COREState(user_input="x", plan=plan)
    state = graph.reasoning_node(state)
    assert [b.step_id for b in state.blocked_actions] == ["r2"]

    agent.tool_dispatcher.dispatch.return_value = {"result": "ok", "status": "success"}
    state.eval_result = _eval("retry_step", retry_step_id="r")
    state = graph.reasoning_node(state)
    assert [(r.step_id, r.status) for r in state.step_results] == [
        ("r", "success"),
        ("r2", "success"),
    ]
    assert state.blocked_actions == []


def test_route_does_not_retry_around_a_held_write(graph, monkeypatch):
    state = COREState(
        user_input="x",
        eval_result=_eval("retry_step", retry_step_id="w"),
        pending_confirmations=[
            PendingConfirmation(
                step_id="w",
                capability_id="file_operations.write",
                side_effect="write",
                reason="write",
            )
        ],
    )
    assert graph.route_from_evaluation(state) == "conversation"
    monkeypatch.setenv("CORE_CONFIRM_WRITES", "0")
    assert graph.route_from_evaluation(state) == "reasoning"


def test_conversation_reports_dependency_blocks():
    g = object.__new__(cg.COREGraph)
    state = COREState(
        user_input="x",
        intent=UserIntent(type="task", description="t", confidence=0.9),
        plan=_plan(_read("r"), _write("w", deps=["r"])),
        step_results=[StepResult(step_id="r", status="failure", outputs={})],
        eval_result=_eval(),
        blocked_actions=[
            BlockedAction(
                step_id="w",
                step_name="write out.txt",
                kind="dependency_failed",
                cause_step_id="r",
            )
        ],
    )
    with patch.object(cg, "record_pipeline_run"):
        out = g.conversation_node(state)
    assert "Not run: a step they depend on did not succeed: write out.txt" in (
        out.response
    )
    assert "I stopped before finishing" not in out.response


def test_held_write_is_not_a_fault():
    held = PendingConfirmation(
        step_id="w",
        capability_id="file_operations.write",
        side_effect="write",
        reason="write",
        target="a",
    )
    state = COREState(
        user_input="x",
        step_results=[
            StepResult(step_id="w", status="failure", error="Awaiting confirmation"),
            StepResult(step_id="d", status="failure", error="Awaiting confirmation"),
        ],
        pending_confirmations=[held],
        blocked_actions=[
            BlockedAction(step_id="d", kind="awaiting_confirmation", cause_step_id="w")
        ],
    )
    assert "step_failure" not in fault_reasons(state)
    report = fault_report_from_state(state)
    assert report.metadata["pending_confirmations"][0]["capability_id"] == (
        "file_operations.write"
    )
    # A real failure next to it still counts.
    state.step_results.append(StepResult(step_id="x", status="failure"))
    assert "step_failure" in fault_reasons(state)
