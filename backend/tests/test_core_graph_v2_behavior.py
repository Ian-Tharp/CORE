"""
Behavior tests for the live C->O->R->E pipeline in
app/core/langgraph/core_graph_v2.py.

All LLM-backed agents are replaced with MagicMocks; the capability registry
and dispatcher are the real in-process ones (no network, no DB).
record_pipeline_run is patched so no intelligence log side effects leak.

Covers:
- Full compiled-graph runs: task happy path, chat path, clarify gate,
  retry_step loop, revise_plan loop cap, retry_step loop cap.
- Node-level failure handling (each agent raising) and fallbacks.
- Comprehension ambiguity merge + non-fatal ambiguity detector failure.
- get_graph lazy compile and get_core_graph singleton.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import app.core.langgraph.core_graph_v2 as cg
from app.core.agents.comprehension_gate import GateDecision
from app.core.telemetry import reset_telemetry
from app.models.core_state import (
    COREState,
    EvaluationResult,
    ExecutionPlan,
    PlanStep,
    StepResult,
    UserIntent,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Fresh telemetry + no intelligence-log side effects + clean singleton."""
    reset_telemetry()
    monkeypatch.setattr(cg, "record_pipeline_run", MagicMock())
    monkeypatch.setattr(cg, "_core_graph_instance", None)
    yield
    reset_telemetry()


@pytest.fixture
def core():
    """A real COREGraph whose four LLM agents are MagicMocks."""
    with patch.multiple(
        "app.core.langgraph.core_graph_v2",
        ComprehensionAgent=MagicMock(),
        OrchestrationAgent=MagicMock(),
        ReasoningAgent=MagicMock(),
        EvaluationAgent=MagicMock(),
    ):
        graph = cg.COREGraph()
    graph.comprehension_agent.detect_ambiguities.return_value = []
    return graph


@pytest.fixture
def gate_proceed():
    with patch(
        "app.core.agents.comprehension_gate.decide_gate",
        return_value=GateDecision(outcome="proceed"),
    ) as m:
        yield m


def _intent(type_="task", confidence=0.95, **kw):
    return UserIntent(
        type=type_, description="summarise the doc", confidence=confidence, **kw
    )


def _eval(next_action="finalize", confidence=0.9, retry_step_id=None, **kw):
    return EvaluationResult(
        overall_status=kw.pop("overall_status", "success"),
        confidence=confidence,
        meets_requirements=True,
        quality_score=0.8,
        next_action=next_action,
        retry_step_id=retry_step_id,
        **kw,
    )


def _plan(*steps, goal="do the thing"):
    steps = steps or (PlanStep(id="s1", name="think", description="llm step"),)
    return ExecutionPlan(goal=goal, steps=list(steps))


def _run(core, user_input="please summarise"):
    out = core.get_graph().invoke(COREState(user_input=user_input))
    return COREState(**out) if isinstance(out, dict) else out


# ---------------------------------------------------------------------------
# Full-graph behaviour
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_task_happy_path_runs_core_in_order_and_returns_step_output(
        self, core, gate_proceed
    ):
        core.comprehension_agent.analyze_intent.return_value = _intent()
        core.orchestration_agent.create_plan.return_value = _plan()
        core.reasoning_agent.execute_plan.return_value = [
            StepResult(step_id="s1", status="success", outputs={"result": "SUMMARY-42"})
        ]
        core.evaluation_agent.evaluate_execution.return_value = _eval()

        final = _run(core)

        assert final.execution_history == [
            "comprehension",
            "orchestration",
            "reasoning",
            "evaluation",
            "conversation",
        ]
        assert final.response == "SUMMARY-42"
        assert final.errors == []
        assert final.gate_outcome == "proceed"
        assert final.completed_at is not None
        cg.record_pipeline_run.assert_called_once()

    def test_conversation_intent_skips_task_loop(self, core):
        core.comprehension_agent.analyze_intent.return_value = _intent(
            "conversation", 0.9
        )

        final = _run(core, "hello there")

        assert final.execution_history == ["comprehension", "conversation"]
        assert "hello there" in final.response
        core.orchestration_agent.create_plan.assert_not_called()

    def test_clarify_gate_surfaces_gate_message_without_planning(self, core):
        core.comprehension_agent.analyze_intent.return_value = _intent()
        with patch(
            "app.core.agents.comprehension_gate.decide_gate",
            return_value=GateDecision(outcome="clarify", message="Which file?"),
        ):
            final = _run(core)

        assert final.execution_history == ["comprehension", "conversation"]
        assert final.gate_outcome == "clarify"
        assert final.response == "Which file?"
        core.orchestration_agent.create_plan.assert_not_called()

    def test_retry_step_reruns_reasoning_and_replaces_failed_result(
        self, core, gate_proceed
    ):
        core.comprehension_agent.analyze_intent.return_value = _intent()
        core.orchestration_agent.create_plan.return_value = _plan(
            PlanStep(id="s1", name="a", description="a"),
            PlanStep(id="s2", name="b", description="b"),
        )
        core.reasoning_agent.execute_plan.side_effect = [
            [
                StepResult(step_id="s1", status="success", outputs={"result": "A"}),
                StepResult(step_id="s2", status="failure"),
            ],
            [StepResult(step_id="s2", status="success", outputs={"result": "B"})],
        ]
        core.evaluation_agent.evaluate_execution.side_effect = [
            _eval("retry_step", retry_step_id="s2", overall_status="needs_retry"),
            _eval(),
        ]

        final = _run(core)

        assert final.execution_history == [
            "comprehension",
            "orchestration",
            "reasoning",
            "evaluation",
            "reasoning",
            "evaluation",
            "conversation",
        ]
        second_call = core.reasoning_agent.execute_plan.call_args_list[1]
        assert second_call.kwargs["start_from_step"] == "s2"
        # failed s2 result replaced, not duplicated
        assert [(r.step_id, r.status) for r in final.step_results] == [
            ("s1", "success"),
            ("s2", "success"),
        ]
        assert final.response == "A\nB"

    def test_revise_plan_loop_is_capped_and_then_finalises(
        self, core, gate_proceed, monkeypatch
    ):
        monkeypatch.setattr(cg, "_MAX_PLAN_REVISIONS", 2)
        core.comprehension_agent.analyze_intent.return_value = _intent()
        core.orchestration_agent.create_plan.side_effect = lambda **kw: ExecutionPlan(
            goal=f"rev{kw['revision']}",
            steps=[PlanStep(id="s1", name="t", description="t")],
            revision=kw["revision"],
        )
        core.reasoning_agent.execute_plan.return_value = []
        core.evaluation_agent.evaluate_execution.return_value = _eval(
            "revise_plan", feedback="be better", overall_status="needs_revision"
        )

        final = _run(core)

        assert final.plan_revisions == 2
        assert final.execution_history.count("orchestration") == 3
        assert final.execution_history[-1] == "conversation"
        revisions = [
            c.kwargs["revision"]
            for c in core.orchestration_agent.create_plan.call_args_list
        ]
        assert revisions == [1, 2, 3]
        # Revisions receive the evaluator's feedback.
        assert (
            core.orchestration_agent.create_plan.call_args_list[1].kwargs[
                "evaluation_feedback"
            ]
            == "be better"
        )

    def test_persistent_retry_step_is_capped_instead_of_recursing_forever(
        self, core, gate_proceed, monkeypatch
    ):
        """BUG REPRO: a step that keeps failing made the evaluator say
        retry_step forever; with no cap the graph blew LangGraph's recursion
        limit (GraphRecursionError) instead of finalising."""
        monkeypatch.setattr(cg, "_MAX_STEP_RETRIES", 2)
        core.comprehension_agent.analyze_intent.return_value = _intent()
        core.orchestration_agent.create_plan.return_value = _plan()
        core.reasoning_agent.execute_plan.return_value = [
            StepResult(step_id="s1", status="failure")
        ]
        core.evaluation_agent.evaluate_execution.return_value = _eval(
            "retry_step", retry_step_id="s1", overall_status="needs_retry"
        )

        final = _run(core)

        # 1 initial run + 2 retries, then forced finalisation.
        assert final.execution_history.count("reasoning") == 3
        assert final.execution_history[-1] == "conversation"


# ---------------------------------------------------------------------------
# Node-level failure handling
# ---------------------------------------------------------------------------


class TestComprehensionNode:
    def test_agent_exception_records_error_and_falls_back_to_conversation(self, core):
        core.comprehension_agent.analyze_intent.side_effect = RuntimeError("llm down")

        state = core.comprehension_node(COREState(user_input="x"))

        assert state.errors == ["Comprehension failed: llm down"]
        assert state.intent.type == "conversation"
        assert state.intent.confidence == 0.5
        assert core.route_from_comprehension(state) == "conversation"

    def test_detected_ambiguities_are_merged_without_duplicates(
        self, core, gate_proceed
    ):
        core.comprehension_agent.analyze_intent.return_value = _intent(
            ambiguities=["which file?"]
        )
        core.comprehension_agent.detect_ambiguities.return_value = [
            "which file?",
            "what format?",
        ]

        state = core.comprehension_node(COREState(user_input="x"))

        assert state.intent.ambiguities == ["which file?", "what format?"]
        # The gate sees the merged ambiguities.
        assert gate_proceed.call_args.args[0].ambiguities == [
            "which file?",
            "what format?",
        ]

    def test_ambiguity_detector_failure_is_non_fatal(self, core, gate_proceed):
        core.comprehension_agent.analyze_intent.return_value = _intent()
        core.comprehension_agent.detect_ambiguities.side_effect = ValueError("regex")

        state = core.comprehension_node(COREState(user_input="x"))

        assert state.errors == []
        assert state.intent.type == "task"
        assert state.gate_outcome == "proceed"
        assert state.gate_message is None

    def test_real_gate_refuses_when_named_tool_is_absent(self, core):
        core.comprehension_agent.analyze_intent.return_value = _intent(
            requires_tools=True, tools_needed=["teleporter_xyz"]
        )

        state = core.comprehension_node(COREState(user_input="teleport me"))

        assert state.gate_outcome == "refuse"
        assert state.gate_message
        assert core.route_from_comprehension(state) == "conversation"


class TestOrchestrationNode:
    def test_agent_exception_records_error_and_keeps_no_plan(self, core):
        core.orchestration_agent.create_plan.side_effect = RuntimeError("boom")
        state = COREState(user_input="x", intent=_intent())

        state = core.orchestration_node(state)

        assert state.errors == ["Orchestration failed: boom"]
        assert state.plan is None

    def test_invented_tool_step_is_dropped_with_warning(self, core):
        core.orchestration_agent.create_plan.return_value = _plan(
            PlanStep(id="ok", name="ok", description="llm"),
            PlanStep(id="bad", name="bad", description="x", tool="made_up_tool"),
        )
        state = core.orchestration_node(COREState(user_input="x", intent=_intent()))

        assert [s.id for s in state.plan.steps] == ["ok"]
        assert state.grounding_report["dropped"][0]["reason"] == "invented_tool"
        assert any("grounding dropped 1" in w for w in state.warnings)

    def test_none_plan_is_tolerated(self, core):
        core.orchestration_agent.create_plan.return_value = None
        state = core.orchestration_node(COREState(user_input="x", intent=_intent()))
        assert state.plan is None
        assert state.errors == []
        assert state.grounding_report is None


class TestReasoningNode:
    def test_no_plan_records_error_without_calling_agent(self, core):
        state = core.reasoning_node(COREState(user_input="x"))
        assert state.errors == ["No plan available for reasoning"]
        core.reasoning_agent.execute_plan.assert_not_called()

    def test_agent_exception_records_error(self, core):
        core.reasoning_agent.execute_plan.side_effect = RuntimeError("tool crash")
        state = core.reasoning_node(COREState(user_input="x", plan=_plan()))
        assert state.errors == ["Reasoning failed: tool crash"]
        assert state.step_results == []

    def test_enable_tools_config_is_forwarded(self, core):
        core.reasoning_agent.execute_plan.return_value = []
        state = COREState(user_input="x", plan=_plan())
        state.config = {"enable_tools": False}
        core.reasoning_node(state)
        kwargs = core.reasoning_agent.execute_plan.call_args.kwargs
        assert kwargs["enable_tools"] is False
        assert kwargs["start_from_step"] is None


class TestEvaluationNode:
    def test_agent_exception_defaults_to_finalize_to_avoid_loops(self, core):
        core.evaluation_agent.evaluate_execution.side_effect = RuntimeError("judge")
        state = core.evaluation_node(COREState(user_input="x", plan=_plan()))

        assert state.errors == ["Evaluation failed: judge"]
        assert state.eval_result.next_action == "finalize"
        assert state.eval_result.overall_status == "success"
        assert core.route_from_evaluation(state) == "conversation"

    def test_none_evaluation_routes_to_conversation(self, core):
        core.evaluation_agent.evaluate_execution.return_value = None
        state = core.evaluation_node(COREState(user_input="x"))
        assert state.eval_result is None
        assert core.route_from_evaluation(state) == "conversation"


class TestConversationNode:
    def test_gate_message_takes_precedence(self, core):
        state = COREState(
            user_input="x",
            gate_message="Please clarify",
            intent=_intent("conversation"),
        )
        assert core.conversation_node(state).response == "Please clarify"

    def test_internal_failure_yields_safe_error_response(self, core):
        state = COREState(user_input="x", plan=_plan(), eval_result=_eval())
        # A step result whose outputs blow up on iteration.
        bad = MagicMock()
        bad.outputs.items.side_effect = RuntimeError("corrupt outputs")
        state.step_results = [bad]

        state = core.conversation_node(state)

        assert state.response == "An error occurred while formulating the response."
        assert state.errors == ["Conversation failed: corrupt outputs"]
        cg.record_pipeline_run.assert_called_once_with(state)

    def test_intelligence_log_failure_is_non_fatal(self, core):
        cg.record_pipeline_run.side_effect = RuntimeError("disk full")
        state = core.conversation_node(
            COREState(user_input="hi", intent=_intent("conversation"))
        )
        assert "hi" in state.response
        assert state.errors == []

    def test_no_plan_or_eval_gives_default_message(self, core):
        state = core.conversation_node(COREState(user_input="x", intent=_intent()))
        assert state.response.startswith("Task processed.")

    def test_low_confidence_appends_metadata_and_warning(self, core):
        state = COREState(
            user_input="x",
            plan=_plan(),
            eval_result=_eval(confidence=0.4),
            step_results=[
                StepResult(
                    step_id="s1",
                    status="success",
                    outputs={"commit_sha": "abc", "summary": "named"},
                    artifacts=["out.txt"],
                )
            ],
        )
        resp = core.conversation_node(state).response
        assert "named" in resp
        assert "abc" not in resp  # technical output filtered
        assert "out.txt" in resp
        assert "Low confidence (40%)" in resp


# ---------------------------------------------------------------------------
# Graph construction / singleton
# ---------------------------------------------------------------------------


class TestGraphConstruction:
    def test_get_graph_compiles_lazily_once(self, core):
        assert core.compiled_graph is None
        g1 = core.get_graph()
        g2 = core.get_graph()
        assert g1 is g2
        assert set(g1.get_graph().nodes) >= {
            "comprehension",
            "orchestration",
            "reasoning",
            "evaluation",
            "conversation",
        }

    def test_get_core_graph_returns_initialised_singleton(self):
        with patch.multiple(
            "app.core.langgraph.core_graph_v2",
            ComprehensionAgent=MagicMock(),
            OrchestrationAgent=MagicMock(),
            ReasoningAgent=MagicMock(),
            EvaluationAgent=MagicMock(),
        ):
            first = cg.get_core_graph()
            second = cg.get_core_graph()
        assert first is second
        assert first.compiled_graph is not None


class TestRetryLoopGuard:
    def test_retry_allowed_below_cap_and_counted_per_plan_revision(
        self, core, monkeypatch
    ):
        monkeypatch.setattr(cg, "_MAX_STEP_RETRIES", 1)
        state = COREState(user_input="x", eval_result=_eval("retry_step"))
        state.execution_history = ["comprehension", "orchestration", "reasoning"]
        assert core.route_from_evaluation(state) == "reasoning"
        state.execution_history += ["evaluation", "reasoning"]
        assert core.route_from_evaluation(state) == "conversation"
        # A fresh plan revision resets the retry budget.
        state.execution_history += ["evaluation", "orchestration", "reasoning"]
        assert core.route_from_evaluation(state) == "reasoning"


class TestConversationLogFallback:
    def test_meaningful_logs_used_when_no_outputs(self, core):
        state = COREState(
            user_input="x",
            plan=_plan(),
            eval_result=_eval(),
            step_results=[
                StepResult(
                    step_id="s1",
                    status="success",
                    logs=["Executing step", "Tool: shell", "Found 3 matches"],
                )
            ],
        )
        assert core.conversation_node(state).response == "Found 3 matches"

    def test_empty_results_say_task_completed(self, core):
        state = COREState(user_input="x", plan=_plan(), eval_result=_eval())
        assert core.conversation_node(state).response == "Task completed."
