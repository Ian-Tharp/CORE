"""
Graph-level tests for Phase 3 of the comprehension-grounded gate
(docs/architecture/comprehension-grounded-gate.md): memory grounding in
comprehension_node and deterministic per-step checks in reasoning/evaluation.

Every agent LLM call is mocked; the capability registry and dispatcher are real
(offline). Memory uses LangGraph's InMemoryStore.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langgraph.store.memory import InMemoryStore

from app.core.langgraph.core_graph_v2 import COREGraph
from app.core.memory.grounding import StoreMemoryGrounder
from app.models.core_state import (
    COREState,
    EvaluationResult,
    ExecutionPlan,
    PlanStep,
    StepResult,
    UserIntent,
)

FLAG = "CORE_COMPREHENSION_MEMORY_GROUNDING"
USER = "u1"

# A registry-covered task whose wording trips neither detect_ambiguities nor any
# knowledge cue: the gate proceeds with or without memory grounding.
PLAIN_TASK = "read main.py and list files in the workspace"
# Same capability, plus a topic-bearing knowledge cue ("my preferred stack").
STACK_TASK = "list files in the workspace using my preferred stack"


def _task_intent(description: str, **overrides) -> UserIntent:
    fields = dict(
        type="task",
        description=description,
        confidence=0.9,
        requires_tools=True,
        tools_needed=["file_operations"],
    )
    fields.update(overrides)
    return UserIntent(**fields)


def _analyze(intent: UserIntent):
    """side_effect for analyze_intent: a fresh copy per call (the node mutates it)."""
    return lambda *_a, **_k: intent.model_copy(deep=True)


def _comprehend(graph: COREGraph, user_input: str, intent: UserIntent):
    state = COREState(user_input=user_input, user_id=USER)
    with patch.object(
        graph.comprehension_agent, "analyze_intent", side_effect=_analyze(intent)
    ):
        state = graph.comprehension_node(state)
    return state, graph.route_from_comprehension(state)


def _snapshot(state: COREState, route: str) -> dict:
    return {
        "gate_outcome": state.gate_outcome,
        "gate_message": state.gate_message,
        "route": route,
        "ambiguities": list(state.intent.ambiguities),
    }


def _grounder_with(*facts: str) -> StoreMemoryGrounder:
    grounder = StoreMemoryGrounder(InMemoryStore())
    for fact in facts:
        grounder.remember(fact, kind="fact", user_id=USER)
    return grounder


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(FLAG, "1")


# ---------------------------------------------------------------------------
# Memory grounding in comprehension_node
# ---------------------------------------------------------------------------

FLAG_OFF_CASES = {
    "proceed_task": (PLAIN_TASK, _task_intent(PLAIN_TASK)),
    "low_confidence_clarify": (PLAIN_TASK, _task_intent(PLAIN_TASK, confidence=0.3)),
    "missing_tool_refuse": (
        "browse example.com",
        _task_intent("browse example.com", tools_needed=["browser"]),
    ),
    "conversation": (
        "hello there, how are you",
        UserIntent(type="conversation", description="greeting", confidence=0.95),
    ),
    "like_last_time": (
        "write the release notes like last time",
        _task_intent("write the release notes like last time"),
    ),
    "preferred_stack": (STACK_TASK, _task_intent(STACK_TASK)),
}


class TestFlagOffUnchanged:
    @pytest.mark.parametrize(
        "user_input,intent", FLAG_OFF_CASES.values(), ids=FLAG_OFF_CASES.keys()
    )
    def test_routing_identical_to_pre_phase3(self, flag_off, user_input, intent):
        graph = COREGraph()
        state, route = _comprehend(graph, user_input, intent)

        baseline_graph = COREGraph()
        with patch.object(baseline_graph, "_ground_in_memory", lambda *a, **k: None):
            base_state, base_route = _comprehend(baseline_graph, user_input, intent)

        assert _snapshot(state, route) == _snapshot(base_state, base_route)
        assert state.memory_grounding is None
        assert graph.memory_grounder is None  # never built when off

    def test_expected_outcomes_sanity(self, flag_off):
        """Pin the baseline outcomes the parametrized cases rely on."""
        expected = {
            "proceed_task": ("proceed", "orchestration"),
            "low_confidence_clarify": ("clarify", "conversation"),
            "missing_tool_refuse": ("refuse", "conversation"),
            "conversation": ("proceed", "conversation"),
            "like_last_time": ("clarify", "conversation"),
            "preferred_stack": ("proceed", "orchestration"),
        }
        graph = COREGraph()
        for name, (user_input, intent) in FLAG_OFF_CASES.items():
            state, route = _comprehend(graph, user_input, intent)
            assert (state.gate_outcome, route) == expected[name], name


class TestFlagOnMemoryGrounding:
    def test_knowledge_gap_clarifies(self, flag_on):
        graph = COREGraph()
        graph.memory_grounder = _grounder_with()  # empty memory

        state, route = _comprehend(graph, STACK_TASK, _task_intent(STACK_TASK))

        assert state.gate_outcome == "clarify"
        assert route == "conversation"
        assert "my preferred stack" in state.gate_message
        assert state.memory_grounding["gaps"] == ["my preferred stack"]
        assert state.memory_grounding["hits"] == []

    def test_like_last_time_gap_recorded(self, flag_on):
        graph = COREGraph()
        graph.memory_grounder = _grounder_with()
        text = "write the release notes like last time"

        state, route = _comprehend(graph, text, _task_intent(text))

        assert state.gate_outcome == "clarify"
        assert route == "conversation"
        assert "like last time" in state.gate_message
        assert "like last time" in state.memory_grounding["gaps"]

    def test_lazily_builds_default_grounder(self, flag_on, monkeypatch):
        monkeypatch.delenv("CORE_MEMORY_STORE", raising=False)
        graph = COREGraph()
        assert graph.memory_grounder is None

        state, route = _comprehend(graph, STACK_TASK, _task_intent(STACK_TASK))

        assert isinstance(graph.memory_grounder, StoreMemoryGrounder)
        assert state.gate_outcome == "clarify"  # fresh store knows nothing

    def test_memory_covers_cue_proceeds(self, flag_on):
        fact = "stack: FastAPI and PostgreSQL for the workspace"
        graph = COREGraph()
        graph.memory_grounder = _grounder_with(fact)

        state, route = _comprehend(graph, STACK_TASK, _task_intent(STACK_TASK))

        assert state.gate_outcome == "proceed"
        assert route == "orchestration"
        assert state.memory_grounding["gaps"] == []
        assert [h["content"] for h in state.memory_grounding["hits"]] == [fact]
        assert fact in state.intent.context_retrieved

    def test_other_users_memory_does_not_cover(self, flag_on):
        graph = COREGraph()
        grounder = StoreMemoryGrounder(InMemoryStore())
        grounder.remember(
            "stack: FastAPI and PostgreSQL for the workspace",
            kind="fact",
            user_id="someone-else",
        )
        graph.memory_grounder = grounder

        state, route = _comprehend(graph, STACK_TASK, _task_intent(STACK_TASK))

        assert state.gate_outcome == "clarify"
        assert state.memory_grounding["gaps"] == ["my preferred stack"]

    def test_no_cues_same_as_flag_off(self, flag_on, monkeypatch):
        graph = COREGraph()
        graph.memory_grounder = _grounder_with()
        on_state, on_route = _comprehend(graph, PLAIN_TASK, _task_intent(PLAIN_TASK))

        monkeypatch.delenv(FLAG)
        off_state, off_route = _comprehend(
            COREGraph(), PLAIN_TASK, _task_intent(PLAIN_TASK)
        )

        assert _snapshot(on_state, on_route) == _snapshot(off_state, off_route)
        assert on_state.gate_outcome == "proceed"
        assert on_state.memory_grounding == {"hits": [], "cues": [], "gaps": []}

    def test_grounder_error_degrades_to_no_memory(self, flag_on):
        class Broken:
            def ground(self, *a, **k):
                raise RuntimeError("store down")

        graph = COREGraph()
        graph.memory_grounder = Broken()

        state, route = _comprehend(graph, STACK_TASK, _task_intent(STACK_TASK))

        assert not state.errors
        assert any(
            "memory grounding failed" in w and "store down" in w for w in state.warnings
        )
        assert state.memory_grounding is None
        assert state.gate_outcome == "proceed"
        assert route == "orchestration"
        assert state.intent.ambiguities == []


# ---------------------------------------------------------------------------
# Per-step evaluation through reasoning_node → evaluation_node
# ---------------------------------------------------------------------------

_SCHEMA = {
    "type": "object",
    "properties": {"content": {"type": "string"}},
    "required": ["content"],
}


def _eval_result() -> EvaluationResult:
    return EvaluationResult(
        overall_status="success",
        confidence=0.9,
        meets_requirements=True,
        quality_score=0.9,
        next_action="finalize",
    )


def _plan_state() -> tuple[COREState, PlanStep, PlanStep]:
    read = PlanStep(
        name="read",
        description="read main.py",
        tool="file_operations",
        params={"action": "read", "path": "main.py"},
        output_schema=_SCHEMA,
    )
    listing = PlanStep(
        name="list",
        description="list the workspace",
        tool="file_operations",
        params={"action": "list", "path": "."},
    )
    state = COREState(
        user_input=PLAIN_TASK,
        intent=_task_intent(PLAIN_TASK),
        plan=ExecutionPlan(goal="read and list", steps=[read, listing]),
    )
    return state, read, listing


def _run_reason_and_eval(graph: COREGraph, state: COREState, results):
    with (
        patch.object(graph.reasoning_agent, "execute_plan", return_value=results),
        patch.object(
            graph.evaluation_agent, "evaluate_execution", return_value=_eval_result()
        ) as evaluate,
    ):
        state = graph.reasoning_node(state)
        state = graph.evaluation_node(state)
    return state, evaluate


class TestStepChecksThroughNodes:
    def test_clean_run_keeps_evaluate_call_unchanged(self):
        graph = COREGraph()
        state, read, listing = _plan_state()
        results = [
            StepResult(step_id=read.id, status="success", outputs={"content": "x"}),
            StepResult(step_id=listing.id, status="success", outputs={"files": ["a"]}),
        ]

        state, evaluate = _run_reason_and_eval(graph, state, results)

        assert [c["step_id"] for c in state.step_checks] == [read.id, listing.id]
        assert all(c["passed"] for c in state.step_checks)
        evaluate.assert_called_once()
        assert set(evaluate.call_args.kwargs) == {
            "user_input",
            "intent",
            "plan",
            "step_results",
        }

    def test_success_with_empty_outputs_is_flagged(self):
        graph = COREGraph()
        state, read, listing = _plan_state()
        results = [
            StepResult(step_id=read.id, status="success", outputs={"content": "x"}),
            StepResult(step_id=listing.id, status="success", outputs={}),
        ]

        state, evaluate = _run_reason_and_eval(graph, state, results)

        failing = [c for c in state.step_checks if not c["passed"]]
        assert [c["step_id"] for c in failing] == [listing.id]
        assert failing[0]["failures"] == ["no_output"]
        passed = evaluate.call_args.kwargs["step_checks"]
        assert [c["step_id"] for c in passed] == [listing.id]

    def test_output_schema_violation_is_flagged(self):
        graph = COREGraph()
        state, read, listing = _plan_state()
        results = [
            StepResult(step_id=read.id, status="success", outputs={"content": 42}),
            StepResult(step_id=listing.id, status="success", outputs={"files": ["a"]}),
        ]

        state, evaluate = _run_reason_and_eval(graph, state, results)

        passed = evaluate.call_args.kwargs["step_checks"]
        assert [c["step_id"] for c in passed] == [read.id]
        assert passed[0]["failures"] == ["schema_mismatch"]

    def test_reasoning_error_still_records_checks(self):
        graph = COREGraph()
        state, read, listing = _plan_state()
        with patch.object(
            graph.reasoning_agent, "execute_plan", side_effect=RuntimeError("boom")
        ):
            state = graph.reasoning_node(state)

        assert any("Reasoning failed" in e for e in state.errors)
        # Steps are still "pending" with no result → skipped, not failed.
        assert state.step_checks == []


# ---------------------------------------------------------------------------
# Full compiled-graph runs
# ---------------------------------------------------------------------------


def _draft_plan(**_kwargs) -> ExecutionPlan:
    return ExecutionPlan(
        goal="read main.py",
        steps=[
            PlanStep(
                name="read",
                description="read main.py",
                tool="file_operations",
                params={"action": "read", "path": "main.py"},
            )
        ],
    )


def _execute(plan, **_kwargs):
    return [
        StepResult(step_id=s.id, status="success", outputs={"content": "print(1)"})
        for s in plan.steps
    ]


def _invoke(graph: COREGraph, user_input: str, intent: UserIntent):
    compiled = graph.get_graph()
    with (
        patch.object(
            graph.comprehension_agent, "analyze_intent", side_effect=_analyze(intent)
        ),
        patch.object(
            graph.orchestration_agent, "create_plan", side_effect=_draft_plan
        ) as create_plan,
        patch.object(
            graph.reasoning_agent, "execute_plan", side_effect=_execute
        ) as execute_plan,
        patch.object(
            graph.evaluation_agent, "evaluate_execution", return_value=_eval_result()
        ) as evaluate,
    ):
        out = compiled.invoke(COREState(user_input=user_input, user_id=USER))
    final = out if isinstance(out, COREState) else COREState(**out)
    return final, create_plan, execute_plan, evaluate


class TestCompiledGraph:
    def test_full_loop_flag_off(self, flag_off):
        graph = COREGraph()
        final, create_plan, execute_plan, evaluate = _invoke(
            graph, PLAIN_TASK, _task_intent(PLAIN_TASK)
        )

        assert final.execution_history == [
            "comprehension",
            "orchestration",
            "reasoning",
            "evaluation",
            "conversation",
        ]
        assert final.gate_outcome == "proceed"
        assert final.memory_grounding is None
        assert len(final.plan.steps) == 1  # survived registry grounding
        assert len(final.step_checks) == 1 and final.step_checks[0]["passed"]
        assert "step_checks" not in evaluate.call_args.kwargs
        assert "print(1)" in final.response
        assert graph.memory_grounder is None

    def test_full_loop_flag_on_gap_short_circuits(self, flag_on):
        graph = COREGraph()
        graph.memory_grounder = _grounder_with()
        final, create_plan, execute_plan, evaluate = _invoke(
            graph, STACK_TASK, _task_intent(STACK_TASK)
        )

        assert final.execution_history == ["comprehension", "conversation"]
        assert final.gate_outcome == "clarify"
        assert final.response == final.gate_message
        assert "my preferred stack" in final.response
        assert final.memory_grounding["gaps"] == ["my preferred stack"]
        create_plan.assert_not_called()
        execute_plan.assert_not_called()
        evaluate.assert_not_called()
