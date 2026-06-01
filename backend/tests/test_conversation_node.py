"""
Tests for COREGraph.conversation_node — pure response-building logic.

Sentinel-value contract:
- Remove 'result'/'output'/'content' key handling → task outputs never shown → test fails
- Remove technical-key exclusion → raw DB metadata shown to user → test fails
- Remove log fallback → step with logs but no outputs yields empty response → test fails
- Remove artifact list → generated files never mentioned → test fails
- Remove show_metadata/low-confidence gate → metadata always/never shown → test fails
- Remove conversation-intent branch → "I understand" response missing → test fails
- Remove default fallback → None response when no plan/eval → test fails
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.models.core_state import (
    COREState,
    UserIntent,
    ExecutionPlan,
    PlanStep,
    StepResult,
    EvaluationResult,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_graph():
    """Create a COREGraph without calling __init__ (avoids LLM agent setup)."""
    from app.core.langgraph.core_graph_v2 import COREGraph

    graph = object.__new__(COREGraph)
    return graph


def _make_state(**kwargs) -> COREState:
    defaults = {"user_input": "test input"}
    defaults.update(kwargs)
    return COREState(**defaults)


def _make_intent(intent_type: str = "task") -> UserIntent:
    return UserIntent(type=intent_type, description="test", confidence=0.9)


def _make_plan_with_steps(*step_names: str) -> ExecutionPlan:
    steps = [
        PlanStep(id=f"step-{i}", name=n, description=f"do {n}")
        for i, n in enumerate(step_names)
    ]
    return ExecutionPlan(goal="test goal", steps=steps)


def _make_eval(confidence: float = 0.9, quality_score: float = 0.9) -> EvaluationResult:
    return EvaluationResult(
        overall_status="success",
        confidence=confidence,
        meets_requirements=True,
        quality_score=quality_score,
        feedback="ok",
        next_action="finalize",
    )


def _make_step_result(
    step_id: str = "step-0", outputs: dict | None = None, logs: list | None = None
) -> StepResult:
    return StepResult(
        step_id=step_id,
        status="success",
        outputs=outputs or {},
        logs=logs or [],
    )


# Context manager shim for tracer spans (no-op)
class _FakeSpan:
    def set_attribute(self, *a, **kw):
        pass

    def record_exception(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class _FakeTracer:
    def start_as_current_span(self, *a, **kw):
        return _FakeSpan()


# ===========================================================================
# Conversation intent branch
# ===========================================================================


class TestConversationIntentBranch:
    def test_conversation_intent_produces_echo_response(self):
        """
        SENTINEL — conversation intent must produce an echoing response.
        Remove conversation branch → state.response stays None → test fails.
        """
        graph = _make_graph()
        state = _make_state(
            user_input="SENTINEL_USER_MSG", intent=_make_intent("conversation")
        )

        with patch(
            "app.core.langgraph.core_graph_v2.get_tracer", return_value=_FakeTracer()
        ), patch("app.core.langgraph.core_graph_v2.record_pipeline_run"):
            graph.conversation_node(state)

        assert state.response is not None
        assert "SENTINEL_USER_MSG" in state.response

    def test_conversation_response_includes_user_input(self):
        """User input must appear in conversation response."""
        graph = _make_graph()
        state = _make_state(
            user_input="SENTINEL_INPUT", intent=_make_intent("conversation")
        )

        with patch(
            "app.core.langgraph.core_graph_v2.get_tracer", return_value=_FakeTracer()
        ), patch("app.core.langgraph.core_graph_v2.record_pipeline_run"):
            graph.conversation_node(state)

        assert "SENTINEL_INPUT" in state.response


# ===========================================================================
# Task execution branch — output extraction
# ===========================================================================


class TestTaskOutputExtraction:
    def test_result_key_included_in_response(self):
        """
        SENTINEL — 'result' key in step outputs must appear in the response.
        Remove 'result' from the included keys → task results never shown → test fails.
        """
        graph = _make_graph()
        step_result = _make_step_result(outputs={"result": "SENTINEL_RESULT_VALUE"})
        state = _make_state(
            intent=_make_intent("task"),
            plan=_make_plan_with_steps("step-a"),
            step_results=[step_result],
            eval_result=_make_eval(),
        )

        with patch(
            "app.core.langgraph.core_graph_v2.get_tracer", return_value=_FakeTracer()
        ), patch("app.core.langgraph.core_graph_v2.record_pipeline_run"):
            graph.conversation_node(state)

        assert "SENTINEL_RESULT_VALUE" in state.response

    def test_output_key_included_in_response(self):
        """'output' key must also be included."""
        graph = _make_graph()
        step_result = _make_step_result(outputs={"output": "SENTINEL_OUTPUT_VALUE"})
        state = _make_state(
            intent=_make_intent("task"),
            plan=_make_plan_with_steps("step-a"),
            step_results=[step_result],
            eval_result=_make_eval(),
        )

        with patch(
            "app.core.langgraph.core_graph_v2.get_tracer", return_value=_FakeTracer()
        ), patch("app.core.langgraph.core_graph_v2.record_pipeline_run"):
            graph.conversation_node(state)

        assert "SENTINEL_OUTPUT_VALUE" in state.response

    def test_content_key_included_in_response(self):
        """'content' key must also be included."""
        graph = _make_graph()
        step_result = _make_step_result(outputs={"content": "SENTINEL_CONTENT_VALUE"})
        state = _make_state(
            intent=_make_intent("task"),
            plan=_make_plan_with_steps("step-a"),
            step_results=[step_result],
            eval_result=_make_eval(),
        )

        with patch(
            "app.core.langgraph.core_graph_v2.get_tracer", return_value=_FakeTracer()
        ), patch("app.core.langgraph.core_graph_v2.record_pipeline_run"):
            graph.conversation_node(state)

        assert "SENTINEL_CONTENT_VALUE" in state.response

    def test_technical_keys_excluded(self):
        """
        SENTINEL — technical keys ('files_modified', 'query_result', etc.) must be excluded.
        Include all keys → raw DB row counts shown to user → test fails.
        """
        graph = _make_graph()
        step_result = _make_step_result(
            outputs={
                "files_modified": ["auth.py"],
                "query_result": [{"id": 1}],
                "rows_affected": 5,
            }
        )
        state = _make_state(
            intent=_make_intent("task"),
            plan=_make_plan_with_steps("step-a"),
            step_results=[step_result],
            eval_result=_make_eval(),
        )

        with patch(
            "app.core.langgraph.core_graph_v2.get_tracer", return_value=_FakeTracer()
        ), patch("app.core.langgraph.core_graph_v2.record_pipeline_run"):
            graph.conversation_node(state)

        response = state.response or ""
        # Technical keys must not appear in the human-facing response
        assert "rows_affected" not in response
        assert "query_result" not in response

    def test_named_custom_key_included(self):
        """Non-technical named keys (e.g. 'summary') must be included."""
        graph = _make_graph()
        step_result = _make_step_result(outputs={"summary": "SENTINEL_SUMMARY"})
        state = _make_state(
            intent=_make_intent("task"),
            plan=_make_plan_with_steps("step-a"),
            step_results=[step_result],
            eval_result=_make_eval(),
        )

        with patch(
            "app.core.langgraph.core_graph_v2.get_tracer", return_value=_FakeTracer()
        ), patch("app.core.langgraph.core_graph_v2.record_pipeline_run"):
            graph.conversation_node(state)

        assert "SENTINEL_SUMMARY" in state.response


# ===========================================================================
# Task execution branch — log fallback
# ===========================================================================


class TestLogFallback:
    def test_logs_used_when_no_outputs(self):
        """
        SENTINEL — when outputs is empty, non-technical logs must be used as fallback.
        Remove log fallback → steps with no outputs yield empty response → test fails.
        """
        graph = _make_graph()
        step_result = _make_step_result(
            outputs={},
            logs=["SENTINEL_LOG_CONTENT"],
        )
        state = _make_state(
            intent=_make_intent("task"),
            plan=_make_plan_with_steps("step-a"),
            step_results=[step_result],
            eval_result=_make_eval(),
        )

        with patch(
            "app.core.langgraph.core_graph_v2.get_tracer", return_value=_FakeTracer()
        ), patch("app.core.langgraph.core_graph_v2.record_pipeline_run"):
            graph.conversation_node(state)

        assert "SENTINEL_LOG_CONTENT" in state.response

    def test_technical_logs_filtered_in_fallback(self):
        """
        SENTINEL — logs starting with 'Executing', 'Tool:', etc. must be filtered out.
        Include all logs → LLM call traces shown to user → test fails.
        """
        graph = _make_graph()
        step_result = _make_step_result(
            outputs={},
            logs=["Executing tool foo", "Tool: bar", "SENTINEL_GOOD_LOG"],
        )
        state = _make_state(
            intent=_make_intent("task"),
            plan=_make_plan_with_steps("step-a"),
            step_results=[step_result],
            eval_result=_make_eval(),
        )

        with patch(
            "app.core.langgraph.core_graph_v2.get_tracer", return_value=_FakeTracer()
        ), patch("app.core.langgraph.core_graph_v2.record_pipeline_run"):
            graph.conversation_node(state)

        response = state.response or ""
        assert "SENTINEL_GOOD_LOG" in response
        assert "Executing tool foo" not in response


# ===========================================================================
# Task execution branch — artifacts
# ===========================================================================


class TestArtifactListing:
    def test_artifacts_appended_to_response(self):
        """
        SENTINEL — generated files must be listed in the response.
        Remove artifact section → user doesn't know files were created → test fails.
        """
        graph = _make_graph()
        step_result = StepResult(
            step_id="step-0",
            status="success",
            outputs={"result": "done"},
            artifacts=["SENTINEL_ARTIFACT.py"],
        )
        state = _make_state(
            intent=_make_intent("task"),
            plan=_make_plan_with_steps("step-a"),
            step_results=[step_result],
            eval_result=_make_eval(),
        )

        with patch(
            "app.core.langgraph.core_graph_v2.get_tracer", return_value=_FakeTracer()
        ), patch("app.core.langgraph.core_graph_v2.record_pipeline_run"):
            graph.conversation_node(state)

        assert "SENTINEL_ARTIFACT.py" in state.response


# ===========================================================================
# Task execution branch — metadata gate
# ===========================================================================


class TestMetadataGate:
    def test_metadata_hidden_by_default(self):
        """
        SENTINEL — metadata must NOT appear when show_metadata=False and confidence >= 0.7.
        Remove gate → ✓ completed metadata shown for every response → test fails.
        """
        graph = _make_graph()
        step_result = _make_step_result(outputs={"result": "done"})
        plan = _make_plan_with_steps("step-a")
        plan.steps[0].status = "completed"
        eval_result = _make_eval(confidence=0.9)
        state = _make_state(
            intent=_make_intent("task"),
            plan=plan,
            step_results=[step_result],
            eval_result=eval_result,
            config={"show_metadata": False},
        )

        with patch(
            "app.core.langgraph.core_graph_v2.get_tracer", return_value=_FakeTracer()
        ), patch("app.core.langgraph.core_graph_v2.record_pipeline_run"):
            graph.conversation_node(state)

        assert "Completed:" not in state.response and "✓" not in state.response

    def test_metadata_shown_when_show_metadata_flag_set(self):
        """
        SENTINEL — metadata must appear when show_metadata=True.
        Remove show_metadata check → debug info never visible → test fails.
        """
        graph = _make_graph()
        step_result = _make_step_result(outputs={"result": "done"})
        plan = _make_plan_with_steps("step-a")
        plan.steps[0].status = "completed"
        state = _make_state(
            intent=_make_intent("task"),
            plan=plan,
            step_results=[step_result],
            eval_result=_make_eval(confidence=0.9),
            config={"show_metadata": True},
        )

        with patch(
            "app.core.langgraph.core_graph_v2.get_tracer", return_value=_FakeTracer()
        ), patch("app.core.langgraph.core_graph_v2.record_pipeline_run"):
            graph.conversation_node(state)

        assert "Completed:" in state.response or "✓" in state.response

    def test_low_confidence_shows_warning(self):
        """
        SENTINEL — confidence < 0.7 must trigger a low-confidence warning.
        Remove low-confidence branch → user doesn't know to review output → test fails.
        """
        graph = _make_graph()
        step_result = _make_step_result(outputs={"result": "done"})
        plan = _make_plan_with_steps("step-a")
        state = _make_state(
            intent=_make_intent("task"),
            plan=plan,
            step_results=[step_result],
            eval_result=_make_eval(confidence=0.5),  # below 0.7
            config={"show_metadata": False},
        )

        with patch(
            "app.core.langgraph.core_graph_v2.get_tracer", return_value=_FakeTracer()
        ), patch("app.core.langgraph.core_graph_v2.record_pipeline_run"):
            graph.conversation_node(state)

        assert (
            "confidence" in state.response.lower() or "review" in state.response.lower()
        )


# ===========================================================================
# Default fallback branch
# ===========================================================================


class TestDefaultFallback:
    def test_no_plan_no_eval_yields_default_response(self):
        """
        SENTINEL — when there's no plan and no eval_result, must produce default response.
        Let state.response stay None → caller gets None → crashes serialisation → test fails.
        """
        graph = _make_graph()
        state = _make_state(intent=_make_intent("task"))  # task but no plan or eval

        with patch(
            "app.core.langgraph.core_graph_v2.get_tracer", return_value=_FakeTracer()
        ), patch("app.core.langgraph.core_graph_v2.record_pipeline_run"):
            graph.conversation_node(state)

        assert state.response is not None
        assert len(state.response) > 0


# ===========================================================================
# Completion bookkeeping
# ===========================================================================


class TestCompletionBookkeeping:
    def test_completed_at_set_after_node(self):
        """conversation_node must set state.completed_at."""
        graph = _make_graph()
        state = _make_state(intent=_make_intent("conversation"))
        assert state.completed_at is None

        with patch(
            "app.core.langgraph.core_graph_v2.get_tracer", return_value=_FakeTracer()
        ), patch("app.core.langgraph.core_graph_v2.record_pipeline_run"):
            graph.conversation_node(state)

        assert state.completed_at is not None

    def test_conversation_node_added_to_execution_history(self):
        """'conversation' must be appended to execution_history."""
        graph = _make_graph()
        state = _make_state(intent=_make_intent("conversation"))

        with patch(
            "app.core.langgraph.core_graph_v2.get_tracer", return_value=_FakeTracer()
        ), patch("app.core.langgraph.core_graph_v2.record_pipeline_run"):
            graph.conversation_node(state)

        assert "conversation" in state.execution_history
