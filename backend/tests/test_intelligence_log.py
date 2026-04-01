"""
Tests for Intelligence Layer logging (pipeline run summaries).

Sentinel-value contract:
- Remove record_pipeline_run() call from conversation_node → log not emitted → test fails
- Remove intent fields from entry → intent_type absent → test fails
- Remove eval fields from entry → eval_status absent → test fails
- Remove step counts → steps_succeeded absent → test fails
- Return None from record_pipeline_run → entry not inspectable → test fails
"""

from __future__ import annotations

import json
import logging
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from app.core.intelligence_log import record_pipeline_run, get_intelligence_logger
from app.models.core_state import (
    COREState,
    UserIntent,
    ExecutionPlan,
    PlanStep,
    StepResult,
    EvaluationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_intent(type_: str = "task", category: str = None) -> UserIntent:
    return UserIntent(
        type=type_,
        description="do something",
        confidence=0.9,
        task_category=category,
    )


def _make_plan(*step_names: str) -> ExecutionPlan:
    steps = [PlanStep(name=n, description=f"do {n}") for n in step_names]
    return ExecutionPlan(goal="test goal", steps=list(steps))


def _make_result(status: str = "success") -> StepResult:
    return StepResult(step_id="s1", status=status, outputs={"result": "ok"})


def _make_eval(status: str = "success", quality: float = 0.9) -> EvaluationResult:
    return EvaluationResult(
        overall_status=status,
        confidence=0.85,
        meets_requirements=True,
        quality_score=quality,
        feedback="ok",
        next_action="finalize",
    )


def _full_state() -> COREState:
    state = COREState(user_input="build a login page", session_id="sess-test")
    state.intent = _make_intent("task", "code")
    state.plan = _make_plan("step-a", "step-b", "step-c")
    state.step_results = [_make_result("success"), _make_result("success"), _make_result("failure")]
    state.eval_result = _make_eval("needs_retry", 0.6)
    state.plan_revisions = 1
    state.response = "Done."
    state.execution_history = ["comprehension", "orchestration", "reasoning", "evaluation", "conversation"]
    return state


# ---------------------------------------------------------------------------
# record_pipeline_run — schema
# ---------------------------------------------------------------------------

class TestRecordPipelineRunSchema:
    def test_returns_dict(self):
        """record_pipeline_run must return a dict (not None)."""
        state = COREState(user_input="hi")
        entry = record_pipeline_run(state)
        assert isinstance(entry, dict)

    def test_event_field_is_correct(self):
        """
        SENTINEL TEST — event field must be 'pipeline.run_complete'.
        Change the event key → assertions in dashboards break → test fails.
        """
        state = COREState(user_input="test")
        entry = record_pipeline_run(state)
        assert entry["event"] == "pipeline.run_complete"

    def test_required_keys_present(self):
        """
        SENTINEL TEST — all required schema keys must be present.
        Remove any key → downstream consumers break → test fails.
        """
        required_keys = {
            "event", "session_id", "user_input_len",
            "intent_type", "task_category", "intent_confidence",
            "plan_goal", "plan_step_count", "plan_revisions",
            "steps_succeeded", "steps_failed",
            "eval_status", "eval_quality", "eval_confidence", "eval_next_action",
            "response_len", "nodes_visited", "errors_count", "timestamp",
        }
        state = COREState(user_input="hi")
        entry = record_pipeline_run(state)
        missing = required_keys - set(entry.keys())
        assert not missing, f"Missing keys in intelligence log entry: {missing}"

    def test_timestamp_is_iso_format(self):
        """timestamp must be parseable ISO-8601."""
        from datetime import datetime
        state = COREState(user_input="hi")
        entry = record_pipeline_run(state)
        # Should not raise
        dt = datetime.fromisoformat(entry["timestamp"])
        assert dt is not None


# ---------------------------------------------------------------------------
# record_pipeline_run — field values
# ---------------------------------------------------------------------------

class TestRecordPipelineRunValues:
    def test_intent_fields_populated(self):
        """
        SENTINEL TEST — intent_type and task_category must come from state.intent.
        Remove intent fields → None values → test fails.
        """
        state = COREState(user_input="test")
        state.intent = _make_intent("task", "code")
        entry = record_pipeline_run(state)
        assert entry["intent_type"] == "task"
        assert entry["task_category"] == "code"
        assert abs(entry["intent_confidence"] - 0.9) < 0.001

    def test_plan_fields_populated(self):
        """
        SENTINEL TEST — plan_goal and plan_step_count must come from state.plan.
        """
        state = COREState(user_input="test")
        state.plan = _make_plan("a", "b", "c")
        entry = record_pipeline_run(state)
        assert entry["plan_goal"] == "test goal"
        assert entry["plan_step_count"] == 3

    def test_step_counts_correct(self):
        """
        SENTINEL TEST — steps_succeeded and steps_failed must be counted from step_results.
        Hardcode 0 → counts wrong → test fails.
        """
        state = COREState(user_input="test")
        state.step_results = [
            _make_result("success"),
            _make_result("success"),
            _make_result("failure"),
        ]
        entry = record_pipeline_run(state)
        assert entry["steps_succeeded"] == 2
        assert entry["steps_failed"] == 1

    def test_eval_fields_populated(self):
        """
        SENTINEL TEST — eval_status, eval_quality, eval_confidence must come from eval_result.
        """
        state = COREState(user_input="test")
        state.eval_result = _make_eval("needs_retry", 0.65)
        entry = record_pipeline_run(state)
        assert entry["eval_status"] == "needs_retry"
        assert abs(entry["eval_quality"] - 0.65) < 0.001
        assert entry["eval_next_action"] == "finalize"

    def test_plan_revisions_field(self):
        """
        SENTINEL TEST — plan_revisions must be recorded from state.plan_revisions.
        """
        state = COREState(user_input="test")
        state.plan_revisions = 3
        entry = record_pipeline_run(state)
        assert entry["plan_revisions"] == 3

    def test_nodes_visited_is_list(self):
        state = COREState(user_input="test")
        state.execution_history = ["comprehension", "orchestration", "conversation"]
        entry = record_pipeline_run(state)
        assert isinstance(entry["nodes_visited"], list)
        assert "comprehension" in entry["nodes_visited"]
        assert "conversation" in entry["nodes_visited"]

    def test_response_len_correct(self):
        state = COREState(user_input="test")
        state.response = "A" * 150
        entry = record_pipeline_run(state)
        assert entry["response_len"] == 150

    def test_user_input_len_correct(self):
        state = COREState(user_input="hello world")
        entry = record_pipeline_run(state)
        assert entry["user_input_len"] == len("hello world")

    def test_none_intent_produces_none_fields(self):
        """None intent must not crash; fields must be None."""
        state = COREState(user_input="test")
        # No intent set
        entry = record_pipeline_run(state)
        assert entry["intent_type"] is None
        assert entry["task_category"] is None
        assert entry["intent_confidence"] is None

    def test_none_plan_produces_zero_step_count(self):
        """No plan → plan_step_count == 0."""
        state = COREState(user_input="test")
        entry = record_pipeline_run(state)
        assert entry["plan_step_count"] == 0
        assert entry["plan_goal"] is None

    def test_none_eval_produces_none_fields(self):
        """No eval_result → eval fields are None."""
        state = COREState(user_input="test")
        entry = record_pipeline_run(state)
        assert entry["eval_status"] is None
        assert entry["eval_quality"] is None


# ---------------------------------------------------------------------------
# record_pipeline_run — logger emission
# ---------------------------------------------------------------------------

class TestIntelligenceLogEmission:
    def test_entry_emitted_via_intelligence_logger(self):
        """
        SENTINEL TEST — record_pipeline_run must emit to the 'core.intelligence' logger.
        Remove _log.info() call → logger never receives message → test fails.
        """
        intel_logger = get_intelligence_logger()
        records: list = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _Capture()
        intel_logger.addHandler(handler)
        intel_logger.setLevel(logging.INFO)

        try:
            state = COREState(user_input="sentinel log test")
            record_pipeline_run(state)
        finally:
            intel_logger.removeHandler(handler)

        assert len(records) == 1, "Expected exactly one log record"
        assert "pipeline.run_complete" in records[0].getMessage()

    def test_emitted_message_is_valid_json(self):
        """
        SENTINEL TEST — the emitted log message must be parseable JSON.
        Log as plain text → json.loads raises → test fails.
        """
        intel_logger = get_intelligence_logger()
        records: list = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _Capture()
        intel_logger.addHandler(handler)
        intel_logger.setLevel(logging.INFO)

        try:
            state = COREState(user_input="json log test")
            record_pipeline_run(state)
        finally:
            intel_logger.removeHandler(handler)

        msg = records[0].getMessage()
        parsed = json.loads(msg)  # must not raise
        assert parsed["event"] == "pipeline.run_complete"


# ---------------------------------------------------------------------------
# conversation_node integration
# ---------------------------------------------------------------------------

class TestConversationNodeEmitsIntelligenceLog:
    """
    SENTINEL TEST — conversation_node must call record_pipeline_run().
    Remove the call → no log entry → spy not called → test fails.
    """

    def test_conversation_node_calls_record_pipeline_run(self):
        from app.core.langgraph.core_graph_v2 import COREGraph

        graph = COREGraph.__new__(COREGraph)

        state = COREState(user_input="test intelligence integration")
        state.intent = _make_intent("conversation")

        with patch("app.core.langgraph.core_graph_v2.record_pipeline_run") as spy:
            spy.return_value = {"event": "pipeline.run_complete"}
            graph.conversation_node(state)

        spy.assert_called_once_with(state)

    def test_conversation_node_continues_despite_log_error(self):
        """If record_pipeline_run raises, conversation_node must not propagate the error."""
        from app.core.langgraph.core_graph_v2 import COREGraph

        graph = COREGraph.__new__(COREGraph)
        state = COREState(user_input="test resilience")
        state.intent = _make_intent("conversation")

        with patch("app.core.langgraph.core_graph_v2.record_pipeline_run",
                   side_effect=RuntimeError("log exploded")):
            # Must not raise
            result = graph.conversation_node(state)

        assert result is not None
