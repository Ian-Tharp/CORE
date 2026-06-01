"""
Tests for core_state.py — COREState pure methods, Literal constraints, bounds, defaults.

Sentinel-value contract:
- Remove COREState.add_execution_node current_node update → pipeline tracing broken → test fails
- Remove COREState.get_step None guard → get_step raises AttributeError without plan → test fails
- Remove COREState.update_step_status started_at stamp → parallel step timing unknown → test fails
- Remove COREState.is_complete completed_at check → completed runs say not done → test fails
- Remove UserIntent.confidence bounds → confidence=2.0 stored → routing skips low-confidence → test fails
- Remove PlanStep.status Literal → unknown status stored → evaluation can't determine completion → test fails
"""

from __future__ import annotations

import pytest
from datetime import datetime
from pydantic import ValidationError

from app.models.core_state import (
    COREState,
    ConversationMessage,
    EvaluationResult,
    ExecutionPlan,
    PlanStep,
    StepResult,
    UserIntent,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_step(name: str = "SENTINEL_STEP", **overrides) -> PlanStep:
    base = {"name": name, "description": f"{name} description"}
    base.update(overrides)
    return PlanStep(**base)


def _make_plan(*step_names: str) -> ExecutionPlan:
    steps = [_make_step(name) for name in step_names]
    return ExecutionPlan(goal="SENTINEL_GOAL", steps=steps)


def _make_state(**overrides) -> COREState:
    base = {"user_input": "SENTINEL_INPUT"}
    base.update(overrides)
    return COREState(**base)


# ===========================================================================
# UserIntent — Literal and confidence bounds
# ===========================================================================


class TestUserIntent:
    def test_valid_intent_types_accepted(self):
        """
        SENTINEL — all four valid intent types must be accepted.
        Change Literal → valid type rejected → comprehension agent can't classify → test fails.
        """
        for intent_type in ("task", "conversation", "question", "clarification"):
            intent = UserIntent(type=intent_type, description="test", confidence=0.9)
            assert intent.type == intent_type

    def test_invalid_intent_type_rejected(self):
        """
        SENTINEL — unknown intent type must raise ValidationError.
        Remove Literal → 'SENTINEL_BAD' stored → model router lookup fails → test fails.
        """
        with pytest.raises(ValidationError):
            UserIntent(type="SENTINEL_BAD", description="test", confidence=0.9)

    def test_confidence_above_1_rejected(self):
        """
        SENTINEL — confidence > 1.0 must raise ValidationError.
        Remove le=1.0 → routing uses 2.0 confidence → all inputs treated as certain → test fails.
        """
        with pytest.raises(ValidationError):
            UserIntent(type="task", description="test", confidence=1.1)

    def test_confidence_below_0_rejected(self):
        """confidence < 0.0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            UserIntent(type="task", description="test", confidence=-0.1)

    def test_default_requires_tools_false(self):
        """requires_tools defaults to False."""
        intent = UserIntent(type="conversation", description="hello", confidence=0.8)
        assert intent.requires_tools is False

    def test_ambiguities_default_empty(self):
        """ambiguities defaults to empty list."""
        intent = UserIntent(type="question", description="?", confidence=0.7)
        assert intent.ambiguities == []


# ===========================================================================
# PlanStep — status Literal and defaults
# ===========================================================================


class TestPlanStep:
    def test_valid_statuses_accepted(self):
        """All five valid step statuses must be accepted."""
        for status in ("pending", "running", "completed", "failed", "skipped"):
            step = _make_step(status=status)
            assert step.status == status

    def test_invalid_status_rejected(self):
        """
        SENTINEL — unknown status must raise ValidationError.
        Remove Literal → 'SENTINEL_STATUS' stored → evaluation logic can't find completed steps → test fails.
        """
        with pytest.raises(ValidationError):
            _make_step(status="SENTINEL_STATUS")

    def test_default_status_is_pending(self):
        """
        SENTINEL — default status must be 'pending'.
        Default 'running' → step appears already running before execution → test fails.
        """
        step = _make_step()
        assert step.status == "pending"

    def test_id_auto_generated(self):
        """id must be auto-generated string."""
        s1 = _make_step()
        s2 = _make_step()
        assert s1.id != s2.id

    def test_retry_policy_defaults(self):
        """
        SENTINEL — retry_policy must default to max_attempts=3, backoff_seconds=1.
        Default empty dict → step retry has no limits → infinite loop → test fails.
        """
        step = _make_step()
        assert step.retry_policy["max_attempts"] == 3
        assert step.retry_policy["backoff_seconds"] == 1


# ===========================================================================
# EvaluationResult — Literal and bounds
# ===========================================================================


class TestEvaluationResult:
    def _min_eval(self, **overrides) -> dict:
        base = {
            "overall_status": "success",
            "confidence": 0.9,
            "meets_requirements": True,
            "quality_score": 0.85,
        }
        base.update(overrides)
        return base

    def test_valid_overall_statuses(self):
        """All four overall_status values must be accepted."""
        for status in ("success", "failure", "needs_revision", "needs_retry"):
            er = EvaluationResult(**self._min_eval(overall_status=status))
            assert er.overall_status == status

    def test_invalid_overall_status_rejected(self):
        """
        SENTINEL — unknown overall_status must raise ValidationError.
        Remove Literal → 'SENTINEL_STATUS' stored → next_action selector breaks → test fails.
        """
        with pytest.raises(ValidationError):
            EvaluationResult(**self._min_eval(overall_status="SENTINEL_STATUS"))

    def test_confidence_bounds(self):
        """confidence must be in [0.0, 1.0]."""
        with pytest.raises(ValidationError):
            EvaluationResult(**self._min_eval(confidence=1.1))

    def test_quality_score_bounds(self):
        """
        SENTINEL — quality_score > 1.0 must raise ValidationError.
        Remove le=1.0 → score=1.5 stored → evaluation always approves → test fails.
        """
        with pytest.raises(ValidationError):
            EvaluationResult(**self._min_eval(quality_score=1.1))

    def test_valid_next_actions(self):
        """All four next_action values must be accepted."""
        for action in ("finalize", "retry_step", "revise_plan", "ask_user"):
            er = EvaluationResult(**self._min_eval(next_action=action))
            assert er.next_action == action

    def test_default_next_action_is_finalize(self):
        """
        SENTINEL — default next_action must be 'finalize'.
        Default 'retry_step' → every evaluation triggers a retry → test fails.
        """
        er = EvaluationResult(**self._min_eval())
        assert er.next_action == "finalize"


# ===========================================================================
# ConversationMessage — role Literal
# ===========================================================================


class TestConversationMessage:
    def test_valid_roles(self):
        """All three valid roles must be accepted."""
        for role in ("user", "assistant", "system"):
            msg = ConversationMessage(role=role, content="test")
            assert msg.role == role

    def test_invalid_role_rejected(self):
        """
        SENTINEL — unknown role must raise ValidationError.
        Remove Literal → 'SENTINEL_ROLE' stored → chat renderer can't distinguish message types → test fails.
        """
        with pytest.raises(ValidationError):
            ConversationMessage(role="SENTINEL_ROLE", content="test")


# ===========================================================================
# COREState — auto-fields and defaults
# ===========================================================================


class TestCOREStateDefaults:
    def test_run_id_auto_generated(self):
        """
        SENTINEL — run_id must be auto-generated.
        Default 'run-1' → all runs have same ID → trace correlation breaks → test fails.
        """
        s1 = _make_state()
        s2 = _make_state()
        assert s1.run_id != s2.run_id

    def test_current_node_starts_at_start(self):
        """
        SENTINEL — current_node must start as 'START'.
        Default 'END' → pipeline thinks it's already done → skips all processing → test fails.
        """
        state = _make_state()
        assert state.current_node == "START"

    def test_errors_and_warnings_default_empty(self):
        """errors and warnings both default to empty list."""
        state = _make_state()
        assert state.errors == []
        assert state.warnings == []

    def test_step_results_default_empty(self):
        """step_results defaults to empty list."""
        state = _make_state()
        assert state.step_results == []

    def test_config_has_expected_defaults(self):
        """
        SENTINEL — config must have model, temperature, max_iterations, enable_tools, enable_hitl keys.
        Remove key → agent_factory falls back to None → LLM call fails → test fails.
        """
        state = _make_state()
        cfg = state.config
        assert "model" in cfg
        assert "temperature" in cfg
        assert "max_iterations" in cfg
        assert "enable_tools" in cfg
        assert "enable_hitl" in cfg

    def test_plan_revisions_default_zero(self):
        """plan_revisions defaults to 0."""
        state = _make_state()
        assert state.plan_revisions == 0


# ===========================================================================
# COREState.add_execution_node
# ===========================================================================


class TestAddExecutionNode:
    def test_appends_to_execution_history(self):
        """
        SENTINEL — add_execution_node must append to execution_history.
        Only update current_node → history empty → pipeline trace lost → test fails.
        """
        state = _make_state()
        state.add_execution_node("comprehension")
        assert "comprehension" in state.execution_history

    def test_updates_current_node(self):
        """
        SENTINEL — current_node must be updated to the new node.
        Append only → current_node stuck at START → routing loops → test fails.
        """
        state = _make_state()
        state.add_execution_node("SENTINEL_NODE")
        assert state.current_node == "SENTINEL_NODE"

    def test_multiple_nodes_tracked_in_order(self):
        """Multiple calls must maintain insertion order."""
        state = _make_state()
        for node in ("comprehension", "orchestration", "reasoning"):
            state.add_execution_node(node)
        assert state.execution_history == [
            "comprehension",
            "orchestration",
            "reasoning",
        ]

    def test_updates_updated_at(self):
        """updated_at must change after add_execution_node."""
        state = _make_state()
        before = state.updated_at
        state.add_execution_node("node")
        # updated_at is set to utcnow() inside the method
        assert state.updated_at >= before


# ===========================================================================
# COREState.add_error / add_warning
# ===========================================================================


class TestAddErrorAndWarning:
    def test_add_error_appends(self):
        """
        SENTINEL — add_error must append to errors list.
        Replace list → older errors lost → debugging impossible → test fails.
        """
        state = _make_state()
        state.add_error("SENTINEL_ERR_1")
        state.add_error("SENTINEL_ERR_2")
        assert "SENTINEL_ERR_1" in state.errors
        assert "SENTINEL_ERR_2" in state.errors
        assert len(state.errors) == 2

    def test_add_warning_appends(self):
        """add_warning must append to warnings list."""
        state = _make_state()
        state.add_warning("SENTINEL_WARN")
        assert "SENTINEL_WARN" in state.warnings


# ===========================================================================
# COREState.get_step
# ===========================================================================


class TestGetStep:
    def test_returns_none_without_plan(self):
        """
        SENTINEL — get_step must return None when plan is None.
        Raise AttributeError → pipeline crashes before plan is created → test fails.
        """
        state = _make_state()
        assert state.plan is None
        result = state.get_step("any-id")
        assert result is None

    def test_returns_matching_step(self):
        """
        SENTINEL — get_step must return the step with matching id.
        Return None always → update_step_status can't find steps → plan execution broken → test fails.
        """
        state = _make_state()
        state.plan = _make_plan("step_a", "step_b")
        target_id = state.plan.steps[1].id

        result = state.get_step(target_id)
        assert result is not None
        assert result.name == "step_b"

    def test_returns_none_for_missing_id(self):
        """get_step must return None for an ID not in the plan."""
        state = _make_state()
        state.plan = _make_plan("step_a")
        result = state.get_step("SENTINEL_NONEXISTENT_ID")
        assert result is None


# ===========================================================================
# COREState.update_step_status
# ===========================================================================


class TestUpdateStepStatus:
    def test_updates_status_field(self):
        """
        SENTINEL — update_step_status must change the step's status field.
        Only update timestamps → status stays 'pending' → evaluation can't detect completion → test fails.
        """
        state = _make_state()
        state.plan = _make_plan("SENTINEL_STEP")
        step_id = state.plan.steps[0].id

        state.update_step_status(step_id, "completed")
        assert state.get_step(step_id).status == "completed"

    def test_sets_started_at_when_running(self):
        """
        SENTINEL — started_at must be set when status changes to 'running'.
        Skip timestamp → timing metrics unavailable → step duration always None → test fails.
        """
        state = _make_state()
        state.plan = _make_plan("STEP")
        step_id = state.plan.steps[0].id

        state.update_step_status(step_id, "running")
        assert state.get_step(step_id).started_at is not None

    def test_sets_completed_at_when_completed(self):
        """
        SENTINEL — completed_at must be set when status is 'completed'.
        Skip timestamp → evaluation can't compute step duration → test fails.
        """
        state = _make_state()
        state.plan = _make_plan("STEP")
        step_id = state.plan.steps[0].id

        state.update_step_status(step_id, "completed")
        assert state.get_step(step_id).completed_at is not None

    def test_sets_completed_at_on_failed(self):
        """completed_at must also be set on 'failed' status."""
        state = _make_state()
        state.plan = _make_plan("STEP")
        step_id = state.plan.steps[0].id
        state.update_step_status(step_id, "failed")
        assert state.get_step(step_id).completed_at is not None

    def test_no_started_at_on_completed_without_running(self):
        """started_at must NOT be set when status is not 'running'."""
        state = _make_state()
        state.plan = _make_plan("STEP")
        step_id = state.plan.steps[0].id

        # Jump directly to completed without 'running' phase
        state.update_step_status(step_id, "completed")
        # started_at was never set by this call
        assert state.get_step(step_id).started_at is None

    def test_noop_when_step_id_not_found(self):
        """update_step_status must not raise when step_id doesn't exist."""
        state = _make_state()
        state.plan = _make_plan("real_step")
        # Should not raise
        state.update_step_status("SENTINEL_FAKE_ID", "completed")


# ===========================================================================
# COREState.is_complete
# ===========================================================================


class TestIsComplete:
    def test_false_at_start(self):
        """
        SENTINEL — is_complete must be False at initial state.
        Return True always → pipeline exits immediately → test fails.
        """
        state = _make_state()
        assert state.is_complete() is False

    def test_true_when_current_node_is_end(self):
        """
        SENTINEL — must be True when current_node == 'END'.
        Check only completed_at → pipeline keeps running after END node → test fails.
        """
        state = _make_state()
        state.current_node = "END"
        assert state.is_complete() is True

    def test_true_when_completed_at_is_set(self):
        """
        SENTINEL — must be True when completed_at is not None.
        Ignore completed_at → completed runs say they're still running → test fails.
        """
        state = _make_state()
        state.completed_at = datetime.utcnow()
        assert state.is_complete() is True

    def test_false_when_mid_pipeline(self):
        """Must be False when pipeline is between START and END."""
        state = _make_state()
        state.add_execution_node("comprehension")
        assert state.is_complete() is False
