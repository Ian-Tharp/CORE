"""
Tests for COREState methods and Pydantic model validators.

Sentinel-value contract:
- Remove execution_history.append → visited nodes not tracked → test fails
- Remove current_node update → node always reports 'START' → test fails
- Remove get_step plan-None guard → AttributeError on missing plan → test fails
- Remove update_step_status started_at stamp → timing data lost → test fails
- Remove validate_trait_values bounds check → out-of-range traits accepted → test fails
- Remove content_must_not_be_empty strip → whitespace-only content accepted → test fails
"""

from __future__ import annotations

import pytest
from datetime import datetime
from typing import Optional

from pydantic import ValidationError

from app.models.core_state import (
    COREState,
    UserIntent,
    ExecutionPlan,
    PlanStep,
    StepResult,
    EvaluationResult,
)
from app.models.agent_models import PersonalityTraits
from app.models.comprehension_models import ComprehensionInput, SourceType


# ===========================================================================
# Helpers
# ===========================================================================


def _make_state(**kwargs) -> COREState:
    defaults = {"user_input": "test input"}
    defaults.update(kwargs)
    return COREState(**defaults)


def _make_plan(*step_names: str) -> ExecutionPlan:
    steps = [
        PlanStep(id=f"step-{i}", name=n, description=f"do {n}")
        for i, n in enumerate(step_names)
    ]
    return ExecutionPlan(goal="test goal", steps=steps)


# ===========================================================================
# COREState.add_execution_node
# ===========================================================================


class TestAddExecutionNode:
    def test_appends_to_execution_history(self):
        """
        SENTINEL — visited node must be appended to execution_history.
        Skip append → graph can't reconstruct visited nodes → test fails.
        """
        state = _make_state()
        state.add_execution_node("comprehension")
        assert "comprehension" in state.execution_history

    def test_updates_current_node(self):
        """
        SENTINEL — current_node must be updated to the new node.
        Skip update → current_node always 'START' → routing logic confused → test fails.
        """
        state = _make_state()
        state.add_execution_node("SENTINEL_NODE")
        assert state.current_node == "SENTINEL_NODE"

    def test_multiple_nodes_tracked_in_order(self):
        """Multiple calls build an ordered history."""
        state = _make_state()
        state.add_execution_node("comprehension")
        state.add_execution_node("orchestration")
        state.add_execution_node("reasoning")
        assert state.execution_history == [
            "comprehension",
            "orchestration",
            "reasoning",
        ]

    def test_updates_updated_at(self):
        """updated_at must be refreshed on each call."""
        state = _make_state()
        before = state.updated_at
        state.add_execution_node("evaluation")
        assert state.updated_at >= before


# ===========================================================================
# COREState.add_error / add_warning
# ===========================================================================


class TestAddError:
    def test_appends_to_errors(self):
        """
        SENTINEL — error message must be appended to errors list.
        Skip append → errors silently lost → debugging impossible → test fails.
        """
        state = _make_state()
        state.add_error("SENTINEL_ERROR")
        assert "SENTINEL_ERROR" in state.errors

    def test_multiple_errors_accumulate(self):
        """Multiple errors must all be preserved."""
        state = _make_state()
        state.add_error("err1")
        state.add_error("err2")
        assert "err1" in state.errors
        assert "err2" in state.errors
        assert len(state.errors) == 2


class TestAddWarning:
    def test_appends_to_warnings(self):
        """Warning message must appear in warnings list."""
        state = _make_state()
        state.add_warning("SENTINEL_WARNING")
        assert "SENTINEL_WARNING" in state.warnings

    def test_errors_and_warnings_are_separate(self):
        """Errors go to errors, warnings go to warnings (not mixed)."""
        state = _make_state()
        state.add_error("an error")
        state.add_warning("a warning")
        assert "a warning" not in state.errors
        assert "an error" not in state.warnings


# ===========================================================================
# COREState.get_step
# ===========================================================================


class TestGetStep:
    def test_returns_none_when_no_plan(self):
        """
        SENTINEL — must return None when state.plan is None (no crash).
        Remove None guard → AttributeError on plan.steps → test fails.
        """
        state = _make_state()
        result = state.get_step("any-id")
        assert result is None

    def test_returns_matching_step(self):
        """
        SENTINEL — must return the step with the matching ID.
        Use linear scan wrong → returns wrong step → execution targets wrong step → test fails.
        """
        plan = _make_plan("step-alpha")
        step_id = plan.steps[0].id
        state = _make_state(plan=plan)
        result = state.get_step(step_id)
        assert result is not None
        assert result.name == "step-alpha"

    def test_returns_none_for_unknown_step_id(self):
        """Unknown step ID must return None without error."""
        state = _make_state(plan=_make_plan("step-a"))
        result = state.get_step("SENTINEL_NONEXISTENT_ID")
        assert result is None

    def test_finds_second_step_in_multi_step_plan(self):
        """get_step must search all steps, not just the first."""
        plan = _make_plan("step-1", "step-2", "step-3")
        last_id = plan.steps[-1].id
        state = _make_state(plan=plan)
        result = state.get_step(last_id)
        assert result is not None
        assert result.name == "step-3"


# ===========================================================================
# COREState.update_step_status
# ===========================================================================


class TestUpdateStepStatus:
    def test_updates_step_status(self):
        """
        SENTINEL — step.status must be changed to the new value.
        Skip assignment → step always 'pending' → evaluation thinks nothing ran → test fails.
        """
        plan = _make_plan("SENTINEL_STEP")
        step_id = plan.steps[0].id
        state = _make_state(plan=plan)
        state.update_step_status(step_id, "running")
        step = state.get_step(step_id)
        assert step.status == "running"

    def test_running_sets_started_at(self):
        """
        SENTINEL — 'running' status must set step.started_at timestamp.
        Skip stamp → no way to measure step latency → test fails.
        """
        plan = _make_plan("step-a")
        step_id = plan.steps[0].id
        state = _make_state(plan=plan)
        assert state.get_step(step_id).started_at is None
        state.update_step_status(step_id, "running")
        assert state.get_step(step_id).started_at is not None

    def test_completed_sets_completed_at(self):
        """
        SENTINEL — 'completed' must set step.completed_at.
        Skip stamp → duration calculation yields None → test fails.
        """
        plan = _make_plan("step-a")
        step_id = plan.steps[0].id
        state = _make_state(plan=plan)
        state.update_step_status(step_id, "completed")
        assert state.get_step(step_id).completed_at is not None

    def test_failed_sets_completed_at(self):
        """'failed' status must also set completed_at (end time)."""
        plan = _make_plan("step-a")
        step_id = plan.steps[0].id
        state = _make_state(plan=plan)
        state.update_step_status(step_id, "failed")
        assert state.get_step(step_id).completed_at is not None

    def test_skipped_sets_completed_at(self):
        """'skipped' status must set completed_at."""
        plan = _make_plan("step-a")
        step_id = plan.steps[0].id
        state = _make_state(plan=plan)
        state.update_step_status(step_id, "skipped")
        assert state.get_step(step_id).completed_at is not None

    def test_pending_does_not_set_timestamps(self):
        """'pending' is the initial state and must not set started_at/completed_at."""
        plan = _make_plan("step-a")
        step_id = plan.steps[0].id
        state = _make_state(plan=plan)
        # 'running' then back to 'pending' scenario
        state.update_step_status(step_id, "pending")
        step = state.get_step(step_id)
        assert step.started_at is None

    def test_no_op_when_step_not_found(self):
        """Updating a non-existent step must not raise."""
        state = _make_state(plan=_make_plan("real-step"))
        state.update_step_status("SENTINEL_NONEXISTENT", "completed")  # must not raise


# ===========================================================================
# COREState.is_complete
# ===========================================================================


class TestIsComplete:
    def test_false_at_start(self):
        """Initial state (current_node='START', no completed_at) must be incomplete."""
        state = _make_state()
        assert state.is_complete() is False

    def test_true_when_current_node_is_end(self):
        """
        SENTINEL — current_node == 'END' must make is_complete() return True.
        Use '!= END' check → pipeline never finishes → test fails.
        """
        state = _make_state()
        state.current_node = "END"
        assert state.is_complete() is True

    def test_true_when_completed_at_set(self):
        """
        SENTINEL — a non-None completed_at must make is_complete() return True.
        Ignore completed_at → completed jobs re-enter the pipeline → test fails.
        """
        state = _make_state()
        state.completed_at = datetime.utcnow()
        assert state.is_complete() is True

    def test_false_when_in_reasoning_node(self):
        """Mid-pipeline node must not be complete."""
        state = _make_state()
        state.current_node = "reasoning"
        assert state.is_complete() is False


# ===========================================================================
# PersonalityTraits.validate_trait_values
# ===========================================================================


class TestPersonalityTraitsValidator:
    def test_valid_traits_accepted(self):
        """Trait values in [0.0, 1.0] must be accepted without error."""
        traits = PersonalityTraits(
            traits={"curiosity": 0.9, "precision": 0.5, "creativity": 0.0}
        )
        assert traits.traits["curiosity"] == pytest.approx(0.9)

    def test_above_1_rejected(self):
        """
        SENTINEL — trait value > 1.0 must raise ValidationError.
        Remove bounds check → traits like 1.5 silently stored → behaviour undefined → test fails.
        """
        with pytest.raises(ValidationError, match="1.0"):
            PersonalityTraits(traits={"bad_trait": 1.5})

    def test_below_0_rejected(self):
        """Trait value < 0.0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            PersonalityTraits(traits={"bad_trait": -0.1})

    def test_boundary_zero_accepted(self):
        """0.0 is a valid trait value (lower boundary)."""
        traits = PersonalityTraits(traits={"muted": 0.0})
        assert traits.traits["muted"] == pytest.approx(0.0)

    def test_boundary_one_accepted(self):
        """1.0 is a valid trait value (upper boundary)."""
        traits = PersonalityTraits(traits={"max_trait": 1.0})
        assert traits.traits["max_trait"] == pytest.approx(1.0)

    def test_empty_traits_accepted(self):
        """Empty traits dict is valid (no constraints to violate)."""
        traits = PersonalityTraits(traits={})
        assert traits.traits == {}


# ===========================================================================
# ComprehensionInput.content_must_not_be_empty
# ===========================================================================


class TestComprehensionInputValidator:
    def test_valid_content_accepted(self):
        """Non-empty content must be accepted and stripped."""
        inp = ComprehensionInput(
            content="  Hello world  ",
            source_type=SourceType.USER,
        )
        assert inp.content == "Hello world"

    def test_empty_string_rejected(self):
        """
        SENTINEL — empty string must raise ValidationError.
        Remove empty check → LLM receives empty prompt → test fails.
        """
        with pytest.raises(ValidationError, match="empty"):
            ComprehensionInput(content="", source_type=SourceType.USER)

    def test_whitespace_only_rejected(self):
        """
        SENTINEL — whitespace-only string must raise ValidationError.
        Skip strip check → whitespace-only content accepted → empty prompt sent → test fails.
        """
        with pytest.raises(ValidationError, match="empty"):
            ComprehensionInput(content="   \t\n  ", source_type=SourceType.USER)

    def test_content_stripped_on_accept(self):
        """Leading/trailing whitespace must be stripped from accepted content."""
        inp = ComprehensionInput(
            content="\t  SENTINEL_CONTENT  \n", source_type=SourceType.AGENT
        )
        assert inp.content == "SENTINEL_CONTENT"
