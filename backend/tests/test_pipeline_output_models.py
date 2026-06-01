"""
Tests for the small pipeline output models:
  - reasoning_output.py  (ReasoningTaskStatus, ReasoningOutput)
  - comprehension_output.py (ComprehensionOutput)
  - evaluation_output.py (EvaluationOutput)
  - orchestration_output.py (OrchestrationOutput)
  - user_input.py (UserInput)

Sentinel-value contract:
- Remove ReasoningTaskStatus.ACTIVE → agent can't mark task active → test fails
- Remove ComprehensionOutput.WITHIN_EXPERTISE → comprehension can't classify in-scope → test fails
- Remove EvaluationOutput.SATISFACTORY → evaluation can't approve → test fails
- Remove UserInput.model optional → UserInput requires model field → breaking change → test fails
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.reasoning_output import ReasoningOutput, ReasoningTaskStatus
from app.models.comprehension_output import ComprehensionOutput
from app.models.evaluation_output import EvaluationOutput
from app.models.orchestration_output import OrchestrationOutput
from app.models.user_input import UserInput


# ===========================================================================
# ReasoningTaskStatus enum
# ===========================================================================


class TestReasoningTaskStatus:
    def test_active_defined(self):
        """
        SENTINEL — ACTIVE must be defined.
        Remove ACTIVE → reasoning agent can't mark steps running → test fails.
        """
        assert hasattr(ReasoningTaskStatus, "ACTIVE")

    def test_created_defined(self):
        """CREATED must be defined."""
        assert hasattr(ReasoningTaskStatus, "CREATED")

    def test_error_defined(self):
        """ERROR must be defined."""
        assert hasattr(ReasoningTaskStatus, "ERROR")

    def test_pending_defined(self):
        """PENDING must be defined."""
        assert hasattr(ReasoningTaskStatus, "PENDING")

    def test_awaiting_approval_defined(self):
        """AWAITING_APPROVAL must be defined."""
        assert hasattr(ReasoningTaskStatus, "AWAITING_APPROVAL")

    def test_awaiting_human_intervention_defined(self):
        """AWAITING_HUMAN_INTERVENTION must be defined."""
        assert hasattr(ReasoningTaskStatus, "AWAITING_HUMAN_INTERVENTION")

    def test_all_six_statuses_defined(self):
        """Exactly 6 statuses must be defined."""
        assert len(ReasoningTaskStatus) == 6

    def test_error_has_string_value(self):
        """
        SENTINEL — ERROR must equal 'Error' (string, not tuple).
        Change to tuple → status serialisation broken → test fails.
        """
        assert ReasoningTaskStatus.ERROR.value == "Error"


# ===========================================================================
# ReasoningOutput model
# ===========================================================================


class TestReasoningOutput:
    def test_valid_reasoning_output(self):
        """Valid task_id and task_status must be accepted."""
        ro = ReasoningOutput(
            task_id="SENTINEL_TASK_ID",
            task_status=ReasoningTaskStatus.ACTIVE,
        )
        assert ro.task_id == "SENTINEL_TASK_ID"
        assert ro.task_status == ReasoningTaskStatus.ACTIVE

    def test_task_id_required(self):
        """task_id is required — omitting must raise ValidationError."""
        with pytest.raises(ValidationError):
            ReasoningOutput(task_status=ReasoningTaskStatus.ACTIVE)


# ===========================================================================
# ComprehensionOutput enum
# ===========================================================================


class TestComprehensionOutput:
    def test_within_expertise_value(self):
        """
        SENTINEL — WITHIN_EXPERTISE must equal 'Within Expertise'.
        Change string → routing check fails → in-scope tasks rejected → test fails.
        """
        assert ComprehensionOutput.WITHIN_EXPERTISE.value == "Within Expertise"

    def test_outside_expertise_value(self):
        """
        SENTINEL — OUTSIDE_EXPERTISE must equal 'Outside Expertise'.
        Change string → out-of-scope tasks incorrectly routed → test fails.
        """
        assert ComprehensionOutput.OUTSIDE_EXPERTISE.value == "Outside Expertise"

    def test_exactly_two_values(self):
        """Exactly two ComprehensionOutput values must be defined."""
        assert len(ComprehensionOutput) == 2


# ===========================================================================
# EvaluationOutput enum
# ===========================================================================


class TestEvaluationOutput:
    def test_satisfactory_value(self):
        """
        SENTINEL — SATISFACTORY must equal 'Satisfactory'.
        Change string → evaluation approval check fails → test fails.
        """
        assert EvaluationOutput.SATISFACTORY.value == "Satisfactory"

    def test_unsatisfactory_value(self):
        """UNSATISFACTORY must equal 'Unsatisfactory'."""
        assert EvaluationOutput.UNSATISFACTORY.value == "Unsatisfactory"

    def test_exactly_two_values(self):
        """Exactly two EvaluationOutput values must be defined."""
        assert len(EvaluationOutput) == 2


# ===========================================================================
# OrchestrationOutput model
# ===========================================================================


class TestOrchestrationOutput:
    def test_valid_orchestration_output(self):
        """Valid model with both required fields must be accepted."""
        oo = OrchestrationOutput(
            orchestration_id="SENTINEL_ORCH_ID",
            overall_plan="SENTINEL_OVERALL_PLAN",
            task_plan="SENTINEL_TASK_PLAN",
        )
        assert oo.orchestration_id == "SENTINEL_ORCH_ID"
        assert oo.overall_plan == "SENTINEL_OVERALL_PLAN"

    def test_orchestration_id_required(self):
        """orchestration_id is required."""
        with pytest.raises(ValidationError):
            OrchestrationOutput(overall_plan="plan", task_plan="plan")


# ===========================================================================
# UserInput model
# ===========================================================================


class TestUserInput:
    def test_minimal_user_input_accepted(self):
        """
        SENTINEL — UserInput with just message_id and user_input must be accepted.
        Make model required → UI calls without model choice fail → breaking change → test fails.
        """
        ui = UserInput(message_id="SENTINEL_MSG_ID", user_input="SENTINEL_INPUT")
        assert ui.message_id == "SENTINEL_MSG_ID"
        assert ui.user_input == "SENTINEL_INPUT"

    def test_model_optional_defaults_none(self):
        """
        SENTINEL — model must default to None (optional field).
        Make required → all callers without model choice break → test fails.
        """
        ui = UserInput(message_id="m", user_input="test")
        assert ui.model is None

    def test_comprehension_text_optional(self):
        """comprehension_text defaults to None."""
        ui = UserInput(message_id="m", user_input="test")
        assert ui.comprehension_text is None

    def test_orchestration_plan_optional(self):
        """orchestration_plan defaults to None."""
        ui = UserInput(message_id="m", user_input="test")
        assert ui.orchestration_plan is None

    def test_prior_context_stored_when_provided(self):
        """Prior-step context fields must be stored when provided."""
        ui = UserInput(
            message_id="m",
            user_input="test",
            comprehension_text="SENTINEL_COMP",
            orchestration_plan=["step1", "step2"],
            reasoning_text="SENTINEL_REASON",
        )
        assert ui.comprehension_text == "SENTINEL_COMP"
        assert ui.orchestration_plan == ["step1", "step2"]
        assert ui.reasoning_text == "SENTINEL_REASON"
