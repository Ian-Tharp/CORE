"""
Tests for EvaluationAgent LLM-based evaluation.

Sentinel-value contract: every LLM path test would FAIL if you removed
the _llm_evaluate call from evaluate_execution or replaced it with
rule-based-only logic.

Coverage:
- evaluate_execution() calls LLM (not rule-based as primary)
- user_input flows into LLM prompt
- intent context flows into LLM prompt
- step results flow into LLM prompt
- LLM response fields (confidence, next_action, feedback) flow into EvaluationResult
- Invalid LLM fields are sanitised to valid values
- LLM failure triggers _rule_based_evaluate fallback (no crash)
- _rule_based_evaluate: all-success → finalize, all-fail → revise_plan, partial → retry_step
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from app.core.agents.evaluation_agent import EvaluationAgent
from app.models.core_state import (
    UserIntent,
    ExecutionPlan,
    PlanStep,
    StepResult,
    EvaluationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(name: str = "Step", desc: str = "Do something") -> PlanStep:
    return PlanStep(name=name, description=desc)


def _make_plan(*steps: PlanStep, goal: str = "Test goal") -> ExecutionPlan:
    return ExecutionPlan(goal=goal, steps=list(steps))


def _make_result(
    status: str = "success", step_id: str = None, outputs: dict = None
) -> StepResult:
    r = StepResult(
        step_id=step_id or "step-1",
        status=status,
        outputs=outputs or {"result": "output text"},
        error="boom" if status == "failure" else None,
    )
    return r


def _make_intent(type_: str = "task", desc: str = "Build something") -> UserIntent:
    return UserIntent(type=type_, description=desc, confidence=0.9)


def _llm_response(data: dict) -> MagicMock:
    """Build a mock OpenAI completion response from a dict."""
    msg = MagicMock()
    msg.content = json.dumps(data)
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _valid_eval_json(**overrides) -> dict:
    base = {
        "overall_status": "success",
        "confidence": 0.85,
        "meets_requirements": True,
        "quality_score": 0.9,
        "feedback": "Task completed as requested.",
        "next_action": "finalize",
        "retry_step_id": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# evaluate_execution calls LLM (primary path)
# ---------------------------------------------------------------------------


class TestEvaluateExecutionCallsLLM:
    def setup_method(self):
        self.agent = EvaluationAgent()

    def test_evaluate_execution_calls_llm_client(self):
        """
        SENTINEL TEST — evaluate_execution must call the LLM client.
        Replace _llm_evaluate with rule-based → LLM not called → test fails.
        """
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _llm_response(
            _valid_eval_json()
        )

        plan = _make_plan(_make_step())
        results = [_make_result()]

        with patch(
            "app.core.agents.evaluation_agent.get_openai_client_sync",
            return_value=mock_client,
        ):
            self.agent.evaluate_execution(
                "Build a login page", _make_intent(), plan, results
            )

        mock_client.chat.completions.create.assert_called_once()

    def test_user_input_in_llm_prompt(self):
        """
        SENTINEL TEST — user_input must appear in the user message sent to LLM.
        Remove the user_input line from context → this fails.
        """
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _llm_response(
            _valid_eval_json()
        )

        plan = _make_plan(_make_step())
        results = [_make_result()]

        with patch(
            "app.core.agents.evaluation_agent.get_openai_client_sync",
            return_value=mock_client,
        ):
            self.agent.evaluate_execution(
                "SENTINEL_USER_REQUEST", _make_intent(), plan, results
            )

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "SENTINEL_USER_REQUEST" in user_msg["content"]

    def test_intent_context_in_llm_prompt(self):
        """
        SENTINEL TEST — intent type and description must appear in the LLM context.
        Remove intent_line from context → intent absent → test fails.
        """
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _llm_response(
            _valid_eval_json()
        )

        plan = _make_plan(_make_step())
        results = [_make_result()]
        intent = _make_intent(type_="task", desc="SENTINEL_INTENT_DESC")

        with patch(
            "app.core.agents.evaluation_agent.get_openai_client_sync",
            return_value=mock_client,
        ):
            self.agent.evaluate_execution("query", intent, plan, results)

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "SENTINEL_INTENT_DESC" in user_msg["content"]
        assert "task" in user_msg["content"]

    def test_plan_goal_in_llm_prompt(self):
        """Plan goal must appear in the prompt."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _llm_response(
            _valid_eval_json()
        )

        plan = _make_plan(_make_step(), goal="SENTINEL_PLAN_GOAL")
        results = [_make_result()]

        with patch(
            "app.core.agents.evaluation_agent.get_openai_client_sync",
            return_value=mock_client,
        ):
            self.agent.evaluate_execution("query", _make_intent(), plan, results)

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "SENTINEL_PLAN_GOAL" in user_msg["content"]

    def test_step_output_in_llm_prompt(self):
        """
        SENTINEL TEST — step result outputs must appear in the user prompt.
        Remove step_lines from context → output absent → test fails.
        """
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _llm_response(
            _valid_eval_json()
        )

        plan = _make_plan(_make_step())
        results = [_make_result(outputs={"result": "SENTINEL_STEP_OUTPUT"})]

        with patch(
            "app.core.agents.evaluation_agent.get_openai_client_sync",
            return_value=mock_client,
        ):
            self.agent.evaluate_execution("query", _make_intent(), plan, results)

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "SENTINEL_STEP_OUTPUT" in user_msg["content"]

    def test_system_prompt_sent_to_llm(self):
        """The system prompt with evaluation instructions must be in the messages."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _llm_response(
            _valid_eval_json()
        )

        plan = _make_plan(_make_step())

        with patch(
            "app.core.agents.evaluation_agent.get_openai_client_sync",
            return_value=mock_client,
        ):
            self.agent.evaluate_execution(
                "query", _make_intent(), plan, [_make_result()]
            )

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        sys_msg = next(m for m in messages if m["role"] == "system")
        # System prompt instructs on evaluation format
        assert "next_action" in sys_msg["content"]
        assert "finalize" in sys_msg["content"]

    def test_low_temperature_used(self):
        """Temperature must be low (<=0.3) for consistent evaluation."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _llm_response(
            _valid_eval_json()
        )

        plan = _make_plan(_make_step())

        with patch(
            "app.core.agents.evaluation_agent.get_openai_client_sync",
            return_value=mock_client,
        ):
            self.agent.evaluate_execution(
                "query", _make_intent(), plan, [_make_result()]
            )

        kwargs = mock_client.chat.completions.create.call_args[1]
        assert kwargs.get("temperature", 1.0) <= 0.3


# ---------------------------------------------------------------------------
# LLM response flows into EvaluationResult
# ---------------------------------------------------------------------------


class TestLLMResponseFlowsIntoResult:
    def setup_method(self):
        self.agent = EvaluationAgent()

    def test_confidence_from_llm_response(self):
        """
        SENTINEL TEST — confidence value from LLM JSON must appear in result.
        Hardcode confidence=0.5 in result → value doesn't match LLM → fails.
        """
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _llm_response(
            _valid_eval_json(confidence=0.77)
        )

        plan = _make_plan(_make_step())

        with patch(
            "app.core.agents.evaluation_agent.get_openai_client_sync",
            return_value=mock_client,
        ):
            result = self.agent.evaluate_execution(
                "query", _make_intent(), plan, [_make_result()]
            )

        assert abs(result.confidence - 0.77) < 0.001

    def test_feedback_from_llm_response(self):
        """
        SENTINEL TEST — feedback string from LLM must appear in result.
        """
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _llm_response(
            _valid_eval_json(feedback="SENTINEL_LLM_FEEDBACK")
        )

        plan = _make_plan(_make_step())

        with patch(
            "app.core.agents.evaluation_agent.get_openai_client_sync",
            return_value=mock_client,
        ):
            result = self.agent.evaluate_execution(
                "query", _make_intent(), plan, [_make_result()]
            )

        assert "SENTINEL_LLM_FEEDBACK" in result.feedback

    def test_next_action_from_llm_response(self):
        """
        SENTINEL TEST — next_action from LLM must drive the routing decision.
        """
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _llm_response(
            _valid_eval_json(
                next_action="revise_plan",
                overall_status="needs_revision",
            )
        )

        plan = _make_plan(_make_step())

        with patch(
            "app.core.agents.evaluation_agent.get_openai_client_sync",
            return_value=mock_client,
        ):
            result = self.agent.evaluate_execution(
                "query", _make_intent(), plan, [_make_result()]
            )

        assert result.next_action == "revise_plan"

    def test_quality_score_from_llm_response(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _llm_response(
            _valid_eval_json(quality_score=0.42)
        )

        plan = _make_plan(_make_step())

        with patch(
            "app.core.agents.evaluation_agent.get_openai_client_sync",
            return_value=mock_client,
        ):
            result = self.agent.evaluate_execution(
                "q", _make_intent(), plan, [_make_result()]
            )

        assert abs(result.quality_score - 0.42) < 0.001

    def test_retry_step_id_from_llm_response(self):
        """retry_step_id from LLM flows into the result."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _llm_response(
            _valid_eval_json(
                next_action="retry_step",
                overall_status="needs_retry",
                retry_step_id="SENTINEL_STEP_ID",
            )
        )

        plan = _make_plan(_make_step())

        with patch(
            "app.core.agents.evaluation_agent.get_openai_client_sync",
            return_value=mock_client,
        ):
            result = self.agent.evaluate_execution(
                "q", _make_intent(), plan, [_make_result()]
            )

        assert result.retry_step_id == "SENTINEL_STEP_ID"

    def test_meets_requirements_from_llm_response(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _llm_response(
            _valid_eval_json(meets_requirements=False, overall_status="failure")
        )

        plan = _make_plan(_make_step())

        with patch(
            "app.core.agents.evaluation_agent.get_openai_client_sync",
            return_value=mock_client,
        ):
            result = self.agent.evaluate_execution(
                "q", _make_intent(), plan, [_make_result()]
            )

        assert result.meets_requirements is False


# ---------------------------------------------------------------------------
# Invalid LLM fields — validation / sanitisation
# ---------------------------------------------------------------------------


class TestLLMFieldValidation:
    def setup_method(self):
        self.agent = EvaluationAgent()

    def test_invalid_next_action_defaults_to_finalize(self):
        """
        SENTINEL TEST — garbage next_action from LLM must be sanitised to 'finalize'.
        Remove the validation block → invalid value passes through → test fails.
        """
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _llm_response(
            _valid_eval_json(next_action="GARBAGE_ACTION")
        )

        plan = _make_plan(_make_step())

        with patch(
            "app.core.agents.evaluation_agent.get_openai_client_sync",
            return_value=mock_client,
        ):
            result = self.agent.evaluate_execution("q", None, plan, [_make_result()])

        assert result.next_action == "finalize"

    def test_invalid_overall_status_defaults_to_success(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _llm_response(
            _valid_eval_json(overall_status="GARBAGE_STATUS")
        )

        plan = _make_plan(_make_step())

        with patch(
            "app.core.agents.evaluation_agent.get_openai_client_sync",
            return_value=mock_client,
        ):
            result = self.agent.evaluate_execution("q", None, plan, [_make_result()])

        assert result.overall_status in {
            "success",
            "failure",
            "needs_revision",
            "needs_retry",
        }


# ---------------------------------------------------------------------------
# Fallback to rule-based when LLM fails
# ---------------------------------------------------------------------------


class TestRuleBasedFallback:
    def setup_method(self):
        self.agent = EvaluationAgent()

    def test_llm_exception_triggers_rule_based_fallback(self):
        """
        SENTINEL TEST — when LLM raises, evaluation must NOT crash.
        _rule_based_evaluate is called as fallback.
        Remove fallback → exception propagates to caller → test fails.
        """
        plan = _make_plan(_make_step())
        results = [_make_result("success")]

        with patch(
            "app.core.agents.evaluation_agent.get_openai_client_sync",
            side_effect=RuntimeError("LLM unavailable"),
        ):
            result = self.agent.evaluate_execution(
                "query", _make_intent(), plan, results
            )

        # Fallback for all-success → finalize
        assert result.next_action == "finalize"
        assert isinstance(result, EvaluationResult)

    def test_llm_bad_json_triggers_fallback(self):
        """If LLM returns unparseable JSON, rule-based fallback fires."""
        mock_client = MagicMock()
        bad_resp = MagicMock()
        bad_resp.choices[0].message.content = "This is not JSON at all!"
        mock_client.chat.completions.create.return_value = bad_resp

        plan = _make_plan(_make_step())
        results = [_make_result("failure")]

        with patch(
            "app.core.agents.evaluation_agent.get_openai_client_sync",
            return_value=mock_client,
        ):
            result = self.agent.evaluate_execution("query", None, plan, results)

        # All steps failed → rule-based → revise_plan
        assert result.next_action == "revise_plan"

    def test_llm_empty_response_triggers_fallback(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _llm_response_empty()

        plan = _make_plan(_make_step())

        with patch(
            "app.core.agents.evaluation_agent.get_openai_client_sync",
            return_value=mock_client,
        ):
            result = self.agent.evaluate_execution(
                "query", None, plan, [_make_result("success")]
            )

        assert isinstance(result, EvaluationResult)

    def test_rule_based_all_success_returns_finalize(self):
        """_rule_based_evaluate: all steps succeed → finalize, confidence=0.9."""
        results = [_make_result("success"), _make_result("success")]
        result = self.agent._rule_based_evaluate(results)
        assert result.next_action == "finalize"
        assert result.overall_status == "success"
        assert result.confidence == 0.9

    def test_rule_based_all_failure_returns_revise_plan(self):
        """
        SENTINEL TEST — all steps failed → _rule_based_evaluate must return revise_plan.
        """
        step_a = _make_step("A")
        step_b = _make_step("B")
        results = [
            _make_result("failure", step_id=step_a.id),
            _make_result("failure", step_id=step_b.id),
        ]
        result = self.agent._rule_based_evaluate(results)
        assert result.next_action == "revise_plan"
        assert result.overall_status == "needs_revision"

    def test_rule_based_partial_failure_returns_retry_step(self):
        """
        SENTINEL TEST — some steps failed → _rule_based_evaluate must return retry_step
        with the first failed step's ID.
        """
        ok = _make_result("success", step_id="s-ok")
        fail = _make_result("failure", step_id="s-fail")
        result = self.agent._rule_based_evaluate([ok, fail])
        assert result.next_action == "retry_step"
        assert result.retry_step_id == "s-fail"

    def test_rule_based_no_steps_returns_ask_user(self):
        """No steps → edge case → ask_user."""
        result = self.agent._rule_based_evaluate([])
        assert result.next_action == "ask_user"


# ---------------------------------------------------------------------------
# Multiple steps — all outputs in prompt
# ---------------------------------------------------------------------------


class TestMultipleStepsInPrompt:
    def setup_method(self):
        self.agent = EvaluationAgent()

    def test_all_step_outputs_appear_in_llm_prompt(self):
        """
        SENTINEL TEST — all step results must be serialised into the LLM prompt.
        Truncating to 1 step would hide step 2's sentinel → test fails.
        """
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _llm_response(
            _valid_eval_json()
        )

        plan = _make_plan(_make_step("A"), _make_step("B"))
        results = [
            _make_result(outputs={"result": "SENTINEL_OUTPUT_ALPHA"}),
            _make_result(outputs={"result": "SENTINEL_OUTPUT_BETA"}),
        ]

        with patch(
            "app.core.agents.evaluation_agent.get_openai_client_sync",
            return_value=mock_client,
        ):
            self.agent.evaluate_execution("query", _make_intent(), plan, results)

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "SENTINEL_OUTPUT_ALPHA" in user_msg["content"]
        assert "SENTINEL_OUTPUT_BETA" in user_msg["content"]

    def test_failed_step_error_appears_in_prompt(self):
        """Step errors must be included so the LLM can reason about them."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _llm_response(
            _valid_eval_json()
        )

        plan = _make_plan(_make_step())
        fail_result = StepResult(
            step_id="s1",
            status="failure",
            outputs={},
            error="SENTINEL_ERROR_MESSAGE",
        )

        with patch(
            "app.core.agents.evaluation_agent.get_openai_client_sync",
            return_value=mock_client,
        ):
            self.agent.evaluate_execution("query", None, plan, [fail_result])

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "SENTINEL_ERROR_MESSAGE" in user_msg["content"]


# ---------------------------------------------------------------------------
# Helper for empty LLM response
# ---------------------------------------------------------------------------


def _llm_response_empty() -> MagicMock:
    msg = MagicMock()
    msg.content = ""
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp
