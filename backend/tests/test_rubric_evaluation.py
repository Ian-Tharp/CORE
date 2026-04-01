"""
Tests for rubric-based evaluation in EvaluationAgent.

Sentinel-value contract:
- Remove get_rubric_for_intent() call → rubric absent from prompt → test fails
- Remove _RUBRICS["code"] entry → code rubric absent → test fails
- Remove rubric injection in _llm_evaluate → rubric never reaches LLM → test fails
- Remove intent-type normalisation → "Code" (caps) not found → test fails
- Remove task_category preference over type → category-specific rubric not injected → test fails
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from app.core.agents.evaluation_agent import (
    EvaluationAgent,
    get_rubric_for_intent,
    get_rubric_for_user_intent,
    _RUBRICS,
    _DEFAULT_RUBRIC,
)
from app.models.core_state import (
    UserIntent,
    ExecutionPlan,
    PlanStep,
    StepResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plan(*names: str) -> ExecutionPlan:
    steps = [PlanStep(name=n, description=f"do {n}") for n in names]
    return ExecutionPlan(goal="test goal", steps=list(steps))


def _make_result(status: str = "success") -> StepResult:
    return StepResult(step_id="s1", status=status, outputs={"result": "ok"})


def _make_intent(
    category: str,
    desc: str = "do something",
    base_type: str = "task",
) -> UserIntent:
    """Create a UserIntent with task_category for rubric selection."""
    return UserIntent(type=base_type, description=desc, confidence=0.9, task_category=category)


def _mock_llm_response(data: dict = None) -> MagicMock:
    data = data or {
        "overall_status": "success",
        "confidence": 0.9,
        "meets_requirements": True,
        "quality_score": 0.9,
        "feedback": "done",
        "next_action": "finalize",
        "retry_step_id": None,
    }
    msg = MagicMock()
    msg.content = json.dumps(data)
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# get_rubric_for_intent unit tests
# ---------------------------------------------------------------------------

class TestGetRubricForIntent:
    def test_code_type_returns_code_rubric(self):
        """
        SENTINEL TEST — 'code' intent must return the code-specific rubric.
        Remove _RUBRICS['code'] → returns default → 'syntax' not in result → fails.
        """
        rubric = get_rubric_for_intent("code")
        assert "syntax" in rubric.lower() or "compile" in rubric.lower(), (
            "Code rubric should mention compilation/syntax"
        )

    def test_research_type_returns_research_rubric(self):
        """
        SENTINEL TEST — 'research' intent must return the research rubric.
        Remove _RUBRICS['research'] → fails.
        """
        rubric = get_rubric_for_intent("research")
        assert "source" in rubric.lower() or "claim" in rubric.lower(), (
            "Research rubric should mention sources/claims"
        )

    def test_file_management_type_returns_file_rubric(self):
        """
        SENTINEL TEST — 'file_management' intent must return file rubric.
        Remove _RUBRICS['file_management'] → default rubric → 'file' specifics absent → fails.
        """
        rubric = get_rubric_for_intent("file_management")
        assert "file" in rubric.lower()

    def test_data_analysis_type_returns_data_rubric(self):
        rubric = get_rubric_for_intent("data_analysis")
        assert "analysis" in rubric.lower() or "statistical" in rubric.lower()

    def test_unknown_type_returns_default_rubric(self):
        """
        SENTINEL TEST — unknown intent type must return _DEFAULT_RUBRIC, not raise.
        Remove fallback → KeyError → test fails.
        """
        rubric = get_rubric_for_intent("unknown_intent_xyz")
        assert rubric == _DEFAULT_RUBRIC

    def test_none_type_returns_default_rubric(self):
        """None intent type must return the default rubric (no crash)."""
        rubric = get_rubric_for_intent(None)
        assert rubric == _DEFAULT_RUBRIC

    def test_empty_string_returns_default_rubric(self):
        rubric = get_rubric_for_intent("")
        assert rubric == _DEFAULT_RUBRIC

    def test_case_insensitive_normalisation(self):
        """
        SENTINEL TEST — 'Code' and 'CODE' must resolve to the same rubric as 'code'.
        Remove .lower() normalisation → upper-case not found → default returned → test fails.
        """
        rubric_lower = get_rubric_for_intent("code")
        rubric_upper = get_rubric_for_intent("CODE")
        rubric_mixed = get_rubric_for_intent("Code")
        assert rubric_lower == rubric_upper == rubric_mixed

    def test_hyphen_normalised(self):
        """'file-management' and 'file_management' must resolve to the same rubric."""
        rubric_hyphen = get_rubric_for_intent("file-management")
        rubric_under = get_rubric_for_intent("file_management")
        assert rubric_hyphen == rubric_under

    def test_all_registered_rubrics_are_non_empty(self):
        """Each rubric in _RUBRICS must be a non-empty string."""
        for intent_type, rubric in _RUBRICS.items():
            assert isinstance(rubric, str) and len(rubric) > 0, (
                f"Rubric for '{intent_type}' is empty"
            )

    def test_default_rubric_is_non_empty(self):
        assert isinstance(_DEFAULT_RUBRIC, str) and len(_DEFAULT_RUBRIC) > 0


# ---------------------------------------------------------------------------
# Rubric injection into LLM prompt
# ---------------------------------------------------------------------------

class TestRubricInjectedIntoLLMPrompt:
    """
    SENTINEL TESTS — the rubric text must flow from get_rubric_for_intent()
    into the user message sent to the LLM.  If the injection call is removed,
    the rubric keyword won't appear in the captured prompt.
    """

    def setup_method(self):
        self.agent = EvaluationAgent()

    def _capture_user_prompt(self, mock_client) -> str:
        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        return next(m["content"] for m in messages if m["role"] == "user")

    def test_code_rubric_injected_for_code_intent(self):
        """
        SENTINEL TEST — when intent type is 'code', the code rubric must appear
        in the LLM user message.
        Remove rubric injection → code criteria absent → test fails.
        """
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_llm_response()

        plan = _make_plan("step1")
        intent = _make_intent("code", desc="Write a Python sort function")

        with patch("app.core.agents.evaluation_agent.get_openai_client_sync",
                   return_value=mock_client):
            self.agent.evaluate_execution("write a sort function", intent, plan, [_make_result()])

        prompt = self._capture_user_prompt(mock_client)
        code_rubric = get_rubric_for_intent("code")
        # The rubric keyword must appear verbatim in the prompt
        assert any(line.strip() in prompt for line in code_rubric.splitlines() if line.strip()), (
            f"Code rubric not found in prompt. Prompt: {prompt[:400]}"
        )

    def test_research_rubric_injected_for_research_intent(self):
        """
        SENTINEL TEST — 'research' intent injects research rubric into prompt.
        """
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_llm_response()

        plan = _make_plan("step1")
        intent = _make_intent("research", desc="Research quantum computing")

        with patch("app.core.agents.evaluation_agent.get_openai_client_sync",
                   return_value=mock_client):
            self.agent.evaluate_execution("research quantum", intent, plan, [_make_result()])

        prompt = self._capture_user_prompt(mock_client)
        research_rubric = get_rubric_for_intent("research")
        assert any(line.strip() in prompt for line in research_rubric.splitlines() if line.strip()), (
            f"Research rubric not found in prompt. Prompt: {prompt[:400]}"
        )

    def test_different_intents_inject_different_rubrics(self):
        """
        SENTINEL TEST — code and research intents must inject DIFFERENT rubrics.
        Return the same rubric always → rubric text identical → test fails.
        """
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_llm_response()

        plan = _make_plan("step1")

        prompts = {}
        for intent_type in ("code", "research"):
            mock_client.reset_mock()
            with patch("app.core.agents.evaluation_agent.get_openai_client_sync",
                       return_value=mock_client):
                self.agent.evaluate_execution(
                    "do something",
                    _make_intent(intent_type),
                    plan,
                    [_make_result()],
                )
            prompts[intent_type] = self._capture_user_prompt(mock_client)

        assert prompts["code"] != prompts["research"], (
            "Code and research prompts are identical — rubric injection is broken"
        )

    def test_task_rubric_injected_for_generic_task(self):
        """'task' task_category injects the task rubric."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_llm_response()

        plan = _make_plan("step1")
        intent = _make_intent("task", desc="Do a thing")

        with patch("app.core.agents.evaluation_agent.get_openai_client_sync",
                   return_value=mock_client):
            self.agent.evaluate_execution("do a thing", intent, plan, [_make_result()])

        prompt = self._capture_user_prompt(mock_client)
        task_rubric = get_rubric_for_intent("task")
        assert any(line.strip() in prompt for line in task_rubric.splitlines() if line.strip())

    def test_unknown_category_injects_default_rubric(self):
        """An unknown task_category injects the default rubric without raising."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_llm_response()

        plan = _make_plan("step1")
        # Use a valid UserIntent type but an unknown task_category
        intent = UserIntent(
            type="task",
            description="do exotic thing",
            confidence=0.9,
            task_category="some_exotic_category_xyz",
        )

        with patch("app.core.agents.evaluation_agent.get_openai_client_sync",
                   return_value=mock_client):
            self.agent.evaluate_execution("do exotic thing", intent, plan, [_make_result()])

        prompt = self._capture_user_prompt(mock_client)
        # Unknown category → falls back to "task" type → task rubric injected
        task_rubric = get_rubric_for_intent("task")
        assert any(line.strip() in prompt for line in task_rubric.splitlines() if line.strip())

    def test_rubric_present_even_with_no_intent(self):
        """When intent is None, the default rubric must still be injected."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_llm_response()

        plan = _make_plan("step1")

        with patch("app.core.agents.evaluation_agent.get_openai_client_sync",
                   return_value=mock_client):
            self.agent.evaluate_execution("do something", None, plan, [_make_result()])

        prompt = self._capture_user_prompt(mock_client)
        # Default rubric lines must appear somewhere in the prompt
        assert any(line.strip() in prompt for line in _DEFAULT_RUBRIC.splitlines() if line.strip())


# ---------------------------------------------------------------------------
# Rubric doesn't break existing evaluation flow
# ---------------------------------------------------------------------------

class TestRubricDoesNotBreakEvaluation:
    def setup_method(self):
        self.agent = EvaluationAgent()

    def test_evaluation_result_still_valid_with_rubric_injected(self):
        """Adding a rubric must not corrupt the EvaluationResult parsing."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_llm_response({
            "overall_status": "success",
            "confidence": 0.88,
            "meets_requirements": True,
            "quality_score": 0.92,
            "feedback": "All good.",
            "next_action": "finalize",
            "retry_step_id": None,
        })

        plan = _make_plan("step1")
        intent = _make_intent("code")  # task_category="code"

        with patch("app.core.agents.evaluation_agent.get_openai_client_sync",
                   return_value=mock_client):
            result = self.agent.evaluate_execution("write code", intent, plan, [_make_result()])

        assert result.next_action == "finalize"
        assert abs(result.confidence - 0.88) < 0.001
        assert result.overall_status == "success"

    def test_llm_failure_still_falls_back_to_rule_based(self):
        """With rubric injection enabled, LLM failure must still fall back gracefully."""
        plan = _make_plan("step1")
        intent = _make_intent("research")  # task_category="research"

        with patch("app.core.agents.evaluation_agent.get_openai_client_sync",
                   side_effect=RuntimeError("LLM down")):
            result = self.agent.evaluate_execution("research something", intent, plan, [_make_result()])

        # Rule-based for single success → finalize
        assert result.next_action == "finalize"
