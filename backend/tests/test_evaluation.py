"""
Evaluation Engine Tests

Comprehensive tests for the CORE Evaluation Engine following AAA format.
Covers quality scoring, verdict determination, plan completion, retry logic,
feedback, metrics, and edge cases.

20+ test scenarios.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4, UUID
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.evaluation_models import (
    EvaluationInput,
    EvaluateStepInput,
    EvaluationResult,
    EvaluationMetrics,
    EvaluationVerdict,
    EvaluationThresholds,
    HumanFeedbackInput,
    PlanCompletionStatus,
    PlanStep,
    QualityScore,
    RetryDecision,
    StepEvaluation,
    StepResult,
    StepStatus,
    Verdict,
    EvaluationHistoryFilter,
)
from app.services.evaluation_service import (
    evaluate,
    evaluate_step,
    should_retry,
    _score_quality,
    _heuristic_accuracy,
    _heuristic_completeness,
    _heuristic_relevance,
    _heuristic_coherence,
    _check_plan_completion,
    _determine_verdict,
    _build_retry_decision,
    _generate_feedback,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_plan_steps():
    """Three-step plan for testing."""
    return [
        PlanStep(step_index=0, description="Research the topic", expected_output="Summary of findings", required=True),
        PlanStep(step_index=1, description="Analyze the data", expected_output="Analysis report", required=True),
        PlanStep(step_index=2, description="Write conclusion", expected_output="Conclusion paragraph", required=False),
    ]


@pytest.fixture
def all_completed_step_results():
    """All steps completed successfully."""
    return [
        StepResult(step_index=0, status=StepStatus.COMPLETED, output="Found relevant research data."),
        StepResult(step_index=1, status=StepStatus.COMPLETED, output="Data analysis shows strong trend."),
        StepResult(step_index=2, status=StepStatus.COMPLETED, output="In conclusion, the findings support the hypothesis."),
    ]


@pytest.fixture
def partial_step_results():
    """One step failed, one partial."""
    return [
        StepResult(step_index=0, status=StepStatus.COMPLETED, output="Found relevant research data."),
        StepResult(step_index=1, status=StepStatus.FAILED, error="Analysis service unavailable"),
        StepResult(step_index=2, status=StepStatus.PARTIAL, output="Partial conclusion drafted."),
    ]


@pytest.fixture
def good_evaluation_input(sample_plan_steps, all_completed_step_results):
    """Evaluation input with good results."""
    return EvaluationInput(
        task_id=uuid4(),
        original_intent="Research and analyze climate change impacts on agriculture",
        plan_steps=sample_plan_steps,
        step_results=all_completed_step_results,
        final_output=(
            "Climate change has significant impacts on agriculture. "
            "Research shows rising temperatures affect crop yields. "
            "Data analysis reveals a 15% decline in wheat production over the last decade. "
            "In conclusion, adaptive strategies are urgently needed to mitigate these impacts."
        ),
        agent_id=uuid4(),
        model_used="claude-sonnet",
        execution_duration_ms=5000,
        retry_count=0,
        max_retries=3,
    )


@pytest.fixture
def poor_evaluation_input(sample_plan_steps, partial_step_results):
    """Evaluation input with poor results."""
    return EvaluationInput(
        task_id=uuid4(),
        original_intent="Research and analyze climate change impacts on agriculture",
        plan_steps=sample_plan_steps,
        step_results=partial_step_results,
        final_output="Some data was found.",
        agent_id=uuid4(),
        model_used="claude-sonnet",
        execution_duration_ms=3000,
        retry_count=0,
        max_retries=3,
    )


# =============================================================================
# 1. QUALITY SCORE MODEL TESTS
# =============================================================================

class TestQualityScore:
    """Tests for the QualityScore model."""

    def test_compute_overall_default_weights(self):
        # Arrange
        score = QualityScore(accuracy=0.9, completeness=0.8, relevance=0.7, coherence=0.6)

        # Act
        overall = score.compute_overall()

        # Assert
        assert 0.0 <= overall <= 1.0
        assert score.overall == overall
        expected = (0.9 * 0.35 + 0.8 * 0.25 + 0.7 * 0.25 + 0.6 * 0.15) / 1.0
        assert abs(overall - expected) < 0.001

    def test_compute_overall_custom_weights(self):
        # Arrange
        score = QualityScore(accuracy=1.0, completeness=0.0, relevance=0.0, coherence=0.0)
        weights = {"accuracy": 1.0, "completeness": 0.0, "relevance": 0.0, "coherence": 0.0}

        # Act
        overall = score.compute_overall(weights=weights)

        # Assert
        assert overall == 1.0

    def test_compute_overall_all_zero(self):
        # Arrange
        score = QualityScore(accuracy=0.0, completeness=0.0, relevance=0.0, coherence=0.0)

        # Act
        overall = score.compute_overall()

        # Assert
        assert overall == 0.0


# =============================================================================
# 2. HEURISTIC ACCURACY TESTS
# =============================================================================

class TestHeuristicAccuracy:
    """Tests for the accuracy scoring heuristic."""

    def test_empty_output_returns_zero(self):
        # Arrange / Act
        result = _heuristic_accuracy("", [])

        # Assert
        assert result == 0.0

    def test_nonempty_output_no_steps_returns_base(self):
        # Arrange / Act
        result = _heuristic_accuracy("Some output text", [])

        # Assert
        assert result == 0.4

    def test_all_steps_completed_high_accuracy(self):
        # Arrange
        results = [
            StepResult(step_index=0, status=StepStatus.COMPLETED, output="ok"),
            StepResult(step_index=1, status=StepStatus.COMPLETED, output="ok"),
        ]

        # Act
        result = _heuristic_accuracy("Output text", results)

        # Assert
        assert result == 1.0  # 0.4 base + 0.6 * (0 errors / 2)

    def test_all_steps_failed_low_accuracy(self):
        # Arrange
        results = [
            StepResult(step_index=0, status=StepStatus.FAILED, error="fail"),
            StepResult(step_index=1, status=StepStatus.FAILED, error="fail"),
        ]

        # Act
        result = _heuristic_accuracy("Output text", results)

        # Assert
        assert result == 0.4  # base only, 0.6 * 0.0


# =============================================================================
# 3. HEURISTIC COMPLETENESS TESTS
# =============================================================================

class TestHeuristicCompleteness:
    """Tests for the completeness scoring heuristic."""

    def test_no_plan_steps_returns_half(self):
        # Arrange / Act
        result = _heuristic_completeness([], [])

        # Assert
        assert result == 0.5

    def test_all_completed(self, sample_plan_steps, all_completed_step_results):
        # Arrange / Act
        result = _heuristic_completeness(sample_plan_steps, all_completed_step_results)

        # Assert
        assert result == 1.0

    def test_one_failed_one_partial(self, sample_plan_steps, partial_step_results):
        # Arrange / Act
        result = _heuristic_completeness(sample_plan_steps, partial_step_results)

        # Assert
        # 1 completed + 0.5 * 1 partial = 1.5 / 3 = 0.5
        assert abs(result - 0.5) < 0.001


# =============================================================================
# 4. HEURISTIC RELEVANCE TESTS
# =============================================================================

class TestHeuristicRelevance:
    """Tests for the relevance scoring heuristic."""

    def test_empty_intent_returns_zero(self):
        # Arrange / Act
        result = _heuristic_relevance("", "some output")

        # Assert
        assert result == 0.0

    def test_empty_output_returns_zero(self):
        # Arrange / Act
        result = _heuristic_relevance("some intent", "")

        # Assert
        assert result == 0.0

    def test_high_overlap(self):
        # Arrange
        intent = "climate change impacts agriculture"
        output = "climate change has significant impacts on agriculture production"

        # Act
        result = _heuristic_relevance(intent, output)

        # Assert
        assert result >= 0.5  # Strong overlap expected

    def test_no_overlap(self):
        # Arrange
        intent = "quantum physics experiments"
        output = "The best recipe for chocolate cake involves butter and sugar"

        # Act
        result = _heuristic_relevance(intent, output)

        # Assert
        assert result < 0.3


# =============================================================================
# 5. HEURISTIC COHERENCE TESTS
# =============================================================================

class TestHeuristicCoherence:
    """Tests for the coherence scoring heuristic."""

    def test_empty_output_returns_zero(self):
        # Arrange / Act
        result = _heuristic_coherence("")

        # Assert
        assert result == 0.0

    def test_short_output_low_coherence(self):
        # Arrange / Act
        result = _heuristic_coherence("Hello")

        # Assert
        assert result == 0.3  # base only

    def test_structured_output_high_coherence(self):
        # Arrange
        output = (
            "First, we analyze the data.\n"
            "Second, we interpret findings.\n"
            "- Key point one.\n"
            "- Key point two.\n"
            "In conclusion, the results are significant. "
            "This has implications for future research."
        )

        # Act
        result = _heuristic_coherence(output)

        # Assert
        assert result >= 0.7


# =============================================================================
# 6. PLAN COMPLETION TESTS
# =============================================================================

class TestPlanCompletion:
    """Tests for plan completion checking."""

    def test_empty_plan_vacuously_complete(self):
        # Arrange / Act
        result = _check_plan_completion([], [])

        # Assert
        assert result.total_steps == 0
        assert result.completion_rate == 1.0
        assert result.required_steps_met is True

    def test_all_steps_completed(self, sample_plan_steps, all_completed_step_results):
        # Arrange / Act
        result = _check_plan_completion(sample_plan_steps, all_completed_step_results)

        # Assert
        assert result.total_steps == 3
        assert result.completed_steps == 3
        assert result.failed_steps == 0
        assert result.completion_rate == 1.0
        assert result.required_steps_met is True

    def test_required_step_failed(self, sample_plan_steps, partial_step_results):
        # Arrange / Act
        result = _check_plan_completion(sample_plan_steps, partial_step_results)

        # Assert
        assert result.required_steps_met is False
        assert result.failed_steps == 1

    def test_missing_step_results(self, sample_plan_steps):
        # Arrange — only one result for three steps
        results = [StepResult(step_index=0, status=StepStatus.COMPLETED, output="done")]

        # Act
        result = _check_plan_completion(sample_plan_steps, results)

        # Assert
        assert result.completed_steps == 1
        assert result.required_steps_met is False  # step 1 is required but not started

    def test_step_evaluations_populated(self, sample_plan_steps, all_completed_step_results):
        # Arrange / Act
        result = _check_plan_completion(sample_plan_steps, all_completed_step_results)

        # Assert
        assert len(result.step_evaluations) == 3
        for se in result.step_evaluations:
            assert se.quality_score > 0
            assert se.meets_criteria is True


# =============================================================================
# 7. VERDICT DETERMINATION TESTS
# =============================================================================

class TestVerdictDetermination:
    """Tests for the verdict determination logic."""

    def test_approve_high_quality(self):
        # Arrange
        quality = QualityScore(accuracy=0.9, completeness=0.8, relevance=0.9, coherence=0.8, overall=0.87)
        plan = PlanCompletionStatus(required_steps_met=True)

        # Act
        verdict = _determine_verdict(quality, plan, retry_count=0, max_retries=3)

        # Assert
        assert verdict.verdict == Verdict.APPROVE

    def test_retry_mediocre_quality(self):
        # Arrange
        quality = QualityScore(accuracy=0.5, completeness=0.5, relevance=0.5, coherence=0.5, overall=0.5)
        plan = PlanCompletionStatus(required_steps_met=True)

        # Act
        verdict = _determine_verdict(quality, plan, retry_count=0, max_retries=3)

        # Assert
        assert verdict.verdict == Verdict.RETRY

    def test_escalate_max_retries_exhausted(self):
        # Arrange
        quality = QualityScore(accuracy=0.5, completeness=0.5, relevance=0.5, coherence=0.5, overall=0.5)
        plan = PlanCompletionStatus(required_steps_met=True)

        # Act
        verdict = _determine_verdict(quality, plan, retry_count=3, max_retries=3)

        # Assert
        assert verdict.verdict == Verdict.ESCALATE

    def test_escalate_very_low_quality(self):
        # Arrange
        quality = QualityScore(accuracy=0.1, completeness=0.1, relevance=0.1, coherence=0.1, overall=0.1)
        plan = PlanCompletionStatus(required_steps_met=False)

        # Act
        verdict = _determine_verdict(quality, plan, retry_count=0, max_retries=3)

        # Assert
        assert verdict.verdict == Verdict.ESCALATE

    def test_refine_when_plan_steps_unmet_but_decent_quality(self):
        # Arrange
        quality = QualityScore(accuracy=0.6, completeness=0.5, relevance=0.6, coherence=0.6, overall=0.58)
        plan = PlanCompletionStatus(required_steps_met=False)

        # Act
        verdict = _determine_verdict(quality, plan, retry_count=0, max_retries=3)

        # Assert
        assert verdict.verdict == Verdict.REFINE

    def test_improvements_populated_on_low_dimensions(self):
        # Arrange
        quality = QualityScore(accuracy=0.3, completeness=0.4, relevance=0.3, coherence=0.4, overall=0.35)
        plan = PlanCompletionStatus(required_steps_met=True)

        # Act
        verdict = _determine_verdict(quality, plan, retry_count=0, max_retries=3)

        # Assert
        assert len(verdict.suggested_improvements) > 0


# =============================================================================
# 8. RETRY LOGIC TESTS
# =============================================================================

class TestRetryLogic:
    """Tests for retry decision building."""

    def test_no_retry_on_approve(self):
        # Arrange
        verdict = EvaluationVerdict(verdict=Verdict.APPROVE, reasoning="Good", confidence=0.9)
        quality = QualityScore(overall=0.9)

        # Act
        decision = _build_retry_decision(verdict, quality, retry_count=0, max_retries=3)

        # Assert
        assert decision.should_retry is False
        assert decision.escalate is False

    def test_retry_with_exponential_backoff(self):
        # Arrange
        verdict = EvaluationVerdict(verdict=Verdict.RETRY, reasoning="Needs improvement", confidence=0.7)
        quality = QualityScore(overall=0.5)

        # Act
        d0 = _build_retry_decision(verdict, quality, retry_count=0, max_retries=3)
        d1 = _build_retry_decision(verdict, quality, retry_count=1, max_retries=3)
        d2 = _build_retry_decision(verdict, quality, retry_count=2, max_retries=3)

        # Assert — backoff increases
        assert d0.should_retry is True
        assert d1.should_retry is True
        assert d2.should_retry is True
        assert d0.delay_ms < d1.delay_ms < d2.delay_ms

    def test_escalation_on_escalate_verdict(self):
        # Arrange
        verdict = EvaluationVerdict(verdict=Verdict.ESCALATE, reasoning="Too many failures", confidence=0.9)
        quality = QualityScore(overall=0.2)

        # Act
        decision = _build_retry_decision(verdict, quality, retry_count=3, max_retries=3)

        # Assert
        assert decision.should_retry is False
        assert decision.escalate is True

    def test_retry_delay_capped(self):
        # Arrange
        verdict = EvaluationVerdict(verdict=Verdict.RETRY, reasoning="Retry", confidence=0.7)
        quality = QualityScore(overall=0.5)

        # Act — high retry count should still cap
        decision = _build_retry_decision(verdict, quality, retry_count=20, max_retries=100)

        # Assert
        assert decision.delay_ms <= EvaluationThresholds.RETRY_MAX_DELAY_MS

    def test_adjustments_contain_focus_area(self):
        # Arrange
        verdict = EvaluationVerdict(verdict=Verdict.RETRY, reasoning="Retry needed", confidence=0.6,
                                     suggested_improvements=["Improve accuracy"])
        quality = QualityScore(accuracy=0.3, completeness=0.8, relevance=0.8, coherence=0.8, overall=0.5)

        # Act
        decision = _build_retry_decision(verdict, quality, retry_count=0, max_retries=3)

        # Assert
        assert decision.adjustments.get("focus") == "accuracy"


# =============================================================================
# 9. FEEDBACK GENERATION TESTS
# =============================================================================

class TestFeedbackGeneration:
    """Tests for generating human-readable feedback."""

    def test_feedback_contains_verdict(self):
        # Arrange
        quality = QualityScore(overall=0.85, accuracy=0.9, completeness=0.8, relevance=0.85, coherence=0.8)
        plan = PlanCompletionStatus(total_steps=2, completed_steps=2, required_steps_met=True, completion_rate=1.0)
        verdict = EvaluationVerdict(verdict=Verdict.APPROVE, reasoning="All good", confidence=0.9)

        # Act
        feedback = _generate_feedback(quality, plan, verdict)

        # Assert
        assert "APPROVE" in feedback

    def test_feedback_contains_plan_info(self):
        # Arrange
        quality = QualityScore(overall=0.5)
        plan = PlanCompletionStatus(total_steps=3, completed_steps=1, required_steps_met=False, completion_rate=0.33)
        verdict = EvaluationVerdict(verdict=Verdict.RETRY, reasoning="Incomplete", confidence=0.6)

        # Act
        feedback = _generate_feedback(quality, plan, verdict)

        # Assert
        assert "1/3" in feedback
        assert "NO" in feedback  # required_steps_met = False

    def test_feedback_contains_improvements(self):
        # Arrange
        quality = QualityScore(overall=0.4)
        plan = PlanCompletionStatus()
        verdict = EvaluationVerdict(
            verdict=Verdict.RETRY,
            reasoning="Low quality",
            confidence=0.5,
            suggested_improvements=["Fix accuracy", "Add details"],
        )

        # Act
        feedback = _generate_feedback(quality, plan, verdict)

        # Assert
        assert "Fix accuracy" in feedback
        assert "Add details" in feedback


# =============================================================================
# 10. FULL EVALUATE INTEGRATION TESTS
# =============================================================================

class TestEvaluateIntegration:
    """Integration tests for the full evaluate() pipeline."""

    @pytest.mark.asyncio
    async def test_good_input_gets_approved(self, good_evaluation_input):
        # Arrange — mock repository and memory
        with patch("app.services.evaluation_service.evaluation_repository") as mock_repo, \
             patch("app.services.evaluation_service.memory_repository") as mock_mem:
            mock_repo.store_evaluation = AsyncMock(return_value=uuid4())
            mock_mem.create_procedural_memory = AsyncMock(return_value=uuid4())

            # Act
            result = await evaluate(good_evaluation_input)

            # Assert
            assert result.verdict.verdict == Verdict.APPROVE
            assert result.quality_score.overall >= EvaluationThresholds.APPROVE_QUALITY
            assert result.plan_completion.required_steps_met is True
            mock_repo.store_evaluation.assert_called_once()
            # Approved → should store success pattern
            mock_mem.create_procedural_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_poor_input_gets_retry(self, poor_evaluation_input):
        # Arrange
        with patch("app.services.evaluation_service.evaluation_repository") as mock_repo, \
             patch("app.services.evaluation_service.memory_repository") as mock_mem:
            mock_repo.store_evaluation = AsyncMock(return_value=uuid4())
            mock_mem.create_procedural_memory = AsyncMock(return_value=uuid4())

            # Act
            result = await evaluate(poor_evaluation_input)

            # Assert
            assert result.verdict.verdict in (Verdict.RETRY, Verdict.REFINE, Verdict.ESCALATE)
            assert result.retry_decision is not None
            mock_repo.store_evaluation.assert_called_once()

    @pytest.mark.asyncio
    async def test_max_retries_escalates(self, poor_evaluation_input):
        # Arrange
        poor_evaluation_input.retry_count = 3
        poor_evaluation_input.max_retries = 3

        with patch("app.services.evaluation_service.evaluation_repository") as mock_repo, \
             patch("app.services.evaluation_service.memory_repository") as mock_mem:
            mock_repo.store_evaluation = AsyncMock(return_value=uuid4())
            mock_mem.create_procedural_memory = AsyncMock(return_value=uuid4())

            # Act
            result = await evaluate(poor_evaluation_input)

            # Assert
            assert result.verdict.verdict == Verdict.ESCALATE
            assert result.retry_decision is not None
            assert result.retry_decision.escalate is True


# =============================================================================
# 11. STEP EVALUATION TESTS
# =============================================================================

class TestStepEvaluation:
    """Tests for individual step evaluation."""

    @pytest.mark.asyncio
    async def test_completed_step_high_quality(self):
        # Arrange
        step_input = EvaluateStepInput(
            task_id=uuid4(),
            step_plan=PlanStep(step_index=0, description="Fetch data", required=True),
            step_result=StepResult(step_index=0, status=StepStatus.COMPLETED, output="Data fetched."),
            original_intent="Get data from API",
        )

        # Act
        result = await evaluate_step(step_input)

        # Assert
        assert result.quality_score >= 0.8
        assert result.meets_criteria is True

    @pytest.mark.asyncio
    async def test_failed_step_low_quality(self):
        # Arrange
        step_input = EvaluateStepInput(
            task_id=uuid4(),
            step_plan=PlanStep(step_index=0, description="Fetch data", required=True),
            step_result=StepResult(step_index=0, status=StepStatus.FAILED, error="Timeout"),
            original_intent="Get data from API",
        )

        # Act
        result = await evaluate_step(step_input)

        # Assert
        assert result.quality_score <= 0.2
        assert result.meets_criteria is False


# =============================================================================
# 12. SHOULD_RETRY TESTS
# =============================================================================

class TestShouldRetry:
    """Tests for the should_retry() service function."""

    @pytest.mark.asyncio
    async def test_approved_no_retry(self):
        # Arrange
        eval_result = EvaluationResult(
            task_id=uuid4(),
            verdict=EvaluationVerdict(verdict=Verdict.APPROVE, reasoning="Good", confidence=0.9),
        )

        # Act
        decision = await should_retry(eval_result)

        # Assert
        assert decision.should_retry is False

    @pytest.mark.asyncio
    async def test_retry_verdict_triggers_retry(self):
        # Arrange
        eval_result = EvaluationResult(
            task_id=uuid4(),
            quality_score=QualityScore(overall=0.5),
            verdict=EvaluationVerdict(verdict=Verdict.RETRY, reasoning="Mediocre", confidence=0.7),
            retry_decision=RetryDecision(should_retry=True, retry_count=1, max_retries=3, reason="retry"),
        )

        # Act
        decision = await should_retry(eval_result)

        # Assert
        assert decision.should_retry is True


# =============================================================================
# 13. EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Edge case and boundary tests."""

    @pytest.mark.asyncio
    async def test_empty_output(self):
        # Arrange
        inp = EvaluationInput(
            task_id=uuid4(),
            original_intent="Do something useful",
            final_output="",
            retry_count=0,
            max_retries=3,
        )

        with patch("app.services.evaluation_service.evaluation_repository") as mock_repo, \
             patch("app.services.evaluation_service.memory_repository") as mock_mem:
            mock_repo.store_evaluation = AsyncMock(return_value=uuid4())
            mock_mem.create_procedural_memory = AsyncMock(return_value=uuid4())

            # Act
            result = await evaluate(inp)

            # Assert — empty output should not be approved
            assert result.verdict.verdict != Verdict.APPROVE
            assert result.quality_score.accuracy == 0.0

    @pytest.mark.asyncio
    async def test_whitespace_only_output(self):
        # Arrange
        inp = EvaluationInput(
            task_id=uuid4(),
            original_intent="Do something useful",
            final_output="   \n\t  ",
            retry_count=0,
            max_retries=3,
        )

        with patch("app.services.evaluation_service.evaluation_repository") as mock_repo, \
             patch("app.services.evaluation_service.memory_repository") as mock_mem:
            mock_repo.store_evaluation = AsyncMock(return_value=uuid4())
            mock_mem.create_procedural_memory = AsyncMock(return_value=uuid4())

            # Act
            result = await evaluate(inp)

            # Assert
            assert result.verdict.verdict != Verdict.APPROVE

    @pytest.mark.asyncio
    async def test_no_plan_steps(self):
        # Arrange
        inp = EvaluationInput(
            task_id=uuid4(),
            original_intent="Quick question answer",
            plan_steps=[],
            step_results=[],
            final_output="The answer is 42. This is a well-known reference from Douglas Adams' Hitchhiker's Guide.",
            retry_count=0,
            max_retries=3,
        )

        with patch("app.services.evaluation_service.evaluation_repository") as mock_repo, \
             patch("app.services.evaluation_service.memory_repository") as mock_mem:
            mock_repo.store_evaluation = AsyncMock(return_value=uuid4())
            mock_mem.create_procedural_memory = AsyncMock(return_value=uuid4())

            # Act
            result = await evaluate(inp)

            # Assert — plan completion should be vacuously complete
            assert result.plan_completion.total_steps == 0
            assert result.plan_completion.required_steps_met is True

    @pytest.mark.asyncio
    async def test_conflicting_step_results(self, sample_plan_steps):
        """Steps results that partially conflict — some pass, some fail."""
        # Arrange
        results = [
            StepResult(step_index=0, status=StepStatus.COMPLETED, output="Done"),
            StepResult(step_index=1, status=StepStatus.COMPLETED, output="Done analysis"),
            StepResult(step_index=2, status=StepStatus.FAILED, error="Could not write conclusion"),
        ]
        inp = EvaluationInput(
            task_id=uuid4(),
            original_intent="Research climate impacts",
            plan_steps=sample_plan_steps,
            step_results=results,
            final_output="Research shows climate impacts are significant. Analysis confirms this with data.",
            retry_count=0,
            max_retries=3,
        )

        with patch("app.services.evaluation_service.evaluation_repository") as mock_repo, \
             patch("app.services.evaluation_service.memory_repository") as mock_mem:
            mock_repo.store_evaluation = AsyncMock(return_value=uuid4())
            mock_mem.create_procedural_memory = AsyncMock(return_value=uuid4())

            # Act
            result = await evaluate(inp)

            # Assert — step 2 is optional, so required should still be met
            assert result.plan_completion.required_steps_met is True
            assert result.plan_completion.failed_steps == 1


# =============================================================================
# 14. HUMAN FEEDBACK TESTS
# =============================================================================

class TestHumanFeedback:
    """Tests for human feedback recording."""

    @pytest.mark.asyncio
    async def test_record_agreeing_feedback(self):
        # Arrange
        eval_id = uuid4()
        feedback = HumanFeedbackInput(
            evaluation_id=eval_id,
            agree_with_verdict=True,
            feedback_text="Looks correct",
        )

        with patch("app.services.evaluation_service.evaluation_repository") as mock_repo:
            mock_repo.record_human_feedback = AsyncMock(return_value=True)

            # Act
            from app.services.evaluation_service import record_human_feedback
            result = await record_human_feedback(eval_id, feedback)

            # Assert
            assert result is True
            mock_repo.record_human_feedback.assert_called_once_with(eval_id, feedback)

    @pytest.mark.asyncio
    async def test_record_disagreeing_feedback(self):
        # Arrange
        eval_id = uuid4()
        feedback = HumanFeedbackInput(
            evaluation_id=eval_id,
            agree_with_verdict=False,
            corrected_verdict=Verdict.APPROVE,
            quality_override=0.9,
            feedback_text="This was actually good output",
        )

        with patch("app.services.evaluation_service.evaluation_repository") as mock_repo:
            mock_repo.record_human_feedback = AsyncMock(return_value=True)

            # Act
            from app.services.evaluation_service import record_human_feedback
            result = await record_human_feedback(eval_id, feedback)

            # Assert
            assert result is True


# =============================================================================
# 15. METRICS TESTS
# =============================================================================

class TestMetrics:
    """Tests for evaluation metrics aggregation."""

    @pytest.mark.asyncio
    async def test_get_metrics_delegates(self):
        # Arrange
        expected = EvaluationMetrics(
            total_evaluations=100,
            approved_count=70,
            retry_count=20,
            refine_count=5,
            escalation_count=5,
            approval_rate=0.7,
            retry_rate=0.2,
            escalation_rate=0.05,
            avg_quality_score=0.75,
        )

        with patch("app.services.evaluation_service.evaluation_repository") as mock_repo:
            mock_repo.get_evaluation_metrics = AsyncMock(return_value=expected)

            # Act
            from app.services.evaluation_service import get_evaluation_metrics
            result = await get_evaluation_metrics()

            # Assert
            assert result.total_evaluations == 100
            assert result.approval_rate == 0.7


# =============================================================================
# 16. MEMORY INTEGRATION TESTS
# =============================================================================

class TestMemoryIntegration:
    """Tests for procedural memory storage on evaluation."""

    @pytest.mark.asyncio
    async def test_approved_stores_success_pattern(self, good_evaluation_input):
        # Arrange
        with patch("app.services.evaluation_service.evaluation_repository") as mock_repo, \
             patch("app.services.evaluation_service.memory_repository") as mock_mem:
            mock_repo.store_evaluation = AsyncMock(return_value=uuid4())
            mock_mem.create_procedural_memory = AsyncMock(return_value=uuid4())

            # Act
            result = await evaluate(good_evaluation_input)

            # Assert
            assert result.verdict.verdict == Verdict.APPROVE
            call_args = mock_mem.create_procedural_memory.call_args
            memory = call_args[0][0]
            assert memory.metadata["learning_type"] == "success"

    @pytest.mark.asyncio
    async def test_failed_stores_failure_pattern(self, poor_evaluation_input):
        # Arrange
        with patch("app.services.evaluation_service.evaluation_repository") as mock_repo, \
             patch("app.services.evaluation_service.memory_repository") as mock_mem:
            mock_repo.store_evaluation = AsyncMock(return_value=uuid4())
            mock_mem.create_procedural_memory = AsyncMock(return_value=uuid4())

            # Act
            result = await evaluate(poor_evaluation_input)

            # Assert
            if result.verdict.verdict in (Verdict.RETRY, Verdict.ESCALATE):
                call_args = mock_mem.create_procedural_memory.call_args
                memory = call_args[0][0]
                assert memory.metadata["learning_type"] == "failure"


# =============================================================================
# 17. MODEL VALIDATION TESTS
# =============================================================================

class TestModelValidation:
    """Tests for Pydantic model validation."""

    def test_quality_score_bounds(self):
        # Arrange / Act / Assert — should not accept out-of-range
        with pytest.raises(Exception):
            QualityScore(accuracy=1.5)

    def test_quality_score_valid(self):
        # Arrange / Act
        score = QualityScore(accuracy=0.5, completeness=0.6, relevance=0.7, coherence=0.8)

        # Assert
        assert score.accuracy == 0.5

    def test_evaluation_input_requires_fields(self):
        # Arrange / Act / Assert — task_id and original_intent required
        with pytest.raises(Exception):
            EvaluationInput(final_output="test")

    def test_verdict_enum_values(self):
        # Arrange / Act / Assert
        assert Verdict.APPROVE.value == "approve"
        assert Verdict.RETRY.value == "retry"
        assert Verdict.REFINE.value == "refine"
        assert Verdict.ESCALATE.value == "escalate"

    def test_step_status_enum_values(self):
        # Arrange / Act / Assert
        assert StepStatus.COMPLETED.value == "completed"
        assert StepStatus.FAILED.value == "failed"
        assert StepStatus.PARTIAL.value == "partial"
        assert StepStatus.SKIPPED.value == "skipped"
        assert StepStatus.NOT_STARTED.value == "not_started"


# =============================================================================
# 18. EVALUATION RESULT SERIALIZATION
# =============================================================================

class TestSerialization:
    """Tests for model serialization."""

    def test_evaluation_result_json_round_trip(self):
        # Arrange
        result = EvaluationResult(
            task_id=uuid4(),
            quality_score=QualityScore(accuracy=0.9, completeness=0.8, relevance=0.7, coherence=0.6, overall=0.8),
            verdict=EvaluationVerdict(verdict=Verdict.APPROVE, reasoning="Good", confidence=0.9),
        )

        # Act
        json_str = result.json()
        restored = EvaluationResult.parse_raw(json_str)

        # Assert
        assert restored.task_id == result.task_id
        assert restored.verdict.verdict == Verdict.APPROVE
        assert restored.quality_score.accuracy == 0.9


# =============================================================================
# 19. THRESHOLD CONSTANTS
# =============================================================================

class TestThresholds:
    """Tests to verify threshold constants are sensible."""

    def test_approve_higher_than_retry_min(self):
        # Assert
        assert EvaluationThresholds.APPROVE_QUALITY > EvaluationThresholds.RETRY_QUALITY_MIN

    def test_retry_base_delay_positive(self):
        assert EvaluationThresholds.RETRY_BASE_DELAY_MS > 0

    def test_retry_max_delay_greater_than_base(self):
        assert EvaluationThresholds.RETRY_MAX_DELAY_MS > EvaluationThresholds.RETRY_BASE_DELAY_MS

    def test_max_retries_positive(self):
        assert EvaluationThresholds.MAX_RETRIES > 0


# =============================================================================
# 20. BOUNDARY VERDICT TESTS
# =============================================================================

class TestBoundaryVerdicts:
    """Boundary condition tests for verdict thresholds."""

    def test_exactly_at_approve_threshold(self):
        # Arrange
        quality = QualityScore(overall=EvaluationThresholds.APPROVE_QUALITY)
        plan = PlanCompletionStatus(required_steps_met=True)

        # Act
        verdict = _determine_verdict(quality, plan, retry_count=0, max_retries=3)

        # Assert
        assert verdict.verdict == Verdict.APPROVE

    def test_just_below_approve_threshold(self):
        # Arrange
        quality = QualityScore(overall=EvaluationThresholds.APPROVE_QUALITY - 0.01)
        plan = PlanCompletionStatus(required_steps_met=True)

        # Act
        verdict = _determine_verdict(quality, plan, retry_count=0, max_retries=3)

        # Assert
        assert verdict.verdict != Verdict.APPROVE

    def test_at_retry_min_threshold(self):
        # Arrange
        quality = QualityScore(overall=EvaluationThresholds.RETRY_QUALITY_MIN)
        plan = PlanCompletionStatus(required_steps_met=True)

        # Act
        verdict = _determine_verdict(quality, plan, retry_count=0, max_retries=3)

        # Assert
        assert verdict.verdict == Verdict.RETRY  # At threshold, not below

    def test_below_retry_min_escalates(self):
        # Arrange
        quality = QualityScore(overall=EvaluationThresholds.RETRY_QUALITY_MIN - 0.01)
        plan = PlanCompletionStatus(required_steps_met=True)

        # Act
        verdict = _determine_verdict(quality, plan, retry_count=0, max_retries=3)

        # Assert
        assert verdict.verdict == Verdict.ESCALATE


# =============================================================================
# 21. HISTORY FILTER TESTS
# =============================================================================

class TestHistoryFilter:
    """Tests for history filter model."""

    def test_empty_filter(self):
        # Arrange / Act
        f = EvaluationHistoryFilter()

        # Assert
        assert f.task_id is None
        assert f.verdict is None

    def test_filter_with_verdict(self):
        # Arrange / Act
        f = EvaluationHistoryFilter(verdict=Verdict.APPROVE)

        # Assert
        assert f.verdict == Verdict.APPROVE

    def test_filter_with_quality_range(self):
        # Arrange / Act
        f = EvaluationHistoryFilter(min_quality=0.5, max_quality=0.9)

        # Assert
        assert f.min_quality == 0.5
        assert f.max_quality == 0.9
