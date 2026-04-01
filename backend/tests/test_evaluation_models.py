"""
Tests for evaluation_models.py — Pydantic bounds, QualityScore.compute_overall, constants.

Sentinel-value contract:
- Remove accuracy ge=0.0/le=1.0 → out-of-range scores stored → math broken → test fails
- Remove QualityScore default 0.0 → unset dimensions produce errors → test fails
- Remove custom-weights path → caller can't tune weighting → test fails
- Remove EvaluationThresholds.APPROVE_QUALITY 0.7 → approval threshold changes → routing wrong → test fails
- Remove HumanFeedbackInput.quality_override bounds → human can submit score of 5.0 → test fails
"""

from __future__ import annotations

import pytest
from uuid import UUID
from pydantic import ValidationError

from app.models.evaluation_models import (
    EvaluationHistoryFilter,
    EvaluationResult,
    EvaluationThresholds,
    EvaluationVerdict,
    HumanFeedbackInput,
    PlanCompletionStatus,
    QualityScore,
    RetryDecision,
    StepEvaluation,
    StepStatus,
    Verdict,
)
from uuid import uuid4


# ===========================================================================
# QualityScore — bounds
# ===========================================================================

class TestQualityScoreBounds:
    def test_accuracy_above_1_rejected(self):
        """
        SENTINEL — accuracy > 1.0 must raise ValidationError.
        Remove le=1.0 → invalid score stored → weighted average > 1 → downstream breaks → test fails.
        """
        with pytest.raises(ValidationError):
            QualityScore(accuracy=1.1)

    def test_accuracy_below_0_rejected(self):
        """accuracy < 0.0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            QualityScore(accuracy=-0.1)

    def test_completeness_bounds_enforced(self):
        """completeness out-of-range must raise ValidationError."""
        with pytest.raises(ValidationError):
            QualityScore(completeness=1.5)

    def test_relevance_bounds_enforced(self):
        """relevance out-of-range must raise ValidationError."""
        with pytest.raises(ValidationError):
            QualityScore(relevance=-0.01)

    def test_coherence_bounds_enforced(self):
        """coherence out-of-range must raise ValidationError."""
        with pytest.raises(ValidationError):
            QualityScore(coherence=2.0)

    def test_overall_bounds_enforced(self):
        """overall out-of-range must raise ValidationError."""
        with pytest.raises(ValidationError):
            QualityScore(overall=1.1)

    def test_boundary_values_accepted(self):
        """Exactly 0.0 and 1.0 must be accepted for all dimensions."""
        qs = QualityScore(accuracy=0.0, completeness=1.0, relevance=0.5, coherence=0.0)
        assert qs.accuracy == 0.0
        assert qs.completeness == 1.0

    def test_all_defaults_are_zero(self):
        """
        SENTINEL — all dimensions default to 0.0.
        Default to 0.5 → unscored output looks good → approval thresholds bypassed → test fails.
        """
        qs = QualityScore()
        assert qs.accuracy == 0.0
        assert qs.completeness == 0.0
        assert qs.relevance == 0.0
        assert qs.coherence == 0.0
        assert qs.overall == 0.0


# ===========================================================================
# QualityScore.compute_overall — weighted math
# ===========================================================================

class TestQualityScoreComputeOverall:
    def test_default_weights_formula(self):
        """
        SENTINEL — compute_overall must use accuracy:0.35, completeness:0.25, relevance:0.25, coherence:0.15.
        Change weights → approval boundary shifts → wrong tasks approved → test fails.
        """
        qs = QualityScore(accuracy=1.0, completeness=1.0, relevance=1.0, coherence=1.0)
        score = qs.compute_overall()
        assert score == pytest.approx(1.0)

    def test_weighted_average_correct(self):
        """
        SENTINEL — weighted math must be exact.
        Use simple average → different boundary → routing decisions change → test fails.
        """
        # accuracy=1.0 (weight 0.35), others=0.0 → score = 0.35/1.0 = 0.35
        qs = QualityScore(accuracy=1.0, completeness=0.0, relevance=0.0, coherence=0.0)
        score = qs.compute_overall()
        assert score == pytest.approx(0.35, abs=1e-6)

    def test_coherence_weight_is_015(self):
        """
        SENTINEL — coherence weight must be 0.15.
        Change to 0.25 → total weights = 1.10 → normalised score wrong → test fails.
        """
        # coherence=1.0 (weight 0.15), others=0.0 → score = 0.15
        qs = QualityScore(accuracy=0.0, completeness=0.0, relevance=0.0, coherence=1.0)
        score = qs.compute_overall()
        assert score == pytest.approx(0.15, abs=1e-6)

    def test_custom_weights_override_defaults(self):
        """
        SENTINEL — caller-supplied weights must override defaults.
        Ignore custom weights → callers can't tune scoring → test fails.
        """
        qs = QualityScore(accuracy=1.0, completeness=0.0, relevance=0.0, coherence=0.0)
        # Give accuracy all the weight
        score = qs.compute_overall(weights={"accuracy": 1.0, "completeness": 0.0, "relevance": 0.0, "coherence": 0.0})
        assert score == pytest.approx(1.0)

    def test_zero_total_weight_returns_zero(self):
        """All-zero custom weights must return 0.0 without dividing by zero."""
        qs = QualityScore(accuracy=1.0)
        score = qs.compute_overall(weights={"accuracy": 0.0, "completeness": 0.0, "relevance": 0.0, "coherence": 0.0})
        assert score == 0.0

    def test_compute_overall_updates_overall_field(self):
        """compute_overall must update the .overall attribute in-place."""
        qs = QualityScore(accuracy=0.8, completeness=0.6, relevance=0.7, coherence=0.9)
        returned = qs.compute_overall()
        assert qs.overall == pytest.approx(returned)
        assert qs.overall > 0.0


# ===========================================================================
# EvaluationThresholds — constant values
# ===========================================================================

class TestEvaluationThresholds:
    def test_approve_quality_is_07(self):
        """
        SENTINEL — APPROVE_QUALITY must be 0.7.
        Change to 0.5 → lower-quality outputs approved → test fails.
        """
        assert EvaluationThresholds.APPROVE_QUALITY == pytest.approx(0.7)

    def test_retry_quality_min_is_03(self):
        """
        SENTINEL — RETRY_QUALITY_MIN must be 0.3.
        Change to 0.5 → escalation triggers too early → test fails.
        """
        assert EvaluationThresholds.RETRY_QUALITY_MIN == pytest.approx(0.3)

    def test_max_retries_is_3(self):
        """
        SENTINEL — MAX_RETRIES must be 3.
        Change to 1 → tasks escalate immediately → no retry opportunity → test fails.
        """
        assert EvaluationThresholds.MAX_RETRIES == 3

    def test_retry_base_delay_ms(self):
        """RETRY_BASE_DELAY_MS must be 1000 (1 second)."""
        assert EvaluationThresholds.RETRY_BASE_DELAY_MS == 1000

    def test_retry_max_delay_ms(self):
        """RETRY_MAX_DELAY_MS must be 30000 (30 seconds)."""
        assert EvaluationThresholds.RETRY_MAX_DELAY_MS == 30000


# ===========================================================================
# HumanFeedbackInput — quality_override bounds
# ===========================================================================

class TestHumanFeedbackInput:
    def test_quality_override_above_1_rejected(self):
        """
        SENTINEL — quality_override > 1.0 must raise ValidationError.
        Remove le=1.0 → human can set score=5 → scoring breakdown → test fails.
        """
        with pytest.raises(ValidationError):
            HumanFeedbackInput(
                evaluation_id=uuid4(),
                agree_with_verdict=True,
                quality_override=1.5,
            )

    def test_quality_override_below_0_rejected(self):
        """quality_override < 0.0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            HumanFeedbackInput(
                evaluation_id=uuid4(),
                agree_with_verdict=False,
                quality_override=-0.1,
            )

    def test_quality_override_boundary_accepted(self):
        """0.0 and 1.0 must be accepted as quality_override."""
        low = HumanFeedbackInput(evaluation_id=uuid4(), agree_with_verdict=True, quality_override=0.0)
        high = HumanFeedbackInput(evaluation_id=uuid4(), agree_with_verdict=True, quality_override=1.0)
        assert low.quality_override == 0.0
        assert high.quality_override == 1.0

    def test_no_quality_override_defaults_none(self):
        """quality_override defaults to None when not supplied."""
        fb = HumanFeedbackInput(evaluation_id=uuid4(), agree_with_verdict=True)
        assert fb.quality_override is None


# ===========================================================================
# EvaluationHistoryFilter — bounds
# ===========================================================================

class TestEvaluationHistoryFilter:
    def test_min_quality_above_1_rejected(self):
        """min_quality > 1.0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            EvaluationHistoryFilter(min_quality=1.1)

    def test_max_quality_below_0_rejected(self):
        """max_quality < 0.0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            EvaluationHistoryFilter(max_quality=-0.5)

    def test_valid_filter_accepted(self):
        """Valid filter with all fields must be accepted."""
        f = EvaluationHistoryFilter(
            verdict=Verdict.APPROVE,
            min_quality=0.5,
            max_quality=0.9,
        )
        assert f.verdict == Verdict.APPROVE
        assert f.min_quality == pytest.approx(0.5)


# ===========================================================================
# EvaluationResult — auto-fields
# ===========================================================================

class TestEvaluationResult:
    def _make_verdict(self) -> EvaluationVerdict:
        return EvaluationVerdict(verdict=Verdict.APPROVE, reasoning="SENTINEL_REASON")

    def test_id_auto_generated(self):
        """
        SENTINEL — id must be auto-generated UUID.
        Default None → downstream can't look up evaluation by ID → test fails.
        """
        result = EvaluationResult(task_id=uuid4(), verdict=self._make_verdict())
        assert isinstance(result.id, UUID)

    def test_created_at_auto_populated(self):
        """created_at must be set automatically."""
        result = EvaluationResult(task_id=uuid4(), verdict=self._make_verdict())
        assert result.created_at is not None

    def test_default_quality_score_all_zeros(self):
        """quality_score defaults to all-zero QualityScore."""
        result = EvaluationResult(task_id=uuid4(), verdict=self._make_verdict())
        assert result.quality_score.accuracy == 0.0
        assert result.quality_score.overall == 0.0


# ===========================================================================
# RetryDecision — delay_ms bounds
# ===========================================================================

class TestRetryDecision:
    def test_delay_ms_negative_rejected(self):
        """
        SENTINEL — delay_ms < 0 must raise ValidationError.
        Remove ge=0 → negative delay stored → clients sleep negative time → test fails.
        """
        with pytest.raises(ValidationError):
            RetryDecision(should_retry=True, delay_ms=-1)

    def test_zero_delay_accepted(self):
        """delay_ms=0 must be accepted (immediate retry)."""
        rd = RetryDecision(should_retry=True, delay_ms=0)
        assert rd.delay_ms == 0

    def test_defaults_sensible(self):
        """should_retry=False, escalate=False, delay_ms=0 defaults."""
        rd = RetryDecision(should_retry=False)
        assert rd.escalate is False
        assert rd.delay_ms == 0


# ===========================================================================
# PlanCompletionStatus — completion_rate bounds
# ===========================================================================

class TestPlanCompletionStatus:
    def test_completion_rate_above_1_rejected(self):
        """completion_rate > 1.0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            PlanCompletionStatus(completion_rate=1.1)

    def test_completion_rate_below_0_rejected(self):
        """completion_rate < 0.0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            PlanCompletionStatus(completion_rate=-0.1)

    def test_completion_rate_boundary_accepted(self):
        """0.0 and 1.0 must be accepted."""
        low = PlanCompletionStatus(completion_rate=0.0)
        high = PlanCompletionStatus(completion_rate=1.0)
        assert low.completion_rate == 0.0
        assert high.completion_rate == 1.0

    def test_required_steps_met_defaults_false(self):
        """
        SENTINEL — required_steps_met defaults to False.
        Default True → incomplete tasks falsely flagged as complete → approval issued → test fails.
        """
        status = PlanCompletionStatus()
        assert status.required_steps_met is False
