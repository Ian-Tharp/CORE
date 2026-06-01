"""
Tests for evaluation_service.py — pure scoring and decision logic.

Sentinel-value contract:
- Remove empty-output check in _heuristic_accuracy → 0.4 returned for empty output → test fails
- Remove error-rate contribution → all-failed steps score 1.0 → test fails
- Remove completeness fraction → always 0.5 → test fails
- Remove stop-word filtering → common words inflate relevance → test fails
- Remove coherence word-count bonus → long output scored same as 1-word → test fails
- Remove APPROVE threshold check → any quality score gets approved → test fails
- Remove ESCALATE retry-count check → retries run forever → test fails
- Remove ESCALATE low-quality check → very-low-quality never escalates → test fails
- Remove REFINE branch → plan failures always RETRY not REFINE → test fails
- Remove exponential backoff → delay stays at base forever → test fails
- Remove compute_overall weighted math → overall always 0.0 → test fails
"""

from __future__ import annotations

import pytest

from app.models.evaluation_models import (
    EvaluationThresholds,
    EvaluationVerdict,
    PlanCompletionStatus,
    PlanStep,
    QualityScore,
    RetryDecision,
    StepResult,
    StepStatus,
    Verdict,
)
from app.services.evaluation_service import (
    _build_retry_decision,
    _check_plan_completion,
    _determine_verdict,
    _evaluate_step_quality,
    _generate_feedback,
    _heuristic_accuracy,
    _heuristic_coherence,
    _heuristic_completeness,
    _heuristic_relevance,
    _step_feedback,
    _step_meets_criteria,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_step(
    index: int = 0, required: bool = True, description: str = "do something"
) -> PlanStep:
    return PlanStep(step_index=index, description=description, required=required)


def _make_result(
    index: int = 0,
    status: StepStatus = StepStatus.COMPLETED,
    output: str | None = None,
    error: str | None = None,
) -> StepResult:
    return StepResult(step_index=index, status=status, output=output, error=error)


def _make_quality(
    accuracy: float = 0.8,
    completeness: float = 0.8,
    relevance: float = 0.8,
    coherence: float = 0.8,
) -> QualityScore:
    q = QualityScore(
        accuracy=accuracy,
        completeness=completeness,
        relevance=relevance,
        coherence=coherence,
    )
    q.compute_overall()
    return q


def _make_completion(
    required_met: bool = True, rate: float = 1.0, total: int = 1, completed: int = 1
) -> PlanCompletionStatus:
    return PlanCompletionStatus(
        total_steps=total,
        completed_steps=completed,
        completion_rate=rate,
        required_steps_met=required_met,
    )


def _make_verdict(
    v: Verdict = Verdict.APPROVE, improvements: list | None = None
) -> EvaluationVerdict:
    return EvaluationVerdict(
        verdict=v,
        reasoning="test",
        confidence=0.8,
        suggested_improvements=improvements or [],
    )


# ===========================================================================
# QualityScore.compute_overall
# ===========================================================================


class TestQualityScoreComputeOverall:
    def test_weighted_average_correct(self):
        """
        SENTINEL — overall must be weighted average (0.35/0.25/0.25/0.15).
        Return simple mean → incorrect overall → approve/retry threshold wrong → test fails.
        """
        q = QualityScore(accuracy=1.0, completeness=0.0, relevance=0.0, coherence=0.0)
        overall = q.compute_overall()
        # Only accuracy contributes: 1.0 * 0.35 / 1.0
        assert overall == pytest.approx(0.35, abs=0.001)

    def test_all_ones_gives_one(self):
        """All perfect scores must yield overall=1.0."""
        q = QualityScore(accuracy=1.0, completeness=1.0, relevance=1.0, coherence=1.0)
        q.compute_overall()
        assert q.overall == pytest.approx(1.0)

    def test_all_zeros_gives_zero(self):
        """All zero scores must yield overall=0.0."""
        q = QualityScore(accuracy=0.0, completeness=0.0, relevance=0.0, coherence=0.0)
        q.compute_overall()
        assert q.overall == pytest.approx(0.0)

    def test_returns_overall_value(self):
        """compute_overall() must return the same value stored in self.overall."""
        q = QualityScore(accuracy=0.5, completeness=0.5, relevance=0.5, coherence=0.5)
        ret = q.compute_overall()
        assert ret == q.overall

    def test_custom_weights_applied(self):
        """Custom weights must override defaults."""
        q = QualityScore(accuracy=1.0, completeness=0.0, relevance=0.0, coherence=0.0)
        # With equal weights 0.25 each, all-accuracy-1 result = 0.25
        overall = q.compute_overall(
            weights={
                "accuracy": 0.25,
                "completeness": 0.25,
                "relevance": 0.25,
                "coherence": 0.25,
            }
        )
        assert overall == pytest.approx(0.25, abs=0.001)


# ===========================================================================
# _heuristic_accuracy
# ===========================================================================


class TestHeuristicAccuracy:
    def test_empty_output_returns_zero(self):
        """
        SENTINEL — empty output must return 0.0.
        Skip empty check → 0.4 returned for empty → approve logic confused → test fails.
        """
        assert _heuristic_accuracy("", []) == pytest.approx(0.0)

    def test_whitespace_only_returns_zero(self):
        """Whitespace-only output is treated as empty."""
        assert _heuristic_accuracy("   \t\n", []) == pytest.approx(0.0)

    def test_non_empty_no_steps_returns_base(self):
        """
        SENTINEL — non-empty output with no steps returns 0.4 base.
        Use 0.5 → threshold boundary changes → test fails.
        """
        assert _heuristic_accuracy("SENTINEL_OUTPUT", []) == pytest.approx(0.4)

    def test_all_completed_steps_gives_max(self):
        """
        SENTINEL — all-completed steps must yield 1.0 (0.4 + 0.6).
        Omit non-error-rate bonus → stays at 0.4 regardless → test fails.
        """
        steps = [_make_result(i, StepStatus.COMPLETED) for i in range(3)]
        score = _heuristic_accuracy("SENTINEL", steps)
        assert score == pytest.approx(1.0)

    def test_all_failed_steps_reduces_score(self):
        """
        SENTINEL — all failed steps must give accuracy close to 0.4 (base only).
        Skip error penalty → failed steps score 1.0 → test fails.
        """
        steps = [_make_result(i, StepStatus.FAILED) for i in range(3)]
        score = _heuristic_accuracy("SENTINEL", steps)
        assert score == pytest.approx(0.4)

    def test_mixed_steps_partial_score(self):
        """Half failed → score between 0.4 and 1.0."""
        steps = [
            _make_result(0, StepStatus.COMPLETED),
            _make_result(1, StepStatus.FAILED),
        ]
        score = _heuristic_accuracy("SENTINEL", steps)
        assert 0.4 < score < 1.0

    def test_score_capped_at_one(self):
        """Score must never exceed 1.0."""
        steps = [_make_result(i, StepStatus.COMPLETED) for i in range(10)]
        score = _heuristic_accuracy("big output" * 100, steps)
        assert score <= 1.0


# ===========================================================================
# _heuristic_completeness
# ===========================================================================


class TestHeuristicCompleteness:
    def test_no_plan_returns_half(self):
        """
        SENTINEL — no plan steps must return 0.5.
        Return 0.0 → low-quality score triggers unnecessary retry → test fails.
        """
        assert _heuristic_completeness([], []) == pytest.approx(0.5)

    def test_all_completed_returns_one(self):
        """
        SENTINEL — all steps completed must return 1.0.
        Use count instead of fraction → multi-step tasks score > 1 → test fails.
        """
        steps = [_make_step(i) for i in range(3)]
        results = [_make_result(i, StepStatus.COMPLETED) for i in range(3)]
        assert _heuristic_completeness(steps, results) == pytest.approx(1.0)

    def test_partial_steps_contribute_half(self):
        """
        SENTINEL — partial steps count as 0.5 in the completion fraction.
        Count partial as 0 → partial work ignored → test fails.
        """
        steps = [_make_step(0), _make_step(1)]
        results = [
            _make_result(0, StepStatus.PARTIAL),
            _make_result(1, StepStatus.PARTIAL),
        ]
        # (0 + 0.5*2) / 2 = 0.5
        assert _heuristic_completeness(steps, results) == pytest.approx(0.5)

    def test_missing_result_treated_as_zero(self):
        """Step with no result counts as 0 completed."""
        steps = [_make_step(0), _make_step(1)]
        results = [_make_result(0, StepStatus.COMPLETED)]  # step 1 missing
        # (1 + 0) / 2 = 0.5
        assert _heuristic_completeness(steps, results) == pytest.approx(0.5)

    def test_failed_steps_count_as_zero(self):
        """Failed steps must not contribute to completion score."""
        steps = [_make_step(0)]
        results = [_make_result(0, StepStatus.FAILED)]
        assert _heuristic_completeness(steps, results) == pytest.approx(0.0)


# ===========================================================================
# _heuristic_relevance
# ===========================================================================


class TestHeuristicRelevance:
    def test_empty_intent_returns_zero(self):
        """Empty intent must return 0.0."""
        assert _heuristic_relevance("", "SENTINEL") == pytest.approx(0.0)

    def test_empty_output_returns_zero(self):
        """Empty output must return 0.0."""
        assert _heuristic_relevance("SENTINEL", "") == pytest.approx(0.0)

    def test_full_keyword_overlap_returns_one(self):
        """
        SENTINEL — identical intent/output (non-stop words) must return 1.0.
        Use character overlap → different result → test fails.
        """
        text = "machine learning optimization algorithm"
        assert _heuristic_relevance(text, text) == pytest.approx(1.0)

    def test_no_overlap_returns_zero(self):
        """
        SENTINEL — no shared keywords returns 0.0.
        Return 0.5 default → irrelevant output approved → test fails.
        """
        assert _heuristic_relevance(
            "machine learning", "cooking recipes"
        ) == pytest.approx(0.0)

    def test_stop_words_excluded_from_overlap(self):
        """
        SENTINEL — stop words must not count as meaningful overlap.
        Include stop words → 'the is a' has full overlap → test fails.
        """
        # Intent and output share only stop words
        score = _heuristic_relevance("the is a to", "the is a to")
        # After removing stop words, intent_words is empty → returns 0.5
        assert score == pytest.approx(0.5)

    def test_partial_overlap_proportional(self):
        """Partial keyword match returns a value in (0, 1)."""
        intent = "machine learning algorithm data"
        output = "machine learning basics"
        score = _heuristic_relevance(intent, output)
        assert 0.0 < score < 1.0

    def test_score_capped_at_one(self):
        """Score must never exceed 1.0."""
        assert _heuristic_relevance("word", "word word word") <= 1.0


# ===========================================================================
# _heuristic_coherence
# ===========================================================================


class TestHeuristicCoherence:
    def test_empty_output_returns_zero(self):
        """
        SENTINEL — empty output must return 0.0.
        Return 0.3 → empty output gets coherence credit → test fails.
        """
        assert _heuristic_coherence("") == pytest.approx(0.0)

    def test_single_word_returns_base(self):
        """
        SENTINEL — single word returns base score 0.3.
        Skip base → one-word output scores 0 → test fails.
        """
        assert _heuristic_coherence("SENTINEL") == pytest.approx(0.3)

    def test_ten_or_more_words_adds_bonus(self):
        """
        SENTINEL — output with >= 10 words must score > 0.3.
        Remove word-count bonus → long, well-structured text scores same as 1 word → test fails.
        """
        output = " ".join(["word"] * 10)
        assert _heuristic_coherence(output) > 0.3

    def test_fifty_or_more_words_adds_extra_bonus(self):
        """50+ words must score higher than 10-49 words."""
        output_10 = " ".join(["word"] * 10)
        output_50 = " ".join(["word"] * 50)
        assert _heuristic_coherence(output_50) > _heuristic_coherence(output_10)

    def test_multiple_sentences_adds_bonus(self):
        """
        SENTINEL — output with 2+ sentences must score higher than identical text without periods.
        Remove sentence bonus → structured output ranks same as unstructured → test fails.
        """
        # 2 words — below word-count bonus threshold (10), so only base or sentence bonus applies
        no_sentences = "SENTINEL output"  # base=0.3, no sentence bonus
        two_sentences = "SENTINEL. output."  # base=0.3 + sentence bonus 0.2 = 0.5
        assert _heuristic_coherence(two_sentences) > _heuristic_coherence(no_sentences)

    def test_score_capped_at_one(self):
        """Very long, well-structured output must not exceed 1.0."""
        output = "- " + ". ".join(["SENTINEL word word word word word"] * 20) + "\n"
        assert _heuristic_coherence(output) <= 1.0


# ===========================================================================
# _evaluate_step_quality
# ===========================================================================


class TestEvaluateStepQuality:
    def test_no_result_returns_zero(self):
        """
        SENTINEL — None result must return 0.0.
        Return 0.5 → missing results inflate quality → test fails.
        """
        assert _evaluate_step_quality(_make_step(), None) == pytest.approx(0.0)

    def test_completed_with_output_returns_one(self):
        """
        SENTINEL — completed step with non-empty output must return 1.0.
        Return 0.8 always → output bonus never applied → test fails.
        """
        result = _make_result(0, StepStatus.COMPLETED, output="SENTINEL_OUTPUT")
        assert _evaluate_step_quality(_make_step(), result) == pytest.approx(1.0)

    def test_completed_without_output_returns_point_eight(self):
        """
        SENTINEL — completed step without output must return 0.8.
        Always return 1.0 → missing output not penalised → test fails.
        """
        result = _make_result(0, StepStatus.COMPLETED, output=None)
        assert _evaluate_step_quality(_make_step(), result) == pytest.approx(0.8)

    def test_partial_returns_point_five(self):
        """
        SENTINEL — partial step must return 0.5.
        Return 0.0 → partial work ignored → test fails.
        """
        result = _make_result(0, StepStatus.PARTIAL)
        assert _evaluate_step_quality(_make_step(), result) == pytest.approx(0.5)

    def test_failed_returns_point_one(self):
        """
        SENTINEL — failed step must return 0.1.
        Return 0.0 → failed step same as missing → test fails.
        """
        result = _make_result(0, StepStatus.FAILED)
        assert _evaluate_step_quality(_make_step(), result) == pytest.approx(0.1)

    def test_skipped_required_returns_zero(self):
        """
        SENTINEL — skipping a required step returns 0.0.
        Return 0.3 always → required-skip treated same as optional → test fails.
        """
        step = _make_step(required=True)
        result = _make_result(0, StepStatus.SKIPPED)
        assert _evaluate_step_quality(step, result) == pytest.approx(0.0)

    def test_skipped_optional_returns_point_three(self):
        """
        SENTINEL — skipping an optional step returns 0.3.
        Return 0.0 → optional skips penalised same as required → test fails.
        """
        step = _make_step(required=False)
        result = _make_result(0, StepStatus.SKIPPED)
        assert _evaluate_step_quality(step, result) == pytest.approx(0.3)


# ===========================================================================
# _step_meets_criteria
# ===========================================================================


class TestStepMeetsCriteria:
    def test_no_result_returns_false(self):
        """No result → criteria not met."""
        assert _step_meets_criteria(_make_step(), None, 0.9) is False

    def test_failed_never_meets_criteria(self):
        """
        SENTINEL — failed step must return False regardless of quality.
        Remove failed check → failed+high-quality flagged as met → test fails.
        """
        result = _make_result(0, StepStatus.FAILED)
        assert _step_meets_criteria(_make_step(), result, 0.9) is False

    def test_completed_high_quality_meets_criteria(self):
        """
        SENTINEL — completed step with quality >= 0.6 must return True.
        Use quality >= 0.8 threshold → most completed steps fail → test fails.
        """
        result = _make_result(0, StepStatus.COMPLETED)
        assert _step_meets_criteria(_make_step(), result, 0.6) is True

    def test_completed_low_quality_fails_criteria(self):
        """Completed step with quality < 0.6 must return False."""
        result = _make_result(0, StepStatus.COMPLETED)
        assert _step_meets_criteria(_make_step(), result, 0.59) is False

    def test_partial_optional_adequate_quality_meets(self):
        """
        SENTINEL — partial optional step with quality >= 0.5 must return True.
        Require COMPLETED for criteria → partial work never acceptable → test fails.
        """
        step = _make_step(required=False)
        result = _make_result(0, StepStatus.PARTIAL)
        assert _step_meets_criteria(step, result, 0.5) is True

    def test_partial_required_never_meets(self):
        """Partial required step must not meet criteria."""
        step = _make_step(required=True)
        result = _make_result(0, StepStatus.PARTIAL)
        assert _step_meets_criteria(step, result, 0.9) is False


# ===========================================================================
# _step_feedback
# ===========================================================================


class TestStepFeedback:
    def test_no_result_mentions_not_started(self):
        """
        SENTINEL — no result must produce 'Not started' message.
        Return empty string → caller can't distinguish not-started from completed → test fails.
        """
        fb = _step_feedback(_make_step(0), None, 0.0, False)
        assert "not started" in fb.lower() or "Not started" in fb

    def test_success_message_includes_quality(self):
        """
        SENTINEL — success feedback must include quality score.
        Return generic message → no numeric feedback → test fails.
        """
        result = _make_result(0, StepStatus.COMPLETED)
        fb = _step_feedback(_make_step(0), result, 0.95, True)
        assert "0.95" in fb

    def test_failure_message_includes_error(self):
        """
        SENTINEL — failure feedback must include the error message.
        Return generic failure message → root cause lost → test fails.
        """
        result = _make_result(0, StepStatus.FAILED, error="SENTINEL_ERROR_MSG")
        fb = _step_feedback(_make_step(0), result, 0.1, False)
        assert "SENTINEL_ERROR_MSG" in fb

    def test_skipped_message_says_skipped(self):
        """Skipped step feedback must mention 'skipped'."""
        result = _make_result(0, StepStatus.SKIPPED)
        fb = _step_feedback(_make_step(0), result, 0.0, False)
        assert "kipped" in fb  # 'Skipped' or 'skipped'

    def test_partial_message_mentions_partial(self):
        """Partial feedback must mention 'partial' or quality."""
        result = _make_result(0, StepStatus.PARTIAL)
        fb = _step_feedback(_make_step(0), result, 0.5, False)
        assert "partial" in fb.lower() or "0.50" in fb


# ===========================================================================
# _check_plan_completion
# ===========================================================================


class TestCheckPlanCompletion:
    def test_no_steps_returns_complete(self):
        """
        SENTINEL — empty plan must return completion_rate=1.0 and required_steps_met=True.
        Return rate=0.0 → empty plan treated as failed → test fails.
        """
        status = _check_plan_completion([], [])
        assert status.completion_rate == pytest.approx(1.0)
        assert status.required_steps_met is True

    def test_all_completed_full_rate(self):
        """All steps completed → completion_rate=1.0."""
        steps = [_make_step(i) for i in range(3)]
        results = [_make_result(i, StepStatus.COMPLETED) for i in range(3)]
        status = _check_plan_completion(steps, results)
        assert status.completion_rate == pytest.approx(1.0)
        assert status.required_steps_met is True

    def test_failed_required_step_breaks_required_met(self):
        """
        SENTINEL — failing a required step must set required_steps_met=False.
        Skip required tracking → required failures invisible → test fails.
        """
        steps = [_make_step(0, required=True)]
        results = [_make_result(0, StepStatus.FAILED)]
        status = _check_plan_completion(steps, results)
        assert status.required_steps_met is False

    def test_failed_optional_step_keeps_required_met(self):
        """Failing an optional step must not break required_steps_met."""
        steps = [_make_step(0, required=False)]
        results = [_make_result(0, StepStatus.FAILED)]
        status = _check_plan_completion(steps, results)
        assert status.required_steps_met is True

    def test_partial_adds_half_to_completion(self):
        """
        SENTINEL — partial steps count as 0.5 in completion_rate.
        Count partial as 1 → inflated completion → test fails.
        """
        steps = [_make_step(0), _make_step(1)]
        results = [
            _make_result(0, StepStatus.PARTIAL),
            _make_result(1, StepStatus.PARTIAL),
        ]
        status = _check_plan_completion(steps, results)
        assert status.completion_rate == pytest.approx(0.5)

    def test_total_steps_counted_correctly(self):
        """total_steps must equal the number of plan steps."""
        steps = [_make_step(i) for i in range(5)]
        status = _check_plan_completion(steps, [])
        assert status.total_steps == 5

    def test_step_evaluations_generated(self):
        """step_evaluations list must have one entry per plan step."""
        steps = [_make_step(i) for i in range(3)]
        results = [_make_result(i, StepStatus.COMPLETED) for i in range(3)]
        status = _check_plan_completion(steps, results)
        assert len(status.step_evaluations) == 3


# ===========================================================================
# _determine_verdict
# ===========================================================================


class TestDetermineVerdict:
    def test_high_quality_all_steps_met_approves(self):
        """
        SENTINEL — quality >= 0.7 and required_steps_met must produce APPROVE.
        Remove approve branch → every call retries regardless of quality → test fails.
        """
        quality = _make_quality(0.9, 0.9, 0.9, 0.9)
        completion = _make_completion(required_met=True)
        verdict = _determine_verdict(quality, completion, retry_count=0, max_retries=3)
        assert verdict.verdict == Verdict.APPROVE

    def test_max_retries_exhausted_escalates(self):
        """
        SENTINEL — retry_count >= max_retries must produce ESCALATE.
        Skip retry-count check → infinite retry loop → test fails.
        """
        quality = _make_quality(0.5, 0.5, 0.5, 0.5)
        completion = _make_completion(required_met=True)
        verdict = _determine_verdict(quality, completion, retry_count=3, max_retries=3)
        assert verdict.verdict == Verdict.ESCALATE

    def test_below_min_quality_escalates(self):
        """
        SENTINEL — quality < RETRY_QUALITY_MIN (0.3) must produce ESCALATE.
        Remove low-quality check → very-low-quality tasks retry instead of escalating → test fails.
        """
        quality = _make_quality(0.0, 0.0, 0.0, 0.0)  # overall well below 0.3
        completion = _make_completion(required_met=True)
        verdict = _determine_verdict(quality, completion, retry_count=0, max_retries=3)
        assert verdict.verdict == Verdict.ESCALATE

    def test_required_steps_missing_decent_quality_refines(self):
        """
        SENTINEL — required steps not met + quality >= 0.5 must produce REFINE.
        Fall through to RETRY → plan problems never refined → test fails.
        """
        quality = _make_quality(0.6, 0.6, 0.6, 0.6)  # overall ~0.6 (above 0.5)
        completion = _make_completion(required_met=False)
        verdict = _determine_verdict(quality, completion, retry_count=0, max_retries=3)
        assert verdict.verdict == Verdict.REFINE

    def test_medium_quality_retries(self):
        """
        SENTINEL — medium quality (0.3–0.69), steps met, retries remaining → RETRY.
        Return ESCALATE always → never retries → test fails.
        """
        quality = _make_quality(0.5, 0.5, 0.5, 0.5)  # overall ~0.5
        completion = _make_completion(required_met=True)
        verdict = _determine_verdict(quality, completion, retry_count=0, max_retries=3)
        assert verdict.verdict == Verdict.RETRY

    def test_approve_includes_confidence(self):
        """Approve verdict must have a positive confidence."""
        quality = _make_quality(0.9, 0.9, 0.9, 0.9)
        completion = _make_completion(required_met=True)
        verdict = _determine_verdict(quality, completion, 0, 3)
        assert verdict.confidence > 0.0

    def test_low_accuracy_adds_improvement_suggestion(self):
        """
        SENTINEL — accuracy < 0.6 must add 'accuracy' improvement suggestion.
        Skip per-dimension suggestions → users don't know what to fix → test fails.
        """
        quality = _make_quality(
            accuracy=0.3, completeness=0.8, relevance=0.8, coherence=0.8
        )
        completion = _make_completion(required_met=True)
        verdict = _determine_verdict(quality, completion, 0, 3)
        assert any("accuracy" in s.lower() for s in verdict.suggested_improvements)


# ===========================================================================
# _build_retry_decision
# ===========================================================================


class TestBuildRetryDecision:
    def test_approve_verdict_no_retry(self):
        """
        SENTINEL — APPROVE verdict must produce should_retry=False.
        Ignore verdict → all evaluations retry → test fails.
        """
        v = _make_verdict(Verdict.APPROVE)
        q = _make_quality()
        decision = _build_retry_decision(v, q, retry_count=0, max_retries=3)
        assert decision.should_retry is False

    def test_escalate_verdict_sets_escalate_flag(self):
        """
        SENTINEL — ESCALATE verdict must set escalate=True and should_retry=False.
        Treat ESCALATE as retry → human review never triggered → test fails.
        """
        v = _make_verdict(Verdict.ESCALATE)
        q = _make_quality()
        decision = _build_retry_decision(v, q, retry_count=3, max_retries=3)
        assert decision.escalate is True
        assert decision.should_retry is False

    def test_retry_verdict_sets_should_retry(self):
        """
        SENTINEL — RETRY verdict must produce should_retry=True.
        Return should_retry=False → tasks never retried → test fails.
        """
        v = _make_verdict(Verdict.RETRY)
        q = _make_quality(0.5, 0.5, 0.5, 0.5)
        decision = _build_retry_decision(v, q, retry_count=0, max_retries=3)
        assert decision.should_retry is True

    def test_exponential_backoff_increases_delay(self):
        """
        SENTINEL — delay must increase with retry_count (exponential backoff).
        Use flat delay → retries hammer the system → test fails.
        """
        v = _make_verdict(Verdict.RETRY)
        q = _make_quality(0.5, 0.5, 0.5, 0.5)
        delay_0 = _build_retry_decision(v, q, retry_count=0, max_retries=5).delay_ms
        delay_1 = _build_retry_decision(v, q, retry_count=1, max_retries=5).delay_ms
        assert delay_1 > delay_0

    def test_delay_capped_at_max(self):
        """
        SENTINEL — delay must be capped at RETRY_MAX_DELAY_MS.
        Remove cap → delay overflows → test fails.
        """
        v = _make_verdict(Verdict.RETRY)
        q = _make_quality(0.5, 0.5, 0.5, 0.5)
        decision = _build_retry_decision(v, q, retry_count=50, max_retries=100)
        assert decision.delay_ms <= EvaluationThresholds.RETRY_MAX_DELAY_MS

    def test_retry_count_incremented(self):
        """
        SENTINEL — returned retry_count must be input retry_count + 1.
        Return same count → retry_count never advances → test fails.
        """
        v = _make_verdict(Verdict.RETRY)
        q = _make_quality(0.5, 0.5, 0.5, 0.5)
        decision = _build_retry_decision(v, q, retry_count=2, max_retries=5)
        assert decision.retry_count == 3

    def test_low_accuracy_adds_focus_adjustment(self):
        """
        SENTINEL — quality.accuracy < 0.5 must add 'accuracy' to adjustments.
        Skip per-dimension adjustments → retries don't know what to improve → test fails.
        """
        v = _make_verdict(Verdict.RETRY)
        q = _make_quality(accuracy=0.2, completeness=0.8, relevance=0.8, coherence=0.8)
        decision = _build_retry_decision(v, q, retry_count=0, max_retries=3)
        assert decision.adjustments.get("focus") == "accuracy"


# ===========================================================================
# _generate_feedback
# ===========================================================================


class TestGenerateFeedback:
    def test_verdict_included_in_feedback(self):
        """
        SENTINEL — verdict string must appear in feedback.
        Return empty string → caller has no text to show → test fails.
        """
        quality = _make_quality()
        completion = _make_completion()
        verdict = _make_verdict(Verdict.APPROVE)
        fb = _generate_feedback(quality, completion, verdict)
        assert "APPROVE" in fb.upper() or "approve" in fb.lower()

    def test_overall_quality_score_included(self):
        """
        SENTINEL — numeric quality score must appear in feedback.
        Use descriptive text only → no numeric reference → test fails.
        """
        quality = _make_quality(0.85, 0.85, 0.85, 0.85)
        completion = _make_completion()
        verdict = _make_verdict()
        fb = _generate_feedback(quality, completion, verdict)
        # The overall score is some computed float; check it's there
        assert "0." in fb  # some float appears

    def test_plan_completion_stats_included_when_steps_exist(self):
        """
        SENTINEL — completed_steps / total_steps must appear when plan has steps.
        Skip plan section → completion stats hidden → test fails.
        """
        quality = _make_quality()
        completion = _make_completion(total=3, completed=2, rate=2 / 3)
        verdict = _make_verdict()
        fb = _generate_feedback(quality, completion, verdict)
        assert "3" in fb  # total_steps=3 appears somewhere

    def test_improvements_listed_when_present(self):
        """
        SENTINEL — improvement suggestions must appear when populated.
        Skip improvements section → retry guidance lost → test fails.
        """
        quality = _make_quality()
        completion = _make_completion()
        verdict = _make_verdict(improvements=["SENTINEL_IMPROVEMENT"])
        fb = _generate_feedback(quality, completion, verdict)
        assert "SENTINEL_IMPROVEMENT" in fb
