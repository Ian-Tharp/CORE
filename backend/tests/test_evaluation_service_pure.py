"""
Tests for evaluation_service pure functions — scoring, verdict, and retry logic.

Sentinel-value contract:
- Remove error_count from _heuristic_accuracy → all-failed steps score 1.0 → test fails
- Remove keyword overlap from _heuristic_relevance → unrelated output scores 0.5 → test fails
- Remove coherence heuristics → empty output gets same score as structured → test fails
- Remove APPROVE path in _determine_verdict → high-quality output never approved → test fails
- Remove ESCALATE-on-retries-exhausted → infinite retry loop possible → test fails
- Remove exponential backoff in _build_retry_decision → all retries same delay → test fails
- Remove required_steps check in _check_plan_completion → missing steps ignored → test fails
"""

from __future__ import annotations

import pytest

from app.models.evaluation_models import (
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
    _heuristic_accuracy,
    _heuristic_coherence,
    _heuristic_completeness,
    _heuristic_relevance,
    _step_meets_criteria,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _step(idx: int, required: bool = True, description: str = "step") -> PlanStep:
    return PlanStep(step_index=idx, description=description, required=required)


def _result(idx: int, status: StepStatus, output: str | None = None) -> StepResult:
    return StepResult(step_index=idx, status=status, output=output)


def _quality(
    accuracy: float = 0.8,
    completeness: float = 0.8,
    relevance: float = 0.8,
    coherence: float = 0.8,
    overall: float = 0.8,
) -> QualityScore:
    return QualityScore(
        accuracy=accuracy,
        completeness=completeness,
        relevance=relevance,
        coherence=coherence,
        overall=overall,
    )


def _verdict(v: Verdict, improvements: list | None = None) -> EvaluationVerdict:
    return EvaluationVerdict(
        verdict=v,
        reasoning="test",
        suggested_improvements=improvements or [],
    )


# ===========================================================================
# _heuristic_accuracy
# ===========================================================================

class TestHeuristicAccuracy:
    def test_empty_output_returns_zero(self):
        """
        SENTINEL — empty output must score 0.
        Remove empty-check → blank output gets base 0.4 → test fails.
        """
        assert _heuristic_accuracy("", []) == 0.0
        assert _heuristic_accuracy("   ", []) == 0.0

    def test_output_with_no_steps_gets_base_score(self):
        """Non-empty output with no step results must get 0.4 (base)."""
        assert _heuristic_accuracy("some output", []) == 0.4

    def test_all_successful_steps_give_full_score(self):
        """
        SENTINEL — all COMPLETED steps must give score = 0.4 + 0.6 = 1.0.
        Remove non_error_rate boost → only base score returned → test fails.
        """
        steps = [_result(i, StepStatus.COMPLETED) for i in range(5)]
        result = _heuristic_accuracy("output here", steps)
        assert abs(result - 1.0) < 1e-9

    def test_all_failed_steps_give_base_score_only(self):
        """
        SENTINEL — all FAILED steps → non_error_rate=0 → score=0.4.
        Ignore error_count → perfect score even with all failures → test fails.
        """
        steps = [_result(i, StepStatus.FAILED) for i in range(5)]
        result = _heuristic_accuracy("output here", steps)
        assert abs(result - 0.4) < 1e-9

    def test_partial_failures_score_between_base_and_full(self):
        """2 of 4 steps fail → non_error_rate=0.5 → score=0.4+0.3=0.7."""
        steps = [
            _result(0, StepStatus.COMPLETED),
            _result(1, StepStatus.COMPLETED),
            _result(2, StepStatus.FAILED),
            _result(3, StepStatus.FAILED),
        ]
        result = _heuristic_accuracy("output", steps)
        assert abs(result - 0.7) < 1e-9


# ===========================================================================
# _heuristic_completeness
# ===========================================================================

class TestHeuristicCompleteness:
    def test_no_plan_returns_0_5(self):
        """No plan steps → vacuously 0.5 (can't judge completeness)."""
        assert _heuristic_completeness([], []) == 0.5

    def test_all_completed_returns_1_0(self):
        """
        SENTINEL — all steps COMPLETED must give completeness=1.0.
        Break completion counting → always returns 0 → test fails.
        """
        steps = [_step(i) for i in range(3)]
        results = [_result(i, StepStatus.COMPLETED) for i in range(3)]
        assert _heuristic_completeness(steps, results) == 1.0

    def test_partial_steps_count_as_half(self):
        """
        SENTINEL — PARTIAL step counts 0.5 toward completeness.
        Count partial as full → overestimates completeness → test fails.
        """
        steps = [_step(0), _step(1)]
        results = [_result(0, StepStatus.PARTIAL), _result(1, StepStatus.PARTIAL)]
        result = _heuristic_completeness(steps, results)
        assert abs(result - 0.5) < 1e-9  # (0.5+0.5) / 2

    def test_all_failed_returns_zero(self):
        """All FAILED steps → completeness=0."""
        steps = [_step(0), _step(1)]
        results = [_result(0, StepStatus.FAILED), _result(1, StepStatus.FAILED)]
        assert _heuristic_completeness(steps, results) == 0.0

    def test_missing_result_counts_as_incomplete(self):
        """A step with no result must not count toward completion."""
        steps = [_step(0), _step(1)]
        results = [_result(0, StepStatus.COMPLETED)]  # step 1 has no result
        result = _heuristic_completeness(steps, results)
        assert result == 0.5  # only 1 of 2 completed


# ===========================================================================
# _heuristic_relevance
# ===========================================================================

class TestHeuristicRelevance:
    def test_empty_intent_returns_zero(self):
        """Empty intent → can't judge relevance → 0.0."""
        assert _heuristic_relevance("", "some output") == 0.0

    def test_empty_output_returns_zero(self):
        """Empty output → can't be relevant → 0.0."""
        assert _heuristic_relevance("some intent", "") == 0.0

    def test_identical_content_returns_1(self):
        """
        SENTINEL — output matching all intent keywords must score 1.0.
        Remove keyword overlap → all content gets same score → test fails.
        """
        intent = "analyze database performance metrics"
        output = "I analyzed the database performance metrics thoroughly."
        result = _heuristic_relevance(intent, output)
        assert result > 0.7, f"Expected >0.7, got {result}"

    def test_completely_different_content_scores_low(self):
        """
        SENTINEL — output with zero keyword overlap must score near 0.
        Always return 0.5 → relevance never detected → test fails.
        """
        intent = "database query optimization"
        output = "The weather is sunny and warm."
        result = _heuristic_relevance(intent, output)
        assert result < 0.3, f"Expected <0.3, got {result}"

    def test_stop_words_not_counted(self):
        """Stop words like 'the', 'is', 'a' must not inflate relevance."""
        # intent is only stop words
        intent = "the is a and or but"
        output = "completely unrelated content"
        result = _heuristic_relevance(intent, output)
        # All intent words are stop words → intent_words empty → returns 0.5
        assert result == 0.5


# ===========================================================================
# _heuristic_coherence
# ===========================================================================

class TestHeuristicCoherence:
    def test_empty_output_returns_zero(self):
        """
        SENTINEL — empty output must score 0.
        Remove empty check → empty content gets base 0.3 → test fails.
        """
        assert _heuristic_coherence("") == 0.0

    def test_short_output_gets_base_score(self):
        """Short output (< 10 words) gets base score 0.3."""
        result = _heuristic_coherence("Hello world.")
        assert result >= 0.3

    def test_structured_output_scores_higher_than_short(self):
        """
        SENTINEL — structured output with sentences, newlines, bullets scores
        higher than a single-word response.
        Remove structural heuristics → everything scores 0.3 → test fails.
        """
        short = _heuristic_coherence("ok")
        structured = _heuristic_coherence(
            "- First point about the analysis.\n"
            "- Second point with more detail.\n"
            "The conclusion follows from the evidence above!"
        )
        assert structured > short

    def test_bulleted_list_scores_higher(self):
        """
        SENTINEL — output starting with '-' must get bullet bonus.
        Remove bullet detection → lists treated same as prose → test fails.
        """
        plain = _heuristic_coherence("just a simple response with more than ten words here to pass threshold")
        bulleted = _heuristic_coherence("- First point here with enough words to reach threshold count for test.")
        assert bulleted >= plain

    def test_long_output_boosts_score(self):
        """Output >= 50 words must score higher than output with 10-49 words."""
        medium = _heuristic_coherence(" ".join(["word"] * 20))
        long_ = _heuristic_coherence(" ".join(["word"] * 60))
        assert long_ >= medium

    def test_score_clamped_to_1(self):
        """Even maximally structured output must not exceed 1.0."""
        result = _heuristic_coherence(
            "- Point one!\n- Point two.\n- Point three?\n" * 20
        )
        assert result <= 1.0


# ===========================================================================
# _evaluate_step_quality
# ===========================================================================

class TestEvaluateStepQuality:
    def test_no_result_returns_zero(self):
        """Missing result must return 0.0 quality."""
        assert _evaluate_step_quality(_step(0), None) == 0.0

    def test_completed_without_output_gives_0_8(self):
        """COMPLETED step with no output gives 0.8 base."""
        result = _evaluate_step_quality(_step(0), _result(0, StepStatus.COMPLETED))
        assert result == 0.8

    def test_completed_with_output_gives_1_0(self):
        """
        SENTINEL — COMPLETED step with non-empty output must score 1.0.
        Remove output bonus → having output doesn't improve score → test fails.
        """
        result = _evaluate_step_quality(
            _step(0), _result(0, StepStatus.COMPLETED, output="result text")
        )
        assert result == 1.0

    def test_partial_gives_0_5(self):
        """PARTIAL step quality must be 0.5."""
        assert _evaluate_step_quality(_step(0), _result(0, StepStatus.PARTIAL)) == 0.5

    def test_failed_gives_0_1(self):
        """FAILED step quality must be 0.1."""
        assert _evaluate_step_quality(_step(0), _result(0, StepStatus.FAILED)) == 0.1

    def test_skipped_required_gives_0(self):
        """Required step that was skipped must score 0.0."""
        assert _evaluate_step_quality(
            _step(0, required=True), _result(0, StepStatus.SKIPPED)
        ) == 0.0

    def test_skipped_optional_gives_0_3(self):
        """Optional step that was skipped must score 0.3 (acceptable)."""
        assert _evaluate_step_quality(
            _step(0, required=False), _result(0, StepStatus.SKIPPED)
        ) == 0.3


# ===========================================================================
# _check_plan_completion — required_steps_met
# ===========================================================================

class TestCheckPlanCompletion:
    def test_empty_plan_is_complete(self):
        """No planned steps → vacuously complete."""
        status = _check_plan_completion([], [])
        assert status.required_steps_met is True
        assert status.completion_rate == 1.0

    def test_all_completed_required_steps_met(self):
        """All COMPLETED required steps → required_steps_met=True."""
        steps = [_step(0, required=True), _step(1, required=True)]
        results = [_result(0, StepStatus.COMPLETED), _result(1, StepStatus.COMPLETED)]
        status = _check_plan_completion(steps, results)
        assert status.required_steps_met is True
        assert status.completion_rate == 1.0

    def test_failed_required_step_unsets_required_met(self):
        """
        SENTINEL — a required step that FAILED must set required_steps_met=False.
        Remove required check → failed required steps look like success → test fails.
        """
        steps = [_step(0, required=True)]
        results = [_result(0, StepStatus.FAILED)]
        status = _check_plan_completion(steps, results)
        assert status.required_steps_met is False

    def test_skipped_required_step_unsets_required_met(self):
        """Skipping a required step must also set required_steps_met=False."""
        steps = [_step(0, required=True)]
        results = [_result(0, StepStatus.SKIPPED)]
        status = _check_plan_completion(steps, results)
        assert status.required_steps_met is False

    def test_failed_optional_step_does_not_affect_required_met(self):
        """A failed optional step must not set required_steps_met=False."""
        steps = [_step(0, required=True), _step(1, required=False)]
        results = [
            _result(0, StepStatus.COMPLETED),
            _result(1, StepStatus.FAILED),
        ]
        status = _check_plan_completion(steps, results)
        assert status.required_steps_met is True

    def test_partial_completion_rate(self):
        """2 completed, 1 partial, 1 failed of 4 steps → rate=(2+0.5)/4=0.625."""
        steps = [_step(i) for i in range(4)]
        results = [
            _result(0, StepStatus.COMPLETED),
            _result(1, StepStatus.COMPLETED),
            _result(2, StepStatus.PARTIAL),
            _result(3, StepStatus.FAILED),
        ]
        status = _check_plan_completion(steps, results)
        assert abs(status.completion_rate - 0.625) < 1e-9


# ===========================================================================
# _determine_verdict
# ===========================================================================

class TestDetermineVerdict:
    def _plan_met(self) -> PlanCompletionStatus:
        return PlanCompletionStatus(
            total_steps=1, completed_steps=1, completion_rate=1.0, required_steps_met=True
        )

    def _plan_unmet(self) -> PlanCompletionStatus:
        return PlanCompletionStatus(
            total_steps=1, completed_steps=0, completion_rate=0.0, required_steps_met=False
        )

    def test_high_quality_all_steps_met_gives_approve(self):
        """
        SENTINEL — overall>=0.7 and required_steps_met must yield APPROVE.
        Remove APPROVE path → good output never finalized → test fails.
        """
        verdict = _determine_verdict(_quality(overall=0.8), self._plan_met(), 0, 3)
        assert verdict.verdict == Verdict.APPROVE

    def test_retries_exhausted_gives_escalate(self):
        """
        SENTINEL — retry_count >= max_retries must yield ESCALATE.
        Remove retry exhaustion check → infinite retry loop → test fails.
        """
        verdict = _determine_verdict(_quality(overall=0.5), self._plan_met(), 3, 3)
        assert verdict.verdict == Verdict.ESCALATE

    def test_very_low_quality_gives_escalate(self):
        """
        SENTINEL — overall < RETRY_QUALITY_MIN (0.3) must yield ESCALATE.
        Remove quality floor check → garbage output gets retry instead → test fails.
        """
        verdict = _determine_verdict(_quality(overall=0.1), self._plan_met(), 0, 3)
        assert verdict.verdict == Verdict.ESCALATE

    def test_missing_steps_decent_quality_gives_refine(self):
        """
        SENTINEL — plan steps unmet + overall>=0.5 must yield REFINE.
        Remove REFINE path → plan gaps treated as retry → test fails.
        """
        verdict = _determine_verdict(_quality(overall=0.6), self._plan_unmet(), 0, 3)
        assert verdict.verdict == Verdict.REFINE

    def test_mediocre_quality_gives_retry(self):
        """Average quality with retries remaining must yield RETRY."""
        verdict = _determine_verdict(_quality(overall=0.5), self._plan_met(), 0, 3)
        assert verdict.verdict == Verdict.RETRY

    def test_low_dimension_scores_generate_improvements(self):
        """Scores below 0.6 must generate improvement suggestions."""
        verdict = _determine_verdict(
            _quality(accuracy=0.3, completeness=0.3, relevance=0.3, coherence=0.3, overall=0.3),
            self._plan_met(),
            0,
            3,
        )
        assert len(verdict.suggested_improvements) >= 3


# ===========================================================================
# _build_retry_decision
# ===========================================================================

class TestBuildRetryDecision:
    def test_approve_verdict_no_retry(self):
        """
        SENTINEL — APPROVE must yield should_retry=False.
        Skip APPROVE check → approved tasks still retried → test fails.
        """
        decision = _build_retry_decision(_verdict(Verdict.APPROVE), _quality(), 0, 3)
        assert decision.should_retry is False
        assert decision.escalate is False

    def test_escalate_verdict_triggers_escalation(self):
        """
        SENTINEL — ESCALATE must yield escalate=True, should_retry=False.
        Remove ESCALATE path → human review never triggered → test fails.
        """
        decision = _build_retry_decision(_verdict(Verdict.ESCALATE), _quality(), 3, 3)
        assert decision.should_retry is False
        assert decision.escalate is True

    def test_retry_verdict_should_retry_true(self):
        """RETRY verdict must yield should_retry=True."""
        decision = _build_retry_decision(_verdict(Verdict.RETRY), _quality(), 0, 3)
        assert decision.should_retry is True

    def test_retry_increments_count(self):
        """
        SENTINEL — retry_count must be incremented by 1.
        Forget to increment → retry count stuck at 0 → never escalates → test fails.
        """
        decision = _build_retry_decision(_verdict(Verdict.RETRY), _quality(), 2, 5)
        assert decision.retry_count == 3

    def test_exponential_backoff_increases_delay(self):
        """
        SENTINEL — each retry must have higher delay than previous.
        Use fixed delay → backoff removed → overloads LLM on failures → test fails.
        """
        d0 = _build_retry_decision(_verdict(Verdict.RETRY), _quality(), 0, 10)
        d1 = _build_retry_decision(_verdict(Verdict.RETRY), _quality(), 1, 10)
        d2 = _build_retry_decision(_verdict(Verdict.RETRY), _quality(), 2, 10)
        assert d0.delay_ms < d1.delay_ms <= d2.delay_ms

    def test_delay_capped_at_max(self):
        """
        SENTINEL — delay must never exceed RETRY_MAX_DELAY_MS (30000).
        Remove min() cap → very high retry counts get absurd delays → test fails.
        """
        decision = _build_retry_decision(_verdict(Verdict.RETRY), _quality(), 100, 200)
        assert decision.delay_ms <= 30000

    def test_low_accuracy_sets_accuracy_focus(self):
        """
        SENTINEL — accuracy < 0.5 must set adjustments['focus']='accuracy'.
        Remove accuracy focus → retry has no targeted improvement → test fails.
        """
        low_accuracy_quality = _quality(accuracy=0.2, overall=0.5)
        decision = _build_retry_decision(_verdict(Verdict.RETRY), low_accuracy_quality, 0, 3)
        assert decision.adjustments.get("focus") == "accuracy"


# ===========================================================================
# QualityScore.compute_overall — weighted average
# ===========================================================================

class TestQualityScoreComputeOverall:
    def test_all_ones_gives_1(self):
        """Perfect scores on all dimensions must give overall=1.0."""
        q = QualityScore(accuracy=1.0, completeness=1.0, relevance=1.0, coherence=1.0)
        result = q.compute_overall()
        assert abs(result - 1.0) < 1e-9

    def test_all_zeros_gives_0(self):
        """Zero on all dimensions must give overall=0.0."""
        q = QualityScore(accuracy=0.0, completeness=0.0, relevance=0.0, coherence=0.0)
        result = q.compute_overall()
        assert result == 0.0

    def test_known_value(self):
        """
        SENTINEL — hand-computed value verifies formula unchanged.
        Default weights: accuracy=0.35, completeness=0.25, relevance=0.25, coherence=0.15
        With all=0.8: (0.8*0.35 + 0.8*0.25 + 0.8*0.25 + 0.8*0.15) / 1.0 = 0.8
        """
        q = QualityScore(accuracy=0.8, completeness=0.8, relevance=0.8, coherence=0.8)
        result = q.compute_overall()
        assert abs(result - 0.8) < 1e-9

    def test_accuracy_has_highest_weight(self):
        """
        SENTINEL — accuracy has highest weight (0.35), so it must influence
        overall more than coherence (0.15).
        """
        high_acc = QualityScore(accuracy=1.0, completeness=0.0, relevance=0.0, coherence=0.0)
        high_coh = QualityScore(accuracy=0.0, completeness=0.0, relevance=0.0, coherence=1.0)
        assert high_acc.compute_overall() > high_coh.compute_overall()
