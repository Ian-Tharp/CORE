"""
Evaluation Service

Core business logic for the CORE Evaluation Engine.
Scores output quality, determines verdicts, manages retries, and feeds learnings
back into procedural memory to close the CORE loop.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4
from datetime import datetime

from app.models.evaluation_models import (
    EvaluationInput,
    EvaluateStepInput,
    EvaluationResult,
    EvaluationMetrics,
    EvaluationHistoryFilter,
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
)
from app.repository import evaluation_repository, memory_repository
from app.repository.memory_repository import ProceduralMemory

logger = logging.getLogger(__name__)


# =============================================================================
# CORE EVALUATION
# =============================================================================

async def evaluate(evaluation_input: EvaluationInput) -> EvaluationResult:
    """
    Evaluate the output of a reasoning phase against the original intent.

    Pipeline:
      1. Score output quality on multiple dimensions
      2. Check plan step completion
      3. Determine verdict (approve / retry / refine / escalate)
      4. Build retry decision if needed
      5. Store result
      6. Feed learnings into procedural memory
    """
    start_ms = _now_ms()

    # 1. Score quality
    quality = await _score_quality(
        original_intent=evaluation_input.original_intent,
        final_output=evaluation_input.final_output,
        plan_steps=evaluation_input.plan_steps,
        step_results=evaluation_input.step_results,
    )

    # 2. Check plan completion
    plan_completion = _check_plan_completion(
        plan_steps=evaluation_input.plan_steps,
        step_results=evaluation_input.step_results,
    )

    # 3. Determine verdict
    verdict = _determine_verdict(
        quality=quality,
        plan_completion=plan_completion,
        retry_count=evaluation_input.retry_count,
        max_retries=evaluation_input.max_retries,
    )

    # 4. Build retry decision (if not approved)
    retry_decision: Optional[RetryDecision] = None
    if verdict.verdict != Verdict.APPROVE:
        retry_decision = _build_retry_decision(
            verdict=verdict,
            quality=quality,
            retry_count=evaluation_input.retry_count,
            max_retries=evaluation_input.max_retries,
        )

    # 5. Generate feedback
    feedback = _generate_feedback(quality, plan_completion, verdict)
    improvements = verdict.suggested_improvements

    eval_duration_ms = _now_ms() - start_ms

    result = EvaluationResult(
        id=uuid4(),
        task_id=evaluation_input.task_id,
        agent_id=evaluation_input.agent_id,
        quality_score=quality,
        verdict=verdict,
        plan_completion=plan_completion,
        feedback=feedback,
        suggested_improvements=improvements,
        retry_decision=retry_decision,
        model_used=evaluation_input.model_used,
        execution_duration_ms=evaluation_input.execution_duration_ms,
        evaluation_duration_ms=eval_duration_ms,
        metadata=evaluation_input.metadata,
        created_at=datetime.utcnow(),
    )

    # 6. Persist
    try:
        await evaluation_repository.store_evaluation(result)
    except Exception as exc:
        logger.error("Failed to persist evaluation for task %s: %s", evaluation_input.task_id, exc)

    # 7. Feed learnings into memory
    try:
        await _store_learnings(evaluation_input, result)
    except Exception as exc:
        logger.error("Failed to store learnings for task %s: %s", evaluation_input.task_id, exc)

    return result


async def evaluate_step(step_input: EvaluateStepInput) -> StepEvaluation:
    """Evaluate a single plan step in isolation."""
    step_plan = step_input.step_plan
    step_result = step_input.step_result

    quality = _evaluate_step_quality(step_plan, step_result)
    meets = _step_meets_criteria(step_plan, step_result, quality)
    fb = _step_feedback(step_plan, step_result, quality, meets)

    return StepEvaluation(
        step_index=step_plan.step_index,
        status=step_result.status,
        quality_score=quality,
        feedback=fb,
        meets_criteria=meets,
    )


async def should_retry(evaluation: EvaluationResult) -> RetryDecision:
    """
    Determine whether to retry based on a completed evaluation.

    Returns a RetryDecision with delay, adjustments, and escalation info.
    """
    if evaluation.verdict.verdict == Verdict.APPROVE:
        return RetryDecision(
            should_retry=False,
            retry_count=0,
            reason="Output approved — no retry needed.",
        )

    retry_count = evaluation.retry_decision.retry_count if evaluation.retry_decision else 0
    max_retries = evaluation.retry_decision.max_retries if evaluation.retry_decision else EvaluationThresholds.MAX_RETRIES

    return _build_retry_decision(
        verdict=evaluation.verdict,
        quality=evaluation.quality_score,
        retry_count=retry_count,
        max_retries=max_retries,
    )


async def get_evaluation_metrics(
    agent_id: Optional[UUID] = None,
    created_after: Optional[datetime] = None,
    created_before: Optional[datetime] = None,
) -> EvaluationMetrics:
    """Return aggregate evaluation metrics (delegates to repository)."""
    return await evaluation_repository.get_evaluation_metrics(
        agent_id=agent_id,
        created_after=created_after,
        created_before=created_before,
    )


async def record_human_feedback(
    evaluation_id: UUID,
    feedback: HumanFeedbackInput,
) -> bool:
    """
    Record human-in-the-loop feedback on an evaluation.

    If the human disagrees with the verdict, this signal can be used
    to recalibrate thresholds or retrain scoring heuristics over time.
    """
    success = await evaluation_repository.record_human_feedback(evaluation_id, feedback)
    if success and not feedback.agree_with_verdict:
        logger.info(
            "Human disagreed with verdict on evaluation %s — corrected to %s",
            evaluation_id,
            feedback.corrected_verdict,
        )
    return success


async def get_evaluations_for_task(task_id: UUID) -> List[EvaluationResult]:
    """Return all evaluations for a given task."""
    return await evaluation_repository.get_evaluations_for_task(task_id)


async def list_evaluations(
    filter_params: Optional[EvaluationHistoryFilter] = None,
    limit: int = 50,
    offset: int = 0,
):
    """Return paginated evaluation history."""
    return await evaluation_repository.list_evaluations(filter_params, limit, offset)


# =============================================================================
# QUALITY SCORING (INTERNAL)
# =============================================================================

async def _score_quality(
    original_intent: str,
    final_output: str,
    plan_steps: List[PlanStep],
    step_results: List[StepResult],
) -> QualityScore:
    """
    Score the output on multiple quality dimensions.

    Currently uses heuristic scoring. A future version can incorporate
    LLM-as-judge and embedding-based semantic similarity.
    """

    # ── Accuracy ─────────────────────────────────────────────────────────
    # Heuristic: output non-empty + low error rate in steps
    accuracy = _heuristic_accuracy(final_output, step_results)

    # ── Completeness ─────────────────────────────────────────────────────
    # Fraction of plan steps completed successfully
    completeness = _heuristic_completeness(plan_steps, step_results)

    # ── Relevance ────────────────────────────────────────────────────────
    # Keyword overlap between intent and output (simple baseline)
    relevance = _heuristic_relevance(original_intent, final_output)

    # ── Coherence ────────────────────────────────────────────────────────
    # Basic structural quality check
    coherence = _heuristic_coherence(final_output)

    score = QualityScore(
        accuracy=accuracy,
        completeness=completeness,
        relevance=relevance,
        coherence=coherence,
    )
    score.compute_overall()
    return score


def _heuristic_accuracy(final_output: str, step_results: List[StepResult]) -> float:
    """Score accuracy based on output presence and step error rate."""
    if not final_output or not final_output.strip():
        return 0.0

    # Base score for having non-empty output
    score = 0.4

    if not step_results:
        return score

    # Proportion of steps without errors
    error_count = sum(1 for sr in step_results if sr.status == StepStatus.FAILED)
    total = len(step_results)
    non_error_rate = 1.0 - (error_count / total) if total > 0 else 0.0
    score += 0.6 * non_error_rate

    return min(score, 1.0)


def _heuristic_completeness(
    plan_steps: List[PlanStep],
    step_results: List[StepResult],
) -> float:
    """Score completeness as fraction of plan steps completed."""
    if not plan_steps:
        # No plan → evaluate just on whether output exists
        return 0.5

    result_map = {sr.step_index: sr for sr in step_results}

    completed = 0
    partial = 0
    for step in plan_steps:
        sr = result_map.get(step.step_index)
        if sr and sr.status == StepStatus.COMPLETED:
            completed += 1
        elif sr and sr.status == StepStatus.PARTIAL:
            partial += 1

    total = len(plan_steps)
    return (completed + 0.5 * partial) / total if total > 0 else 0.0


def _heuristic_relevance(intent: str, output: str) -> float:
    """Score relevance via simple keyword overlap ratio."""
    if not intent or not output:
        return 0.0

    intent_words = set(intent.lower().split())
    output_words = set(output.lower().split())

    # Remove very common stop words
    stop = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "shall",
            "should", "may", "might", "can", "could", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "and",
            "or", "but", "not", "so", "if", "than", "that", "this", "it", "i"}
    intent_words -= stop
    output_words -= stop

    if not intent_words:
        return 0.5

    overlap = intent_words & output_words
    return min(len(overlap) / len(intent_words), 1.0)


def _heuristic_coherence(output: str) -> float:
    """Score coherence based on structural heuristics."""
    if not output or not output.strip():
        return 0.0

    score = 0.3  # Base for non-empty

    # Longer, structured output → higher coherence
    word_count = len(output.split())
    if word_count >= 10:
        score += 0.2
    if word_count >= 50:
        score += 0.1

    # Sentences (period-terminated)
    sentence_count = output.count(".") + output.count("!") + output.count("?")
    if sentence_count >= 2:
        score += 0.2

    # Paragraphs / structure
    if "\n" in output:
        score += 0.1

    # Bullet / list structure
    if any(output.strip().startswith(c) for c in ["-", "•", "1.", "*"]):
        score += 0.1

    return min(score, 1.0)


# =============================================================================
# PLAN COMPLETION
# =============================================================================

def _check_plan_completion(
    plan_steps: List[PlanStep],
    step_results: List[StepResult],
) -> PlanCompletionStatus:
    """Build a PlanCompletionStatus from the step data."""
    if not plan_steps:
        return PlanCompletionStatus(
            total_steps=0,
            completed_steps=0,
            completion_rate=1.0,  # Vacuously complete
            required_steps_met=True,
        )

    result_map = {sr.step_index: sr for sr in step_results}

    completed = 0
    partial = 0
    failed = 0
    skipped = 0
    evals: List[StepEvaluation] = []
    required_met = True

    for step in plan_steps:
        sr = result_map.get(step.step_index)
        status = sr.status if sr else StepStatus.NOT_STARTED

        if status == StepStatus.COMPLETED:
            completed += 1
        elif status == StepStatus.PARTIAL:
            partial += 1
        elif status == StepStatus.FAILED:
            failed += 1
            if step.required:
                required_met = False
        elif status == StepStatus.SKIPPED:
            skipped += 1
            if step.required:
                required_met = False
        else:
            # NOT_STARTED
            if step.required:
                required_met = False

        q = _evaluate_step_quality(step, sr) if sr else 0.0
        meets = _step_meets_criteria(step, sr, q) if sr else False
        fb = _step_feedback(step, sr, q, meets) if sr else "Step not started."

        evals.append(StepEvaluation(
            step_index=step.step_index,
            status=status,
            quality_score=q,
            feedback=fb,
            meets_criteria=meets,
        ))

    total = len(plan_steps)
    completion_rate = (completed + 0.5 * partial) / total if total > 0 else 0.0

    return PlanCompletionStatus(
        total_steps=total,
        completed_steps=completed,
        partial_steps=partial,
        failed_steps=failed,
        skipped_steps=skipped,
        completion_rate=completion_rate,
        required_steps_met=required_met,
        step_evaluations=evals,
    )


def _evaluate_step_quality(step: PlanStep, result: Optional[StepResult]) -> float:
    """Compute a quality score for a single step."""
    if not result:
        return 0.0

    if result.status == StepStatus.COMPLETED:
        base = 0.8
        # Bonus if output exists
        if result.output and result.output.strip():
            base += 0.2
        return min(base, 1.0)
    elif result.status == StepStatus.PARTIAL:
        return 0.5
    elif result.status == StepStatus.FAILED:
        return 0.1
    elif result.status == StepStatus.SKIPPED:
        return 0.0 if step.required else 0.3
    return 0.0


def _step_meets_criteria(step: PlanStep, result: Optional[StepResult], quality: float) -> bool:
    """Check whether a step meets its criteria."""
    if not result:
        return False
    if result.status == StepStatus.FAILED:
        return False
    if result.status == StepStatus.COMPLETED and quality >= 0.6:
        return True
    if result.status == StepStatus.PARTIAL and quality >= 0.5 and not step.required:
        return True
    return False


def _step_feedback(step: PlanStep, result: Optional[StepResult], quality: float, meets: bool) -> str:
    """Generate feedback for a single step."""
    if not result:
        return f"Step {step.step_index}: Not started — '{step.description}'."

    if meets:
        return f"Step {step.step_index}: Completed successfully (quality={quality:.2f})."

    if result.status == StepStatus.FAILED:
        err = result.error or "unknown error"
        return f"Step {step.step_index}: Failed — {err}."

    if result.status == StepStatus.PARTIAL:
        return f"Step {step.step_index}: Partially completed (quality={quality:.2f}). More work needed."

    if result.status == StepStatus.SKIPPED:
        return f"Step {step.step_index}: Skipped."

    return f"Step {step.step_index}: Status {result.status.value} (quality={quality:.2f})."


# =============================================================================
# VERDICT DETERMINATION
# =============================================================================

def _determine_verdict(
    quality: QualityScore,
    plan_completion: PlanCompletionStatus,
    retry_count: int,
    max_retries: int,
) -> EvaluationVerdict:
    """
    Determine the evaluation verdict based on scores and plan completion.
    """
    overall = quality.overall
    improvements: List[str] = []

    # Collect improvement suggestions per dimension
    if quality.accuracy < 0.6:
        improvements.append("Improve factual accuracy of the output.")
    if quality.completeness < 0.6:
        improvements.append("Address all parts of the original intent more thoroughly.")
    if quality.relevance < 0.6:
        improvements.append("Focus the output more closely on the original intent.")
    if quality.coherence < 0.6:
        improvements.append("Improve structure and logical flow of the output.")

    if not plan_completion.required_steps_met:
        improvements.append("Ensure all required plan steps are completed.")

    # ── Approve ──────────────────────────────────────────────────────────
    if (
        overall >= EvaluationThresholds.APPROVE_QUALITY
        and plan_completion.required_steps_met
    ):
        return EvaluationVerdict(
            verdict=Verdict.APPROVE,
            reasoning=f"Output quality ({overall:.2f}) meets threshold and all required steps completed.",
            confidence=min(overall, 1.0),
            suggested_improvements=improvements,
        )

    # ── Escalate (exhausted retries or very low quality) ─────────────────
    if retry_count >= max_retries:
        return EvaluationVerdict(
            verdict=Verdict.ESCALATE,
            reasoning=f"Maximum retries ({max_retries}) exhausted. Escalating for human review.",
            confidence=0.9,
            suggested_improvements=improvements,
        )

    if overall < EvaluationThresholds.RETRY_QUALITY_MIN:
        return EvaluationVerdict(
            verdict=Verdict.ESCALATE,
            reasoning=f"Quality ({overall:.2f}) below minimum retry threshold ({EvaluationThresholds.RETRY_QUALITY_MIN}). Needs human review.",
            confidence=0.85,
            suggested_improvements=improvements,
        )

    # ── Refine (plan problems, decent quality) ───────────────────────────
    if not plan_completion.required_steps_met and overall >= 0.5:
        return EvaluationVerdict(
            verdict=Verdict.REFINE,
            reasoning=f"Output quality acceptable ({overall:.2f}) but required plan steps not met. Plan refinement needed.",
            confidence=0.7,
            suggested_improvements=improvements,
        )

    # ── Retry ────────────────────────────────────────────────────────────
    return EvaluationVerdict(
        verdict=Verdict.RETRY,
        reasoning=f"Output quality ({overall:.2f}) below approval threshold ({EvaluationThresholds.APPROVE_QUALITY}). Retrying.",
        confidence=0.7,
        suggested_improvements=improvements,
    )


# =============================================================================
# RETRY LOGIC
# =============================================================================

def _build_retry_decision(
    verdict: EvaluationVerdict,
    quality: QualityScore,
    retry_count: int,
    max_retries: int,
) -> RetryDecision:
    """Build a RetryDecision with exponential backoff."""
    if verdict.verdict == Verdict.APPROVE:
        return RetryDecision(
            should_retry=False,
            retry_count=retry_count,
            max_retries=max_retries,
            reason="Approved — no retry necessary.",
        )

    if verdict.verdict == Verdict.ESCALATE:
        return RetryDecision(
            should_retry=False,
            escalate=True,
            retry_count=retry_count,
            max_retries=max_retries,
            reason=verdict.reasoning,
        )

    # Exponential backoff
    delay = min(
        EvaluationThresholds.RETRY_BASE_DELAY_MS * (2 ** retry_count),
        EvaluationThresholds.RETRY_MAX_DELAY_MS,
    )

    adjustments: Dict[str, Any] = {}
    if verdict.suggested_improvements:
        adjustments["improvements"] = verdict.suggested_improvements
    if quality.accuracy < 0.5:
        adjustments["focus"] = "accuracy"
    elif quality.completeness < 0.5:
        adjustments["focus"] = "completeness"

    return RetryDecision(
        should_retry=True,
        delay_ms=delay,
        retry_count=retry_count + 1,
        max_retries=max_retries,
        escalate=False,
        reason=verdict.reasoning,
        adjustments=adjustments,
    )


# =============================================================================
# FEEDBACK GENERATION
# =============================================================================

def _generate_feedback(
    quality: QualityScore,
    plan_completion: PlanCompletionStatus,
    verdict: EvaluationVerdict,
) -> str:
    """Generate human-readable feedback summarising the evaluation."""
    lines: List[str] = []

    lines.append(f"Verdict: {verdict.verdict.value.upper()}")
    lines.append(f"Overall quality: {quality.overall:.2f} "
                 f"(accuracy={quality.accuracy:.2f}, completeness={quality.completeness:.2f}, "
                 f"relevance={quality.relevance:.2f}, coherence={quality.coherence:.2f})")

    if plan_completion.total_steps > 0:
        lines.append(
            f"Plan completion: {plan_completion.completed_steps}/{plan_completion.total_steps} steps completed "
            f"(rate={plan_completion.completion_rate:.0%}), "
            f"required steps met: {'yes' if plan_completion.required_steps_met else 'NO'}"
        )

    lines.append(f"Reasoning: {verdict.reasoning}")

    if verdict.suggested_improvements:
        lines.append("Improvements:")
        for imp in verdict.suggested_improvements:
            lines.append(f"  - {imp}")

    return "\n".join(lines)


# =============================================================================
# MEMORY INTEGRATION
# =============================================================================

async def _store_learnings(
    evaluation_input: EvaluationInput,
    result: EvaluationResult,
) -> None:
    """
    Feed learnings back into procedural memory.

    On APPROVE → store the successful pattern so future tasks can benefit.
    On failure → store failure pattern for avoidance learning.
    """
    if result.verdict.verdict == Verdict.APPROVE:
        # Store successful procedure
        procedure = ProceduralMemory(
            id=uuid4(),
            role="evaluation_learner",
            procedure_name=f"successful_pattern_task_{evaluation_input.task_id}",
            content=(
                f"Task intent: {evaluation_input.original_intent[:500]}\n"
                f"Quality: {result.quality_score.overall:.2f}\n"
                f"Verdict: approved"
            ),
            steps=[s.description for s in evaluation_input.plan_steps] if evaluation_input.plan_steps else [],
            embedding=[],  # Would be populated by embedding service in production
            metadata={
                "task_id": str(evaluation_input.task_id),
                "quality_overall": result.quality_score.overall,
                "accuracy": result.quality_score.accuracy,
                "model_used": evaluation_input.model_used,
                "learning_type": "success",
            },
            success_rate=1.0,
            confidence=result.verdict.confidence,
        )
        try:
            await memory_repository.create_procedural_memory(procedure)
            logger.info("Stored successful pattern for task %s", evaluation_input.task_id)
        except Exception as exc:
            logger.warning("Could not store procedural memory: %s", exc)

    elif result.verdict.verdict in (Verdict.ESCALATE, Verdict.RETRY):
        # Store failure pattern for learning
        procedure = ProceduralMemory(
            id=uuid4(),
            role="evaluation_learner",
            procedure_name=f"failure_pattern_task_{evaluation_input.task_id}",
            content=(
                f"Task intent: {evaluation_input.original_intent[:500]}\n"
                f"Quality: {result.quality_score.overall:.2f}\n"
                f"Verdict: {result.verdict.verdict.value}\n"
                f"Issues: {'; '.join(result.suggested_improvements)}"
            ),
            steps=[s.description for s in evaluation_input.plan_steps] if evaluation_input.plan_steps else [],
            embedding=[],
            metadata={
                "task_id": str(evaluation_input.task_id),
                "quality_overall": result.quality_score.overall,
                "verdict": result.verdict.verdict.value,
                "model_used": evaluation_input.model_used,
                "learning_type": "failure",
                "improvements": result.suggested_improvements,
            },
            success_rate=0.0,
            confidence=result.verdict.confidence,
        )
        try:
            await memory_repository.create_procedural_memory(procedure)
            logger.info("Stored failure pattern for task %s", evaluation_input.task_id)
        except Exception as exc:
            logger.warning("Could not store procedural memory: %s", exc)


# =============================================================================
# HELPERS
# =============================================================================

def _now_ms() -> int:
    """Current time in milliseconds."""
    return int(time.time() * 1000)
