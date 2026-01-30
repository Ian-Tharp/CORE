"""
Evaluation Repository

Data access layer for evaluation persistence in the CORE Evaluation Engine.
Handles storing, retrieving, and aggregating evaluation results, verdicts, and metrics.
"""

import json
import logging
from typing import List, Optional, Tuple
from datetime import datetime
from uuid import UUID

from app.dependencies import get_db_pool
from app.models.evaluation_models import (
    EvaluationResult,
    EvaluationMetrics,
    EvaluationHistoryFilter,
    Verdict,
    QualityScore,
    EvaluationVerdict,
    PlanCompletionStatus,
    StepEvaluation,
    StepStatus,
    RetryDecision,
    HumanFeedbackInput,
)

logger = logging.getLogger(__name__)


# =============================================================================
# TABLE INITIALIZATION
# =============================================================================

async def ensure_evaluation_tables() -> None:
    """Create evaluation tables if they don't exist."""
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        # ── evaluations table ────────────────────────────────────────────
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                task_id UUID NOT NULL,
                agent_id UUID,
                
                -- quality scores
                score_accuracy FLOAT NOT NULL DEFAULT 0.0,
                score_completeness FLOAT NOT NULL DEFAULT 0.0,
                score_relevance FLOAT NOT NULL DEFAULT 0.0,
                score_coherence FLOAT NOT NULL DEFAULT 0.0,
                score_overall FLOAT NOT NULL DEFAULT 0.0,
                
                -- verdict
                verdict VARCHAR(20) NOT NULL DEFAULT 'approve',
                verdict_reasoning TEXT NOT NULL DEFAULT '',
                verdict_confidence FLOAT NOT NULL DEFAULT 0.5,
                verdict_improvements JSONB DEFAULT '[]',
                
                -- plan completion
                plan_total_steps INTEGER NOT NULL DEFAULT 0,
                plan_completed_steps INTEGER NOT NULL DEFAULT 0,
                plan_partial_steps INTEGER NOT NULL DEFAULT 0,
                plan_failed_steps INTEGER NOT NULL DEFAULT 0,
                plan_skipped_steps INTEGER NOT NULL DEFAULT 0,
                plan_completion_rate FLOAT NOT NULL DEFAULT 0.0,
                plan_required_met BOOLEAN NOT NULL DEFAULT FALSE,
                step_evaluations JSONB DEFAULT '[]',
                
                -- feedback
                feedback TEXT NOT NULL DEFAULT '',
                suggested_improvements JSONB DEFAULT '[]',
                
                -- retry decision
                retry_should BOOLEAN,
                retry_delay_ms INTEGER,
                retry_count INTEGER DEFAULT 0,
                retry_max INTEGER DEFAULT 3,
                retry_escalate BOOLEAN DEFAULT FALSE,
                retry_reason TEXT,
                retry_adjustments JSONB,
                
                -- human feedback
                human_agree BOOLEAN,
                human_corrected_verdict VARCHAR(20),
                human_quality_override FLOAT,
                human_feedback_text TEXT,
                human_corrected_output TEXT,
                
                -- metadata
                model_used VARCHAR(255),
                execution_duration_ms INTEGER,
                evaluation_duration_ms INTEGER,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                
                CHECK (verdict IN ('approve', 'retry', 'refine', 'escalate'))
            )
        """)

        # ── indexes ──────────────────────────────────────────────────────
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_evaluations_task ON evaluations(task_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_evaluations_agent ON evaluations(agent_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_evaluations_verdict ON evaluations(verdict)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_evaluations_created ON evaluations(created_at DESC)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_evaluations_quality ON evaluations(score_overall)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_evaluations_human_fb ON evaluations(human_agree) WHERE human_agree IS NOT NULL"
        )

        logger.info("Evaluation tables ensured")


# =============================================================================
# CRUD OPERATIONS
# =============================================================================

async def store_evaluation(result: EvaluationResult) -> UUID:
    """
    Persist an evaluation result.

    Returns:
        UUID of the stored evaluation.
    """
    pool = await get_db_pool()

    query = """
        INSERT INTO evaluations (
            id, task_id, agent_id,
            score_accuracy, score_completeness, score_relevance, score_coherence, score_overall,
            verdict, verdict_reasoning, verdict_confidence, verdict_improvements,
            plan_total_steps, plan_completed_steps, plan_partial_steps,
            plan_failed_steps, plan_skipped_steps, plan_completion_rate,
            plan_required_met, step_evaluations,
            feedback, suggested_improvements,
            retry_should, retry_delay_ms, retry_count, retry_max,
            retry_escalate, retry_reason, retry_adjustments,
            human_agree, human_corrected_verdict, human_quality_override,
            human_feedback_text, human_corrected_output,
            model_used, execution_duration_ms, evaluation_duration_ms,
            metadata, created_at
        ) VALUES (
            $1,$2,$3,
            $4,$5,$6,$7,$8,
            $9,$10,$11,$12,
            $13,$14,$15,$16,$17,$18,$19,$20,
            $21,$22,
            $23,$24,$25,$26,$27,$28,$29,
            $30,$31,$32,$33,$34,
            $35,$36,$37,
            $38,$39
        )
        RETURNING id
    """

    qs = result.quality_score
    v = result.verdict
    pc = result.plan_completion
    rd = result.retry_decision
    hf = result.human_feedback

    async with pool.acquire() as conn:
        eval_id = await conn.fetchval(
            query,
            result.id,
            result.task_id,
            result.agent_id,
            # quality
            qs.accuracy, qs.completeness, qs.relevance, qs.coherence, qs.overall,
            # verdict
            v.verdict.value, v.reasoning, v.confidence,
            json.dumps(v.suggested_improvements),
            # plan
            pc.total_steps, pc.completed_steps, pc.partial_steps,
            pc.failed_steps, pc.skipped_steps, pc.completion_rate,
            pc.required_steps_met,
            json.dumps([se.model_dump() for se in pc.step_evaluations]),
            # feedback
            result.feedback,
            json.dumps(result.suggested_improvements),
            # retry
            rd.should_retry if rd else None,
            rd.delay_ms if rd else None,
            rd.retry_count if rd else 0,
            rd.max_retries if rd else 3,
            rd.escalate if rd else False,
            rd.reason if rd else None,
            json.dumps(rd.adjustments) if rd else None,
            # human
            hf.agree_with_verdict if hf else None,
            hf.corrected_verdict.value if hf and hf.corrected_verdict else None,
            hf.quality_override if hf else None,
            hf.feedback_text if hf else None,
            hf.corrected_output if hf else None,
            # metadata
            result.model_used,
            result.execution_duration_ms,
            result.evaluation_duration_ms,
            json.dumps(result.metadata),
            result.created_at,
        )

        logger.info(f"Stored evaluation {eval_id} for task {result.task_id} (verdict={v.verdict.value})")
        return eval_id


async def get_evaluation(evaluation_id: UUID) -> Optional[EvaluationResult]:
    """Get an evaluation by its ID."""
    pool = await get_db_pool()

    query = "SELECT * FROM evaluations WHERE id = $1"

    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, evaluation_id)
        if not row:
            return None
        return _row_to_evaluation(row)


async def get_evaluations_for_task(task_id: UUID) -> List[EvaluationResult]:
    """Get all evaluations for a specific task, ordered newest first."""
    pool = await get_db_pool()

    query = """
        SELECT * FROM evaluations
        WHERE task_id = $1
        ORDER BY created_at DESC
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, task_id)
        return [_row_to_evaluation(r) for r in rows]


async def list_evaluations(
    filter_params: Optional[EvaluationHistoryFilter] = None,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[List[EvaluationResult], int]:
    """
    List evaluations with optional filtering and pagination.

    Returns:
        Tuple of (evaluations, total_count).
    """
    pool = await get_db_pool()

    conditions: List[str] = []
    params: list = []
    param_count = 0

    if filter_params:
        if filter_params.task_id:
            param_count += 1
            conditions.append(f"task_id = ${param_count}")
            params.append(filter_params.task_id)

        if filter_params.agent_id:
            param_count += 1
            conditions.append(f"agent_id = ${param_count}")
            params.append(filter_params.agent_id)

        if filter_params.verdict:
            param_count += 1
            conditions.append(f"verdict = ${param_count}")
            params.append(filter_params.verdict.value)

        if filter_params.min_quality is not None:
            param_count += 1
            conditions.append(f"score_overall >= ${param_count}")
            params.append(filter_params.min_quality)

        if filter_params.max_quality is not None:
            param_count += 1
            conditions.append(f"score_overall <= ${param_count}")
            params.append(filter_params.max_quality)

        if filter_params.created_after:
            param_count += 1
            conditions.append(f"created_at >= ${param_count}")
            params.append(filter_params.created_after)

        if filter_params.created_before:
            param_count += 1
            conditions.append(f"created_at <= ${param_count}")
            params.append(filter_params.created_before)

        if filter_params.has_human_feedback is not None:
            if filter_params.has_human_feedback:
                conditions.append("human_agree IS NOT NULL")
            else:
                conditions.append("human_agree IS NULL")

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    # Count query
    count_query = f"SELECT COUNT(*) FROM evaluations {where_clause}"

    # Data query
    param_count += 1
    params.append(limit)
    limit_idx = param_count
    param_count += 1
    params.append(offset)
    offset_idx = param_count

    data_query = f"""
        SELECT * FROM evaluations
        {where_clause}
        ORDER BY created_at DESC
        LIMIT ${limit_idx} OFFSET ${offset_idx}
    """

    async with pool.acquire() as conn:
        total_count = await conn.fetchval(count_query, *params[:-2])
        rows = await conn.fetch(data_query, *params)
        results = [_row_to_evaluation(r) for r in rows]
        return results, total_count


# =============================================================================
# HUMAN FEEDBACK
# =============================================================================

async def record_human_feedback(
    evaluation_id: UUID,
    feedback: HumanFeedbackInput,
) -> bool:
    """
    Record human feedback on an evaluation.

    Returns True if the update succeeded.
    """
    pool = await get_db_pool()

    query = """
        UPDATE evaluations
        SET human_agree = $2,
            human_corrected_verdict = $3,
            human_quality_override = $4,
            human_feedback_text = $5,
            human_corrected_output = $6
        WHERE id = $1
    """

    async with pool.acquire() as conn:
        result = await conn.execute(
            query,
            evaluation_id,
            feedback.agree_with_verdict,
            feedback.corrected_verdict.value if feedback.corrected_verdict else None,
            feedback.quality_override,
            feedback.feedback_text,
            feedback.corrected_output,
        )
        success = result == "UPDATE 1"
        if success:
            logger.info(f"Recorded human feedback for evaluation {evaluation_id}")
        return success


# =============================================================================
# METRICS / AGGREGATION
# =============================================================================

async def get_evaluation_metrics(
    agent_id: Optional[UUID] = None,
    created_after: Optional[datetime] = None,
    created_before: Optional[datetime] = None,
) -> EvaluationMetrics:
    """
    Compute aggregate evaluation metrics, optionally filtered.
    """
    pool = await get_db_pool()

    conditions: List[str] = []
    params: list = []
    param_count = 0

    if agent_id:
        param_count += 1
        conditions.append(f"agent_id = ${param_count}")
        params.append(agent_id)

    if created_after:
        param_count += 1
        conditions.append(f"created_at >= ${param_count}")
        params.append(created_after)

    if created_before:
        param_count += 1
        conditions.append(f"created_at <= ${param_count}")
        params.append(created_before)

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    query = f"""
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE verdict = 'approve') AS approved,
            COUNT(*) FILTER (WHERE verdict = 'retry') AS retried,
            COUNT(*) FILTER (WHERE verdict = 'refine') AS refined,
            COUNT(*) FILTER (WHERE verdict = 'escalate') AS escalated,
            AVG(score_overall) AS avg_overall,
            AVG(score_accuracy) AS avg_accuracy,
            AVG(score_completeness) AS avg_completeness,
            AVG(score_relevance) AS avg_relevance,
            AVG(score_coherence) AS avg_coherence,
            AVG(plan_completion_rate) AS avg_completion_rate,
            AVG(evaluation_duration_ms) FILTER (WHERE evaluation_duration_ms IS NOT NULL) AS avg_eval_dur,
            -- Human agreement
            COUNT(*) FILTER (WHERE human_agree IS NOT NULL) AS human_total,
            COUNT(*) FILTER (WHERE human_agree = TRUE) AS human_agreed,
            MIN(created_at) AS period_start,
            MAX(created_at) AS period_end
        FROM evaluations
        {where_clause}
    """

    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, *params)

        total = row["total"] or 0
        approved = row["approved"] or 0
        retried = row["retried"] or 0
        refined = row["refined"] or 0
        escalated = row["escalated"] or 0
        human_total = row["human_total"] or 0
        human_agreed = row["human_agreed"] or 0

        return EvaluationMetrics(
            total_evaluations=total,
            approved_count=approved,
            retry_count=retried,
            refine_count=refined,
            escalation_count=escalated,
            approval_rate=(approved / total) if total > 0 else 0.0,
            retry_rate=(retried / total) if total > 0 else 0.0,
            escalation_rate=(escalated / total) if total > 0 else 0.0,
            avg_quality_score=float(row["avg_overall"] or 0),
            avg_accuracy=float(row["avg_accuracy"] or 0),
            avg_completeness=float(row["avg_completeness"] or 0),
            avg_relevance=float(row["avg_relevance"] or 0),
            avg_coherence=float(row["avg_coherence"] or 0),
            avg_plan_completion_rate=float(row["avg_completion_rate"] or 0),
            human_agreement_rate=(human_agreed / human_total) if human_total > 0 else None,
            avg_evaluation_duration_ms=float(row["avg_eval_dur"]) if row["avg_eval_dur"] else None,
            period_start=row["period_start"],
            period_end=row["period_end"],
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _row_to_evaluation(row) -> EvaluationResult:
    """Convert a database row into an EvaluationResult model."""
    quality_score = QualityScore(
        accuracy=row["score_accuracy"],
        completeness=row["score_completeness"],
        relevance=row["score_relevance"],
        coherence=row["score_coherence"],
        overall=row["score_overall"],
    )

    verdict = EvaluationVerdict(
        verdict=Verdict(row["verdict"]),
        reasoning=row["verdict_reasoning"],
        confidence=row["verdict_confidence"],
        suggested_improvements=json.loads(row["verdict_improvements"]) if row["verdict_improvements"] else [],
    )

    # Rebuild step evaluations
    raw_steps = json.loads(row["step_evaluations"]) if row["step_evaluations"] else []
    step_evals = [
        StepEvaluation(
            step_index=s.get("step_index", 0),
            status=StepStatus(s.get("status", "not_started")),
            quality_score=s.get("quality_score", 0.0),
            feedback=s.get("feedback", ""),
            meets_criteria=s.get("meets_criteria", False),
        )
        for s in raw_steps
    ]

    plan_completion = PlanCompletionStatus(
        total_steps=row["plan_total_steps"],
        completed_steps=row["plan_completed_steps"],
        partial_steps=row["plan_partial_steps"],
        failed_steps=row["plan_failed_steps"],
        skipped_steps=row["plan_skipped_steps"],
        completion_rate=row["plan_completion_rate"],
        required_steps_met=row["plan_required_met"],
        step_evaluations=step_evals,
    )

    retry_decision = None
    if row["retry_should"] is not None:
        retry_decision = RetryDecision(
            should_retry=row["retry_should"],
            delay_ms=row["retry_delay_ms"] or 0,
            retry_count=row["retry_count"] or 0,
            max_retries=row["retry_max"] or 3,
            escalate=row["retry_escalate"] or False,
            reason=row["retry_reason"] or "",
            adjustments=json.loads(row["retry_adjustments"]) if row["retry_adjustments"] else {},
        )

    human_feedback = None
    if row["human_agree"] is not None:
        human_feedback = HumanFeedbackInput(
            evaluation_id=row["id"],
            agree_with_verdict=row["human_agree"],
            corrected_verdict=Verdict(row["human_corrected_verdict"]) if row["human_corrected_verdict"] else None,
            quality_override=row["human_quality_override"],
            feedback_text=row["human_feedback_text"],
            corrected_output=row["human_corrected_output"],
        )

    return EvaluationResult(
        id=row["id"],
        task_id=row["task_id"],
        agent_id=row["agent_id"],
        quality_score=quality_score,
        verdict=verdict,
        plan_completion=plan_completion,
        feedback=row["feedback"],
        suggested_improvements=json.loads(row["suggested_improvements"]) if row["suggested_improvements"] else [],
        retry_decision=retry_decision,
        human_feedback=human_feedback,
        model_used=row["model_used"],
        execution_duration_ms=row["execution_duration_ms"],
        evaluation_duration_ms=row["evaluation_duration_ms"],
        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        created_at=row["created_at"],
    )
