"""
Comprehension Repository

Data access layer for comprehension result persistence and analytics.
Handles storing, retrieving, and analyzing comprehension results for
the CORE Comprehension Engine learning loop.
"""

import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from uuid import UUID

from app.dependencies import get_db_pool
from app.models.comprehension_models import (
    ComprehensionResult,
    ComprehensionInput,
    IntentAnalysis,
    ContextMatch,
    CapabilityMatch,
    ComplexityScore,
    ComprehensionFeedback,
    ComprehensionStatus,
    SourceType,
    ActionType,
    UrgencyLevel,
    HandlingMode,
    MemoryMatch,
    MatchedCapability,
    ExtractedEntity,
)

logger = logging.getLogger(__name__)


# =============================================================================
# TABLE INITIALIZATION
# =============================================================================

async def ensure_comprehension_tables() -> None:
    """Create comprehension tables if they don't exist."""
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        # Comprehension results table — stores every analysis result
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS comprehension_results (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                input_id UUID NOT NULL,
                input_content TEXT NOT NULL,
                source_type VARCHAR(50) NOT NULL,
                source_id VARCHAR(255),
                conversation_id VARCHAR(255),
                input_metadata JSONB DEFAULT '{}',
                input_timestamp TIMESTAMP WITH TIME ZONE,

                -- Intent analysis
                action_type VARCHAR(50) NOT NULL,
                intent_summary TEXT NOT NULL,
                entities JSONB DEFAULT '[]',
                urgency VARCHAR(20) NOT NULL DEFAULT 'medium',
                intent_confidence FLOAT NOT NULL DEFAULT 0.5,
                keywords JSONB DEFAULT '[]',
                requires_clarification BOOLEAN DEFAULT FALSE,
                clarification_questions JSONB DEFAULT '[]',

                -- Context matching
                context_matches JSONB DEFAULT '{}',
                has_relevant_context BOOLEAN DEFAULT FALSE,
                best_match_score FLOAT DEFAULT 0.0,

                -- Capability matching
                capability_matches JSONB DEFAULT '{}',
                matched_agent_ids JSONB DEFAULT '[]',
                has_capable_agents BOOLEAN DEFAULT FALSE,
                best_agent_id VARCHAR(255),
                best_agent_score FLOAT DEFAULT 0.0,

                -- Complexity
                complexity JSONB DEFAULT '{}',

                -- Routing suggestions
                suggested_task_type VARCHAR(100) NOT NULL DEFAULT 'general',
                suggested_priority INTEGER NOT NULL DEFAULT 5 CHECK (suggested_priority >= 1 AND suggested_priority <= 10),
                handling_mode VARCHAR(50) NOT NULL DEFAULT 'single_agent',
                suggested_capabilities JSONB DEFAULT '[]',

                -- Meta
                confidence FLOAT NOT NULL DEFAULT 0.5 CHECK (confidence >= 0.0 AND confidence <= 1.0),
                status VARCHAR(50) NOT NULL DEFAULT 'completed',
                processing_time_ms INTEGER,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

                -- Feedback
                feedback_score FLOAT CHECK (feedback_score IS NULL OR (feedback_score >= 0.0 AND feedback_score <= 1.0)),
                feedback_notes TEXT,
                feedback_submitted_at TIMESTAMP WITH TIME ZONE,
                feedback_submitted_by VARCHAR(255)
            )
        """)

        # Indexes for common query patterns
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_comprehension_created ON comprehension_results(created_at DESC)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_comprehension_source_type ON comprehension_results(source_type)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_comprehension_action_type ON comprehension_results(action_type)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_comprehension_status ON comprehension_results(status)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_comprehension_confidence ON comprehension_results(confidence)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_comprehension_conversation ON comprehension_results(conversation_id) WHERE conversation_id IS NOT NULL"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_comprehension_task_type ON comprehension_results(suggested_task_type)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_comprehension_feedback ON comprehension_results(feedback_score) WHERE feedback_score IS NOT NULL"
        )

        # GIN index for full-text search on input content
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_comprehension_input_content_gin ON comprehension_results USING GIN(to_tsvector('english', input_content))"
        )

        logger.info("Comprehension tables ensured")


# =============================================================================
# STORE COMPREHENSION RESULT
# =============================================================================

async def store_comprehension_result(result: ComprehensionResult) -> UUID:
    """
    Store a comprehension result for analytics and caching.

    Args:
        result: ComprehensionResult to persist

    Returns:
        UUID of the stored result
    """
    pool = await get_db_pool()

    query = """
        INSERT INTO comprehension_results (
            id, input_id, input_content, source_type, source_id,
            conversation_id, input_metadata, input_timestamp,
            action_type, intent_summary, entities, urgency,
            intent_confidence, keywords, requires_clarification,
            clarification_questions,
            context_matches, has_relevant_context, best_match_score,
            capability_matches, matched_agent_ids, has_capable_agents,
            best_agent_id, best_agent_score,
            complexity,
            suggested_task_type, suggested_priority, handling_mode,
            suggested_capabilities,
            confidence, status, processing_time_ms, created_at
        ) VALUES (
            $1, $2, $3, $4, $5,
            $6, $7, $8,
            $9, $10, $11, $12,
            $13, $14, $15,
            $16,
            $17, $18, $19,
            $20, $21, $22,
            $23, $24,
            $25,
            $26, $27, $28,
            $29,
            $30, $31, $32, $33
        )
        RETURNING id
    """

    async with pool.acquire() as conn:
        result_id = await conn.fetchval(
            query,
            result.id,
            result.input.id,
            result.input.content,
            result.input.source_type.value,
            result.input.source_id,
            result.input.conversation_id,
            json.dumps(result.input.metadata),
            result.input.timestamp,
            result.intent.action_type.value,
            result.intent.summary,
            json.dumps([e.model_dump() for e in result.intent.entities]),
            result.intent.urgency.value,
            result.intent.confidence,
            json.dumps(result.intent.keywords),
            result.intent.requires_clarification,
            json.dumps(result.intent.clarification_questions),
            json.dumps({
                "semantic_matches": [m.model_dump() for m in result.context.semantic_matches],
                "episodic_matches": [m.model_dump() for m in result.context.episodic_matches],
                "procedural_matches": [m.model_dump() for m in result.context.procedural_matches],
            }),
            result.context.has_relevant_context,
            result.context.best_match_score,
            json.dumps({
                "matched_capabilities": [c.model_dump() for c in result.capabilities.matched_capabilities],
                "matched_tools": result.capabilities.matched_tools,
            }),
            json.dumps(result.capabilities.matched_agent_ids),
            result.capabilities.has_capable_agents,
            result.capabilities.best_agent_id,
            result.capabilities.best_agent_score,
            json.dumps(result.complexity.model_dump()),
            result.suggested_task_type,
            result.suggested_priority,
            result.handling_mode.value,
            json.dumps(result.suggested_capabilities),
            result.confidence,
            result.status.value,
            result.processing_time_ms,
            result.created_at,
        )

        logger.info(f"Stored comprehension result: {result_id}")
        return result_id


# =============================================================================
# RETRIEVE COMPREHENSION RESULTS
# =============================================================================

async def get_comprehension_result(result_id: UUID) -> Optional[ComprehensionResult]:
    """Get a comprehension result by ID."""
    pool = await get_db_pool()

    query = "SELECT * FROM comprehension_results WHERE id = $1"

    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, result_id)
        if not row:
            return None
        return _row_to_comprehension_result(row)


async def list_comprehension_results(
    limit: int = 50,
    offset: int = 0,
    source_type: Optional[str] = None,
    action_type: Optional[str] = None,
    status: Optional[str] = None,
    min_confidence: Optional[float] = None,
    conversation_id: Optional[str] = None,
    created_after: Optional[datetime] = None,
    created_before: Optional[datetime] = None,
) -> Tuple[List[ComprehensionResult], int]:
    """
    List comprehension results with filtering and pagination.

    Returns:
        Tuple of (results, total_count)
    """
    pool = await get_db_pool()

    conditions: List[str] = []
    params: list = []
    param_count = 0

    if source_type:
        param_count += 1
        conditions.append(f"source_type = ${param_count}")
        params.append(source_type)

    if action_type:
        param_count += 1
        conditions.append(f"action_type = ${param_count}")
        params.append(action_type)

    if status:
        param_count += 1
        conditions.append(f"status = ${param_count}")
        params.append(status)

    if min_confidence is not None:
        param_count += 1
        conditions.append(f"confidence >= ${param_count}")
        params.append(min_confidence)

    if conversation_id:
        param_count += 1
        conditions.append(f"conversation_id = ${param_count}")
        params.append(conversation_id)

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

    # Total count
    count_query = f"SELECT COUNT(*) FROM comprehension_results {where_clause}"
    count_params = list(params)

    # Results with pagination
    param_count += 1
    params.append(limit)
    limit_idx = param_count
    param_count += 1
    params.append(offset)
    offset_idx = param_count

    results_query = f"""
        SELECT * FROM comprehension_results
        {where_clause}
        ORDER BY created_at DESC
        LIMIT ${limit_idx} OFFSET ${offset_idx}
    """

    async with pool.acquire() as conn:
        total_count = await conn.fetchval(count_query, *count_params)
        rows = await conn.fetch(results_query, *params)
        results = [_row_to_comprehension_result(row) for row in rows]
        return results, total_count or 0


async def find_similar_comprehension(
    input_content: str,
    max_age_hours: int = 24,
    limit: int = 5,
) -> List[ComprehensionResult]:
    """
    Find past comprehension results for similar inputs (dedup/caching).

    Uses PostgreSQL full-text search to find similar past inputs.
    """
    pool = await get_db_pool()

    cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)

    query = """
        SELECT *, ts_rank(
            to_tsvector('english', input_content),
            plainto_tsquery('english', $1)
        ) AS rank
        FROM comprehension_results
        WHERE created_at > $2
          AND status = 'completed'
          AND to_tsvector('english', input_content) @@ plainto_tsquery('english', $1)
        ORDER BY rank DESC
        LIMIT $3
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, input_content, cutoff, limit)
        return [_row_to_comprehension_result(row) for row in rows]


# =============================================================================
# FEEDBACK
# =============================================================================

async def store_feedback(feedback: ComprehensionFeedback) -> bool:
    """
    Store feedback on a comprehension result.

    Updates the existing comprehension result with feedback data.
    """
    pool = await get_db_pool()

    query = """
        UPDATE comprehension_results
        SET feedback_score = $2,
            feedback_notes = $3,
            feedback_submitted_at = $4,
            feedback_submitted_by = $5
        WHERE id = $1
        RETURNING id
    """

    async with pool.acquire() as conn:
        result = await conn.fetchval(
            query,
            feedback.comprehension_id,
            feedback.score,
            feedback.notes,
            feedback.submitted_at,
            feedback.submitted_by,
        )
        if result:
            logger.info(f"Stored feedback for comprehension {feedback.comprehension_id}: {feedback.score}")
            return True
        return False


# =============================================================================
# ANALYTICS
# =============================================================================

async def get_comprehension_analytics(
    days: int = 30,
) -> Dict[str, Any]:
    """
    Get comprehension analytics over a time window.

    Returns accuracy metrics, action type distribution, processing times, etc.
    """
    pool = await get_db_pool()
    cutoff = datetime.utcnow() - timedelta(days=days)

    query = """
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE status = 'completed') AS completed,
            COUNT(*) FILTER (WHERE status = 'failed') AS failed,
            COUNT(*) FILTER (WHERE status = 'partial') AS partial,
            AVG(confidence) AS avg_confidence,
            AVG(processing_time_ms) FILTER (WHERE processing_time_ms IS NOT NULL) AS avg_processing_time_ms,
            AVG(feedback_score) FILTER (WHERE feedback_score IS NOT NULL) AS avg_feedback_score,
            COUNT(*) FILTER (WHERE feedback_score IS NOT NULL) AS feedback_count,
            COUNT(DISTINCT conversation_id) FILTER (WHERE conversation_id IS NOT NULL) AS unique_conversations,
            COUNT(DISTINCT source_id) FILTER (WHERE source_id IS NOT NULL) AS unique_sources
        FROM comprehension_results
        WHERE created_at > $1
    """

    action_type_query = """
        SELECT action_type, COUNT(*) AS count
        FROM comprehension_results
        WHERE created_at > $1
        GROUP BY action_type
        ORDER BY count DESC
    """

    source_type_query = """
        SELECT source_type, COUNT(*) AS count
        FROM comprehension_results
        WHERE created_at > $1
        GROUP BY source_type
        ORDER BY count DESC
    """

    handling_mode_query = """
        SELECT handling_mode, COUNT(*) AS count
        FROM comprehension_results
        WHERE created_at > $1
        GROUP BY handling_mode
        ORDER BY count DESC
    """

    async with pool.acquire() as conn:
        overview = await conn.fetchrow(query, cutoff)
        action_rows = await conn.fetch(action_type_query, cutoff)
        source_rows = await conn.fetch(source_type_query, cutoff)
        handling_rows = await conn.fetch(handling_mode_query, cutoff)

        return {
            "period_days": days,
            "overview": {
                "total": overview["total"] or 0,
                "completed": overview["completed"] or 0,
                "failed": overview["failed"] or 0,
                "partial": overview["partial"] or 0,
                "avg_confidence": float(overview["avg_confidence"]) if overview["avg_confidence"] else 0.0,
                "avg_processing_time_ms": float(overview["avg_processing_time_ms"]) if overview["avg_processing_time_ms"] else None,
                "avg_feedback_score": float(overview["avg_feedback_score"]) if overview["avg_feedback_score"] else None,
                "feedback_count": overview["feedback_count"] or 0,
                "unique_conversations": overview["unique_conversations"] or 0,
                "unique_sources": overview["unique_sources"] or 0,
            },
            "action_type_distribution": {row["action_type"]: row["count"] for row in action_rows},
            "source_type_distribution": {row["source_type"]: row["count"] for row in source_rows},
            "handling_mode_distribution": {row["handling_mode"]: row["count"] for row in handling_rows},
        }


async def get_accuracy_over_time(
    days: int = 30,
    bucket_days: int = 1,
) -> List[Dict[str, Any]]:
    """
    Get comprehension accuracy over time for trend analysis.
    
    Groups results into time buckets and returns average confidence
    and feedback scores per bucket.
    """
    pool = await get_db_pool()
    cutoff = datetime.utcnow() - timedelta(days=days)

    query = """
        SELECT
            date_trunc('day', created_at) AS bucket,
            COUNT(*) AS total,
            AVG(confidence) AS avg_confidence,
            AVG(feedback_score) FILTER (WHERE feedback_score IS NOT NULL) AS avg_feedback,
            COUNT(*) FILTER (WHERE feedback_score IS NOT NULL) AS feedback_count
        FROM comprehension_results
        WHERE created_at > $1
        GROUP BY bucket
        ORDER BY bucket ASC
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, cutoff)
        return [
            {
                "date": row["bucket"].isoformat() if row["bucket"] else None,
                "total": row["total"],
                "avg_confidence": float(row["avg_confidence"]) if row["avg_confidence"] else 0.0,
                "avg_feedback": float(row["avg_feedback"]) if row["avg_feedback"] else None,
                "feedback_count": row["feedback_count"] or 0,
            }
            for row in rows
        ]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _row_to_comprehension_result(row) -> ComprehensionResult:
    """Convert a database row to a ComprehensionResult model."""
    # Parse JSON fields
    entities_raw = _parse_json(row["entities"], [])
    keywords = _parse_json(row["keywords"], [])
    clarification_questions = _parse_json(row["clarification_questions"], [])
    context_raw = _parse_json(row["context_matches"], {})
    capability_raw = _parse_json(row["capability_matches"], {})
    matched_agent_ids = _parse_json(row["matched_agent_ids"], [])
    complexity_raw = _parse_json(row["complexity"], {})
    suggested_capabilities = _parse_json(row["suggested_capabilities"], [])
    input_metadata = _parse_json(row["input_metadata"], {})

    # Build entities
    entities = []
    for e in entities_raw:
        if isinstance(e, dict):
            entities.append(ExtractedEntity(**e))

    # Build context matches
    semantic_matches = []
    for m in context_raw.get("semantic_matches", []):
        if isinstance(m, dict):
            # UUID fields need conversion
            m["memory_id"] = m.get("memory_id", str(uuid4_safe()))
            semantic_matches.append(MemoryMatch(**m))

    episodic_matches = []
    for m in context_raw.get("episodic_matches", []):
        if isinstance(m, dict):
            m["memory_id"] = m.get("memory_id", str(uuid4_safe()))
            episodic_matches.append(MemoryMatch(**m))

    procedural_matches = []
    for m in context_raw.get("procedural_matches", []):
        if isinstance(m, dict):
            m["memory_id"] = m.get("memory_id", str(uuid4_safe()))
            procedural_matches.append(MemoryMatch(**m))

    # Build capability matches
    matched_capabilities = []
    for c in capability_raw.get("matched_capabilities", []):
        if isinstance(c, dict):
            matched_capabilities.append(MatchedCapability(**c))

    matched_tools = capability_raw.get("matched_tools", [])

    # Build complexity
    complexity = ComplexityScore(
        overall=complexity_raw.get("overall", 0.5),
        reasoning_depth=complexity_raw.get("reasoning_depth", 0.5),
        breadth=complexity_raw.get("breadth", 0.5),
        novelty=complexity_raw.get("novelty", 0.5),
        estimated_duration_seconds=complexity_raw.get("estimated_duration_seconds"),
    )

    return ComprehensionResult(
        id=row["id"],
        input=ComprehensionInput(
            id=row["input_id"],
            content=row["input_content"],
            source_type=SourceType(row["source_type"]),
            source_id=row["source_id"],
            conversation_id=row["conversation_id"],
            metadata=input_metadata,
            timestamp=row["input_timestamp"] or datetime.utcnow(),
        ),
        intent=IntentAnalysis(
            action_type=ActionType(row["action_type"]),
            summary=row["intent_summary"],
            entities=entities,
            urgency=UrgencyLevel(row["urgency"]),
            confidence=row["intent_confidence"],
            keywords=keywords,
            requires_clarification=row["requires_clarification"],
            clarification_questions=clarification_questions,
        ),
        context=ContextMatch(
            semantic_matches=semantic_matches,
            episodic_matches=episodic_matches,
            procedural_matches=procedural_matches,
            has_relevant_context=row["has_relevant_context"],
            best_match_score=row["best_match_score"],
        ),
        capabilities=CapabilityMatch(
            matched_capabilities=matched_capabilities,
            matched_agent_ids=matched_agent_ids,
            matched_tools=matched_tools,
            has_capable_agents=row["has_capable_agents"],
            best_agent_id=row["best_agent_id"],
            best_agent_score=row["best_agent_score"],
        ),
        complexity=complexity,
        suggested_task_type=row["suggested_task_type"],
        suggested_priority=row["suggested_priority"],
        handling_mode=HandlingMode(row["handling_mode"]),
        suggested_capabilities=suggested_capabilities,
        confidence=row["confidence"],
        status=ComprehensionStatus(row["status"]),
        processing_time_ms=row["processing_time_ms"],
        created_at=row["created_at"],
        feedback_score=row["feedback_score"],
        feedback_notes=row["feedback_notes"],
    )


def _parse_json(value: Any, default: Any = None) -> Any:
    """Parse a JSON value that might be a string or already parsed."""
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return default
    return value


def uuid4_safe():
    """Generate a UUID4, import-safe."""
    from uuid import uuid4
    return uuid4()
