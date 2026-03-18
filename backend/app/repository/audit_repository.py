"""
Audit Repository

Data access layer for the audit log — persistent record of security-sensitive
and administrative operations for compliance, debugging, and forensics.

Pattern: asyncpg + get_db_pool (no SQLAlchemy).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from app.dependencies import get_db_pool

logger = logging.getLogger(__name__)


# =============================================================================
# TABLE INITIALISATION
# =============================================================================

async def ensure_audit_tables() -> None:
    """Create audit log tables (idempotent)."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    actor VARCHAR(255) NOT NULL,
                    action VARCHAR(128) NOT NULL,
                    resource_type VARCHAR(128),
                    resource_id VARCHAR(255),
                    detail JSONB DEFAULT '{}',
                    ip_address VARCHAR(45),
                    correlation_id VARCHAR(64),
                    outcome VARCHAR(32) NOT NULL DEFAULT 'success'
                )
                """
            )
            # Indexes for common query patterns
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp DESC)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_log_actor ON audit_log(actor)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_log_resource ON audit_log(resource_type, resource_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_log_outcome ON audit_log(outcome)"
            )
    logger.info("Audit log tables ensured")


# =============================================================================
# WRITE
# =============================================================================

async def record(
    *,
    actor: str,
    action: str,
    resource_type: str | None = None,
    resource_id: str | None = None,
    detail: Dict[str, Any] | None = None,
    ip_address: str | None = None,
    correlation_id: str | None = None,
    outcome: str = "success",
) -> Optional[str]:
    """
    Persist a single audit event.

    Args:
        actor: Who performed the action (API key name, agent id, "system").
        action: Verb describing the operation (e.g. "api_key.create").
        resource_type: Optional type of resource affected ("api_key", "webhook", etc.).
        resource_id: Optional identifier of the resource.
        detail: Arbitrary JSON metadata (redacted values, before/after, etc.).
        ip_address: Client IP when available.
        correlation_id: Request correlation id for tracing.
        outcome: "success", "failure", "denied".

    Returns:
        The audit event UUID on success, None on failure.
    """
    event_id = str(uuid.uuid4())
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO audit_log
                    (id, actor, action, resource_type, resource_id,
                     detail, ip_address, correlation_id, outcome)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9)
                """,
                uuid.UUID(event_id),
                actor,
                action,
                resource_type,
                resource_id,
                json.dumps(detail or {}),
                ip_address,
                correlation_id,
                outcome,
            )
        return event_id
    except Exception as exc:
        logger.error("Failed to record audit event: %s", exc)
        return None


# =============================================================================
# READ
# =============================================================================

async def get_events(
    *,
    actor: str | None = None,
    action: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    outcome: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Query audit events with optional filters.

    Returns a list of dicts ordered by timestamp descending.
    """
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            clauses: list[str] = []
            params: list[Any] = []
            idx = 1

            if actor:
                clauses.append(f"actor = ${idx}")
                params.append(actor)
                idx += 1
            if action:
                clauses.append(f"action = ${idx}")
                params.append(action)
                idx += 1
            if resource_type:
                clauses.append(f"resource_type = ${idx}")
                params.append(resource_type)
                idx += 1
            if resource_id:
                clauses.append(f"resource_id = ${idx}")
                params.append(resource_id)
                idx += 1
            if outcome:
                clauses.append(f"outcome = ${idx}")
                params.append(outcome)
                idx += 1
            if since:
                clauses.append(f"timestamp >= ${idx}")
                params.append(since)
                idx += 1
            if until:
                clauses.append(f"timestamp <= ${idx}")
                params.append(until)
                idx += 1

            where = "WHERE " + " AND ".join(clauses) if clauses else ""
            params.extend([limit, offset])

            rows = await conn.fetch(
                f"""
                SELECT id, timestamp, actor, action, resource_type,
                       resource_id, detail, ip_address, correlation_id, outcome
                FROM audit_log
                {where}
                ORDER BY timestamp DESC
                LIMIT ${idx} OFFSET ${idx + 1}
                """,
                *params,
            )

            return [
                {
                    "id": str(row["id"]),
                    "timestamp": row["timestamp"].isoformat(),
                    "actor": row["actor"],
                    "action": row["action"],
                    "resource_type": row["resource_type"],
                    "resource_id": row["resource_id"],
                    "detail": json.loads(row["detail"]) if row["detail"] else {},
                    "ip_address": row["ip_address"],
                    "correlation_id": row["correlation_id"],
                    "outcome": row["outcome"],
                }
                for row in rows
            ]
    except Exception as exc:
        logger.error("Failed to query audit events: %s", exc)
        return []


async def count_events(
    *,
    actor: str | None = None,
    action: str | None = None,
    outcome: str | None = None,
    since: datetime | None = None,
) -> int:
    """Count audit events matching the given filters."""
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            clauses: list[str] = []
            params: list[Any] = []
            idx = 1

            if actor:
                clauses.append(f"actor = ${idx}")
                params.append(actor)
                idx += 1
            if action:
                clauses.append(f"action = ${idx}")
                params.append(action)
                idx += 1
            if outcome:
                clauses.append(f"outcome = ${idx}")
                params.append(outcome)
                idx += 1
            if since:
                clauses.append(f"timestamp >= ${idx}")
                params.append(since)
                idx += 1

            where = "WHERE " + " AND ".join(clauses) if clauses else ""

            result = await conn.fetchval(
                f"SELECT COUNT(*) FROM audit_log {where}",
                *params,
            )
            return result or 0
    except Exception as exc:
        logger.error("Failed to count audit events: %s", exc)
        return 0


async def get_summary(hours: int = 24) -> Dict[str, Any]:
    """
    Aggregated audit summary over a time window.

    Returns counts by action, actor, and outcome.
    """
    try:
        pool = await get_db_pool()
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        async with pool.acquire() as conn:
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM audit_log WHERE timestamp >= $1",
                since,
            )

            by_action = await conn.fetch(
                """
                SELECT action, COUNT(*) AS cnt
                FROM audit_log WHERE timestamp >= $1
                GROUP BY action ORDER BY cnt DESC
                """,
                since,
            )

            by_actor = await conn.fetch(
                """
                SELECT actor, COUNT(*) AS cnt
                FROM audit_log WHERE timestamp >= $1
                GROUP BY actor ORDER BY cnt DESC LIMIT 20
                """,
                since,
            )

            by_outcome = await conn.fetch(
                """
                SELECT outcome, COUNT(*) AS cnt
                FROM audit_log WHERE timestamp >= $1
                GROUP BY outcome ORDER BY cnt DESC
                """,
                since,
            )

            return {
                "period_hours": hours,
                "total_events": total or 0,
                "by_action": {row["action"]: row["cnt"] for row in by_action},
                "by_actor": {row["actor"]: row["cnt"] for row in by_actor},
                "by_outcome": {row["outcome"]: row["cnt"] for row in by_outcome},
            }
    except Exception as exc:
        logger.error("Failed to generate audit summary: %s", exc)
        return {"period_hours": hours, "total_events": 0, "by_action": {}, "by_actor": {}, "by_outcome": {}}


async def prune_old_events(keep_days: int = 90) -> int:
    """
    Delete audit events older than *keep_days*.

    Returns the number of deleted rows.
    """
    try:
        pool = await get_db_pool()
        cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM audit_log WHERE timestamp < $1",
                cutoff,
            )
            deleted = int(result.split()[-1])
            if deleted:
                logger.info("Pruned %d audit events older than %d days", deleted, keep_days)
            return deleted
    except Exception as exc:
        logger.error("Failed to prune audit events: %s", exc)
        return 0