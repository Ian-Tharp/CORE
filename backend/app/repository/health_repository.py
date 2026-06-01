"""
Health Repository

Data access layer for health snapshot persistence.

Stores health check results over time for trend analysis,
incident investigation, and SLA monitoring.

Pattern: asyncpg + get_db_pool (no SQLAlchemy).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.dependencies import get_db_pool

logger = logging.getLogger(__name__)


async def record_snapshot(
    overall_status: str,
    services: Dict[str, Dict[str, Any]],
    total_latency_ms: float,
    summary: Dict[str, Any],
) -> Optional[str]:
    """
    Persist a health check snapshot.

    Args:
        overall_status: Aggregated status (healthy/degraded/unhealthy).
        services: Per-service check results dict.
        total_latency_ms: Total wall-clock time for all checks.
        summary: Count summary (healthy/degraded/unhealthy/unknown).

    Returns:
        The snapshot UUID on success, None on failure.
    """
    snapshot_id = str(uuid.uuid4())
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO health_snapshots
                    (id, overall_status, services, total_latency_ms, summary, created_at)
                VALUES ($1, $2, $3::jsonb, $4, $5::jsonb, $6)
                """,
                snapshot_id,
                overall_status,
                json.dumps(services),
                total_latency_ms,
                json.dumps(summary),
                datetime.now(timezone.utc),
            )
        return snapshot_id
    except Exception as e:
        # Health recording should never break the health endpoint itself
        logger.warning(f"Failed to record health snapshot: {e}")
        return None


async def get_history(
    limit: int = 50,
    offset: int = 0,
    status_filter: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Query health snapshot history with optional filters.

    Args:
        limit: Max rows to return (capped at 500).
        offset: Pagination offset.
        status_filter: Filter by overall_status value.
        since: Only snapshots after this timestamp.
        until: Only snapshots before this timestamp.

    Returns:
        List of snapshot dicts ordered newest-first.
    """
    limit = min(limit, 500)
    conditions: List[str] = []
    params: List[Any] = []
    idx = 1

    if status_filter:
        conditions.append(f"overall_status = ${idx}")
        params.append(status_filter)
        idx += 1
    if since:
        conditions.append(f"created_at >= ${idx}")
        params.append(since)
        idx += 1
    if until:
        conditions.append(f"created_at <= ${idx}")
        params.append(until)
        idx += 1

    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""

    params.append(limit)
    limit_idx = idx
    idx += 1
    params.append(offset)
    offset_idx = idx

    query = f"""
        SELECT id, overall_status, services, total_latency_ms, summary, created_at
        FROM health_snapshots
        {where}
        ORDER BY created_at DESC
        LIMIT ${limit_idx} OFFSET ${offset_idx}
    """

    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [
            {
                "id": row["id"],
                "overall_status": row["overall_status"],
                "services": (
                    json.loads(row["services"])
                    if isinstance(row["services"], str)
                    else row["services"]
                ),
                "total_latency_ms": float(row["total_latency_ms"]),
                "summary": (
                    json.loads(row["summary"])
                    if isinstance(row["summary"], str)
                    else row["summary"]
                ),
                "created_at": (
                    row["created_at"].isoformat() if row["created_at"] else None
                ),
            }
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Failed to query health history: {e}")
        return []


async def get_snapshot(snapshot_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single snapshot by id."""
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, overall_status, services, total_latency_ms, summary, created_at "
                "FROM health_snapshots WHERE id = $1",
                snapshot_id,
            )
        if not row:
            return None
        return {
            "id": row["id"],
            "overall_status": row["overall_status"],
            "services": (
                json.loads(row["services"])
                if isinstance(row["services"], str)
                else row["services"]
            ),
            "total_latency_ms": float(row["total_latency_ms"]),
            "summary": (
                json.loads(row["summary"])
                if isinstance(row["summary"], str)
                else row["summary"]
            ),
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }
    except Exception as e:
        logger.error(f"Failed to fetch health snapshot {snapshot_id}: {e}")
        return None


async def get_status_summary(hours: int = 24) -> Dict[str, Any]:
    """
    Get aggregated health status summary over a time window.

    Returns counts by status, average latency, and incident windows
    (periods where status was not healthy).
    """
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            # Counts by status
            status_rows = await conn.fetch(
                """
                SELECT overall_status, COUNT(*) AS cnt
                FROM health_snapshots
                WHERE created_at >= NOW() - make_interval(hours => $1)
                GROUP BY overall_status
                """,
                hours,
            )

            # Average latency
            avg_latency = await conn.fetchval(
                """
                SELECT AVG(total_latency_ms)
                FROM health_snapshots
                WHERE created_at >= NOW() - make_interval(hours => $1)
                """,
                hours,
            )

            # Total snapshots
            total = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM health_snapshots
                WHERE created_at >= NOW() - make_interval(hours => $1)
                """,
                hours,
            )

        by_status = {row["overall_status"]: row["cnt"] for row in status_rows}
        healthy_count = by_status.get("healthy", 0)
        uptime_pct = round(healthy_count / total * 100, 2) if total else 0.0

        return {
            "window_hours": hours,
            "total_checks": total or 0,
            "by_status": by_status,
            "avg_latency_ms": round(float(avg_latency), 2) if avg_latency else 0.0,
            "uptime_pct": uptime_pct,
        }
    except Exception as e:
        logger.error(f"Failed to compute health summary: {e}")
        return {
            "window_hours": hours,
            "total_checks": 0,
            "by_status": {},
            "avg_latency_ms": 0.0,
            "uptime_pct": 0.0,
        }


async def prune_old_snapshots(keep_days: int = 30) -> int:
    """
    Delete health snapshots older than *keep_days*.

    Returns the number of deleted rows.
    """
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM health_snapshots WHERE created_at < NOW() - make_interval(days => $1)",
                keep_days,
            )
            # asyncpg returns "DELETE <n>"
            deleted = int(result.split()[-1])
        logger.info(f"Pruned {deleted} health snapshots older than {keep_days} days")
        return deleted
    except Exception as e:
        logger.error(f"Failed to prune health snapshots: {e}")
        return 0
