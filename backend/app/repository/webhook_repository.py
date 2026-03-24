"""
Webhook Repository

Data access layer for webhook registration and delivery persistence.

Pattern: asyncpg + get_db_pool (no SQLAlchemy).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.dependencies import get_db_pool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Webhook Registrations
# ---------------------------------------------------------------------------


async def create_registration(
    webhook_id: str,
    name: str,
    url: str,
    events: List[str],
    secret: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    max_retries: int = 3,
) -> Optional[str]:
    """Persist a new webhook registration. Returns the id on success."""
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO webhook_registrations
                    (id, name, url, events, secret, headers, max_retries, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $8)
                """,
                webhook_id,
                name,
                url,
                events,
                secret,
                json.dumps(headers or {}),
                max_retries,
                datetime.now(timezone.utc),
            )
        return webhook_id
    except Exception as e:
        logger.error(f"Failed to create webhook registration: {e}")
        return None


async def get_registration(webhook_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single webhook registration by id."""
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM webhook_registrations WHERE id = $1",
                webhook_id,
            )
        if not row:
            return None
        return _row_to_dict(row)
    except Exception as e:
        logger.error(f"Failed to fetch webhook {webhook_id}: {e}")
        return None


async def list_registrations(
    active_only: bool = False,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """List webhook registrations with optional active filter."""
    try:
        pool = await get_db_pool()
        conditions = []
        params: list = []
        idx = 1

        if active_only:
            conditions.append(f"is_active = ${idx}")
            params.append(True)
            idx += 1

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""

        params.append(min(limit, 500))
        limit_idx = idx
        idx += 1
        params.append(offset)
        offset_idx = idx

        query = f"""
            SELECT * FROM webhook_registrations
            {where}
            ORDER BY created_at DESC
            LIMIT ${limit_idx} OFFSET ${offset_idx}
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [_row_to_dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to list webhook registrations: {e}")
        return []


async def update_registration(
    webhook_id: str,
    **fields,
) -> bool:
    """Update specific fields on a webhook registration."""
    allowed = {"name", "url", "events", "secret", "headers", "max_retries", "is_active"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return False

    set_clauses = []
    params: list = []
    idx = 1

    for key, value in updates.items():
        if key == "headers":
            set_clauses.append(f"{key} = ${idx}::jsonb")
            params.append(json.dumps(value))
        else:
            set_clauses.append(f"{key} = ${idx}")
            params.append(value)
        idx += 1

    set_clauses.append(f"updated_at = ${idx}")
    params.append(datetime.now(timezone.utc))
    idx += 1

    params.append(webhook_id)

    query = f"""
        UPDATE webhook_registrations
        SET {", ".join(set_clauses)}
        WHERE id = ${idx}
    """

    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(query, *params)
        return result.endswith("1")
    except Exception as e:
        logger.error(f"Failed to update webhook {webhook_id}: {e}")
        return False


async def delete_registration(webhook_id: str) -> bool:
    """Delete a webhook registration (cascades to deliveries)."""
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM webhook_registrations WHERE id = $1",
                webhook_id,
            )
        return result.endswith("1")
    except Exception as e:
        logger.error(f"Failed to delete webhook {webhook_id}: {e}")
        return False


async def increment_delivery_count(webhook_id: str) -> None:
    """Increment successful delivery counter and update last_delivery_at."""
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE webhook_registrations
                SET delivery_count = delivery_count + 1,
                    last_delivery_at = $2,
                    updated_at = $2
                WHERE id = $1
                """,
                webhook_id,
                datetime.now(timezone.utc),
            )
    except Exception as e:
        logger.warning(f"Failed to increment delivery count for {webhook_id}: {e}")


async def increment_failure_count(webhook_id: str, error: str) -> None:
    """Increment failure counter and set last_error."""
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE webhook_registrations
                SET failure_count = failure_count + 1,
                    last_error = $2,
                    updated_at = $3
                WHERE id = $1
                """,
                webhook_id,
                error[:1000],
                datetime.now(timezone.utc),
            )
    except Exception as e:
        logger.warning(f"Failed to increment failure count for {webhook_id}: {e}")


# ---------------------------------------------------------------------------
# Webhook Deliveries
# ---------------------------------------------------------------------------


async def record_delivery(
    delivery_id: str,
    webhook_id: str,
    event: str,
    payload: Dict[str, Any],
    attempts: int = 0,
    status_code: Optional[int] = None,
    response_body: Optional[str] = None,
    error: Optional[str] = None,
    delivered_at: Optional[datetime] = None,
) -> Optional[str]:
    """Record a webhook delivery attempt."""
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO webhook_deliveries
                    (id, webhook_id, event, payload, attempts, status_code,
                     response_body, error, delivered_at, created_at)
                VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8, $9, $10)
                """,
                delivery_id,
                webhook_id,
                event,
                json.dumps(payload),
                attempts,
                status_code,
                response_body[:500] if response_body else None,
                error[:1000] if error else None,
                delivered_at,
                datetime.now(timezone.utc),
            )
        return delivery_id
    except Exception as e:
        logger.warning(f"Failed to record webhook delivery: {e}")
        return None


async def get_deliveries(
    webhook_id: Optional[str] = None,
    event_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Query delivery history with optional filters."""
    try:
        pool = await get_db_pool()
        conditions = []
        params: list = []
        idx = 1

        if webhook_id:
            conditions.append(f"webhook_id = ${idx}")
            params.append(webhook_id)
            idx += 1
        if event_filter:
            conditions.append(f"event = ${idx}")
            params.append(event_filter)
            idx += 1

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""

        params.append(min(limit, 500))
        limit_idx = idx
        idx += 1
        params.append(offset)
        offset_idx = idx

        query = f"""
            SELECT * FROM webhook_deliveries
            {where}
            ORDER BY created_at DESC
            LIMIT ${limit_idx} OFFSET ${offset_idx}
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [_delivery_to_dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to query webhook deliveries: {e}")
        return []


async def prune_old_deliveries(keep_days: int = 30) -> int:
    """Delete delivery records older than keep_days. Returns deleted count."""
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM webhook_deliveries WHERE created_at < NOW() - make_interval(days => $1)",
                keep_days,
            )
            deleted = int(result.split()[-1])
        logger.info(f"Pruned {deleted} webhook deliveries older than {keep_days} days")
        return deleted
    except Exception as e:
        logger.error(f"Failed to prune webhook deliveries: {e}")
        return 0


async def get_delivery_stats(hours: int = 24) -> Dict[str, Any]:
    """Get delivery statistics over a time window."""
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            total = await conn.fetchval(
                """
                SELECT COUNT(*) FROM webhook_deliveries
                WHERE created_at >= NOW() - make_interval(hours => $1)
                """,
                hours,
            )
            successful = await conn.fetchval(
                """
                SELECT COUNT(*) FROM webhook_deliveries
                WHERE created_at >= NOW() - make_interval(hours => $1)
                  AND delivered_at IS NOT NULL
                """,
                hours,
            )
            by_event = await conn.fetch(
                """
                SELECT event, COUNT(*) as cnt
                FROM webhook_deliveries
                WHERE created_at >= NOW() - make_interval(hours => $1)
                GROUP BY event
                """,
                hours,
            )

        total = total or 0
        successful = successful or 0
        return {
            "window_hours": hours,
            "total_deliveries": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": round(successful / total, 4) if total > 0 else 0.0,
            "by_event": {row["event"]: row["cnt"] for row in by_event},
        }
    except Exception as e:
        logger.error(f"Failed to compute delivery stats: {e}")
        return {"window_hours": hours, "total_deliveries": 0, "successful": 0, "failed": 0, "success_rate": 0.0, "by_event": {}}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row) -> Dict[str, Any]:
    """Convert a webhook_registrations row to dict."""
    headers = row["headers"]
    if isinstance(headers, str):
        headers = json.loads(headers)
    return {
        "id": row["id"],
        "name": row["name"],
        "url": row["url"],
        "events": list(row["events"]),
        "secret": row["secret"],
        "headers": headers,
        "max_retries": row["max_retries"],
        "is_active": row["is_active"],
        "delivery_count": row["delivery_count"],
        "failure_count": row["failure_count"],
        "last_delivery_at": row["last_delivery_at"].isoformat() if row["last_delivery_at"] else None,
        "last_error": row["last_error"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
    }


def _delivery_to_dict(row) -> Dict[str, Any]:
    """Convert a webhook_deliveries row to dict."""
    payload = row["payload"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return {
        "id": row["id"],
        "webhook_id": row["webhook_id"],
        "event": row["event"],
        "payload": payload,
        "attempts": row["attempts"],
        "status_code": row["status_code"],
        "response_body": row["response_body"],
        "error": row["error"],
        "delivered_at": row["delivered_at"].isoformat() if row["delivered_at"] else None,
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
    }