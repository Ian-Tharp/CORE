"""
Run Repository - PostgreSQL persistence for CORE pipeline runs.

Follows the same asyncpg patterns as conversation_repository.py:
  pool = await get_db_pool()
  async with pool.acquire() as conn: ...

Tables (see init.sql / setup_db_schema):
  core_runs       - one row per pipeline execution
  core_run_events - ordered event log per run (node starts, step results, etc.)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.dependencies import get_db_pool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize(obj: Any) -> str:
    """JSON-serialize with fallback for datetimes / UUIDs."""
    return json.dumps(obj, default=str)


def _row_to_dict(row) -> Dict[str, Any]:
    """Convert an asyncpg Record to a plain dict, serializing datetimes/UUIDs."""
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
        elif isinstance(v, UUID):
            d[k] = str(v)
    return d


# ---------------------------------------------------------------------------
# Run CRUD
# ---------------------------------------------------------------------------

async def create_run(
    *,
    run_id: str,
    user_id: Optional[str] = None,
    input_text: str,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Insert a new run row and return it."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO core_runs (run_id, user_id, input_text, status, state, created_at, updated_at)
            VALUES ($1::uuid, $2, $3, 'running', $4::jsonb, NOW(), NOW())
            RETURNING *
            """,
            run_id,
            user_id,
            input_text,
            _serialize(state) if state else None,
        )
        logger.info("Created run %s", run_id)
        return _row_to_dict(row)


async def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single run by id, or None."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM core_runs WHERE run_id = $1::uuid",
            run_id,
        )
        return _row_to_dict(row) if row else None


async def update_run_status(run_id: str, status: str) -> None:
    """Set status + bump updated_at."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE core_runs
            SET status = $2, updated_at = NOW()
            WHERE run_id = $1::uuid
            """,
            run_id,
            status,
        )


async def update_run_state(run_id: str, state: Dict[str, Any]) -> None:
    """Persist latest COREState snapshot."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE core_runs
            SET state = $2::jsonb, updated_at = NOW()
            WHERE run_id = $1::uuid
            """,
            run_id,
            _serialize(state),
        )


async def complete_run(
    run_id: str,
    result: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
) -> None:
    """Mark a run as completed with its result payload."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE core_runs
            SET status       = 'completed',
                result       = $2::jsonb,
                state        = COALESCE($3::jsonb, state),
                updated_at   = NOW(),
                completed_at = NOW()
            WHERE run_id = $1::uuid
            """,
            run_id,
            _serialize(result),
            _serialize(state) if state else None,
        )
        logger.info("Completed run %s", run_id)


async def fail_run(run_id: str, error: str) -> None:
    """Mark a run as failed with an error message."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE core_runs
            SET status       = 'failed',
                error        = $2,
                updated_at   = NOW(),
                completed_at = NOW()
            WHERE run_id = $1::uuid
            """,
            run_id,
            error,
        )
        logger.warning("Failed run %s: %s", run_id, error)


async def list_runs(
    *,
    user_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """List runs with optional filters, newest first. Paginated."""
    pool = await get_db_pool()

    clauses: List[str] = []
    params: List[Any] = []
    idx = 1

    if user_id is not None:
        clauses.append(f"user_id = ${idx}")
        params.append(user_id)
        idx += 1
    if status is not None:
        clauses.append(f"status = ${idx}")
        params.append(status)
        idx += 1

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    query = f"""
        SELECT run_id, user_id, input_text, status, error,
               created_at, updated_at, completed_at
        FROM core_runs
        {where}
        ORDER BY created_at DESC
        LIMIT ${idx} OFFSET ${idx + 1}
    """
    params.extend([limit, offset])

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
        return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------

async def log_event(
    run_id: str,
    event_type: str,
    step_name: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """Append an event to the run's event log. Non-fatal on failure."""
    pool = await get_db_pool()
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO core_run_events (event_id, run_id, event_type, step_name, data, created_at)
                VALUES (gen_random_uuid(), $1::uuid, $2, $3, $4::jsonb, NOW())
                """,
                run_id,
                event_type,
                step_name,
                _serialize(data) if data else None,
            )
    except Exception as exc:
        # Event logging must never break a run
        logger.error("Failed to log event for run %s: %s", run_id, exc, exc_info=True)


async def get_run_events(run_id: str, *, limit: int = 200) -> List[Dict[str, Any]]:
    """Return events for a run in chronological order."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT event_id, run_id, event_type, step_name, data, created_at
            FROM core_run_events
            WHERE run_id = $1::uuid
            ORDER BY created_at ASC
            LIMIT $2
            """,
            run_id,
            limit,
        )
        return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Stats & cleanup (used by admin controller)
# ---------------------------------------------------------------------------

async def get_run_stats() -> Dict[str, Any]:
    """Aggregate statistics for the admin dashboard."""
    pool = await get_db_pool()
    try:
        async with pool.acquire() as conn:
            total = await conn.fetchval("SELECT COUNT(*) FROM core_runs") or 0
            completed = await conn.fetchval("SELECT COUNT(*) FROM core_runs WHERE status = 'completed'") or 0
            failed = await conn.fetchval("SELECT COUNT(*) FROM core_runs WHERE status = 'failed'") or 0
            recent = await conn.fetchval(
                "SELECT COUNT(*) FROM core_runs WHERE created_at > NOW() - INTERVAL '1 hour'"
            ) or 0
        return {
            "total_runs": total,
            "completed_runs": completed,
            "failed_runs": failed,
            "success_rate": (completed / total) if total > 0 else 0,
            "runs_last_hour": recent,
        }
    except Exception as exc:
        logger.error("Failed to get run stats: %s", exc, exc_info=True)
        return {
            "total_runs": 0,
            "completed_runs": 0,
            "failed_runs": 0,
            "success_rate": 0,
            "runs_last_hour": 0,
            "error": str(exc),
        }


async def cleanup_old_runs(days: int = 30) -> int:
    """Delete runs older than *days*. Returns count deleted."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "DELETE FROM core_runs WHERE created_at < NOW() - make_interval(days => $1) RETURNING run_id",
            days,
        )
        count = len(rows)
        logger.info("Deleted %d runs older than %d days", count, days)
        return count
