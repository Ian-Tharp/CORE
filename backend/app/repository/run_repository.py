"""
Run Repository

Data access layer for CORE run persistence.
Handles storing, retrieving, and updating run state in PostgreSQL.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID

from app.dependencies import get_db_pool
from app.models.core_state import COREState

logger = logging.getLogger(__name__)


async def ensure_runs_table() -> None:
    """Create the runs table if it doesn't exist."""
    pool = await get_db_pool()
    
    async with pool.acquire() as conn:
        # Create main runs table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS core_runs (
                id SERIAL PRIMARY KEY,
                run_id VARCHAR(255) UNIQUE NOT NULL,
                conversation_id VARCHAR(255),
                user_id VARCHAR(255),
                user_input TEXT NOT NULL,
                state JSONB,
                status VARCHAR(50) DEFAULT 'pending',
                current_node VARCHAR(100),
                response TEXT,
                error TEXT,
                config JSONB DEFAULT '{}',
                execution_history JSONB DEFAULT '[]',
                step_results JSONB DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP
            )
        """)
        
        # Create events table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS core_run_events (
                event_id SERIAL PRIMARY KEY,
                run_id VARCHAR(255) NOT NULL REFERENCES core_runs(run_id) ON DELETE CASCADE,
                event_type VARCHAR(50) NOT NULL,
                event_data JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_core_runs_status ON core_runs(status)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_core_runs_user ON core_runs(user_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_core_runs_created ON core_runs(created_at DESC)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_core_run_events_run ON core_run_events(run_id)"
        )
        
        logger.info("Core runs tables ensured")


async def create_run(state: COREState) -> str:
    """
    Create a new run record in the database.
    
    Args:
        state: Initial COREState
        
    Returns:
        run_id of the created run
    """
    pool = await get_db_pool()
    
    query = """
        INSERT INTO core_runs (
            run_id, conversation_id, user_id, user_input,
            state, status, current_node, config, started_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9
        )
        RETURNING run_id
    """
    
    try:
        async with pool.acquire() as conn:
            result = await conn.fetchval(
                query,
                state.run_id,
                state.conversation_id,
                state.user_id,
                state.user_input,
                json.dumps(state.model_dump(), default=str),
                "running",
                state.current_node,
                json.dumps(state.config),
                datetime.utcnow()
            )
            
            logger.info(f"Created run {result}")
            return str(result)
            
    except Exception as e:
        logger.error(f"Failed to create run: {e}", exc_info=True)
        raise


async def update_run(
    run_id: str,
    state: Optional[COREState] = None,
    status: Optional[str] = None,
    current_node: Optional[str] = None,
    response: Optional[str] = None,
    error: Optional[str] = None,
    step_results: Optional[List[Dict]] = None,
    completed: bool = False
) -> None:
    """
    Update an existing run record.
    
    Args:
        run_id: Run to update
        state: Updated COREState (optional)
        status: New status (optional)
        current_node: Current execution node (optional)
        response: Final response text (optional)
        error: Error message if failed (optional)
        step_results: Updated step results (optional)
        completed: Whether run is complete (sets completed_at)
    """
    pool = await get_db_pool()
    
    # Build dynamic update
    updates = []
    params = []
    param_count = 1
    
    if state is not None:
        updates.append(f"state = ${param_count}")
        params.append(json.dumps(state.model_dump(), default=str))
        param_count += 1
        
        updates.append(f"execution_history = ${param_count}")
        params.append(json.dumps(state.execution_history))
        param_count += 1
    
    if status is not None:
        updates.append(f"status = ${param_count}")
        params.append(status)
        param_count += 1
    
    if current_node is not None:
        updates.append(f"current_node = ${param_count}")
        params.append(current_node)
        param_count += 1
    
    if response is not None:
        updates.append(f"response = ${param_count}")
        params.append(response)
        param_count += 1
    
    if error is not None:
        updates.append(f"error = ${param_count}")
        params.append(error)
        param_count += 1
    
    if step_results is not None:
        updates.append(f"step_results = ${param_count}")
        params.append(json.dumps(step_results, default=str))
        param_count += 1
    
    if completed:
        updates.append(f"completed_at = ${param_count}")
        params.append(datetime.utcnow())
        param_count += 1
    
    if not updates:
        return
    
    # Add run_id as final parameter
    params.append(run_id)
    
    query = f"""
        UPDATE core_runs
        SET {', '.join(updates)}
        WHERE run_id = ${param_count}
    """
    
    try:
        async with pool.acquire() as conn:
            await conn.execute(query, *params)
            logger.debug(f"Updated run {run_id}")
            
    except Exception as e:
        logger.error(f"Failed to update run {run_id}: {e}", exc_info=True)
        raise


async def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a run by ID.
    
    Args:
        run_id: Run identifier
        
    Returns:
        Run record as dict, or None if not found
    """
    pool = await get_db_pool()
    
    query = """
        SELECT 
            run_id, conversation_id, user_id, user_input,
            state, status, current_node, response, error,
            config, execution_history, step_results,
            created_at, updated_at, started_at, completed_at
        FROM core_runs
        WHERE run_id = $1
    """
    
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, run_id)
            
            if row is None:
                return None
            
            return dict(row)
            
    except Exception as e:
        logger.error(f"Failed to get run {run_id}: {e}", exc_info=True)
        raise


async def list_runs(
    user_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    List runs with optional filtering.
    
    Args:
        user_id: Filter by user
        status: Filter by status
        limit: Maximum results
        offset: Pagination offset
        
    Returns:
        List of run records
    """
    pool = await get_db_pool()
    
    query_parts = ["SELECT run_id, user_id, user_input, status, current_node, created_at, completed_at FROM core_runs"]
    conditions = []
    params = []
    param_count = 1
    
    if user_id:
        conditions.append(f"user_id = ${param_count}")
        params.append(user_id)
        param_count += 1
    
    if status:
        conditions.append(f"status = ${param_count}")
        params.append(status)
        param_count += 1
    
    if conditions:
        query_parts.append("WHERE " + " AND ".join(conditions))
    
    query_parts.append("ORDER BY created_at DESC")
    query_parts.append(f"LIMIT ${param_count}")
    params.append(limit)
    param_count += 1
    
    query_parts.append(f"OFFSET ${param_count}")
    params.append(offset)
    
    query = " ".join(query_parts)
    
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
            
    except Exception as e:
        logger.error(f"Failed to list runs: {e}", exc_info=True)
        raise


async def add_run_event(
    run_id: str,
    event_type: str,
    event_data: Dict[str, Any]
) -> None:
    """
    Add an event to the run event log.
    
    Args:
        run_id: Run to add event to
        event_type: Type of event (node_start, step_executed, etc.)
        event_data: Event payload
    """
    pool = await get_db_pool()
    
    query = """
        INSERT INTO core_run_events (run_id, event_type, event_data)
        VALUES ($1, $2, $3)
    """
    
    try:
        async with pool.acquire() as conn:
            await conn.execute(query, run_id, event_type, json.dumps(event_data, default=str))
            
    except Exception as e:
        logger.error(f"Failed to add run event: {e}", exc_info=True)
        # Don't raise - event logging shouldn't fail the run


async def get_run_events(run_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get events for a run.
    
    Args:
        run_id: Run identifier
        limit: Maximum events to return
        
    Returns:
        List of events in chronological order
    """
    pool = await get_db_pool()
    
    query = """
        SELECT event_id, event_type, event_data, created_at
        FROM core_run_events
        WHERE run_id = $1
        ORDER BY created_at ASC
        LIMIT $2
    """
    
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, run_id, limit)
            return [dict(row) for row in rows]
            
    except Exception as e:
        logger.error(f"Failed to get run events: {e}", exc_info=True)
        raise


async def delete_old_runs(days: int = 30) -> int:
    """
    Delete runs older than specified days.
    
    Args:
        days: Delete runs older than this many days
        
    Returns:
        Number of runs deleted
    """
    pool = await get_db_pool()
    
    query = """
        DELETE FROM core_runs
        WHERE created_at < NOW() - INTERVAL '%s days'
        RETURNING run_id
    """
    
    try:
        async with pool.acquire() as conn:
            result = await conn.fetch(query % days)
            count = len(result)
            logger.info(f"Deleted {count} old runs (older than {days} days)")
            return count
            
    except Exception as e:
        logger.error(f"Failed to delete old runs: {e}", exc_info=True)
        raise


async def save_run(run_data: Dict[str, Any]) -> str:
    """
    Save a new run or update existing (upsert).
    
    This is a convenience function that handles both create and update.
    Used by the engine controller for simple persistence.
    
    Args:
        run_data: COREState serialized to dict
        
    Returns:
        run_id
    """
    pool = await get_db_pool()
    
    run_id = run_data.get("run_id")
    if not run_id:
        raise ValueError("run_id is required")
    
    # Extract fields
    intent = run_data.get("intent") or {}
    plan = run_data.get("plan") or {}
    
    # Determine status
    status = "running"
    if run_data.get("completed_at"):
        status = "completed"
    if run_data.get("errors"):
        status = "error"
    
    try:
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO core_runs (
                    run_id, conversation_id, user_id, user_input,
                    state, status, current_node, response, error,
                    config, execution_history, step_results,
                    started_at, completed_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
                )
                ON CONFLICT (run_id) DO UPDATE SET
                    state = EXCLUDED.state,
                    status = EXCLUDED.status,
                    current_node = EXCLUDED.current_node,
                    response = EXCLUDED.response,
                    error = EXCLUDED.error,
                    execution_history = EXCLUDED.execution_history,
                    step_results = EXCLUDED.step_results,
                    updated_at = CURRENT_TIMESTAMP,
                    completed_at = EXCLUDED.completed_at
            """,
                run_id,
                run_data.get("conversation_id"),
                run_data.get("user_id"),
                run_data.get("user_input", ""),
                json.dumps(run_data, default=str),
                status,
                run_data.get("current_node"),
                run_data.get("response"),
                json.dumps(run_data.get("errors", [])) if run_data.get("errors") else None,
                json.dumps(run_data.get("config", {})),
                json.dumps(run_data.get("execution_history", [])),
                json.dumps([r for r in run_data.get("step_results", [])], default=str),
                run_data.get("started_at") or datetime.utcnow(),
                run_data.get("completed_at")
            )
        
        logger.debug(f"Saved run {run_id} with status {status}")
        return run_id
        
    except Exception as e:
        logger.error(f"Failed to save run {run_id}: {e}", exc_info=True)
        raise


async def get_run_stats() -> Dict[str, Any]:
    """
    Get run statistics.
    
    Returns:
        Dict with run statistics
    """
    pool = await get_db_pool()
    
    try:
        async with pool.acquire() as conn:
            # Total counts
            total = await conn.fetchval("SELECT COUNT(*) FROM core_runs")
            completed = await conn.fetchval(
                "SELECT COUNT(*) FROM core_runs WHERE status = 'completed'"
            )
            errors = await conn.fetchval(
                "SELECT COUNT(*) FROM core_runs WHERE status = 'error'"
            )
            
            # Runs in last hour
            recent_count = await conn.fetchval(
                """
                SELECT COUNT(*) FROM core_runs 
                WHERE created_at > NOW() - INTERVAL '1 hour'
                """
            )
        
        return {
            "total_runs": total or 0,
            "completed_runs": completed or 0,
            "error_runs": errors or 0,
            "success_rate": (completed / total) if total and total > 0 else 0,
            "runs_last_hour": recent_count or 0
        }
    except Exception as e:
        logger.error(f"Failed to get run stats: {e}", exc_info=True)
        return {
            "total_runs": 0,
            "completed_runs": 0,
            "error_runs": 0,
            "success_rate": 0,
            "runs_last_hour": 0,
            "error": str(e)
        }


async def cleanup_old_runs(days: int = 30) -> int:
    """
    Alias for delete_old_runs for API consistency.
    """
    return await delete_old_runs(days)
