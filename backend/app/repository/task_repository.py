"""
Task Repository

Data access layer for task persistence in the CORE Task Routing Engine.
Handles storing, retrieving, and updating tasks, assignments, and results.
"""

import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from uuid import UUID

from app.dependencies import get_db_pool
from app.models.task_models import (
    Task, TaskAssignment, TaskResult, TaskFilter, TaskMetrics, 
    AgentTaskMetrics, TaskStatus, AgentResponse
)

logger = logging.getLogger(__name__)


# =============================================================================
# TABLE INITIALIZATION
# =============================================================================

async def ensure_task_tables() -> None:
    """Create task tables if they don't exist."""
    pool = await get_db_pool()
    
    async with pool.acquire() as conn:
        # Create tasks table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                task_type VARCHAR(100) NOT NULL,
                payload JSONB NOT NULL DEFAULT '{}',
                priority INTEGER NOT NULL DEFAULT 5 CHECK (priority >= 1 AND priority <= 10),
                required_capabilities JSONB DEFAULT '[]',
                preferred_model VARCHAR(255),
                status VARCHAR(50) NOT NULL DEFAULT 'queued',
                assigned_agent_id UUID REFERENCES agent_instances(id),
                result JSONB,
                human_override BOOLEAN DEFAULT FALSE,
                override_reason TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                assigned_at TIMESTAMP WITH TIME ZONE,
                completed_at TIMESTAMP WITH TIME ZONE,
                duration_ms INTEGER,
                CHECK (status IN ('queued', 'assigned', 'running', 'completed', 'failed', 'refused', 'cancelled'))
            )
        """)
        
        # Create task_assignments table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS task_assignments (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                task_id UUID NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                agent_id UUID NOT NULL REFERENCES agent_instances(id),
                agent_response VARCHAR(50) NOT NULL,
                refusal_reason TEXT,
                suggested_agent VARCHAR(255),
                confidence_score FLOAT NOT NULL DEFAULT 0.5 CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
                assigned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                CHECK (agent_response IN ('accept', 'refuse', 'suggest_alternative'))
            )
        """)
        
        # Create task_results table  
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS task_results (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                task_id UUID NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                agent_id UUID NOT NULL REFERENCES agent_instances(id),
                status VARCHAR(50) NOT NULL,
                result JSONB DEFAULT '{}',
                duration_ms INTEGER NOT NULL,
                model_used VARCHAR(255) NOT NULL,
                tokens_used INTEGER,
                error_message TEXT,
                completed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                CHECK (status IN ('completed', 'failed', 'partial'))
            )
        """)
        
        # Create indexes for performance
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_type ON tasks(task_type)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority DESC)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at DESC)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_assigned_agent ON tasks(assigned_agent_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_task_assignments_task ON task_assignments(task_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_task_assignments_agent ON task_assignments(agent_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_task_results_task ON task_results(task_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_task_results_agent ON task_results(agent_id)")
        
        logger.info("Task tables ensured")


# =============================================================================
# TASK CRUD OPERATIONS
# =============================================================================

async def create_task(task: Task) -> UUID:
    """
    Create a new task.
    
    Args:
        task: Task model instance
        
    Returns:
        UUID of created task
    """
    pool = await get_db_pool()
    
    query = """
        INSERT INTO tasks (
            id, task_type, payload, priority, required_capabilities,
            preferred_model, status, assigned_agent_id, result, 
            human_override, override_reason, created_at, assigned_at, 
            completed_at, duration_ms
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
        )
        RETURNING id
    """
    
    async with pool.acquire() as conn:
        result = await conn.fetchval(
            query,
            task.id,
            task.task_type,
            json.dumps(task.payload),
            task.priority,
            json.dumps(task.required_capabilities),
            task.preferred_model,
            task.status,
            task.assigned_agent_id,
            json.dumps(task.result) if task.result else None,
            task.human_override,
            task.override_reason,
            task.created_at,
            task.assigned_at,
            task.completed_at,
            task.duration_ms
        )
        
        logger.info(f"Created task: {result} (type: {task.task_type})")
        return result


async def get_task(task_id: UUID) -> Optional[Task]:
    """Get a task by ID."""
    pool = await get_db_pool()
    
    query = "SELECT * FROM tasks WHERE id = $1"
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, task_id)
        
        if not row:
            return None
            
        return _row_to_task(row)


async def update_task_status(
    task_id: UUID, 
    status: str, 
    assigned_agent_id: Optional[UUID] = None,
    result: Optional[Dict[str, Any]] = None,
    duration_ms: Optional[int] = None
) -> bool:
    """
    Update task status and related fields.
    
    Args:
        task_id: Task to update
        status: New status
        assigned_agent_id: Agent assigned to task (optional)
        result: Task result data (optional)
        duration_ms: Task duration (optional)
        
    Returns:
        True if update succeeded
    """
    pool = await get_db_pool()
    
    # Build dynamic update query
    set_clauses = ["status = $2"]
    params = [task_id, status]
    param_count = 2
    
    if status == TaskStatus.ASSIGNED and assigned_agent_id:
        param_count += 1
        set_clauses.append(f"assigned_agent_id = ${param_count}")
        set_clauses.append(f"assigned_at = NOW()")
        params.append(assigned_agent_id)
    
    if status in [TaskStatus.COMPLETED, TaskStatus.FAILED] and result is not None:
        param_count += 1
        set_clauses.append(f"result = ${param_count}")
        set_clauses.append(f"completed_at = NOW()")
        params.append(json.dumps(result))
        
        if duration_ms is not None:
            param_count += 1
            set_clauses.append(f"duration_ms = ${param_count}")
            params.append(duration_ms)
    
    query = f"""
        UPDATE tasks 
        SET {', '.join(set_clauses)}
        WHERE id = $1
    """
    
    async with pool.acquire() as conn:
        result = await conn.execute(query, *params)
        return result == "UPDATE 1"


async def get_queued_tasks(priority_order: bool = True, limit: int = 50) -> List[Task]:
    """
    Get queued tasks, optionally ordered by priority.
    
    Args:
        priority_order: Whether to order by priority (highest first)
        limit: Maximum number of tasks to return
        
    Returns:
        List of queued tasks
    """
    pool = await get_db_pool()
    
    order_clause = "ORDER BY priority DESC, created_at ASC" if priority_order else "ORDER BY created_at ASC"
    
    query = f"""
        SELECT * FROM tasks 
        WHERE status = 'queued'
        {order_clause}
        LIMIT $1
    """
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, limit)
        
        return [_row_to_task(row) for row in rows]


async def get_tasks_by_agent(agent_id: UUID, limit: int = 50) -> List[Task]:
    """Get tasks assigned to a specific agent."""
    pool = await get_db_pool()
    
    query = """
        SELECT * FROM tasks 
        WHERE assigned_agent_id = $1
        ORDER BY created_at DESC
        LIMIT $2
    """
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, agent_id, limit)
        
        return [_row_to_task(row) for row in rows]


async def list_tasks(
    filter_params: Optional[TaskFilter] = None,
    limit: int = 50,
    offset: int = 0
) -> Tuple[List[Task], int]:
    """
    List tasks with optional filtering and pagination.
    
    Args:
        filter_params: Optional filter criteria
        limit: Maximum number of results
        offset: Number of results to skip
        
    Returns:
        Tuple of (tasks, total_count)
    """
    pool = await get_db_pool()
    
    # Build WHERE clause
    conditions = []
    params = []
    param_count = 0
    
    if filter_params:
        if filter_params.status:
            param_count += 1
            conditions.append(f"status = ${param_count}")
            params.append(filter_params.status)
            
        if filter_params.task_type:
            param_count += 1
            conditions.append(f"task_type = ${param_count}")
            params.append(filter_params.task_type)
            
        if filter_params.assigned_agent_id:
            param_count += 1
            conditions.append(f"assigned_agent_id = ${param_count}")
            params.append(filter_params.assigned_agent_id)
            
        if filter_params.priority_min:
            param_count += 1
            conditions.append(f"priority >= ${param_count}")
            params.append(filter_params.priority_min)
            
        if filter_params.priority_max:
            param_count += 1
            conditions.append(f"priority <= ${param_count}")
            params.append(filter_params.priority_max)
            
        if filter_params.created_after:
            param_count += 1
            conditions.append(f"created_at >= ${param_count}")
            params.append(filter_params.created_after)
            
        if filter_params.created_before:
            param_count += 1
            conditions.append(f"created_at <= ${param_count}")
            params.append(filter_params.created_before)
            
        if filter_params.human_override is not None:
            param_count += 1
            conditions.append(f"human_override = ${param_count}")
            params.append(filter_params.human_override)
    
    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)
    
    # Get total count
    count_query = f"SELECT COUNT(*) FROM tasks {where_clause}"
    
    # Get tasks with pagination
    param_count += 1
    params.append(limit)
    param_count += 1
    params.append(offset)
    
    tasks_query = f"""
        SELECT * FROM tasks 
        {where_clause}
        ORDER BY created_at DESC
        LIMIT ${param_count - 1} OFFSET ${param_count}
    """
    
    async with pool.acquire() as conn:
        # Get total count
        total_count = await conn.fetchval(count_query, *params[:-2])
        
        # Get tasks
        rows = await conn.fetch(tasks_query, *params)
        tasks = [_row_to_task(row) for row in rows]
        
        return tasks, total_count


async def get_task_metrics(agent_id: Optional[UUID] = None) -> TaskMetrics:
    """
    Get task routing analytics.
    
    Args:
        agent_id: Optional agent ID to filter metrics
        
    Returns:
        TaskMetrics with aggregated data
    """
    pool = await get_db_pool()
    
    # Build WHERE clause for agent filtering
    agent_filter = ""
    params = []
    if agent_id:
        agent_filter = "WHERE assigned_agent_id = $1"
        params.append(agent_id)
    
    query = f"""
        SELECT 
            COUNT(*) as total_tasks,
            COUNT(*) FILTER (WHERE status = 'completed') as completed_tasks,
            COUNT(*) FILTER (WHERE status = 'failed') as failed_tasks,
            COUNT(*) FILTER (WHERE status = 'refused') as refused_tasks,
            COUNT(*) FILTER (WHERE status = 'queued') as queue_depth,
            AVG(duration_ms) FILTER (WHERE status = 'completed' AND duration_ms IS NOT NULL) as avg_completion_time_ms,
            AVG(EXTRACT(EPOCH FROM (assigned_at - created_at)) * 1000) FILTER (
                WHERE assigned_at IS NOT NULL
            ) as avg_queue_wait_time_ms
        FROM tasks
        {agent_filter}
    """
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, *params)
        
        total = row['total_tasks'] or 0
        completed = row['completed_tasks'] or 0
        failed = row['failed_tasks'] or 0
        refused = row['refused_tasks'] or 0
        
        success_rate = (completed / total) if total > 0 else 0.0
        refusal_rate = (refused / total) if total > 0 else 0.0
        
        return TaskMetrics(
            total_tasks=total,
            completed_tasks=completed,
            failed_tasks=failed,
            refused_tasks=refused,
            avg_completion_time_ms=row['avg_completion_time_ms'],
            success_rate=success_rate,
            refusal_rate=refusal_rate,
            queue_depth=row['queue_depth'] or 0,
            avg_queue_wait_time_ms=row['avg_queue_wait_time_ms']
        )


async def get_agent_task_metrics() -> List[AgentTaskMetrics]:
    """Get task metrics for all agents."""
    pool = await get_db_pool()
    
    query = """
        SELECT 
            ai.id as agent_id,
            ai.agent_role,
            COUNT(t.id) as total_assigned,
            COUNT(*) FILTER (WHERE t.status = 'completed') as completed,
            COUNT(*) FILTER (WHERE t.status = 'failed') as failed,
            COUNT(*) FILTER (WHERE t.status = 'refused') as refused,
            COUNT(*) FILTER (WHERE t.status IN ('assigned', 'running')) as current_load,
            AVG(t.duration_ms) FILTER (WHERE t.status = 'completed' AND t.duration_ms IS NOT NULL) as avg_duration_ms
        FROM agent_instances ai
        LEFT JOIN tasks t ON ai.id = t.assigned_agent_id
        WHERE ai.status = 'ready'
        GROUP BY ai.id, ai.agent_role
        ORDER BY ai.agent_role, ai.created_at
    """
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query)
        
        metrics = []
        for row in rows:
            total = row['total_assigned'] or 0
            completed = row['completed'] or 0
            failed = row['failed'] or 0
            refused = row['refused'] or 0
            
            success_rate = (completed / total) if total > 0 else 0.0
            refusal_rate = (refused / total) if total > 0 else 0.0
            
            metrics.append(AgentTaskMetrics(
                agent_id=row['agent_id'],
                agent_role=row['agent_role'],
                total_assigned=total,
                completed=completed,
                failed=failed,
                refused=refused,
                avg_duration_ms=row['avg_duration_ms'],
                success_rate=success_rate,
                refusal_rate=refusal_rate,
                current_load=row['current_load'] or 0
            ))
        
        return metrics


# =============================================================================
# TASK ASSIGNMENT OPERATIONS
# =============================================================================

async def create_task_assignment(assignment: TaskAssignment) -> UUID:
    """Create a task assignment record."""
    pool = await get_db_pool()
    
    query = """
        INSERT INTO task_assignments (
            task_id, agent_id, agent_response, refusal_reason,
            suggested_agent, confidence_score, assigned_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7
        )
        RETURNING id
    """
    
    async with pool.acquire() as conn:
        result = await conn.fetchval(
            query,
            assignment.task_id,
            assignment.agent_id,
            assignment.agent_response,
            assignment.refusal_reason,
            assignment.suggested_agent,
            assignment.confidence_score,
            assignment.assigned_at
        )
        
        logger.debug(f"Created task assignment: {result}")
        return result


async def get_task_assignments(task_id: UUID) -> List[TaskAssignment]:
    """Get all assignments for a task."""
    pool = await get_db_pool()
    
    query = """
        SELECT * FROM task_assignments 
        WHERE task_id = $1
        ORDER BY assigned_at DESC
    """
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, task_id)
        
        assignments = []
        for row in rows:
            assignments.append(TaskAssignment(
                task_id=row['task_id'],
                agent_id=row['agent_id'],
                agent_response=row['agent_response'],
                refusal_reason=row['refusal_reason'],
                suggested_agent=row['suggested_agent'],
                confidence_score=row['confidence_score'],
                assigned_at=row['assigned_at']
            ))
        
        return assignments


# =============================================================================
# TASK RESULT OPERATIONS
# =============================================================================

async def create_task_result(result: TaskResult) -> UUID:
    """Create a task result record."""
    pool = await get_db_pool()
    
    query = """
        INSERT INTO task_results (
            task_id, agent_id, status, result, duration_ms,
            model_used, tokens_used, error_message, completed_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9
        )
        RETURNING id
    """
    
    async with pool.acquire() as conn:
        result_id = await conn.fetchval(
            query,
            result.task_id,
            result.agent_id,
            result.status,
            json.dumps(result.result),
            result.duration_ms,
            result.model_used,
            result.tokens_used,
            result.error_message,
            result.completed_at
        )
        
        logger.info(f"Created task result: {result_id} (status: {result.status})")
        return result_id


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _row_to_task(row) -> Task:
    """Convert database row to Task model."""
    return Task(
        id=row['id'],
        task_type=row['task_type'],
        payload=json.loads(row['payload']) if row['payload'] else {},
        priority=row['priority'],
        required_capabilities=json.loads(row['required_capabilities']) if row['required_capabilities'] else [],
        preferred_model=row['preferred_model'],
        status=row['status'],
        assigned_agent_id=row['assigned_agent_id'],
        result=json.loads(row['result']) if row['result'] else None,
        human_override=row['human_override'],
        override_reason=row['override_reason'],
        created_at=row['created_at'],
        assigned_at=row['assigned_at'],
        completed_at=row['completed_at'],
        duration_ms=row['duration_ms']
    )