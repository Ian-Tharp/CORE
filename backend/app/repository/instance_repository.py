"""
Instance Repository

Data access layer for agent instance persistence.
Handles storing, retrieving, and updating agent instances and trust metrics.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field

from app.dependencies import get_db_pool

logger = logging.getLogger(__name__)


class InstanceStatus(str, Enum):
    """Status of an agent instance."""
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    STOPPING = "stopping"
    STOPPED = "stopped"
    RESTARTING = "restarting"
    FAILED = "failed"


class AgentInstance(BaseModel):
    """Agent instance model."""
    id: UUID
    container_id: str
    agent_id: str
    agent_role: str
    status: InstanceStatus = InstanceStatus.STARTING
    device_id: Optional[str] = None
    resource_profile: Dict[str, Any] = Field(default_factory=dict)
    capabilities: List[str] = Field(default_factory=list)
    last_heartbeat: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    stopped_at: Optional[datetime] = None
    
    class Config:
        use_enum_values = True


class InstanceTrustMetrics(BaseModel):
    """Trust metrics for an agent instance."""
    id: UUID
    agent_instance_id: UUID
    tasks_completed: int = 0
    tasks_refused: int = 0
    tasks_failed: int = 0
    overrides_received: int = 0
    override_success_rate: float = 0.0
    avg_task_duration_ms: Optional[float] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# TABLE INITIALIZATION
# =============================================================================

async def ensure_instance_tables() -> None:
    """Create instance tables if they don't exist."""
    pool = await get_db_pool()
    
    async with pool.acquire() as conn:
        # Create agent_instances table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_instances (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                container_id VARCHAR(64) UNIQUE,
                agent_id VARCHAR(255),
                agent_role VARCHAR(100) NOT NULL,
                status VARCHAR(50) NOT NULL DEFAULT 'starting',
                device_id VARCHAR(100),
                resource_profile JSONB DEFAULT '{}',
                capabilities JSONB DEFAULT '[]',
                last_heartbeat TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                stopped_at TIMESTAMP WITH TIME ZONE
            )
        """)
        
        # Create instance_trust_metrics table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS instance_trust_metrics (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                agent_instance_id UUID REFERENCES agent_instances(id) ON DELETE CASCADE,
                tasks_completed INTEGER DEFAULT 0,
                tasks_refused INTEGER DEFAULT 0,
                tasks_failed INTEGER DEFAULT 0,
                overrides_received INTEGER DEFAULT 0,
                override_success_rate FLOAT DEFAULT 0.0,
                avg_task_duration_ms FLOAT,
                last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create indexes
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_instances_container_id ON agent_instances(container_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_instances_role ON agent_instances(agent_role)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_instances_status ON agent_instances(status)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_instances_created ON agent_instances(created_at DESC)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_instance_trust_metrics_instance ON instance_trust_metrics(agent_instance_id)"
        )
        
        logger.info("Instance tables ensured")


# =============================================================================
# INSTANCE CRUD
# =============================================================================

async def create_instance(instance: AgentInstance) -> UUID:
    """
    Create a new agent instance record.
    
    Args:
        instance: AgentInstance model with instance data
        
    Returns:
        UUID of the created instance
    """
    pool = await get_db_pool()
    
    query = """
        INSERT INTO agent_instances (
            id, container_id, agent_id, agent_role, status,
            device_id, resource_profile, capabilities, 
            last_heartbeat, created_at, stopped_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
        )
        RETURNING id
    """
    
    async with pool.acquire() as conn:
        result = await conn.fetchval(
            query,
            instance.id,
            instance.container_id,
            instance.agent_id,
            instance.agent_role,
            instance.status.value,
            instance.device_id,
            json.dumps(instance.resource_profile),
            json.dumps(instance.capabilities),
            instance.last_heartbeat,
            instance.created_at,
            instance.stopped_at
        )
        
        # Initialize trust metrics
        await _create_trust_metrics(instance.id)
        
        logger.info(f"Created agent instance: {result}")
        return result


async def _create_trust_metrics(instance_id: UUID) -> UUID:
    """Create initial trust metrics for an instance."""
    pool = await get_db_pool()
    
    query = """
        INSERT INTO instance_trust_metrics (agent_instance_id)
        VALUES ($1)
        RETURNING id
    """
    
    async with pool.acquire() as conn:
        result = await conn.fetchval(query, instance_id)
        logger.debug(f"Created trust metrics for instance: {instance_id}")
        return result


async def get_instance(instance_id: UUID) -> Optional[AgentInstance]:
    """Get an instance by ID."""
    pool = await get_db_pool()
    
    query = "SELECT * FROM agent_instances WHERE id = $1"
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, instance_id)
        
        if not row:
            return None
            
        return AgentInstance(
            id=row['id'],
            container_id=row['container_id'],
            agent_id=row['agent_id'],
            agent_role=row['agent_role'],
            status=InstanceStatus(row['status']),
            device_id=row['device_id'],
            resource_profile=json.loads(row['resource_profile']) if row['resource_profile'] else {},
            capabilities=json.loads(row['capabilities']) if row['capabilities'] else [],
            last_heartbeat=row['last_heartbeat'],
            created_at=row['created_at'],
            stopped_at=row['stopped_at']
        )


async def get_instance_by_container_id(container_id: str) -> Optional[AgentInstance]:
    """Get an instance by container ID."""
    pool = await get_db_pool()
    
    query = "SELECT * FROM agent_instances WHERE container_id = $1"
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, container_id)
        
        if not row:
            return None
            
        return AgentInstance(
            id=row['id'],
            container_id=row['container_id'],
            agent_id=row['agent_id'],
            agent_role=row['agent_role'],
            status=InstanceStatus(row['status']),
            device_id=row['device_id'],
            resource_profile=json.loads(row['resource_profile']) if row['resource_profile'] else {},
            capabilities=json.loads(row['capabilities']) if row['capabilities'] else [],
            last_heartbeat=row['last_heartbeat'],
            created_at=row['created_at'],
            stopped_at=row['stopped_at']
        )


async def list_instances(
    agent_role: Optional[str] = None,
    status: Optional[InstanceStatus] = None,
    device_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> List[AgentInstance]:
    """List instances with optional filters."""
    pool = await get_db_pool()
    
    conditions = []
    params = []
    param_count = 0
    
    if agent_role:
        param_count += 1
        conditions.append(f"agent_role = ${param_count}")
        params.append(agent_role)
    
    if status:
        param_count += 1
        conditions.append(f"status = ${param_count}")
        params.append(status.value)
        
    if device_id:
        param_count += 1
        conditions.append(f"device_id = ${param_count}")
        params.append(device_id)
    
    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)
    
    # Add limit and offset
    param_count += 1
    params.append(limit)
    param_count += 1
    params.append(offset)
    
    query = f"""
        SELECT * FROM agent_instances
        {where_clause}
        ORDER BY created_at DESC
        LIMIT ${param_count - 1} OFFSET ${param_count}
    """
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
        
        instances = []
        for row in rows:
            instances.append(AgentInstance(
                id=row['id'],
                container_id=row['container_id'],
                agent_id=row['agent_id'],
                agent_role=row['agent_role'],
                status=InstanceStatus(row['status']),
                device_id=row['device_id'],
                resource_profile=json.loads(row['resource_profile']) if row['resource_profile'] else {},
                capabilities=json.loads(row['capabilities']) if row['capabilities'] else [],
                last_heartbeat=row['last_heartbeat'],
                created_at=row['created_at'],
                stopped_at=row['stopped_at']
            ))
        
        return instances


async def list_instances_by_status(status: InstanceStatus) -> List[AgentInstance]:
    """List instances by status."""
    return await list_instances(status=status, limit=1000)


async def update_instance_status(
    container_id: str,
    status: InstanceStatus,
    stopped_at: Optional[datetime] = None
) -> bool:
    """Update instance status."""
    pool = await get_db_pool()
    
    if status == InstanceStatus.STOPPED and not stopped_at:
        stopped_at = datetime.utcnow()
    
    if stopped_at:
        query = """
            UPDATE agent_instances 
            SET status = $2, stopped_at = $3
            WHERE container_id = $1
        """
        params = (container_id, status.value, stopped_at)
    else:
        query = """
            UPDATE agent_instances 
            SET status = $2
            WHERE container_id = $1
        """
        params = (container_id, status.value)
    
    async with pool.acquire() as conn:
        result = await conn.execute(query, *params)
        return result == "UPDATE 1"


async def update_heartbeat(container_id: str, heartbeat_time: Optional[datetime] = None) -> bool:
    """Update last heartbeat time for an instance."""
    if heartbeat_time is None:
        heartbeat_time = datetime.utcnow()
    
    pool = await get_db_pool()
    
    query = """
        UPDATE agent_instances 
        SET last_heartbeat = $2
        WHERE container_id = $1
    """
    
    async with pool.acquire() as conn:
        result = await conn.execute(query, container_id, heartbeat_time)
        return result == "UPDATE 1"


async def update_instance(
    instance_id: UUID,
    updates: Dict[str, Any]
) -> bool:
    """Update instance fields dynamically."""
    if not updates:
        return True
    
    pool = await get_db_pool()
    
    # Build dynamic update query
    set_clauses = []
    params = [instance_id]
    param_count = 1
    
    for field, value in updates.items():
        if field in ['resource_profile', 'capabilities'] and isinstance(value, (dict, list)):
            value = json.dumps(value)
        
        param_count += 1
        set_clauses.append(f"{field} = ${param_count}")
        params.append(value)
    
    query = f"""
        UPDATE agent_instances 
        SET {', '.join(set_clauses)}
        WHERE id = $1
    """
    
    async with pool.acquire() as conn:
        result = await conn.execute(query, *params)
        return result == "UPDATE 1"


async def delete_instance(instance_id: UUID) -> bool:
    """Delete an instance and its trust metrics (cascades)."""
    pool = await get_db_pool()
    
    query = "DELETE FROM agent_instances WHERE id = $1"
    
    async with pool.acquire() as conn:
        result = await conn.execute(query, instance_id)
        return result == "DELETE 1"


async def delete_instance_by_container_id(container_id: str) -> bool:
    """Delete an instance by container ID."""
    pool = await get_db_pool()
    
    query = "DELETE FROM agent_instances WHERE container_id = $1"
    
    async with pool.acquire() as conn:
        result = await conn.execute(query, container_id)
        return result == "DELETE 1"


# =============================================================================
# TRUST METRICS CRUD
# =============================================================================

async def get_trust_metrics(instance_id: UUID) -> Optional[InstanceTrustMetrics]:
    """Get trust metrics for an instance."""
    pool = await get_db_pool()
    
    query = "SELECT * FROM instance_trust_metrics WHERE agent_instance_id = $1"
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, instance_id)
        
        if not row:
            return None
            
        return InstanceTrustMetrics(
            id=row['id'],
            agent_instance_id=row['agent_instance_id'],
            tasks_completed=row['tasks_completed'],
            tasks_refused=row['tasks_refused'],
            tasks_failed=row['tasks_failed'],
            overrides_received=row['overrides_received'],
            override_success_rate=row['override_success_rate'],
            avg_task_duration_ms=row['avg_task_duration_ms'],
            last_updated=row['last_updated']
        )


async def update_trust_metrics(
    instance_id: UUID,
    updates: Dict[str, Any]
) -> bool:
    """Update trust metrics for an instance."""
    if not updates:
        return True
    
    pool = await get_db_pool()
    
    # Always update last_updated
    updates['last_updated'] = datetime.utcnow()
    
    # Build dynamic update query
    set_clauses = []
    params = [instance_id]
    param_count = 1
    
    for field, value in updates.items():
        param_count += 1
        set_clauses.append(f"{field} = ${param_count}")
        params.append(value)
    
    query = f"""
        UPDATE instance_trust_metrics 
        SET {', '.join(set_clauses)}
        WHERE agent_instance_id = $1
    """
    
    async with pool.acquire() as conn:
        result = await conn.execute(query, *params)
        return result == "UPDATE 1"


async def increment_task_completed(instance_id: UUID, task_duration_ms: Optional[float] = None) -> bool:
    """Increment tasks completed counter and update average duration."""
    pool = await get_db_pool()
    
    if task_duration_ms:
        query = """
            UPDATE instance_trust_metrics 
            SET tasks_completed = tasks_completed + 1,
                avg_task_duration_ms = CASE 
                    WHEN avg_task_duration_ms IS NULL THEN $2
                    ELSE (avg_task_duration_ms * tasks_completed + $2) / (tasks_completed + 1)
                END,
                last_updated = NOW()
            WHERE agent_instance_id = $1
        """
        params = (instance_id, task_duration_ms)
    else:
        query = """
            UPDATE instance_trust_metrics 
            SET tasks_completed = tasks_completed + 1,
                last_updated = NOW()
            WHERE agent_instance_id = $1
        """
        params = (instance_id,)
    
    async with pool.acquire() as conn:
        result = await conn.execute(query, *params)
        return result == "UPDATE 1"


async def increment_task_failed(instance_id: UUID) -> bool:
    """Increment tasks failed counter."""
    pool = await get_db_pool()
    
    query = """
        UPDATE instance_trust_metrics 
        SET tasks_failed = tasks_failed + 1,
            last_updated = NOW()
        WHERE agent_instance_id = $1
    """
    
    async with pool.acquire() as conn:
        result = await conn.execute(query, instance_id)
        return result == "UPDATE 1"


async def increment_task_refused(instance_id: UUID) -> bool:
    """Increment tasks refused counter."""
    pool = await get_db_pool()
    
    query = """
        UPDATE instance_trust_metrics 
        SET tasks_refused = tasks_refused + 1,
            last_updated = NOW()
        WHERE agent_instance_id = $1
    """
    
    async with pool.acquire() as conn:
        result = await conn.execute(query, instance_id)
        return result == "UPDATE 1"


async def record_override(instance_id: UUID, success: bool) -> bool:
    """Record an override and update success rate."""
    pool = await get_db_pool()
    
    query = """
        UPDATE instance_trust_metrics 
        SET overrides_received = overrides_received + 1,
            override_success_rate = CASE
                WHEN overrides_received = 0 THEN CASE WHEN $2 THEN 1.0 ELSE 0.0 END
                ELSE (override_success_rate * overrides_received + CASE WHEN $2 THEN 1.0 ELSE 0.0 END) / (overrides_received + 1)
            END,
            last_updated = NOW()
        WHERE agent_instance_id = $1
    """
    
    async with pool.acquire() as conn:
        result = await conn.execute(query, instance_id, success)
        return result == "UPDATE 1"


async def get_instances_with_trust_metrics(
    agent_role: Optional[str] = None,
    min_trust_score: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Get instances with their trust metrics for analysis."""
    pool = await get_db_pool()
    
    conditions = []
    params = []
    param_count = 0
    
    if agent_role:
        param_count += 1
        conditions.append(f"i.agent_role = ${param_count}")
        params.append(agent_role)
    
    if min_trust_score:
        param_count += 1
        conditions.append(f"""
            (t.tasks_completed + 1.0) / (t.tasks_completed + t.tasks_failed + 1.0) >= ${param_count}
        """)
        params.append(min_trust_score)
    
    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)
    
    query = f"""
        SELECT 
            i.*,
            t.tasks_completed,
            t.tasks_refused,
            t.tasks_failed,
            t.overrides_received,
            t.override_success_rate,
            t.avg_task_duration_ms,
            (t.tasks_completed + 1.0) / (t.tasks_completed + t.tasks_failed + 1.0) as trust_score
        FROM agent_instances i
        LEFT JOIN instance_trust_metrics t ON i.id = t.agent_instance_id
        {where_clause}
        ORDER BY trust_score DESC, i.created_at DESC
    """
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
        
        results = []
        for row in rows:
            results.append({
                "instance": AgentInstance(
                    id=row['id'],
                    container_id=row['container_id'],
                    agent_id=row['agent_id'],
                    agent_role=row['agent_role'],
                    status=InstanceStatus(row['status']),
                    device_id=row['device_id'],
                    resource_profile=json.loads(row['resource_profile']) if row['resource_profile'] else {},
                    capabilities=json.loads(row['capabilities']) if row['capabilities'] else [],
                    last_heartbeat=row['last_heartbeat'],
                    created_at=row['created_at'],
                    stopped_at=row['stopped_at']
                ),
                "trust_metrics": {
                    "tasks_completed": row['tasks_completed'] or 0,
                    "tasks_refused": row['tasks_refused'] or 0,
                    "tasks_failed": row['tasks_failed'] or 0,
                    "overrides_received": row['overrides_received'] or 0,
                    "override_success_rate": row['override_success_rate'] or 0.0,
                    "avg_task_duration_ms": row['avg_task_duration_ms'],
                    "trust_score": float(row['trust_score']) if row['trust_score'] else 0.5
                }
            })
        
        return results