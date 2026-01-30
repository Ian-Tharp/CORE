"""
Memory Repository

Data access layer for LangMem three-tier memory persistence.
Handles storing, retrieving, and searching semantic, episodic, and procedural memories.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from enum import Enum

from pydantic import BaseModel, Field

from app.dependencies import get_db_pool

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Types of memory in the three-tier system."""
    SEMANTIC = "semantic"      # Shared knowledge
    EPISODIC = "episodic"      # Personal experiences
    PROCEDURAL = "procedural"  # Role-based procedures


class MemoryItem(BaseModel):
    """Base memory item model."""
    id: UUID
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    access_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: Optional[datetime] = None


class SemanticMemory(MemoryItem):
    """Semantic memory - shared knowledge accessible to all agents."""
    source_agent_id: Optional[str] = None


class EpisodicMemory(MemoryItem):
    """Episodic memory - personal experiences for specific agents."""
    agent_id: str
    memory_type: str = "experience"  # experience, conversation, task_result
    importance: float = 0.5
    consolidated: bool = False
    expires_at: Optional[datetime] = None


class ProceduralMemory(MemoryItem):
    """Procedural memory - role-based learned procedures."""
    role: str
    procedure_name: str
    steps: List[str]
    success_rate: float = 0.0
    usage_count: int = 0
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class MemoryStats(BaseModel):
    """Memory statistics for an agent."""
    agent_id: str
    semantic_count: int = 0
    episodic_count: int = 0
    procedural_count: int = 0
    total_access_count: int = 0
    last_memory_created: Optional[datetime] = None


# =============================================================================
# TABLE INITIALIZATION
# =============================================================================

async def ensure_memory_tables() -> None:
    """Create memory tables if they don't exist."""
    pool = await get_db_pool()
    
    async with pool.acquire() as conn:
        # Enable pgvector extension if not already enabled
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Semantic memories table - shared knowledge
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS semantic_memories (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                content TEXT NOT NULL,
                embedding vector(768),  -- Default dimension, will be updated based on model
                metadata JSONB DEFAULT '{}',
                source_agent_id VARCHAR(255),
                confidence FLOAT DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                last_accessed TIMESTAMP WITH TIME ZONE
            )
        """)
        
        # Episodic memories table - personal experiences
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS episodic_memories (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                agent_id VARCHAR(255) NOT NULL,
                content TEXT NOT NULL,
                embedding vector(768),
                memory_type VARCHAR(50) DEFAULT 'experience',
                metadata JSONB DEFAULT '{}',
                importance FLOAT DEFAULT 0.5,
                confidence FLOAT DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                consolidated BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                last_accessed TIMESTAMP WITH TIME ZONE,
                expires_at TIMESTAMP WITH TIME ZONE
            )
        """)
        
        # Procedural memories table - role-based procedures
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS procedural_memories (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                role VARCHAR(100) NOT NULL,
                procedure_name VARCHAR(255) NOT NULL,
                content TEXT NOT NULL,
                steps JSONB NOT NULL,
                embedding vector(768),
                metadata JSONB DEFAULT '{}',
                success_rate FLOAT DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                confidence FLOAT DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                last_accessed TIMESTAMP WITH TIME ZONE,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create indexes for efficient querying
        # Vector similarity indexes (cosine distance)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_embedding ON semantic_memories USING hnsw (embedding vector_cosine_ops)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodic_embedding ON episodic_memories USING hnsw (embedding vector_cosine_ops)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_procedural_embedding ON procedural_memories USING hnsw (embedding vector_cosine_ops)"
        )
        
        # Regular indexes for filtering
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_created ON semantic_memories(created_at DESC)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodic_agent_created ON episodic_memories(agent_id, created_at DESC)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_procedural_role ON procedural_memories(role)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodic_consolidated ON episodic_memories(consolidated)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodic_expires ON episodic_memories(expires_at) WHERE expires_at IS NOT NULL"
        )
        
        # GIN indexes for JSONB metadata
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_metadata ON semantic_memories USING GIN(metadata)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodic_metadata ON episodic_memories USING GIN(metadata)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_procedural_metadata ON procedural_memories USING GIN(metadata)"
        )
        
        logger.info("Memory tables ensured")


# =============================================================================
# SEMANTIC MEMORY CRUD
# =============================================================================

async def create_semantic_memory(memory: SemanticMemory) -> UUID:
    """Create a new semantic memory."""
    pool = await get_db_pool()
    
    query = """
        INSERT INTO semantic_memories (
            id, content, embedding, metadata, source_agent_id, 
            confidence, created_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7
        )
        RETURNING id
    """
    
    async with pool.acquire() as conn:
        result = await conn.fetchval(
            query,
            memory.id,
            memory.content,
            memory.embedding,
            json.dumps(memory.metadata),
            memory.source_agent_id,
            memory.confidence,
            memory.created_at
        )
        
        logger.info(f"Created semantic memory: {result}")
        return result


async def search_semantic_memories(
    query_embedding: List[float],
    limit: int = 10,
    threshold: float = 0.7
) -> List[SemanticMemory]:
    """Search semantic memories by vector similarity."""
    pool = await get_db_pool()
    
    query = """
        SELECT *, 1 - (embedding <=> $1) as similarity
        FROM semantic_memories
        WHERE 1 - (embedding <=> $1) > $2
        ORDER BY embedding <=> $1
        LIMIT $3
    """
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, query_embedding, threshold, limit)
        
        memories = []
        for row in rows:
            memory = SemanticMemory(
                id=row['id'],
                content=row['content'],
                embedding=list(row['embedding']),
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                source_agent_id=row['source_agent_id'],
                confidence=row['confidence'],
                access_count=row['access_count'],
                created_at=row['created_at'],
                last_accessed=row['last_accessed']
            )
            memory.metadata['similarity'] = float(row['similarity'])
            memories.append(memory)
        
        # Update access counts
        if memories:
            await _update_access_counts('semantic_memories', [m.id for m in memories])
        
        return memories


async def get_semantic_memory(memory_id: UUID) -> Optional[SemanticMemory]:
    """Get semantic memory by ID."""
    pool = await get_db_pool()
    
    query = "SELECT * FROM semantic_memories WHERE id = $1"
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, memory_id)
        
        if not row:
            return None
        
        # Update access count
        await _update_access_counts('semantic_memories', [memory_id])
        
        return SemanticMemory(
            id=row['id'],
            content=row['content'],
            embedding=list(row['embedding']),
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            source_agent_id=row['source_agent_id'],
            confidence=row['confidence'],
            access_count=row['access_count'] + 1,
            created_at=row['created_at'],
            last_accessed=datetime.utcnow()
        )


# =============================================================================
# EPISODIC MEMORY CRUD
# =============================================================================

async def create_episodic_memory(memory: EpisodicMemory) -> UUID:
    """Create a new episodic memory."""
    pool = await get_db_pool()
    
    query = """
        INSERT INTO episodic_memories (
            id, agent_id, content, embedding, memory_type, metadata,
            importance, confidence, consolidated, created_at, expires_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
        )
        RETURNING id
    """
    
    async with pool.acquire() as conn:
        result = await conn.fetchval(
            query,
            memory.id,
            memory.agent_id,
            memory.content,
            memory.embedding,
            memory.memory_type,
            json.dumps(memory.metadata),
            memory.importance,
            memory.confidence,
            memory.consolidated,
            memory.created_at,
            memory.expires_at
        )
        
        logger.debug(f"Created episodic memory for agent {memory.agent_id}: {result}")
        return result


async def search_episodic_memories(
    agent_id: str,
    query_embedding: Optional[List[float]] = None,
    limit: int = 10,
    threshold: float = 0.7,
    memory_type: Optional[str] = None
) -> List[EpisodicMemory]:
    """Search episodic memories for a specific agent."""
    pool = await get_db_pool()
    
    conditions = ["agent_id = $1"]
    params = [agent_id]
    param_count = 1
    
    if memory_type:
        param_count += 1
        conditions.append(f"memory_type = ${param_count}")
        params.append(memory_type)
    
    if query_embedding:
        param_count += 1
        conditions.append(f"1 - (embedding <=> ${param_count}) > ${param_count + 1}")
        params.extend([query_embedding, threshold])
        order_by = f"embedding <=> ${param_count}"
        param_count += 1
    else:
        order_by = "created_at DESC"
    
    params.append(limit)
    
    query = f"""
        SELECT *, 
               CASE WHEN ${{len(params) - 1}} > 1 THEN 1 - (embedding <=> ${len(params) - 2}) 
                    ELSE NULL END as similarity
        FROM episodic_memories
        WHERE {' AND '.join(conditions)}
        ORDER BY {order_by}
        LIMIT ${len(params)}
    """
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
        
        memories = []
        for row in rows:
            memory = EpisodicMemory(
                id=row['id'],
                agent_id=row['agent_id'],
                content=row['content'],
                embedding=list(row['embedding']),
                memory_type=row['memory_type'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                importance=row['importance'],
                confidence=row['confidence'],
                access_count=row['access_count'],
                consolidated=row['consolidated'],
                created_at=row['created_at'],
                last_accessed=row['last_accessed'],
                expires_at=row['expires_at']
            )
            if row['similarity']:
                memory.metadata['similarity'] = float(row['similarity'])
            memories.append(memory)
        
        # Update access counts
        if memories:
            await _update_access_counts('episodic_memories', [m.id for m in memories])
        
        return memories


async def consolidate_episodic_memories(
    agent_id: str,
    consolidation_threshold_hours: int = 24
) -> int:
    """Consolidate short-term episodic memories to long-term."""
    pool = await get_db_pool()
    
    cutoff_time = datetime.utcnow() - timedelta(hours=consolidation_threshold_hours)
    
    query = """
        UPDATE episodic_memories 
        SET consolidated = TRUE
        WHERE agent_id = $1 
        AND created_at < $2 
        AND consolidated = FALSE
        AND importance > 0.3
        RETURNING id
    """
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, agent_id, cutoff_time)
        count = len(rows)
        
        if count > 0:
            logger.info(f"Consolidated {count} episodic memories for agent {agent_id}")
        
        return count


async def expire_old_memories(days_to_keep: int = 90) -> int:
    """Delete expired episodic memories."""
    pool = await get_db_pool()
    
    query = """
        DELETE FROM episodic_memories 
        WHERE expires_at IS NOT NULL 
        AND expires_at < NOW()
        OR (created_at < NOW() - INTERVAL '%s days' AND importance < 0.2)
        RETURNING id
    """
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query % days_to_keep)
        count = len(rows)
        
        if count > 0:
            logger.info(f"Expired {count} old episodic memories")
        
        return count


# =============================================================================
# PROCEDURAL MEMORY CRUD
# =============================================================================

async def create_procedural_memory(memory: ProceduralMemory) -> UUID:
    """Create a new procedural memory."""
    pool = await get_db_pool()
    
    query = """
        INSERT INTO procedural_memories (
            id, role, procedure_name, content, steps, embedding,
            metadata, success_rate, confidence, created_at, updated_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
        )
        RETURNING id
    """
    
    async with pool.acquire() as conn:
        result = await conn.fetchval(
            query,
            memory.id,
            memory.role,
            memory.procedure_name,
            memory.content,
            json.dumps(memory.steps),
            memory.embedding,
            json.dumps(memory.metadata),
            memory.success_rate,
            memory.confidence,
            memory.created_at,
            memory.updated_at
        )
        
        logger.debug(f"Created procedural memory for role {memory.role}: {result}")
        return result


async def search_procedural_memories(
    role: Optional[str] = None,
    query_embedding: Optional[List[float]] = None,
    limit: int = 10,
    threshold: float = 0.7
) -> List[ProceduralMemory]:
    """Search procedural memories by role and/or similarity."""
    pool = await get_db_pool()
    
    conditions = []
    params = []
    param_count = 0
    
    if role:
        param_count += 1
        conditions.append(f"role = ${param_count}")
        params.append(role)
    
    if query_embedding:
        param_count += 1
        conditions.append(f"1 - (embedding <=> ${param_count}) > ${param_count + 1}")
        params.extend([query_embedding, threshold])
        order_by = f"embedding <=> ${param_count}"
        param_count += 1
    else:
        order_by = "updated_at DESC"
    
    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)
    
    params.append(limit)
    
    query = f"""
        SELECT *, 
               CASE WHEN ${param_count} > 0 THEN 1 - (embedding <=> ${param_count - 1}) 
                    ELSE NULL END as similarity
        FROM procedural_memories
        {where_clause}
        ORDER BY {order_by}
        LIMIT ${len(params)}
    """
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
        
        memories = []
        for row in rows:
            memory = ProceduralMemory(
                id=row['id'],
                role=row['role'],
                procedure_name=row['procedure_name'],
                content=row['content'],
                steps=json.loads(row['steps']) if row['steps'] else [],
                embedding=list(row['embedding']),
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                success_rate=row['success_rate'],
                usage_count=row['usage_count'],
                confidence=row['confidence'],
                access_count=row['access_count'],
                created_at=row['created_at'],
                last_accessed=row['last_accessed'],
                updated_at=row['updated_at']
            )
            if row['similarity']:
                memory.metadata['similarity'] = float(row['similarity'])
            memories.append(memory)
        
        # Update access counts and usage counts
        if memories:
            await _update_access_counts('procedural_memories', [m.id for m in memories])
            await _update_usage_counts([m.id for m in memories])
        
        return memories


async def update_procedure_success_rate(
    procedure_id: UUID,
    success: bool
) -> bool:
    """Update procedure success rate based on execution result."""
    pool = await get_db_pool()
    
    query = """
        UPDATE procedural_memories 
        SET success_rate = CASE
                WHEN usage_count = 0 THEN CASE WHEN $2 THEN 1.0 ELSE 0.0 END
                ELSE (success_rate * usage_count + CASE WHEN $2 THEN 1.0 ELSE 0.0 END) / (usage_count + 1)
            END,
            updated_at = NOW()
        WHERE id = $1
    """
    
    async with pool.acquire() as conn:
        result = await conn.execute(query, procedure_id, success)
        return result == "UPDATE 1"


async def _update_usage_counts(procedure_ids: List[UUID]) -> None:
    """Update usage counts for procedures."""
    if not procedure_ids:
        return
        
    pool = await get_db_pool()
    
    query = """
        UPDATE procedural_memories 
        SET usage_count = usage_count + 1,
            updated_at = NOW()
        WHERE id = ANY($1)
    """
    
    async with pool.acquire() as conn:
        await conn.execute(query, procedure_ids)


# =============================================================================
# CROSS-TIER OPERATIONS
# =============================================================================

async def get_relevant_context(
    query_embedding: List[float],
    agent_id: Optional[str] = None,
    limit_per_tier: int = 5,
    threshold: float = 0.7
) -> Dict[str, List[MemoryItem]]:
    """Get relevant context from all memory tiers."""
    
    # Search semantic memories (always included)
    semantic_memories = await search_semantic_memories(
        query_embedding, limit=limit_per_tier, threshold=threshold
    )
    
    # Search episodic memories (if agent_id provided)
    episodic_memories = []
    if agent_id:
        episodic_memories = await search_episodic_memories(
            agent_id, query_embedding, limit=limit_per_tier, threshold=threshold
        )
    
    # Search procedural memories (global)
    procedural_memories = await search_procedural_memories(
        query_embedding=query_embedding, limit=limit_per_tier, threshold=threshold
    )
    
    return {
        "semantic": semantic_memories,
        "episodic": episodic_memories,
        "procedural": procedural_memories
    }


async def get_memory_stats(agent_id: str) -> MemoryStats:
    """Get memory statistics for an agent."""
    pool = await get_db_pool()
    
    # Count semantic memories created by this agent
    semantic_count_query = "SELECT COUNT(*) FROM semantic_memories WHERE source_agent_id = $1"
    
    # Count episodic memories for this agent
    episodic_count_query = "SELECT COUNT(*) FROM episodic_memories WHERE agent_id = $1"
    
    # Count procedural memories (not agent-specific, but we'll count all)
    procedural_count_query = "SELECT COUNT(*) FROM procedural_memories"
    
    # Total access counts for this agent's memories
    access_count_query = """
        SELECT 
            COALESCE(SUM(s.access_count), 0) + COALESCE(SUM(e.access_count), 0) as total_access
        FROM 
            (SELECT access_count FROM semantic_memories WHERE source_agent_id = $1) s
        FULL OUTER JOIN 
            (SELECT access_count FROM episodic_memories WHERE agent_id = $1) e ON TRUE
    """
    
    # Last memory created
    last_memory_query = """
        SELECT MAX(created_at) as last_created FROM (
            SELECT created_at FROM semantic_memories WHERE source_agent_id = $1
            UNION ALL
            SELECT created_at FROM episodic_memories WHERE agent_id = $1
        ) memories
    """
    
    async with pool.acquire() as conn:
        semantic_count = await conn.fetchval(semantic_count_query, agent_id)
        episodic_count = await conn.fetchval(episodic_count_query, agent_id)
        procedural_count = await conn.fetchval(procedural_count_query)
        total_access = await conn.fetchval(access_count_query, agent_id) or 0
        last_created = await conn.fetchval(last_memory_query, agent_id)
        
        return MemoryStats(
            agent_id=agent_id,
            semantic_count=semantic_count or 0,
            episodic_count=episodic_count or 0,
            procedural_count=procedural_count or 0,
            total_access_count=total_access,
            last_memory_created=last_created
        )


async def clear_agent_memories(
    agent_id: str,
    tier: Optional[MemoryType] = None
) -> Dict[str, int]:
    """Clear memories for an agent (optionally by tier)."""
    pool = await get_db_pool()
    counts = {}
    
    async with pool.acquire() as conn:
        if not tier or tier == MemoryType.SEMANTIC:
            # Clear semantic memories created by this agent
            result = await conn.execute(
                "DELETE FROM semantic_memories WHERE source_agent_id = $1",
                agent_id
            )
            counts['semantic'] = int(result.split()[-1])
        
        if not tier or tier == MemoryType.EPISODIC:
            # Clear episodic memories for this agent
            result = await conn.execute(
                "DELETE FROM episodic_memories WHERE agent_id = $1",
                agent_id
            )
            counts['episodic'] = int(result.split()[-1])
        
        if not tier or tier == MemoryType.PROCEDURAL:
            # Note: Procedural memories are role-based, not agent-specific
            # We could clear by role if we had that information
            counts['procedural'] = 0
    
    return counts


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def _update_access_counts(table_name: str, memory_ids: List[UUID]) -> None:
    """Update access counts and last_accessed for memories."""
    if not memory_ids:
        return
        
    pool = await get_db_pool()
    
    query = f"""
        UPDATE {table_name} 
        SET access_count = access_count + 1,
            last_accessed = NOW()
        WHERE id = ANY($1)
    """
    
    async with pool.acquire() as conn:
        await conn.execute(query, memory_ids)


async def update_embedding_dimensions(new_dimensions: int) -> None:
    """Update embedding vector dimensions in all tables."""
    if new_dimensions <= 0:
        raise ValueError("Dimensions must be positive")
    
    pool = await get_db_pool()
    
    async with pool.acquire() as conn:
        # This is a destructive operation - should only be done during migrations
        logger.warning(f"Updating embedding dimensions to {new_dimensions}")
        
        # Drop existing indexes
        await conn.execute("DROP INDEX IF EXISTS idx_semantic_embedding")
        await conn.execute("DROP INDEX IF EXISTS idx_episodic_embedding")  
        await conn.execute("DROP INDEX IF EXISTS idx_procedural_embedding")
        
        # Alter column types
        await conn.execute(f"ALTER TABLE semantic_memories ALTER COLUMN embedding TYPE vector({new_dimensions})")
        await conn.execute(f"ALTER TABLE episodic_memories ALTER COLUMN embedding TYPE vector({new_dimensions})")
        await conn.execute(f"ALTER TABLE procedural_memories ALTER COLUMN embedding TYPE vector({new_dimensions})")
        
        # Recreate indexes
        await conn.execute(
            f"CREATE INDEX idx_semantic_embedding ON semantic_memories USING hnsw (embedding vector_cosine_ops)"
        )
        await conn.execute(
            f"CREATE INDEX idx_episodic_embedding ON episodic_memories USING hnsw (embedding vector_cosine_ops)"
        )
        await conn.execute(
            f"CREATE INDEX idx_procedural_embedding ON procedural_memories USING hnsw (embedding vector_cosine_ops)"
        )
        
        logger.info(f"Updated embedding dimensions to {new_dimensions}")