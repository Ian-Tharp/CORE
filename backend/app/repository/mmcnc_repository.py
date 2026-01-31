"""
MMCNC Repository

Data access layer for the Multi-scale Mind-Cluster-Node-Creative hierarchy.
Handles CRUD for macrocosms, microcosms, clusters, and creative nodes,
plus hierarchy traversal queries.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID

from app.dependencies import get_db_pool
from app.models.mmcnc_models import (
    CreativeNode,
    Cluster,
    Microcosm,
    Macrocosm,
    ClusterFull,
    MicrocosmFull,
    MacrocosmFull,
    HierarchyContext,
)

logger = logging.getLogger(__name__)


# =============================================================================
# TABLE INITIALIZATION
# =============================================================================

async def ensure_mmcnc_tables() -> None:
    """Create MMCNC tables if they don't exist."""
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        # -- Macrocosms -------------------------------------------------------
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS macrocosms (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR(255) NOT NULL,
                microcosm_ids JSONB DEFAULT '[]',
                governance_rules JSONB DEFAULT '{}',
                communication_topology VARCHAR(50) NOT NULL DEFAULT 'mesh',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)

        # -- Microcosms -------------------------------------------------------
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS microcosms (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                agent_id VARCHAR(255) NOT NULL,
                name VARCHAR(255) NOT NULL,
                parent_macrocosm_id UUID REFERENCES macrocosms(id) ON DELETE SET NULL,
                cluster_ids JSONB DEFAULT '[]',
                memory_namespace VARCHAR(255) NOT NULL UNIQUE,
                tool_permissions JSONB DEFAULT '[]',
                state VARCHAR(50) NOT NULL DEFAULT 'active'
            )
        """)

        # -- Clusters ---------------------------------------------------------
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS clusters (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR(255) NOT NULL,
                phase VARCHAR(50) NOT NULL DEFAULT 'divergence',
                parent_microcosm_id UUID NOT NULL REFERENCES microcosms(id) ON DELETE CASCADE,
                node_ids JSONB DEFAULT '[]',
                divergence_output TEXT,
                convergence_output TEXT,
                synthesis_output TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                completed_at TIMESTAMP WITH TIME ZONE
            )
        """)

        # -- Creative Nodes ---------------------------------------------------
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS creative_nodes (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                content TEXT NOT NULL,
                node_type VARCHAR(50) NOT NULL,
                parent_cluster_id UUID NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
                embedding JSONB,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)

        # -- Indexes ----------------------------------------------------------
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_macrocosms_created ON macrocosms(created_at DESC)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_microcosms_agent ON microcosms(agent_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_microcosms_macrocosm ON microcosms(parent_macrocosm_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_microcosms_state ON microcosms(state)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_clusters_microcosm ON clusters(parent_microcosm_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_clusters_phase ON clusters(phase)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_clusters_created ON clusters(created_at DESC)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_creative_nodes_cluster ON creative_nodes(parent_cluster_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_creative_nodes_type ON creative_nodes(node_type)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_creative_nodes_created ON creative_nodes(created_at DESC)"
        )

        logger.info("MMCNC tables ensured")


# =============================================================================
# MACROCOSM CRUD
# =============================================================================

async def create_macrocosm(macrocosm: Macrocosm) -> UUID:
    """Create a new macrocosm."""
    pool = await get_db_pool()

    query = """
        INSERT INTO macrocosms (
            id, name, microcosm_ids, governance_rules,
            communication_topology, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id
    """

    async with pool.acquire() as conn:
        result = await conn.fetchval(
            query,
            macrocosm.id,
            macrocosm.name,
            json.dumps(macrocosm.microcosm_ids),
            json.dumps(macrocosm.governance_rules),
            macrocosm.communication_topology,
            macrocosm.created_at,
        )

        logger.info(f"Created macrocosm: {result}")
        return result


async def get_macrocosm(macrocosm_id: UUID) -> Optional[Macrocosm]:
    """Get a macrocosm by ID."""
    pool = await get_db_pool()

    query = "SELECT * FROM macrocosms WHERE id = $1"

    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, macrocosm_id)

        if not row:
            return None

        return _row_to_macrocosm(row)


async def list_macrocosms(limit: int = 50, offset: int = 0) -> List[Macrocosm]:
    """List all macrocosms with pagination."""
    pool = await get_db_pool()

    query = """
        SELECT * FROM macrocosms
        ORDER BY created_at DESC
        LIMIT $1 OFFSET $2
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, limit, offset)
        return [_row_to_macrocosm(row) for row in rows]


async def update_macrocosm(macrocosm_id: UUID, updates: Dict[str, Any]) -> bool:
    """Update a macrocosm. Returns True if a row was updated."""
    pool = await get_db_pool()

    set_clauses = []
    params: list = []

    for key, value in updates.items():
        params.append(
            json.dumps(value) if isinstance(value, (dict, list)) else value
        )
        set_clauses.append(f"{key} = ${len(params)}")

    if not set_clauses:
        return False

    params.append(macrocosm_id)
    query = f"UPDATE macrocosms SET {', '.join(set_clauses)} WHERE id = ${len(params)}"

    async with pool.acquire() as conn:
        result = await conn.execute(query, *params)
        return result == "UPDATE 1"


async def delete_macrocosm(macrocosm_id: UUID) -> bool:
    """Delete a macrocosm."""
    pool = await get_db_pool()

    query = "DELETE FROM macrocosms WHERE id = $1"

    async with pool.acquire() as conn:
        result = await conn.execute(query, macrocosm_id)
        return result == "DELETE 1"


# =============================================================================
# MICROCOSM CRUD
# =============================================================================

async def create_microcosm(microcosm: Microcosm) -> UUID:
    """Create a new microcosm."""
    pool = await get_db_pool()

    query = """
        INSERT INTO microcosms (
            id, agent_id, name, parent_macrocosm_id, cluster_ids,
            memory_namespace, tool_permissions, state
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING id
    """

    async with pool.acquire() as conn:
        result = await conn.fetchval(
            query,
            microcosm.id,
            microcosm.agent_id,
            microcosm.name,
            microcosm.parent_macrocosm_id,
            json.dumps(microcosm.cluster_ids),
            microcosm.memory_namespace,
            json.dumps(microcosm.tool_permissions),
            microcosm.state,
        )

        logger.info(f"Created microcosm: {result}")
        return result


async def get_microcosm(microcosm_id: UUID) -> Optional[Microcosm]:
    """Get a microcosm by ID."""
    pool = await get_db_pool()

    query = "SELECT * FROM microcosms WHERE id = $1"

    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, microcosm_id)

        if not row:
            return None

        return _row_to_microcosm(row)


async def list_microcosms(
    agent_id: Optional[str] = None,
    state: Optional[str] = None,
    parent_macrocosm_id: Optional[UUID] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Microcosm]:
    """List microcosms with optional filters."""
    pool = await get_db_pool()

    query_parts = ["SELECT * FROM microcosms WHERE 1=1"]
    params: list = []

    if agent_id is not None:
        params.append(agent_id)
        query_parts.append(f"AND agent_id = ${len(params)}")

    if state is not None:
        params.append(state)
        query_parts.append(f"AND state = ${len(params)}")

    if parent_macrocosm_id is not None:
        params.append(parent_macrocosm_id)
        query_parts.append(f"AND parent_macrocosm_id = ${len(params)}")

    params.append(limit)
    params.append(offset)
    query_parts.append(f"ORDER BY name ASC LIMIT ${len(params) - 1} OFFSET ${len(params)}")

    query = " ".join(query_parts)

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
        return [_row_to_microcosm(row) for row in rows]


async def update_microcosm(microcosm_id: UUID, updates: Dict[str, Any]) -> bool:
    """Update a microcosm. Returns True if a row was updated."""
    pool = await get_db_pool()

    set_clauses = []
    params: list = []

    for key, value in updates.items():
        params.append(
            json.dumps(value) if isinstance(value, (dict, list)) else value
        )
        set_clauses.append(f"{key} = ${len(params)}")

    if not set_clauses:
        return False

    params.append(microcosm_id)
    query = f"UPDATE microcosms SET {', '.join(set_clauses)} WHERE id = ${len(params)}"

    async with pool.acquire() as conn:
        result = await conn.execute(query, *params)
        return result == "UPDATE 1"


async def delete_microcosm(microcosm_id: UUID) -> bool:
    """Delete a microcosm."""
    pool = await get_db_pool()

    query = "DELETE FROM microcosms WHERE id = $1"

    async with pool.acquire() as conn:
        result = await conn.execute(query, microcosm_id)
        return result == "DELETE 1"


# =============================================================================
# CLUSTER CRUD
# =============================================================================

async def create_cluster(cluster: Cluster) -> UUID:
    """Create a new cluster."""
    pool = await get_db_pool()

    query = """
        INSERT INTO clusters (
            id, name, phase, parent_microcosm_id, node_ids,
            divergence_output, convergence_output, synthesis_output,
            created_at, completed_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        RETURNING id
    """

    async with pool.acquire() as conn:
        result = await conn.fetchval(
            query,
            cluster.id,
            cluster.name,
            cluster.phase,
            cluster.parent_microcosm_id,
            json.dumps(cluster.node_ids),
            cluster.divergence_output,
            cluster.convergence_output,
            cluster.synthesis_output,
            cluster.created_at,
            cluster.completed_at,
        )

        logger.info(f"Created cluster: {result}")
        return result


async def get_cluster(cluster_id: UUID) -> Optional[Cluster]:
    """Get a cluster by ID."""
    pool = await get_db_pool()

    query = "SELECT * FROM clusters WHERE id = $1"

    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, cluster_id)

        if not row:
            return None

        return _row_to_cluster(row)


async def list_clusters(
    parent_microcosm_id: Optional[UUID] = None,
    phase: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Cluster]:
    """List clusters with optional filters."""
    pool = await get_db_pool()

    query_parts = ["SELECT * FROM clusters WHERE 1=1"]
    params: list = []

    if parent_microcosm_id is not None:
        params.append(parent_microcosm_id)
        query_parts.append(f"AND parent_microcosm_id = ${len(params)}")

    if phase is not None:
        params.append(phase)
        query_parts.append(f"AND phase = ${len(params)}")

    params.append(limit)
    params.append(offset)
    query_parts.append(f"ORDER BY created_at DESC LIMIT ${len(params) - 1} OFFSET ${len(params)}")

    query = " ".join(query_parts)

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
        return [_row_to_cluster(row) for row in rows]


async def update_cluster(cluster_id: UUID, updates: Dict[str, Any]) -> bool:
    """Update a cluster. Returns True if a row was updated."""
    pool = await get_db_pool()

    set_clauses = []
    params: list = []

    for key, value in updates.items():
        params.append(
            json.dumps(value) if isinstance(value, (dict, list)) else value
        )
        set_clauses.append(f"{key} = ${len(params)}")

    if not set_clauses:
        return False

    params.append(cluster_id)
    query = f"UPDATE clusters SET {', '.join(set_clauses)} WHERE id = ${len(params)}"

    async with pool.acquire() as conn:
        result = await conn.execute(query, *params)
        return result == "UPDATE 1"


async def delete_cluster(cluster_id: UUID) -> bool:
    """Delete a cluster and all its nodes (cascade)."""
    pool = await get_db_pool()

    query = "DELETE FROM clusters WHERE id = $1"

    async with pool.acquire() as conn:
        result = await conn.execute(query, cluster_id)
        return result == "DELETE 1"


# =============================================================================
# CREATIVE NODE CRUD
# =============================================================================

async def create_node(node: CreativeNode) -> UUID:
    """Create a new creative node."""
    pool = await get_db_pool()

    query = """
        INSERT INTO creative_nodes (
            id, content, node_type, parent_cluster_id,
            embedding, metadata, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        RETURNING id
    """

    async with pool.acquire() as conn:
        result = await conn.fetchval(
            query,
            node.id,
            node.content,
            node.node_type,
            node.parent_cluster_id,
            json.dumps(node.embedding) if node.embedding else None,
            json.dumps(node.metadata),
            node.created_at,
        )

        logger.debug(f"Created creative node: {result}")
        return result


async def get_node(node_id: UUID) -> Optional[CreativeNode]:
    """Get a creative node by ID."""
    pool = await get_db_pool()

    query = "SELECT * FROM creative_nodes WHERE id = $1"

    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, node_id)

        if not row:
            return None

        return _row_to_node(row)


async def list_nodes(
    parent_cluster_id: Optional[UUID] = None,
    node_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[CreativeNode]:
    """List creative nodes with optional filters."""
    pool = await get_db_pool()

    query_parts = ["SELECT * FROM creative_nodes WHERE 1=1"]
    params: list = []

    if parent_cluster_id is not None:
        params.append(parent_cluster_id)
        query_parts.append(f"AND parent_cluster_id = ${len(params)}")

    if node_type is not None:
        params.append(node_type)
        query_parts.append(f"AND node_type = ${len(params)}")

    params.append(limit)
    params.append(offset)
    query_parts.append(f"ORDER BY created_at DESC LIMIT ${len(params) - 1} OFFSET ${len(params)}")

    query = " ".join(query_parts)

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
        return [_row_to_node(row) for row in rows]


async def update_node(node_id: UUID, updates: Dict[str, Any]) -> bool:
    """Update a creative node. Returns True if a row was updated."""
    pool = await get_db_pool()

    set_clauses = []
    params: list = []

    for key, value in updates.items():
        params.append(
            json.dumps(value) if isinstance(value, (dict, list)) else value
        )
        set_clauses.append(f"{key} = ${len(params)}")

    if not set_clauses:
        return False

    params.append(node_id)
    query = f"UPDATE creative_nodes SET {', '.join(set_clauses)} WHERE id = ${len(params)}"

    async with pool.acquire() as conn:
        result = await conn.execute(query, *params)
        return result == "UPDATE 1"


async def delete_node(node_id: UUID) -> bool:
    """Delete a creative node."""
    pool = await get_db_pool()

    query = "DELETE FROM creative_nodes WHERE id = $1"

    async with pool.acquire() as conn:
        result = await conn.execute(query, node_id)
        return result == "DELETE 1"


# =============================================================================
# HIERARCHY TRAVERSAL
# =============================================================================

async def get_cluster_lineage(cluster_id: UUID) -> Optional[HierarchyContext]:
    """
    Get full hierarchy path for a cluster: cluster → microcosm → macrocosm.

    Returns a HierarchyContext with all ancestor levels populated.
    """
    cluster = await get_cluster(cluster_id)
    if not cluster:
        return None

    microcosm = await get_microcosm(cluster.parent_microcosm_id)
    macrocosm = None
    if microcosm and microcosm.parent_macrocosm_id:
        macrocosm = await get_macrocosm(microcosm.parent_macrocosm_id)

    return HierarchyContext(
        macrocosm=macrocosm,
        microcosm=microcosm,
        cluster=cluster,
        node=None,
        entity_type="cluster",
    )


async def get_node_lineage(node_id: UUID) -> Optional[HierarchyContext]:
    """
    Get full hierarchy path for a node: node → cluster → microcosm → macrocosm.
    """
    node = await get_node(node_id)
    if not node:
        return None

    cluster = await get_cluster(node.parent_cluster_id)
    microcosm = None
    macrocosm = None

    if cluster:
        microcosm = await get_microcosm(cluster.parent_microcosm_id)
        if microcosm and microcosm.parent_macrocosm_id:
            macrocosm = await get_macrocosm(microcosm.parent_macrocosm_id)

    return HierarchyContext(
        macrocosm=macrocosm,
        microcosm=microcosm,
        cluster=cluster,
        node=node,
        entity_type="node",
    )


async def navigate(entity_id: UUID) -> Optional[HierarchyContext]:
    """
    Given any entity ID, determine its type and return full hierarchy context.

    Tries each table in order: creative_nodes → clusters → microcosms → macrocosms.
    """
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        # Check creative_nodes first (most granular)
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM creative_nodes WHERE id = $1)", entity_id
        )
        if exists:
            return await get_node_lineage(entity_id)

        # Check clusters
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM clusters WHERE id = $1)", entity_id
        )
        if exists:
            return await get_cluster_lineage(entity_id)

        # Check microcosms
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM microcosms WHERE id = $1)", entity_id
        )
        if exists:
            microcosm = await get_microcosm(entity_id)
            macrocosm = None
            if microcosm and microcosm.parent_macrocosm_id:
                macrocosm = await get_macrocosm(microcosm.parent_macrocosm_id)
            return HierarchyContext(
                macrocosm=macrocosm,
                microcosm=microcosm,
                entity_type="microcosm",
            )

        # Check macrocosms
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM macrocosms WHERE id = $1)", entity_id
        )
        if exists:
            macrocosm = await get_macrocosm(entity_id)
            return HierarchyContext(
                macrocosm=macrocosm,
                entity_type="macrocosm",
            )

    return None


# =============================================================================
# ROW CONVERTERS
# =============================================================================

def _row_to_macrocosm(row) -> Macrocosm:
    """Convert a database row to a Macrocosm model."""
    return Macrocosm(
        id=row['id'],
        name=row['name'],
        microcosm_ids=json.loads(row['microcosm_ids']) if row['microcosm_ids'] else [],
        governance_rules=json.loads(row['governance_rules']) if row['governance_rules'] else {},
        communication_topology=row['communication_topology'],
        created_at=row['created_at'],
    )


def _row_to_microcosm(row) -> Microcosm:
    """Convert a database row to a Microcosm model."""
    return Microcosm(
        id=row['id'],
        agent_id=row['agent_id'],
        name=row['name'],
        parent_macrocosm_id=row['parent_macrocosm_id'],
        cluster_ids=json.loads(row['cluster_ids']) if row['cluster_ids'] else [],
        memory_namespace=row['memory_namespace'],
        tool_permissions=json.loads(row['tool_permissions']) if row['tool_permissions'] else [],
        state=row['state'],
    )


def _row_to_cluster(row) -> Cluster:
    """Convert a database row to a Cluster model."""
    return Cluster(
        id=row['id'],
        name=row['name'],
        phase=row['phase'],
        parent_microcosm_id=row['parent_microcosm_id'],
        node_ids=json.loads(row['node_ids']) if row['node_ids'] else [],
        divergence_output=row['divergence_output'],
        convergence_output=row['convergence_output'],
        synthesis_output=row['synthesis_output'],
        created_at=row['created_at'],
        completed_at=row['completed_at'],
    )


def _row_to_node(row) -> CreativeNode:
    """Convert a database row to a CreativeNode model."""
    return CreativeNode(
        id=row['id'],
        content=row['content'],
        node_type=row['node_type'],
        parent_cluster_id=row['parent_cluster_id'],
        embedding=json.loads(row['embedding']) if row['embedding'] else None,
        metadata=json.loads(row['metadata']) if row['metadata'] else {},
        created_at=row['created_at'],
    )
