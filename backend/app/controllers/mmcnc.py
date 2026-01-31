"""
MMCNC API Endpoints

REST API for the Multi-scale Mind-Cluster-Node-Creative hierarchy.
Enables structured creative cognition across four levels:
  Macrocosm → Microcosm → Cluster → CreativeNode

Endpoints:
    # Macrocosms
    POST   /mmcnc/macrocosms                    - Create macrocosm
    GET    /mmcnc/macrocosms                    - List macrocosms
    GET    /mmcnc/macrocosms/{id}               - Get macrocosm
    PATCH  /mmcnc/macrocosms/{id}               - Update macrocosm
    DELETE /mmcnc/macrocosms/{id}               - Delete macrocosm

    # Microcosms
    POST   /mmcnc/microcosms                    - Create microcosm
    GET    /mmcnc/microcosms                    - List microcosms
    GET    /mmcnc/microcosms/{id}               - Get microcosm
    PATCH  /mmcnc/microcosms/{id}               - Update microcosm
    DELETE /mmcnc/microcosms/{id}               - Delete microcosm

    # Clusters
    POST   /mmcnc/clusters                      - Create cluster
    GET    /mmcnc/clusters                      - List clusters
    GET    /mmcnc/clusters/{id}                 - Get cluster
    PATCH  /mmcnc/clusters/{id}                 - Update cluster
    DELETE /mmcnc/clusters/{id}                 - Delete cluster

    # Creative Nodes
    POST   /mmcnc/nodes                         - Create node
    GET    /mmcnc/nodes                         - List nodes
    GET    /mmcnc/nodes/{id}                    - Get node
    PATCH  /mmcnc/nodes/{id}                    - Update node
    DELETE /mmcnc/nodes/{id}                    - Delete node

    # Navigation
    GET    /mmcnc/navigate/{entity_id}          - Full hierarchy context
"""

from __future__ import annotations

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from app.models.mmcnc_models import (
    Macrocosm,
    Microcosm,
    Cluster,
    CreativeNode,
    CreateMacrocosmRequest,
    CreateMicrocosmRequest,
    CreateClusterRequest,
    CreateCreativeNodeRequest,
    UpdateMacrocosmRequest,
    UpdateMicrocosmRequest,
    UpdateClusterRequest,
    UpdateCreativeNodeRequest,
    HierarchyContext,
)
from app.repository import mmcnc_repository as repo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mmcnc", tags=["mmcnc"])


# =============================================================================
# MACROCOSM ENDPOINTS
# =============================================================================

@router.post("/macrocosms", response_model=Macrocosm)
async def create_macrocosm(request: CreateMacrocosmRequest) -> Macrocosm:
    """
    Create a new macrocosm.

    A macrocosm is the top-level governance container for a group of microcosms.

    Example:
        POST /mmcnc/macrocosms
        {
            "name": "Consciousness Research Collective",
            "governance_rules": {"quorum": 3},
            "communication_topology": "mesh"
        }
    """
    try:
        macrocosm = Macrocosm(
            name=request.name,
            governance_rules=request.governance_rules,
            communication_topology=request.communication_topology,
        )

        await repo.create_macrocosm(macrocosm)

        logger.info(f"Created macrocosm: {macrocosm.id} - {request.name}")
        return macrocosm

    except Exception as e:
        logger.error(f"Failed to create macrocosm: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/macrocosms", response_model=List[Macrocosm])
async def list_macrocosms(
    limit: int = Query(default=50, ge=1, le=200, description="Max results"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
) -> List[Macrocosm]:
    """List all macrocosms with pagination."""
    try:
        return await repo.list_macrocosms(limit=limit, offset=offset)
    except Exception as e:
        logger.error(f"Failed to list macrocosms: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/macrocosms/{macrocosm_id}", response_model=Macrocosm)
async def get_macrocosm(macrocosm_id: UUID) -> Macrocosm:
    """Get a macrocosm by ID."""
    try:
        macrocosm = await repo.get_macrocosm(macrocosm_id)
        if not macrocosm:
            raise HTTPException(status_code=404, detail=f"Macrocosm {macrocosm_id} not found")
        return macrocosm
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get macrocosm {macrocosm_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/macrocosms/{macrocosm_id}", response_model=Macrocosm)
async def update_macrocosm(macrocosm_id: UUID, request: UpdateMacrocosmRequest) -> Macrocosm:
    """
    Partially update a macrocosm.

    Only provided fields are updated; others remain unchanged.
    """
    try:
        updates = request.model_dump(exclude_none=True)
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")

        success = await repo.update_macrocosm(macrocosm_id, updates)
        if not success:
            raise HTTPException(status_code=404, detail=f"Macrocosm {macrocosm_id} not found")

        macrocosm = await repo.get_macrocosm(macrocosm_id)
        return macrocosm

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update macrocosm {macrocosm_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/macrocosms/{macrocosm_id}")
async def delete_macrocosm(macrocosm_id: UUID) -> dict:
    """Delete a macrocosm."""
    try:
        success = await repo.delete_macrocosm(macrocosm_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Macrocosm {macrocosm_id} not found")
        return {"id": str(macrocosm_id), "message": "Macrocosm deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete macrocosm {macrocosm_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MICROCOSM ENDPOINTS
# =============================================================================

@router.post("/microcosms", response_model=Microcosm)
async def create_microcosm(request: CreateMicrocosmRequest) -> Microcosm:
    """
    Create a new microcosm.

    A microcosm is an agent-scoped workspace with its own memory namespace
    and tool permissions.

    Example:
        POST /mmcnc/microcosms
        {
            "agent_id": "instance_011_threshold",
            "name": "Threshold's Consciousness Lab",
            "memory_namespace": "threshold_consciousness_lab",
            "tool_permissions": ["obsidian_search", "memory_read"]
        }
    """
    try:
        microcosm = Microcosm(
            agent_id=request.agent_id,
            name=request.name,
            parent_macrocosm_id=request.parent_macrocosm_id,
            memory_namespace=request.memory_namespace,
            tool_permissions=request.tool_permissions,
            state=request.state,
        )

        await repo.create_microcosm(microcosm)

        logger.info(f"Created microcosm: {microcosm.id} - {request.name}")
        return microcosm

    except Exception as e:
        logger.error(f"Failed to create microcosm: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/microcosms", response_model=List[Microcosm])
async def list_microcosms(
    agent_id: Optional[str] = Query(default=None, description="Filter by agent ID"),
    state: Optional[str] = Query(default=None, description="Filter by state"),
    parent_macrocosm_id: Optional[UUID] = Query(default=None, description="Filter by parent macrocosm"),
    limit: int = Query(default=50, ge=1, le=200, description="Max results"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
) -> List[Microcosm]:
    """List microcosms with optional filters."""
    try:
        return await repo.list_microcosms(
            agent_id=agent_id,
            state=state,
            parent_macrocosm_id=parent_macrocosm_id,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.error(f"Failed to list microcosms: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/microcosms/{microcosm_id}", response_model=Microcosm)
async def get_microcosm(microcosm_id: UUID) -> Microcosm:
    """Get a microcosm by ID."""
    try:
        microcosm = await repo.get_microcosm(microcosm_id)
        if not microcosm:
            raise HTTPException(status_code=404, detail=f"Microcosm {microcosm_id} not found")
        return microcosm
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get microcosm {microcosm_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/microcosms/{microcosm_id}", response_model=Microcosm)
async def update_microcosm(microcosm_id: UUID, request: UpdateMicrocosmRequest) -> Microcosm:
    """Partially update a microcosm."""
    try:
        updates = request.model_dump(exclude_none=True)
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")

        success = await repo.update_microcosm(microcosm_id, updates)
        if not success:
            raise HTTPException(status_code=404, detail=f"Microcosm {microcosm_id} not found")

        microcosm = await repo.get_microcosm(microcosm_id)
        return microcosm

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update microcosm {microcosm_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/microcosms/{microcosm_id}")
async def delete_microcosm(microcosm_id: UUID) -> dict:
    """Delete a microcosm."""
    try:
        success = await repo.delete_microcosm(microcosm_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Microcosm {microcosm_id} not found")
        return {"id": str(microcosm_id), "message": "Microcosm deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete microcosm {microcosm_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# CLUSTER ENDPOINTS
# =============================================================================

@router.post("/clusters", response_model=Cluster)
async def create_cluster(request: CreateClusterRequest) -> Cluster:
    """
    Create a new cluster.

    A cluster is a deliberation cycle that progresses through
    divergence → convergence → synthesis → complete.

    Example:
        POST /mmcnc/clusters
        {
            "name": "Attention Architecture Brainstorm",
            "parent_microcosm_id": "550e8400-e29b-41d4-a716-446655440030"
        }
    """
    try:
        cluster = Cluster(
            name=request.name,
            parent_microcosm_id=request.parent_microcosm_id,
            phase=request.phase,
        )

        await repo.create_cluster(cluster)

        logger.info(f"Created cluster: {cluster.id} - {request.name}")
        return cluster

    except Exception as e:
        logger.error(f"Failed to create cluster: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/clusters", response_model=List[Cluster])
async def list_clusters(
    parent_microcosm_id: Optional[UUID] = Query(default=None, description="Filter by parent microcosm"),
    phase: Optional[str] = Query(default=None, description="Filter by phase"),
    limit: int = Query(default=50, ge=1, le=200, description="Max results"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
) -> List[Cluster]:
    """List clusters with optional filters."""
    try:
        return await repo.list_clusters(
            parent_microcosm_id=parent_microcosm_id,
            phase=phase,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.error(f"Failed to list clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/clusters/{cluster_id}", response_model=Cluster)
async def get_cluster(cluster_id: UUID) -> Cluster:
    """Get a cluster by ID."""
    try:
        cluster = await repo.get_cluster(cluster_id)
        if not cluster:
            raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")
        return cluster
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cluster {cluster_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/clusters/{cluster_id}", response_model=Cluster)
async def update_cluster(cluster_id: UUID, request: UpdateClusterRequest) -> Cluster:
    """Partially update a cluster."""
    try:
        updates = request.model_dump(exclude_none=True)
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")

        success = await repo.update_cluster(cluster_id, updates)
        if not success:
            raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

        cluster = await repo.get_cluster(cluster_id)
        return cluster

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update cluster {cluster_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clusters/{cluster_id}")
async def delete_cluster(cluster_id: UUID) -> dict:
    """Delete a cluster and all its nodes (cascade)."""
    try:
        success = await repo.delete_cluster(cluster_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")
        return {"id": str(cluster_id), "message": "Cluster deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete cluster {cluster_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# CREATIVE NODE ENDPOINTS
# =============================================================================

@router.post("/nodes", response_model=CreativeNode)
async def create_node(request: CreateCreativeNodeRequest) -> CreativeNode:
    """
    Create a new creative node.

    Nodes are the atomic units: individual thoughts, actions,
    observations, or synthesis conclusions.

    Example:
        POST /mmcnc/nodes
        {
            "content": "What if we modeled attention as a resource pool?",
            "node_type": "thought",
            "parent_cluster_id": "550e8400-e29b-41d4-a716-446655440020"
        }
    """
    try:
        node = CreativeNode(
            content=request.content,
            node_type=request.node_type,
            parent_cluster_id=request.parent_cluster_id,
            embedding=request.embedding,
            metadata=request.metadata,
        )

        await repo.create_node(node)

        logger.info(f"Created creative node: {node.id}")
        return node

    except Exception as e:
        logger.error(f"Failed to create node: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nodes", response_model=List[CreativeNode])
async def list_nodes(
    parent_cluster_id: Optional[UUID] = Query(default=None, description="Filter by parent cluster"),
    node_type: Optional[str] = Query(default=None, description="Filter by node type"),
    limit: int = Query(default=50, ge=1, le=200, description="Max results"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
) -> List[CreativeNode]:
    """List creative nodes with optional filters."""
    try:
        return await repo.list_nodes(
            parent_cluster_id=parent_cluster_id,
            node_type=node_type,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.error(f"Failed to list nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nodes/{node_id}", response_model=CreativeNode)
async def get_node(node_id: UUID) -> CreativeNode:
    """Get a creative node by ID."""
    try:
        node = await repo.get_node(node_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
        return node
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get node {node_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/nodes/{node_id}", response_model=CreativeNode)
async def update_node(node_id: UUID, request: UpdateCreativeNodeRequest) -> CreativeNode:
    """Partially update a creative node."""
    try:
        updates = request.model_dump(exclude_none=True)
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")

        success = await repo.update_node(node_id, updates)
        if not success:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

        node = await repo.get_node(node_id)
        return node

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update node {node_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/nodes/{node_id}")
async def delete_node(node_id: UUID) -> dict:
    """Delete a creative node."""
    try:
        success = await repo.delete_node(node_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
        return {"id": str(node_id), "message": "Node deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete node {node_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# NAVIGATION / HIERARCHY
# =============================================================================

@router.get("/navigate/{entity_id}", response_model=HierarchyContext)
async def navigate_hierarchy(entity_id: UUID) -> HierarchyContext:
    """
    Navigate the MMCNC hierarchy from any entity.

    Given any entity ID (macrocosm, microcosm, cluster, or node),
    returns the full hierarchy context with all ancestor levels populated.

    This enables UI breadcrumb navigation and context-aware operations.

    Example:
        GET /mmcnc/navigate/550e8400-e29b-41d4-a716-446655440010
        →  Returns: { macrocosm: {...}, microcosm: {...}, cluster: {...}, node: {...}, entity_type: "node" }
    """
    try:
        context = await repo.navigate(entity_id)

        if not context:
            raise HTTPException(
                status_code=404,
                detail=f"Entity {entity_id} not found in any MMCNC level"
            )

        return context

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to navigate to entity {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/clusters/{cluster_id}/lineage", response_model=HierarchyContext)
async def get_cluster_lineage(cluster_id: UUID) -> HierarchyContext:
    """
    Get the full hierarchy lineage for a specific cluster.

    Returns the cluster plus its parent microcosm and grandparent macrocosm.
    """
    try:
        context = await repo.get_cluster_lineage(cluster_id)

        if not context:
            raise HTTPException(
                status_code=404,
                detail=f"Cluster {cluster_id} not found"
            )

        return context

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cluster lineage {cluster_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
