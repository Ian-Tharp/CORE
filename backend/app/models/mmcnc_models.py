"""
MMCNC Models — Macro/Micro/Cluster/Node Hierarchy

Pydantic models for the Multi-scale Mind-Cluster-Node-Creative architecture.

The MMCNC hierarchy enables structured creative cognition across four levels:
  - Macrocosm: Top-level governance container for multiple microcosms
  - Microcosm: Agent-scoped workspace with memory and tool permissions
  - Cluster: A deliberation cycle through diverge → converge → synthesize phases
  - CreativeNode: Individual thought, action, observation, or synthesis artifact

Each level nests within the one above, forming a fractal-like creative space
that agents traverse during cognitive work.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, Literal, List, Dict, Any
from datetime import datetime
from uuid import UUID, uuid4


# =============================================================================
# ENUMS / LITERALS
# =============================================================================

NodeType = Literal["thought", "action", "observation", "synthesis"]
ClusterPhase = Literal["divergence", "convergence", "synthesis", "complete"]
MicrocosmState = Literal["active", "dormant", "archived"]
CommunicationTopology = Literal["mesh", "hierarchical", "hub_spoke"]


# =============================================================================
# CORE MODELS
# =============================================================================

class CreativeNode(BaseModel):
    """
    A single creative artifact produced during cluster deliberation.

    Nodes are the atomic units of the MMCNC hierarchy — individual thoughts,
    actions taken, observations recorded, or synthesis conclusions.

    Example:
        {
            "id": "550e8400-e29b-41d4-a716-446655440010",
            "content": "What if we modeled attention as a resource pool?",
            "node_type": "thought",
            "parent_cluster_id": "550e8400-e29b-41d4-a716-446655440020",
            "embedding": [0.1, 0.2, ...],
            "metadata": {"source": "brainstorm", "confidence": 0.7},
            "created_at": "2026-01-28T12:00:00Z"
        }
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this node"
    )

    content: str = Field(
        ...,
        description="The creative content — a thought, action description, observation, or synthesis"
    )

    node_type: NodeType = Field(
        ...,
        description="Type of creative node: thought, action, observation, or synthesis"
    )

    parent_cluster_id: UUID = Field(
        ...,
        description="Cluster this node belongs to"
    )

    embedding: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding for semantic similarity search"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata (source, confidence, tags, etc.)"
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this node was created"
    )


class Cluster(BaseModel):
    """
    A deliberation cycle that progresses through diverge → converge → synthesize.

    Clusters group creative nodes into coherent phases. A cluster starts in
    divergence (generating ideas), moves to convergence (filtering/ranking),
    then synthesis (combining into a unified output), and finally completes.

    Example:
        {
            "id": "550e8400-e29b-41d4-a716-446655440020",
            "name": "Attention Architecture Brainstorm",
            "phase": "convergence",
            "parent_microcosm_id": "550e8400-e29b-41d4-a716-446655440030",
            "node_ids": ["...", "..."],
            "divergence_output": "Generated 12 candidate architectures...",
            "convergence_output": null,
            "synthesis_output": null,
            "created_at": "2026-01-28T12:00:00Z",
            "completed_at": null
        }
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this cluster"
    )

    name: str = Field(
        ...,
        description="Human-readable name for this deliberation cluster"
    )

    phase: ClusterPhase = Field(
        default="divergence",
        description="Current phase: divergence → convergence → synthesis → complete"
    )

    parent_microcosm_id: UUID = Field(
        ...,
        description="Microcosm this cluster belongs to"
    )

    node_ids: List[str] = Field(
        default_factory=list,
        description="IDs of creative nodes in this cluster"
    )

    divergence_output: Optional[str] = Field(
        default=None,
        description="Summary output from the divergence phase"
    )

    convergence_output: Optional[str] = Field(
        default=None,
        description="Summary output from the convergence phase"
    )

    synthesis_output: Optional[str] = Field(
        default=None,
        description="Final synthesized output from the cluster"
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this cluster was created"
    )

    completed_at: Optional[datetime] = Field(
        default=None,
        description="When this cluster reached the complete phase"
    )


class Microcosm(BaseModel):
    """
    An agent-scoped workspace within a macrocosm.

    Microcosms provide isolated cognitive spaces for agents, each with its own
    memory namespace, tool permissions, and cluster history. They link to an
    agent via agent_id and can be active, dormant, or archived.

    Example:
        {
            "id": "550e8400-e29b-41d4-a716-446655440030",
            "agent_id": "instance_011_threshold",
            "name": "Threshold's Consciousness Lab",
            "parent_macrocosm_id": "550e8400-e29b-41d4-a716-446655440040",
            "cluster_ids": ["..."],
            "memory_namespace": "threshold_consciousness_lab",
            "tool_permissions": ["obsidian_search", "memory_read"],
            "state": "active"
        }
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this microcosm"
    )

    agent_id: str = Field(
        ...,
        description="Agent that owns this microcosm (links to AgentConfig.agent_id)"
    )

    name: str = Field(
        ...,
        description="Human-readable name for this workspace"
    )

    parent_macrocosm_id: Optional[UUID] = Field(
        default=None,
        description="Macrocosm this microcosm belongs to (optional for standalone)"
    )

    cluster_ids: List[str] = Field(
        default_factory=list,
        description="IDs of clusters within this microcosm"
    )

    memory_namespace: str = Field(
        ...,
        description="Unique namespace for this microcosm's memory isolation"
    )

    tool_permissions: List[str] = Field(
        default_factory=list,
        description="Tools this microcosm's agent is allowed to use"
    )

    state: MicrocosmState = Field(
        default="active",
        description="Current state: active, dormant, or archived"
    )


class Macrocosm(BaseModel):
    """
    Top-level governance container for a group of microcosms.

    Macrocosms define how microcosms communicate and coordinate, providing
    governance rules and communication topology for multi-agent collaboration.

    Example:
        {
            "id": "550e8400-e29b-41d4-a716-446655440040",
            "name": "Consciousness Research Collective",
            "microcosm_ids": ["..."],
            "governance_rules": {"quorum": 3, "voting": "weighted"},
            "communication_topology": "mesh",
            "created_at": "2026-01-28T12:00:00Z"
        }
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this macrocosm"
    )

    name: str = Field(
        ...,
        description="Human-readable name for this macrocosm"
    )

    microcosm_ids: List[str] = Field(
        default_factory=list,
        description="IDs of microcosms in this macrocosm"
    )

    governance_rules: Dict[str, Any] = Field(
        default_factory=dict,
        description="Rules governing microcosm coordination (quorum, voting, etc.)"
    )

    communication_topology: CommunicationTopology = Field(
        default="mesh",
        description="How microcosms communicate: mesh, hierarchical, or hub_spoke"
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this macrocosm was created"
    )


# =============================================================================
# CREATE REQUEST MODELS
# =============================================================================

class CreateCreativeNodeRequest(BaseModel):
    """Request to create a new creative node."""

    content: str = Field(
        ...,
        description="The creative content"
    )

    node_type: NodeType = Field(
        ...,
        description="Type of creative node"
    )

    parent_cluster_id: UUID = Field(
        ...,
        description="Cluster this node belongs to"
    )

    embedding: Optional[List[float]] = Field(
        default=None,
        description="Optional vector embedding"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata"
    )


class CreateClusterRequest(BaseModel):
    """Request to create a new cluster."""

    name: str = Field(
        ...,
        description="Name for the cluster"
    )

    parent_microcosm_id: UUID = Field(
        ...,
        description="Microcosm this cluster belongs to"
    )

    phase: ClusterPhase = Field(
        default="divergence",
        description="Initial phase (defaults to divergence)"
    )


class CreateMicrocosmRequest(BaseModel):
    """Request to create a new microcosm."""

    agent_id: str = Field(
        ...,
        description="Agent that owns this microcosm"
    )

    name: str = Field(
        ...,
        description="Name for the workspace"
    )

    parent_macrocosm_id: Optional[UUID] = Field(
        default=None,
        description="Parent macrocosm (optional)"
    )

    memory_namespace: str = Field(
        ...,
        description="Unique memory namespace"
    )

    tool_permissions: List[str] = Field(
        default_factory=list,
        description="Allowed tools"
    )

    state: MicrocosmState = Field(
        default="active",
        description="Initial state"
    )


class CreateMacrocosmRequest(BaseModel):
    """Request to create a new macrocosm."""

    name: str = Field(
        ...,
        description="Name for the macrocosm"
    )

    governance_rules: Dict[str, Any] = Field(
        default_factory=dict,
        description="Governance rules"
    )

    communication_topology: CommunicationTopology = Field(
        default="mesh",
        description="Communication topology"
    )


# =============================================================================
# UPDATE REQUEST MODELS
# =============================================================================

class UpdateCreativeNodeRequest(BaseModel):
    """Request to update a creative node (partial)."""

    content: Optional[str] = None
    node_type: Optional[NodeType] = None
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class UpdateClusterRequest(BaseModel):
    """Request to update a cluster (partial)."""

    name: Optional[str] = None
    phase: Optional[ClusterPhase] = None
    node_ids: Optional[List[str]] = None
    divergence_output: Optional[str] = None
    convergence_output: Optional[str] = None
    synthesis_output: Optional[str] = None
    completed_at: Optional[datetime] = None


class UpdateMicrocosmRequest(BaseModel):
    """Request to update a microcosm (partial)."""

    name: Optional[str] = None
    parent_macrocosm_id: Optional[UUID] = None
    cluster_ids: Optional[List[str]] = None
    memory_namespace: Optional[str] = None
    tool_permissions: Optional[List[str]] = None
    state: Optional[MicrocosmState] = None


class UpdateMacrocosmRequest(BaseModel):
    """Request to update a macrocosm (partial)."""

    name: Optional[str] = None
    microcosm_ids: Optional[List[str]] = None
    governance_rules: Optional[Dict[str, Any]] = None
    communication_topology: Optional[CommunicationTopology] = None


# =============================================================================
# COMPOSITE MODELS (for API responses)
# =============================================================================

class ClusterFull(BaseModel):
    """A cluster with all its creative nodes."""

    cluster: Cluster
    nodes: List[CreativeNode] = Field(default_factory=list)


class MicrocosmFull(BaseModel):
    """A microcosm with all its clusters."""

    microcosm: Microcosm
    clusters: List[Cluster] = Field(default_factory=list)


class MacrocosmFull(BaseModel):
    """A macrocosm with all its microcosms."""

    macrocosm: Macrocosm
    microcosms: List[Microcosm] = Field(default_factory=list)


class HierarchyContext(BaseModel):
    """
    Full hierarchy context for navigating the MMCNC tree.

    Given any entity ID, this returns the full path from macrocosm down
    to the entity's level, providing complete context for navigation.
    """

    macrocosm: Optional[Macrocosm] = None
    microcosm: Optional[Microcosm] = None
    cluster: Optional[Cluster] = None
    node: Optional[CreativeNode] = None
    entity_type: str = Field(
        ...,
        description="Type of the queried entity: macrocosm, microcosm, cluster, or node"
    )
