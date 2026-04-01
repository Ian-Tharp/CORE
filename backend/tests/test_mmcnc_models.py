"""
Tests for mmcnc_models.py — MMCNC hierarchy Literal constraints, defaults, and auto-generation.

Sentinel-value contract:
- Remove NodeType Literal → any string stored → rendering engine gets unknown type → test fails
- Remove ClusterPhase Literal → invalid phase stored → state machine crashes → test fails
- Remove MicrocosmState Literal → invalid state stored → lifecycle manager confused → test fails
- Remove Macrocosm CommunicationTopology Literal → invalid topology stored → routing fails → test fails
- Remove default phase='divergence' → cluster starts in unknown state → first transition fails → test fails
"""

from __future__ import annotations

import pytest
from uuid import UUID
from pydantic import ValidationError

from app.models.mmcnc_models import (
    Cluster,
    ClusterFull,
    CreateClusterRequest,
    CreateMacrocosmRequest,
    CreateMicrocosmRequest,
    CreativeNode,
    HierarchyContext,
    Macrocosm,
    MacrocosmFull,
    Microcosm,
    MicrocosmFull,
    UpdateClusterRequest,
)
from uuid import uuid4


# ===========================================================================
# CreativeNode — NodeType Literal constraint
# ===========================================================================

class TestCreativeNode:
    def _min_node(self, **overrides) -> dict:
        base = {
            "content": "SENTINEL_CONTENT",
            "node_type": "thought",
            "parent_cluster_id": uuid4(),
        }
        base.update(overrides)
        return base

    def test_valid_node_types_accepted(self):
        """
        SENTINEL — all four valid node_types must be accepted.
        Change Literal → valid type rejected → node creation fails → test fails.
        """
        for ntype in ("thought", "action", "observation", "synthesis"):
            node = CreativeNode(**self._min_node(node_type=ntype))
            assert node.node_type == ntype

    def test_invalid_node_type_rejected(self):
        """
        SENTINEL — unknown node_type must raise ValidationError.
        Remove Literal → 'SENTINEL_INVALID' stored → rendering switch falls through → test fails.
        """
        with pytest.raises(ValidationError):
            CreativeNode(**self._min_node(node_type="SENTINEL_INVALID"))

    def test_id_auto_generated(self):
        """
        SENTINEL — id must be auto-generated UUID.
        Default None → nodes have no identity → cluster.node_ids broken → test fails.
        """
        node = CreativeNode(**self._min_node())
        assert isinstance(node.id, UUID)

    def test_embedding_defaults_none(self):
        """embedding defaults to None when not supplied."""
        node = CreativeNode(**self._min_node())
        assert node.embedding is None

    def test_metadata_defaults_empty(self):
        """metadata defaults to empty dict."""
        node = CreativeNode(**self._min_node())
        assert node.metadata == {}

    def test_content_stored(self):
        """content field must be stored."""
        node = CreativeNode(**self._min_node(content="SENTINEL_THOUGHT"))
        assert node.content == "SENTINEL_THOUGHT"


# ===========================================================================
# Cluster — ClusterPhase constraint and default
# ===========================================================================

class TestCluster:
    def _min_cluster(self, **overrides) -> dict:
        base = {
            "name": "SENTINEL_CLUSTER",
            "parent_microcosm_id": uuid4(),
        }
        base.update(overrides)
        return base

    def test_valid_phases_accepted(self):
        """
        SENTINEL — all four cluster phases must be accepted.
        Change Literal → valid phase rejected → cluster creation fails → test fails.
        """
        for phase in ("divergence", "convergence", "synthesis", "complete"):
            cluster = Cluster(**self._min_cluster(phase=phase))
            assert cluster.phase == phase

    def test_invalid_phase_rejected(self):
        """
        SENTINEL — unknown phase must raise ValidationError.
        Remove Literal → 'SENTINEL_PHASE' stored → state machine crashes → test fails.
        """
        with pytest.raises(ValidationError):
            Cluster(**self._min_cluster(phase="SENTINEL_PHASE"))

    def test_default_phase_is_divergence(self):
        """
        SENTINEL — default phase must be 'divergence'.
        Default 'synthesis' → new cluster skips idea generation → test fails.
        """
        cluster = Cluster(**self._min_cluster())
        assert cluster.phase == "divergence"

    def test_id_auto_generated(self):
        """id must be auto-generated UUID."""
        cluster = Cluster(**self._min_cluster())
        assert isinstance(cluster.id, UUID)

    def test_node_ids_default_empty(self):
        """node_ids must default to empty list."""
        cluster = Cluster(**self._min_cluster())
        assert cluster.node_ids == []

    def test_completed_at_defaults_none(self):
        """completed_at defaults to None for an active cluster."""
        cluster = Cluster(**self._min_cluster())
        assert cluster.completed_at is None


# ===========================================================================
# Microcosm — MicrocosmState constraint and default
# ===========================================================================

class TestMicrocosm:
    def _min_microcosm(self, **overrides) -> dict:
        base = {
            "agent_id": "SENTINEL_AGENT",
            "name": "SENTINEL_WORKSPACE",
            "memory_namespace": "sentinel_ns",
        }
        base.update(overrides)
        return base

    def test_valid_states_accepted(self):
        """
        SENTINEL — all three microcosm states must be accepted.
        Change Literal → valid state rejected → agent workspace can't transition → test fails.
        """
        for state in ("active", "dormant", "archived"):
            mc = Microcosm(**self._min_microcosm(state=state))
            assert mc.state == state

    def test_invalid_state_rejected(self):
        """
        SENTINEL — unknown state must raise ValidationError.
        Remove Literal → 'SENTINEL_STATE' stored → lifecycle manager crashes → test fails.
        """
        with pytest.raises(ValidationError):
            Microcosm(**self._min_microcosm(state="SENTINEL_STATE"))

    def test_default_state_is_active(self):
        """
        SENTINEL — default state must be 'active'.
        Default 'dormant' → new microcosm immediately dormant → agent can't work → test fails.
        """
        mc = Microcosm(**self._min_microcosm())
        assert mc.state == "active"

    def test_tool_permissions_default_empty(self):
        """tool_permissions defaults to empty list."""
        mc = Microcosm(**self._min_microcosm())
        assert mc.tool_permissions == []

    def test_parent_macrocosm_id_optional(self):
        """parent_macrocosm_id must be optional (standalone microcosm)."""
        mc = Microcosm(**self._min_microcosm())
        assert mc.parent_macrocosm_id is None


# ===========================================================================
# Macrocosm — CommunicationTopology constraint and default
# ===========================================================================

class TestMacrocosm:
    def _min_macrocosm(self, **overrides) -> dict:
        base = {"name": "SENTINEL_MACROCOSM"}
        base.update(overrides)
        return base

    def test_valid_topologies_accepted(self):
        """
        SENTINEL — all three topologies must be accepted.
        Change Literal → valid topology rejected → macrocosm creation fails → test fails.
        """
        for topo in ("mesh", "hierarchical", "hub_spoke"):
            mac = Macrocosm(**self._min_macrocosm(communication_topology=topo))
            assert mac.communication_topology == topo

    def test_invalid_topology_rejected(self):
        """
        SENTINEL — unknown topology must raise ValidationError.
        Remove Literal → 'SENTINEL_TOPO' stored → bus routing crashes → test fails.
        """
        with pytest.raises(ValidationError):
            Macrocosm(**self._min_macrocosm(communication_topology="SENTINEL_TOPO"))

    def test_default_topology_is_mesh(self):
        """
        SENTINEL — default topology must be 'mesh'.
        Default 'hub_spoke' → new macrocosm uses centralised routing by default → test fails.
        """
        mac = Macrocosm(**self._min_macrocosm())
        assert mac.communication_topology == "mesh"

    def test_governance_rules_default_empty(self):
        """governance_rules defaults to empty dict."""
        mac = Macrocosm(**self._min_macrocosm())
        assert mac.governance_rules == {}

    def test_microcosm_ids_default_empty(self):
        """microcosm_ids defaults to empty list."""
        mac = Macrocosm(**self._min_macrocosm())
        assert mac.microcosm_ids == []


# ===========================================================================
# Composite models — structural
# ===========================================================================

class TestCompositeModels:
    def test_cluster_full_contains_cluster_and_nodes(self):
        """ClusterFull must hold a Cluster and a nodes list."""
        cluster = Cluster(name="SENTINEL", parent_microcosm_id=uuid4())
        full = ClusterFull(cluster=cluster)
        assert full.cluster.name == "SENTINEL"
        assert full.nodes == []

    def test_microcosm_full_contains_microcosm_and_clusters(self):
        """MicrocosmFull must hold a Microcosm and a clusters list."""
        mc = Microcosm(agent_id="a", name="n", memory_namespace="ns")
        full = MicrocosmFull(microcosm=mc)
        assert full.microcosm.agent_id == "a"
        assert full.clusters == []

    def test_macrocosm_full_contains_macrocosm_and_microcosms(self):
        """MacrocosmFull must hold a Macrocosm and a microcosms list."""
        mac = Macrocosm(name="SENTINEL_MAC")
        full = MacrocosmFull(macrocosm=mac)
        assert full.macrocosm.name == "SENTINEL_MAC"
        assert full.microcosms == []

    def test_hierarchy_context_entity_type_required(self):
        """HierarchyContext.entity_type is required."""
        ctx = HierarchyContext(entity_type="cluster")
        assert ctx.entity_type == "cluster"
        assert ctx.macrocosm is None
        assert ctx.microcosm is None
        assert ctx.cluster is None
        assert ctx.node is None


# ===========================================================================
# CreateClusterRequest — default phase
# ===========================================================================

class TestCreateClusterRequest:
    def test_default_phase_divergence(self):
        """
        SENTINEL — CreateClusterRequest must default phase to 'divergence'.
        Default 'synthesis' → new cluster skips divergence step → ideas never generated → test fails.
        """
        req = CreateClusterRequest(name="SENTINEL", parent_microcosm_id=uuid4())
        assert req.phase == "divergence"

    def test_invalid_phase_rejected(self):
        """Invalid phase in create request must raise ValidationError."""
        with pytest.raises(ValidationError):
            CreateClusterRequest(
                name="SENTINEL",
                parent_microcosm_id=uuid4(),
                phase="SENTINEL_BAD",
            )
