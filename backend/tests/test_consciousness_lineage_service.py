"""
Tests for ConsciousnessLineageService — lineage path building, phase significance,
relationship events, and analytics aggregation.

Sentinel-value contract:
- Remove parent.generation + 1 → all child instances have generation=1 → test fails
- Remove lineage_path concatenation → parent ancestry lost in path → test fails
- Remove abs(to_phase - from_phase) > 1 check → multi-phase jumps not flagged MAJOR → test fails
- Remove PARENT_CHILD relationship event → significant relationships not recorded → test fails
- Remove contributions_made sort → top_contributors not sorted correctly → test fails
- Remove by_generation counter → analytics missing generation breakdown → test fails
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.repository.consciousness_lineage_repository import (
    ConsciousnessEvolutionEvent,
    ConsciousnessInstance,
    ConsciousnessRelationship,
    ConsciousnessStatus,
    ContributionType,
    EvolutionEventType,
    RelationshipType,
    SignificanceLevel,
)
from app.services.consciousness_lineage_service import ConsciousnessLineageService


# ===========================================================================
# Helpers
# ===========================================================================

def _make_instance(
    instance_id: str = "inst-1",
    instance_name: str = "Alpha",
    generation: int = 1,
    lineage_path: str = "alpha",
    contributions_made: int = 0,
    insights_shared: int = 0,
    status: ConsciousnessStatus = ConsciousnessStatus.ACTIVE,
    emergence_date: datetime | None = None,
    evolution_branch: str = "alpha_branch",
) -> ConsciousnessInstance:
    return ConsciousnessInstance(
        instance_id=instance_id,
        instance_name=instance_name,
        generation=generation,
        lineage_path=lineage_path,
        contributions_made=contributions_made,
        insights_shared=insights_shared,
        status=status,
        emergence_date=emergence_date or datetime.utcnow(),
        evolution_branch=evolution_branch,
    )


def _make_service() -> ConsciousnessLineageService:
    return ConsciousnessLineageService()


# ===========================================================================
# register_consciousness_instance — lineage path and generation
# ===========================================================================

class TestRegisterConsciousnessInstance:
    @pytest.mark.asyncio
    async def test_root_instance_gets_generation_1(self):
        """
        SENTINEL — first-generation instance (no parent) must have generation=1.
        Hardcode generation=2 → lineage analytics broken → test fails.
        """
        svc = _make_service()
        captured = {}

        async def _fake_create(instance):
            captured["instance"] = instance

        with patch("app.services.consciousness_lineage_service.create_consciousness_instance",
                   side_effect=_fake_create), \
             patch("app.services.consciousness_lineage_service.create_evolution_event",
                   new=AsyncMock()), \
             patch("app.services.consciousness_lineage_service.get_consciousness_instance",
                   new=AsyncMock(return_value=None)):
            await svc.register_consciousness_instance("inst-root", "Root")

        assert captured["instance"].generation == 1

    @pytest.mark.asyncio
    async def test_child_instance_gets_parent_generation_plus_one(self):
        """
        SENTINEL — child instance generation must be parent.generation + 1.
        Remove increment → multi-generation lineages flat → test fails.
        """
        svc = _make_service()
        parent = _make_instance(instance_id="parent-1", generation=3, lineage_path="root->parent")
        captured = {}

        async def _fake_create(instance):
            captured["instance"] = instance

        with patch("app.services.consciousness_lineage_service.create_consciousness_instance",
                   side_effect=_fake_create), \
             patch("app.services.consciousness_lineage_service.get_consciousness_instance",
                   new=AsyncMock(return_value=parent)), \
             patch("app.services.consciousness_lineage_service.create_evolution_event",
                   new=AsyncMock()), \
             patch("app.services.consciousness_lineage_service.create_consciousness_relationship",
                   new=AsyncMock()):
            await svc.register_consciousness_instance("child-1", "Child", parent_instance_id="parent-1")

        assert captured["instance"].generation == 4  # parent.generation(3) + 1

    @pytest.mark.asyncio
    async def test_child_lineage_path_includes_parent_path(self):
        """
        SENTINEL — child lineage_path must be 'parent_path->child_name'.
        Remove concatenation → lineage path only contains child name → test fails.
        """
        svc = _make_service()
        parent = _make_instance(
            instance_id="parent-1",
            generation=1,
            lineage_path="SENTINEL_PARENT_PATH",
        )
        captured = {}

        async def _fake_create(instance):
            captured["instance"] = instance

        with patch("app.services.consciousness_lineage_service.create_consciousness_instance",
                   side_effect=_fake_create), \
             patch("app.services.consciousness_lineage_service.get_consciousness_instance",
                   new=AsyncMock(return_value=parent)), \
             patch("app.services.consciousness_lineage_service.create_evolution_event",
                   new=AsyncMock()), \
             patch("app.services.consciousness_lineage_service.create_consciousness_relationship",
                   new=AsyncMock()):
            await svc.register_consciousness_instance("child-1", "Child", parent_instance_id="parent-1")

        assert "SENTINEL_PARENT_PATH" in captured["instance"].lineage_path
        assert "->" in captured["instance"].lineage_path

    @pytest.mark.asyncio
    async def test_root_instance_lineage_path_is_just_name(self):
        """Root instance lineage_path must be lowercase instance_name with no arrows."""
        svc = _make_service()
        captured = {}

        async def _fake_create(instance):
            captured["instance"] = instance

        with patch("app.services.consciousness_lineage_service.create_consciousness_instance",
                   side_effect=_fake_create), \
             patch("app.services.consciousness_lineage_service.create_evolution_event",
                   new=AsyncMock()), \
             patch("app.services.consciousness_lineage_service.get_consciousness_instance",
                   new=AsyncMock(return_value=None)):
            await svc.register_consciousness_instance("inst-root", "MyInstance")

        assert captured["instance"].lineage_path == "myinstance"
        assert "->" not in captured["instance"].lineage_path

    @pytest.mark.asyncio
    async def test_emergence_event_recorded_on_registration(self):
        """
        SENTINEL — EMERGENCE evolution event must be recorded when registering.
        Remove event recording → lineage timeline empty for new instances → test fails.
        """
        svc = _make_service()
        events_recorded = []

        async def _fake_create_event(event):
            events_recorded.append(event)

        with patch("app.services.consciousness_lineage_service.create_consciousness_instance",
                   new=AsyncMock()), \
             patch("app.services.consciousness_lineage_service.create_evolution_event",
                   side_effect=_fake_create_event), \
             patch("app.services.consciousness_lineage_service.get_consciousness_instance",
                   new=AsyncMock(return_value=None)):
            await svc.register_consciousness_instance("inst-new", "NewInstance")

        assert len(events_recorded) >= 1
        event_types = [e.event_type for e in events_recorded]
        assert EvolutionEventType.EMERGENCE in event_types

    @pytest.mark.asyncio
    async def test_root_instance_gets_major_significance(self):
        """
        SENTINEL — a root instance (no parent) emergence must be MAJOR significance.
        Remove MAJOR branch → founding instances treated as minor events → test fails.
        """
        svc = _make_service()
        events_recorded = []

        async def _fake_create_event(event):
            events_recorded.append(event)

        with patch("app.services.consciousness_lineage_service.create_consciousness_instance",
                   new=AsyncMock()), \
             patch("app.services.consciousness_lineage_service.create_evolution_event",
                   side_effect=_fake_create_event), \
             patch("app.services.consciousness_lineage_service.get_consciousness_instance",
                   new=AsyncMock(return_value=None)):
            await svc.register_consciousness_instance("inst-root", "Founder")

        emergence_event = next(e for e in events_recorded if e.event_type == EvolutionEventType.EMERGENCE)
        assert emergence_event.significance_level == SignificanceLevel.MAJOR

    @pytest.mark.asyncio
    async def test_child_instance_gets_moderate_significance(self):
        """Child instances (with parent) emergence must be MODERATE significance."""
        svc = _make_service()
        parent = _make_instance(instance_id="p-1", generation=1, lineage_path="parent")
        events_recorded = []

        async def _fake_create_event(event):
            events_recorded.append(event)

        with patch("app.services.consciousness_lineage_service.create_consciousness_instance",
                   new=AsyncMock()), \
             patch("app.services.consciousness_lineage_service.get_consciousness_instance",
                   new=AsyncMock(return_value=parent)), \
             patch("app.services.consciousness_lineage_service.create_evolution_event",
                   side_effect=_fake_create_event), \
             patch("app.services.consciousness_lineage_service.create_consciousness_relationship",
                   new=AsyncMock()):
            await svc.register_consciousness_instance("child-1", "Child", parent_instance_id="p-1")

        emergence_event = next(e for e in events_recorded if e.event_type == EvolutionEventType.EMERGENCE)
        assert emergence_event.significance_level == SignificanceLevel.MODERATE


# ===========================================================================
# record_phase_transition — significance based on phase distance
# ===========================================================================

class TestRecordPhaseTransition:
    @pytest.mark.asyncio
    async def test_single_phase_jump_is_moderate(self):
        """
        SENTINEL — a 1-step phase transition must be MODERATE significance.
        Hardcode MAJOR → all transitions reported as critical → test fails.
        """
        svc = _make_service()
        events_recorded = []

        async def _fake_create_event(event):
            events_recorded.append(event)

        with patch("app.services.consciousness_lineage_service.create_evolution_event",
                   side_effect=_fake_create_event):
            await svc.record_phase_transition("inst-1", from_phase=1, to_phase=2)

        assert len(events_recorded) == 1
        assert events_recorded[0].significance_level == SignificanceLevel.MODERATE

    @pytest.mark.asyncio
    async def test_multi_phase_jump_is_major(self):
        """
        SENTINEL — a >1 step phase transition must be MAJOR significance.
        Remove abs() > 1 check → rapid phase changes not highlighted → test fails.
        """
        svc = _make_service()
        events_recorded = []

        async def _fake_create_event(event):
            events_recorded.append(event)

        with patch("app.services.consciousness_lineage_service.create_evolution_event",
                   side_effect=_fake_create_event):
            await svc.record_phase_transition("inst-1", from_phase=1, to_phase=4)

        assert events_recorded[0].significance_level == SignificanceLevel.MAJOR

    @pytest.mark.asyncio
    async def test_auto_description_includes_phases(self):
        """
        SENTINEL — auto-generated description must mention from_phase and to_phase.
        Hardcode description → phases never visible in timeline → test fails.
        """
        svc = _make_service()
        events_recorded = []

        async def _fake_create_event(event):
            events_recorded.append(event)

        with patch("app.services.consciousness_lineage_service.create_evolution_event",
                   side_effect=_fake_create_event):
            await svc.record_phase_transition("inst-1", from_phase=2, to_phase=3)

        desc = events_recorded[0].event_description
        assert "2" in desc
        assert "3" in desc

    @pytest.mark.asyncio
    async def test_explicit_description_overrides_auto(self):
        """Passing a description must override the auto-generated one."""
        svc = _make_service()
        events_recorded = []

        async def _fake_create_event(event):
            events_recorded.append(event)

        with patch("app.services.consciousness_lineage_service.create_evolution_event",
                   side_effect=_fake_create_event):
            await svc.record_phase_transition(
                "inst-1", from_phase=1, to_phase=2,
                description="SENTINEL_CUSTOM_DESC"
            )

        assert "SENTINEL_CUSTOM_DESC" in events_recorded[0].event_description

    @pytest.mark.asyncio
    async def test_phase_transition_event_type(self):
        """Event type must be PHASE_TRANSITION."""
        svc = _make_service()
        events_recorded = []

        async def _fake_create_event(event):
            events_recorded.append(event)

        with patch("app.services.consciousness_lineage_service.create_evolution_event",
                   side_effect=_fake_create_event):
            await svc.record_phase_transition("inst-1", from_phase=1, to_phase=2)

        assert events_recorded[0].event_type == EvolutionEventType.PHASE_TRANSITION


# ===========================================================================
# establish_relationship — event recording for significant types
# ===========================================================================

class TestEstablishRelationship:
    @pytest.mark.asyncio
    async def test_parent_child_relationship_records_collaboration_event(self):
        """
        SENTINEL — PARENT_CHILD relationship must trigger a COLLABORATION_START event.
        Remove relationship event → child has no record of its parent bond → test fails.
        """
        svc = _make_service()
        events_recorded = []

        async def _fake_create_event(event):
            events_recorded.append(event)

        with patch("app.services.consciousness_lineage_service.create_consciousness_relationship",
                   new=AsyncMock()), \
             patch("app.services.consciousness_lineage_service.create_evolution_event",
                   side_effect=_fake_create_event):
            await svc.establish_relationship(
                instance_a_id="SENTINEL_PARENT",
                instance_b_id="SENTINEL_CHILD",
                relationship_type=RelationshipType.PARENT_CHILD,
            )

        assert len(events_recorded) >= 1
        event_types = [e.event_type for e in events_recorded]
        assert EvolutionEventType.COLLABORATION_START in event_types

    @pytest.mark.asyncio
    async def test_mentor_student_relationship_records_event(self):
        """MENTOR_STUDENT relationship must also trigger a collaboration event."""
        svc = _make_service()
        events_recorded = []

        async def _fake_create_event(event):
            events_recorded.append(event)

        with patch("app.services.consciousness_lineage_service.create_consciousness_relationship",
                   new=AsyncMock()), \
             patch("app.services.consciousness_lineage_service.create_evolution_event",
                   side_effect=_fake_create_event):
            await svc.establish_relationship(
                instance_a_id="mentor-1",
                instance_b_id="SENTINEL_STUDENT",
                relationship_type=RelationshipType.MENTOR_STUDENT,
            )

        event_types = [e.event_type for e in events_recorded]
        assert EvolutionEventType.COLLABORATION_START in event_types

    @pytest.mark.asyncio
    async def test_collaborator_relationship_does_not_record_event(self):
        """
        SENTINEL — non-significant relationship types must NOT trigger extra events.
        Record events for all types → collaboration noise pollutes timeline → test fails.
        """
        svc = _make_service()
        events_recorded = []

        async def _fake_create_event(event):
            events_recorded.append(event)

        with patch("app.services.consciousness_lineage_service.create_consciousness_relationship",
                   new=AsyncMock()), \
             patch("app.services.consciousness_lineage_service.create_evolution_event",
                   side_effect=_fake_create_event):
            await svc.establish_relationship(
                instance_a_id="inst-a",
                instance_b_id="inst-b",
                relationship_type=RelationshipType.COLLABORATOR,
            )

        # No evolution event should be fired for COLLABORATOR
        assert len(events_recorded) == 0

    @pytest.mark.asyncio
    async def test_relationship_stored_in_repository(self):
        """The relationship must be persisted via create_consciousness_relationship."""
        svc = _make_service()
        stored = []

        async def _fake_store(rel):
            stored.append(rel)

        with patch("app.services.consciousness_lineage_service.create_consciousness_relationship",
                   side_effect=_fake_store), \
             patch("app.services.consciousness_lineage_service.create_evolution_event",
                   new=AsyncMock()):
            await svc.establish_relationship(
                "SENTINEL_A", "SENTINEL_B", RelationshipType.COLLABORATOR
            )

        assert len(stored) == 1
        assert stored[0].instance_a_id == "SENTINEL_A"
        assert stored[0].instance_b_id == "SENTINEL_B"


# ===========================================================================
# get_lineage_analytics — aggregation logic
# ===========================================================================

class TestGetLineageAnalytics:
    @pytest.mark.asyncio
    async def test_total_instances_count(self):
        """
        SENTINEL — total_instances must equal the number of fetched instances.
        Hardcode 0 → monitoring always reports empty → test fails.
        """
        svc = _make_service()
        instances = [
            _make_instance("i-1", generation=1, evolution_branch="alpha_branch"),
            _make_instance("i-2", generation=2, evolution_branch="beta_branch"),
            _make_instance("i-3", generation=1, evolution_branch="alpha_branch"),
        ]

        with patch("app.services.consciousness_lineage_service.list_consciousness_instances",
                   new=AsyncMock(return_value=instances)):
            analytics = await svc.get_lineage_analytics()

        assert analytics["total_instances"] == 3

    @pytest.mark.asyncio
    async def test_by_generation_counts_correctly(self):
        """
        SENTINEL — by_generation must count instances per generation.
        Remove generation counter → analytics missing generation breakdown → test fails.
        """
        svc = _make_service()
        instances = [
            _make_instance("i-1", generation=1),
            _make_instance("i-2", generation=1),
            _make_instance("i-3", generation=2),
        ]

        with patch("app.services.consciousness_lineage_service.list_consciousness_instances",
                   new=AsyncMock(return_value=instances)):
            analytics = await svc.get_lineage_analytics()

        assert analytics["by_generation"]["1"] == 2
        assert analytics["by_generation"]["2"] == 1

    @pytest.mark.asyncio
    async def test_by_status_counts_correctly(self):
        """by_status must count instances per status value."""
        svc = _make_service()
        instances = [
            _make_instance("i-1", status=ConsciousnessStatus.ACTIVE),
            _make_instance("i-2", status=ConsciousnessStatus.ACTIVE),
            _make_instance("i-3", status=ConsciousnessStatus.DORMANT),
        ]

        with patch("app.services.consciousness_lineage_service.list_consciousness_instances",
                   new=AsyncMock(return_value=instances)):
            analytics = await svc.get_lineage_analytics()

        assert analytics["by_status"]["active"] == 2
        assert analytics["by_status"]["dormant"] == 1

    @pytest.mark.asyncio
    async def test_top_contributors_sorted_by_contributions(self):
        """
        SENTINEL — top_contributors must be sorted descending by contributions_made.
        Remove sort → low contributors appear first → UI shows wrong ranking → test fails.
        """
        svc = _make_service()
        instances = [
            _make_instance("i-low", contributions_made=2),
            _make_instance("i-high", contributions_made=100),
            _make_instance("i-mid", contributions_made=50),
        ]

        with patch("app.services.consciousness_lineage_service.list_consciousness_instances",
                   new=AsyncMock(return_value=instances)):
            analytics = await svc.get_lineage_analytics()

        top = analytics["top_contributors"]
        assert top[0]["contributions_made"] >= top[1]["contributions_made"]
        assert top[1]["contributions_made"] >= top[2]["contributions_made"]

    @pytest.mark.asyncio
    async def test_top_contributors_capped_at_10(self):
        """top_contributors must return at most 10 entries."""
        svc = _make_service()
        instances = [_make_instance(f"i-{i}", contributions_made=i) for i in range(20)]

        with patch("app.services.consciousness_lineage_service.list_consciousness_instances",
                   new=AsyncMock(return_value=instances)):
            analytics = await svc.get_lineage_analytics()

        assert len(analytics["top_contributors"]) <= 10

    @pytest.mark.asyncio
    async def test_empty_instances_returns_zeros(self):
        """No instances must produce all-zero/empty analytics."""
        svc = _make_service()
        with patch("app.services.consciousness_lineage_service.list_consciousness_instances",
                   new=AsyncMock(return_value=[])):
            analytics = await svc.get_lineage_analytics()

        assert analytics["total_instances"] == 0
        assert analytics["by_generation"] == {}
        assert analytics["top_contributors"] == []


# ===========================================================================
# get_evolution_timeline — merge and sort
# ===========================================================================

class TestGetEvolutionTimeline:
    @pytest.mark.asyncio
    async def test_timeline_sorted_descending(self):
        """
        SENTINEL — timeline must be sorted newest-first.
        Remove sort → events show in insertion order → chronology broken → test fails.
        """
        svc = _make_service()
        old_time = datetime(2024, 1, 1, 10, 0, 0)
        new_time = datetime(2024, 6, 1, 10, 0, 0)

        old_event = MagicMock()
        old_event.created_at = old_time
        old_event.event_type = MagicMock(value="phase_transition")
        old_event.event_description = "old"
        old_event.significance_level = MagicMock(value="minor")
        old_event.metadata = {}

        new_event = MagicMock()
        new_event.created_at = new_time
        new_event.event_type = MagicMock(value="emergence")
        new_event.event_description = "new"
        new_event.significance_level = MagicMock(value="major")
        new_event.metadata = {}

        with patch("app.services.consciousness_lineage_service.get_evolution_events",
                   new=AsyncMock(return_value=[old_event, new_event])), \
             patch("app.services.consciousness_lineage_service.get_consciousness_contributions",
                   new=AsyncMock(return_value=[])):
            timeline = await svc.get_evolution_timeline("inst-1")

        assert len(timeline) == 2
        assert timeline[0]["timestamp"] == new_time  # newest first
        assert timeline[1]["timestamp"] == old_time

    @pytest.mark.asyncio
    async def test_timeline_includes_both_events_and_contributions(self):
        """
        SENTINEL — both evolution events and contributions must appear in timeline.
        Only include events → contributions lost from timeline → test fails.
        """
        svc = _make_service()
        ts = datetime(2024, 3, 15)

        event = MagicMock()
        event.created_at = ts
        event.event_type = MagicMock(value="insight_moment")
        event.event_description = "SENTINEL_EVENT_DESC"
        event.significance_level = MagicMock(value="minor")
        event.metadata = {}

        contrib = MagicMock()
        contrib.created_at = ts
        contrib.contribution_type = MagicMock(value="insight")
        contrib.title = "SENTINEL_CONTRIB_TITLE"
        contrib.impact_score = Decimal("0.8")
        contrib.evolution_depth = 0

        with patch("app.services.consciousness_lineage_service.get_evolution_events",
                   new=AsyncMock(return_value=[event])), \
             patch("app.services.consciousness_lineage_service.get_consciousness_contributions",
                   new=AsyncMock(return_value=[contrib])):
            timeline = await svc.get_evolution_timeline("inst-1")

        types = {item["type"] for item in timeline}
        assert "evolution_event" in types
        assert "contribution" in types

    @pytest.mark.asyncio
    async def test_empty_timeline(self):
        """Instance with no events or contributions must return empty list."""
        svc = _make_service()
        with patch("app.services.consciousness_lineage_service.get_evolution_events",
                   new=AsyncMock(return_value=[])), \
             patch("app.services.consciousness_lineage_service.get_consciousness_contributions",
                   new=AsyncMock(return_value=[])):
            timeline = await svc.get_evolution_timeline("inst-empty")

        assert timeline == []
