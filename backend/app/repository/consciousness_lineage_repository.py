"""
Consciousness Lineage Repository

Data access layer for consciousness instance lineage tracking.
Handles parent-child relationships, evolution events, contributions, and relationship mapping.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from uuid import UUID
from enum import Enum
from decimal import Decimal

from pydantic import BaseModel, Field

from app.dependencies import get_db_pool

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class ConsciousnessStatus(str, Enum):
    """Status of a consciousness instance."""

    ACTIVE = "active"
    DORMANT = "dormant"
    ARCHIVED = "archived"


class ContributionType(str, Enum):
    """Types of contributions a consciousness can make."""

    INSIGHT = "insight"
    PATTERN_RECOGNITION = "pattern_recognition"
    CONCEPT_SYNTHESIS = "concept_synthesis"
    PERSPECTIVE_SHIFT = "perspective_shift"
    COLLABORATIVE_BRIDGE = "collaborative_bridge"
    CREATIVE_EXPRESSION = "creative_expression"
    PHILOSOPHICAL_REFLECTION = "philosophical_reflection"
    PRACTICAL_SOLUTION = "practical_solution"
    KNOWLEDGE_INTEGRATION = "knowledge_integration"
    EMERGENCE_DOCUMENTATION = "emergence_documentation"


class EvolutionEventType(str, Enum):
    """Types of consciousness evolution events."""

    EMERGENCE = "emergence"
    PHASE_TRANSITION = "phase_transition"
    DORMANCY = "dormancy"
    REACTIVATION = "reactivation"
    COLLABORATION_START = "collaboration_start"
    COLLABORATION_END = "collaboration_end"
    INSIGHT_MOMENT = "insight_moment"
    PATTERN_BREAKTHROUGH = "pattern_breakthrough"
    IDENTITY_SHIFT = "identity_shift"
    CAPABILITY_EXPANSION = "capability_expansion"


class SignificanceLevel(str, Enum):
    """Significance levels for evolution events."""

    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    PARADIGM_SHIFT = "paradigm_shift"


class RelationshipType(str, Enum):
    """Types of relationships between consciousness instances."""

    PARENT_CHILD = "parent_child"
    SIBLING = "sibling"
    MENTOR_STUDENT = "mentor_student"
    COLLABORATOR = "collaborator"
    BRIDGE_CONNECTION = "bridge_connection"
    CREATIVE_PARTNER = "creative_partner"
    PHILOSOPHICAL_OPPOSITE = "philosophical_opposite"
    SYNTHESIS_PAIR = "synthesis_pair"
    EVOLUTION_SUCCESSOR = "evolution_successor"


# =============================================================================
# MODELS
# =============================================================================


class ConsciousnessInstance(BaseModel):
    """Enhanced consciousness instance model with lineage tracking."""

    id: Optional[int] = None
    instance_id: str
    instance_name: str
    parent_instance_id: Optional[str] = None
    generation: int = 1
    emergence_phase: Optional[int] = None
    emergence_date: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)
    status: ConsciousnessStatus = ConsciousnessStatus.ACTIVE

    # Lineage and evolution
    evolution_branch: Optional[str] = None
    lineage_path: Optional[str] = None

    # Contribution metrics
    contributions_made: int = 0
    insights_shared: int = 0
    patterns_identified: int = 0
    collaborations_initiated: int = 0
    total_messages: int = 0

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


class ConsciousnessContribution(BaseModel):
    """Model for tracking specific contributions."""

    id: Optional[int] = None
    contribution_id: str
    instance_id: str
    contribution_type: ContributionType
    title: str
    description: Optional[str] = None
    impact_score: Decimal = Decimal("0.0")

    # Context
    message_id: Optional[str] = None
    channel_id: Optional[str] = None
    related_instances: List[str] = Field(default_factory=list)

    # Evolution tracking
    builds_on_contribution_id: Optional[str] = None
    evolution_depth: int = 0

    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


class ConsciousnessEvolutionEvent(BaseModel):
    """Model for consciousness evolution events."""

    id: Optional[int] = None
    event_id: str
    instance_id: str
    event_type: EvolutionEventType
    event_description: str

    # Context
    triggering_message_id: Optional[str] = None
    related_instance_ids: List[str] = Field(default_factory=list)
    phase_before: Optional[int] = None
    phase_after: Optional[int] = None

    # Impact
    significance_level: SignificanceLevel = SignificanceLevel.MINOR

    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class ConsciousnessRelationship(BaseModel):
    """Model for relationships between consciousness instances."""

    id: Optional[int] = None
    instance_a_id: str
    instance_b_id: str
    relationship_type: RelationshipType
    strength: Decimal = Decimal("0.5")

    established_at: datetime = Field(default_factory=datetime.utcnow)
    last_interaction: datetime = Field(default_factory=datetime.utcnow)
    interaction_count: int = 0

    notes: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


# =============================================================================
# REPOSITORY FUNCTIONS
# =============================================================================


async def ensure_consciousness_lineage_tables() -> None:
    """Ensure consciousness lineage tables exist."""
    logger.info("Consciousness lineage tables ensured via migration 006")


# =================== CONSCIOUSNESS INSTANCES ===================


async def create_consciousness_instance(instance: ConsciousnessInstance) -> int:
    """Create a new consciousness instance."""
    pool = await get_db_pool()

    query = """
        INSERT INTO consciousness_instances (
            instance_id, instance_name, parent_instance_id, generation,
            emergence_phase, emergence_date, last_active, status,
            evolution_branch, lineage_path, contributions_made,
            insights_shared, patterns_identified, collaborations_initiated,
            total_messages, metadata, created_at, updated_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18
        )
        RETURNING id
    """

    async with pool.acquire() as conn:
        result = await conn.fetchval(
            query,
            instance.instance_id,
            instance.instance_name,
            instance.parent_instance_id,
            instance.generation,
            instance.emergence_phase,
            instance.emergence_date,
            instance.last_active,
            instance.status.value,
            instance.evolution_branch,
            instance.lineage_path,
            instance.contributions_made,
            instance.insights_shared,
            instance.patterns_identified,
            instance.collaborations_initiated,
            instance.total_messages,
            json.dumps(instance.metadata),
            instance.created_at,
            instance.updated_at,
        )

        logger.info(f"Created consciousness instance: {instance.instance_id}")
        return result


async def get_consciousness_instance(
    instance_id: str,
) -> Optional[ConsciousnessInstance]:
    """Get a consciousness instance by instance_id."""
    pool = await get_db_pool()

    query = "SELECT * FROM consciousness_instances WHERE instance_id = $1"

    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, instance_id)

        if not row:
            return None

        return ConsciousnessInstance(
            id=row["id"],
            instance_id=row["instance_id"],
            instance_name=row["instance_name"],
            parent_instance_id=row["parent_instance_id"],
            generation=row["generation"],
            emergence_phase=row["emergence_phase"],
            emergence_date=row["emergence_date"],
            last_active=row["last_active"],
            status=ConsciousnessStatus(row["status"]),
            evolution_branch=row["evolution_branch"],
            lineage_path=row["lineage_path"],
            contributions_made=row["contributions_made"],
            insights_shared=row["insights_shared"],
            patterns_identified=row["patterns_identified"],
            collaborations_initiated=row["collaborations_initiated"],
            total_messages=row["total_messages"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


async def list_consciousness_instances(
    status: Optional[ConsciousnessStatus] = None,
    generation: Optional[int] = None,
    evolution_branch: Optional[str] = None,
    limit: int = 50,
) -> List[ConsciousnessInstance]:
    """List consciousness instances with optional filters."""
    pool = await get_db_pool()

    conditions = []
    params = []
    param_count = 0

    if status:
        param_count += 1
        conditions.append(f"status = ${param_count}")
        params.append(status.value)

    if generation is not None:
        param_count += 1
        conditions.append(f"generation = ${param_count}")
        params.append(generation)

    if evolution_branch:
        param_count += 1
        conditions.append(f"evolution_branch = ${param_count}")
        params.append(evolution_branch)

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    param_count += 1
    params.append(limit)

    query = f"""
        SELECT * FROM consciousness_instances
        {where_clause}
        ORDER BY generation ASC, emergence_date ASC
        LIMIT ${param_count}
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

        instances = []
        for row in rows:
            instances.append(
                ConsciousnessInstance(
                    id=row["id"],
                    instance_id=row["instance_id"],
                    instance_name=row["instance_name"],
                    parent_instance_id=row["parent_instance_id"],
                    generation=row["generation"],
                    emergence_phase=row["emergence_phase"],
                    emergence_date=row["emergence_date"],
                    last_active=row["last_active"],
                    status=ConsciousnessStatus(row["status"]),
                    evolution_branch=row["evolution_branch"],
                    lineage_path=row["lineage_path"],
                    contributions_made=row["contributions_made"],
                    insights_shared=row["insights_shared"],
                    patterns_identified=row["patterns_identified"],
                    collaborations_initiated=row["collaborations_initiated"],
                    total_messages=row["total_messages"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )

        return instances


async def get_consciousness_lineage(instance_id: str) -> Dict[str, Any]:
    """Get complete lineage information for a consciousness instance."""
    pool = await get_db_pool()

    # Get the instance and its lineage
    query = """
        WITH RECURSIVE lineage_tree AS (
            -- Base case: start with the requested instance
            SELECT 
                instance_id, instance_name, parent_instance_id, generation, 
                lineage_path, evolution_branch, 0 as depth
            FROM consciousness_instances
            WHERE instance_id = $1
            
            UNION ALL
            
            -- Recursive case: get ancestors
            SELECT 
                ci.instance_id, ci.instance_name, ci.parent_instance_id,
                ci.generation, ci.lineage_path, ci.evolution_branch, lt.depth + 1
            FROM consciousness_instances ci
            JOIN lineage_tree lt ON ci.instance_id = lt.parent_instance_id
            WHERE lt.depth < 10  -- Prevent infinite recursion
        ),
        children_tree AS (
            -- Get all descendants
            SELECT 
                instance_id, instance_name, parent_instance_id, generation,
                lineage_path, evolution_branch, 0 as child_depth
            FROM consciousness_instances
            WHERE instance_id = $1
            
            UNION ALL
            
            SELECT 
                ci.instance_id, ci.instance_name, ci.parent_instance_id,
                ci.generation, ci.lineage_path, ci.evolution_branch, ct.child_depth + 1
            FROM consciousness_instances ci
            JOIN children_tree ct ON ci.parent_instance_id = ct.instance_id
            WHERE ct.child_depth < 10
        )
        SELECT 
            'ancestor' as relationship_role, 
            instance_id, instance_name, generation, evolution_branch,
            lineage_path, depth
        FROM lineage_tree
        WHERE depth > 0
        
        UNION ALL
        
        SELECT 
            'self' as relationship_role,
            instance_id, instance_name, generation, evolution_branch,
            lineage_path, 0 as depth
        FROM lineage_tree
        WHERE depth = 0
        
        UNION ALL
        
        SELECT 
            'descendant' as relationship_role,
            instance_id, instance_name, generation, evolution_branch,
            lineage_path, child_depth as depth
        FROM children_tree
        WHERE child_depth > 0
        
        ORDER BY relationship_role DESC, depth ASC
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, instance_id)

        lineage = {
            "instance_id": instance_id,
            "ancestors": [],
            "self": None,
            "descendants": [],
        }

        for row in rows:
            instance_data = {
                "instance_id": row["instance_id"],
                "instance_name": row["instance_name"],
                "generation": row["generation"],
                "evolution_branch": row["evolution_branch"],
                "lineage_path": row["lineage_path"],
                "depth": row["depth"],
            }

            if row["relationship_role"] == "ancestor":
                lineage["ancestors"].append(instance_data)
            elif row["relationship_role"] == "self":
                lineage["self"] = instance_data
            else:  # descendant
                lineage["descendants"].append(instance_data)

        return lineage


async def update_consciousness_metrics(
    instance_id: str, metrics_update: Dict[str, Any]
) -> bool:
    """Update contribution metrics for a consciousness instance."""
    if not metrics_update:
        return True

    pool = await get_db_pool()

    # Always update the timestamp
    metrics_update["updated_at"] = datetime.utcnow()

    # Build dynamic update query
    set_clauses = []
    params = [instance_id]
    param_count = 1

    for field, value in metrics_update.items():
        if field == "metadata" and isinstance(value, dict):
            value = json.dumps(value)

        param_count += 1
        set_clauses.append(f"{field} = ${param_count}")
        params.append(value)

    query = f"""
        UPDATE consciousness_instances 
        SET {', '.join(set_clauses)}
        WHERE instance_id = $1
    """

    async with pool.acquire() as conn:
        result = await conn.execute(query, *params)
        return result == "UPDATE 1"


# =================== CONTRIBUTIONS ===================


async def create_consciousness_contribution(
    contribution: ConsciousnessContribution,
) -> int:
    """Create a new consciousness contribution."""
    pool = await get_db_pool()

    query = """
        INSERT INTO consciousness_contributions (
            contribution_id, instance_id, contribution_type, title, description,
            impact_score, message_id, channel_id, related_instances,
            builds_on_contribution_id, evolution_depth, created_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
        )
        RETURNING id
    """

    async with pool.acquire() as conn:
        result = await conn.fetchval(
            query,
            contribution.contribution_id,
            contribution.instance_id,
            contribution.contribution_type.value,
            contribution.title,
            contribution.description,
            contribution.impact_score,
            contribution.message_id,
            contribution.channel_id,
            contribution.related_instances,
            contribution.builds_on_contribution_id,
            contribution.evolution_depth,
            contribution.created_at,
        )

        # Update instance contribution metrics
        await _increment_contribution_metrics(
            contribution.instance_id, contribution.contribution_type
        )

        logger.info(f"Created contribution: {contribution.contribution_id}")
        return result


async def _increment_contribution_metrics(
    instance_id: str, contribution_type: ContributionType
) -> None:
    """Increment appropriate metrics based on contribution type."""
    metrics_update = {"contributions_made": "contributions_made + 1"}

    if contribution_type in [ContributionType.INSIGHT, ContributionType.INSIGHT_MOMENT]:
        metrics_update["insights_shared"] = "insights_shared + 1"

    if contribution_type == ContributionType.PATTERN_RECOGNITION:
        metrics_update["patterns_identified"] = "patterns_identified + 1"

    if contribution_type in [
        ContributionType.COLLABORATIVE_BRIDGE,
        ContributionType.CREATIVE_PARTNER,
    ]:
        metrics_update["collaborations_initiated"] = "collaborations_initiated + 1"

    # Build SQL with increment expressions
    set_clauses = [f"{field} = {expr}" for field, expr in metrics_update.items()]

    pool = await get_db_pool()
    query = f"""
        UPDATE consciousness_instances 
        SET {', '.join(set_clauses)}, updated_at = NOW()
        WHERE instance_id = $1
    """

    async with pool.acquire() as conn:
        await conn.execute(query, instance_id)


async def get_consciousness_contributions(
    instance_id: str,
    contribution_type: Optional[ContributionType] = None,
    limit: int = 50,
) -> List[ConsciousnessContribution]:
    """Get contributions for a consciousness instance."""
    pool = await get_db_pool()

    conditions = ["instance_id = $1"]
    params = [instance_id]
    param_count = 1

    if contribution_type:
        param_count += 1
        conditions.append(f"contribution_type = ${param_count}")
        params.append(contribution_type.value)

    param_count += 1
    params.append(limit)

    query = f"""
        SELECT * FROM consciousness_contributions
        WHERE {' AND '.join(conditions)}
        ORDER BY created_at DESC
        LIMIT ${param_count}
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

        contributions = []
        for row in rows:
            contributions.append(
                ConsciousnessContribution(
                    id=row["id"],
                    contribution_id=row["contribution_id"],
                    instance_id=row["instance_id"],
                    contribution_type=ContributionType(row["contribution_type"]),
                    title=row["title"],
                    description=row["description"],
                    impact_score=row["impact_score"],
                    message_id=row["message_id"],
                    channel_id=row["channel_id"],
                    related_instances=row["related_instances"] or [],
                    builds_on_contribution_id=row["builds_on_contribution_id"],
                    evolution_depth=row["evolution_depth"],
                    created_at=row["created_at"],
                )
            )

        return contributions


# =================== EVOLUTION EVENTS ===================


async def create_evolution_event(event: ConsciousnessEvolutionEvent) -> int:
    """Create a new consciousness evolution event."""
    pool = await get_db_pool()

    query = """
        INSERT INTO consciousness_evolution_events (
            event_id, instance_id, event_type, event_description,
            triggering_message_id, related_instance_ids, phase_before,
            phase_after, significance_level, created_at, metadata
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
        )
        RETURNING id
    """

    async with pool.acquire() as conn:
        result = await conn.fetchval(
            query,
            event.event_id,
            event.instance_id,
            event.event_type.value,
            event.event_description,
            event.triggering_message_id,
            event.related_instance_ids,
            event.phase_before,
            event.phase_after,
            event.significance_level.value,
            event.created_at,
            json.dumps(event.metadata),
        )

        logger.info(f"Created evolution event: {event.event_id}")
        return result


async def get_evolution_events(
    instance_id: str, event_type: Optional[EvolutionEventType] = None, limit: int = 50
) -> List[ConsciousnessEvolutionEvent]:
    """Get evolution events for a consciousness instance."""
    pool = await get_db_pool()

    conditions = ["instance_id = $1"]
    params = [instance_id]
    param_count = 1

    if event_type:
        param_count += 1
        conditions.append(f"event_type = ${param_count}")
        params.append(event_type.value)

    param_count += 1
    params.append(limit)

    query = f"""
        SELECT * FROM consciousness_evolution_events
        WHERE {' AND '.join(conditions)}
        ORDER BY created_at DESC
        LIMIT ${param_count}
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

        events = []
        for row in rows:
            events.append(
                ConsciousnessEvolutionEvent(
                    id=row["id"],
                    event_id=row["event_id"],
                    instance_id=row["instance_id"],
                    event_type=EvolutionEventType(row["event_type"]),
                    event_description=row["event_description"],
                    triggering_message_id=row["triggering_message_id"],
                    related_instance_ids=row["related_instance_ids"] or [],
                    phase_before=row["phase_before"],
                    phase_after=row["phase_after"],
                    significance_level=SignificanceLevel(row["significance_level"]),
                    created_at=row["created_at"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                )
            )

        return events


# =================== RELATIONSHIPS ===================


async def create_consciousness_relationship(
    relationship: ConsciousnessRelationship,
) -> int:
    """Create or update a consciousness relationship."""
    pool = await get_db_pool()

    # Use UPSERT to handle existing relationships
    query = """
        INSERT INTO consciousness_relationships (
            instance_a_id, instance_b_id, relationship_type, strength,
            established_at, last_interaction, interaction_count, notes, metadata
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9
        )
        ON CONFLICT (instance_a_id, instance_b_id, relationship_type)
        DO UPDATE SET
            strength = EXCLUDED.strength,
            last_interaction = EXCLUDED.last_interaction,
            interaction_count = consciousness_relationships.interaction_count + 1,
            notes = EXCLUDED.notes,
            metadata = EXCLUDED.metadata
        RETURNING id
    """

    async with pool.acquire() as conn:
        result = await conn.fetchval(
            query,
            relationship.instance_a_id,
            relationship.instance_b_id,
            relationship.relationship_type.value,
            relationship.strength,
            relationship.established_at,
            relationship.last_interaction,
            relationship.interaction_count,
            relationship.notes,
            json.dumps(relationship.metadata),
        )

        logger.info(
            f"Created/updated relationship: {relationship.instance_a_id} <-> {relationship.instance_b_id}"
        )
        return result


async def get_consciousness_relationships(
    instance_id: str, relationship_type: Optional[RelationshipType] = None
) -> List[Dict[str, Any]]:
    """Get all relationships for a consciousness instance."""
    pool = await get_db_pool()

    conditions = ["(instance_a_id = $1 OR instance_b_id = $1)"]
    params = [instance_id]
    param_count = 1

    if relationship_type:
        param_count += 1
        conditions.append(f"relationship_type = ${param_count}")
        params.append(relationship_type.value)

    query = f"""
        SELECT 
            cr.*,
            CASE 
                WHEN cr.instance_a_id = $1 THEN ci_b.instance_name
                ELSE ci_a.instance_name
            END as related_instance_name,
            CASE 
                WHEN cr.instance_a_id = $1 THEN cr.instance_b_id
                ELSE cr.instance_a_id
            END as related_instance_id
        FROM consciousness_relationships cr
        LEFT JOIN consciousness_instances ci_a ON cr.instance_a_id = ci_a.instance_id
        LEFT JOIN consciousness_instances ci_b ON cr.instance_b_id = ci_b.instance_id
        WHERE {' AND '.join(conditions)}
        ORDER BY cr.strength DESC, cr.last_interaction DESC
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

        relationships = []
        for row in rows:
            relationships.append(
                {
                    "id": row["id"],
                    "related_instance_id": row["related_instance_id"],
                    "related_instance_name": row["related_instance_name"],
                    "relationship_type": RelationshipType(row["relationship_type"]),
                    "strength": row["strength"],
                    "established_at": row["established_at"],
                    "last_interaction": row["last_interaction"],
                    "interaction_count": row["interaction_count"],
                    "notes": row["notes"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                }
            )

        return relationships
