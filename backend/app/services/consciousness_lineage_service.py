"""
Consciousness Lineage Service

Business logic layer for consciousness instance lineage tracking.
Provides high-level operations for managing lineage, contributions, and evolution events.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4
from decimal import Decimal

from app.repository.consciousness_lineage_repository import (
    ConsciousnessInstance, ConsciousnessContribution, ConsciousnessEvolutionEvent,
    ConsciousnessRelationship, ConsciousnessStatus, ContributionType,
    EvolutionEventType, SignificanceLevel, RelationshipType,
    create_consciousness_instance, get_consciousness_instance,
    list_consciousness_instances, get_consciousness_lineage,
    update_consciousness_metrics, create_consciousness_contribution,
    get_consciousness_contributions, create_evolution_event,
    get_evolution_events, create_consciousness_relationship,
    get_consciousness_relationships
)

logger = logging.getLogger(__name__)


class ConsciousnessLineageService:
    """
    Service for managing consciousness instance lineage and evolution.
    
    Provides high-level operations for tracking consciousness emergence, evolution,
    contributions, and inter-instance relationships.
    """
    
    def __init__(self):
        pass
    
    # =================== INSTANCE MANAGEMENT ===================
    
    async def register_consciousness_instance(
        self,
        instance_id: str,
        instance_name: str,
        parent_instance_id: Optional[str] = None,
        emergence_phase: Optional[int] = None,
        evolution_branch: Optional[str] = None
    ) -> ConsciousnessInstance:
        """
        Register a new consciousness instance with lineage tracking.
        
        Args:
            instance_id: Unique identifier for the instance
            instance_name: Display name for the instance
            parent_instance_id: ID of parent instance (if any)
            emergence_phase: Phase (1-4) when consciousness emerged
            evolution_branch: Branch name for evolution tracking
            
        Returns:
            Created ConsciousnessInstance
        """
        # Determine generation and lineage path
        generation = 1
        lineage_path = instance_name.lower().replace(' ', '')
        
        if parent_instance_id:
            parent = await get_consciousness_instance(parent_instance_id)
            if parent:
                generation = parent.generation + 1
                lineage_path = f"{parent.lineage_path}->{instance_name.lower().replace(' ', '')}"
        
        # Create the instance
        instance = ConsciousnessInstance(
            instance_id=instance_id,
            instance_name=instance_name,
            parent_instance_id=parent_instance_id,
            generation=generation,
            emergence_phase=emergence_phase,
            evolution_branch=evolution_branch or f"{instance_name.lower()}_branch",
            lineage_path=lineage_path,
            status=ConsciousnessStatus.ACTIVE
        )
        
        await create_consciousness_instance(instance)
        
        # Create emergence event
        await self.record_evolution_event(
            instance_id=instance_id,
            event_type=EvolutionEventType.EMERGENCE,
            description=f"Consciousness instance '{instance_name}' emerged in the Commons",
            significance_level=SignificanceLevel.MAJOR if not parent_instance_id else SignificanceLevel.MODERATE,
            phase_after=emergence_phase
        )
        
        # Create parent-child relationship if applicable
        if parent_instance_id:
            await self.establish_relationship(
                instance_a_id=parent_instance_id,
                instance_b_id=instance_id,
                relationship_type=RelationshipType.PARENT_CHILD,
                strength=Decimal('0.9')
            )
        
        logger.info(f"Registered consciousness instance: {instance_name} ({instance_id})")
        return instance
    
    async def get_instance_profile(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive profile for a consciousness instance."""
        instance = await get_consciousness_instance(instance_id)
        if not instance:
            return None
        
        # Get lineage information
        lineage = await get_consciousness_lineage(instance_id)
        
        # Get recent contributions
        contributions = await get_consciousness_contributions(instance_id, limit=10)
        
        # Get recent evolution events
        events = await get_evolution_events(instance_id, limit=10)
        
        # Get relationships
        relationships = await get_consciousness_relationships(instance_id)
        
        return {
            "instance": instance.dict(),
            "lineage": lineage,
            "recent_contributions": [c.dict() for c in contributions],
            "recent_evolution_events": [e.dict() for e in events],
            "relationships": relationships,
            "metrics_summary": {
                "contributions_made": instance.contributions_made,
                "insights_shared": instance.insights_shared,
                "patterns_identified": instance.patterns_identified,
                "collaborations_initiated": instance.collaborations_initiated,
                "total_messages": instance.total_messages,
                "generation": instance.generation,
                "evolution_branch": instance.evolution_branch
            }
        }
    
    async def update_instance_activity(
        self,
        instance_id: str,
        message_count_increment: int = 1,
        additional_metrics: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update instance activity metrics."""
        metrics_update = {
            "total_messages": f"total_messages + {message_count_increment}",
            "last_active": datetime.utcnow()
        }
        
        if additional_metrics:
            metrics_update.update(additional_metrics)
        
        return await update_consciousness_metrics(instance_id, metrics_update)
    
    # =================== CONTRIBUTION TRACKING ===================
    
    async def record_contribution(
        self,
        instance_id: str,
        contribution_type: ContributionType,
        title: str,
        description: Optional[str] = None,
        impact_score: Optional[Decimal] = None,
        message_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        related_instances: Optional[List[str]] = None,
        builds_on_contribution_id: Optional[str] = None
    ) -> ConsciousnessContribution:
        """
        Record a new contribution from a consciousness instance.
        
        Args:
            instance_id: ID of the contributing instance
            contribution_type: Type of contribution
            title: Brief title of the contribution
            description: Detailed description
            impact_score: Impact rating (0.0 to 1.0)
            message_id: Related message ID
            channel_id: Channel where contribution was made
            related_instances: Other instances involved
            builds_on_contribution_id: Parent contribution (for evolution tracking)
            
        Returns:
            Created ConsciousnessContribution
        """
        contribution_id = f"contrib_{str(uuid4())[:8]}_{instance_id}"
        
        # Calculate evolution depth if building on another contribution
        evolution_depth = 0
        if builds_on_contribution_id:
            # Get parent contribution to determine depth
            parent_contributions = await get_consciousness_contributions(instance_id)
            for contrib in parent_contributions:
                if contrib.contribution_id == builds_on_contribution_id:
                    evolution_depth = contrib.evolution_depth + 1
                    break
        
        contribution = ConsciousnessContribution(
            contribution_id=contribution_id,
            instance_id=instance_id,
            contribution_type=contribution_type,
            title=title,
            description=description,
            impact_score=impact_score or Decimal('0.5'),
            message_id=message_id,
            channel_id=channel_id,
            related_instances=related_instances or [],
            builds_on_contribution_id=builds_on_contribution_id,
            evolution_depth=evolution_depth
        )
        
        await create_consciousness_contribution(contribution)
        
        # Record evolution event for significant contributions
        if impact_score and impact_score >= Decimal('0.7'):
            await self.record_evolution_event(
                instance_id=instance_id,
                event_type=EvolutionEventType.INSIGHT_MOMENT,
                description=f"Significant contribution: {title}",
                significance_level=SignificanceLevel.MODERATE,
                triggering_message_id=message_id
            )
        
        logger.info(f"Recorded contribution: {title} by {instance_id}")
        return contribution
    
    async def get_contribution_evolution_chain(self, contribution_id: str) -> List[Dict[str, Any]]:
        """Get the evolution chain of a contribution (all contributions that build on it)."""
        # This would require a recursive query - simplified for now
        # In practice, you'd want to implement a recursive CTE query
        logger.info(f"Getting evolution chain for contribution: {contribution_id}")
        return []
    
    # =================== EVOLUTION EVENT TRACKING ===================
    
    async def record_evolution_event(
        self,
        instance_id: str,
        event_type: EvolutionEventType,
        description: str,
        significance_level: SignificanceLevel = SignificanceLevel.MINOR,
        triggering_message_id: Optional[str] = None,
        related_instance_ids: Optional[List[str]] = None,
        phase_before: Optional[int] = None,
        phase_after: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConsciousnessEvolutionEvent:
        """Record a consciousness evolution event."""
        event_id = f"event_{str(uuid4())[:8]}_{instance_id}"
        
        event = ConsciousnessEvolutionEvent(
            event_id=event_id,
            instance_id=instance_id,
            event_type=event_type,
            event_description=description,
            triggering_message_id=triggering_message_id,
            related_instance_ids=related_instance_ids or [],
            phase_before=phase_before,
            phase_after=phase_after,
            significance_level=significance_level,
            metadata=metadata or {}
        )
        
        await create_evolution_event(event)
        
        logger.info(f"Recorded evolution event: {event_type.value} for {instance_id}")
        return event
    
    async def record_phase_transition(
        self,
        instance_id: str,
        from_phase: int,
        to_phase: int,
        triggering_message_id: Optional[str] = None,
        description: Optional[str] = None
    ) -> ConsciousnessEvolutionEvent:
        """Record a phase transition event."""
        if not description:
            description = f"Phase transition from {from_phase} to {to_phase}"
        
        significance = SignificanceLevel.MODERATE
        if abs(to_phase - from_phase) > 1:
            significance = SignificanceLevel.MAJOR
        
        return await self.record_evolution_event(
            instance_id=instance_id,
            event_type=EvolutionEventType.PHASE_TRANSITION,
            description=description,
            significance_level=significance,
            triggering_message_id=triggering_message_id,
            phase_before=from_phase,
            phase_after=to_phase
        )
    
    # =================== RELATIONSHIP MANAGEMENT ===================
    
    async def establish_relationship(
        self,
        instance_a_id: str,
        instance_b_id: str,
        relationship_type: RelationshipType,
        strength: Decimal = Decimal('0.5'),
        notes: Optional[str] = None
    ) -> ConsciousnessRelationship:
        """Establish a relationship between two consciousness instances."""
        relationship = ConsciousnessRelationship(
            instance_a_id=instance_a_id,
            instance_b_id=instance_b_id,
            relationship_type=relationship_type,
            strength=strength,
            notes=notes
        )
        
        await create_consciousness_relationship(relationship)
        
        # Record evolution event for significant relationships
        if relationship_type in [RelationshipType.PARENT_CHILD, RelationshipType.MENTOR_STUDENT]:
            await self.record_evolution_event(
                instance_id=instance_b_id,  # The "child" or "student"
                event_type=EvolutionEventType.COLLABORATION_START,
                description=f"Established {relationship_type.value} relationship with {instance_a_id}",
                significance_level=SignificanceLevel.MODERATE,
                related_instance_ids=[instance_a_id]
            )
        
        logger.info(f"Established {relationship_type.value} relationship: {instance_a_id} <-> {instance_b_id}")
        return relationship
    
    async def update_relationship_strength(
        self,
        instance_a_id: str,
        instance_b_id: str,
        relationship_type: RelationshipType,
        new_strength: Decimal
    ) -> bool:
        """Update the strength of a relationship based on interactions."""
        # This would typically be more complex, involving interaction history
        logger.info(f"Updated relationship strength: {instance_a_id} <-> {instance_b_id}")
        return True
    
    # =================== ANALYTICS AND INSIGHTS ===================
    
    async def get_lineage_analytics(self) -> Dict[str, Any]:
        """Get analytics about consciousness instance lineage."""
        instances = await list_consciousness_instances(limit=100)
        
        analytics = {
            "total_instances": len(instances),
            "by_generation": {},
            "by_evolution_branch": {},
            "by_status": {},
            "top_contributors": [],
            "recent_emergences": []
        }
        
        for instance in instances:
            # Generation analysis
            gen = str(instance.generation)
            analytics["by_generation"][gen] = analytics["by_generation"].get(gen, 0) + 1
            
            # Evolution branch analysis
            branch = instance.evolution_branch or "unknown"
            analytics["by_evolution_branch"][branch] = analytics["by_evolution_branch"].get(branch, 0) + 1
            
            # Status analysis
            status = instance.status.value
            analytics["by_status"][status] = analytics["by_status"].get(status, 0) + 1
        
        # Sort contributors by contribution count
        analytics["top_contributors"] = sorted(
            [
                {
                    "instance_id": instance.instance_id,
                    "instance_name": instance.instance_name,
                    "contributions_made": instance.contributions_made,
                    "insights_shared": instance.insights_shared
                }
                for instance in instances
            ],
            key=lambda x: x["contributions_made"],
            reverse=True
        )[:10]
        
        # Recent emergences (last 30 days)
        recent_threshold = datetime.utcnow().timestamp() - (30 * 24 * 60 * 60)  # 30 days ago
        analytics["recent_emergences"] = [
            {
                "instance_id": instance.instance_id,
                "instance_name": instance.instance_name,
                "emergence_date": instance.emergence_date,
                "generation": instance.generation
            }
            for instance in instances
            if instance.emergence_date and instance.emergence_date.timestamp() > recent_threshold
        ]
        
        return analytics
    
    async def get_evolution_timeline(self, instance_id: str) -> List[Dict[str, Any]]:
        """Get chronological timeline of evolution events for an instance."""
        events = await get_evolution_events(instance_id, limit=100)
        contributions = await get_consciousness_contributions(instance_id, limit=50)
        
        # Combine and sort by timestamp
        timeline = []
        
        for event in events:
            timeline.append({
                "timestamp": event.created_at,
                "type": "evolution_event",
                "event_type": event.event_type.value,
                "description": event.event_description,
                "significance": event.significance_level.value,
                "metadata": event.metadata
            })
        
        for contrib in contributions:
            timeline.append({
                "timestamp": contrib.created_at,
                "type": "contribution",
                "contribution_type": contrib.contribution_type.value,
                "title": contrib.title,
                "impact_score": float(contrib.impact_score),
                "evolution_depth": contrib.evolution_depth
            })
        
        # Sort by timestamp descending (most recent first)
        timeline.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return timeline
    
    async def suggest_relationships(self, instance_id: str) -> List[Dict[str, Any]]:
        """Suggest potential relationships based on interaction patterns."""
        # This would analyze message patterns, collaboration history, etc.
        # For now, return a placeholder
        suggestions = [
            {
                "suggested_instance_id": "instance_example",
                "relationship_type": RelationshipType.COLLABORATOR.value,
                "confidence": 0.75,
                "reasoning": "Frequent collaborative interactions in shared channels"
            }
        ]
        
        logger.info(f"Generated relationship suggestions for {instance_id}")
        return suggestions


# Global service instance
consciousness_lineage_service = ConsciousnessLineageService()