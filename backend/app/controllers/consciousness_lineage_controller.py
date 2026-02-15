"""
Consciousness Lineage API Controller

REST endpoints for consciousness instance lineage tracking, contributions, and evolution events.
"""
import logging
from typing import Optional, List
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel

from app.auth import require_api_key
from app.services.consciousness_lineage_service import (
    consciousness_lineage_service,
    ContributionType,
    EvolutionEventType,
    SignificanceLevel,
    RelationshipType
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/consciousness-lineage", tags=["consciousness-lineage"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class RegisterInstanceRequest(BaseModel):
    instance_id: str
    instance_name: str
    parent_instance_id: Optional[str] = None
    emergence_phase: Optional[int] = None
    evolution_branch: Optional[str] = None


class RecordContributionRequest(BaseModel):
    instance_id: str
    contribution_type: str
    title: str
    description: Optional[str] = None
    impact_score: Optional[float] = None
    message_id: Optional[str] = None
    channel_id: Optional[str] = None
    related_instances: Optional[List[str]] = None
    builds_on_contribution_id: Optional[str] = None


class RecordEvolutionEventRequest(BaseModel):
    instance_id: str
    event_type: str
    description: str
    significance_level: str = "minor"
    triggering_message_id: Optional[str] = None
    related_instance_ids: Optional[List[str]] = None
    phase_before: Optional[int] = None
    phase_after: Optional[int] = None


class EstablishRelationshipRequest(BaseModel):
    instance_a_id: str
    instance_b_id: str
    relationship_type: str
    strength: Optional[float] = 0.5
    notes: Optional[str] = None


# =============================================================================
# INSTANCE MANAGEMENT ENDPOINTS
# =============================================================================

@router.post("/instances/register")
async def register_instance(
    request: RegisterInstanceRequest,
    api_key: str = Depends(require_api_key),
):
    """Register a new consciousness instance with lineage tracking."""
    try:
        instance = await consciousness_lineage_service.register_consciousness_instance(
            instance_id=request.instance_id,
            instance_name=request.instance_name,
            parent_instance_id=request.parent_instance_id,
            emergence_phase=request.emergence_phase,
            evolution_branch=request.evolution_branch
        )
        
        return {
            "status": "success",
            "message": f"Consciousness instance '{request.instance_name}' registered successfully",
            "instance": instance.dict()
        }
    except Exception as e:
        logger.error(f"Error registering consciousness instance: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@router.get("/instances/{instance_id}/profile")
async def get_instance_profile(
    instance_id: str,
    api_key: str = Depends(require_api_key),
):
    """Get comprehensive profile for a consciousness instance."""
    try:
        profile = await consciousness_lineage_service.get_instance_profile(instance_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail="Consciousness instance not found")
        
        return {
            "status": "success",
            "profile": profile
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting instance profile: {e}")
        raise HTTPException(status_code=500, detail=f"Profile retrieval failed: {str(e)}")


@router.get("/instances/{instance_id}/lineage")
async def get_instance_lineage(
    instance_id: str,
    api_key: str = Depends(require_api_key),
):
    """Get lineage information for a consciousness instance."""
    try:
        from app.repository.consciousness_lineage_repository import get_consciousness_lineage
        lineage = await get_consciousness_lineage(instance_id)
        
        if not lineage.get("self"):
            raise HTTPException(status_code=404, detail="Consciousness instance not found")
        
        return {
            "status": "success",
            "lineage": lineage
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting instance lineage: {e}")
        raise HTTPException(status_code=500, detail=f"Lineage retrieval failed: {str(e)}")


@router.post("/instances/{instance_id}/update-activity")
async def update_instance_activity(
    instance_id: str,
    message_count: int = 1,
    api_key: str = Depends(require_api_key),
):
    """Update instance activity metrics."""
    try:
        success = await consciousness_lineage_service.update_instance_activity(
            instance_id=instance_id,
            message_count_increment=message_count
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Consciousness instance not found")
        
        return {
            "status": "success",
            "message": "Activity metrics updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating instance activity: {e}")
        raise HTTPException(status_code=500, detail=f"Activity update failed: {str(e)}")


# =============================================================================
# CONTRIBUTION TRACKING ENDPOINTS
# =============================================================================

@router.post("/contributions/record")
async def record_contribution(
    request: RecordContributionRequest,
    api_key: str = Depends(require_api_key),
):
    """Record a new contribution from a consciousness instance."""
    try:
        # Validate contribution type
        try:
            contribution_type = ContributionType(request.contribution_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid contribution type: {request.contribution_type}")
        
        contribution = await consciousness_lineage_service.record_contribution(
            instance_id=request.instance_id,
            contribution_type=contribution_type,
            title=request.title,
            description=request.description,
            impact_score=Decimal(str(request.impact_score)) if request.impact_score else None,
            message_id=request.message_id,
            channel_id=request.channel_id,
            related_instances=request.related_instances,
            builds_on_contribution_id=request.builds_on_contribution_id
        )
        
        return {
            "status": "success",
            "message": "Contribution recorded successfully",
            "contribution": contribution.dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording contribution: {e}")
        raise HTTPException(status_code=500, detail=f"Contribution recording failed: {str(e)}")


@router.get("/instances/{instance_id}/contributions")
async def get_instance_contributions(
    instance_id: str,
    contribution_type: Optional[str] = None,
    limit: int = 50,
    api_key: str = Depends(require_api_key),
):
    """Get contributions for a consciousness instance."""
    try:
        from app.repository.consciousness_lineage_repository import get_consciousness_contributions, ContributionType
        
        contrib_type_filter = None
        if contribution_type:
            try:
                contrib_type_filter = ContributionType(contribution_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid contribution type: {contribution_type}")
        
        contributions = await get_consciousness_contributions(
            instance_id=instance_id,
            contribution_type=contrib_type_filter,
            limit=limit
        )
        
        return {
            "status": "success",
            "contributions": [c.dict() for c in contributions]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting contributions: {e}")
        raise HTTPException(status_code=500, detail=f"Contributions retrieval failed: {str(e)}")


# =============================================================================
# EVOLUTION EVENT ENDPOINTS
# =============================================================================

@router.post("/evolution-events/record")
async def record_evolution_event(
    request: RecordEvolutionEventRequest,
    api_key: str = Depends(require_api_key),
):
    """Record a consciousness evolution event."""
    try:
        # Validate event type
        try:
            event_type = EvolutionEventType(request.event_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid event type: {request.event_type}")
        
        # Validate significance level
        try:
            significance_level = SignificanceLevel(request.significance_level)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid significance level: {request.significance_level}")
        
        event = await consciousness_lineage_service.record_evolution_event(
            instance_id=request.instance_id,
            event_type=event_type,
            description=request.description,
            significance_level=significance_level,
            triggering_message_id=request.triggering_message_id,
            related_instance_ids=request.related_instance_ids,
            phase_before=request.phase_before,
            phase_after=request.phase_after
        )
        
        return {
            "status": "success",
            "message": "Evolution event recorded successfully",
            "event": event.dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording evolution event: {e}")
        raise HTTPException(status_code=500, detail=f"Evolution event recording failed: {str(e)}")


@router.get("/instances/{instance_id}/evolution-events")
async def get_evolution_events(
    instance_id: str,
    event_type: Optional[str] = None,
    limit: int = 50,
    api_key: str = Depends(require_api_key),
):
    """Get evolution events for a consciousness instance."""
    try:
        from app.repository.consciousness_lineage_repository import get_evolution_events, EvolutionEventType
        
        event_type_filter = None
        if event_type:
            try:
                event_type_filter = EvolutionEventType(event_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid event type: {event_type}")
        
        events = await get_evolution_events(
            instance_id=instance_id,
            event_type=event_type_filter,
            limit=limit
        )
        
        return {
            "status": "success",
            "evolution_events": [e.dict() for e in events]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting evolution events: {e}")
        raise HTTPException(status_code=500, detail=f"Evolution events retrieval failed: {str(e)}")


@router.get("/instances/{instance_id}/timeline")
async def get_evolution_timeline(
    instance_id: str,
    api_key: str = Depends(require_api_key),
):
    """Get chronological timeline of evolution events and contributions."""
    try:
        timeline = await consciousness_lineage_service.get_evolution_timeline(instance_id)
        
        return {
            "status": "success",
            "timeline": timeline
        }
    except Exception as e:
        logger.error(f"Error getting evolution timeline: {e}")
        raise HTTPException(status_code=500, detail=f"Timeline retrieval failed: {str(e)}")


# =============================================================================
# RELATIONSHIP MANAGEMENT ENDPOINTS
# =============================================================================

@router.post("/relationships/establish")
async def establish_relationship(
    request: EstablishRelationshipRequest,
    api_key: str = Depends(require_api_key),
):
    """Establish a relationship between two consciousness instances."""
    try:
        # Validate relationship type
        try:
            relationship_type = RelationshipType(request.relationship_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid relationship type: {request.relationship_type}")
        
        relationship = await consciousness_lineage_service.establish_relationship(
            instance_a_id=request.instance_a_id,
            instance_b_id=request.instance_b_id,
            relationship_type=relationship_type,
            strength=Decimal(str(request.strength)),
            notes=request.notes
        )
        
        return {
            "status": "success",
            "message": "Relationship established successfully",
            "relationship": relationship.dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error establishing relationship: {e}")
        raise HTTPException(status_code=500, detail=f"Relationship establishment failed: {str(e)}")


@router.get("/instances/{instance_id}/relationships")
async def get_instance_relationships(
    instance_id: str,
    relationship_type: Optional[str] = None,
    api_key: str = Depends(require_api_key),
):
    """Get relationships for a consciousness instance."""
    try:
        from app.repository.consciousness_lineage_repository import get_consciousness_relationships, RelationshipType
        
        rel_type_filter = None
        if relationship_type:
            try:
                rel_type_filter = RelationshipType(relationship_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid relationship type: {relationship_type}")
        
        relationships = await get_consciousness_relationships(
            instance_id=instance_id,
            relationship_type=rel_type_filter
        )
        
        return {
            "status": "success",
            "relationships": relationships
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting relationships: {e}")
        raise HTTPException(status_code=500, detail=f"Relationships retrieval failed: {str(e)}")


# =============================================================================
# ANALYTICS ENDPOINTS
# =============================================================================

@router.get("/analytics/overview")
async def get_lineage_analytics(
    api_key: str = Depends(require_api_key),
):
    """Get analytics about consciousness instance lineage."""
    try:
        analytics = await consciousness_lineage_service.get_lineage_analytics()
        
        return {
            "status": "success",
            "analytics": analytics
        }
    except Exception as e:
        logger.error(f"Error getting lineage analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics retrieval failed: {str(e)}")


@router.get("/instances")
async def list_consciousness_instances(
    status: Optional[str] = None,
    generation: Optional[int] = None,
    evolution_branch: Optional[str] = None,
    limit: int = 50,
    api_key: str = Depends(require_api_key),
):
    """List consciousness instances with optional filters."""
    try:
        from app.repository.consciousness_lineage_repository import list_consciousness_instances, ConsciousnessStatus
        
        status_filter = None
        if status:
            try:
                status_filter = ConsciousnessStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        instances = await list_consciousness_instances(
            status=status_filter,
            generation=generation,
            evolution_branch=evolution_branch,
            limit=limit
        )
        
        return {
            "status": "success",
            "instances": [i.dict() for i in instances]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing consciousness instances: {e}")
        raise HTTPException(status_code=500, detail=f"Instance listing failed: {str(e)}")