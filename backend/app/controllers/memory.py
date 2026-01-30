"""
Memory Controller

REST API endpoints for the three-tier memory system.
Provides access to semantic, episodic, and procedural memory operations.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from uuid import UUID

from fastapi import APIRouter, HTTPException, status, Query, Depends
from pydantic import BaseModel, Field, validator

from app.services.memory_service import memory_service, MemoryContext, Procedure, MemoryStats

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/memory", tags=["memory"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class StoreKnowledgeRequest(BaseModel):
    """Request to store semantic knowledge."""
    content: str = Field(..., description="Knowledge content to store")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    source_agent_id: Optional[str] = Field(None, description="ID of agent storing the knowledge")
    
    @validator('content')
    def content_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()


class StoreExperienceRequest(BaseModel):
    """Request to store agent experience."""
    agent_id: str = Field(..., description="Agent ID for the experience")
    content: str = Field(..., description="Experience content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Experience metadata")
    
    @validator('content')
    def content_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()


class StoreProcedureRequest(BaseModel):
    """Request to store a procedure."""
    role: str = Field(..., description="Role this procedure applies to")
    procedure_name: str = Field(..., description="Name of the procedure")
    steps: List[str] = Field(..., description="Procedure steps")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Procedure metadata")
    
    @validator('steps')
    def steps_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('Steps cannot be empty')
        return [step.strip() for step in v if step.strip()]


class UpdateProcedureOutcomeRequest(BaseModel):
    """Request to update procedure success rate."""
    procedure_id: UUID = Field(..., description="Procedure ID")
    success: bool = Field(..., description="Whether the procedure execution was successful")


class MemoryResponse(BaseModel):
    """Generic memory response."""
    id: UUID
    content: str
    confidence: float
    access_count: int
    created_at: str
    similarity: Optional[float] = None


class SemanticMemoryResponse(MemoryResponse):
    """Semantic memory response."""
    source_agent_id: Optional[str] = None
    metadata: Dict[str, Any]


class EpisodicMemoryResponse(MemoryResponse):
    """Episodic memory response."""
    agent_id: str
    memory_type: str
    importance: float
    consolidated: bool
    metadata: Dict[str, Any]


class ProceduralMemoryResponse(MemoryResponse):
    """Procedural memory response."""
    role: str
    procedure_name: str
    steps: List[str]
    success_rate: float
    usage_count: int
    metadata: Dict[str, Any]


class MemorySearchResponse(BaseModel):
    """Memory search results."""
    query: str
    results: List[MemoryResponse]
    total: int
    threshold: float


class ContextResponse(BaseModel):
    """Memory context response."""
    query: str
    semantic: List[str]
    episodic: List[str]
    procedural: List[str]
    total_items: int
    relevance_threshold: float


class BulkImportRequest(BaseModel):
    """Request to import memories in bulk."""
    memories: List[Dict[str, Any]] = Field(..., description="List of memories to import")


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================

async def get_memory_service():
    """Get the memory service instance."""
    if not memory_service._initialized:
        await memory_service.initialize()
    return memory_service


# =============================================================================
# SEMANTIC MEMORY ENDPOINTS
# =============================================================================

@router.post("/knowledge", status_code=status.HTTP_201_CREATED)
async def store_knowledge(
    request: StoreKnowledgeRequest,
    service = Depends(get_memory_service)
) -> Dict[str, Any]:
    """Store shared knowledge accessible to all agents."""
    try:
        memory_id = await service.store_knowledge(
            content=request.content,
            metadata=request.metadata,
            source_agent_id=request.source_agent_id
        )
        
        return {
            "memory_id": memory_id,
            "type": "semantic",
            "message": "Knowledge stored successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to store knowledge: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store knowledge: {str(e)}"
        )


@router.get("/knowledge/search")
async def search_knowledge(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    threshold: float = Query(0.7, ge=0.0, le=1.0, description="Similarity threshold"),
    service = Depends(get_memory_service)
) -> MemorySearchResponse:
    """Search shared knowledge by semantic similarity."""
    try:
        memories = await service.search_knowledge(
            query=q,
            limit=limit,
            threshold=threshold
        )
        
        results = []
        for memory in memories:
            results.append(SemanticMemoryResponse(
                id=memory.id,
                content=memory.content,
                confidence=memory.confidence,
                access_count=memory.access_count,
                created_at=memory.created_at.isoformat(),
                source_agent_id=memory.source_agent_id,
                metadata=memory.metadata,
                similarity=memory.metadata.get('similarity')
            ))
        
        return MemorySearchResponse(
            query=q,
            results=results,
            total=len(results),
            threshold=threshold
        )
        
    except Exception as e:
        logger.error(f"Failed to search knowledge: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search knowledge: {str(e)}"
        )


# =============================================================================
# EPISODIC MEMORY ENDPOINTS
# =============================================================================

@router.post("/experience", status_code=status.HTTP_201_CREATED)
async def store_experience(
    request: StoreExperienceRequest,
    service = Depends(get_memory_service)
) -> Dict[str, Any]:
    """Store agent experience."""
    try:
        memory_id = await service.store_experience(
            agent_id=request.agent_id,
            content=request.content,
            metadata=request.metadata
        )
        
        return {
            "memory_id": memory_id,
            "type": "episodic",
            "agent_id": request.agent_id,
            "message": "Experience stored successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to store experience: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store experience: {str(e)}"
        )


@router.get("/experience/{agent_id}")
async def get_agent_experiences(
    agent_id: str,
    q: Optional[str] = Query(None, description="Optional search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    memory_type: Optional[str] = Query(None, description="Filter by memory type"),
    service = Depends(get_memory_service)
) -> List[EpisodicMemoryResponse]:
    """Get agent experiences."""
    try:
        memories = await service.get_agent_experiences(
            agent_id=agent_id,
            query=q,
            limit=limit,
            memory_type=memory_type
        )
        
        results = []
        for memory in memories:
            results.append(EpisodicMemoryResponse(
                id=memory.id,
                content=memory.content,
                confidence=memory.confidence,
                access_count=memory.access_count,
                created_at=memory.created_at.isoformat(),
                agent_id=memory.agent_id,
                memory_type=memory.memory_type,
                importance=memory.importance,
                consolidated=memory.consolidated,
                metadata=memory.metadata,
                similarity=memory.metadata.get('similarity')
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to get experiences for agent {agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get experiences: {str(e)}"
        )


@router.post("/experience/{agent_id}/consolidate")
async def consolidate_agent_experiences(
    agent_id: str,
    service = Depends(get_memory_service)
) -> Dict[str, Any]:
    """Consolidate short-term experiences to long-term for an agent."""
    try:
        count = await service.consolidate_experiences(agent_id)
        
        return {
            "agent_id": agent_id,
            "consolidated_count": count,
            "message": f"Consolidated {count} experiences"
        }
        
    except Exception as e:
        logger.error(f"Failed to consolidate experiences for agent {agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to consolidate experiences: {str(e)}"
        )


# =============================================================================
# PROCEDURAL MEMORY ENDPOINTS
# =============================================================================

@router.post("/procedure", status_code=status.HTTP_201_CREATED)
async def store_procedure(
    request: StoreProcedureRequest,
    service = Depends(get_memory_service)
) -> Dict[str, Any]:
    """Store a learned procedure."""
    try:
        memory_id = await service.store_procedure(
            role=request.role,
            procedure_name=request.procedure_name,
            steps=request.steps,
            metadata=request.metadata
        )
        
        return {
            "memory_id": memory_id,
            "type": "procedural",
            "role": request.role,
            "procedure_name": request.procedure_name,
            "message": "Procedure stored successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to store procedure: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store procedure: {str(e)}"
        )


@router.get("/procedures/{role}")
async def get_role_procedures(
    role: str,
    service = Depends(get_memory_service)
) -> List[Procedure]:
    """Get all procedures for a role."""
    try:
        procedures = await service.get_role_procedures(role)
        return procedures
        
    except Exception as e:
        logger.error(f"Failed to get procedures for role {role}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get procedures: {str(e)}"
        )


@router.get("/procedures/search")
async def search_procedures(
    q: str = Query(..., description="Search query"),
    role: Optional[str] = Query(None, description="Filter by role"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    service = Depends(get_memory_service)
) -> List[Procedure]:
    """Search procedures by semantic similarity."""
    try:
        procedures = await service.search_procedures(
            query=q,
            role=role
        )
        
        return procedures[:limit]  # Apply limit
        
    except Exception as e:
        logger.error(f"Failed to search procedures: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search procedures: {str(e)}"
        )


@router.patch("/procedure/outcome")
async def update_procedure_outcome(
    request: UpdateProcedureOutcomeRequest,
    service = Depends(get_memory_service)
) -> Dict[str, Any]:
    """Update procedure success rate based on execution outcome."""
    try:
        success = await service.update_procedure_outcome(
            procedure_id=request.procedure_id,
            success=request.success
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Procedure not found"
            )
        
        return {
            "procedure_id": request.procedure_id,
            "success": request.success,
            "message": "Procedure outcome updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update procedure outcome: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update procedure outcome: {str(e)}"
        )


# =============================================================================
# CROSS-TIER ENDPOINTS
# =============================================================================

@router.get("/context")
async def get_relevant_context(
    q: str = Query(..., description="Query for retrieving relevant context"),
    agent_id: Optional[str] = Query(None, description="Agent ID for personalized context"),
    limit: int = Query(5, ge=1, le=20, description="Items per memory tier"),
    threshold: float = Query(0.7, ge=0.0, le=1.0, description="Relevance threshold"),
    service = Depends(get_memory_service)
) -> ContextResponse:
    """Get relevant context from all memory tiers."""
    try:
        context = await service.get_relevant_context(
            query=q,
            agent_id=agent_id,
            limit_per_tier=limit,
            relevance_threshold=threshold
        )
        
        return ContextResponse(
            query=q,
            semantic=context.semantic,
            episodic=context.episodic,
            procedural=context.procedural,
            total_items=context.total_items,
            relevance_threshold=context.relevance_threshold
        )
        
    except Exception as e:
        logger.error(f"Failed to get relevant context: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get relevant context: {str(e)}"
        )


# =============================================================================
# MEMORY MANAGEMENT ENDPOINTS
# =============================================================================

@router.get("/stats/{agent_id}")
async def get_memory_statistics(
    agent_id: str,
    service = Depends(get_memory_service)
) -> MemoryStats:
    """Get memory statistics for an agent."""
    try:
        stats = await service.get_agent_memory_summary(agent_id)
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get memory stats for agent {agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get memory statistics: {str(e)}"
        )


@router.delete("/agent/{agent_id}")
async def clear_agent_memories(
    agent_id: str,
    tier: Optional[str] = Query(None, description="Memory tier to clear (semantic/episodic/procedural)"),
    service = Depends(get_memory_service)
) -> Dict[str, Any]:
    """Clear memories for an agent."""
    try:
        if tier and tier not in ["semantic", "episodic", "procedural"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid tier. Must be one of: semantic, episodic, procedural"
            )
        
        counts = await service.clear_agent_memories(agent_id, tier)
        total_cleared = sum(counts.values())
        
        return {
            "agent_id": agent_id,
            "tier": tier or "all",
            "counts": counts,
            "total_cleared": total_cleared,
            "message": f"Cleared {total_cleared} memories"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear memories for agent {agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear memories: {str(e)}"
        )


@router.post("/cleanup")
async def cleanup_expired_memories(
    service = Depends(get_memory_service)
) -> Dict[str, Any]:
    """Clean up expired memories (admin operation)."""
    try:
        result = await service.cleanup_expired_memories()
        
        return {
            "operation": "cleanup",
            "result": result,
            "message": "Memory cleanup completed"
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup memories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup memories: {str(e)}"
        )


@router.post("/bulk-import", status_code=status.HTTP_201_CREATED)
async def bulk_import_memories(
    request: BulkImportRequest,
    service = Depends(get_memory_service)
) -> Dict[str, Any]:
    """Import memories in bulk."""
    try:
        if not request.memories:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No memories provided for import"
            )
        
        counts = await service.bulk_import_memories(request.memories)
        
        return {
            "operation": "bulk_import",
            "total_provided": len(request.memories),
            "counts": counts,
            "message": f"Imported memories: {counts}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to bulk import memories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to import memories: {str(e)}"
        )


@router.get("/health")
async def memory_health_check(
    service = Depends(get_memory_service)
) -> Dict[str, Any]:
    """Check memory service health."""
    try:
        health = await service.health_check()
        
        # Determine overall status
        overall_healthy = (
            health.get("initialized", False) and
            health.get("embedding_service", False) and
            health.get("database", False)
        )
        
        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "details": health
        }
        
    except Exception as e:
        logger.error(f"Memory health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }