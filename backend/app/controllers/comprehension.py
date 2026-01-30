"""
Comprehension Controller

REST API endpoints for the Comprehension Engine.
Provides the HTTP interface for analyzing inputs, querying capabilities,
viewing history, and submitting feedback on comprehension accuracy.

This is the entry point for the CORE loop:
  POST /comprehension/analyze → ComprehensionResult → Task Router
"""

import logging
from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Depends, status
from pydantic import BaseModel, Field, field_validator

from app.models.comprehension_models import (
    ComprehensionInput,
    ComprehensionResult,
    ComprehensionFeedback,
    SystemCapability,
    SourceType,
    ComprehensionStatus,
)
from app.services.comprehension_service import comprehension_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/comprehension", tags=["comprehension"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class AnalyzeRequest(BaseModel):
    """Request model for analyzing input through the Comprehension Engine."""
    content: str = Field(..., description="Raw input text to analyze")
    source_type: str = Field(
        default="user",
        description="Source type: user, agent, system, tool_output"
    )
    source_id: Optional[str] = Field(None, description="ID of the source (user_id, agent_id, etc.)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    conversation_id: Optional[str] = Field(None, description="Conversation context if applicable")

    @field_validator('content')
    @classmethod
    def content_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()

    @field_validator('source_type')
    @classmethod
    def validate_source_type(cls, v: str) -> str:
        valid_types = ['user', 'agent', 'system', 'tool_output']
        if v not in valid_types:
            raise ValueError(f'source_type must be one of: {valid_types}')
        return v


class AnalyzeResponse(BaseModel):
    """Response model for comprehension analysis."""
    comprehension_id: str = Field(..., description="Unique comprehension result ID")
    intent_summary: str = Field(..., description="Summary of parsed intent")
    action_type: str = Field(..., description="Classified action type")
    urgency: str = Field(..., description="Assessed urgency level")
    confidence: float = Field(..., description="Overall comprehension confidence")
    suggested_task_type: str = Field(..., description="Suggested task type for routing")
    suggested_priority: int = Field(..., description="Suggested priority (1-10)")
    handling_mode: str = Field(..., description="single_agent, multi_agent, no_agent, or human_required")
    has_relevant_context: bool = Field(..., description="Whether relevant memory context was found")
    has_capable_agents: bool = Field(..., description="Whether capable agents were found")
    context_match_count: int = Field(default=0, description="Number of context matches found")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in ms")
    status: str = Field(..., description="Comprehension status")
    # Full result (included for programmatic consumers)
    full_result: ComprehensionResult = Field(..., description="Full comprehension result")


class FeedbackRequest(BaseModel):
    """Request model for submitting feedback on comprehension accuracy."""
    comprehension_id: UUID = Field(..., description="ID of the comprehension result to rate")
    score: float = Field(ge=0.0, le=1.0, description="Accuracy score (0=wrong, 1=perfect)")
    correct_action_type: Optional[str] = Field(None, description="What the action type should have been")
    correct_task_type: Optional[str] = Field(None, description="What the task type should have been")
    notes: Optional[str] = Field(None, description="Additional feedback notes")
    submitted_by: Optional[str] = Field(None, description="Who submitted the feedback")


class HistoryResponse(BaseModel):
    """Response model for comprehension history."""
    results: List[ComprehensionResult]
    total_count: int
    page: int
    page_size: int


class CapabilityListResponse(BaseModel):
    """Response model for system capabilities."""
    capabilities: List[SystemCapability]
    total: int


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================

async def get_comprehension_service():
    """Get the comprehension service instance."""
    if not comprehension_service._initialized:
        await comprehension_service.initialize()
    return comprehension_service


# =============================================================================
# CORE ENDPOINTS
# =============================================================================

@router.post("/analyze", status_code=status.HTTP_200_OK)
async def analyze_input(
    request: AnalyzeRequest,
    service=Depends(get_comprehension_service),
) -> AnalyzeResponse:
    """
    Analyze input through the Comprehension Engine.

    This is the primary endpoint — the entry point for the CORE loop.
    Takes raw input, parses intent, searches memory for context,
    matches capabilities, and returns a structured ComprehensionResult.

    The result can be passed directly to the Task Router for orchestration.
    """
    try:
        # Build ComprehensionInput from request
        comp_input = ComprehensionInput(
            content=request.content,
            source_type=SourceType(request.source_type),
            source_id=request.source_id,
            metadata=request.metadata,
            conversation_id=request.conversation_id,
        )

        # Run comprehension
        result = await service.comprehend(comp_input)

        # Build response
        return AnalyzeResponse(
            comprehension_id=str(result.id),
            intent_summary=result.intent.summary,
            action_type=result.intent.action_type.value,
            urgency=result.intent.urgency.value,
            confidence=result.confidence,
            suggested_task_type=result.suggested_task_type,
            suggested_priority=result.suggested_priority,
            handling_mode=result.handling_mode.value,
            has_relevant_context=result.context.has_relevant_context,
            has_capable_agents=result.capabilities.has_capable_agents,
            context_match_count=result.context.total_matches,
            processing_time_ms=result.processing_time_ms,
            status=result.status.value,
            full_result=result,
        )

    except Exception as e:
        logger.error(f"Comprehension analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comprehension analysis failed: {str(e)}",
        )


@router.get("/capabilities", status_code=status.HTTP_200_OK)
async def list_capabilities(
    service=Depends(get_comprehension_service),
) -> CapabilityListResponse:
    """
    List all system capabilities.

    Returns agents, tools, and services registered in the system
    and what they can do. Useful for understanding system coverage.
    """
    try:
        capabilities = await service.get_capability_registry()
        return CapabilityListResponse(
            capabilities=capabilities,
            total=len(capabilities),
        )

    except Exception as e:
        logger.error(f"Failed to list capabilities: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list capabilities: {str(e)}",
        )


@router.get("/history", status_code=status.HTTP_200_OK)
async def get_comprehension_history(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Results per page"),
    source_type: Optional[str] = Query(None, description="Filter by source type"),
    action_type: Optional[str] = Query(None, description="Filter by action type"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence"),
    conversation_id: Optional[str] = Query(None, description="Filter by conversation"),
    service=Depends(get_comprehension_service),
) -> HistoryResponse:
    """
    Get past comprehension results with filtering and pagination.

    Useful for analytics, debugging, and understanding how the
    Comprehension Engine is performing over time.
    """
    try:
        from app.repository import comprehension_repository

        offset = (page - 1) * page_size
        results, total_count = await comprehension_repository.list_comprehension_results(
            limit=page_size,
            offset=offset,
            source_type=source_type,
            action_type=action_type,
            status=status_filter,
            min_confidence=min_confidence,
            conversation_id=conversation_id,
        )

        return HistoryResponse(
            results=results,
            total_count=total_count,
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        logger.error(f"Failed to get comprehension history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get comprehension history: {str(e)}",
        )


@router.post("/feedback", status_code=status.HTTP_200_OK)
async def submit_feedback(
    request: FeedbackRequest,
    service=Depends(get_comprehension_service),
) -> Dict[str, Any]:
    """
    Submit feedback on comprehension accuracy.

    This feeds into the learning loop — over time, the system can track
    which types of inputs it comprehends well and which it struggles with.
    """
    try:
        feedback = ComprehensionFeedback(
            comprehension_id=request.comprehension_id,
            score=request.score,
            correct_action_type=request.correct_action_type,
            correct_task_type=request.correct_task_type,
            notes=request.notes,
            submitted_by=request.submitted_by,
        )

        success = await service.submit_feedback(feedback)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Comprehension result {request.comprehension_id} not found",
            )

        return {
            "comprehension_id": str(request.comprehension_id),
            "score": request.score,
            "message": "Feedback recorded successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit feedback: {str(e)}",
        )


# =============================================================================
# ANALYTICS ENDPOINTS
# =============================================================================

@router.get("/analytics", status_code=status.HTTP_200_OK)
async def get_analytics(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    service=Depends(get_comprehension_service),
) -> Dict[str, Any]:
    """
    Get comprehension analytics over a time window.

    Returns accuracy metrics, action type distribution, processing times,
    and trends for the dashboard.
    """
    try:
        analytics = await service.get_analytics(days=days)
        return analytics

    except Exception as e:
        logger.error(f"Failed to get comprehension analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics: {str(e)}",
        )


@router.get("/analytics/accuracy", status_code=status.HTTP_200_OK)
async def get_accuracy_trend(
    days: int = Query(30, ge=1, le=365, description="Number of days"),
    service=Depends(get_comprehension_service),
) -> List[Dict[str, Any]]:
    """
    Get comprehension accuracy over time for trend analysis.

    Returns daily buckets with average confidence and feedback scores.
    """
    try:
        from app.repository import comprehension_repository

        trend = await comprehension_repository.get_accuracy_over_time(days=days)
        return trend

    except Exception as e:
        logger.error(f"Failed to get accuracy trend: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get accuracy trend: {str(e)}",
        )


# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================

@router.get("/health", status_code=status.HTTP_200_OK)
async def comprehension_health(
    service=Depends(get_comprehension_service),
) -> Dict[str, Any]:
    """Check comprehension service health."""
    try:
        health = await service.health_check()
        overall_healthy = health.get("initialized", False)

        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "details": health,
        }

    except Exception as e:
        logger.error(f"Comprehension health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@router.post("/complexity", status_code=status.HTTP_200_OK)
async def analyze_complexity(
    content: str = Query(..., description="Text to analyze for complexity"),
    service=Depends(get_comprehension_service),
) -> Dict[str, Any]:
    """
    Analyze text complexity without full comprehension.

    Standalone endpoint for quick complexity estimation.
    Useful for the Task Router to pre-assess incoming tasks.
    """
    try:
        score = await service.analyze_complexity(content)
        return score.model_dump()

    except Exception as e:
        logger.error(f"Complexity analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Complexity analysis failed: {str(e)}",
        )
