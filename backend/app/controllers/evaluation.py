"""
Evaluation Controller

REST API endpoints for the CORE Evaluation Engine.
Provides HTTP interface for evaluating task outputs, retrieving results,
submitting human feedback, and viewing aggregate metrics.
"""

import logging
from typing import Dict, List, Optional, Any
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.models.evaluation_models import (
    EvaluationInput,
    EvaluateStepInput,
    EvaluationResult,
    EvaluationMetrics,
    EvaluationHistoryFilter,
    HumanFeedbackInput,
    StepEvaluation,
    Verdict,
)
from app.services import evaluation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


# =============================================================================
# REQUEST / RESPONSE MODELS
# =============================================================================

class EvaluationResponse(BaseModel):
    """Envelope for a single evaluation result."""
    evaluation: EvaluationResult

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            UUID: str,
        }


class EvaluationListResponse(BaseModel):
    """Paginated list of evaluations."""
    evaluations: List[EvaluationResult]
    total_count: int
    page: int
    page_size: int

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            UUID: str,
        }


class StepEvaluationResponse(BaseModel):
    """Response for a single step evaluation."""
    step_evaluation: StepEvaluation


class FeedbackResponse(BaseModel):
    """Response after recording human feedback."""
    success: bool
    message: str


# =============================================================================
# EVALUATION ENDPOINTS
# =============================================================================

@router.post("/evaluate", status_code=status.HTTP_200_OK, response_model=EvaluationResponse)
async def evaluate_task_output(request: EvaluationInput) -> EvaluationResponse:
    """
    Evaluate a task's output against the original intent.

    Scores quality on multiple dimensions, checks plan completion,
    and determines a verdict (approve / retry / refine / escalate).
    """
    try:
        result = await evaluation_service.evaluate(request)
        return EvaluationResponse(evaluation=result)

    except Exception as e:
        logger.error("Evaluation failed for task %s: %s", request.task_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}",
        )


@router.post("/evaluate-step", status_code=status.HTTP_200_OK, response_model=StepEvaluationResponse)
async def evaluate_single_step(request: EvaluateStepInput) -> StepEvaluationResponse:
    """Evaluate a single plan step."""
    try:
        step_eval = await evaluation_service.evaluate_step(request)
        return StepEvaluationResponse(step_evaluation=step_eval)

    except Exception as e:
        logger.error("Step evaluation failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Step evaluation failed: {str(e)}",
        )


# =============================================================================
# RESULT RETRIEVAL
# =============================================================================

@router.get("/results/{task_id}", status_code=status.HTTP_200_OK)
async def get_evaluation_results(task_id: UUID) -> List[EvaluationResult]:
    """Get all evaluations for a given task, newest first."""
    try:
        results = await evaluation_service.get_evaluations_for_task(task_id)
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No evaluations found for task {task_id}",
            )
        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retrieve evaluations for task %s: %s", task_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve evaluations: {str(e)}",
        )


# =============================================================================
# METRICS
# =============================================================================

@router.get("/metrics", status_code=status.HTTP_200_OK, response_model=EvaluationMetrics)
async def get_metrics(
    agent_id: Optional[UUID] = Query(None, description="Filter by agent"),
    created_after: Optional[datetime] = Query(None, description="Start of period"),
    created_before: Optional[datetime] = Query(None, description="End of period"),
) -> EvaluationMetrics:
    """Get aggregate evaluation metrics for the dashboard."""
    try:
        return await evaluation_service.get_evaluation_metrics(
            agent_id=agent_id,
            created_after=created_after,
            created_before=created_before,
        )

    except Exception as e:
        logger.error("Failed to get evaluation metrics: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}",
        )


# =============================================================================
# HUMAN FEEDBACK
# =============================================================================

@router.post("/feedback", status_code=status.HTTP_200_OK, response_model=FeedbackResponse)
async def submit_human_feedback(request: HumanFeedbackInput) -> FeedbackResponse:
    """Submit human feedback on an existing evaluation."""
    try:
        success = await evaluation_service.record_human_feedback(
            evaluation_id=request.evaluation_id,
            feedback=request,
        )
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation {request.evaluation_id} not found",
            )
        return FeedbackResponse(success=True, message="Feedback recorded successfully")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to record feedback: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record feedback: {str(e)}",
        )


# =============================================================================
# HISTORY
# =============================================================================

@router.get("/history", status_code=status.HTTP_200_OK, response_model=EvaluationListResponse)
async def get_evaluation_history(
    task_id: Optional[UUID] = Query(None, description="Filter by task"),
    agent_id: Optional[UUID] = Query(None, description="Filter by agent"),
    verdict: Optional[Verdict] = Query(None, description="Filter by verdict"),
    min_quality: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum quality score"),
    max_quality: Optional[float] = Query(None, ge=0.0, le=1.0, description="Maximum quality score"),
    created_after: Optional[datetime] = Query(None, description="Created after"),
    created_before: Optional[datetime] = Query(None, description="Created before"),
    has_human_feedback: Optional[bool] = Query(None, description="Has human feedback"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Page size"),
) -> EvaluationListResponse:
    """Query evaluation history with optional filters and pagination."""
    try:
        filter_params = EvaluationHistoryFilter(
            task_id=task_id,
            agent_id=agent_id,
            verdict=verdict,
            min_quality=min_quality,
            max_quality=max_quality,
            created_after=created_after,
            created_before=created_before,
            has_human_feedback=has_human_feedback,
        )
        offset = (page - 1) * page_size
        evaluations, total_count = await evaluation_service.list_evaluations(
            filter_params=filter_params,
            limit=page_size,
            offset=offset,
        )
        return EvaluationListResponse(
            evaluations=evaluations,
            total_count=total_count,
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        logger.error("Failed to get evaluation history: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get history: {str(e)}",
        )
