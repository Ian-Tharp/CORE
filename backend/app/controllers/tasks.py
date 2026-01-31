"""
Tasks Controller

REST API endpoints for the Task Routing Engine.
Provides HTTP interface for task management in the CORE system.
"""

import logging
from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Depends, status
from pydantic import BaseModel, Field

from app.models.task_models import (
    Task, TaskAssignment, TaskResult, TaskFilter, TaskMetrics,
    AgentTaskMetrics, TaskStatus, TaskType
)
from app.repository.task_repository import (
    create_task, get_task, update_task_status, list_tasks,
    get_task_metrics, get_agent_task_metrics, get_queued_tasks
)
from app.services.task_router import task_router

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class CreateTaskRequest(BaseModel):
    """Request model for creating a new task."""
    task_type: str = Field(..., description="Type of task")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Task payload")
    priority: int = Field(default=5, ge=1, le=10, description="Task priority")
    required_capabilities: List[str] = Field(default_factory=list, description="Required capabilities")
    preferred_model: Optional[str] = Field(None, description="Preferred LLM model")


class TaskResponse(BaseModel):
    """Response model for task data."""
    task: Task
    assignments: List[TaskAssignment] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            UUID: str
        }


class TaskListResponse(BaseModel):
    """Response model for task list."""
    tasks: List[Task]
    total_count: int
    page: int
    page_size: int
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            UUID: str
        }


class AssignTaskRequest(BaseModel):
    """Request model for manual task assignment."""
    agent_id: UUID = Field(..., description="Agent to assign task to")
    reason: str = Field(..., description="Reason for manual assignment")


class TaskCompletionRequest(BaseModel):
    """Request model for marking task as completed."""
    result: Dict[str, Any] = Field(..., description="Task result data")
    duration_ms: int = Field(..., description="Task duration in milliseconds")
    model_used: str = Field(..., description="LLM model used")
    tokens_used: Optional[int] = Field(None, description="Tokens consumed")


class TaskFailureRequest(BaseModel):
    """Request model for marking task as failed."""
    error: str = Field(..., description="Error message")
    duration_ms: int = Field(..., description="Time spent before failure")
    model_used: str = Field(..., description="LLM model used")


class RoutingAnalyticsResponse(BaseModel):
    """Response model for routing analytics."""
    overview: Dict[str, Any]
    agent_performance: List[Dict[str, Any]]
    routing_efficiency: Dict[str, Any]


# =============================================================================
# TASK CRUD ENDPOINTS
# =============================================================================

@router.post("/", status_code=status.HTTP_201_CREATED)
async def submit_task(request: CreateTaskRequest) -> Dict[str, str]:
    """
    Submit a new task for routing.
    
    The task will be automatically routed to the best available agent
    based on capabilities, load, and performance history.
    """
    try:
        # Create task
        task = Task(
            task_type=request.task_type,
            payload=request.payload,
            priority=request.priority,
            required_capabilities=request.required_capabilities,
            preferred_model=request.preferred_model
        )
        
        # Save to database
        task_id = await create_task(task)
        
        # Route task asynchronously
        # Note: In production, this should be done via a task queue
        try:
            assignment = await task_router.route_task(task)
            if assignment:
                logger.info(f"Task {task_id} routed to agent {assignment.agent_id}")
            else:
                logger.info(f"Task {task_id} queued - no available agents")
        except Exception as e:
            logger.error(f"Failed to route task {task_id}: {e}")
            # Task is saved but not routed - it will be picked up later
        
        return {"task_id": str(task_id)}
        
    except Exception as e:
        logger.error(f"Failed to create task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create task: {str(e)}"
        )


@router.get("/", status_code=status.HTTP_200_OK)
async def list_tasks_endpoint(
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    task_type: Optional[str] = Query(None, description="Filter by task type"),
    assigned_agent_id: Optional[UUID] = Query(None, description="Filter by assigned agent"),
    priority_min: Optional[int] = Query(None, ge=1, le=10, description="Minimum priority"),
    priority_max: Optional[int] = Query(None, ge=1, le=10, description="Maximum priority"),
    human_override: Optional[bool] = Query(None, description="Filter by human override"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Page size")
) -> TaskListResponse:
    """List tasks with optional filtering and pagination."""
    try:
        # Build filter
        task_filter = TaskFilter(
            status=status_filter,
            task_type=task_type,
            assigned_agent_id=assigned_agent_id,
            priority_min=priority_min,
            priority_max=priority_max,
            human_override=human_override
        )
        
        # Get tasks
        offset = (page - 1) * page_size
        tasks, total_count = await list_tasks(task_filter, page_size, offset)
        
        return TaskListResponse(
            tasks=tasks,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list tasks: {str(e)}"
        )


@router.get("/queue", status_code=status.HTTP_200_OK)
async def get_queued_tasks_endpoint(
    priority_order: bool = Query(True, description="Order by priority"),
    limit: int = Query(50, ge=1, le=200, description="Maximum tasks to return")
) -> List[Task]:
    """Get queued tasks waiting for assignment."""
    try:
        tasks = await get_queued_tasks(priority_order, limit)
        return tasks
        
    except Exception as e:
        logger.error(f"Failed to get queued tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get queued tasks: {str(e)}"
        )


@router.get("/{task_id}", status_code=status.HTTP_200_OK)
async def get_task_details(task_id: UUID) -> TaskResponse:
    """Get detailed task information including assignments."""
    try:
        task = await get_task(task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        # TODO: Get task assignments when implemented
        assignments = []
        
        return TaskResponse(task=task, assignments=assignments)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task: {str(e)}"
        )


# =============================================================================
# TASK MANAGEMENT ENDPOINTS
# =============================================================================

@router.post("/{task_id}/assign", status_code=status.HTTP_200_OK)
async def assign_task_manually(task_id: UUID, request: AssignTaskRequest) -> Dict[str, str]:
    """
    Manually assign a task to a specific agent (human override).
    
    This bypasses the normal routing algorithm and assigns the task directly.
    The override is recorded for trust evolution tracking.
    """
    try:
        # Get task to verify it exists
        task = await get_task(task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        if task.status not in [TaskStatus.QUEUED, TaskStatus.FAILED]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot assign task with status '{task.status}'"
            )
        
        # Update task with human override
        success = await update_task_status(
            task_id,
            TaskStatus.ASSIGNED,
            assigned_agent_id=request.agent_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update task status"
            )
        
        # TODO: Record override in trust metrics
        # TODO: Create assignment record
        # TODO: Notify agent
        
        logger.info(f"Task {task_id} manually assigned to agent {request.agent_id}: {request.reason}")
        
        return {"message": "Task assigned successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to assign task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to assign task: {str(e)}"
        )


@router.post("/{task_id}/complete", status_code=status.HTTP_200_OK)
async def complete_task(task_id: UUID, request: TaskCompletionRequest) -> Dict[str, str]:
    """Mark a task as completed with results."""
    try:
        # Get task to verify it exists and get agent ID
        task = await get_task(task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        if task.status != TaskStatus.RUNNING:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot complete task with status '{task.status}'"
            )
        
        if not task.assigned_agent_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Task has no assigned agent"
            )
        
        # Handle completion via task router
        await task_router.handle_task_completion(
            task_id=task_id,
            result=request.result,
            duration_ms=request.duration_ms,
            agent_id=task.assigned_agent_id,
            model_used=request.model_used,
            tokens_used=request.tokens_used
        )
        
        logger.info(f"Task {task_id} completed successfully")
        
        return {"message": "Task completed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to complete task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to complete task: {str(e)}"
        )


@router.post("/{task_id}/fail", status_code=status.HTTP_200_OK)
async def fail_task(task_id: UUID, request: TaskFailureRequest) -> Dict[str, str]:
    """Mark a task as failed with error details."""
    try:
        # Get task to verify it exists and get agent ID
        task = await get_task(task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        if task.status not in [TaskStatus.ASSIGNED, TaskStatus.RUNNING]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot fail task with status '{task.status}'"
            )
        
        if not task.assigned_agent_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Task has no assigned agent"
            )
        
        # Handle failure via task router
        await task_router.handle_task_failure(
            task_id=task_id,
            error=request.error,
            agent_id=task.assigned_agent_id,
            duration_ms=request.duration_ms,
            model_used=request.model_used
        )
        
        logger.info(f"Task {task_id} marked as failed: {request.error}")
        
        return {"message": "Task marked as failed"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to mark task as failed {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to mark task as failed: {str(e)}"
        )


@router.post("/{task_id}/cancel", status_code=status.HTTP_200_OK)
async def cancel_task(task_id: UUID) -> Dict[str, str]:
    """Cancel a queued or running task."""
    try:
        # Get task to verify it exists
        task = await get_task(task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel task with status '{task.status}'"
            )
        
        # Update status
        success = await update_task_status(task_id, TaskStatus.CANCELLED)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to cancel task"
            )
        
        logger.info(f"Task {task_id} cancelled")
        
        return {"message": "Task cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel task: {str(e)}"
        )


@router.post("/{task_id}/retry", status_code=status.HTTP_200_OK)
async def retry_task(task_id: UUID) -> Dict[str, str]:
    """Retry a failed task."""
    try:
        # Get task to verify it exists
        task = await get_task(task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        if task.status != TaskStatus.FAILED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Can only retry failed tasks, task status is '{task.status}'"
            )
        
        # Reset task to queued status
        success = await update_task_status(
            task_id,
            TaskStatus.QUEUED,
            assigned_agent_id=None  # Clear previous assignment
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reset task status"
            )
        
        # Try to route again
        task.status = TaskStatus.QUEUED
        task.assigned_agent_id = None
        
        try:
            assignment = await task_router.route_task(task)
            if assignment:
                logger.info(f"Retried task {task_id} routed to agent {assignment.agent_id}")
            else:
                logger.info(f"Retried task {task_id} queued - no available agents")
        except Exception as e:
            logger.error(f"Failed to route retried task {task_id}: {e}")
        
        return {"message": "Task retry initiated"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retry task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retry task: {str(e)}"
        )


# =============================================================================
# ANALYTICS ENDPOINTS
# =============================================================================

@router.get("/analytics/overview", status_code=status.HTTP_200_OK)
async def get_task_analytics() -> TaskMetrics:
    """Get overall task routing analytics."""
    try:
        metrics = await get_task_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get task analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics: {str(e)}"
        )


@router.get("/analytics/agents", status_code=status.HTTP_200_OK)
async def get_agent_analytics() -> List[AgentTaskMetrics]:
    """Get per-agent task performance analytics."""
    try:
        metrics = await get_agent_task_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get agent analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent analytics: {str(e)}"
        )


@router.get("/analytics/routing", status_code=status.HTTP_200_OK)
async def get_routing_analytics() -> RoutingAnalyticsResponse:
    """Get comprehensive routing analytics for dashboard."""
    try:
        analytics = await task_router.get_routing_analytics()
        return RoutingAnalyticsResponse(**analytics)
        
    except Exception as e:
        logger.error(f"Failed to get routing analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get routing analytics: {str(e)}"
        )


# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================

@router.get("/types", status_code=status.HTTP_200_OK)
async def get_task_types() -> List[str]:
    """Get available task types."""
    return [
        TaskType.RESEARCH,
        TaskType.CODE,
        TaskType.ANALYSIS,
        TaskType.MONITORING,
        TaskType.WRITING,
        TaskType.COMMUNICATION,
        TaskType.PLANNING,
        TaskType.EVALUATION
    ]


@router.get("/statuses", status_code=status.HTTP_200_OK)
async def get_task_statuses() -> List[str]:
    """Get available task statuses."""
    return [
        TaskStatus.QUEUED,
        TaskStatus.ASSIGNED,
        TaskStatus.RUNNING,
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.REFUSED,
        TaskStatus.CANCELLED
    ]