"""
Task Models

Data models for the Task Routing Engine in the CORE Orchestration phase.
These models represent tasks, assignments, and results as they flow through the CORE loop.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class Task(BaseModel):
    """
    A task to be routed to an agent in the CORE system.
    
    Tasks flow through the CORE loop:
    1. Comprehension analyzes the task requirements
    2. Orchestration routes it to the best agent (this is where routing happens)
    3. Reasoning executes the task
    4. Evaluation checks the output
    """
    id: UUID = Field(default_factory=uuid4, description="Unique task identifier")
    task_type: str = Field(..., description="Type of task (research, code, analysis, monitoring, etc.)")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Task-specific data and parameters")
    priority: int = Field(default=5, ge=1, le=10, description="Task priority (1-10, higher = more urgent)")
    required_capabilities: List[str] = Field(default_factory=list, description="Required agent capabilities")
    preferred_model: Optional[str] = Field(None, description="Preferred LLM model (e.g., 'ollama/llama3.2' or 'claude-sonnet')")
    status: str = Field(default="queued", description="Current task status")
    assigned_agent_id: Optional[UUID] = Field(None, description="ID of agent assigned to this task")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result data")
    human_override: bool = Field(default=False, description="Was this assignment overridden by human?")
    override_reason: Optional[str] = Field(None, description="Reason for human override")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Task creation timestamp")
    assigned_at: Optional[datetime] = Field(None, description="Task assignment timestamp")
    completed_at: Optional[datetime] = Field(None, description="Task completion timestamp")
    duration_ms: Optional[int] = Field(None, description="Task execution duration in milliseconds")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            UUID: str
        }


class TaskAssignment(BaseModel):
    """
    Represents an assignment of a task to an agent.
    
    Contains the agent's response to the assignment and any feedback.
    """
    task_id: UUID = Field(..., description="ID of the task being assigned")
    agent_id: UUID = Field(..., description="ID of the agent receiving the assignment")
    agent_response: str = Field(..., description="Agent's response (accept, refuse, suggest_alternative)")
    refusal_reason: Optional[str] = Field(None, description="Reason for task refusal")
    suggested_agent: Optional[str] = Field(None, description="Agent suggested as alternative")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Agent's confidence in handling this task")
    assigned_at: datetime = Field(default_factory=datetime.utcnow, description="Assignment timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            UUID: str
        }


class TaskResult(BaseModel):
    """
    Result of a completed task execution.
    
    Contains the outcome, performance metrics, and metadata about task execution.
    """
    task_id: UUID = Field(..., description="ID of the completed task")
    agent_id: UUID = Field(..., description="ID of the agent that executed the task")
    status: str = Field(..., description="Final task status (completed, failed, partial)")
    result: Dict[str, Any] = Field(default_factory=dict, description="Task result data")
    duration_ms: int = Field(..., description="Task execution duration in milliseconds")
    model_used: str = Field(..., description="LLM model used for task execution")
    tokens_used: Optional[int] = Field(None, description="Number of tokens consumed")
    error_message: Optional[str] = Field(None, description="Error message if task failed")
    completed_at: datetime = Field(default_factory=datetime.utcnow, description="Completion timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            UUID: str
        }


class TaskFilter(BaseModel):
    """Filter criteria for task queries."""
    status: Optional[str] = None
    task_type: Optional[str] = None
    assigned_agent_id: Optional[UUID] = None
    priority_min: Optional[int] = Field(None, ge=1, le=10)
    priority_max: Optional[int] = Field(None, ge=1, le=10)
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    human_override: Optional[bool] = None


class TaskMetrics(BaseModel):
    """Analytics metrics for task routing performance."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    refused_tasks: int = 0
    avg_completion_time_ms: Optional[float] = None
    success_rate: float = 0.0
    refusal_rate: float = 0.0
    queue_depth: int = 0
    avg_queue_wait_time_ms: Optional[float] = None


class AgentTaskMetrics(BaseModel):
    """Task metrics for a specific agent."""
    agent_id: UUID
    agent_role: str
    total_assigned: int = 0
    completed: int = 0
    failed: int = 0
    refused: int = 0
    avg_duration_ms: Optional[float] = None
    success_rate: float = 0.0
    refusal_rate: float = 0.0
    current_load: int = 0  # Number of currently assigned tasks


class TaskRoutingScore(BaseModel):
    """Routing score calculation for agent selection."""
    agent_id: UUID
    total_score: float = Field(ge=0.0, le=1.0, description="Combined routing score")
    capability_match_score: float = Field(ge=0.0, le=1.0, description="How well capabilities match requirements")
    load_score: float = Field(ge=0.0, le=1.0, description="Current load score (lower load = higher score)")
    trust_score: float = Field(ge=0.0, le=1.0, description="Trust score from historical performance")
    model_preference_score: float = Field(ge=0.0, le=1.0, description="Model preference alignment score")
    task_type_performance_score: float = Field(ge=0.0, le=1.0, description="Historical performance on this task type")


# Task status constants
class TaskStatus:
    """Constants for task status values."""
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUSED = "refused"
    CANCELLED = "cancelled"


# Task response constants
class AgentResponse:
    """Constants for agent response values."""
    ACCEPT = "accept"
    REFUSE = "refuse"
    SUGGEST_ALTERNATIVE = "suggest_alternative"


# Task type constants
class TaskType:
    """Constants for common task types."""
    RESEARCH = "research"
    CODE = "code"
    ANALYSIS = "analysis"
    MONITORING = "monitoring"
    WRITING = "writing"
    COMMUNICATION = "communication"
    PLANNING = "planning"
    EVALUATION = "evaluation"