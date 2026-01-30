"""
CORE State Schema - Defines the shared state for the CORE cognitive graph.

This state flows through the CORE pipeline:
START → Comprehension → Orchestration → Reasoning → Evaluation → Conversation → END
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from uuid import UUID, uuid4


# --- Core State Schema ---

class UserIntent(BaseModel):
    """Result of comprehension analysis."""
    type: Literal["task", "conversation", "question", "clarification"]
    description: str
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in intent classification")
    requires_tools: bool = False
    tools_needed: List[str] = Field(default_factory=list)
    context_retrieved: Optional[str] = None
    ambiguities: List[str] = Field(default_factory=list, description="Things that need clarification")


class PlanStep(BaseModel):
    """A single step in the execution plan."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    tool: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list, description="Step IDs this depends on")
    requires_hitl: bool = False  # Human-in-the-loop checkpoint
    retry_policy: Dict[str, Any] = Field(
        default_factory=lambda: {"max_attempts": 3, "backoff_seconds": 1}
    )
    status: Literal["pending", "running", "completed", "failed", "skipped"] = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ExecutionPlan(BaseModel):
    """The orchestrated plan for accomplishing the task."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    goal: str
    steps: List[PlanStep]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    revision: int = 1
    reasoning: str = Field(default="", description="Why this plan was chosen")


class StepResult(BaseModel):
    """Result of executing a plan step."""
    step_id: str
    status: Literal["success", "failure", "partial"]
    outputs: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list, description="Paths to generated files/data")
    logs: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    attempt: int = 1
    duration_seconds: float = 0.0


class EvaluationResult(BaseModel):
    """Result of evaluating the execution outcomes."""
    overall_status: Literal["success", "failure", "needs_revision", "needs_retry"]
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the outcome")
    meets_requirements: bool
    quality_score: float = Field(ge=0.0, le=1.0, description="Quality of the output")
    feedback: str = Field(default="", description="Detailed evaluation feedback")
    next_action: Literal["finalize", "retry_step", "revise_plan", "ask_user"] = "finalize"
    retry_step_id: Optional[str] = None
    revision_suggestions: List[str] = Field(default_factory=list)


class ConversationMessage(BaseModel):
    """A message in the conversation flow."""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class COREState(BaseModel):
    """
    Complete state that flows through the CORE graph.

    This state is shared across all nodes and represents the current execution context.
    """
    # Unique identifiers
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None

    # Input
    user_input: str
    user_message: Optional[ConversationMessage] = None

    # Comprehension outputs
    intent: Optional[UserIntent] = None

    # Orchestration outputs
    plan: Optional[ExecutionPlan] = None

    # Reasoning outputs
    step_results: List[StepResult] = Field(default_factory=list)
    current_step_id: Optional[str] = None

    # Evaluation outputs
    eval_result: Optional[EvaluationResult] = None

    # Conversation outputs
    response: Optional[str] = None
    messages: List[ConversationMessage] = Field(default_factory=list)

    # Execution metadata
    current_node: str = "START"
    execution_history: List[str] = Field(default_factory=list, description="Nodes visited in order")
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Configuration
    config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_iterations": 10,
            "enable_tools": True,
            "enable_hitl": False,
        }
    )

    def add_execution_node(self, node_name: str) -> None:
        """Track that we visited a node."""
        self.execution_history.append(node_name)
        self.current_node = node_name
        self.updated_at = datetime.utcnow()

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.updated_at = datetime.utcnow()

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
        self.updated_at = datetime.utcnow()

    def get_step(self, step_id: str) -> Optional[PlanStep]:
        """Get a plan step by ID."""
        if not self.plan:
            return None
        for step in self.plan.steps:
            if step.id == step_id:
                return step
        return None

    def update_step_status(
        self,
        step_id: str,
        status: Literal["pending", "running", "completed", "failed", "skipped"]
    ) -> None:
        """Update a step's status."""
        step = self.get_step(step_id)
        if step:
            step.status = status
            if status == "running":
                step.started_at = datetime.utcnow()
            elif status in ["completed", "failed", "skipped"]:
                step.completed_at = datetime.utcnow()
            self.updated_at = datetime.utcnow()

    def is_complete(self) -> bool:
        """Check if execution is complete."""
        return self.current_node == "END" or self.completed_at is not None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
