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
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in intent classification"
    )
    requires_tools: bool = False
    tools_needed: List[str] = Field(default_factory=list)
    context_retrieved: Optional[str] = None
    ambiguities: List[str] = Field(
        default_factory=list, description="Things that need clarification"
    )
    task_category: Optional[str] = Field(
        default=None,
        description=(
            "Fine-grained task category for rubric-based evaluation "
            "(e.g. 'code', 'research', 'file_management', 'data_analysis', 'communication'). "
            "Optional — used by EvaluationAgent to select task-specific quality rubrics."
        ),
    )


class PlanStep(BaseModel):
    """A single step in the execution plan."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    tool: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(
        default_factory=list, description="Step IDs this depends on"
    )
    # Optional JSON Schema the step's StepResult.outputs must satisfy. Checked
    # by the deterministic per-step evaluation (app/evaluation/step_checks.py).
    output_schema: Optional[Dict[str, Any]] = None
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
    artifacts: List[str] = Field(
        default_factory=list, description="Paths to generated files/data"
    )
    logs: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    attempt: int = 1
    duration_seconds: float = 0.0
    # Tool the dispatcher actually ran for this step, or None when the step was
    # answered by the LLM, dry-run, or blocked. This is the tool-call trace that
    # claim verification checks against: a planned tool is not an executed one.
    dispatched_tool: Optional[str] = None


class EvaluationResult(BaseModel):
    """Result of evaluating the execution outcomes."""

    overall_status: Literal["success", "failure", "needs_revision", "needs_retry"]
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the outcome")
    meets_requirements: bool
    quality_score: float = Field(ge=0.0, le=1.0, description="Quality of the output")
    feedback: str = Field(default="", description="Detailed evaluation feedback")
    next_action: Literal["finalize", "retry_step", "revise_plan", "ask_user"] = (
        "finalize"
    )
    retry_step_id: Optional[str] = None
    revision_suggestions: List[str] = Field(default_factory=list)


class RefusalRecord(BaseModel):
    """An unresolved refusal or abstention that blocks further actions this run.

    ``source="gate"``: the comprehension gate chose clarify/refuse.
    ``source="reasoning"``: a step's LLM output declined to do the work.
    Only the user resolves it, by sending a new turn (a new run). Nothing inside
    the run (retry, plan revision) clears it.
    """

    source: Literal["gate", "reasoning"]
    reason: str
    step_id: Optional[str] = None
    excerpt: str = ""


class BlockedAction(BaseModel):
    """A plan step that was not run, and why.

    ``kind="refusal"``: a refusal was unresolved (see refusal_guard.py).
    ``kind="dependency_failed"``: a step it depends on failed, was blocked, or
    never ran (dependency cycle).
    ``kind="awaiting_confirmation"``: a step it depends on is held for the
    user's confirmation (see action_guard.py).
    """

    step_id: str
    step_name: str = ""
    tool: Optional[str] = None
    reason: str = ""
    kind: Literal["refusal", "dependency_failed", "awaiting_confirmation"] = "refusal"
    # The dependency that caused a dependency block (None for refusal blocks).
    cause_step_id: Optional[str] = None


class PendingConfirmation(BaseModel):
    """A dispatchable step held until the user confirms it.

    Held because its capability changes something (``side_effect="write"``)
    or because the plan marked it ``requires_hitl``. It was never dispatched.
    The user approves it by re-sending the request with the capability id in
    ``config["approved_capabilities"]`` (a new run).
    """

    step_id: str
    step_name: str = ""
    tool: Optional[str] = None
    capability_id: str
    side_effect: str
    reason: Literal["write", "requires_hitl"]
    # Short human-readable target (path, url or endpoint), for the response.
    target: str = ""


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

    # Comprehension gate (front verifier): "proceed" | "clarify" | "refuse".
    # gate_message carries the user-facing text for clarify/refuse.
    gate_outcome: Optional[str] = None
    gate_message: Optional[str] = None

    # Orchestration outputs
    plan: Optional[ExecutionPlan] = None

    # Orchestration grounding (Phase 2): structured report of plan steps dropped
    # for being invented tools / unknown actions / schema-invalid params, plus any
    # orphaned dependents. Optional + default so existing construction is unaffected.
    grounding_report: Optional[Dict[str, Any]] = None

    # Memory grounding (Phase 3): relevant facts/episodes, knowledge cues and
    # gaps found by comprehension. None when the feature flag is off.
    memory_grounding: Optional[Dict[str, Any]] = None

    # Reasoning outputs
    step_results: List[StepResult] = Field(default_factory=list)
    current_step_id: Optional[str] = None

    # Abstention guard: the refusal that latched this run (if any) and the steps
    # it kept from running. See app/core/agents/refusal_guard.py.
    refusal: Optional[RefusalRecord] = None
    blocked_actions: List[BlockedAction] = Field(default_factory=list)

    # Confirm tier: write (and requires_hitl) steps held for the user's
    # go-ahead instead of being dispatched. See app/core/agents/action_guard.py.
    pending_confirmations: List[PendingConfirmation] = Field(default_factory=list)

    # Claim verification: action claims in LLM step outputs checked against the
    # dispatched tool-call trace. See app/evaluation/claims.py.
    claim_report: Optional[Dict[str, Any]] = None

    # Per-step evaluation (Phase 3): one deterministic check per executed step
    # (StepCheck.to_dict()); failing ones are passed to the final evaluation.
    step_checks: List[Dict[str, Any]] = Field(default_factory=list)

    # Evaluation outputs
    eval_result: Optional[EvaluationResult] = None
    plan_revisions: int = Field(
        default=0, description="Number of plan revisions attempted this run"
    )

    # Conversation outputs
    response: Optional[str] = None
    messages: List[ConversationMessage] = Field(default_factory=list)

    # Execution metadata
    current_node: str = "START"
    execution_history: List[str] = Field(
        default_factory=list, description="Nodes visited in order"
    )
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
        status: Literal["pending", "running", "completed", "failed", "skipped"],
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
