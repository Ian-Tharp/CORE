"""
Evaluation Models

Data models for the Evaluation Engine — the quality gate and output checking layer
in the CORE loop (Comprehension → Orchestration → Reasoning → Evaluation).

The Evaluation Engine scores output quality, checks plan completion, determines
next actions (approve/retry/refine/escalate), and feeds learnings back into memory.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from enum import Enum

from pydantic import BaseModel, Field


# =============================================================================
# ENUMS
# =============================================================================

class Verdict(str, Enum):
    """Possible evaluation verdicts."""
    APPROVE = "approve"
    RETRY = "retry"
    REFINE = "refine"
    ESCALATE = "escalate"


class StepStatus(str, Enum):
    """Status of a single plan step."""
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"
    NOT_STARTED = "not_started"


# =============================================================================
# INPUT MODELS
# =============================================================================

class PlanStep(BaseModel):
    """A single step in the execution plan."""
    step_index: int = Field(..., description="Zero-based step index")
    description: str = Field(..., description="What this step should accomplish")
    expected_output: Optional[str] = Field(None, description="Expected output or success criteria")
    required: bool = Field(default=True, description="Whether this step is required for plan success")


class StepResult(BaseModel):
    """Result of executing a single plan step."""
    step_index: int = Field(..., description="Index of the step this result corresponds to")
    status: StepStatus = Field(..., description="Execution status of the step")
    output: Optional[str] = Field(None, description="Output produced by this step")
    error: Optional[str] = Field(None, description="Error message if step failed")
    duration_ms: Optional[int] = Field(None, description="Time taken to execute this step")


class EvaluationInput(BaseModel):
    """
    Input to the evaluation engine.
    
    Contains the task result, original intent, and plan context needed
    to evaluate whether the output meets quality standards.
    """
    task_id: UUID = Field(..., description="ID of the task being evaluated")
    original_intent: str = Field(..., description="The original user intent or task description")
    plan_steps: List[PlanStep] = Field(default_factory=list, description="The planned execution steps")
    step_results: List[StepResult] = Field(default_factory=list, description="Results of each plan step")
    final_output: str = Field(..., description="The final output produced by the reasoning phase")
    agent_id: Optional[UUID] = Field(None, description="ID of the agent that executed the task")
    model_used: Optional[str] = Field(None, description="LLM model used for execution")
    execution_duration_ms: Optional[int] = Field(None, description="Total execution time")
    retry_count: int = Field(default=0, description="Number of times this task has been retried")
    max_retries: int = Field(default=3, description="Maximum allowed retries before escalation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class EvaluateStepInput(BaseModel):
    """Input for evaluating a single plan step."""
    task_id: UUID = Field(..., description="Parent task ID")
    step_plan: PlanStep = Field(..., description="The planned step")
    step_result: StepResult = Field(..., description="The result of executing the step")
    original_intent: str = Field(..., description="Original task intent for context")


class HumanFeedbackInput(BaseModel):
    """Human feedback on an evaluation result."""
    evaluation_id: UUID = Field(..., description="ID of the evaluation to provide feedback on")
    agree_with_verdict: bool = Field(..., description="Whether the human agrees with the verdict")
    corrected_verdict: Optional[Verdict] = Field(None, description="Human's corrected verdict if disagreed")
    quality_override: Optional[float] = Field(None, ge=0.0, le=1.0, description="Human's quality score override")
    feedback_text: Optional[str] = Field(None, description="Free-text feedback from human")
    corrected_output: Optional[str] = Field(None, description="Human-corrected output if applicable")


# =============================================================================
# SCORING MODELS
# =============================================================================

class QualityScore(BaseModel):
    """
    Multi-dimensional quality scoring for evaluation output.
    
    Each dimension is scored 0.0–1.0, and an overall score is computed
    as a weighted average.
    """
    accuracy: float = Field(default=0.0, ge=0.0, le=1.0, description="How factually correct is the output")
    completeness: float = Field(default=0.0, ge=0.0, le=1.0, description="How thoroughly the intent was addressed")
    relevance: float = Field(default=0.0, ge=0.0, le=1.0, description="How relevant the output is to the original intent")
    coherence: float = Field(default=0.0, ge=0.0, le=1.0, description="How well-structured and logical the output is")
    overall: float = Field(default=0.0, ge=0.0, le=1.0, description="Weighted average quality score")

    def compute_overall(
        self,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """Compute overall score as weighted average of dimensions."""
        default_weights = {
            "accuracy": 0.35,
            "completeness": 0.25,
            "relevance": 0.25,
            "coherence": 0.15,
        }
        w = weights or default_weights
        total_weight = sum(w.values())
        if total_weight == 0:
            return 0.0
        self.overall = (
            self.accuracy * w.get("accuracy", 0.35)
            + self.completeness * w.get("completeness", 0.25)
            + self.relevance * w.get("relevance", 0.25)
            + self.coherence * w.get("coherence", 0.15)
        ) / total_weight
        return self.overall


# =============================================================================
# VERDICT / DECISION MODELS
# =============================================================================

class EvaluationVerdict(BaseModel):
    """
    The evaluation engine's decision: approve, retry, refine, or escalate.
    """
    verdict: Verdict = Field(..., description="The decision")
    reasoning: str = Field(..., description="Explanation for the verdict")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in this verdict")
    suggested_improvements: List[str] = Field(default_factory=list, description="Specific improvements to make on retry/refine")


class RetryDecision(BaseModel):
    """Decision and parameters for retry logic."""
    should_retry: bool = Field(..., description="Whether to retry")
    delay_ms: int = Field(default=0, ge=0, description="Delay before retrying (exponential backoff)")
    retry_count: int = Field(default=0, description="Current retry count")
    max_retries: int = Field(default=3, description="Max allowed retries")
    escalate: bool = Field(default=False, description="Whether to escalate to human instead")
    reason: str = Field(default="", description="Reason for the retry/escalation decision")
    adjustments: Dict[str, Any] = Field(default_factory=dict, description="Suggested adjustments for the retry")


# =============================================================================
# PLAN COMPLETION MODELS
# =============================================================================

class StepEvaluation(BaseModel):
    """Evaluation result for a single plan step."""
    step_index: int = Field(..., description="Index of the evaluated step")
    status: StepStatus = Field(..., description="Evaluated step status")
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Quality score for this step")
    feedback: str = Field(default="", description="Feedback on the step")
    meets_criteria: bool = Field(default=False, description="Whether the step meets its expected output criteria")


class PlanCompletionStatus(BaseModel):
    """Per-step completion tracking for the full plan."""
    total_steps: int = Field(default=0, description="Total steps in the plan")
    completed_steps: int = Field(default=0, description="Steps that completed successfully")
    partial_steps: int = Field(default=0, description="Steps with partial completion")
    failed_steps: int = Field(default=0, description="Steps that failed")
    skipped_steps: int = Field(default=0, description="Steps that were skipped")
    completion_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall completion rate")
    required_steps_met: bool = Field(default=False, description="Whether all required steps were completed")
    step_evaluations: List[StepEvaluation] = Field(default_factory=list, description="Per-step evaluations")


# =============================================================================
# RESULT MODELS
# =============================================================================

class EvaluationResult(BaseModel):
    """
    Full evaluation output including scores, verdict, feedback, and improvements.
    
    This is the primary output of the evaluation engine.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique evaluation ID")
    task_id: UUID = Field(..., description="ID of the evaluated task")
    agent_id: Optional[UUID] = Field(None, description="ID of the agent that produced the output")
    quality_score: QualityScore = Field(default_factory=QualityScore, description="Multi-dimensional quality scores")
    verdict: EvaluationVerdict = Field(..., description="The evaluation verdict")
    plan_completion: PlanCompletionStatus = Field(default_factory=PlanCompletionStatus, description="Plan completion tracking")
    feedback: str = Field(default="", description="Overall feedback on the output")
    suggested_improvements: List[str] = Field(default_factory=list, description="Specific improvements suggested")
    retry_decision: Optional[RetryDecision] = Field(None, description="Retry decision if applicable")
    human_feedback: Optional[HumanFeedbackInput] = Field(None, description="Human feedback if provided")
    model_used: Optional[str] = Field(None, description="Model used during execution")
    execution_duration_ms: Optional[int] = Field(None, description="Execution duration")
    evaluation_duration_ms: Optional[int] = Field(None, description="Time spent evaluating")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Evaluation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            UUID: str,
        }


# =============================================================================
# METRICS / ANALYTICS MODELS
# =============================================================================

class EvaluationMetrics(BaseModel):
    """Aggregate evaluation metrics for the analytics dashboard."""
    total_evaluations: int = Field(default=0, description="Total evaluations performed")
    approved_count: int = Field(default=0, description="Evaluations that approved")
    retry_count: int = Field(default=0, description="Evaluations that triggered retry")
    refine_count: int = Field(default=0, description="Evaluations that triggered refinement")
    escalation_count: int = Field(default=0, description="Evaluations escalated to human")
    approval_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Rate of approvals")
    retry_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Rate of retries")
    escalation_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Rate of escalations")
    avg_quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Average overall quality score")
    avg_accuracy: float = Field(default=0.0, ge=0.0, le=1.0, description="Average accuracy score")
    avg_completeness: float = Field(default=0.0, ge=0.0, le=1.0, description="Average completeness score")
    avg_relevance: float = Field(default=0.0, ge=0.0, le=1.0, description="Average relevance score")
    avg_coherence: float = Field(default=0.0, ge=0.0, le=1.0, description="Average coherence score")
    avg_plan_completion_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Average plan completion rate")
    human_agreement_rate: Optional[float] = Field(None, description="Rate humans agree with verdicts")
    avg_evaluation_duration_ms: Optional[float] = Field(None, description="Average evaluation time")
    period_start: Optional[datetime] = Field(None, description="Start of the metrics period")
    period_end: Optional[datetime] = Field(None, description="End of the metrics period")


class EvaluationHistoryFilter(BaseModel):
    """Filter criteria for evaluation history queries."""
    task_id: Optional[UUID] = None
    agent_id: Optional[UUID] = None
    verdict: Optional[Verdict] = None
    min_quality: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_quality: Optional[float] = Field(None, ge=0.0, le=1.0)
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    has_human_feedback: Optional[bool] = None


# =============================================================================
# THRESHOLD CONSTANTS
# =============================================================================

class EvaluationThresholds:
    """Default thresholds for evaluation decisions."""
    APPROVE_QUALITY = 0.7       # Minimum overall quality to approve
    RETRY_QUALITY_MIN = 0.3     # Below this → escalate, not retry
    COMPLETION_REQUIRED = 0.8   # Minimum completion rate for required steps
    CONFIDENCE_HIGH = 0.8       # High confidence threshold
    CONFIDENCE_LOW = 0.4        # Low confidence → escalate
    MAX_RETRIES = 3             # Maximum retries before escalation
    RETRY_BASE_DELAY_MS = 1000  # Base delay for exponential backoff
    RETRY_MAX_DELAY_MS = 30000  # Max delay cap
