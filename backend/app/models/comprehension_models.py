"""
Comprehension Engine Models

Data models for the Comprehension phase of the CORE loop.
The Comprehension Engine is the intake/analysis layer that:
  1. Parses intent from raw input
  2. Runs semantic similarity against LangMem knowledge base
  3. Matches capabilities against available agents/tools
  4. Produces a structured ComprehensionResult that feeds into the Task Router

Flow: Raw Input → Comprehension → Orchestration → Reasoning → Evaluation
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Literal
from uuid import UUID, uuid4
from enum import Enum

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class SourceType(str, Enum):
    """Source of the comprehension input."""
    USER = "user"                # Direct user request
    AGENT = "agent"              # Message from another agent
    SYSTEM = "system"            # System-generated event
    TOOL_OUTPUT = "tool_output"  # Output from a tool execution


class ActionType(str, Enum):
    """Classified action type for parsed intent."""
    QUERY = "query"              # Information retrieval / question
    COMMAND = "command"           # Direct action request
    ANALYSIS = "analysis"        # Analysis or evaluation request
    CREATION = "creation"        # Create something new (code, doc, etc.)
    MONITORING = "monitoring"    # Ongoing monitoring / observation
    CONVERSATION = "conversation"  # Conversational / social interaction
    PLANNING = "planning"        # Planning / strategy
    UNKNOWN = "unknown"          # Could not classify


class UrgencyLevel(str, Enum):
    """Urgency level of the input."""
    CRITICAL = "critical"    # Immediate action needed
    HIGH = "high"            # Important, time-sensitive
    MEDIUM = "medium"        # Normal priority
    LOW = "low"              # Can wait, background task
    NONE = "none"            # No urgency


class HandlingMode(str, Enum):
    """Whether single-agent or multi-agent handling is needed."""
    SINGLE_AGENT = "single_agent"
    MULTI_AGENT = "multi_agent"
    NO_AGENT = "no_agent"        # Can be answered from context alone
    HUMAN_REQUIRED = "human_required"  # Needs human intervention


class ComprehensionStatus(str, Enum):
    """Status of a comprehension result for tracking."""
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some steps succeeded, others didn't


# =============================================================================
# INPUT MODEL
# =============================================================================

class ComprehensionInput(BaseModel):
    """
    Raw input to the Comprehension Engine.
    
    This is the entry point for anything entering the CORE loop.
    Can come from users, agents, system events, or tool outputs.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique input identifier")
    content: str = Field(..., description="Raw input text content")
    source_type: SourceType = Field(..., description="Where this input came from")
    source_id: Optional[str] = Field(None, description="ID of the source (user_id, agent_id, tool_name)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context from the source")
    conversation_id: Optional[str] = Field(None, description="Conversation context if applicable")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the input was received")

    @field_validator('content')
    @classmethod
    def content_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()


# =============================================================================
# INTENT ANALYSIS
# =============================================================================

class ExtractedEntity(BaseModel):
    """An entity extracted from the input."""
    name: str = Field(..., description="Entity name or value")
    entity_type: str = Field(..., description="Type of entity (person, topic, tool, etc.)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in extraction")


class IntentAnalysis(BaseModel):
    """
    Parsed intent from the raw input.
    
    Produced by LLM-based intent parsing. Contains the classified action,
    extracted entities, urgency assessment, and confidence scores.
    """
    action_type: ActionType = Field(..., description="Classified action type")
    summary: str = Field(..., description="Brief summary of what is being requested")
    entities: List[ExtractedEntity] = Field(default_factory=list, description="Extracted entities")
    urgency: UrgencyLevel = Field(default=UrgencyLevel.MEDIUM, description="Assessed urgency level")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence in intent parsing")
    keywords: List[str] = Field(default_factory=list, description="Key terms extracted from input")
    requires_clarification: bool = Field(default=False, description="Whether the input is ambiguous")
    clarification_questions: List[str] = Field(
        default_factory=list, 
        description="Suggested clarification questions if ambiguous"
    )


# =============================================================================
# CONTEXT MATCHING
# =============================================================================

class MemoryMatch(BaseModel):
    """A single memory match from semantic search."""
    memory_id: UUID = Field(..., description="ID of the matched memory")
    content: str = Field(..., description="Memory content")
    similarity: float = Field(ge=0.0, le=1.0, description="Semantic similarity score")
    memory_tier: str = Field(..., description="Which tier: semantic, episodic, procedural")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Memory metadata")


class ContextMatch(BaseModel):
    """
    Semantic similarity results from all memory tiers.
    
    Aggregates results from searching semantic, episodic, and procedural
    memories against the input.
    """
    semantic_matches: List[MemoryMatch] = Field(
        default_factory=list, description="Matches from shared semantic memory"
    )
    episodic_matches: List[MemoryMatch] = Field(
        default_factory=list, description="Matches from agent episodic memory"
    )
    procedural_matches: List[MemoryMatch] = Field(
        default_factory=list, description="Matches from procedural memory"
    )
    has_relevant_context: bool = Field(default=False, description="Whether any relevant context was found")
    best_match_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Highest similarity score")

    @property
    def total_matches(self) -> int:
        return len(self.semantic_matches) + len(self.episodic_matches) + len(self.procedural_matches)

    @property
    def all_matches(self) -> List[MemoryMatch]:
        return self.semantic_matches + self.episodic_matches + self.procedural_matches


# =============================================================================
# CAPABILITY MATCHING
# =============================================================================

class MatchedCapability(BaseModel):
    """A capability that matches the input requirements."""
    capability_name: str = Field(..., description="Name of the matched capability")
    description: str = Field(default="", description="Description of the capability")
    match_score: float = Field(ge=0.0, le=1.0, description="How well this capability matches")
    source_type: str = Field(..., description="Where this capability comes from: agent, tool, service")
    source_id: str = Field(..., description="ID of the agent/tool/service providing this capability")


class CapabilityMatch(BaseModel):
    """
    Matched capabilities from agents, tools, and services.
    
    Identifies what in the system can handle the parsed intent.
    """
    matched_capabilities: List[MatchedCapability] = Field(
        default_factory=list, description="Capabilities that match the input"
    )
    matched_agent_ids: List[str] = Field(
        default_factory=list, description="Agent IDs that can handle this"
    )
    matched_tools: List[str] = Field(
        default_factory=list, description="Tools that could be useful"
    )
    has_capable_agents: bool = Field(default=False, description="Whether any agent can handle this")
    best_agent_id: Optional[str] = Field(None, description="ID of the best-matched agent")
    best_agent_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Best agent match score")


# =============================================================================
# COMPLEXITY ANALYSIS
# =============================================================================

class ComplexityScore(BaseModel):
    """
    Estimated complexity of the task for routing decisions.
    
    Used by the Task Router to determine resource allocation.
    """
    overall: float = Field(ge=0.0, le=1.0, description="Overall complexity score (0=trivial, 1=extremely complex)")
    reasoning_depth: float = Field(
        ge=0.0, le=1.0, description="How much reasoning is needed"
    )
    breadth: float = Field(
        ge=0.0, le=1.0, description="How many different areas/skills are involved"
    )
    novelty: float = Field(
        ge=0.0, le=1.0, description="How novel/unprecedented this request is"
    )
    estimated_duration_seconds: Optional[int] = Field(
        None, description="Estimated time to complete"
    )


# =============================================================================
# COMPREHENSION RESULT (MAIN OUTPUT)
# =============================================================================

class ComprehensionResult(BaseModel):
    """
    The full output of the Comprehension Engine.
    
    This is the structured result that feeds directly into the Task Router
    (Orchestration phase). Contains everything needed to make routing decisions:
    - What is being asked (intent)
    - What context is relevant (memory matches)
    - What can handle it (capability matches)
    - How to route it (suggested task type, priority, handling mode)
    
    This model bridges Comprehension → Orchestration in the CORE loop.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique comprehension result ID")
    
    # Original input
    input: ComprehensionInput = Field(..., description="The original input that was analyzed")
    
    # Analysis results
    intent: IntentAnalysis = Field(..., description="Parsed intent analysis")
    context: ContextMatch = Field(default_factory=ContextMatch, description="Memory context matches")
    capabilities: CapabilityMatch = Field(default_factory=CapabilityMatch, description="Capability matches")
    complexity: ComplexityScore = Field(
        default_factory=lambda: ComplexityScore(
            overall=0.5, reasoning_depth=0.5, breadth=0.5, novelty=0.5
        ), 
        description="Task complexity analysis"
    )
    
    # Routing suggestions (for the Task Router)
    suggested_task_type: str = Field(default="general", description="Suggested task type for routing")
    suggested_priority: int = Field(default=5, ge=1, le=10, description="Suggested priority (1-10)")
    handling_mode: HandlingMode = Field(
        default=HandlingMode.SINGLE_AGENT, description="Whether single or multi-agent handling is needed"
    )
    suggested_capabilities: List[str] = Field(
        default_factory=list, description="Capabilities the task router should look for"
    )
    
    # Comprehension metadata
    confidence: float = Field(ge=0.0, le=1.0, description="Overall comprehension confidence")
    status: ComprehensionStatus = Field(
        default=ComprehensionStatus.COMPLETED, description="Status of comprehension"
    )
    processing_time_ms: Optional[int] = Field(None, description="Time taken to comprehend")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When this result was created")
    
    # Feedback tracking
    feedback_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Feedback score on comprehension accuracy"
    )
    feedback_notes: Optional[str] = Field(None, description="Feedback notes")

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            UUID: str
        }

    def to_task_kwargs(self) -> Dict[str, Any]:
        """
        Convert comprehension result to kwargs suitable for creating a Task.
        
        This is the bridge method that connects Comprehension → Orchestration.
        Returns a dict compatible with Task model constructor.
        """
        return {
            "task_type": self.suggested_task_type,
            "payload": {
                "original_input": self.input.content,
                "source_type": self.input.source_type.value,
                "intent_summary": self.intent.summary,
                "action_type": self.intent.action_type.value,
                "entities": [e.model_dump() for e in self.intent.entities],
                "context_matches": len(self.context.all_matches),
                "complexity_score": self.complexity.overall,
                "comprehension_id": str(self.id),
            },
            "priority": self.suggested_priority,
            "required_capabilities": self.suggested_capabilities,
        }


# =============================================================================
# FEEDBACK MODEL
# =============================================================================

class ComprehensionFeedback(BaseModel):
    """
    Feedback on comprehension accuracy for the learning loop.
    
    Used to track whether the Comprehension Engine correctly understood
    the input. This feeds back into improving future comprehension.
    """
    comprehension_id: UUID = Field(..., description="ID of the comprehension result being rated")
    score: float = Field(ge=0.0, le=1.0, description="Accuracy score (0=wrong, 1=perfect)")
    correct_action_type: Optional[str] = Field(None, description="What the action type should have been")
    correct_task_type: Optional[str] = Field(None, description="What the task type should have been")
    notes: Optional[str] = Field(None, description="Additional feedback notes")
    submitted_by: Optional[str] = Field(None, description="Who submitted the feedback")
    submitted_at: datetime = Field(default_factory=datetime.utcnow, description="When feedback was submitted")


# =============================================================================
# CAPABILITY REGISTRY
# =============================================================================

class SystemCapability(BaseModel):
    """
    A capability registered in the system.
    
    Used by get_capability_registry() to return what the system can do.
    """
    name: str = Field(..., description="Capability name")
    description: str = Field(default="", description="What this capability does")
    provider_type: str = Field(..., description="Type of provider: agent, tool, service")
    provider_id: str = Field(..., description="ID of the provider")
    provider_name: str = Field(default="", description="Human-readable provider name")
    is_available: bool = Field(default=True, description="Whether currently available")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
