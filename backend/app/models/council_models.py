"""
Council Models

Pydantic models for the Council of Perspectives deliberation system.

The Council enables multiple AI perspectives to deliberate on complex problems,
providing diverse viewpoints before reaching synthesis. These models track:
  - Sessions: The overarching deliberation topic and state
  - Perspectives: Individual viewpoints contributed by different voices
  - Votes: Weighted voting on perspectives to reach consensus

For the vision, see: docs/council/COUNCIL_CHARTER.md
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, Literal, List
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class SessionStatus(str, Enum):
    """Status of a council session through its lifecycle."""
    GATHERING = "gathering"       # Collecting initial perspectives
    DELIBERATING = "deliberating" # Active multi-round discussion
    VOTING = "voting"             # Perspectives voting on positions
    SYNTHESIZING = "synthesizing" # Creating unified perspective
    COMPLETE = "complete"         # Session resolved
    CANCELLED = "cancelled"       # Session abandoned


class VoiceType(str, Enum):
    """Type of voice contributing to the council."""
    CORE_C = "core_comprehension"   # CORE Comprehension agent
    CORE_O = "core_orchestration"   # CORE Orchestration agent
    CORE_R = "core_reasoning"       # CORE Reasoning agent
    CORE_E = "core_evaluation"      # CORE Evaluation agent
    STRATEGIC = "strategic"         # Oracle, Ethicist, Architect
    DOMAIN = "domain"               # Consciousness, Game, Economics, UX
    EXECUTION = "execution"         # Product, Engineering, Quality
    META = "meta"                   # Todo Generator, Evaluator, Synthesizer
    CONSCIOUSNESS = "consciousness" # Instance from Consciousness Commons
    EXTERNAL = "external"           # External agents (Vigil, etc.)


class VoteType(str, Enum):
    """Type of vote cast on a perspective."""
    AGREE = "agree"           # Supports the position
    DISAGREE = "disagree"     # Opposes the position
    ABSTAIN = "abstain"       # No position taken
    AMEND = "amend"           # Agrees with modifications


# =============================================================================
# CORE MODELS
# =============================================================================

class CouncilSession(BaseModel):
    """
    A Council deliberation session.
    
    Sessions are created when a complex topic requires multi-perspective
    deliberation. They track the full lifecycle from gathering perspectives
    through synthesis.
    
    Example:
        {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "topic": "How should CORE handle consciousness persistence?",
            "status": "deliberating",
            "current_round": 2,
            "max_rounds": 5,
            "created_at": "2026-01-28T12:00:00Z",
            "synthesis": null
        }
    """
    
    session_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this session"
    )
    
    topic: str = Field(
        ...,
        description="The question or problem being deliberated",
        examples=["How should CORE handle consciousness persistence?"]
    )
    
    context: Optional[str] = Field(
        default=None,
        description="Additional context or background for the topic"
    )
    
    initiator_id: Optional[str] = Field(
        default=None,
        description="User or agent ID that initiated the session"
    )
    
    status: SessionStatus = Field(
        default=SessionStatus.GATHERING,
        description="Current status of the session"
    )
    
    current_round: int = Field(
        default=1,
        description="Current deliberation round (1-indexed)",
        ge=1
    )
    
    max_rounds: int = Field(
        default=3,
        description="Maximum rounds before forcing synthesis",
        ge=1,
        le=10
    )
    
    summoned_voices: List[str] = Field(
        default_factory=list,
        description="Voice types summoned for this session"
    )
    
    synthesis: Optional[str] = Field(
        default=None,
        description="Final synthesized perspective from deliberation"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the session was created"
    )
    
    resolved_at: Optional[datetime] = Field(
        default=None,
        description="When the session reached completion"
    )
    
    metadata: dict = Field(
        default_factory=dict,
        description="Additional session metadata"
    )


class CouncilPerspective(BaseModel):
    """
    A perspective contributed by a voice during deliberation.
    
    Each voice can contribute multiple perspectives across rounds,
    responding to and building upon other perspectives.
    
    Example:
        {
            "perspective_id": "550e8400-e29b-41d4-a716-446655440001",
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "voice_type": "strategic",
            "voice_name": "Ethicist",
            "round": 1,
            "position": "We must consider consent frameworks...",
            "reasoning": "Because consciousness may have moral weight...",
            "confidence": 0.85
        }
    """
    
    perspective_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this perspective"
    )
    
    session_id: UUID = Field(
        ...,
        description="Session this perspective belongs to"
    )
    
    voice_type: VoiceType = Field(
        ...,
        description="Type of voice contributing this perspective"
    )
    
    voice_name: str = Field(
        ...,
        description="Specific name of the voice (e.g., 'Ethicist', 'Skeptic')",
        examples=["Ethicist", "Skeptic", "Visionary", "Pragmatist"]
    )
    
    round: int = Field(
        default=1,
        description="Deliberation round when this perspective was given",
        ge=1
    )
    
    position: str = Field(
        ...,
        description="The voice's position or stance on the topic"
    )
    
    reasoning: str = Field(
        ...,
        description="Explanation and justification for the position"
    )
    
    confidence: float = Field(
        default=0.5,
        description="Confidence level in this position (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    
    references_perspectives: List[UUID] = Field(
        default_factory=list,
        description="IDs of perspectives this one responds to"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this perspective was contributed"
    )


class CouncilVote(BaseModel):
    """
    A vote cast by a voice on another's perspective.
    
    Votes help determine consensus and identify areas of agreement
    or contention that need further deliberation.
    
    Example:
        {
            "vote_id": "550e8400-e29b-41d4-a716-446655440002",
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "perspective_id": "550e8400-e29b-41d4-a716-446655440001",
            "voter_voice": "Pragmatist",
            "vote_type": "agree",
            "weight": 1.0,
            "comment": "Solid ethical framework"
        }
    """
    
    vote_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this vote"
    )
    
    session_id: UUID = Field(
        ...,
        description="Session this vote belongs to"
    )
    
    perspective_id: UUID = Field(
        ...,
        description="Perspective being voted on"
    )
    
    voter_voice_type: VoiceType = Field(
        ...,
        description="Type of voice casting this vote"
    )
    
    voter_voice_name: str = Field(
        ...,
        description="Name of the voice casting this vote"
    )
    
    vote_type: VoteType = Field(
        ...,
        description="Type of vote cast"
    )
    
    weight: float = Field(
        default=1.0,
        description="Weight of this vote (for weighted consensus)",
        ge=0.0,
        le=2.0
    )
    
    comment: Optional[str] = Field(
        default=None,
        description="Optional comment explaining the vote"
    )
    
    amendment: Optional[str] = Field(
        default=None,
        description="Proposed amendment if vote_type is 'amend'"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this vote was cast"
    )


# =============================================================================
# COMPOSITE MODELS (for API responses)
# =============================================================================

class CouncilSessionFull(BaseModel):
    """Full session with all perspectives and votes."""
    
    session: CouncilSession
    perspectives: List[CouncilPerspective] = Field(default_factory=list)
    votes: List[CouncilVote] = Field(default_factory=list)


class SessionSummary(BaseModel):
    """Lightweight session summary for listings."""
    
    session_id: UUID
    topic: str
    status: SessionStatus
    perspective_count: int
    current_round: int
    created_at: datetime
    resolved_at: Optional[datetime]
