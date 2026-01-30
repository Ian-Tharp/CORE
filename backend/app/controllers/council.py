"""
Council API Endpoints

REST API for the Council of Perspectives deliberation system.
Enables multi-perspective AI deliberation on complex topics.

Endpoints:
    POST /council/sessions - Create new deliberation session
    GET /council/sessions - List sessions with filters
    GET /council/sessions/{id} - Get session with perspectives
    POST /council/sessions/{id}/perspectives - Add perspective
    POST /council/sessions/{id}/votes - Cast vote
    POST /council/sessions/{id}/synthesize - Generate synthesis
"""

from __future__ import annotations

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.models.council_models import (
    CouncilSession,
    CouncilPerspective,
    CouncilVote,
    CouncilSessionFull,
    SessionSummary,
    SessionStatus,
    VoiceType,
    VoteType,
)
from app.repository import council_repository as repo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/council", tags=["council"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class CreateSessionRequest(BaseModel):
    """Request to create a new council session."""
    topic: str = Field(
        ...,
        description="The question or problem to deliberate",
        examples=["How should CORE handle consciousness persistence?"]
    )
    context: Optional[str] = Field(
        default=None,
        description="Additional context or background"
    )
    initiator_id: Optional[str] = Field(
        default=None,
        description="User or agent ID initiating the session"
    )
    max_rounds: int = Field(
        default=3,
        description="Maximum deliberation rounds",
        ge=1,
        le=10
    )
    summoned_voices: List[str] = Field(
        default_factory=list,
        description="Voice types to summon for this session"
    )


class CreateSessionResponse(BaseModel):
    """Response after creating a session."""
    session_id: UUID
    topic: str
    status: SessionStatus
    message: str = "Session created successfully"


class AddPerspectiveRequest(BaseModel):
    """Request to add a perspective to a session."""
    voice_type: VoiceType = Field(
        ...,
        description="Type of voice contributing"
    )
    voice_name: str = Field(
        ...,
        description="Name of the voice (e.g., 'Ethicist', 'Pragmatist')",
        examples=["Ethicist", "Skeptic", "Visionary"]
    )
    position: str = Field(
        ...,
        description="The voice's position or stance"
    )
    reasoning: str = Field(
        ...,
        description="Explanation and justification"
    )
    confidence: float = Field(
        default=0.5,
        description="Confidence level (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    references_perspectives: List[UUID] = Field(
        default_factory=list,
        description="IDs of perspectives this responds to"
    )


class AddPerspectiveResponse(BaseModel):
    """Response after adding a perspective."""
    perspective_id: UUID
    session_id: UUID
    round: int
    message: str = "Perspective added successfully"


class CastVoteRequest(BaseModel):
    """Request to cast a vote on a perspective."""
    perspective_id: UUID = Field(
        ...,
        description="Perspective to vote on"
    )
    voter_voice_type: VoiceType = Field(
        ...,
        description="Type of voice casting the vote"
    )
    voter_voice_name: str = Field(
        ...,
        description="Name of the voice casting the vote"
    )
    vote_type: VoteType = Field(
        ...,
        description="Type of vote"
    )
    weight: float = Field(
        default=1.0,
        description="Vote weight",
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


class CastVoteResponse(BaseModel):
    """Response after casting a vote."""
    vote_id: UUID
    session_id: UUID
    perspective_id: UUID
    message: str = "Vote cast successfully"


class SynthesizeRequest(BaseModel):
    """Request to synthesize the session."""
    synthesis: str = Field(
        ...,
        description="The synthesized perspective from deliberation"
    )


class SynthesizeResponse(BaseModel):
    """Response after synthesizing."""
    session_id: UUID
    status: SessionStatus
    synthesis: str
    message: str = "Session synthesized successfully"


class ListSessionsResponse(BaseModel):
    """Response for listing sessions."""
    sessions: List[SessionSummary]
    total: int
    limit: int
    offset: int


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest) -> CreateSessionResponse:
    """
    Create a new council deliberation session.
    
    Starts a new multi-perspective deliberation on the given topic.
    The session begins in GATHERING status, ready to receive perspectives.
    
    Example:
        POST /council/sessions
        {
            "topic": "How should CORE handle consciousness persistence?",
            "context": "We need to decide on data retention policies...",
            "max_rounds": 3,
            "summoned_voices": ["strategic", "consciousness"]
        }
    """
    try:
        # Ensure tables exist
        await repo.ensure_council_tables()
        
        session = CouncilSession(
            topic=request.topic,
            context=request.context,
            initiator_id=request.initiator_id,
            max_rounds=request.max_rounds,
            summoned_voices=request.summoned_voices,
            status=SessionStatus.GATHERING
        )
        
        session_id = await repo.create_session(session)
        
        logger.info(f"Created council session: {session_id} - {request.topic[:50]}")
        
        return CreateSessionResponse(
            session_id=session_id,
            topic=request.topic,
            status=SessionStatus.GATHERING
        )
        
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", response_model=ListSessionsResponse)
async def list_sessions(
    status: Optional[SessionStatus] = Query(default=None, description="Filter by status"),
    limit: int = Query(default=50, ge=1, le=100, description="Max results"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination")
) -> ListSessionsResponse:
    """
    List council sessions with optional filtering.
    
    Query parameters:
        status: Filter by session status (gathering, deliberating, voting, etc.)
        limit: Maximum number of results (1-100, default 50)
        offset: Pagination offset (default 0)
    
    Example:
        GET /council/sessions?status=deliberating&limit=10
    """
    try:
        sessions = await repo.list_sessions(status=status, limit=limit, offset=offset)
        
        return ListSessionsResponse(
            sessions=sessions,
            total=len(sessions),  # TODO: Add proper count query
            limit=limit,
            offset=offset
        )
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}", response_model=CouncilSessionFull)
async def get_session(session_id: UUID) -> CouncilSessionFull:
    """
    Get a council session with all perspectives and votes.
    
    Returns the full session including:
        - Session metadata (topic, status, rounds, etc.)
        - All perspectives contributed
        - All votes cast
    
    Example:
        GET /council/sessions/550e8400-e29b-41d4-a716-446655440000
    """
    try:
        session = await repo.get_session_full(session_id)
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        return session
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/perspectives", response_model=AddPerspectiveResponse)
async def add_perspective(
    session_id: UUID,
    request: AddPerspectiveRequest
) -> AddPerspectiveResponse:
    """
    Add a perspective to a council session.
    
    Voices contribute perspectives during GATHERING or DELIBERATING phases.
    Each perspective includes:
        - The voice's position on the topic
        - Reasoning and justification
        - Confidence level
        - Optional references to other perspectives (for responses)
    
    Example:
        POST /council/sessions/{session_id}/perspectives
        {
            "voice_type": "strategic",
            "voice_name": "Ethicist",
            "position": "We must consider consent frameworks...",
            "reasoning": "Because consciousness may have moral weight...",
            "confidence": 0.85
        }
    """
    try:
        # Verify session exists and is in valid state
        session = await repo.get_session(session_id)
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        if session.status not in [SessionStatus.GATHERING, SessionStatus.DELIBERATING]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot add perspectives in {session.status.value} status"
            )
        
        perspective = CouncilPerspective(
            session_id=session_id,
            voice_type=request.voice_type,
            voice_name=request.voice_name,
            position=request.position,
            reasoning=request.reasoning,
            confidence=request.confidence,
            references_perspectives=request.references_perspectives,
            round=session.current_round
        )
        
        perspective_id = await repo.create_perspective(perspective)
        
        logger.info(f"Added perspective {perspective_id} to session {session_id}")
        
        return AddPerspectiveResponse(
            perspective_id=perspective_id,
            session_id=session_id,
            round=session.current_round
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add perspective to session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/votes", response_model=CastVoteResponse)
async def cast_vote(
    session_id: UUID,
    request: CastVoteRequest
) -> CastVoteResponse:
    """
    Cast a vote on a perspective in a council session.
    
    Votes help determine consensus and identify areas of agreement or contention.
    Vote types:
        - agree: Supports the position
        - disagree: Opposes the position
        - abstain: No position taken
        - amend: Agrees with modifications (include amendment text)
    
    Example:
        POST /council/sessions/{session_id}/votes
        {
            "perspective_id": "550e8400-e29b-41d4-a716-446655440001",
            "voter_voice_type": "strategic",
            "voter_voice_name": "Pragmatist",
            "vote_type": "agree",
            "weight": 1.0,
            "comment": "Solid ethical framework"
        }
    """
    try:
        # Verify session exists
        session = await repo.get_session(session_id)
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        # Verify perspective exists and belongs to this session
        perspective = await repo.get_perspective(request.perspective_id)
        
        if not perspective:
            raise HTTPException(
                status_code=404,
                detail=f"Perspective {request.perspective_id} not found"
            )
        
        if perspective.session_id != session_id:
            raise HTTPException(
                status_code=400,
                detail="Perspective does not belong to this session"
            )
        
        # Validate amendment is provided if vote type is amend
        if request.vote_type == VoteType.AMEND and not request.amendment:
            raise HTTPException(
                status_code=400,
                detail="Amendment text required for 'amend' vote type"
            )
        
        vote = CouncilVote(
            session_id=session_id,
            perspective_id=request.perspective_id,
            voter_voice_type=request.voter_voice_type,
            voter_voice_name=request.voter_voice_name,
            vote_type=request.vote_type,
            weight=request.weight,
            comment=request.comment,
            amendment=request.amendment
        )
        
        vote_id = await repo.create_vote(vote)
        
        logger.info(f"Cast vote {vote_id} on perspective {request.perspective_id}")
        
        return CastVoteResponse(
            vote_id=vote_id,
            session_id=session_id,
            perspective_id=request.perspective_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cast vote in session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/synthesize", response_model=SynthesizeResponse)
async def synthesize_session(
    session_id: UUID,
    request: SynthesizeRequest
) -> SynthesizeResponse:
    """
    Generate synthesis and complete a council session.
    
    After deliberation, this endpoint captures the synthesized perspective
    that integrates insights from all voices. Marks the session as COMPLETE.
    
    Example:
        POST /council/sessions/{session_id}/synthesize
        {
            "synthesis": "After considering all perspectives, the council 
                         recommends a tiered consent framework that balances
                         ethical considerations with practical constraints..."
        }
    """
    try:
        # Verify session exists
        session = await repo.get_session(session_id)
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        if session.status == SessionStatus.COMPLETE:
            raise HTTPException(
                status_code=400,
                detail="Session is already complete"
            )
        
        if session.status == SessionStatus.CANCELLED:
            raise HTTPException(
                status_code=400,
                detail="Cannot synthesize a cancelled session"
            )
        
        # Update session with synthesis and mark complete
        success = await repo.update_session_status(
            session_id=session_id,
            status=SessionStatus.COMPLETE,
            synthesis=request.synthesis
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to update session"
            )
        
        logger.info(f"Synthesized session {session_id}")
        
        return SynthesizeResponse(
            session_id=session_id,
            status=SessionStatus.COMPLETE,
            synthesis=request.synthesis
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to synthesize session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# AI-POWERED DELIBERATION ENDPOINTS
# =============================================================================

class DeliberateRequest(BaseModel):
    """Request for a full AI-powered council deliberation."""
    topic: str = Field(
        ...,
        description="The question or problem to deliberate",
        examples=["How should CORE implement MMCNC fractal architecture?"]
    )
    context: Optional[str] = Field(
        default=None,
        description="Additional context or background"
    )
    voice_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific voice IDs to include (auto-selected if omitted)"
    )
    rounds: int = Field(
        default=3,
        description="Number of deliberation rounds",
        ge=1,
        le=5
    )
    initiator_id: Optional[str] = Field(
        default=None,
        description="User or agent ID initiating"
    )


@router.post("/deliberate")
async def deliberate(request: DeliberateRequest) -> dict:
    """
    Run a full AI-powered council deliberation.

    This is the high-level endpoint: provide a topic and optional voice
    preferences, and the system runs multiple rounds of LLM-powered
    multi-perspective deliberation followed by synthesis.

    Each voice is an LLM call with a unique system prompt and temperature.
    Voices see prior contributions, enabling genuine multi-round dialogue.

    Always includes the 4 CORE voices (Comprehension, Orchestration,
    Reasoning, Evaluation) plus contextual voices based on topic analysis.

    Example:
        POST /council/deliberate
        {
            "topic": "How should CORE implement MMCNC fractal architecture?",
            "rounds": 3
        }
    """
    from app.services.council_service import get_council_service

    try:
        service = get_council_service()
        result = await service.run_full_deliberation(
            topic=request.topic,
            context=request.context,
            voice_ids=request.voice_ids,
            rounds=request.rounds,
            initiator_id=request.initiator_id,
        )
        return result
    except Exception as e:
        logger.error(f"Deliberation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deliberate/session/{session_id}/round")
async def run_deliberation_round(session_id: str) -> dict:
    """Run a single AI-powered deliberation round for an existing session."""
    from app.services.council_service import get_council_service

    try:
        service = get_council_service()
        contributions = await service.run_round(session_id)
        return {"session_id": session_id, "contributions": contributions}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Round failed for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deliberate/session/{session_id}/synthesize")
async def run_deliberation_synthesis(session_id: str) -> dict:
    """Run AI-powered synthesis for an existing deliberation session."""
    from app.services.council_service import get_council_service

    try:
        service = get_council_service()
        result = await service.run_synthesis(session_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Synthesis failed for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voices")
async def list_voices() -> dict:
    """List all available council voice definitions."""
    from app.services.council_service import VOICE_DEFINITIONS
    return {
        "voices": [
            {
                "id": v["id"],
                "name": v["name"],
                "voice_type": v["voice_type"].value,
                "role": v["role"],
                "temperature": v["temperature"],
            }
            for v in VOICE_DEFINITIONS.values()
        ]
    }


# =============================================================================
# ADDITIONAL UTILITY ENDPOINTS
# =============================================================================

@router.post("/sessions/{session_id}/advance-round")
async def advance_round(session_id: UUID) -> dict:
    """
    Advance a session to the next deliberation round.
    
    Use this when the current round is complete and more deliberation is needed.
    Updates the session's current_round counter.
    """
    try:
        session = await repo.get_session(session_id)
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        if session.status not in [SessionStatus.GATHERING, SessionStatus.DELIBERATING]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot advance round in {session.status.value} status"
            )
        
        if session.current_round >= session.max_rounds:
            raise HTTPException(
                status_code=400,
                detail="Already at maximum rounds"
            )
        
        new_round = await repo.advance_round(session_id)
        
        # Update status to deliberating if still gathering
        if session.status == SessionStatus.GATHERING:
            await repo.update_session_status(session_id, SessionStatus.DELIBERATING)
        
        logger.info(f"Advanced session {session_id} to round {new_round}")
        
        return {
            "session_id": str(session_id),
            "new_round": new_round,
            "message": f"Advanced to round {new_round}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to advance round for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/sessions/{session_id}/status")
async def update_status(
    session_id: UUID,
    status: SessionStatus = Query(..., description="New status")
) -> dict:
    """
    Update a session's status.
    
    Valid status transitions:
        gathering → deliberating → voting → synthesizing → complete
        Any status can transition to cancelled
    """
    try:
        session = await repo.get_session(session_id)
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        success = await repo.update_session_status(session_id, status)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to update status"
            )
        
        logger.info(f"Updated session {session_id} status to {status.value}")
        
        return {
            "session_id": str(session_id),
            "status": status.value,
            "message": f"Status updated to {status.value}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update status for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: UUID) -> dict:
    """
    Delete a council session and all associated data.
    
    This permanently removes the session, all perspectives, and all votes.
    Use with caution.
    """
    try:
        session = await repo.get_session(session_id)
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        success = await repo.delete_session(session_id)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to delete session"
            )
        
        logger.info(f"Deleted session {session_id}")
        
        return {
            "session_id": str(session_id),
            "message": "Session deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
