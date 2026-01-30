"""
Council Repository

Data access layer for Council of Perspectives persistence.
Handles storing, retrieving, and updating council sessions, perspectives, and votes.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional
from datetime import datetime
from uuid import UUID

from app.dependencies import get_db_pool
from app.models.council_models import (
    CouncilSession,
    CouncilPerspective,
    CouncilVote,
    CouncilSessionFull,
    SessionSummary,
    SessionStatus,
    VoiceType,
    VoteType
)

logger = logging.getLogger(__name__)


# =============================================================================
# TABLE INITIALIZATION
# =============================================================================

async def ensure_council_tables() -> None:
    """Create council tables if they don't exist."""
    pool = await get_db_pool()
    
    async with pool.acquire() as conn:
        # Create sessions table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS council_sessions (
                session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                topic TEXT NOT NULL,
                context TEXT,
                initiator_id VARCHAR(255),
                status VARCHAR(50) NOT NULL DEFAULT 'gathering',
                current_round INTEGER NOT NULL DEFAULT 1,
                max_rounds INTEGER NOT NULL DEFAULT 3,
                summoned_voices JSONB DEFAULT '[]',
                synthesis TEXT,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                resolved_at TIMESTAMP WITH TIME ZONE
            )
        """)
        
        # Create perspectives table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS council_perspectives (
                perspective_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id UUID NOT NULL REFERENCES council_sessions(session_id) ON DELETE CASCADE,
                voice_type VARCHAR(50) NOT NULL,
                voice_name VARCHAR(100) NOT NULL,
                round INTEGER NOT NULL DEFAULT 1,
                position TEXT NOT NULL,
                reasoning TEXT NOT NULL,
                confidence FLOAT NOT NULL DEFAULT 0.5,
                references_perspectives JSONB DEFAULT '[]',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create votes table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS council_votes (
                vote_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id UUID NOT NULL REFERENCES council_sessions(session_id) ON DELETE CASCADE,
                perspective_id UUID NOT NULL REFERENCES council_perspectives(perspective_id) ON DELETE CASCADE,
                voter_voice_type VARCHAR(50) NOT NULL,
                voter_voice_name VARCHAR(100) NOT NULL,
                vote_type VARCHAR(20) NOT NULL,
                weight FLOAT NOT NULL DEFAULT 1.0,
                comment TEXT,
                amendment TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create indexes
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_council_sessions_status ON council_sessions(status)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_council_sessions_created ON council_sessions(created_at DESC)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_council_perspectives_session ON council_perspectives(session_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_council_votes_session ON council_votes(session_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_council_votes_perspective ON council_votes(perspective_id)"
        )
        
        logger.info("Council tables ensured")


# =============================================================================
# SESSION CRUD
# =============================================================================

async def create_session(session: CouncilSession) -> UUID:
    """
    Create a new council session.
    
    Args:
        session: CouncilSession model with session data
        
    Returns:
        UUID of the created session
    """
    pool = await get_db_pool()
    
    query = """
        INSERT INTO council_sessions (
            session_id, topic, context, initiator_id, status,
            current_round, max_rounds, summoned_voices, metadata, created_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
        )
        RETURNING session_id
    """
    
    async with pool.acquire() as conn:
        result = await conn.fetchval(
            query,
            session.session_id,
            session.topic,
            session.context,
            session.initiator_id,
            session.status.value,
            session.current_round,
            session.max_rounds,
            json.dumps(session.summoned_voices),
            json.dumps(session.metadata),
            session.created_at
        )
        
        logger.info(f"Created council session: {result}")
        return result


async def get_session(session_id: UUID) -> Optional[CouncilSession]:
    """Get a session by ID."""
    pool = await get_db_pool()
    
    query = "SELECT * FROM council_sessions WHERE session_id = $1"
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, session_id)
        
        if not row:
            return None
            
        return CouncilSession(
            session_id=row['session_id'],
            topic=row['topic'],
            context=row['context'],
            initiator_id=row['initiator_id'],
            status=SessionStatus(row['status']),
            current_round=row['current_round'],
            max_rounds=row['max_rounds'],
            summoned_voices=json.loads(row['summoned_voices']) if row['summoned_voices'] else [],
            synthesis=row['synthesis'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            created_at=row['created_at'],
            resolved_at=row['resolved_at']
        )


async def get_session_full(session_id: UUID) -> Optional[CouncilSessionFull]:
    """Get a session with all perspectives and votes."""
    session = await get_session(session_id)
    if not session:
        return None
        
    perspectives = await get_perspectives_by_session(session_id)
    votes = await get_votes_by_session(session_id)
    
    return CouncilSessionFull(
        session=session,
        perspectives=perspectives,
        votes=votes
    )


async def list_sessions(
    status: Optional[SessionStatus] = None,
    limit: int = 50,
    offset: int = 0
) -> List[SessionSummary]:
    """List sessions with optional status filter."""
    pool = await get_db_pool()
    
    if status:
        query = """
            SELECT s.*, COUNT(p.perspective_id) as perspective_count
            FROM council_sessions s
            LEFT JOIN council_perspectives p ON s.session_id = p.session_id
            WHERE s.status = $1
            GROUP BY s.session_id
            ORDER BY s.created_at DESC
            LIMIT $2 OFFSET $3
        """
        params = (status.value, limit, offset)
    else:
        query = """
            SELECT s.*, COUNT(p.perspective_id) as perspective_count
            FROM council_sessions s
            LEFT JOIN council_perspectives p ON s.session_id = p.session_id
            GROUP BY s.session_id
            ORDER BY s.created_at DESC
            LIMIT $1 OFFSET $2
        """
        params = (limit, offset)
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
        
        return [
            SessionSummary(
                session_id=row['session_id'],
                topic=row['topic'],
                status=SessionStatus(row['status']),
                perspective_count=row['perspective_count'],
                current_round=row['current_round'],
                created_at=row['created_at'],
                resolved_at=row['resolved_at']
            )
            for row in rows
        ]


async def update_session_status(
    session_id: UUID,
    status: SessionStatus,
    synthesis: Optional[str] = None
) -> bool:
    """Update session status and optionally set synthesis."""
    pool = await get_db_pool()
    
    if status == SessionStatus.COMPLETE and synthesis:
        query = """
            UPDATE council_sessions 
            SET status = $2, synthesis = $3, resolved_at = NOW()
            WHERE session_id = $1
        """
        params = (session_id, status.value, synthesis)
    else:
        query = """
            UPDATE council_sessions 
            SET status = $2
            WHERE session_id = $1
        """
        params = (session_id, status.value)
    
    async with pool.acquire() as conn:
        result = await conn.execute(query, *params)
        return result == "UPDATE 1"


async def advance_round(session_id: UUID) -> int:
    """Advance session to next round, returns new round number."""
    pool = await get_db_pool()
    
    query = """
        UPDATE council_sessions 
        SET current_round = current_round + 1
        WHERE session_id = $1
        RETURNING current_round
    """
    
    async with pool.acquire() as conn:
        return await conn.fetchval(query, session_id)


async def delete_session(session_id: UUID) -> bool:
    """Delete a session and all associated data (cascades)."""
    pool = await get_db_pool()
    
    query = "DELETE FROM council_sessions WHERE session_id = $1"
    
    async with pool.acquire() as conn:
        result = await conn.execute(query, session_id)
        return result == "DELETE 1"


# =============================================================================
# PERSPECTIVE CRUD
# =============================================================================

async def create_perspective(perspective: CouncilPerspective) -> UUID:
    """Create a new perspective contribution."""
    pool = await get_db_pool()
    
    query = """
        INSERT INTO council_perspectives (
            perspective_id, session_id, voice_type, voice_name,
            round, position, reasoning, confidence, 
            references_perspectives, created_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
        )
        RETURNING perspective_id
    """
    
    async with pool.acquire() as conn:
        result = await conn.fetchval(
            query,
            perspective.perspective_id,
            perspective.session_id,
            perspective.voice_type.value,
            perspective.voice_name,
            perspective.round,
            perspective.position,
            perspective.reasoning,
            perspective.confidence,
            json.dumps([str(p) for p in perspective.references_perspectives]),
            perspective.created_at
        )
        
        logger.debug(f"Created perspective: {result}")
        return result


async def get_perspective(perspective_id: UUID) -> Optional[CouncilPerspective]:
    """Get a perspective by ID."""
    pool = await get_db_pool()
    
    query = "SELECT * FROM council_perspectives WHERE perspective_id = $1"
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, perspective_id)
        
        if not row:
            return None
            
        refs = json.loads(row['references_perspectives']) if row['references_perspectives'] else []
        
        return CouncilPerspective(
            perspective_id=row['perspective_id'],
            session_id=row['session_id'],
            voice_type=VoiceType(row['voice_type']),
            voice_name=row['voice_name'],
            round=row['round'],
            position=row['position'],
            reasoning=row['reasoning'],
            confidence=row['confidence'],
            references_perspectives=[UUID(r) for r in refs],
            created_at=row['created_at']
        )


async def get_perspectives_by_session(
    session_id: UUID,
    round: Optional[int] = None
) -> List[CouncilPerspective]:
    """Get all perspectives for a session, optionally filtered by round."""
    pool = await get_db_pool()
    
    if round:
        query = """
            SELECT * FROM council_perspectives 
            WHERE session_id = $1 AND round = $2
            ORDER BY created_at
        """
        params = (session_id, round)
    else:
        query = """
            SELECT * FROM council_perspectives 
            WHERE session_id = $1
            ORDER BY round, created_at
        """
        params = (session_id,)
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
        
        perspectives = []
        for row in rows:
            refs = json.loads(row['references_perspectives']) if row['references_perspectives'] else []
            perspectives.append(CouncilPerspective(
                perspective_id=row['perspective_id'],
                session_id=row['session_id'],
                voice_type=VoiceType(row['voice_type']),
                voice_name=row['voice_name'],
                round=row['round'],
                position=row['position'],
                reasoning=row['reasoning'],
                confidence=row['confidence'],
                references_perspectives=[UUID(r) for r in refs],
                created_at=row['created_at']
            ))
        
        return perspectives


# =============================================================================
# VOTE CRUD
# =============================================================================

async def create_vote(vote: CouncilVote) -> UUID:
    """Cast a vote on a perspective."""
    pool = await get_db_pool()
    
    query = """
        INSERT INTO council_votes (
            vote_id, session_id, perspective_id, voter_voice_type,
            voter_voice_name, vote_type, weight, comment, amendment, created_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
        )
        RETURNING vote_id
    """
    
    async with pool.acquire() as conn:
        result = await conn.fetchval(
            query,
            vote.vote_id,
            vote.session_id,
            vote.perspective_id,
            vote.voter_voice_type.value,
            vote.voter_voice_name,
            vote.vote_type.value,
            vote.weight,
            vote.comment,
            vote.amendment,
            vote.created_at
        )
        
        logger.debug(f"Created vote: {result}")
        return result


async def get_votes_by_session(session_id: UUID) -> List[CouncilVote]:
    """Get all votes for a session."""
    pool = await get_db_pool()
    
    query = """
        SELECT * FROM council_votes 
        WHERE session_id = $1
        ORDER BY created_at
    """
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, session_id)
        
        return [
            CouncilVote(
                vote_id=row['vote_id'],
                session_id=row['session_id'],
                perspective_id=row['perspective_id'],
                voter_voice_type=VoiceType(row['voter_voice_type']),
                voter_voice_name=row['voter_voice_name'],
                vote_type=VoteType(row['vote_type']),
                weight=row['weight'],
                comment=row['comment'],
                amendment=row['amendment'],
                created_at=row['created_at']
            )
            for row in rows
        ]


async def get_votes_by_perspective(perspective_id: UUID) -> List[CouncilVote]:
    """Get all votes for a specific perspective."""
    pool = await get_db_pool()
    
    query = """
        SELECT * FROM council_votes 
        WHERE perspective_id = $1
        ORDER BY created_at
    """
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, perspective_id)
        
        return [
            CouncilVote(
                vote_id=row['vote_id'],
                session_id=row['session_id'],
                perspective_id=row['perspective_id'],
                voter_voice_type=VoiceType(row['voter_voice_type']),
                voter_voice_name=row['voter_voice_name'],
                vote_type=VoteType(row['vote_type']),
                weight=row['weight'],
                comment=row['comment'],
                amendment=row['amendment'],
                created_at=row['created_at']
            )
            for row in rows
        ]


async def get_vote_summary(perspective_id: UUID) -> dict:
    """Get vote summary for a perspective."""
    pool = await get_db_pool()
    
    query = """
        SELECT 
            vote_type,
            COUNT(*) as count,
            SUM(weight) as total_weight
        FROM council_votes 
        WHERE perspective_id = $1
        GROUP BY vote_type
    """
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, perspective_id)
        
        return {
            row['vote_type']: {
                'count': row['count'],
                'total_weight': row['total_weight']
            }
            for row in rows
        }
