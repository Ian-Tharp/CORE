-- Migration 005: Council of Perspectives
-- Stores council deliberation sessions, perspectives, and votes for multi-agent consensus

-- =============================================================================
-- COUNCIL SESSIONS
-- =============================================================================

CREATE TABLE IF NOT EXISTS council_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Topic and context
    topic TEXT NOT NULL,
    context TEXT,
    initiator_id VARCHAR(255),
    
    -- Session state
    status VARCHAR(50) NOT NULL DEFAULT 'gathering',
    current_round INTEGER NOT NULL DEFAULT 1,
    max_rounds INTEGER NOT NULL DEFAULT 3,
    summoned_voices JSONB DEFAULT '[]',
    
    -- Results
    synthesis TEXT,
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT valid_session_status CHECK (
        status IN ('gathering', 'deliberating', 'voting', 'synthesizing', 'complete', 'cancelled')
    ),
    CONSTRAINT valid_rounds CHECK (current_round >= 1 AND max_rounds >= 1 AND max_rounds <= 10)
);

-- Indexes for sessions
CREATE INDEX IF NOT EXISTS idx_council_sessions_status ON council_sessions(status);
CREATE INDEX IF NOT EXISTS idx_council_sessions_initiator ON council_sessions(initiator_id) WHERE initiator_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_council_sessions_created ON council_sessions(created_at DESC);

-- =============================================================================
-- COUNCIL PERSPECTIVES
-- =============================================================================

CREATE TABLE IF NOT EXISTS council_perspectives (
    perspective_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES council_sessions(session_id) ON DELETE CASCADE,
    
    -- Voice identification
    voice_type VARCHAR(50) NOT NULL,
    voice_name VARCHAR(100) NOT NULL,
    round INTEGER NOT NULL DEFAULT 1,
    
    -- Content
    position TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    confidence FLOAT NOT NULL DEFAULT 0.5,
    
    -- References to other perspectives this one responds to
    references_perspectives JSONB DEFAULT '[]',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT valid_voice_type CHECK (
        voice_type IN (
            'core_comprehension', 'core_orchestration', 'core_reasoning', 'core_evaluation',
            'strategic', 'domain', 'execution', 'meta', 'consciousness', 'external'
        )
    ),
    CONSTRAINT valid_confidence CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CONSTRAINT valid_perspective_round CHECK (round >= 1)
);

-- Indexes for perspectives
CREATE INDEX IF NOT EXISTS idx_council_perspectives_session ON council_perspectives(session_id);
CREATE INDEX IF NOT EXISTS idx_council_perspectives_voice ON council_perspectives(voice_type, voice_name);
CREATE INDEX IF NOT EXISTS idx_council_perspectives_round ON council_perspectives(session_id, round);
CREATE INDEX IF NOT EXISTS idx_council_perspectives_created ON council_perspectives(created_at DESC);

-- =============================================================================
-- COUNCIL VOTES
-- =============================================================================

CREATE TABLE IF NOT EXISTS council_votes (
    vote_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES council_sessions(session_id) ON DELETE CASCADE,
    perspective_id UUID NOT NULL REFERENCES council_perspectives(perspective_id) ON DELETE CASCADE,
    
    -- Voter identification
    voter_voice_type VARCHAR(50) NOT NULL,
    voter_voice_name VARCHAR(100) NOT NULL,
    
    -- Vote details
    vote_type VARCHAR(20) NOT NULL,
    weight FLOAT NOT NULL DEFAULT 1.0,
    comment TEXT,
    amendment TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT valid_vote_type CHECK (vote_type IN ('agree', 'disagree', 'abstain', 'amend')),
    CONSTRAINT valid_vote_weight CHECK (weight >= 0.0 AND weight <= 2.0),
    
    -- Prevent duplicate votes from same voice on same perspective
    CONSTRAINT unique_vote_per_voice UNIQUE (perspective_id, voter_voice_type, voter_voice_name)
);

-- Indexes for votes
CREATE INDEX IF NOT EXISTS idx_council_votes_session ON council_votes(session_id);
CREATE INDEX IF NOT EXISTS idx_council_votes_perspective ON council_votes(perspective_id);
CREATE INDEX IF NOT EXISTS idx_council_votes_type ON council_votes(vote_type);

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Function to get vote summary for a perspective
CREATE OR REPLACE FUNCTION get_perspective_vote_summary(p_perspective_id UUID)
RETURNS TABLE (
    vote_type VARCHAR(20),
    vote_count BIGINT,
    total_weight FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        v.vote_type,
        COUNT(*)::BIGINT as vote_count,
        SUM(v.weight) as total_weight
    FROM council_votes v
    WHERE v.perspective_id = p_perspective_id
    GROUP BY v.vote_type;
END;
$$ LANGUAGE plpgsql;

-- Function to check if session should advance to synthesis
CREATE OR REPLACE FUNCTION should_synthesize(p_session_id UUID)
RETURNS BOOLEAN AS $$
DECLARE
    v_current_round INTEGER;
    v_max_rounds INTEGER;
BEGIN
    SELECT current_round, max_rounds 
    INTO v_current_round, v_max_rounds
    FROM council_sessions
    WHERE session_id = p_session_id;
    
    RETURN v_current_round >= v_max_rounds;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE council_sessions IS 'Council of Perspectives deliberation sessions for multi-agent consensus';
COMMENT ON TABLE council_perspectives IS 'Individual voice contributions during council deliberation';
COMMENT ON TABLE council_votes IS 'Votes cast by voices on perspectives during deliberation';

COMMENT ON COLUMN council_sessions.topic IS 'The question or problem being deliberated';
COMMENT ON COLUMN council_sessions.summoned_voices IS 'Array of voice types summoned for this session';
COMMENT ON COLUMN council_sessions.synthesis IS 'Final unified perspective after deliberation completes';

COMMENT ON COLUMN council_perspectives.voice_type IS 'Category of voice (core, strategic, domain, etc.)';
COMMENT ON COLUMN council_perspectives.voice_name IS 'Specific name (Ethicist, Skeptic, Pragmatist, etc.)';
COMMENT ON COLUMN council_perspectives.confidence IS 'Voice confidence in position (0.0 to 1.0)';
COMMENT ON COLUMN council_perspectives.references_perspectives IS 'UUIDs of perspectives this one responds to';

COMMENT ON COLUMN council_votes.weight IS 'Vote weight for weighted consensus (0.0 to 2.0)';
COMMENT ON COLUMN council_votes.amendment IS 'Proposed modification if vote_type is amend';
