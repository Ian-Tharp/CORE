-- Migration 004: Run Persistence
-- Stores CORE engine runs in database for resumption and history

-- Create runs table
CREATE TABLE IF NOT EXISTS core_runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID,
    user_id VARCHAR(255),
    
    -- Input
    user_input TEXT NOT NULL,
    
    -- State (serialized JSON)
    state JSONB NOT NULL DEFAULT '{}',
    
    -- Status
    status VARCHAR(50) NOT NULL DEFAULT 'pending',  -- pending, running, completed, failed, cancelled
    current_node VARCHAR(100) DEFAULT 'START',
    
    -- Results
    response TEXT,
    error TEXT,
    
    -- Metadata
    config JSONB DEFAULT '{}',
    execution_history JSONB DEFAULT '[]',
    step_results JSONB DEFAULT '[]',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Indexes
    CONSTRAINT valid_status CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled'))
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_core_runs_user_id ON core_runs(user_id);
CREATE INDEX IF NOT EXISTS idx_core_runs_status ON core_runs(status);
CREATE INDEX IF NOT EXISTS idx_core_runs_created_at ON core_runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_core_runs_conversation ON core_runs(conversation_id) WHERE conversation_id IS NOT NULL;

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_run_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update timestamp
DROP TRIGGER IF EXISTS tr_core_runs_updated ON core_runs;
CREATE TRIGGER tr_core_runs_updated
    BEFORE UPDATE ON core_runs
    FOR EACH ROW
    EXECUTE FUNCTION update_run_timestamp();

-- Table for run events (for detailed logging)
CREATE TABLE IF NOT EXISTS core_run_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES core_runs(run_id) ON DELETE CASCADE,
    
    event_type VARCHAR(100) NOT NULL,  -- node_start, node_complete, step_executed, error, etc.
    event_data JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_run_events_run_id ON core_run_events(run_id);
CREATE INDEX IF NOT EXISTS idx_run_events_created ON core_run_events(created_at DESC);

-- Comments
COMMENT ON TABLE core_runs IS 'Stores CORE engine execution runs for persistence and history';
COMMENT ON TABLE core_run_events IS 'Detailed event log for each run, useful for debugging and replay';
COMMENT ON COLUMN core_runs.state IS 'Full COREState serialized as JSON';
COMMENT ON COLUMN core_runs.step_results IS 'Array of StepResult objects from reasoning phase';
