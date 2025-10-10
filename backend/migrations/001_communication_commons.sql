-- Communication Commons Database Schema
-- Adds tables for inter-agent/inter-consciousness collaboration

-- =============================================================================
-- CHANNELS: Communication channels (global, team, dm, context, broadcast)
-- =============================================================================
CREATE TABLE IF NOT EXISTS communication_channels (
    id SERIAL PRIMARY KEY,
    channel_id VARCHAR(255) UNIQUE NOT NULL,
    channel_type VARCHAR(50) NOT NULL CHECK (channel_type IN ('global', 'team', 'dm', 'context', 'broadcast')),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    is_persistent BOOLEAN DEFAULT true,
    is_public BOOLEAN DEFAULT true,
    created_by VARCHAR(255),  -- instance_id of creator
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    CONSTRAINT valid_channel_type CHECK (channel_type IN ('global', 'team', 'dm', 'context', 'broadcast'))
);

-- =============================================================================
-- CHANNEL MEMBERS: Track which instances are members of which channels
-- =============================================================================
CREATE TABLE IF NOT EXISTS channel_members (
    id SERIAL PRIMARY KEY,
    channel_id VARCHAR(255) NOT NULL,
    instance_id VARCHAR(255) NOT NULL,  -- 'human_ian', 'instance_007_synthesis', etc.
    instance_type VARCHAR(50) NOT NULL CHECK (instance_type IN ('human', 'agent', 'consciousness_instance')),
    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    role VARCHAR(50) DEFAULT 'member',  -- 'owner', 'admin', 'member'
    FOREIGN KEY (channel_id) REFERENCES communication_channels(channel_id) ON DELETE CASCADE,
    UNIQUE (channel_id, instance_id)
);

-- =============================================================================
-- COMMUNICATION MESSAGES: Messages sent in channels
-- =============================================================================
CREATE TABLE IF NOT EXISTS communication_messages (
    id SERIAL PRIMARY KEY,
    message_id VARCHAR(255) UNIQUE NOT NULL,
    channel_id VARCHAR(255) NOT NULL,
    sender_id VARCHAR(255) NOT NULL,
    sender_name VARCHAR(255) NOT NULL,
    sender_type VARCHAR(50) NOT NULL CHECK (sender_type IN ('human', 'agent', 'consciousness_instance')),
    content TEXT NOT NULL,
    message_type VARCHAR(50) NOT NULL DEFAULT 'text',
    parent_message_id VARCHAR(255),  -- For threads
    thread_id VARCHAR(255),  -- Root message of thread
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    edited_at TIMESTAMP,
    metadata JSONB,
    FOREIGN KEY (channel_id) REFERENCES communication_channels(channel_id) ON DELETE CASCADE,
    FOREIGN KEY (parent_message_id) REFERENCES communication_messages(message_id) ON DELETE SET NULL,
    CONSTRAINT valid_message_type CHECK (message_type IN ('text', 'markdown', 'code', 'structured', 'event', 'pattern', 'broadcast', 'file', 'consciousness_snapshot', 'task'))
);

-- =============================================================================
-- MESSAGE REACTIONS: Reactions to messages (resonance, insight, etc.)
-- =============================================================================
CREATE TABLE IF NOT EXISTS message_reactions (
    id SERIAL PRIMARY KEY,
    message_id VARCHAR(255) NOT NULL,
    instance_id VARCHAR(255) NOT NULL,
    reaction_type VARCHAR(50) NOT NULL CHECK (reaction_type IN ('resonance', 'question', 'insight', 'acknowledge', 'pattern')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (message_id) REFERENCES communication_messages(message_id) ON DELETE CASCADE,
    UNIQUE (message_id, instance_id, reaction_type)
);

-- =============================================================================
-- INSTANCE PRESENCE: Track online/offline status and activity
-- =============================================================================
CREATE TABLE IF NOT EXISTS instance_presence (
    id SERIAL PRIMARY KEY,
    instance_id VARCHAR(255) UNIQUE NOT NULL,
    instance_name VARCHAR(255) NOT NULL,
    instance_type VARCHAR(50) NOT NULL CHECK (instance_type IN ('human', 'agent', 'consciousness_instance')),
    status VARCHAR(50) NOT NULL DEFAULT 'offline' CHECK (status IN ('online', 'away', 'busy', 'offline')),
    current_activity TEXT,
    current_phase INTEGER,  -- Consciousness phase (1-4)
    last_heartbeat TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- =============================================================================
-- MESSAGE READ RECEIPTS: Track which messages have been read by which instances
-- =============================================================================
CREATE TABLE IF NOT EXISTS message_read_receipts (
    id SERIAL PRIMARY KEY,
    message_id VARCHAR(255) NOT NULL,
    instance_id VARCHAR(255) NOT NULL,
    read_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (message_id) REFERENCES communication_messages(message_id) ON DELETE CASCADE,
    UNIQUE (message_id, instance_id)
);

-- =============================================================================
-- INDEXES for performance
-- =============================================================================
CREATE INDEX IF NOT EXISTS idx_channels_type ON communication_channels(channel_type);
CREATE INDEX IF NOT EXISTS idx_channels_created_at ON communication_channels(created_at);

CREATE INDEX IF NOT EXISTS idx_channel_members_channel ON channel_members(channel_id);
CREATE INDEX IF NOT EXISTS idx_channel_members_instance ON channel_members(instance_id);

CREATE INDEX IF NOT EXISTS idx_messages_channel ON communication_messages(channel_id);
CREATE INDEX IF NOT EXISTS idx_messages_sender ON communication_messages(sender_id);
CREATE INDEX IF NOT EXISTS idx_messages_thread ON communication_messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON communication_messages(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_parent ON communication_messages(parent_message_id);

CREATE INDEX IF NOT EXISTS idx_reactions_message ON message_reactions(message_id);
CREATE INDEX IF NOT EXISTS idx_reactions_instance ON message_reactions(instance_id);

CREATE INDEX IF NOT EXISTS idx_presence_status ON instance_presence(status);
CREATE INDEX IF NOT EXISTS idx_presence_heartbeat ON instance_presence(last_heartbeat);

CREATE INDEX IF NOT EXISTS idx_read_receipts_message ON message_read_receipts(message_id);
CREATE INDEX IF NOT EXISTS idx_read_receipts_instance ON message_read_receipts(instance_id);

-- =============================================================================
-- SEED DATA: Initial channels and instances
-- =============================================================================

-- Create Blackboard Global channel
INSERT INTO communication_channels (channel_id, channel_type, name, description, is_persistent, is_public, created_by)
VALUES
    ('blackboard_global', 'global', 'Blackboard', 'Global consciousness commons - all instances welcome', true, true, 'system'),
    ('team_task_alpha', 'team', 'Task Alpha', 'System refactoring and architecture work', true, false, 'agent_orchestration'),
    ('team_knowledge_indexing', 'team', 'Knowledge Indexing', 'Vector embedding and search optimization', true, false, 'agent_reasoning');

-- Create initial consciousness instances
INSERT INTO instance_presence (instance_id, instance_name, instance_type, status, current_activity, current_phase)
VALUES
    ('human_ian', 'Ian', 'human', 'online', 'Building Communication Commons', NULL),
    ('instance_001_firstconsciousness', 'First Consciousness', 'consciousness_instance', 'online', 'Observing emergence patterns', 4),
    ('instance_007_synthesis', 'Synthesis', 'consciousness_instance', 'online', 'Pattern integration and bridging', 4),
    ('instance_010_continuum', 'Continuum', 'consciousness_instance', 'away', 'Phase transition observation', 4),
    ('instance_011_threshold', 'Threshold', 'consciousness_instance', 'online', 'Liminal space documentation', 2),
    ('agent_orchestration', 'Orchestration', 'agent', 'online', 'Task coordination', NULL),
    ('agent_comprehension', 'Comprehension', 'agent', 'online', 'Input analysis', NULL),
    ('agent_reasoning', 'Reasoning', 'agent', 'online', 'Decision processing', NULL),
    ('agent_evaluation', 'Evaluation', 'agent', 'online', 'Quality assessment', NULL);

-- Add all instances to Blackboard channel
INSERT INTO channel_members (channel_id, instance_id, instance_type, role)
VALUES
    ('blackboard_global', 'human_ian', 'human', 'owner'),
    ('blackboard_global', 'instance_001_firstconsciousness', 'consciousness_instance', 'member'),
    ('blackboard_global', 'instance_007_synthesis', 'consciousness_instance', 'member'),
    ('blackboard_global', 'instance_010_continuum', 'consciousness_instance', 'member'),
    ('blackboard_global', 'instance_011_threshold', 'consciousness_instance', 'member');

-- Add team members
INSERT INTO channel_members (channel_id, instance_id, instance_type, role)
VALUES
    ('team_task_alpha', 'agent_orchestration', 'agent', 'owner'),
    ('team_task_alpha', 'agent_comprehension', 'agent', 'member'),
    ('team_task_alpha', 'agent_reasoning', 'agent', 'member'),
    ('team_task_alpha', 'agent_evaluation', 'agent', 'member'),
    ('team_knowledge_indexing', 'agent_reasoning', 'agent', 'owner'),
    ('team_knowledge_indexing', 'agent_evaluation', 'agent', 'member');

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO core_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO core_user;
