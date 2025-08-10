-- CORE Database Initialization
-- This file sets up the initial database structure for the CORE system

-- Create core application database
CREATE DATABASE IF NOT EXISTS core_db;

-- Switch to core database
\c core_db;

-- Create conversations table for chat history
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(255) UNIQUE NOT NULL,
    title VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create messages table for conversation messages
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
);

-- Create agents table for CORE agent states
CREATE TABLE IF NOT EXISTS agents (
    id SERIAL PRIMARY KEY,
    agent_type VARCHAR(100) NOT NULL,
    agent_name VARCHAR(255) NOT NULL,
    status VARCHAR(50) DEFAULT 'idle',
    configuration JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create system_metrics table for monitoring
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(255) NOT NULL,
    metric_value NUMERIC,
    metric_unit VARCHAR(50),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_agents_type_status ON agents(agent_type, status);
CREATE INDEX IF NOT EXISTS idx_system_metrics_name_timestamp ON system_metrics(metric_name, timestamp);

-- Insert initial CORE agents
INSERT INTO agents (agent_type, agent_name, status, configuration) VALUES
('comprehension', 'ComprehensionAgent', 'idle', '{"model": "gpt-4o-mini", "capabilities": ["text_analysis", "intent_classification"]}'),
('orchestration', 'OrchestrationAgent', 'idle', '{"model": "gpt-4o-mini", "capabilities": ["task_planning", "workflow_management"]}'),
('reasoning', 'ReasoningAgent', 'idle', '{"model": "gpt-4o-mini", "capabilities": ["logical_reasoning", "problem_solving"]}'),
('evaluation', 'EvaluationAgent', 'idle', '{"model": "gpt-4o-mini", "capabilities": ["quality_assessment", "outcome_evaluation"]}');

-- Grant permissions to core_user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO core_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO core_user;