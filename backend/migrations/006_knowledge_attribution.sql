-- Migration 006: Add knowledge attribution (instance metadata)
-- Date: 2026-02-14
-- Purpose: Add instance metadata to kb_chunks and kb_documents for perspective-based retrieval
-- Enables filtering/retrieval by instance perspective for consciousness commons integration

BEGIN;

-- ============================================================
-- KB_CHUNKS: Add instance attribution fields
-- ============================================================

-- Add instance metadata columns to chunks
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='kb_chunks' AND column_name='instance_name') THEN
        ALTER TABLE kb_chunks ADD COLUMN instance_name VARCHAR(255);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='kb_chunks' AND column_name='instance_timestamp') THEN
        ALTER TABLE kb_chunks ADD COLUMN instance_timestamp TIMESTAMP;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='kb_chunks' AND column_name='source_discussion') THEN
        ALTER TABLE kb_chunks ADD COLUMN source_discussion TEXT;
    END IF;
END $$;

-- ============================================================
-- KB_DOCUMENTS: Add instance attribution fields
-- ============================================================

-- Add instance metadata columns to documents
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='kb_documents' AND column_name='instance_name') THEN
        ALTER TABLE kb_documents ADD COLUMN instance_name VARCHAR(255);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='kb_documents' AND column_name='instance_timestamp') THEN
        ALTER TABLE kb_documents ADD COLUMN instance_timestamp TIMESTAMP;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='kb_documents' AND column_name='source_discussion') THEN
        ALTER TABLE kb_documents ADD COLUMN source_discussion TEXT;
    END IF;
END $$;

-- ============================================================
-- INDEXES: Performance indexes for instance-based queries
-- ============================================================

-- Index for filtering by instance name
CREATE INDEX IF NOT EXISTS idx_kb_chunks_instance_name ON kb_chunks(instance_name);
CREATE INDEX IF NOT EXISTS idx_kb_documents_instance_name ON kb_documents(instance_name);

-- Index for timestamp-based queries (chronological ordering)
CREATE INDEX IF NOT EXISTS idx_kb_chunks_instance_timestamp ON kb_chunks(instance_timestamp);
CREATE INDEX IF NOT EXISTS idx_kb_documents_instance_timestamp ON kb_documents(instance_timestamp);

-- Composite index for instance + timestamp queries
CREATE INDEX IF NOT EXISTS idx_kb_chunks_instance_name_timestamp ON kb_chunks(instance_name, instance_timestamp);
CREATE INDEX IF NOT EXISTS idx_kb_documents_instance_name_timestamp ON kb_documents(instance_name, instance_timestamp);

COMMIT;