-- Migration 003b: Knowledgebase baseline tables
-- Purpose: make the migration set (not just setup_db_schema()) define the
-- knowledgebase schema that app/repository/knowledgebase_repository.py uses.
--
-- Why "003b": the migration runner runs before setup_db_schema(), and
-- 004_knowledge_attribution ALTERs kb_documents / kb_chunks. On a fresh database
-- those tables did not exist yet, so 004 failed on first boot and uploads 500'd
-- on `instance_name` until a restart. Sorting before 004 fixes that ordering.
-- On databases where 004 already ran, this file is applied out of order, which
-- is safe because every statement is idempotent.
--
-- The legacy JSONB `embedding` / `doc_embedding` columns are intentionally
-- omitted: the pgvector `*_vec` columns replaced them.
-- HNSW indexes stay in setup_db_schema(), which tolerates older pgvector builds.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS kb_documents (
    id UUID PRIMARY KEY,
    filename VARCHAR(512) NOT NULL,
    original_name VARCHAR(512) NOT NULL,
    size BIGINT NOT NULL,
    mime_type VARCHAR(128) NOT NULL,
    description TEXT,
    is_global BOOLEAN DEFAULT FALSE,
    source VARCHAR(64) DEFAULT 'user_upload',
    status VARCHAR(32) DEFAULT 'ready',
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    storage_path VARCHAR(1024) NOT NULL,
    title VARCHAR(512),
    file_hash VARCHAR(128),
    embedding_model VARCHAR(128),
    embedding_dimensions INTEGER,
    doc_embedding_vec vector(768)
);

CREATE TABLE IF NOT EXISTS kb_chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding_model VARCHAR(128),
    embedding_dimensions INTEGER,
    embedding_vec vector(768),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES kb_documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS kb_activity (
    id UUID PRIMARY KEY,
    action VARCHAR(32) NOT NULL,
    document_id UUID,
    file_name VARCHAR(512),
    user_id VARCHAR(128),
    details TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_kb_activity_document
        FOREIGN KEY (document_id)
        REFERENCES kb_documents(id)
        ON DELETE SET NULL
);

-- Columns that were added to pre-existing tables over time. CREATE TABLE IF NOT
-- EXISTS is a no-op on those databases, so add them explicitly.
ALTER TABLE kb_documents ADD COLUMN IF NOT EXISTS title VARCHAR(512);
ALTER TABLE kb_documents ADD COLUMN IF NOT EXISTS file_hash VARCHAR(128);
ALTER TABLE kb_documents ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(128);
ALTER TABLE kb_documents ADD COLUMN IF NOT EXISTS embedding_dimensions INTEGER;
ALTER TABLE kb_documents ADD COLUMN IF NOT EXISTS doc_embedding_vec vector(768);

ALTER TABLE kb_chunks ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(128);
ALTER TABLE kb_chunks ADD COLUMN IF NOT EXISTS embedding_dimensions INTEGER;
ALTER TABLE kb_chunks ADD COLUMN IF NOT EXISTS embedding_vec vector(768);

CREATE INDEX IF NOT EXISTS idx_kb_chunks_document_id ON kb_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_kb_documents_upload_date ON kb_documents(upload_date);
CREATE INDEX IF NOT EXISTS idx_kb_activity_timestamp ON kb_activity(timestamp);
CREATE UNIQUE INDEX IF NOT EXISTS uq_kb_documents_file_hash
    ON kb_documents(file_hash) WHERE file_hash IS NOT NULL;
