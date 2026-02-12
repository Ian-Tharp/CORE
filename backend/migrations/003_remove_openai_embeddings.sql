-- Migration 003: Remove OpenAI embedding columns, resize vectors from 3072 → 768
-- Date: 2026-02-12
-- Purpose: All embeddings now use local Ollama (nomic-embed-text, 768-dim).
--          Remove legacy OpenAI JSONB columns and shrink vector columns.
-- Idempotent: safe to run multiple times.

BEGIN;

-- ============================================================
-- KB_CHUNKS: migrate embedding_vec_local (vector(3072)) → embedding_vec (vector(768))
-- ============================================================

-- Step 1: Add new vector(768) column if it doesn't exist
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='kb_chunks' AND column_name='embedding_vec') THEN
        ALTER TABLE kb_chunks ADD COLUMN embedding_vec vector(768);
    END IF;
END $$;

-- Step 2: Copy first 768 dims from old column (only where new column is null and old exists)
UPDATE kb_chunks
SET embedding_vec = subvector(embedding_vec_local, 1, 768)
WHERE embedding_vec IS NULL AND embedding_vec_local IS NOT NULL;

-- Step 3: Drop old columns (idempotent via IF EXISTS)
ALTER TABLE kb_chunks DROP COLUMN IF EXISTS embedding;
ALTER TABLE kb_chunks DROP COLUMN IF EXISTS embedding_vec_local;

-- Step 4: Rename local_* columns → standard names
-- Must check existence since this may have already run
DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='kb_chunks' AND column_name='embedding_model') THEN
        ALTER TABLE kb_chunks DROP COLUMN embedding_model;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='kb_chunks' AND column_name='embedding_dimensions') THEN
        ALTER TABLE kb_chunks DROP COLUMN embedding_dimensions;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='kb_chunks' AND column_name='local_embedding_model') THEN
        ALTER TABLE kb_chunks RENAME COLUMN local_embedding_model TO embedding_model;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='kb_chunks' AND column_name='local_embedding_dimensions') THEN
        ALTER TABLE kb_chunks RENAME COLUMN local_embedding_dimensions TO embedding_dimensions;
    END IF;
END $$;

-- ============================================================
-- KB_DOCUMENTS: migrate doc_embedding_vec_local (vector(3072)) → doc_embedding_vec (vector(768))
-- ============================================================

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='kb_documents' AND column_name='doc_embedding_vec') THEN
        ALTER TABLE kb_documents ADD COLUMN doc_embedding_vec vector(768);
    END IF;
END $$;

UPDATE kb_documents
SET doc_embedding_vec = subvector(doc_embedding_vec_local, 1, 768)
WHERE doc_embedding_vec IS NULL AND doc_embedding_vec_local IS NOT NULL;

ALTER TABLE kb_documents DROP COLUMN IF EXISTS doc_embedding;
ALTER TABLE kb_documents DROP COLUMN IF EXISTS doc_embedding_vec_local;

DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='kb_documents' AND column_name='embedding_model') THEN
        ALTER TABLE kb_documents DROP COLUMN embedding_model;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='kb_documents' AND column_name='embedding_dimensions') THEN
        ALTER TABLE kb_documents DROP COLUMN embedding_dimensions;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='kb_documents' AND column_name='local_embedding_model') THEN
        ALTER TABLE kb_documents RENAME COLUMN local_embedding_model TO embedding_model;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='kb_documents' AND column_name='local_embedding_dimensions') THEN
        ALTER TABLE kb_documents RENAME COLUMN local_embedding_dimensions TO embedding_dimensions;
    END IF;
END $$;

-- ============================================================
-- Remove NOT NULL constraint on embedding column if it was carried over
-- (embedding JSONB was NOT NULL; that column is now dropped)
-- ============================================================

-- ============================================================
-- INDEXES: HNSW index for fast cosine similarity search
-- ============================================================

DROP INDEX IF EXISTS idx_kb_chunks_embedding_vec_hnsw;
CREATE INDEX idx_kb_chunks_embedding_vec_hnsw ON kb_chunks
    USING hnsw (embedding_vec vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

DROP INDEX IF EXISTS idx_kb_documents_doc_embedding_vec_hnsw;
CREATE INDEX idx_kb_documents_doc_embedding_vec_hnsw ON kb_documents
    USING hnsw (doc_embedding_vec vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

COMMIT;
