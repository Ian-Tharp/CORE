-- Migration 005: kb_documents.world_id + kb_chunks legacy column cleanup
-- Purpose: fix `column "world_id" of relation "kb_documents" does not exist`
-- on POST /knowledgebase/upload.
--
-- knowledgebase_repository.create_document() inserts world_id and
-- list_documents_by_world() filters on it, but the column was only ever added
-- by setup_db_schema(). That function runs in a single transaction, so when any
-- earlier statement in it failed, the world_id ALTER was rolled back with it.
--
-- world_id is an opaque scope key supplied by callers outside the kernel
-- (world logic is archived from CORE, see docs/adr/006), so it is TEXT, not a
-- UUID foreign key. Databases where setup_db_schema() already created it as
-- UUID are converted in place (the index is rebuilt by ALTER ... TYPE).

DO $$
BEGIN
    IF to_regclass('kb_documents') IS NOT NULL THEN
        ALTER TABLE kb_documents ADD COLUMN IF NOT EXISTS world_id TEXT;

        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'kb_documents'
              AND column_name = 'world_id'
              AND data_type <> 'text'
        ) THEN
            ALTER TABLE kb_documents ALTER COLUMN world_id TYPE TEXT USING world_id::text;
        END IF;

        CREATE INDEX IF NOT EXISTS idx_kb_documents_world_id ON kb_documents(world_id);
    END IF;

    -- Chunk inserts only fill embedding_vec. Older databases still carry the
    -- legacy JSONB `embedding` column as NOT NULL, which rejects every insert.
    -- Databases that ran backend/migrations/003_remove_openai_embeddings.sql
    -- no longer have the column at all, so check before touching it.
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'kb_chunks'
          AND column_name = 'embedding'
          AND is_nullable = 'NO'
    ) THEN
        ALTER TABLE kb_chunks ALTER COLUMN embedding DROP NOT NULL;
    END IF;
END $$;
