import os
import asyncio
from functools import lru_cache
from typing import Optional

import asyncpg
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from openai import AsyncOpenAI
import anthropic
from dotenv import load_dotenv

load_dotenv()


@lru_cache()
def get_4o_llm():
    return ChatOpenAI(
        model="gpt-4o",
        verbose=True,
        temperature=0.25,
        max_retries=3,
        streaming=True,
    )


@lru_cache()
def get_o1_llm():
    return ChatOpenAI(
        model="o1-preview-2024-09-12",
    )


@lru_cache()
def get_claude_4_sonnet():
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        verbose=True,
        temperature=1.0,
        max_retries=3,
        streaming=True,
    )


@lru_cache()
def _get_openai_client() -> AsyncOpenAI:
    """Create and return an authenticated `AsyncOpenAI` client instance."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")

    return AsyncOpenAI(api_key=api_key)


@lru_cache()
def _get_ollama_base_url() -> str:
    """Return the base URL for the Ollama service.

    Defaults to the Docker service name `ollama` on the standard port when not
    explicitly configured via the ``OLLAMA_BASE_URL`` environment variable.
    """
    return os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")


@lru_cache()
def _get_anthropic_client() -> anthropic.Anthropic:
    """Create and return an authenticated Anthropic SDK client instance."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set.")
    return anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Database connection pool (PostgreSQL via asyncpg)
# ---------------------------------------------------------------------------

_DB_POOL: Optional[asyncpg.Pool] = None


async def get_db_pool() -> asyncpg.Pool:
    """Return a singleton asyncpg connection pool.

    The connection details are sourced from environment variables with sensible
    defaults that match the provided docker-compose setup.

    - DB_HOST (default: "postgres")
    - DB_PORT (default: "5432")
    - DB_NAME (default: "core_db")
    - DB_USER (default: "core_user")
    - DB_PASSWORD (default: "core_password")
    """
    global _DB_POOL  # noqa: PLW0603
    if _DB_POOL is not None:
        return _DB_POOL

    db_host = os.getenv("DB_HOST", "postgres")
    db_port = int(os.getenv("DB_PORT", "5432"))
    db_name = os.getenv("DB_NAME", "core_db")
    db_user = os.getenv("DB_USER", "core_user")
    db_password = os.getenv("DB_PASSWORD", "core_password")

    _DB_POOL = await asyncpg.create_pool(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
        database=db_name,
        min_size=1,
        max_size=10,
    )
    return _DB_POOL


async def close_db_pool() -> None:
    """Close the global asyncpg pool if it exists."""
    global _DB_POOL  # noqa: PLW0603
    if _DB_POOL is not None:
        await _DB_POOL.close()
        _DB_POOL = None


async def setup_db_schema() -> None:
    """Ensure required tables exist. Safe to run multiple times.

    This bootstraps the minimum schema needed by the chat/conversations flow so
    fresh developer environments do not depend solely on Docker's init scripts.
    """
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Conversations table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    conversation_id VARCHAR(255) UNIQUE NOT NULL,
                    title VARCHAR(500),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB
                )
                """
            )

            # Messages table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    conversation_id VARCHAR(255) NOT NULL,
                    role VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
                )
                """
            )

            # Indexes
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)"
            )

            # -----------------------------------------------------------------
            # Worlds (HexWorld snapshots) persistence
            # -----------------------------------------------------------------
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS worlds (
                    id UUID PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    origin VARCHAR(32) DEFAULT 'human',
                    tags JSONB DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS world_snapshots (
                    id UUID PRIMARY KEY,
                    world_id UUID NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    -- Minimal schema for HexWorld v2 payloads
                    config JSONB NOT NULL,
                    layers JSONB,
                    -- optional legacy tiles support
                    tiles JSONB,
                    -- optional base64 preview thumbnail (data URL)
                    preview TEXT,
                    CONSTRAINT fk_world
                        FOREIGN KEY(world_id)
                        REFERENCES worlds(id)
                        ON DELETE CASCADE
                )
                """
            )

            # In case the table already exists without 'preview' column (earlier dev envs)
            await conn.execute(
                "ALTER TABLE world_snapshots ADD COLUMN IF NOT EXISTS preview TEXT"
            )

            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_world_snapshots_world_id ON world_snapshots(world_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_worlds_updated_at ON worlds(updated_at)"
            )
            # -- Migrate legacy schemas where JSON fields were TEXT -----------
            # Convert world_snapshots JSON-like TEXT columns to JSONB when needed
            await conn.execute(
                """
                DO $$
                BEGIN
                  IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='world_snapshots' AND column_name='config' AND data_type <> 'jsonb'
                  ) THEN
                    ALTER TABLE world_snapshots ALTER COLUMN config TYPE JSONB USING config::jsonb;
                  END IF;
                  IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='world_snapshots' AND column_name='layers' AND data_type <> 'jsonb'
                  ) THEN
                    ALTER TABLE world_snapshots ALTER COLUMN layers TYPE JSONB USING layers::jsonb;
                  END IF;
                  IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='world_snapshots' AND column_name='tiles' AND data_type <> 'jsonb'
                  ) THEN
                    ALTER TABLE world_snapshots ALTER COLUMN tiles TYPE JSONB USING tiles::jsonb;
                  END IF;
                END$$;
                """
            )

            # Convert wiki_pages.metadata to JSONB when needed
            await conn.execute(
                """
                DO $$
                BEGIN
                  IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='wiki_pages' AND column_name='metadata' AND data_type <> 'jsonb'
                  ) THEN
                    ALTER TABLE wiki_pages ALTER COLUMN metadata TYPE JSONB USING metadata::jsonb;
                  END IF;
                END$$;
                """
            )

            # Convert characters.traits to JSONB when needed
            await conn.execute(
                """
                DO $$
                BEGIN
                  IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='characters' AND column_name='traits' AND data_type <> 'jsonb'
                  ) THEN
                    ALTER TABLE characters ALTER COLUMN traits TYPE JSONB USING traits::jsonb;
                  END IF;
                END$$;
                """
            )

            # Backfill columns if older table definitions exist
            await conn.execute("ALTER TABLE worlds ADD COLUMN IF NOT EXISTS origin VARCHAR(32) DEFAULT 'human'")
            await conn.execute("ALTER TABLE worlds ADD COLUMN IF NOT EXISTS tags JSONB DEFAULT '[]'")

            # -----------------------------------------------------------------
            # Creative Studio: wiki pages and characters
            # -----------------------------------------------------------------
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS wiki_pages (
                    id UUID PRIMARY KEY,
                    world_id UUID,
                    title VARCHAR(500) NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (world_id) REFERENCES worlds(id) ON DELETE SET NULL
                )
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS characters (
                    id UUID PRIMARY KEY,
                    world_id UUID,
                    name VARCHAR(255) NOT NULL,
                    traits JSONB,
                    image_b64 TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (world_id) REFERENCES worlds(id) ON DELETE SET NULL
                )
                """
            )

            # -----------------------------------------------------------------
            # Knowledgebase: documents and chunk embeddings (RAG)
            # -----------------------------------------------------------------
            await conn.execute(
                """
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
                    doc_embedding JSONB,
                    embedding_model VARCHAR(128),
                    embedding_dimensions INTEGER
                )
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kb_chunks (
                    id UUID PRIMARY KEY,
                    document_id UUID NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding JSONB NOT NULL,
                    embedding_model VARCHAR(128) NOT NULL,
                    embedding_dimensions INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES kb_documents(id) ON DELETE CASCADE
                )
                """
            )

            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_kb_chunks_document_id ON kb_chunks(document_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_kb_documents_upload_date ON kb_documents(upload_date)"
            )

            # -----------------------------------------------------------------
            # Knowledgebase: incremental schema upgrades
            # -----------------------------------------------------------------
            # Add human-friendly title column for documents
            await conn.execute(
                "ALTER TABLE kb_documents ADD COLUMN IF NOT EXISTS title VARCHAR(512)"
            )

            # Add file hash for duplicate detection
            await conn.execute(
                "ALTER TABLE kb_documents ADD COLUMN IF NOT EXISTS file_hash VARCHAR(128)"
            )
            await conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_kb_documents_file_hash ON kb_documents(file_hash) WHERE file_hash IS NOT NULL"
            )

            # Activity log for knowledgebase operations (uploads, deletes, processing, etc.)
            await conn.execute(
                """
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
                )
                """
            )

            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_kb_activity_timestamp ON kb_activity(timestamp)"
            )