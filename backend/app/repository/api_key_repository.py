"""
API Key Repository

Data access layer for persistent API key storage.
Replaces the in-memory _API_KEYS dict in security.py with database persistence.

Pattern: asyncpg + get_db_pool (no SQLAlchemy).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.dependencies import get_db_pool

logger = logging.getLogger(__name__)


# =============================================================================
# TABLE INITIALISATION
# =============================================================================

async def ensure_api_key_tables() -> None:
    """Create api_keys table (idempotent)."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    key_hash VARCHAR(64) UNIQUE NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    description TEXT DEFAULT '',
                    permissions JSONB DEFAULT '["*"]',
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP WITH TIME ZONE,
                    request_count BIGINT DEFAULT 0,
                    is_active BOOLEAN DEFAULT TRUE
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_api_keys_name ON api_keys(name)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active) WHERE is_active = TRUE"
            )
    logger.info("API key tables ensured")


# =============================================================================
# WRITE
# =============================================================================

async def store_key(
    *,
    key_hash: str,
    name: str,
    description: str = "",
    permissions: List[str] | None = None,
) -> Optional[str]:
    """
    Persist a new API key record.

    Args:
        key_hash: SHA-256 hash of the raw key.
        name: Human-readable identifier.
        description: Optional description.
        permissions: List of permitted operations.

    Returns:
        The record UUID on success, None on failure.
    """
    record_id = str(uuid.uuid4())
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO api_keys (id, key_hash, name, description, permissions)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                """,
                uuid.UUID(record_id),
                key_hash,
                name,
                description,
                json.dumps(permissions or ["*"]),
            )
        logger.info("Stored API key record for: %s", name)
        return record_id
    except Exception as exc:
        logger.error("Failed to store API key: %s", exc)
        return None


async def update_last_used(key_hash: str) -> None:
    """Bump last_used timestamp and increment request_count for a key."""
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE api_keys
                SET last_used = CURRENT_TIMESTAMP,
                    request_count = request_count + 1
                WHERE key_hash = $1 AND is_active = TRUE
                """,
                key_hash,
            )
    except Exception as exc:
        logger.error("Failed to update API key usage: %s", exc)


async def deactivate_by_name(name: str) -> bool:
    """
    Soft-delete an API key by name.

    Returns True if a key was deactivated.
    """
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE api_keys SET is_active = FALSE
                WHERE name = $1 AND is_active = TRUE
                """,
                name,
            )
            count = int(result.split()[-1])
            if count > 0:
                logger.info("Deactivated API key: %s", name)
            return count > 0
    except Exception as exc:
        logger.error("Failed to deactivate API key: %s", exc)
        return False


# =============================================================================
# READ
# =============================================================================

async def get_by_hash(key_hash: str) -> Optional[Dict[str, Any]]:
    """
    Look up an active API key by its hash.

    Returns key metadata dict or None.
    """
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, key_hash, name, description, permissions,
                       created_at, last_used, request_count
                FROM api_keys
                WHERE key_hash = $1 AND is_active = TRUE
                """,
                key_hash,
            )
            if row is None:
                return None
            return _row_to_dict(row)
    except Exception as exc:
        logger.error("Failed to look up API key: %s", exc)
        return None


async def list_all_active() -> List[Dict[str, Any]]:
    """List all active API keys (without exposing hashes)."""
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, key_hash, name, description, permissions,
                       created_at, last_used, request_count
                FROM api_keys
                WHERE is_active = TRUE
                ORDER BY created_at DESC
                """
            )
            return [_row_to_dict(row) for row in rows]
    except Exception as exc:
        logger.error("Failed to list API keys: %s", exc)
        return []


def _row_to_dict(row) -> Dict[str, Any]:
    """Convert an asyncpg Row to a plain dict."""
    perms = row["permissions"]
    if isinstance(perms, str):
        perms = json.loads(perms)
    return {
        "id": str(row["id"]),
        "key_hash": row["key_hash"],
        "name": row["name"],
        "description": row["description"],
        "permissions": perms,
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "last_used": row["last_used"].isoformat() if row["last_used"] else None,
        "request_count": row["request_count"],
    }