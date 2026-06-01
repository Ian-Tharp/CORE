"""
Repository for Discord bridge persistence.

Stores bridge configuration, Discord-to-CORE channel mappings, and message link
records so the native gateway survives restarts and can correlate messages
across both systems.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from app.dependencies import get_db_pool

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_KEY = "default"


async def ensure_discord_tables() -> None:
    """Create all Discord bridge tables (idempotent)."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS discord_bridge_config (
                    config_key VARCHAR(64) PRIMARY KEY,
                    enabled BOOLEAN DEFAULT FALSE,
                    allowed_users JSONB DEFAULT '[]',
                    default_core_channel VARCHAR(255),
                    auto_create_channels BOOLEAN DEFAULT TRUE,
                    message_prefix TEXT DEFAULT '',
                    response_prefix TEXT DEFAULT '',
                    reconnect_delay_seconds INTEGER DEFAULT 5,
                    max_reconnect_attempts INTEGER DEFAULT 10,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS discord_channel_mappings (
                    discord_channel_id VARCHAR(255) PRIMARY KEY,
                    discord_channel_name VARCHAR(255),
                    discord_guild_id VARCHAR(255),
                    discord_guild_name VARCHAR(255),
                    core_channel_id VARCHAR(255) NOT NULL,
                    core_channel_name VARCHAR(255),
                    require_mention BOOLEAN DEFAULT FALSE,
                    enabled BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_discord_channel_mappings_core_channel
                ON discord_channel_mappings(core_channel_id)
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_discord_channel_mappings_guild
                ON discord_channel_mappings(discord_guild_id)
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS discord_message_links (
                    id SERIAL PRIMARY KEY,
                    core_message_id VARCHAR(255) NOT NULL,
                    core_channel_id VARCHAR(255) NOT NULL,
                    discord_message_id VARCHAR(255) NOT NULL,
                    discord_channel_id VARCHAR(255) NOT NULL,
                    discord_guild_id VARCHAR(255),
                    discord_author_id VARCHAR(255),
                    direction VARCHAR(32) NOT NULL,
                    chunk_index INTEGER DEFAULT 0,
                    total_chunks INTEGER DEFAULT 1,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (discord_message_id, discord_channel_id)
                )
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_discord_message_links_core_message
                ON discord_message_links(core_message_id)
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_discord_message_links_core_channel
                ON discord_message_links(core_channel_id)
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_discord_message_links_created_at
                ON discord_message_links(created_at DESC)
                """
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS discord_delivery_events (
                    id SERIAL PRIMARY KEY,
                    event_id VARCHAR(255) UNIQUE NOT NULL,
                    status VARCHAR(32) NOT NULL,
                    direction VARCHAR(32) NOT NULL,
                    core_message_id VARCHAR(255),
                    core_channel_id VARCHAR(255),
                    discord_message_id VARCHAR(255),
                    discord_channel_id VARCHAR(255),
                    discord_guild_id VARCHAR(255),
                    error TEXT,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_discord_delivery_events_status
                ON discord_delivery_events(status)
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_discord_delivery_events_direction
                ON discord_delivery_events(direction)
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_discord_delivery_events_created_at
                ON discord_delivery_events(created_at DESC)
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_discord_delivery_events_core_channel
                ON discord_delivery_events(core_channel_id)
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_discord_delivery_events_discord_channel
                ON discord_delivery_events(discord_channel_id)
                """
            )

    logger.info("Discord bridge tables ensured")


async def get_bridge_config() -> Optional[Dict[str, Any]]:
    """Fetch the stored Discord bridge configuration."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT *
            FROM discord_bridge_config
            WHERE config_key = $1
            """,
            _DEFAULT_CONFIG_KEY,
        )
        return _row_to_dict(row) if row else None


async def upsert_bridge_config(
    *,
    enabled: bool,
    allowed_users: List[str],
    default_core_channel: Optional[str],
    auto_create_channels: bool,
    message_prefix: str,
    response_prefix: str,
    reconnect_delay_seconds: int,
    max_reconnect_attempts: int,
) -> Dict[str, Any]:
    """Create or update the stored Discord bridge configuration."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO discord_bridge_config (
                config_key,
                enabled,
                allowed_users,
                default_core_channel,
                auto_create_channels,
                message_prefix,
                response_prefix,
                reconnect_delay_seconds,
                max_reconnect_attempts,
                updated_at
            )
            VALUES ($1, $2, $3::jsonb, $4, $5, $6, $7, $8, $9, CURRENT_TIMESTAMP)
            ON CONFLICT (config_key) DO UPDATE SET
                enabled = EXCLUDED.enabled,
                allowed_users = EXCLUDED.allowed_users,
                default_core_channel = EXCLUDED.default_core_channel,
                auto_create_channels = EXCLUDED.auto_create_channels,
                message_prefix = EXCLUDED.message_prefix,
                response_prefix = EXCLUDED.response_prefix,
                reconnect_delay_seconds = EXCLUDED.reconnect_delay_seconds,
                max_reconnect_attempts = EXCLUDED.max_reconnect_attempts,
                updated_at = CURRENT_TIMESTAMP
            RETURNING *
            """,
            _DEFAULT_CONFIG_KEY,
            enabled,
            json.dumps(allowed_users),
            default_core_channel,
            auto_create_channels,
            message_prefix,
            response_prefix,
            reconnect_delay_seconds,
            max_reconnect_attempts,
        )
        return _row_to_dict(row)


async def list_channel_mappings() -> List[Dict[str, Any]]:
    """List all stored Discord channel mappings."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT *
            FROM discord_channel_mappings
            ORDER BY discord_guild_name NULLS LAST, discord_channel_name NULLS LAST, discord_channel_id
            """
        )
        return [_row_to_dict(row) for row in rows]


async def get_channel_mapping(discord_channel_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single stored Discord channel mapping."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT *
            FROM discord_channel_mappings
            WHERE discord_channel_id = $1
            """,
            discord_channel_id,
        )
        return _row_to_dict(row) if row else None


async def upsert_channel_mapping(
    *,
    discord_channel_id: str,
    discord_channel_name: Optional[str],
    discord_guild_id: Optional[str],
    discord_guild_name: Optional[str],
    core_channel_id: str,
    core_channel_name: Optional[str],
    require_mention: bool,
    enabled: bool,
) -> Dict[str, Any]:
    """Create or update a Discord-to-CORE channel mapping."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO discord_channel_mappings (
                discord_channel_id,
                discord_channel_name,
                discord_guild_id,
                discord_guild_name,
                core_channel_id,
                core_channel_name,
                require_mention,
                enabled,
                updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CURRENT_TIMESTAMP)
            ON CONFLICT (discord_channel_id) DO UPDATE SET
                discord_channel_name = EXCLUDED.discord_channel_name,
                discord_guild_id = EXCLUDED.discord_guild_id,
                discord_guild_name = EXCLUDED.discord_guild_name,
                core_channel_id = EXCLUDED.core_channel_id,
                core_channel_name = EXCLUDED.core_channel_name,
                require_mention = EXCLUDED.require_mention,
                enabled = EXCLUDED.enabled,
                updated_at = CURRENT_TIMESTAMP
            RETURNING *
            """,
            discord_channel_id,
            discord_channel_name,
            discord_guild_id,
            discord_guild_name,
            core_channel_id,
            core_channel_name,
            require_mention,
            enabled,
        )
        return _row_to_dict(row)


async def delete_channel_mapping(discord_channel_id: str) -> bool:
    """Delete a stored Discord channel mapping."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            DELETE FROM discord_channel_mappings
            WHERE discord_channel_id = $1
            """,
            discord_channel_id,
        )
        return result == "DELETE 1"


async def create_message_link(
    *,
    core_message_id: str,
    core_channel_id: str,
    discord_message_id: str,
    discord_channel_id: str,
    direction: str,
    discord_guild_id: Optional[str] = None,
    discord_author_id: Optional[str] = None,
    chunk_index: int = 0,
    total_chunks: int = 1,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist a Discord↔CORE message correlation record."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO discord_message_links (
                core_message_id,
                core_channel_id,
                discord_message_id,
                discord_channel_id,
                discord_guild_id,
                discord_author_id,
                direction,
                chunk_index,
                total_chunks,
                metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
            ON CONFLICT (discord_message_id, discord_channel_id) DO UPDATE SET
                core_message_id = EXCLUDED.core_message_id,
                core_channel_id = EXCLUDED.core_channel_id,
                discord_guild_id = EXCLUDED.discord_guild_id,
                discord_author_id = EXCLUDED.discord_author_id,
                direction = EXCLUDED.direction,
                chunk_index = EXCLUDED.chunk_index,
                total_chunks = EXCLUDED.total_chunks,
                metadata = EXCLUDED.metadata
            RETURNING *
            """,
            core_message_id,
            core_channel_id,
            discord_message_id,
            discord_channel_id,
            discord_guild_id,
            discord_author_id,
            direction,
            chunk_index,
            total_chunks,
            json.dumps(metadata or {}),
        )
        return _row_to_dict(row)


async def get_message_link_by_discord_message(
    discord_message_id: str,
    discord_channel_id: str,
) -> Optional[Dict[str, Any]]:
    """Fetch a link record by Discord message identity."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT *
            FROM discord_message_links
            WHERE discord_message_id = $1
              AND discord_channel_id = $2
            """,
            discord_message_id,
            discord_channel_id,
        )
        return _row_to_dict(row) if row else None


async def get_primary_message_link_for_core_message(
    core_message_id: str,
    discord_channel_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch the primary link record for a CORE message."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        if discord_channel_id:
            row = await conn.fetchrow(
                """
                SELECT *
                FROM discord_message_links
                WHERE core_message_id = $1
                  AND discord_channel_id = $2
                ORDER BY chunk_index ASC, created_at ASC
                LIMIT 1
                """,
                core_message_id,
                discord_channel_id,
            )
        else:
            row = await conn.fetchrow(
                """
                SELECT *
                FROM discord_message_links
                WHERE core_message_id = $1
                ORDER BY chunk_index ASC, created_at ASC
                LIMIT 1
                """,
                core_message_id,
            )
        return _row_to_dict(row) if row else None


async def list_message_links_for_core_message(
    core_message_id: str,
) -> List[Dict[str, Any]]:
    """List all Discord link rows for a CORE message."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT *
            FROM discord_message_links
            WHERE core_message_id = $1
            ORDER BY chunk_index ASC, created_at ASC
            """,
            core_message_id,
        )
        return [_row_to_dict(row) for row in rows]


async def list_message_links(
    *,
    limit: int = 50,
    core_message_id: Optional[str] = None,
    core_channel_id: Optional[str] = None,
    discord_channel_id: Optional[str] = None,
    direction: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List Discord link rows with optional filters."""
    pool = await get_db_pool()
    clauses: list[str] = []
    params: list[Any] = []
    idx = 1

    if core_message_id is not None:
        clauses.append(f"core_message_id = ${idx}")
        params.append(core_message_id)
        idx += 1
    if core_channel_id is not None:
        clauses.append(f"core_channel_id = ${idx}")
        params.append(core_channel_id)
        idx += 1
    if discord_channel_id is not None:
        clauses.append(f"discord_channel_id = ${idx}")
        params.append(discord_channel_id)
        idx += 1
    if direction is not None:
        clauses.append(f"direction = ${idx}")
        params.append(direction)
        idx += 1

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT *
            FROM discord_message_links
            {where}
            ORDER BY created_at DESC, id DESC
            LIMIT ${idx}
            """,
            *params,
        )
        return [_row_to_dict(row) for row in rows]


async def count_channel_mappings() -> int:
    """Count stored Discord channel mappings."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval("SELECT COUNT(*) FROM discord_channel_mappings")


async def count_message_links() -> int:
    """Count stored Discord message links."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval("SELECT COUNT(*) FROM discord_message_links")


async def count_message_links_by_direction() -> Dict[str, int]:
    """Count message links grouped by direction."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT direction, COUNT(*) AS cnt
            FROM discord_message_links
            GROUP BY direction
            """
        )
        return {row["direction"]: row["cnt"] for row in rows}


async def create_delivery_event(
    *,
    event_id: str,
    status: str,
    direction: str,
    core_message_id: Optional[str] = None,
    core_channel_id: Optional[str] = None,
    discord_message_id: Optional[str] = None,
    discord_channel_id: Optional[str] = None,
    discord_guild_id: Optional[str] = None,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist a Discord delivery/ingest event."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO discord_delivery_events (
                event_id,
                status,
                direction,
                core_message_id,
                core_channel_id,
                discord_message_id,
                discord_channel_id,
                discord_guild_id,
                error,
                metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
            RETURNING *
            """,
            event_id,
            status,
            direction,
            core_message_id,
            core_channel_id,
            discord_message_id,
            discord_channel_id,
            discord_guild_id,
            error,
            json.dumps(metadata or {}),
        )
        return _row_to_dict(row)


async def list_delivery_events(
    *,
    limit: int = 50,
    status: Optional[str] = None,
    direction: Optional[str] = None,
    core_channel_id: Optional[str] = None,
    discord_channel_id: Optional[str] = None,
    core_message_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List Discord delivery events with optional filters."""
    pool = await get_db_pool()
    clauses: list[str] = []
    params: list[Any] = []
    idx = 1

    if status is not None:
        clauses.append(f"status = ${idx}")
        params.append(status)
        idx += 1
    if direction is not None:
        clauses.append(f"direction = ${idx}")
        params.append(direction)
        idx += 1
    if core_channel_id is not None:
        clauses.append(f"core_channel_id = ${idx}")
        params.append(core_channel_id)
        idx += 1
    if discord_channel_id is not None:
        clauses.append(f"discord_channel_id = ${idx}")
        params.append(discord_channel_id)
        idx += 1
    if core_message_id is not None:
        clauses.append(f"core_message_id = ${idx}")
        params.append(core_message_id)
        idx += 1

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT *
            FROM discord_delivery_events
            {where}
            ORDER BY created_at DESC, id DESC
            LIMIT ${idx}
            """,
            *params,
        )
        return [_row_to_dict(row) for row in rows]


async def count_delivery_events() -> int:
    """Count stored Discord delivery events."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval("SELECT COUNT(*) FROM discord_delivery_events")


async def count_delivery_events_by_status() -> Dict[str, int]:
    """Count Discord delivery events grouped by status."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT status, COUNT(*) AS cnt
            FROM discord_delivery_events
            GROUP BY status
            """
        )
        return {row["status"]: row["cnt"] for row in rows}


async def count_delivery_events_by_direction() -> Dict[str, int]:
    """Count Discord delivery events grouped by direction."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT direction, COUNT(*) AS cnt
            FROM discord_delivery_events
            GROUP BY direction
            """
        )
        return {row["direction"]: row["cnt"] for row in rows}


def _row_to_dict(row) -> Dict[str, Any]:
    """Convert an asyncpg Record to a plain dict with ISO datetime strings."""
    d = dict(row)
    for key, value in d.items():
        if hasattr(value, "isoformat"):
            d[key] = value.isoformat()
    return d
