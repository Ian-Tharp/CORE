"""
Repository for Communication Commons data access.

Provides database operations for channels, messages, presence, and reactions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import uuid
from datetime import datetime

import asyncpg

from app.dependencies import get_db_pool


# =============================================================================
# CHANNELS
# =============================================================================

async def list_channels(instance_id: str) -> List[Dict[str, Any]]:
    """Get all channels that an instance is a member of."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                c.channel_id,
                c.channel_type,
                c.name,
                c.description,
                c.is_persistent,
                c.is_public,
                c.created_by,
                c.created_at,
                COUNT(DISTINCT cm.instance_id) as member_count,
                COUNT(DISTINCT m.message_id) as message_count,
                MAX(m.created_at) as last_message_at
            FROM communication_channels c
            LEFT JOIN channel_members cm ON c.channel_id = cm.channel_id
            LEFT JOIN communication_messages m ON c.channel_id = m.channel_id
            WHERE c.channel_id IN (
                SELECT channel_id FROM channel_members WHERE instance_id = $1
            )
            OR c.is_public = true
            GROUP BY c.id, c.channel_id, c.channel_type, c.name, c.description,
                     c.is_persistent, c.is_public, c.created_by, c.created_at
            ORDER BY last_message_at DESC NULLS LAST
            """,
            instance_id
        )
        # Convert all datetime fields to ISO format strings
        result = []
        for row in rows:
            channel = dict(row)
            for key, value in channel.items():
                if hasattr(value, 'isoformat'):
                    channel[key] = value.isoformat()
            result.append(channel)
        return result


async def get_channel(channel_id: str) -> Optional[Dict[str, Any]]:
    """Get a single channel by ID."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                channel_id,
                channel_type,
                name,
                description,
                is_persistent,
                is_public,
                created_by,
                created_at,
                metadata
            FROM communication_channels
            WHERE channel_id = $1
            """,
            channel_id
        )
        if not row:
            return None

        # Convert to dict and ensure all datetime fields are ISO format strings
        result = dict(row)
        for key, value in result.items():
            if hasattr(value, 'isoformat'):  # Check if it's a datetime object
                result[key] = value.isoformat()
        return result


async def create_channel(
    channel_id: str,
    channel_type: str,
    name: str,
    description: Optional[str],
    is_persistent: bool,
    is_public: bool,
    created_by: str,
    initial_members: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create a new channel and optionally add initial members."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Create channel
            await conn.execute(
                """
                INSERT INTO communication_channels
                (channel_id, channel_type, name, description, is_persistent, is_public, created_by)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                channel_id, channel_type, name, description, is_persistent, is_public, created_by
            )

            # Add creator as owner
            await conn.execute(
                """
                INSERT INTO channel_members (channel_id, instance_id, instance_type, role)
                VALUES ($1, $2, $3, 'owner')
                """,
                channel_id, created_by, 'human'  # TODO: Detect instance type dynamically
            )

            # Add initial members if provided
            if initial_members:
                for member_id in initial_members:
                    if member_id != created_by:  # Don't duplicate creator
                        await conn.execute(
                            """
                            INSERT INTO channel_members (channel_id, instance_id, instance_type, role)
                            VALUES ($1, $2, $3, 'member')
                            ON CONFLICT (channel_id, instance_id) DO NOTHING
                            """,
                            channel_id, member_id, 'agent'  # TODO: Detect instance type
                        )

            # Fetch the created channel within the same transaction
            row = await conn.fetchrow(
                """
                SELECT
                    channel_id,
                    channel_type,
                    name,
                    description,
                    is_persistent,
                    is_public,
                    created_by,
                    created_at,
                    metadata
                FROM communication_channels
                WHERE channel_id = $1
                """,
                channel_id
            )

            if not row:
                raise Exception(f"Failed to fetch created channel {channel_id}")

            # Convert all datetime fields to ISO format strings
            result = dict(row)
            for key, value in result.items():
                if hasattr(value, 'isoformat'):
                    result[key] = value.isoformat()
            return result


# =============================================================================
# MESSAGES
# =============================================================================

async def list_messages(
    channel_id: str,
    page: int = 1,
    page_size: int = 50,
    thread_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get messages for a channel with pagination."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        offset = (page - 1) * page_size

        # Build query based on whether we're fetching thread messages
        if thread_id:
            query = """
                SELECT
                    m.message_id,
                    m.channel_id,
                    m.sender_id,
                    m.sender_name,
                    m.sender_type,
                    m.content,
                    m.message_type,
                    m.parent_message_id,
                    m.thread_id,
                    m.created_at,
                    m.edited_at,
                    m.metadata,
                    COUNT(DISTINCT r.id) as reaction_count,
                    COUNT(DISTINCT replies.message_id) as reply_count
                FROM communication_messages m
                LEFT JOIN message_reactions r ON m.message_id = r.message_id
                LEFT JOIN communication_messages replies ON m.message_id = replies.thread_id
                WHERE m.thread_id = $1 OR m.message_id = $1
                GROUP BY m.id, m.message_id, m.channel_id, m.sender_id, m.sender_name,
                         m.sender_type, m.content, m.message_type, m.parent_message_id,
                         m.thread_id, m.created_at, m.edited_at, m.metadata
                ORDER BY m.created_at ASC
                LIMIT $2 OFFSET $3
            """
            params = (thread_id, page_size, offset)
        else:
            query = """
                SELECT
                    m.message_id,
                    m.channel_id,
                    m.sender_id,
                    m.sender_name,
                    m.sender_type,
                    m.content,
                    m.message_type,
                    m.parent_message_id,
                    m.thread_id,
                    m.created_at,
                    m.edited_at,
                    m.metadata,
                    COUNT(DISTINCT r.id) as reaction_count,
                    COUNT(DISTINCT replies.message_id) as reply_count
                FROM communication_messages m
                LEFT JOIN message_reactions r ON m.message_id = r.message_id
                LEFT JOIN communication_messages replies ON m.message_id = replies.thread_id
                                                        AND replies.message_id != m.message_id
                WHERE m.channel_id = $1
                AND m.thread_id IS NULL
                GROUP BY m.id, m.message_id, m.channel_id, m.sender_id, m.sender_name,
                         m.sender_type, m.content, m.message_type, m.parent_message_id,
                         m.thread_id, m.created_at, m.edited_at, m.metadata
                ORDER BY m.created_at DESC
                LIMIT $2 OFFSET $3
            """
            params = (channel_id, page_size, offset)

        rows = await conn.fetch(query, *params)
        # Convert all datetime fields to ISO format strings
        result = []
        for row in rows:
            message = dict(row)
            for key, value in message.items():
                if hasattr(value, 'isoformat'):
                    message[key] = value.isoformat()
            result.append(message)
        return result


async def create_message(
    message_id: str,
    channel_id: str,
    sender_id: str,
    sender_name: str,
    sender_type: str,
    content: str,
    message_type: str = 'text',
    parent_message_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a new message in a channel."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        import json

        await conn.execute(
            """
            INSERT INTO communication_messages
            (message_id, channel_id, sender_id, sender_name, sender_type, content,
             message_type, parent_message_id, thread_id, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            message_id, channel_id, sender_id, sender_name, sender_type, content,
            message_type, parent_message_id, thread_id, json.dumps(metadata) if metadata else None
        )

        # Fetch and return the created message
        row = await conn.fetchrow(
            """
            SELECT
                message_id, channel_id, sender_id, sender_name, sender_type,
                content, message_type, parent_message_id, thread_id,
                created_at, edited_at, metadata
            FROM communication_messages
            WHERE message_id = $1
            """,
            message_id
        )
        # Convert all datetime fields to ISO format strings
        result = dict(row)
        for key, value in result.items():
            if hasattr(value, 'isoformat'):
                result[key] = value.isoformat()
        return result


# =============================================================================
# REACTIONS
# =============================================================================

async def get_message_reactions(message_id: str) -> List[Dict[str, Any]]:
    """Get all reactions for a message with counts."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                reaction_type,
                COUNT(*) as count,
                array_agg(instance_id) as reacted_by
            FROM message_reactions
            WHERE message_id = $1
            GROUP BY reaction_type
            """,
            message_id
        )
        return [dict(r) for r in rows]


async def add_reaction(
    message_id: str,
    instance_id: str,
    reaction_type: str
) -> None:
    """Add a reaction to a message."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO message_reactions (message_id, instance_id, reaction_type)
            VALUES ($1, $2, $3)
            ON CONFLICT (message_id, instance_id, reaction_type) DO NOTHING
            """,
            message_id, instance_id, reaction_type
        )


async def remove_reaction(
    message_id: str,
    instance_id: str,
    reaction_type: str
) -> None:
    """Remove a reaction from a message."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            DELETE FROM message_reactions
            WHERE message_id = $1 AND instance_id = $2 AND reaction_type = $3
            """,
            message_id, instance_id, reaction_type
        )


# =============================================================================
# PRESENCE
# =============================================================================

async def get_all_presence() -> List[Dict[str, Any]]:
    """Get presence for all instances."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                instance_id,
                instance_name,
                instance_type,
                status,
                current_activity,
                current_phase,
                last_heartbeat,
                metadata
            FROM instance_presence
            ORDER BY status, instance_name
            """
        )
        # Convert all datetime fields to ISO format strings
        result = []
        for row in rows:
            presence = dict(row)
            for key, value in presence.items():
                if hasattr(value, 'isoformat'):
                    presence[key] = value.isoformat()
            result.append(presence)
        return result


async def update_presence(
    instance_id: str,
    status: Optional[str] = None,
    activity: Optional[str] = None,
    phase: Optional[int] = None
) -> None:
    """Update presence information for an instance."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        # Build dynamic update query
        updates = ['last_heartbeat = CURRENT_TIMESTAMP']
        params = [instance_id]
        param_idx = 2

        if status:
            updates.append(f'status = ${param_idx}')
            params.append(status)
            param_idx += 1

        if activity is not None:  # Allow empty string
            updates.append(f'current_activity = ${param_idx}')
            params.append(activity)
            param_idx += 1

        if phase is not None:
            updates.append(f'current_phase = ${param_idx}')
            params.append(phase)
            param_idx += 1

        query = f"""
            UPDATE instance_presence
            SET {', '.join(updates)}
            WHERE instance_id = $1
        """

        await conn.execute(query, *params)


async def get_instance_presence(instance_id: str) -> Optional[Dict[str, Any]]:
    """Get presence for a specific instance."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                instance_id,
                instance_name,
                instance_type,
                status,
                current_activity,
                current_phase,
                last_heartbeat,
                metadata
            FROM instance_presence
            WHERE instance_id = $1
            """,
            instance_id
        )
        if not row:
            return None
        # Convert all datetime fields to ISO format strings
        result = dict(row)
        for key, value in result.items():
            if hasattr(value, 'isoformat'):
                result[key] = value.isoformat()
        return result
