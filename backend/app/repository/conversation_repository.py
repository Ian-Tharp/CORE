from __future__ import annotations

"""PostgreSQL-backed storage for chat conversations.

This replaces the previous JSON-file approach and persists conversations and
messages using the schema provided in `init.sql` (tables `conversations` and
`messages`).
"""

from typing import Any, Dict, List, Optional, TypedDict
import uuid

import asyncpg

from app.dependencies import get_db_pool


class _Message(TypedDict):
    role: str
    content: str


class _Conversation(TypedDict):
    id: str
    title: str
    messages: List[_Message]


# ---------------------------------------------------------------------------
# Public repository API
# ---------------------------------------------------------------------------


async def list_conversations() -> List[Dict[str, Any]]:
    """Return list of conversations with message counts.

    Shape matches controller expectations: `{id, title, messages}` where
    `messages` is the count of messages in the conversation.
    """
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT c.conversation_id AS id,
                   COALESCE(c.title, '') AS title,
                   COUNT(m.id) AS messages,
                   COALESCE(MAX(m.timestamp), MAX(c.updated_at)) AS last_activity
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.conversation_id
            GROUP BY c.conversation_id, c.title
            ORDER BY last_activity DESC NULLS LAST
            """
        )
        # Do not expose last_activity field to the controller response shape
        return [{"id": r["id"], "title": r["title"], "messages": r["messages"]} for r in rows]


async def get_conversation(conv_id: str) -> Optional[_Conversation]:
    """Return a single conversation with its messages or None if not found."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        conv_row = await conn.fetchrow(
            """
            SELECT conversation_id AS id, COALESCE(title, 'New Conversation') AS title
            FROM conversations
            WHERE conversation_id = $1
            """,
            conv_id,
        )
        if conv_row is None:
            return None

        msg_rows = await conn.fetch(
            """
            SELECT role, content
            FROM messages
            WHERE conversation_id = $1
            ORDER BY timestamp ASC, id ASC
            """,
            conv_id,
        )

        return {
            "id": conv_row["id"],
            "title": conv_row["title"],
            "messages": [{"role": r["role"], "content": r["content"]} for r in msg_rows],
        }


async def create_conversation(
    initial_messages: List[_Message] | None = None,
    title: str | None = None,
) -> str:
    """Create a new conversation and return its id (UUID4 string)."""
    conv_id = str(uuid.uuid4())
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO conversations (conversation_id, title)
                VALUES ($1, $2)
                ON CONFLICT (conversation_id) DO NOTHING
                """,
                conv_id,
                title or "New Conversation",
            )

            if initial_messages:
                await _insert_messages(conn, conv_id, initial_messages)
                await conn.execute(
                    """
                    UPDATE conversations
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE conversation_id = $1
                    """,
                    conv_id,
                )

    return conv_id


async def append_message(conv_id: str, message: _Message) -> None:
    """Append message to an existing conversation; create if missing."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Ensure conversation exists
            exists = await conn.fetchval(
                "SELECT 1 FROM conversations WHERE conversation_id = $1",
                conv_id,
            )
            if not exists:
                await conn.execute(
                    """
                    INSERT INTO conversations (conversation_id, title)
                    VALUES ($1, $2)
                    ON CONFLICT (conversation_id) DO NOTHING
                    """,
                    conv_id,
                    "Recovered",
                )

            await _insert_messages(conn, conv_id, [message])
            await conn.execute(
                """
                UPDATE conversations
                SET updated_at = CURRENT_TIMESTAMP
                WHERE conversation_id = $1
                """,
                conv_id,
            )


async def update_title(conv_id: str, title: str) -> None:
    """Update the title of a conversation."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE conversations
            SET title = $2, updated_at = CURRENT_TIMESTAMP
            WHERE conversation_id = $1
            """,
            conv_id,
            title,
        )


async def _insert_messages(conn: asyncpg.Connection, conv_id: str, messages: List[_Message]) -> None:
    """Helper to bulk-insert messages for a given conversation id."""
    if not messages:
        return

    # Use executemany for efficiency
    await conn.executemany(
        """
        INSERT INTO messages (conversation_id, role, content)
        VALUES ($1, $2, $3)
        """,
        [(conv_id, m["role"], m["content"]) for m in messages],
    )
