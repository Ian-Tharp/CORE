"""
Repository for Inter-Agent Communication Bus data access.

Provides database operations for bus messages, subscriptions,
external agent registrations, delivery receipts, and offline queues.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from app.dependencies import get_db_pool

logger = logging.getLogger(__name__)


# =============================================================================
# TABLE INITIALISATION
# =============================================================================

async def ensure_bus_tables() -> None:
    """Create all Communication Bus tables (idempotent)."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            # ----- bus_messages -----
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bus_messages (
                    id SERIAL PRIMARY KEY,
                    message_id VARCHAR(255) UNIQUE NOT NULL,
                    sender_id VARCHAR(255) NOT NULL,
                    recipients JSONB DEFAULT '[]',
                    message_type VARCHAR(64) NOT NULL,
                    topic VARCHAR(255),
                    payload JSONB DEFAULT '{}',
                    priority VARCHAR(16) DEFAULT 'normal',
                    correlation_id VARCHAR(255),
                    reply_to VARCHAR(255),
                    ttl_seconds INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bus_messages_sender ON bus_messages(sender_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bus_messages_type ON bus_messages(message_type)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bus_messages_topic ON bus_messages(topic)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bus_messages_correlation ON bus_messages(correlation_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bus_messages_created ON bus_messages(created_at DESC)"
            )

            # ----- bus_subscriptions -----
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bus_subscriptions (
                    id SERIAL PRIMARY KEY,
                    subscription_id VARCHAR(255) UNIQUE NOT NULL,
                    agent_id VARCHAR(255) NOT NULL,
                    message_types JSONB DEFAULT '[]',
                    topics JSONB DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bus_subs_agent ON bus_subscriptions(agent_id)"
            )

            # ----- bus_external_agents -----
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bus_external_agents (
                    id SERIAL PRIMARY KEY,
                    agent_id VARCHAR(255) UNIQUE NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    capabilities JSONB DEFAULT '[]',
                    webhook_url TEXT NOT NULL,
                    webhook_secret VARCHAR(512),
                    webhook_max_retries INTEGER DEFAULT 3,
                    webhook_retry_backoff_base_ms INTEGER DEFAULT 1000,
                    webhook_timeout_ms INTEGER DEFAULT 5000,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # ----- bus_delivery_receipts -----
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bus_delivery_receipts (
                    id SERIAL PRIMARY KEY,
                    receipt_id VARCHAR(255) UNIQUE NOT NULL,
                    message_id VARCHAR(255) NOT NULL,
                    recipient_id VARCHAR(255) NOT NULL,
                    status VARCHAR(32) DEFAULT 'pending',
                    delivered_at TIMESTAMP,
                    error TEXT
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bus_receipts_message ON bus_delivery_receipts(message_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bus_receipts_recipient ON bus_delivery_receipts(recipient_id)"
            )

            # ----- bus_offline_queue -----
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bus_offline_queue (
                    id SERIAL PRIMARY KEY,
                    agent_id VARCHAR(255) NOT NULL,
                    message_id VARCHAR(255) NOT NULL,
                    queued_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bus_queue_agent ON bus_offline_queue(agent_id)"
            )

    logger.info("Bus tables ensured")


# =============================================================================
# MESSAGES
# =============================================================================

async def store_message(
    message_id: str,
    sender_id: str,
    recipients: List[str],
    message_type: str,
    topic: Optional[str],
    payload: Dict[str, Any],
    priority: str,
    correlation_id: Optional[str],
    reply_to: Optional[str],
    ttl_seconds: Optional[int],
    created_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Persist a bus message and return the stored row."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO bus_messages
                (message_id, sender_id, recipients, message_type, topic,
                 payload, priority, correlation_id, reply_to, ttl_seconds, created_at)
            VALUES ($1, $2, $3::jsonb, $4, $5, $6::jsonb, $7, $8, $9, $10, $11)
            RETURNING *
            """,
            message_id,
            sender_id,
            json.dumps(recipients),
            message_type,
            topic,
            json.dumps(payload),
            priority,
            correlation_id,
            reply_to,
            ttl_seconds,
            created_at or datetime.utcnow(),
        )
        return _row_to_dict(row)


async def get_message(message_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a single bus message by its ``message_id``."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM bus_messages WHERE message_id = $1",
            message_id,
        )
        return _row_to_dict(row) if row else None


async def get_messages_by_correlation(correlation_id: str) -> List[Dict[str, Any]]:
    """Get all messages sharing a correlation ID (request/response chain)."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM bus_messages WHERE correlation_id = $1 ORDER BY created_at",
            correlation_id,
        )
        return [_row_to_dict(r) for r in rows]


async def count_messages(
    since: Optional[datetime] = None,
    message_type: Optional[str] = None,
) -> int:
    """Count messages, optionally filtering by time and/or type."""
    pool = await get_db_pool()
    clauses = []
    params: list = []
    idx = 1
    if since:
        clauses.append(f"created_at >= ${idx}")
        params.append(since)
        idx += 1
    if message_type:
        clauses.append(f"message_type = ${idx}")
        params.append(message_type)
        idx += 1
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    async with pool.acquire() as conn:
        return await conn.fetchval(
            f"SELECT COUNT(*) FROM bus_messages {where}", *params
        )


async def count_messages_by_type() -> Dict[str, int]:
    """Return message counts grouped by type."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT message_type, COUNT(*) AS cnt FROM bus_messages GROUP BY message_type"
        )
        return {r["message_type"]: r["cnt"] for r in rows}


async def count_messages_by_priority() -> Dict[str, int]:
    """Return message counts grouped by priority."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT priority, COUNT(*) AS cnt FROM bus_messages GROUP BY priority"
        )
        return {r["priority"]: r["cnt"] for r in rows}


# =============================================================================
# SUBSCRIPTIONS
# =============================================================================

async def create_subscription(
    subscription_id: str,
    agent_id: str,
    message_types: List[str],
    topics: List[str],
) -> Dict[str, Any]:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO bus_subscriptions (subscription_id, agent_id, message_types, topics)
            VALUES ($1, $2, $3::jsonb, $4::jsonb)
            RETURNING *
            """,
            subscription_id,
            agent_id,
            json.dumps(message_types),
            json.dumps(topics),
        )
        return _row_to_dict(row)


async def delete_subscription(subscription_id: str) -> bool:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM bus_subscriptions WHERE subscription_id = $1",
            subscription_id,
        )
        return result == "DELETE 1"


async def get_subscriptions_for_agent(agent_id: str) -> List[Dict[str, Any]]:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM bus_subscriptions WHERE agent_id = $1",
            agent_id,
        )
        return [_row_to_dict(r) for r in rows]


async def get_all_subscriptions() -> List[Dict[str, Any]]:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM bus_subscriptions")
        return [_row_to_dict(r) for r in rows]


async def count_subscriptions() -> int:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval("SELECT COUNT(*) FROM bus_subscriptions")


# =============================================================================
# EXTERNAL AGENTS
# =============================================================================

async def register_external_agent(
    agent_id: str,
    name: str,
    description: Optional[str],
    capabilities: List[str],
    webhook_url: str,
    webhook_secret: Optional[str],
    webhook_max_retries: int,
    webhook_retry_backoff_base_ms: int,
    webhook_timeout_ms: int,
) -> Dict[str, Any]:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO bus_external_agents
                (agent_id, name, description, capabilities,
                 webhook_url, webhook_secret, webhook_max_retries,
                 webhook_retry_backoff_base_ms, webhook_timeout_ms)
            VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8, $9)
            ON CONFLICT (agent_id) DO UPDATE SET
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                capabilities = EXCLUDED.capabilities,
                webhook_url = EXCLUDED.webhook_url,
                webhook_secret = EXCLUDED.webhook_secret,
                webhook_max_retries = EXCLUDED.webhook_max_retries,
                webhook_retry_backoff_base_ms = EXCLUDED.webhook_retry_backoff_base_ms,
                webhook_timeout_ms = EXCLUDED.webhook_timeout_ms
            RETURNING *
            """,
            agent_id,
            name,
            description,
            json.dumps(capabilities),
            webhook_url,
            webhook_secret,
            webhook_max_retries,
            webhook_retry_backoff_base_ms,
            webhook_timeout_ms,
        )
        return _row_to_dict(row)


async def deregister_external_agent(agent_id: str) -> bool:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM bus_external_agents WHERE agent_id = $1", agent_id
        )
        return result == "DELETE 1"


async def get_external_agent(agent_id: str) -> Optional[Dict[str, Any]]:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM bus_external_agents WHERE agent_id = $1", agent_id
        )
        return _row_to_dict(row) if row else None


async def list_external_agents() -> List[Dict[str, Any]]:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM bus_external_agents ORDER BY created_at")
        return [_row_to_dict(r) for r in rows]


async def count_external_agents() -> int:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval("SELECT COUNT(*) FROM bus_external_agents")


# =============================================================================
# DELIVERY RECEIPTS
# =============================================================================

async def create_delivery_receipt(
    receipt_id: str,
    message_id: str,
    recipient_id: str,
    status: str = "pending",
    error: Optional[str] = None,
) -> Dict[str, Any]:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        delivered_at = datetime.utcnow() if status == "delivered" else None
        row = await conn.fetchrow(
            """
            INSERT INTO bus_delivery_receipts
                (receipt_id, message_id, recipient_id, status, delivered_at, error)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING *
            """,
            receipt_id,
            message_id,
            recipient_id,
            status,
            delivered_at,
            error,
        )
        return _row_to_dict(row)


async def update_delivery_receipt(
    receipt_id: str,
    status: str,
    error: Optional[str] = None,
) -> bool:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        delivered_at = datetime.utcnow() if status == "delivered" else None
        result = await conn.execute(
            """
            UPDATE bus_delivery_receipts
            SET status = $2, delivered_at = COALESCE($3, delivered_at), error = $4
            WHERE receipt_id = $1
            """,
            receipt_id,
            status,
            delivered_at,
            error,
        )
        return result == "UPDATE 1"


async def get_receipts_for_message(message_id: str) -> List[Dict[str, Any]]:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM bus_delivery_receipts WHERE message_id = $1",
            message_id,
        )
        return [_row_to_dict(r) for r in rows]


async def count_receipts_by_status() -> Dict[str, int]:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT status, COUNT(*) AS cnt FROM bus_delivery_receipts GROUP BY status"
        )
        return {r["status"]: r["cnt"] for r in rows}


async def avg_delivery_latency_ms() -> Optional[float]:
    """Average latency (ms) of delivered messages (created → delivered)."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        val = await conn.fetchval(
            """
            SELECT AVG(EXTRACT(EPOCH FROM (dr.delivered_at - bm.created_at)) * 1000)
            FROM bus_delivery_receipts dr
            JOIN bus_messages bm ON bm.message_id = dr.message_id
            WHERE dr.status = 'delivered' AND dr.delivered_at IS NOT NULL
            """
        )
        return float(val) if val is not None else None


# =============================================================================
# OFFLINE QUEUE
# =============================================================================

async def enqueue_offline(agent_id: str, message_id: str) -> None:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO bus_offline_queue (agent_id, message_id)
            VALUES ($1, $2)
            ON CONFLICT DO NOTHING
            """,
            agent_id,
            message_id,
        )


async def drain_offline_queue(agent_id: str) -> List[Dict[str, Any]]:
    """Return and delete all queued messages for *agent_id*."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch(
                """
                SELECT bm.* FROM bus_offline_queue oq
                JOIN bus_messages bm ON bm.message_id = oq.message_id
                WHERE oq.agent_id = $1
                ORDER BY oq.queued_at
                """,
                agent_id,
            )
            await conn.execute(
                "DELETE FROM bus_offline_queue WHERE agent_id = $1", agent_id
            )
        return [_row_to_dict(r) for r in rows]


async def count_offline_queued() -> int:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval("SELECT COUNT(*) FROM bus_offline_queue")


# =============================================================================
# HELPERS
# =============================================================================

def _row_to_dict(row) -> Dict[str, Any]:
    """Convert an asyncpg Record to a plain dict with ISO datetime strings."""
    d = dict(row)
    for key, value in d.items():
        if hasattr(value, "isoformat"):
            d[key] = value.isoformat()
    return d
