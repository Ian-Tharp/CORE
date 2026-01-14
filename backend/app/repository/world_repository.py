from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict
import uuid

import asyncpg
import json

from app.dependencies import get_db_pool


class WorldRecord(TypedDict):
    id: str
    name: str
    origin: str
    created_at: str
    updated_at: str


async def create_world(name: str, *, origin: str = "human", tags: Optional[List[str]] = None) -> str:
    """Create a world and return its UUID.

    Args:
        name: Human-readable world name.
    Returns:
        The created world's UUID string.
    """
    world_id = str(uuid.uuid4())
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        try:
            await conn.execute(
                """
                INSERT INTO worlds (id, name, origin, tags)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (id) DO NOTHING
                """,
                world_id,
                name,
                origin,
                tags or [],
            )
        except Exception:
            # Fallback for older schemas that do not yet have origin/tags
            await conn.execute(
                """
                INSERT INTO worlds (id, name)
                VALUES ($1, $2)
                ON CONFLICT (id) DO NOTHING
                """,
                world_id,
                name,
            )
    return world_id


async def list_worlds(limit: int = 20, offset: int = 0) -> List[WorldRecord]:
    """List worlds ordered by most recently updated."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, name, origin, created_at, updated_at
            FROM worlds
            ORDER BY updated_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
        )
    return [
        {
            "id": str(r["id"]),
            "name": r["name"],
            "origin": r["origin"] or "human",
            "created_at": r["created_at"].isoformat() if r["created_at"] else "",
            "updated_at": r["updated_at"].isoformat() if r["updated_at"] else "",
        }
        for r in rows
    ]


async def create_snapshot(
    world_id: str,
    *,
    config: Dict[str, Any],
    layers: Optional[Dict[str, Any]] = None,
    tiles: Optional[List[Dict[str, Any]]] = None,
    preview: Optional[str] = None,
) -> str:
    """Create a snapshot for a world and bump world's updated_at."""
    snapshot_id = str(uuid.uuid4())
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Pass dicts directly - asyncpg serializes to JSONB properly
            await conn.execute(
                """
                INSERT INTO world_snapshots (id, world_id, config, layers, tiles, preview)
                VALUES ($1, $2, $3::jsonb, $4::jsonb, $5::jsonb, $6)
                """,
                snapshot_id,
                world_id,
                json.dumps(config),  # Cast to jsonb in query
                json.dumps(layers) if layers is not None else None,
                json.dumps(tiles) if tiles is not None else None,
                preview,
            )
            await conn.execute(
                """
                UPDATE worlds
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = $1
                """,
                world_id,
            )
    return snapshot_id


def _parse_jsonb_field(value: Any) -> Any:
    """Parse a JSONB field that might be a string (legacy) or already a dict."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


async def get_latest_snapshot(world_id: str) -> Optional[Dict[str, Any]]:
    """Return the latest snapshot for a world or None."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, created_at, config, layers, tiles, preview
            FROM world_snapshots
            WHERE world_id = $1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            world_id,
        )
        if row is None:
            return None
        return {
            "id": str(row["id"]),
            "created_at": row["created_at"].isoformat() if row["created_at"] else "",
            "config": _parse_jsonb_field(row["config"]),
            "layers": _parse_jsonb_field(row["layers"]),
            "tiles": _parse_jsonb_field(row["tiles"]),
            "preview": row["preview"],
        }


async def list_snapshots(world_id: str, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
    """List snapshots for a world ordered by newest first."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, created_at
            FROM world_snapshots
            WHERE world_id = $1
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
            """,
            world_id,
            limit,
            offset,
        )
    return [
        {
            "id": str(r["id"]),
            "created_at": r["created_at"].isoformat() if r["created_at"] else "",
        }
        for r in rows
    ]


async def delete_world(world_id: str) -> None:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM worlds WHERE id = $1", world_id)


async def delete_snapshot(world_id: str, snapshot_id: str) -> None:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM world_snapshots WHERE id = $1 AND world_id = $2",
            snapshot_id,
            world_id,
        )


async def update_world_name(world_id: str, name: str) -> None:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE worlds
            SET name = $2, updated_at = CURRENT_TIMESTAMP
            WHERE id = $1
            """,
            world_id,
            name,
        )


async def get_world_by_name(name: str) -> Optional[WorldRecord]:
    """Find a world by its exact name (case-insensitive). Returns None if not found."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, name, origin, created_at, updated_at
            FROM worlds
            WHERE LOWER(name) = LOWER($1)
            LIMIT 1
            """,
            name,
        )
        if row is None:
            return None
        return {
            "id": str(row["id"]),
            "name": row["name"],
            "origin": row["origin"] or "human",
            "created_at": row["created_at"].isoformat() if row["created_at"] else "",
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else "",
        }


