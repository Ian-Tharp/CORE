from __future__ import annotations

from typing import Any, Dict, List, Optional
import uuid

from app.dependencies import get_db_pool
import json


async def create_wiki_page(world_id: Optional[str], title: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    page_id = str(uuid.uuid4())
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO wiki_pages (id, world_id, title, content, metadata)
            VALUES ($1, $2, $3, $4, $5)
            """,
            page_id, world_id, title, content, metadata,
        )
    return page_id


async def update_wiki_page(page_id: str, *, title: Optional[str] = None, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE wiki_pages
            SET title = COALESCE($2, title),
                content = COALESCE($3, content),
                metadata = COALESCE($4, metadata),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = $1
            """,
            page_id, title, content, json.dumps(metadata) if metadata is not None else None,
        )


async def list_wiki_pages(world_id: Optional[str] = None) -> List[Dict[str, Any]]:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        if world_id:
            rows = await conn.fetch("SELECT id, world_id, title, content, metadata, created_at, updated_at FROM wiki_pages WHERE world_id = $1 ORDER BY updated_at DESC", world_id)
        else:
            rows = await conn.fetch("SELECT id, world_id, title, content, metadata, created_at, updated_at FROM wiki_pages ORDER BY updated_at DESC")
    return [
        {
            "id": str(r["id"]),
            "world_id": str(r["world_id"]) if r["world_id"] else None,
            "title": r["title"],
            "content": r["content"],
            "metadata": r["metadata"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else "",
            "updated_at": r["updated_at"].isoformat() if r["updated_at"] else "",
        }
        for r in rows
    ]


async def create_character(world_id: Optional[str], name: str, traits: Optional[Dict[str, Any]] = None) -> str:
    character_id = str(uuid.uuid4())
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO characters (id, world_id, name, traits)
            VALUES ($1, $2, $3, $4)
            """,
            character_id, world_id, name, json.dumps(traits) if traits is not None else None,
        )
    return character_id


async def update_character_image(character_id: str, image_b64: str) -> None:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE characters SET image_b64 = $2, updated_at = CURRENT_TIMESTAMP WHERE id = $1",
            character_id, image_b64,
        )


async def list_characters(world_id: Optional[str] = None) -> List[Dict[str, Any]]:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        if world_id:
            rows = await conn.fetch("SELECT id, world_id, name, traits, image_b64, created_at, updated_at FROM characters WHERE world_id = $1 ORDER BY updated_at DESC", world_id)
        else:
            rows = await conn.fetch("SELECT id, world_id, name, traits, image_b64, created_at, updated_at FROM characters ORDER BY updated_at DESC")
    return [
        {
            "id": str(r["id"]),
            "world_id": str(r["world_id"]) if r["world_id"] else None,
            "name": r["name"],
            "traits": r["traits"],
            "image_b64": r["image_b64"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else "",
            "updated_at": r["updated_at"].isoformat() if r["updated_at"] else "",
        }
        for r in rows
    ]


