from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import uuid

import asyncpg

from app.dependencies import get_db_pool
import json as _json


async def create_document(
    *,
    filename: str,
    original_name: str,
    size: int,
    mime_type: str,
    storage_path: str,
    description: Optional[str] = None,
    is_global: bool = False,
    source: str = "user_upload",
    status: str = "ready",
    title: Optional[str] = None,
    file_hash: Optional[str] = None,
) -> str:
    doc_id = str(uuid.uuid4())
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO kb_documents (
                id, filename, original_name, size, mime_type, description, is_global, source, status,
                storage_path, title, file_hash
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
            doc_id,
            filename,
            original_name,
            size,
            mime_type,
            description,
            is_global,
            source,
            status,
            storage_path,
            title,
            file_hash,
        )
    return doc_id


async def update_document_embedding(
    *,
    document_id: str,
    embedding: List[float],
    model: str,
    dimensions: int,
) -> None:
    """Update the document-level vector embedding."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        vec_text = f"[{', '.join(str(x) for x in embedding)}]"
        await conn.execute(
            """
            UPDATE kb_documents
            SET doc_embedding_vec = $2::vector(768),
                embedding_model = $3,
                embedding_dimensions = $4,
                last_modified = CURRENT_TIMESTAMP
            WHERE id = $1
            """,
            document_id,
            vec_text,
            model,
            dimensions,
        )


async def insert_chunk_embeddings(
    *,
    document_id: str,
    items: List[Tuple[int, str, List[float]]],
    model: str,
    dimensions: int,
) -> None:
    """Insert new kb_chunks rows with text and vector embeddings."""
    if not items:
        return
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        records = []
        for (chunk_index, text, embedding) in items:
            vec_text = f"[{', '.join(str(x) for x in embedding)}]"
            records.append((
                str(uuid.uuid4()),
                document_id,
                chunk_index,
                text,
                model,
                dimensions,
                vec_text,
            ))
        await conn.executemany(
            """
            INSERT INTO kb_chunks (
                id, document_id, chunk_index, text, embedding_model, embedding_dimensions, embedding_vec
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7::vector(768)
            )
            """,
            records,
        )


async def update_chunk_embeddings(
    *,
    document_id: str,
    chunks: List[Tuple[int, List[float]]],
    model: str,
    dimensions: int,
) -> None:
    """Update vector embeddings for existing chunks by (document_id, chunk_index)."""
    if not chunks:
        return
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        records = []
        for (chunk_index, embedding) in chunks:
            vec_text = f"[{', '.join(str(x) for x in embedding)}]"
            records.append((document_id, chunk_index, vec_text, model, dimensions))
        await conn.executemany(
            """
            UPDATE kb_chunks
            SET embedding_vec = $3::vector(768),
                embedding_model = $4,
                embedding_dimensions = $5
            WHERE document_id = $1 AND chunk_index = $2
            """,
            records,
        )


async def list_documents(*, q: Optional[str] = None, is_global: Optional[bool] = None) -> List[Dict[str, Any]]:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        where_clauses = []
        params: List[Any] = []
        if q:
            where_clauses.append("(LOWER(filename) LIKE $1 OR LOWER(original_name) LIKE $1 OR LOWER(coalesce(description,'')) LIKE $1)")
            params.append(f"%{q.lower()}%")
        if is_global is not None:
            where_clauses.append("is_global = $%d" % (len(params) + 1))
            params.append(is_global)
        where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        rows = await conn.fetch(
            f"""
            SELECT d.id, d.filename, d.original_name, d.size, d.mime_type, d.upload_date, d.last_modified,
                   d.is_global, coalesce(d.description,'') AS description, d.source, d.status,
                   coalesce(d.title, '') AS title,
                   COALESCE((SELECT COUNT(1) FROM kb_chunks c WHERE c.document_id = d.id), 0) AS chunk_count,
                   d.embedding_model, d.embedding_dimensions
            FROM kb_documents d
            {where_sql}
            ORDER BY d.upload_date DESC
            """,
            *params,
        )
        return [dict(r) for r in rows]


async def get_document(document_id: str) -> Optional[Dict[str, Any]]:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT d.id, d.filename, d.original_name, d.size, d.mime_type, d.upload_date, d.last_modified,
                   d.is_global, coalesce(d.description,'') AS description, d.source, d.status, d.storage_path,
                   d.embedding_model, d.embedding_dimensions,
                   coalesce(d.title, '') AS title,
                   COALESCE((SELECT COUNT(1) FROM kb_chunks c WHERE c.document_id = d.id), 0) AS chunk_count
            FROM kb_documents d
            WHERE d.id = $1
            """,
            document_id,
        )
        return dict(row) if row else None


async def delete_document(document_id: str) -> None:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM kb_documents WHERE id = $1", document_id)


async def list_chunks_for_document(document_id: str) -> List[Dict[str, Any]]:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, document_id, chunk_index, text, embedding_model, embedding_dimensions
            FROM kb_chunks
            WHERE document_id = $1
            ORDER BY chunk_index ASC
            """,
            document_id,
        )
        return [dict(r) for r in rows]


async def list_chunks_for_documents(document_ids: List[str]) -> List[Dict[str, Any]]:
    if not document_ids:
        return []
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, document_id, chunk_index, text, embedding_model, embedding_dimensions
            FROM kb_chunks
            WHERE document_id = ANY($1::uuid[])
            ORDER BY document_id, chunk_index ASC
            """,
            document_ids,
        )
        return [dict(r) for r in rows]


async def search_chunks_by_vector(
    *, query_vec: List[float], limit: int = 20, document_filter: Optional[List[str]] = None, model: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Search chunks by vector similarity (cosine distance via HNSW index)."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        vec_text = f"[{', '.join(str(x) for x in query_vec)}]"
        where = ["embedding_vec IS NOT NULL"]
        params: List[Any] = []
        param_offset = 2  # $1 is vec, $2 is limit
        if document_filter:
            where.append(f"document_id = ANY(${param_offset + 1}::uuid[])")
            params.append(document_filter)
            param_offset += 1
        if model:
            where.append(f"embedding_model = ${param_offset + 1}")
            params.append(model)
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        rows = await conn.fetch(
            f"""
            SELECT id, document_id, chunk_index, text, embedding_model, embedding_dimensions,
                   (embedding_vec <=> $1::vector(768)) AS distance
            FROM kb_chunks
            {where_sql}
            ORDER BY embedding_vec <=> $1::vector(768) ASC
            LIMIT $2
            """,
            vec_text,
            limit,
            *params,
        )
        return [dict(r) for r in rows]


async def update_document_description(document_id: str, description: str) -> None:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE kb_documents
            SET description = $2, last_modified = CURRENT_TIMESTAMP
            WHERE id = $1
            """,
            document_id,
            description,
        )


async def update_document_title_and_description(
    *, document_id: str, title: Optional[str] = None, description: Optional[str] = None
) -> None:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        sets = []
        params: List[Any] = [document_id]
        if title is not None:
            sets.append("title = $%d" % (len(params) + 1))
            params.append(title)
        if description is not None:
            sets.append("description = $%d" % (len(params) + 1))
            params.append(description)
        if not sets:
            return
        set_sql = ", ".join(sets) + ", last_modified = CURRENT_TIMESTAMP"
        await conn.execute(
            f"UPDATE kb_documents SET {set_sql} WHERE id = $1",
            *params,
        )

async def get_document_by_hash(file_hash: str) -> Optional[Dict[str, Any]]:
    if not file_hash:
        return None
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT d.id, d.filename, d.original_name, d.size, d.mime_type, d.upload_date, d.last_modified,
                   d.is_global, coalesce(d.description,'') AS description, d.source, d.status, d.storage_path,
                   d.embedding_model, d.embedding_dimensions,
                   coalesce(d.title, '') AS title,
                   COALESCE((SELECT COUNT(1) FROM kb_chunks c WHERE c.document_id = d.id), 0) AS chunk_count
            FROM kb_documents d
            WHERE file_hash = $1
            """,
            file_hash,
        )
        return dict(row) if row else None

async def insert_activity(
    *,
    action: str,
    document_id: Optional[str],
    file_name: Optional[str],
    user_id: Optional[str] = None,
    details: Optional[str] = None,
) -> str:
    activity_id = str(uuid.uuid4())
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO kb_activity (id, action, document_id, file_name, user_id, details)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            activity_id,
            action,
            document_id,
            file_name,
            user_id,
            details,
        )
    return activity_id


async def list_recent_activity(limit: int = 20) -> List[Dict[str, Any]]:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, action, document_id, file_name, user_id, details, timestamp
            FROM kb_activity
            ORDER BY timestamp DESC
            LIMIT $1
            """,
            limit,
        )
        return [dict(r) for r in rows]
