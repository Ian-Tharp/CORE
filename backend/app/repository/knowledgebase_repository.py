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
    doc_embedding: Optional[List[float]] = None,
    embedding_model: Optional[str] = None,
    embedding_dimensions: Optional[int] = None,
) -> str:
    doc_id = str(uuid.uuid4())
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO kb_documents (
                id, filename, original_name, size, mime_type, description, is_global, source, status,
                storage_path, doc_embedding, embedding_model, embedding_dimensions
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
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
            doc_embedding,
            embedding_model,
            embedding_dimensions,
        )
    return doc_id


async def update_document_embedding(
    *,
    document_id: str,
    embedding: List[float],
    model: str,
    dimensions: int,
) -> None:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE kb_documents
            SET doc_embedding = $2::jsonb, embedding_model = $3, embedding_dimensions = $4,
                last_modified = CURRENT_TIMESTAMP
            WHERE id = $1
            """,
            document_id,
            _json.dumps(embedding),
            model,
            dimensions,
        )


async def insert_chunk_embeddings(
    *,
    document_id: str,
    chunks: List[Tuple[int, str, List[float]]],
    model: str,
    dimensions: int,
) -> None:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        records = [
            (
                str(uuid.uuid4()),
                document_id,
                chunk_index,
                text,
                _json.dumps(embedding),
                model,
                dimensions,
            )
            for (chunk_index, text, embedding) in chunks
        ]
        await conn.executemany(
            """
            INSERT INTO kb_chunks (
                id, document_id, chunk_index, text, embedding, embedding_model, embedding_dimensions
            ) VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
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
            SELECT id, filename, original_name, size, mime_type, upload_date, last_modified,
                   is_global, coalesce(description,'') AS description, source, status
            FROM kb_documents
            {where_sql}
            ORDER BY upload_date DESC
            """,
            *params,
        )
        return [dict(r) for r in rows]


async def get_document(document_id: str) -> Optional[Dict[str, Any]]:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, filename, original_name, size, mime_type, upload_date, last_modified,
                   is_global, coalesce(description,'') AS description, source, status, storage_path,
                   doc_embedding, embedding_model, embedding_dimensions
            FROM kb_documents
            WHERE id = $1
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
            SELECT id, document_id, chunk_index, text, embedding, embedding_model, embedding_dimensions
            FROM kb_chunks
            WHERE document_id = $1
            ORDER BY chunk_index ASC
            """,
            document_id,
        )
        return [dict(r) for r in rows]


async def list_all_doc_embeddings() -> List[Dict[str, Any]]:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, coalesce(description,'') AS description, filename, original_name,
                   doc_embedding, embedding_model, embedding_dimensions
            FROM kb_documents
            WHERE doc_embedding IS NOT NULL
            """
        )
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            emb = d.get("doc_embedding")
            if isinstance(emb, str):
                try:
                    d["doc_embedding"] = _json.loads(emb)
                except Exception:
                    d["doc_embedding"] = []
            out.append(d)
        return out


async def list_chunks_for_documents(document_ids: List[str]) -> List[Dict[str, Any]]:
    if not document_ids:
        return []
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, document_id, chunk_index, text, embedding, embedding_model, embedding_dimensions
            FROM kb_chunks
            WHERE document_id = ANY($1::uuid[])
            ORDER BY document_id, chunk_index ASC
            """,
            document_ids,
        )
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            emb = d.get("embedding")
            if isinstance(emb, str):
                try:
                    d["embedding"] = _json.loads(emb)
                except Exception:
                    d["embedding"] = []
            out.append(d)
        return out


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


