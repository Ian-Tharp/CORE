from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import asyncio
import os
import io
import logging
import math
import mimetypes
import shutil
import time

logger = logging.getLogger(__name__)

from app.repository import knowledgebase_repository as repo
from app.services.ollama_embeddings import embed_texts_via_ollama, embed_texts_batch


def _split_text(text: str, *, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[str]:
    if chunk_size <= 0:
        return [text]
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        if end == length:
            break
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return chunks


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) != len(vec_b):
        n = min(len(vec_a), len(vec_b))
        vec_a = vec_a[:n]
        vec_b = vec_b[:n]
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _embed_texts_local(*, model: str, texts: List[str], use_batch: bool = True) -> Tuple[List[List[float]], int]:
    """Embed texts locally via Ollama and return (vectors, original_dim).

    When use_batch is True, uses the batch /api/embed endpoint (with fallback).
    """
    if use_batch:
        return await embed_texts_batch(model=model, texts=texts)
    return await embed_texts_via_ollama(model=model, texts=texts)


async def process_uploaded_file(
    *,
    storage_path: str,
    original_name: str,
    mime_type: str,
    description: Optional[str],
    is_global: bool,
    file_hash: Optional[str] = None,
    local_model: Optional[str] = None,
) -> str:
    start_time = time.time()
    title, text = await _extract_title_and_text(storage_path, mime_type)
    model = local_model or "nomic-embed-text"

    # Auto-description: simple heuristic summary (first 1-2 sentences)
    auto_description: Optional[str] = None
    if not description:
        snippet = (text or "").strip()[:800]
        if snippet:
            parts = [p.strip() for p in snippet.split(".") if p.strip()]
            auto_description = ". ".join(parts[:2])[:200]

    # Create document entry
    stat = os.stat(storage_path)
    filename = os.path.basename(storage_path)
    doc_id = await repo.create_document(
        filename=filename,
        original_name=original_name,
        size=stat.st_size,
        mime_type=mime_type,
        storage_path=storage_path,
        description=description or auto_description,
        is_global=is_global,
        title=title,
        file_hash=file_hash,
    )

    # Document-level embedding
    title_desc = f"{(title or original_name)}\n\n{(description or auto_description or '')}".strip()
    if title_desc:
        vecs, orig = await _embed_texts_local(model=model, texts=[title_desc])
        if vecs:
            await repo.update_document_embedding(
                document_id=doc_id,
                embedding=vecs[0],
                model=model,
                dimensions=orig,
            )

    # Chunk text and embed
    chunks = _split_text(text)
    vecs_all: List[List[float]] = []
    dims = 0
    batch_size = 64
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        v, od = await _embed_texts_local(model=model, texts=batch)
        if not v:
            continue
        vecs_all.extend(v)
        if od and not dims:
            dims = od
    updates: List[Tuple[int, List[float]]] = []
    for idx, _ in enumerate(chunks):
        if idx < len(vecs_all):
            updates.append((idx, vecs_all[idx]))
    if updates:
        await repo.update_chunk_embeddings(
            document_id=doc_id,
            chunks=updates,
            model=model,
            dimensions=dims or 0,
        )

    # Record document processing performance
    elapsed_ms = (time.time() - start_time) * 1000
    total_chars = len(text or "")
    try:
        from app.services.metrics_service import record_embedding_performance
        record_embedding_performance("document", model, len(chunks), elapsed_ms, total_chars)
    except Exception:
        pass  # Don't fail on metrics errors

    return doc_id


async def reprocess_document(
    *,
    document_id: str,
    storage_path: str,
    original_name: str,
    mime_type: str,
    description: Optional[str] = None,
    local_model: Optional[str] = None,
) -> None:
    """Re-extract text and re-embed an existing document (delete old chunks first)."""
    logger.info("Reprocessing document %s (%s)", document_id, original_name)

    # 1. Extract text with OCR fallback
    title, text = await _extract_title_and_text(storage_path, mime_type)
    logger.info("Extracted %d chars, title=%r for %s", len(text or ""), title, document_id)

    # 2. Delete old chunks
    from app.dependencies import get_db_pool
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        deleted = await conn.execute("DELETE FROM kb_chunks WHERE document_id = $1", document_id)
        logger.info("Deleted old chunks for %s: %s", document_id, deleted)

    # 3. Update document title if extracted
    if title:
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE kb_documents SET title = $1 WHERE id = $2", title, document_id
            )

    model = local_model or "nomic-embed-text"

    # 4. Document-level embedding
    title_desc = f"{(title or original_name)}\n\n{(description or '')}".strip()
    if title_desc:
        vecs, orig = await _embed_texts_local(model=model, texts=[title_desc])
        if vecs:
            await repo.update_document_embedding(
                document_id=document_id,
                embedding=vecs[0],
                model=model,
                dimensions=orig,
            )

    # 5. Chunk text and embed
    chunks = _split_text(text)
    logger.info("Split into %d chunks for %s", len(chunks), document_id)

    if not chunks:
        logger.warning("No text chunks for %s — document may be empty or image-only", document_id)
        return

    vecs_all: List[List[float]] = []
    dims = 0
    batch_size = 64
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        v, od = await _embed_texts_local(model=model, texts=batch)
        if v:
            vecs_all.extend(v)
            if od and not dims:
                dims = od
    # Insert new chunks with embeddings
    chunk_payload: List[Tuple[int, str, List[float]]] = []
    for idx, chunk_text in enumerate(chunks):
        if idx < len(vecs_all):
            chunk_payload.append((idx, chunk_text, vecs_all[idx]))
    if chunk_payload:
        await repo.insert_chunk_embeddings(
            document_id=document_id,
            items=chunk_payload,
            model=model,
            dimensions=dims or 0,
        )
    logger.info("Reprocessing complete for %s: %d chunks embedded", document_id, len(chunks))


async def embed_document_locally(*, document_id: str, model: str) -> None:
    """Generate and persist local embeddings for a document and its chunks via Ollama."""
    doc = await repo.get_document(document_id)
    if not doc:
        return
    chunks = await repo.list_chunks_for_document(document_id)

    # Document-level embedding using title/description fallback to first chunk
    title = doc.get("title") or doc.get("original_name") or doc.get("filename") or ""
    desc = doc.get("description") or ""
    title_desc = (f"{title}\n\n{desc}").strip()
    doc_texts: List[str] = [title_desc] if title_desc else []
    if not doc_texts and chunks:
        doc_texts = [chunks[0].get("text") or ""]
    if doc_texts:
        doc_vecs, original_dim = await _embed_texts_local(model=model, texts=doc_texts)
        if doc_vecs:
            await repo.update_document_embedding(
                document_id=document_id,
                embedding=doc_vecs[0],
                model=model,
                dimensions=original_dim,
            )

    # Chunk-level embeddings
    chunk_texts = [c.get("text") or "" for c in chunks]
    if chunk_texts:
        batch_size = 64
        items: List[Tuple[int, str, List[float]]] = []
        original_dim_total = 0
        for start in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[start : start + batch_size]
            vecs, original_dim = await _embed_texts_local(model=model, texts=batch)
            original_dim_total = original_dim or original_dim_total
            for i, v in enumerate(vecs):
                items.append((start + i, batch[i], v))
        if items:
            update_tuples = [(idx, vec) for idx, _text, vec in items]
            await repo.update_chunk_embeddings(
                document_id=document_id,
                chunks=update_tuples,
                model=model,
                dimensions=original_dim_total or 0,
            )


async def _extract_text(path: str, mime_type: str) -> str:
    # Handle simple text types
    if mime_type.startswith("text/") or mime_type in ("application/json",):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    # Basic PDF support via pypdf if installed
    if mime_type == "application/pdf":
        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(path)
            text = []
            for page in reader.pages:
                try:
                    text.append(page.extract_text() or "")
                except Exception:
                    continue
            return "\n".join(text)
        except Exception:
            return ""

    # Basic DOCX support via python-docx if installed
    if mime_type in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"):
        try:
            import docx  # type: ignore

            doc = docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return ""

    # Fallback: try to guess and read as text
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


_PYMUPDF_MAX_OCR_PAGES = 500
_TESSERACT_AVAILABLE = shutil.which("tesseract") is not None


def _extract_pdf_with_pymupdf(
    path: str, existing_title: Optional[str]
) -> Tuple[str, Optional[str]]:
    """Extract text from a PDF using pymupdf (fitz), with OCR fallback per page."""
    import fitz  # pymupdf

    title = existing_title
    pages_text: List[str] = []

    try:
        doc = fitz.open(path)
    except Exception:
        logger.error("pymupdf could not open %s", path, exc_info=True)
        return "", title

    num_pages = min(len(doc), _PYMUPDF_MAX_OCR_PAGES)
    if len(doc) > _PYMUPDF_MAX_OCR_PAGES:
        logger.warning(
            "PDF has %d pages; limiting OCR to first %d pages: %s",
            len(doc), _PYMUPDF_MAX_OCR_PAGES, path,
        )

    for idx in range(num_pages):
        try:
            page = doc[idx]
            text = page.get_text("text") or ""
            if len(text.strip()) < 20 and _TESSERACT_AVAILABLE:
                try:
                    ocr_text = page.get_text("ocr") or ""
                    if len(ocr_text.strip()) > len(text.strip()):
                        text = ocr_text
                        logger.debug("OCR used for page %d of %s", idx, path)
                except Exception:
                    logger.debug("OCR failed for page %d of %s", idx, path, exc_info=True)
            pages_text.append(text)
        except Exception:
            logger.debug("pymupdf failed on page %d of %s", idx, path, exc_info=True)
            continue

    doc.close()

    full_text = "\n".join(pages_text)

    if not title and pages_text:
        for line in pages_text[0].splitlines():
            candidate = (line or "").strip()
            if 3 <= len(candidate) <= 140:
                title = candidate
                break

    logger.info("pymupdf extracted %d chars from %d pages: %s", len(full_text), num_pages, path)
    return full_text, title


async def _extract_title_and_text(path: str, mime_type: str, *, prefer_pymupdf: bool = False) -> Tuple[Optional[str], str]:
    """Best-effort extraction of a human-friendly title and full text.
    
    If prefer_pymupdf=True, use pymupdf first (faster, no hanging on malformed fonts).
    Falls back to pypdf only if pymupdf fails.
    """
    title: Optional[str] = None

    # PDF path
    if mime_type == "application/pdf":
        if prefer_pymupdf:
            # pymupdf-first path: faster and doesn't hang on font map issues
            try:
                text, fallback_title = _extract_pdf_with_pymupdf(path, None)
                if text and len(text.strip()) >= 100:
                    return fallback_title, text
                logger.info("pymupdf extracted < 100 chars for %s, trying pypdf", path)
            except Exception:
                logger.warning("pymupdf failed for %s, trying pypdf", path, exc_info=True)

        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(path)
            pages_text: List[str] = []
            first_page_text: str = ""
            for idx, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                    pages_text.append(page_text)
                    if idx == 0:
                        first_page_text = page_text
                except Exception:
                    continue

            try:
                meta = getattr(reader, "metadata", None)
                meta_title = None
                if meta is not None:
                    meta_title = getattr(meta, "title", None)
                if isinstance(meta_title, str):
                    cleaned = meta_title.strip()
                    title = cleaned if cleaned else None
            except Exception:
                title = None

            if not title and first_page_text:
                for line in (first_page_text.splitlines() or []):
                    candidate = (line or "").strip()
                    if 3 <= len(candidate) <= 140:
                        title = candidate
                        break

            full_text = "\n".join(pages_text)

            num_pages = len(reader.pages)
            if len(full_text.strip()) < 100 and num_pages > 0:
                logger.info(
                    "pypdf extracted < 100 chars from %d-page PDF, "
                    "falling back to pymupdf OCR: %s",
                    num_pages, path,
                )
                full_text, title = _extract_pdf_with_pymupdf(path, title)
            else:
                logger.debug("pypdf text extraction succeeded for %s", path)

            return title, full_text
        except Exception:
            logger.warning("pypdf failed for %s, trying pymupdf", path, exc_info=True)
            try:
                text, fallback_title = _extract_pdf_with_pymupdf(path, None)
                return fallback_title, text
            except Exception:
                logger.warning("pymupdf also failed for %s", path, exc_info=True)
            pass

    # DOCX path
    if mime_type in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"):
        try:
            import docx  # type: ignore

            doc = docx.Document(path)
            paragraphs = [p.text for p in doc.paragraphs]
            text = "\n".join(paragraphs)
            for p in paragraphs:
                candidate = (p or "").strip()
                if candidate:
                    title = candidate[:140]
                    break
            return title, text
        except Exception:
            pass

    # Plain text and everything else
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        if content:
            for line in content.splitlines():
                candidate = (line or "").strip()
                if candidate:
                    title = candidate[:140]
                    break
        return title, content
    except Exception:
        return None, ""


async def reextract_title_for_document(*, storage_path: str, mime_type: str) -> Optional[str]:
    """Re-extract a best-effort title from the stored document without re-embedding."""
    title, _ = await _extract_title_and_text(storage_path, mime_type)
    return title


async def retrieve_context(
    *,
    query: str,
    mode: str,
    file_id: Optional[str] = None,
    max_docs: int = 5,
    max_chunks: int = 8,
    local_model: Optional[str] = None,
) -> Dict[str, any]:
    model = (local_model or "nomic-embed-text").strip()
    qvecs, _orig = await _embed_texts_local(model=model, texts=[query])
    if not qvecs:
        return {"chunks": [], "doc_ids": []}
    query_vec = qvecs[0]

    # Determine candidate documents
    candidate_doc_ids: List[str] = []
    if mode == "file" and file_id:
        candidate_doc_ids = [file_id]

    # SQL vector search across local vectors
    doc_filter = candidate_doc_ids if mode == "file" else None
    rows = await repo.search_chunks_by_vector(
        query_vec=query_vec, limit=max_chunks, document_filter=doc_filter, model=local_model
    )
    return {"chunks": rows, "doc_ids": list({r.get("document_id") for r in rows})}


async def batch_upload_and_process(
    *,
    file_paths: List[str],
    is_global: bool = False,
    local_model: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict:
    """Process multiple files: extract text, chunk, batch-embed, store.

    Returns dict with doc_ids, stats per doc, and totals.
    """
    import time
    import hashlib

    model = local_model or "nomic-embed-text"
    start_time = time.monotonic()

    # Phase 1: Extract and chunk all files (with per-file timeout)
    file_infos: List[Dict] = []  # {path, title, text, chunks, error}
    extraction_timeout = 120  # seconds per file
    for i, fpath in enumerate(file_paths):
        logger.info("Processing file %d of %d: %s", i + 1, len(file_paths), fpath)
        try:
            mime = mimetypes.guess_type(fpath)[0] or "application/octet-stream"
            # Run extraction with timeout — pypdf can hang on malformed PDFs
            title, text = await asyncio.wait_for(
                _extract_title_and_text(fpath, mime, prefer_pymupdf=True),
                timeout=extraction_timeout,
            )
            # Strip null bytes — some PDFs contain \x00 that PostgreSQL rejects
            if text:
                text = text.replace("\x00", "")
            if title:
                title = title.replace("\x00", "")
            chunks = _split_text(text) if text else []
            file_infos.append({
                "path": fpath, "title": title, "text": text,
                "chunks": chunks, "mime": mime, "error": None,
            })
        except asyncio.TimeoutError:
            logger.error("Extraction timed out after %ds: %s", extraction_timeout, fpath)
            file_infos.append({"path": fpath, "chunks": [], "error": f"extraction timed out after {extraction_timeout}s"})
        except Exception as exc:
            logger.error("Failed to extract %s: %s", fpath, exc)
            file_infos.append({"path": fpath, "chunks": [], "error": str(exc)})

    # Phase 2: Collect all chunks and batch embed
    all_chunks: List[str] = []
    chunk_map: List[Tuple[int, int]] = []  # (file_index, chunk_index_within_file)
    for fi, info in enumerate(file_infos):
        if info.get("error"):
            continue
        for ci, chunk in enumerate(info["chunks"]):
            all_chunks.append(chunk)
            chunk_map.append((fi, ci))

    all_vecs: List[List[float]] = []
    dims = 0
    batch_size = 64
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size if all_chunks else 0
    for batch_idx, start in enumerate(range(0, len(all_chunks), batch_size)):
        logger.info("Embedding batch %d of %d", batch_idx + 1, total_batches)
        batch = all_chunks[start:start + batch_size]
        v, od = await embed_texts_batch(model=model, texts=batch)
        if v:
            all_vecs.extend(v)
            if od and not dims:
                dims = od

    # Phase 3: Store documents and chunks
    doc_ids: List[str] = []
    stats: List[Dict] = []
    vec_offset = 0

    for fi, info in enumerate(file_infos):
        if info.get("error"):
            stats.append({"file": info["path"], "error": info["error"], "chunks": 0})
            continue

        fpath = info["path"]
        try:
            stat = os.stat(fpath)
            filename = os.path.basename(fpath)
            with open(fpath, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            auto_desc = description
            if not auto_desc:
                snippet = (info.get("text") or "").strip()[:800]
                if snippet:
                    parts = [p.strip() for p in snippet.split(".") if p.strip()]
                    auto_desc = ". ".join(parts[:2])[:200]

            doc_id = await repo.create_document(
                filename=filename,
                original_name=filename,
                size=stat.st_size,
                mime_type=info["mime"],
                storage_path=fpath,
                description=auto_desc,
                is_global=is_global,
                title=info.get("title"),
                file_hash=file_hash,
            )

            # Document-level embedding
            title_desc = f"{(info.get('title') or filename)}\n\n{(auto_desc or '')}".strip()
            if title_desc:
                dvecs, dorig = await embed_texts_batch(model=model, texts=[title_desc])
                if dvecs:
                    await repo.update_document_embedding(
                        document_id=doc_id, embedding=dvecs[0],
                        model=model, dimensions=dorig,
                    )

            # Chunk embeddings — INSERT (not update, chunks don't exist yet)
            num_chunks = len(info["chunks"])
            inserts: List[Tuple[int, str, List[float]]] = []
            for ci in range(num_chunks):
                if vec_offset + ci < len(all_vecs):
                    inserts.append((ci, info["chunks"][ci], all_vecs[vec_offset + ci]))
            if inserts:
                await repo.insert_chunk_embeddings(
                    document_id=doc_id, items=inserts,
                    model=model, dimensions=dims or 0,
                )
            vec_offset += num_chunks

            doc_ids.append(doc_id)
            stats.append({"file": fpath, "doc_id": doc_id, "chunks": num_chunks})
        except Exception as exc:
            logger.error("Failed to store %s: %s", fpath, exc)
            stats.append({"file": fpath, "error": str(exc), "chunks": 0})
            vec_offset += len(info["chunks"])

    elapsed = time.monotonic() - start_time
    return {
        "doc_ids": doc_ids,
        "stats": stats,
        "total_chunks": len(all_chunks),
        "total_time": round(elapsed, 2),
    }


def build_rag_messages(original_messages: List[Dict[str, str]], *, context_chunks: List[Dict[str, any]]) -> List[Dict[str, str]]:
    if not context_chunks:
        return original_messages
    ctx_lines = []
    for ch in context_chunks:
        ref = f"doc:{ch['document_id']}#chunk:{ch['chunk_index']}"
        snippet = ch["text"]
        ctx_lines.append(f"[{ref}]\n{snippet}")
    context_block = "\n\n".join(ctx_lines)
    system_msg = {
        "role": "system",
        "content": (
            "You are a precise assistant. Use the provided knowledgebase context when relevant. "
            "If the context is insufficient, say so and answer from your general knowledge.\n\n"
            f"Context:\n{context_block}"
        ),
    }
    return [system_msg] + original_messages
