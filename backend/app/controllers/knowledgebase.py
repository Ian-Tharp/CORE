from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import mimetypes
import uuid
import hashlib
import logging

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.services import knowledgebase_service as svc
from app.repository import knowledgebase_repository as repo
from app.auth import require_api_key


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/knowledgebase", tags=["knowledgebase"])


class UploadData(BaseModel):
    tags: Optional[List[str]] = None
    description: Optional[str] = None
    isGlobal: Optional[bool] = False
    metadata: Optional[Dict[str, Any]] = None
    processImmediately: Optional[bool] = True
    localModel: Optional[str] = None


@router.get("/files")
async def list_files(
    q: Optional[str] = None,
    global_: Optional[bool] = None,
    api_key: str = Depends(require_api_key),
) -> List[Dict[str, Any]]:
    docs = await repo.list_documents(q=q, is_global=global_)
    return [
        {
            "id": d["id"],
            "filename": d["filename"],
            "originalName": d["original_name"],
            "title": d.get("title") or "",
            "chunkCount": d.get("chunk_count", 0),
            "embeddingModel": d.get("embedding_model"),
            "embeddingDimensions": d.get("embedding_dimensions"),
            "size": d["size"],
            "mimeType": d["mime_type"],
            "uploadDate": d["upload_date"],
            "lastModified": d["last_modified"],
            "isGlobal": d["is_global"],
            "description": d.get("description") or "",
            "source": d.get("source") or "user_upload",
            "status": d.get("status") or "ready",
        }
        for d in docs
    ]


@router.get("/files/{file_id}")
async def get_file(
    file_id: str, api_key: str = Depends(require_api_key)
) -> Dict[str, Any]:
    doc = await repo.get_document(file_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="File not found"
        )
    return {
        "id": doc["id"],
        "filename": doc["filename"],
        "originalName": doc["original_name"],
        "title": doc.get("title") or "",
        "chunkCount": doc.get("chunk_count", 0),
        "embeddingModel": doc.get("embedding_model"),
        "embeddingDimensions": doc.get("embedding_dimensions"),
        "size": doc["size"],
        "mimeType": doc["mime_type"],
        "uploadDate": doc["upload_date"],
        "lastModified": doc["last_modified"],
        "isGlobal": doc["is_global"],
        "description": doc.get("description") or "",
        "source": doc.get("source") or "user_upload",
        "status": doc.get("status") or "ready",
    }


@router.delete("/files/{file_id}")
async def delete_file(
    file_id: str, api_key: str = Depends(require_api_key)
) -> Dict[str, str]:
    doc = await repo.get_document(file_id)
    if doc and doc.get("storage_path") and os.path.exists(doc["storage_path"]):
        try:
            os.remove(doc["storage_path"])
        except Exception:
            pass
    await repo.delete_document(file_id)
    try:
        await repo.insert_activity(
            action="delete",
            document_id=file_id,
            file_name=(
                doc.get("title")
                or doc.get("original_name")
                or doc.get("filename")
                or ""
            ),
            user_id=None,
            details=None,
        )
    except Exception:
        pass
    return {"status": "ok"}


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    data: str = Form("{}"),
    api_key: str = Depends(require_api_key),
) -> Dict[str, Any]:
    try:
        import json

        payload = UploadData(**json.loads(data or "{}"))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid form data"
        )

    # Save file to storage
    storage_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "knowledgebase"
    )
    storage_dir = os.path.abspath(storage_dir)
    os.makedirs(storage_dir, exist_ok=True)

    ext = os.path.splitext(file.filename or "")[1]
    stored_name = f"{uuid.uuid4().hex}{ext}"
    storage_path = os.path.join(storage_dir, stored_name)

    MAX_UPLOAD_SIZE = 200 * 1024 * 1024  # 200MB
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large (max {MAX_UPLOAD_SIZE // (1024*1024)}MB)",
        )
    with open(storage_path, "wb") as out:
        out.write(content)

    # Compute a content hash for duplicate detection
    file_hash = hashlib.sha256(content).hexdigest()

    # Reject duplicate files
    existing = await repo.get_document_by_hash(file_hash)
    if existing:
        try:
            os.remove(storage_path)
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="File already exists in knowledgebase",
        )

    # Try to infer mime if missing
    mime_type = file.content_type or (
        mimetypes.guess_type(file.filename or "")[0] or "application/octet-stream"
    )

    # Optionally process immediately
    if payload.processImmediately:
        doc_id = await svc.process_uploaded_file(
            storage_path=storage_path,
            original_name=file.filename or stored_name,
            mime_type=mime_type,
            description=payload.description,
            is_global=bool(payload.isGlobal),
            file_hash=file_hash,
            local_model=payload.localModel,
        )
        doc = await repo.get_document(doc_id)
        # Log upload activity
        try:
            await repo.insert_activity(
                action="upload",
                document_id=doc_id,
                file_name=(
                    doc.get("title")
                    or doc.get("original_name")
                    or doc.get("filename")
                    or ""
                ),
                user_id=None,
                details=f"mime={mime_type}; size={doc.get('size', 0)}",
            )
        except Exception:
            pass
    else:
        doc_id = await repo.create_document(
            filename=stored_name,
            original_name=file.filename or stored_name,
            size=os.stat(storage_path).st_size,
            mime_type=mime_type,
            storage_path=storage_path,
            description=payload.description,
            is_global=bool(payload.isGlobal),
            status="processing",
            file_hash=file_hash,
        )
        doc = await repo.get_document(doc_id)

    return {
        "id": doc["id"],
        "filename": doc["filename"],
        "originalName": doc["original_name"],
        "title": doc.get("title") or "",
        "size": doc["size"],
        "mimeType": doc["mime_type"],
        "uploadDate": doc["upload_date"],
        "lastModified": doc["last_modified"],
        "isGlobal": doc["is_global"],
        "description": doc.get("description") or "",
        "source": doc.get("source") or "user_upload",
        "status": doc.get("status") or "ready",
    }


@router.post("/files/{file_id}/process")
async def process_file(
    file_id: str, api_key: str = Depends(require_api_key)
) -> Dict[str, Any]:
    doc = await repo.get_document(file_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="File not found"
        )
    # Re-process: delete old chunks, re-extract text, re-embed with local Ollama
    await svc.reprocess_document(
        document_id=file_id,
        storage_path=doc["storage_path"],
        original_name=doc["original_name"],
        mime_type=doc["mime_type"],
        description=doc.get("description"),
        local_model="nomic-embed-text",
    )
    try:
        await repo.insert_activity(
            action="process",
            document_id=file_id,
            file_name=(
                doc.get("title")
                or doc.get("original_name")
                or doc.get("filename")
                or ""
            ),
            user_id=None,
            details="re-processed document with OCR fallback",
        )
    except Exception:
        pass
    return {"fileId": file_id, "status": "ready"}


class BatchUploadRequest(BaseModel):
    directory: str
    recursive: bool = True
    extensions: Optional[List[str]] = None
    processImmediately: Optional[bool] = True
    isGlobal: Optional[bool] = False
    localModel: Optional[str] = None
    description: Optional[str] = None


@router.post("/batch-upload")
async def batch_upload(
    payload: BatchUploadRequest, api_key: str = Depends(require_api_key)
) -> Dict[str, Any]:
    """Batch-upload files from a directory.

    NOTE: The directory path must be as seen from INSIDE the Docker container
    (e.g. /data/documents), not the host path.
    """
    import time
    import glob

    directory = payload.directory
    if not os.path.isdir(directory):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Directory not found: {directory}",
        )

    # Scan for files
    extensions = payload.extensions or [".pdf", ".txt", ".md", ".docx", ".json"]
    found_files: List[str] = []

    if payload.recursive:
        for root, _dirs, files in os.walk(directory):
            for fname in files:
                if any(fname.lower().endswith(ext) for ext in extensions):
                    found_files.append(os.path.join(root, fname))
    else:
        for fname in os.listdir(directory):
            fpath = os.path.join(directory, fname)
            if os.path.isfile(fpath) and any(
                fname.lower().endswith(ext) for ext in extensions
            ):
                found_files.append(fpath)

    # Skip duplicates by file hash
    files_to_process: List[str] = []
    skipped = 0
    for fpath in found_files:
        try:
            with open(fpath, "rb") as f:
                fhash = hashlib.sha256(f.read()).hexdigest()
            existing = await repo.get_document_by_hash(fhash)
            if existing:
                skipped += 1
                continue
        except Exception:
            pass
        files_to_process.append(fpath)

    if not files_to_process:
        return {
            "filesFound": len(found_files),
            "filesSkipped": skipped,
            "filesProcessed": 0,
            "totalChunks": 0,
            "totalTime": 0,
            "results": [],
        }

    result = await svc.batch_upload_and_process(
        file_paths=files_to_process,
        is_global=bool(payload.isGlobal),
        local_model=payload.localModel,
        description=payload.description,
    )

    return {
        "filesFound": len(found_files),
        "filesSkipped": skipped,
        "filesProcessed": len(result["doc_ids"]),
        "totalChunks": result["total_chunks"],
        "totalTime": result["total_time"],
        "results": result["stats"],
    }


class SemanticSearchRequest(BaseModel):
    query: str
    limit: int = 10
    localModel: str | None = None


@router.post("/semantic-search")
async def semantic_search(
    payload: SemanticSearchRequest, api_key: str = Depends(require_api_key)
) -> List[Dict[str, Any]]:
    ctx = await svc.retrieve_context(
        query=payload.query,
        mode="all",
        max_docs=payload.limit,
        local_model=(payload.localModel or "nomic-embed-text"),
    )
    # Aggregate best (lowest) chunk distance per document
    best_dist: Dict[str, float] = {}
    for chunk in ctx.get("chunks", []):
        did = chunk.get("document_id")
        dist = chunk.get("distance", 1.0)
        if did and (did not in best_dist or dist < best_dist[did]):
            best_dist[did] = dist

    doc_ids = set(ctx.get("doc_ids", []))
    out: List[Dict[str, Any]] = []
    for doc_id in doc_ids:
        d = await repo.get_document(doc_id)
        if not d:
            continue
        # Convert cosine distance to similarity (1 - distance)
        similarity = round(1.0 - best_dist.get(doc_id, 0.0), 4)
        out.append(
            {
                "id": d["id"],
                "filename": d["filename"],
                "originalName": d["original_name"],
                "title": d.get("title") or "",
                "chunkCount": d.get("chunk_count", 0),
                "embeddingModel": d.get("embedding_model"),
                "embeddingDimensions": d.get("embedding_dimensions"),
                "size": d["size"],
                "mimeType": d["mime_type"],
                "uploadDate": d["upload_date"],
                "lastModified": d["last_modified"],
                "isGlobal": d["is_global"],
                "description": d.get("description") or "",
                "source": d.get("source") or "user_upload",
                "status": d.get("status") or "ready",
                "similarity": similarity,
            }
        )
    # Sort by similarity descending (most relevant first)
    out.sort(key=lambda x: x["similarity"], reverse=True)
    return out


class ChunkSearchRequest(BaseModel):
    query: str
    limit: int = 10
    fileId: str | None = None
    localModel: str | None = None


class InstanceSearchRequest(BaseModel):
    query: str
    instanceName: str
    limit: int = 10
    localModel: str | None = None


@router.post("/chunk-search")
async def chunk_search(
    payload: ChunkSearchRequest, api_key: str = Depends(require_api_key)
) -> List[Dict[str, Any]]:
    """Search at the chunk level — returns actual text passages with similarity scores and source document info."""
    file_filter = [payload.fileId] if payload.fileId else None
    ctx = await svc.retrieve_context(
        query=payload.query,
        mode="file" if payload.fileId else "all",
        file_id=payload.fileId,
        max_docs=payload.limit,
        max_chunks=payload.limit,
        local_model=(payload.localModel or "nomic-embed-text"),
    )
    chunks = ctx.get("chunks", [])
    # Enrich each chunk with document metadata
    doc_cache: Dict[str, Dict[str, Any]] = {}
    out: List[Dict[str, Any]] = []
    for chunk in chunks:
        did = chunk.get("document_id")
        if did and did not in doc_cache:
            doc = await repo.get_document(did)
            doc_cache[did] = doc or {}
        doc = doc_cache.get(did, {})
        similarity = round(1.0 - chunk.get("distance", 0.0), 4)
        out.append(
            {
                "chunkId": str(chunk.get("id", "")),
                "documentId": did,
                "filename": doc.get("original_name") or doc.get("filename", ""),
                "title": doc.get("title") or "",
                "chunkIndex": chunk.get("chunk_index"),
                "text": chunk.get("text", ""),
                "similarity": similarity,
            }
        )
    out.sort(key=lambda x: x["similarity"], reverse=True)
    return out


@router.post("/instance-search")
async def instance_search(
    payload: InstanceSearchRequest, api_key: str = Depends(require_api_key)
) -> List[Dict[str, Any]]:
    """Search chunks filtered by instance perspective - 'what does [instance] think about X?' queries."""
    ctx = await svc.retrieve_context_by_instance(
        query=payload.query,
        instance_name=payload.instanceName,
        max_chunks=payload.limit,
        local_model=(payload.localModel or "nomic-embed-text"),
    )
    chunks = ctx.get("chunks", [])

    # Enrich each chunk with document metadata
    doc_cache: Dict[str, Dict[str, Any]] = {}
    out: List[Dict[str, Any]] = []
    for chunk in chunks:
        did = chunk.get("document_id")
        if did and did not in doc_cache:
            doc = await repo.get_document(did)
            doc_cache[did] = doc or {}
        doc = doc_cache.get(did, {})
        similarity = round(1.0 - chunk.get("distance", 0.0), 4)
        out.append(
            {
                "chunkId": str(chunk.get("id", "")),
                "documentId": did,
                "filename": doc.get("original_name") or doc.get("filename", ""),
                "title": doc.get("title") or "",
                "chunkIndex": chunk.get("chunk_index"),
                "text": chunk.get("text", ""),
                "similarity": similarity,
                "instanceName": chunk.get("instance_name"),
                "instanceTimestamp": chunk.get("instance_timestamp"),
                "sourceDiscussion": chunk.get("source_discussion"),
            }
        )
    out.sort(key=lambda x: x["similarity"], reverse=True)
    return out


@router.get("/instances")
async def list_instances(
    api_key: str = Depends(require_api_key),
) -> List[Dict[str, Any]]:
    """List all instance names that have contributed knowledge with basic stats."""
    instances = await repo.list_instances()
    return [
        {
            "instanceName": r["instance_name"],
            "chunkCount": r["chunk_count"],
            "documentCount": r["document_count"],
            "firstContribution": r["first_contribution"],
            "lastContribution": r["last_contribution"],
        }
        for r in instances
    ]


@router.post("/files/{file_id}/embed-local")
async def embed_local(
    file_id: str, model: str, api_key: str = Depends(require_api_key)
) -> Dict[str, Any]:
    doc = await repo.get_document(file_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="File not found"
        )
    await svc.embed_document_locally(document_id=file_id, model=model)
    updated = await repo.get_document(file_id)
    return {
        "fileId": file_id,
        "status": "ok",
        "embeddingModel": updated.get("embedding_model") if updated else None,
        "embeddingDimensions": updated.get("embedding_dimensions") if updated else None,
    }


@router.post("/reindex")
async def reindex(
    model: str, only_missing: bool = True, api_key: str = Depends(require_api_key)
) -> Dict[str, Any]:
    """Re-embed all documents using a local model. When only_missing is true, skip docs that already have local vectors."""
    docs = await repo.list_documents()
    count = 0
    for d in docs:
        if only_missing and d.get("embedding_model"):
            continue
        await svc.embed_document_locally(document_id=d["id"], model=model)
        count += 1
    return {"status": "ok", "processed": count}


@router.get("/activity")
async def recent_activity(
    limit: int = 20, api_key: str = Depends(require_api_key)
) -> List[Dict[str, Any]]:
    try:
        rows = await repo.list_recent_activity(limit=limit)
        return [
            {
                "id": r["id"],
                "action": r["action"],
                "fileId": r.get("document_id") or "",
                "fileName": r.get("file_name") or "",
                "userId": r.get("user_id") or "",
                "timestamp": r.get("timestamp"),
                "details": r.get("details") or "",
            }
            for r in rows
        ]
    except Exception:
        return []


@router.post("/files/{file_id}/reextract-title")
async def reextract_title(
    file_id: str, api_key: str = Depends(require_api_key)
) -> Dict[str, Any]:
    doc = await repo.get_document(file_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="File not found"
        )

    try:
        title = await svc.reextract_title_for_document(
            storage_path=doc["storage_path"], mime_type=doc["mime_type"]
        )
        if title:
            await repo.update_document_title_and_description(
                document_id=file_id, title=title
            )
            try:
                await repo.insert_activity(
                    action="annotate",
                    document_id=file_id,
                    file_name=title,
                    user_id=None,
                    details="auto re-extracted title",
                )
            except Exception:
                pass
            return {"fileId": file_id, "title": title, "updated": True}
        return {"fileId": file_id, "updated": False}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/stats")
async def get_stats(api_key: str = Depends(require_api_key)) -> Dict[str, Any]:
    """Return knowledgebase statistics."""
    try:
        docs = await repo.list_documents()
        total_files = len(docs)
        total_size = sum(d.get("size", 0) for d in docs)

        return {
            "totalFiles": total_files,
            "totalSize": total_size,
            "filesByType": {},
            "filesBySource": {},
            "totalEmbeddings": 0,
            "processingQueue": 0,
            "recentActivity": [],
        }
    except Exception:
        return {
            "totalFiles": 0,
            "totalSize": 0,
            "filesByType": {},
            "filesBySource": {},
            "totalEmbeddings": 0,
            "processingQueue": 0,
            "recentActivity": [],
        }


@router.get("/tags")
async def get_tags(api_key: str = Depends(require_api_key)) -> List[Dict[str, Any]]:
    """Return available file tags."""
    try:
        return []
    except Exception:
        return []


@router.get("/metrics/performance")
async def get_performance_metrics(
    hours: int = 24,
    operation: Optional[str] = None,
    summary: bool = False,
    api_key: str = Depends(require_api_key),
) -> Dict[str, Any]:
    """Get embedding performance metrics and statistics.

    Args:
        hours: Number of hours of history to retrieve (default 24)
        operation: Filter by operation type (single, batch, document)
        summary: If True, return aggregate summary instead of raw metrics
    """
    try:
        from app.services.metrics_service import (
            get_embedding_metrics,
            get_embedding_summary,
        )

        if summary:
            return get_embedding_summary(hours=hours)
        else:
            metrics = get_embedding_metrics(hours=hours, operation=operation)
            return {
                "metrics": metrics,
                "count": len(metrics),
                "period_hours": hours,
                "filtered_operation": operation,
            }
    except Exception as e:
        logger.error("Failed to retrieve performance metrics: %s", e)
        return {
            "error": "Metrics unavailable",
            "metrics": [],
            "count": 0,
            "period_hours": hours,
        }
