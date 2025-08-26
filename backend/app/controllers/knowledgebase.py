from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import mimetypes
import uuid
import hashlib

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.services import knowledgebase_service as svc
from app.repository import knowledgebase_repository as repo


router = APIRouter(prefix="/knowledgebase", tags=["knowledgebase"])


class UploadData(BaseModel):
    tags: Optional[List[str]] = None
    description: Optional[str] = None
    isGlobal: Optional[bool] = False
    metadata: Optional[Dict[str, Any]] = None
    processImmediately: Optional[bool] = True


@router.get("/files")
async def list_files(q: Optional[str] = None, global_: Optional[bool] = None) -> List[Dict[str, Any]]:
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
async def get_file(file_id: str) -> Dict[str, Any]:
    doc = await repo.get_document(file_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
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
async def delete_file(file_id: str) -> Dict[str, str]:
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
            file_name=(doc.get("title") or doc.get("original_name") or doc.get("filename") or ""),
            user_id=None,
            details=None,
        )
    except Exception:
        # Do not block deletion on logging failures
        pass
    return {"status": "ok"}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...), data: str = Form("{}")) -> Dict[str, Any]:
    try:
        import json

        payload = UploadData(**json.loads(data or "{}"))
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid form data")

    # Save file to storage
    storage_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "knowledgebase")
    storage_dir = os.path.abspath(storage_dir)
    os.makedirs(storage_dir, exist_ok=True)

    ext = os.path.splitext(file.filename or "")[1]
    stored_name = f"{uuid.uuid4().hex}{ext}"
    storage_path = os.path.join(storage_dir, stored_name)

    content = await file.read()
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
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="File already exists in knowledgebase")

    # Try to infer mime if missing
    mime_type = file.content_type or (mimetypes.guess_type(file.filename or "")[0] or "application/octet-stream")

    # Optionally process immediately
    if payload.processImmediately:
        doc_id = await svc.process_uploaded_file(
            storage_path=storage_path,
            original_name=file.filename or stored_name,
            mime_type=mime_type,
            description=payload.description,
            is_global=bool(payload.isGlobal),
            file_hash=file_hash,
        )
        doc = await repo.get_document(doc_id)
        # Log upload activity
        try:
            await repo.insert_activity(
                action="upload",
                document_id=doc_id,
                file_name=(doc.get("title") or doc.get("original_name") or doc.get("filename") or ""),
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
async def process_file(file_id: str) -> Dict[str, Any]:
    doc = await repo.get_document(file_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    # Re-process document (idempotent)
    await svc.process_uploaded_file(
        storage_path=doc["storage_path"],
        original_name=doc["original_name"],
        mime_type=doc["mime_type"],
        description=doc.get("description"),
        is_global=doc.get("is_global", False),
    )
    try:
        await repo.insert_activity(
            action="process",
            document_id=file_id,
            file_name=(doc.get("title") or doc.get("original_name") or doc.get("filename") or ""),
            user_id=None,
            details="re-embedded document",
        )
    except Exception:
        pass
    return {"fileId": file_id, "status": "ready"}


class SemanticSearchRequest(BaseModel):
    query: str
    limit: int = 10


@router.post("/semantic-search")
async def semantic_search(payload: SemanticSearchRequest) -> List[Dict[str, Any]]:
    ctx = await svc.retrieve_context(query=payload.query, mode="all", max_docs=payload.limit)
    doc_ids = set(ctx.get("doc_ids", []))
    out: List[Dict[str, Any]] = []
    for doc_id in doc_ids:
        d = await repo.get_document(doc_id)
        if not d:
            continue
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
                "similarity": 1.0,  # Placeholder; detailed per-doc score optional
            }
        )
    return out


@router.get("/activity")
async def recent_activity(limit: int = 20) -> List[Dict[str, Any]]:
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
        # Return empty list if anything goes wrong to keep UI functional
        return []


@router.post("/files/{file_id}/reextract-title")
async def reextract_title(file_id: str) -> Dict[str, Any]:
    doc = await repo.get_document(file_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

    try:
        title = await svc.reextract_title_for_document(storage_path=doc["storage_path"], mime_type=doc["mime_type"])
        if title:
            await repo.update_document_title_and_description(document_id=file_id, title=title)
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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

