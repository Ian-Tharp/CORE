from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import mimetypes
import uuid

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

    with open(storage_path, "wb") as out:
        out.write(await file.read())

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
        )
        doc = await repo.get_document(doc_id)
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
        )
        doc = await repo.get_document(doc_id)

    return {
        "id": doc["id"],
        "filename": doc["filename"],
        "originalName": doc["original_name"],
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


