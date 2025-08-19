from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, status

from app.dependencies import _get_ollama_base_url


router = APIRouter(prefix="/local-llm", tags=["local-llm"])
logger = logging.getLogger(__name__)


def _ollama_url(path: str) -> str:
    base = _get_ollama_base_url().rstrip("/")
    return f"{base}{path}"


@router.get("/health")
async def health() -> Dict[str, str]:
    """Basic health probe for the local Ollama server."""
    url = _ollama_url("/api/tags")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        return {"status": "healthy"}
    except Exception as exc:  # noqa: BLE001
        logger.error("Ollama health check failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))


@router.get("/models")
async def list_models() -> Dict[str, List[Dict[str, Any]]]:
    """Return the list of locally available models (tags) from Ollama."""
    url = _ollama_url("/api/tags")
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
        # Normalise shape: Ollama returns { models: [ { name, ... }, ... ] }
        return {"models": data.get("models", [])}


@router.post("/pull")
async def pull_model(payload: Dict[str, str]) -> Dict[str, Any]:
    """Trigger a model pull by name (e.g., "llama3.2:latest").

    Returns a simple acknowledgement; callers may poll `/models` to see when it
    becomes available locally. For richer progress streaming, a dedicated SSE
    endpoint can be added later.
    """
    name: Optional[str] = payload.get("name") if payload else None
    if not name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="'name' is required")

    # Fast-path: if already installed, return immediately
    tags_url = _ollama_url("/api/tags")
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            tags_resp = await client.get(tags_url)
            tags_resp.raise_for_status()
            models = (tags_resp.json() or {}).get("models", [])
            if any(m.get("name") == name for m in models):
                return {"status": "ok", "name": name, "already_present": True}
        except Exception:  # noqa: BLE001
            # If tag listing fails, fall through to pull attempt
            pass

    url = _ollama_url("/api/pull")
    async with httpx.AsyncClient(timeout=None) as client:
        # Non-streaming pull; Ollama blocks until complete
        resp = await client.post(url, json={"name": name})
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return {"status": "ok", "name": name, "already_present": False}


