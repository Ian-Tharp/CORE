from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, status

from app.dependencies import (
    _get_ollama_base_url,
    get_local_provider,
    get_ollama_client,
)


router = APIRouter(prefix="/local-llm", tags=["local-llm"])
logger = logging.getLogger(__name__)


def _ollama_url(path: str) -> str:
    base = _get_ollama_base_url().rstrip("/")
    return f"{base}{path}"


def _provider_label(provider: str) -> str:
    return "LM Studio" if provider == "lmstudio" else "Ollama"


@router.get("/provider")
async def provider_info() -> Dict[str, Any]:
    """Describe the active local LLM provider so the UI can label it dynamically.

    `supports_pull` is False for LM Studio — its models are loaded in the LM Studio
    app rather than pulled over an API.
    """
    provider = get_local_provider()
    return {
        "provider": provider,
        "label": _provider_label(provider),
        "supports_pull": provider != "lmstudio",
    }


@router.get("/health")
async def health() -> Dict[str, str]:
    """Health probe for the active local provider (Ollama or LM Studio)."""
    provider = get_local_provider()
    try:
        if provider == "lmstudio":
            # LM Studio is OpenAI-compatible; listing models is a cheap liveness probe.
            await get_ollama_client().models.list()
        else:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(_ollama_url("/api/tags"))
                resp.raise_for_status()
        return {"status": "healthy", "provider": provider}
    except Exception as exc:  # noqa: BLE001
        logger.error("%s health check failed: %s", _provider_label(provider), exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )


@router.get("/models")
async def list_models() -> Dict[str, Any]:
    """Return the locally available models from the active provider.

    Shape is normalised to ``{ provider, models: [ { name, ... } ] }`` for both
    Ollama (native ``/api/tags``) and LM Studio (OpenAI-compatible ``/v1/models``).
    """
    provider = get_local_provider()
    if provider == "lmstudio":
        models = await get_ollama_client().models.list()
        # LM Studio's /v1/models lists every loaded model, including embedding
        # models — filter those out so the chat dropdown only offers chat models.
        return {
            "provider": provider,
            "models": [
                {"name": m.id}
                for m in (models.data or [])
                if "embed" not in (m.id or "").lower()
            ],
        }

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(_ollama_url("/api/tags"))
        resp.raise_for_status()
        data = resp.json()
        # Normalise shape: Ollama returns { models: [ { name, ... }, ... ] }
        return {"provider": provider, "models": data.get("models", [])}


@router.post("/pull")
async def pull_model(payload: Dict[str, str]) -> Dict[str, Any]:
    """Trigger a model pull by name (e.g., "llama3.2:latest").

    Returns a simple acknowledgement; callers may poll `/models` to see when it
    becomes available locally. For richer progress streaming, a dedicated SSE
    endpoint can be added later.
    """
    if get_local_provider() == "lmstudio":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model pulling isn't supported for LM Studio — load models in the LM Studio app.",
        )

    name: Optional[str] = payload.get("name") if payload else None
    if not name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="'name' is required"
        )

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
