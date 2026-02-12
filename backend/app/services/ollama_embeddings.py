from __future__ import annotations

from typing import List, Tuple, Dict

import httpx

from app.dependencies import _get_ollama_base_url
import os


async def embed_texts_via_ollama(*, model: str, texts: List[str]) -> Tuple[List[List[float]], int]:
    """Embed a batch of texts using Ollama's embeddings API.

    Returns (vectors, original_dimensions).
    """
    if not texts:
        return [], 0
    base = _get_ollama_base_url().rstrip("/")
    url = f"{base}/api/embeddings"
    async with httpx.AsyncClient(timeout=None) as client:
        collected: List[List[float]] = []
        detected_dim: int = 0
        for text in texts:
            # Try payload with "input" (newer docs) then fall back to "prompt" (older builds)
            payloads = [
                {"model": model, "input": text},
                {"model": model, "prompt": text},
            ]
            vec = None
            last_exc: Exception | None = None
            for pl in payloads:
                try:
                    r = await client.post(url, json=pl)
                    if r.status_code == 404:
                        continue
                    r.raise_for_status()
                    data = r.json() or {}
                    # Extract vector from response
                    candidate = None
                    if isinstance(data.get("embedding"), list) and data["embedding"]:
                        candidate = data["embedding"]
                    elif isinstance(data.get("embeddings"), list) and data["embeddings"]:
                        candidate = data["embeddings"][0]
                    elif isinstance(data.get("data"), list) and data["data"]:
                        candidate = data["data"][0].get("embedding")
                    if isinstance(candidate, list) and candidate:
                        vec = candidate
                        break
                    continue
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    continue
            if not vec:
                if last_exc is not None:
                    raise last_exc
                raise httpx.HTTPError("Ollama embeddings response missing 'embedding'")

            if detected_dim == 0:
                detected_dim = len(vec)
            collected.append(vec)

        return collected, detected_dim
