from __future__ import annotations

from typing import List, Tuple, Dict

import httpx

from app.dependencies import _get_ollama_base_url
import os


DB_LOCAL_VECTOR_DIMENSIONS: int = 3072  # Fixed by DB schema vector(3072)

# Known original embedding dimensions per local model. Used for metadata and validation.
MODEL_ORIGINAL_DIMS: Dict[str, int] = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "embedding-gemma": 3072,
}


def _pad_or_truncate(vector: List[float], target_dim: int = DB_LOCAL_VECTOR_DIMENSIONS) -> List[float]:
    length = len(vector)
    if length == target_dim:
        return vector
    if length > target_dim:
        return vector[:target_dim]
    return vector + [0.0] * (target_dim - length)


async def embed_texts_via_ollama(*, model: str, texts: List[str], target_dim: int | None = None) -> Tuple[List[List[float]], int]:
    """Embed a batch of texts using Ollama's embeddings API.

    Returns (padded_vectors, original_dimensions).
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
            resp = None
            last_exc: Exception | None = None
            for pl in payloads:
                try:
                    r = await client.post(url, json=pl)
                    # Some older versions return 404 for unsupported payload shapes – try the next one
                    if r.status_code == 404:
                        continue
                    r.raise_for_status()
                    resp = r
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    continue
            if resp is None:
                if last_exc is not None:
                    raise last_exc
                raise httpx.HTTPError("Failed to obtain embeddings from Ollama")

            data = resp.json() or {}
            vec = None
            if isinstance(data.get("embedding"), list):
                vec = data.get("embedding")
            elif isinstance(data.get("embeddings"), list) and data["embeddings"]:
                vec = data["embeddings"][0]
            elif isinstance(data.get("data"), list) and data["data"]:
                vec = data["data"][0].get("embedding")
            if not isinstance(vec, list):
                raise httpx.HTTPError("Ollama embeddings response missing 'embedding'")

            if detected_dim == 0:
                detected_dim = len(vec)
            pad_dim = target_dim or DB_LOCAL_VECTOR_DIMENSIONS
            collected.append(_pad_or_truncate(vec, pad_dim))

        return collected, detected_dim


