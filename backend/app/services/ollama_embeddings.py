from __future__ import annotations

from typing import List, Tuple, Dict
import time

import httpx

from app.dependencies import _get_ollama_base_url
import os


async def embed_texts_via_ollama(*, model: str, texts: List[str]) -> Tuple[List[List[float]], int]:
    """Embed a batch of texts using Ollama's embeddings API.

    Returns (vectors, original_dimensions).
    """
    if not texts:
        return [], 0
    
    start_time = time.time()
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

        # Record performance metrics
        elapsed_ms = (time.time() - start_time) * 1000
        total_chars = sum(len(text) for text in texts)
        try:
            from app.services.metrics_service import record_embedding_performance
            record_embedding_performance("single", model, len(texts), elapsed_ms, total_chars)
        except Exception:
            pass  # Don't fail on metrics errors

        return collected, detected_dim


async def embed_texts_batch(*, model: str, texts: List[str]) -> Tuple[List[List[float]], int]:
    """Embed texts using Ollama's batch /api/embed endpoint (Ollama 0.12+).

    Sends all texts in a single request. Falls back to sequential
    embed_texts_via_ollama() if the batch endpoint is unavailable.

    Returns (vectors, dimensions).
    """
    if not texts:
        return [], 0

    start_time = time.time()
    base = _get_ollama_base_url().rstrip("/")
    url = f"{base}/api/embed"

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(url, json={"model": model, "input": texts})
            if r.status_code == 404:
                # Batch endpoint not available, fall back
                import logging
                logging.getLogger(__name__).warning(
                    "Ollama /api/embed returned 404, falling back to sequential embedding"
                )
                return await embed_texts_via_ollama(model=model, texts=texts)
            r.raise_for_status()
            data = r.json() or {}
            embeddings = data.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(texts):
                raise ValueError(
                    f"Expected {len(texts)} embeddings, got "
                    f"{len(embeddings) if isinstance(embeddings, list) else 'none'}"
                )
            dims = len(embeddings[0]) if embeddings else 0
            
            # Record performance metrics
            elapsed_ms = (time.time() - start_time) * 1000
            total_chars = sum(len(text) for text in texts)
            try:
                from app.services.metrics_service import record_embedding_performance
                record_embedding_performance("batch", model, len(texts), elapsed_ms, total_chars)
            except Exception:
                pass  # Don't fail on metrics errors
                
            return embeddings, dims
    except httpx.HTTPStatusError:
        raise
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Batch embed failed (%s), falling back to sequential", exc
        )
        # Note: fallback will record its own metrics as "single" operation
        return await embed_texts_via_ollama(model=model, texts=texts)
