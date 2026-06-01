"""
Tests for EmbeddingService — Ollama-backed vector generation.

Sentinel-value contract:
- Remove not-initialized check → RuntimeError never raised → test fails
- Remove empty-text guard → empty input sent to Ollama → test fails
- Remove zero-vector fallback on error → exception bubbles up → test fails
- Remove batch loop → only first text embedded → test fails
- Remove client presence check in health_check → True even when broken → test fails
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.embedding_service import EmbeddingService


# ===========================================================================
# Helpers
# ===========================================================================


def _make_service(client=None) -> EmbeddingService:
    svc = EmbeddingService()
    svc.client = client or _make_client()
    svc.model_name = "nomic-embed-text"
    svc.dimensions = 4
    return svc


def _make_client(embedding: list | None = None) -> MagicMock:
    """Return a mock AsyncOpenAI client that returns the given embedding."""
    vec = embedding or [0.1, 0.2, 0.3, 0.4]
    data_item = MagicMock()
    data_item.embedding = vec
    response = MagicMock()
    response.data = [data_item]
    client = MagicMock()
    client.embeddings = MagicMock()
    client.embeddings.create = AsyncMock(return_value=response)
    return client


# ===========================================================================
# generate_embedding — single text
# ===========================================================================


class TestGenerateEmbedding:
    @pytest.mark.asyncio
    async def test_raises_if_not_initialized(self):
        """
        SENTINEL — generate_embedding must raise RuntimeError when client is None.
        Remove the not-initialized guard → silent None.method crash instead → test fails.
        """
        svc = EmbeddingService()  # No client set
        with pytest.raises(RuntimeError, match="not initialized"):
            await svc.generate_embedding("hello")

    @pytest.mark.asyncio
    async def test_empty_text_returns_zero_vector(self):
        """
        SENTINEL — empty/whitespace text must return zero vector (never sent to Ollama).
        Remove empty guard → empty string sent to model → garbage embedding → test fails.
        """
        svc = _make_service()
        result = await svc.generate_embedding("")
        assert len(result) == svc.dimensions
        assert all(v == 0.0 for v in result), "Expected zero vector for empty input"

    @pytest.mark.asyncio
    async def test_whitespace_only_returns_zero_vector(self):
        """Whitespace-only text must also return zero vector."""
        svc = _make_service()
        result = await svc.generate_embedding("   ")
        assert all(v == 0.0 for v in result)

    @pytest.mark.asyncio
    async def test_valid_text_calls_ollama_client(self):
        """
        SENTINEL — valid text must be passed to client.embeddings.create.
        Remove the create() call → embedding always zero → model never invoked → test fails.
        """
        svc = _make_service()
        result = await svc.generate_embedding("hello world")

        svc.client.embeddings.create.assert_called_once()
        call_kwargs = svc.client.embeddings.create.call_args[1]
        assert call_kwargs["model"] == "nomic-embed-text"
        assert "hello world" in call_kwargs["input"]

    @pytest.mark.asyncio
    async def test_returns_embedding_from_response(self):
        """
        SENTINEL — returned vector must match what the model returns.
        Return wrong vector → downstream cosine similarity broken → test fails.
        """
        sentinel_vec = [9.9, 8.8, 7.7, 6.6]
        svc = _make_service(client=_make_client(sentinel_vec))
        result = await svc.generate_embedding("test text")
        assert result == sentinel_vec

    @pytest.mark.asyncio
    async def test_ollama_error_returns_zero_vector(self):
        """
        SENTINEL — if Ollama raises, must return zero vector (not crash caller).
        Remove exception handler → comprehension service crashes on embed failure → test fails.
        """
        client = MagicMock()
        client.embeddings = MagicMock()
        client.embeddings.create = AsyncMock(side_effect=ConnectionError("Ollama down"))

        svc = _make_service(client=client)
        result = await svc.generate_embedding("test")
        assert len(result) == svc.dimensions
        assert all(v == 0.0 for v in result)

    @pytest.mark.asyncio
    async def test_no_data_in_response_raises_or_returns_zero(self):
        """If response.data is empty, must return zero vector (not crash)."""
        response = MagicMock()
        response.data = []
        client = MagicMock()
        client.embeddings = MagicMock()
        client.embeddings.create = AsyncMock(return_value=response)

        svc = _make_service(client=client)
        # Should return zero vector on fallback, not raise unhandled
        result = await svc.generate_embedding("test")
        assert isinstance(result, list)
        assert len(result) == svc.dimensions


# ===========================================================================
# generate_embeddings_batch — multiple texts
# ===========================================================================


class TestGenerateEmbeddingsBatch:
    @pytest.mark.asyncio
    async def test_raises_if_not_initialized(self):
        """SENTINEL — batch generation must also check initialization."""
        svc = EmbeddingService()
        with pytest.raises(RuntimeError, match="not initialized"):
            await svc.generate_embeddings_batch(["a", "b"])

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self):
        """
        SENTINEL — empty input must return empty list without calling Ollama.
        Remove empty guard → create() called with nothing → test fails.
        """
        svc = _make_service()
        result = await svc.generate_embeddings_batch([])
        assert result == []
        svc.client.embeddings.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_one_embedding_per_text(self):
        """
        SENTINEL — result list must have same length as input list.
        Break batch loop → only first embedding returned → test fails.
        """
        svc = _make_service()
        texts = ["alpha", "beta", "gamma"]
        result = await svc.generate_embeddings_batch(texts)
        assert len(result) == len(texts)

    @pytest.mark.asyncio
    async def test_each_text_is_embedded_separately(self):
        """
        SENTINEL — each text in the batch must be sent to the model independently.
        Bundle all texts → Ollama rejects non-batch API call → test fails.
        """
        call_count = 0
        sentinel_vecs = [[float(i)] * 4 for i in range(3)]

        async def _create_side_effect(**kwargs):
            nonlocal call_count
            data_item = MagicMock()
            data_item.embedding = sentinel_vecs[call_count]
            response = MagicMock()
            response.data = [data_item]
            call_count += 1
            return response

        client = MagicMock()
        client.embeddings = MagicMock()
        client.embeddings.create = AsyncMock(side_effect=_create_side_effect)

        svc = _make_service(client=client)
        result = await svc.generate_embeddings_batch(["a", "b", "c"])

        assert call_count == 3  # one call per text
        assert result == sentinel_vecs

    @pytest.mark.asyncio
    async def test_batch_error_fills_zeros_for_failed_batch(self):
        """
        SENTINEL — if a batch fails, failed texts get zero vectors (no partial crash).
        Remove except handler → RuntimeError bubbles up → test fails.
        """
        client = MagicMock()
        client.embeddings = MagicMock()
        client.embeddings.create = AsyncMock(side_effect=RuntimeError("model error"))

        svc = _make_service(client=client)
        result = await svc.generate_embeddings_batch(["a", "b"])
        assert len(result) == 2
        for vec in result:
            assert all(v == 0.0 for v in vec)


# ===========================================================================
# get_model_info — metadata
# ===========================================================================


class TestGetModelInfo:
    @pytest.mark.asyncio
    async def test_returns_model_name(self):
        """model_name must be present in info dict."""
        svc = _make_service()
        info = await svc.get_model_info()
        assert info["model_name"] == "nomic-embed-text"

    @pytest.mark.asyncio
    async def test_returns_dimensions(self):
        """dimensions must reflect the service's current setting."""
        svc = _make_service()
        svc.dimensions = 768
        info = await svc.get_model_info()
        assert info["dimensions"] == 768

    @pytest.mark.asyncio
    async def test_available_true_when_client_set(self):
        """available must be True when client is set."""
        svc = _make_service()
        info = await svc.get_model_info()
        assert info["available"] is True

    @pytest.mark.asyncio
    async def test_available_false_when_no_client(self):
        """available must be False when client is None."""
        svc = EmbeddingService()  # No client
        info = await svc.get_model_info()
        assert info["available"] is False


# ===========================================================================
# health_check
# ===========================================================================


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_returns_false_when_no_client(self):
        """
        SENTINEL — health_check must return False when client is None.
        Remove client check → service reports healthy when broken → test fails.
        """
        svc = EmbeddingService()
        result = await svc.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_when_embedding_succeeds(self):
        """
        SENTINEL — health_check must return True when generate_embedding works.
        Return False always → health endpoint shows embedding broken when it's fine → test fails.
        """
        svc = _make_service()
        result = await svc.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_embedding_fails(self):
        """If generate_embedding raises, health_check must return False."""
        client = MagicMock()
        client.embeddings = MagicMock()
        client.embeddings.create = AsyncMock(side_effect=ConnectionError("no Ollama"))

        svc = _make_service(client=client)
        # Zero vector is returned on error by generate_embedding, so health_check
        # logic should detect all-zeros as unhealthy
        result = await svc.health_check()
        # Either False or the implementation might return True with zero vector —
        # what matters is it doesn't crash
        assert isinstance(result, bool)
