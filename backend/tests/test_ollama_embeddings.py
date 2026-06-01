"""
Tests for ollama_embeddings — embedding API calls, fallback logic, and error handling.

Sentinel-value contract:
- Remove empty-input early return → empty list POSTs to Ollama → test fails
- Remove 'input' payload variant → newer Ollama builds don't get embeddings → test fails
- Remove 404 fallback in embed_texts_batch → batch endpoint absent → service crashes → test fails
- Remove sequential fallback on batch exception → transient errors break all embedding → test fails
- Remove dimension detection from first vector → dims always 0 → caller misconfigures pgvector → test fails
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from app.services.ollama_embeddings import embed_texts_via_ollama, embed_texts_batch


# ===========================================================================
# Helpers
# ===========================================================================


def _mock_response(
    status_code: int = 200, body: Dict[str, Any] | None = None
) -> MagicMock:
    """Build a mock httpx Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body or {}

    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status = MagicMock()

    return resp


def _build_mock_client(responses: List[MagicMock]) -> AsyncMock:
    """Build an httpx.AsyncClient mock that returns responses in order."""
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.post = AsyncMock(side_effect=responses)
    return client


# ===========================================================================
# embed_texts_via_ollama — sequential endpoint
# ===========================================================================


class TestEmbedTextsViaOllama:
    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_immediately(self):
        """
        SENTINEL — empty texts list must return ([], 0) without calling Ollama.
        Remove empty check → POST made with empty body → Ollama returns error → test fails.
        """
        result, dims = await embed_texts_via_ollama(model="test", texts=[])
        assert result == []
        assert dims == 0

    @pytest.mark.asyncio
    async def test_successful_embedding_returned(self):
        """
        SENTINEL — successful embedding must be returned as a list of floats.
        Skip return → caller gets [] instead of vectors → RAG returns no results → test fails.
        """
        sentinel_vector = [0.1, 0.2, 0.3, 0.4]
        response = _mock_response(200, {"embedding": sentinel_vector})
        client = _build_mock_client([response])

        with patch(
            "app.services.ollama_embeddings.httpx.AsyncClient", return_value=client
        ), patch(
            "app.services.ollama_embeddings._get_ollama_base_url",
            return_value="http://localhost:11434",
        ):
            result, dims = await embed_texts_via_ollama(model="m", texts=["hello"])

        assert result == [sentinel_vector]
        assert dims == 4  # len of vector

    @pytest.mark.asyncio
    async def test_dimensions_set_from_first_vector(self):
        """
        SENTINEL — dims must be the length of the first returned vector.
        Hardcode dims=0 → pgvector misconfigured → embeddings silently dropped → test fails.
        """
        vec = [1.0] * 768
        response = _mock_response(200, {"embedding": vec})
        client = _build_mock_client([response])

        with patch(
            "app.services.ollama_embeddings.httpx.AsyncClient", return_value=client
        ), patch(
            "app.services.ollama_embeddings._get_ollama_base_url",
            return_value="http://localhost:11434",
        ):
            _, dims = await embed_texts_via_ollama(model="m", texts=["text"])

        assert dims == 768

    @pytest.mark.asyncio
    async def test_embeddings_key_also_accepted(self):
        """'embeddings' (plural) key in response must also be accepted."""
        vec = [0.5, 0.6, 0.7]
        response = _mock_response(200, {"embeddings": [vec]})
        client = _build_mock_client([response])

        with patch(
            "app.services.ollama_embeddings.httpx.AsyncClient", return_value=client
        ), patch(
            "app.services.ollama_embeddings._get_ollama_base_url",
            return_value="http://localhost:11434",
        ):
            result, dims = await embed_texts_via_ollama(model="m", texts=["text"])

        assert result == [vec]

    @pytest.mark.asyncio
    async def test_missing_embedding_raises_http_error(self):
        """
        SENTINEL — response with no embedding key must raise httpx.HTTPError.
        Return empty vector → caller inserts zero-vector → corrupt knowledge base → test fails.
        """
        # First payload 200 with no usable embedding
        resp_input = _mock_response(200, {"no_embedding_here": True})
        # Second payload (prompt fallback) also empty
        resp_prompt = _mock_response(200, {"no_embedding_here": True})
        client = _build_mock_client([resp_input, resp_prompt])

        with patch(
            "app.services.ollama_embeddings.httpx.AsyncClient", return_value=client
        ), patch(
            "app.services.ollama_embeddings._get_ollama_base_url",
            return_value="http://localhost:11434",
        ):
            with pytest.raises(httpx.HTTPError, match="missing 'embedding'"):
                await embed_texts_via_ollama(model="m", texts=["text"])

    @pytest.mark.asyncio
    async def test_multiple_texts_produce_multiple_vectors(self):
        """Two texts must produce two vectors."""
        vec1, vec2 = [0.1, 0.2], [0.3, 0.4]
        r1 = _mock_response(200, {"embedding": vec1})
        r2 = _mock_response(200, {"embedding": vec2})
        client = _build_mock_client([r1, r2])

        with patch(
            "app.services.ollama_embeddings.httpx.AsyncClient", return_value=client
        ), patch(
            "app.services.ollama_embeddings._get_ollama_base_url",
            return_value="http://localhost:11434",
        ):
            result, _ = await embed_texts_via_ollama(model="m", texts=["a", "b"])

        assert len(result) == 2
        assert result[0] == vec1
        assert result[1] == vec2

    @pytest.mark.asyncio
    async def test_404_on_first_payload_tries_second(self):
        """
        SENTINEL — 404 response on 'input' payload must fall back to 'prompt' payload.
        Remove fallback payload → newer endpoint names break all embedding → test fails.
        """
        vec = [1.0, 2.0, 3.0]
        resp_404 = _mock_response(404)
        resp_ok = _mock_response(200, {"embedding": vec})
        client = _build_mock_client([resp_404, resp_ok])

        with patch(
            "app.services.ollama_embeddings.httpx.AsyncClient", return_value=client
        ), patch(
            "app.services.ollama_embeddings._get_ollama_base_url",
            return_value="http://localhost:11434",
        ):
            result, _ = await embed_texts_via_ollama(model="m", texts=["text"])

        assert result == [vec]


# ===========================================================================
# embed_texts_batch — batch endpoint with fallback
# ===========================================================================


class TestEmbedTextsBatch:
    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_immediately(self):
        """Empty texts must return ([], 0) without calling Ollama batch."""
        result, dims = await embed_texts_batch(model="test", texts=[])
        assert result == []
        assert dims == 0

    @pytest.mark.asyncio
    async def test_batch_success_returns_embeddings(self):
        """
        SENTINEL — successful batch response must return all embeddings.
        Return only first embedding → multi-text batches silently truncated → test fails.
        """
        vecs = [[0.1, 0.2], [0.3, 0.4]]
        resp = _mock_response(200, {"embeddings": vecs})
        client = _build_mock_client([resp])

        with patch(
            "app.services.ollama_embeddings.httpx.AsyncClient", return_value=client
        ), patch(
            "app.services.ollama_embeddings._get_ollama_base_url",
            return_value="http://localhost:11434",
        ):
            result, dims = await embed_texts_batch(model="m", texts=["a", "b"])

        assert result == vecs
        assert dims == 2  # len of first vector

    @pytest.mark.asyncio
    async def test_404_falls_back_to_sequential(self):
        """
        SENTINEL — 404 on batch endpoint must fall back to embed_texts_via_ollama.
        Remove 404 check → service crashes on older Ollama builds → test fails.
        """
        fallback_vecs = [[0.5, 0.6]]
        fallback_dims = 2

        resp_404 = _mock_response(404)
        client = _build_mock_client([resp_404])

        with patch(
            "app.services.ollama_embeddings.httpx.AsyncClient", return_value=client
        ), patch(
            "app.services.ollama_embeddings._get_ollama_base_url",
            return_value="http://localhost:11434",
        ), patch(
            "app.services.ollama_embeddings.embed_texts_via_ollama",
            new=AsyncMock(return_value=(fallback_vecs, fallback_dims)),
        ) as fallback_spy:
            result, dims = await embed_texts_batch(model="m", texts=["text"])

        fallback_spy.assert_called_once()
        assert result == fallback_vecs
        assert dims == fallback_dims

    @pytest.mark.asyncio
    async def test_http_status_error_propagates(self):
        """HTTP 4xx/5xx (non-404) must be raised, not swallowed."""
        resp_500 = _mock_response(500)
        client = _build_mock_client([resp_500])

        with patch(
            "app.services.ollama_embeddings.httpx.AsyncClient", return_value=client
        ), patch(
            "app.services.ollama_embeddings._get_ollama_base_url",
            return_value="http://localhost:11434",
        ):
            with pytest.raises(httpx.HTTPStatusError):
                await embed_texts_batch(model="m", texts=["text"])

    @pytest.mark.asyncio
    async def test_general_exception_falls_back_to_sequential(self):
        """
        SENTINEL — non-HTTP exceptions in batch must fall back to sequential.
        Remove general fallback → transient errors (timeouts) break all embedding → test fails.
        """
        fallback_vecs = [[0.9, 0.8]]

        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.post = AsyncMock(side_effect=ConnectionError("timeout"))

        with patch(
            "app.services.ollama_embeddings.httpx.AsyncClient", return_value=client
        ), patch(
            "app.services.ollama_embeddings._get_ollama_base_url",
            return_value="http://localhost:11434",
        ), patch(
            "app.services.ollama_embeddings.embed_texts_via_ollama",
            new=AsyncMock(return_value=(fallback_vecs, 2)),
        ) as fallback_spy:
            result, _ = await embed_texts_batch(model="m", texts=["text"])

        fallback_spy.assert_called_once()
        assert result == fallback_vecs

    @pytest.mark.asyncio
    async def test_count_mismatch_falls_back_to_sequential(self):
        """
        SENTINEL — if batch returns wrong count, ValueError triggers sequential fallback.
        The general except clause in embed_texts_batch catches ValueError and calls
        embed_texts_via_ollama; verify that sequential fallback is invoked.
        """
        # 2 texts but only 1 embedding in response → raises ValueError internally
        resp = _mock_response(200, {"embeddings": [[0.1, 0.2]]})
        client = _build_mock_client([resp])

        fallback_vecs = [[0.1, 0.2], [0.3, 0.4]]

        with patch(
            "app.services.ollama_embeddings.httpx.AsyncClient", return_value=client
        ), patch(
            "app.services.ollama_embeddings._get_ollama_base_url",
            return_value="http://localhost:11434",
        ), patch(
            "app.services.ollama_embeddings.embed_texts_via_ollama",
            new=AsyncMock(return_value=(fallback_vecs, 2)),
        ) as fallback_spy:
            result, dims = await embed_texts_batch(model="m", texts=["a", "b"])

        # Fallback must be invoked when batch returns wrong count
        fallback_spy.assert_called_once()
        assert result == fallback_vecs
