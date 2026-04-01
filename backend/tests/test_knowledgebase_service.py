"""
Tests for the knowledgebase service — RAG retrieval, chunking, and message building.

Sentinel-value contract:
- Remove chunk overlap logic → overlap=0 always → test fails
- Remove cosine similarity normalisation → wrong scores → test fails
- Remove embedding call in retrieve_context → no query vector → test fails
- Remove repo.search_chunks_by_vector call → no results ever → test fails
- Remove context_block injection in build_rag_messages → RAG context dropped → test fails
- Remove system_msg prepend → context never reaches LLM → test fails
"""

from __future__ import annotations

import math
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.knowledgebase_service import (
    _split_text,
    _cosine_similarity,
    build_rag_messages,
    retrieve_context,
    retrieve_context_by_instance,
)


# ===========================================================================
# _split_text — pure chunking logic
# ===========================================================================

class TestSplitText:
    def test_short_text_returns_single_chunk(self):
        """Text shorter than chunk_size must not be split."""
        result = _split_text("hello world", chunk_size=1200, chunk_overlap=200)
        assert len(result) == 1
        assert result[0] == "hello world"

    def test_long_text_is_split(self):
        """
        SENTINEL — text longer than chunk_size must produce multiple chunks.
        Remove the while loop → always returns single chunk → test fails.
        """
        text = "a" * 3000
        result = _split_text(text, chunk_size=1000, chunk_overlap=0)
        assert len(result) == 3

    def test_overlap_is_respected(self):
        """
        SENTINEL — consecutive chunks must share `chunk_overlap` characters.
        Set overlap to 0 in implementation → chunks don't overlap → test fails.
        """
        text = "abcdefghij"  # 10 chars
        result = _split_text(text, chunk_size=6, chunk_overlap=2)
        # chunk 0: [0:6] = "abcdef"
        # chunk 1 starts at 6-2=4: [4:10] = "efghij"
        assert len(result) == 2
        assert result[0] == "abcdef"
        assert result[1] == "efghij"
        # The last 2 chars of chunk0 == first 2 chars of chunk1 (overlap)
        assert result[0][-2:] == result[1][:2]

    def test_exact_chunk_size_produces_single_chunk(self):
        """Text exactly chunk_size chars → single chunk, no off-by-one split."""
        text = "x" * 100
        result = _split_text(text, chunk_size=100, chunk_overlap=10)
        assert len(result) == 1

    def test_empty_text_returns_empty_list(self):
        """Empty string must return empty list (no content = no chunks, no crash)."""
        result = _split_text("", chunk_size=1200, chunk_overlap=200)
        assert result == []

    def test_chunk_size_zero_returns_full_text(self):
        """chunk_size=0 is a guard-case — must return original text."""
        result = _split_text("hello", chunk_size=0, chunk_overlap=0)
        assert result == ["hello"]

    def test_all_chunks_are_non_empty(self):
        """No chunk in the result should be empty."""
        text = "word " * 500
        result = _split_text(text, chunk_size=300, chunk_overlap=50)
        for chunk in result:
            assert chunk, "Found empty chunk in result"

    def test_total_content_covered(self):
        """
        SENTINEL — the first and last characters of the original text must appear
        in the first and last chunks respectively.
        Skip last chunk → end of text lost → test fails.
        """
        text = "START" + ("x" * 2000) + "END"
        result = _split_text(text, chunk_size=500, chunk_overlap=50)
        assert result[0].startswith("START")
        assert result[-1].endswith("END")


# ===========================================================================
# _cosine_similarity — pure math
# ===========================================================================

class TestCosineSimilarity:
    def test_identical_vectors_return_1(self):
        """
        SENTINEL — identical vectors must return similarity of 1.0.
        Remove normalisation → dot product != 1 for non-unit vectors → test fails.
        """
        v = [1.0, 2.0, 3.0]
        result = _cosine_similarity(v, v)
        assert abs(result - 1.0) < 1e-9

    def test_orthogonal_vectors_return_0(self):
        """Orthogonal vectors must return similarity of 0.0."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(_cosine_similarity(a, b)) < 1e-9

    def test_opposite_vectors_return_minus_1(self):
        """Anti-parallel vectors must return -1.0."""
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-9

    def test_zero_vector_returns_0(self):
        """Zero vector similarity must be 0, not a division error."""
        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
        assert _cosine_similarity([1.0, 1.0], [0.0, 0.0]) == 0.0

    def test_empty_vectors_return_0(self):
        """Empty vectors must return 0 without raising."""
        assert _cosine_similarity([], []) == 0.0

    def test_known_value(self):
        """Hand-computed value: [1,0] vs [1,1]/√2 → cos(45°) = 0.707."""
        a = [1.0, 0.0]
        b = [1.0, 1.0]
        result = _cosine_similarity(a, b)
        expected = 1.0 / math.sqrt(2)
        assert abs(result - expected) < 1e-9

    def test_mismatched_lengths_truncates(self):
        """Different-length vectors must not raise — truncate to shorter."""
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0]
        # Only first 2 elements used → identical in those dims → 1.0
        result = _cosine_similarity(a, b)
        assert abs(result - 1.0) < 1e-9

    def test_positive_similarity_for_similar_vectors(self):
        """Vectors pointing in the same general direction must score > 0."""
        a = [1.0, 1.0, 1.0]
        b = [2.0, 2.0, 2.0]
        assert _cosine_similarity(a, b) > 0.99  # should be exactly 1.0


# ===========================================================================
# build_rag_messages — RAG message construction
# ===========================================================================

class TestBuildRagMessages:
    def _make_chunk(self, doc_id: str, chunk_idx: int, text: str) -> Dict[str, Any]:
        return {
            "document_id": doc_id,
            "chunk_index": chunk_idx,
            "text": text,
        }

    def test_no_chunks_returns_original_messages(self):
        """Empty context chunks must return the original messages unchanged."""
        msgs = [{"role": "user", "content": "hi"}]
        result = build_rag_messages(msgs, context_chunks=[])
        assert result is msgs

    def test_context_injected_as_system_message(self):
        """
        SENTINEL — chunks must be injected as a system message prepended to messages.
        Remove system_msg prepend → context never reaches LLM → test fails.
        """
        chunks = [self._make_chunk("doc-1", 0, "SENTINEL_CONTENT")]
        msgs = [{"role": "user", "content": "What does the doc say?"}]
        result = build_rag_messages(msgs, context_chunks=chunks)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "SENTINEL_CONTENT" in result[0]["content"]

    def test_original_messages_preserved_after_system(self):
        """
        SENTINEL — original user messages must follow the injected system message.
        Overwrite original messages → user question lost → test fails.
        """
        chunks = [self._make_chunk("doc-1", 0, "some context")]
        user_msg = {"role": "user", "content": "SENTINEL_USER_MSG"}
        result = build_rag_messages([user_msg], context_chunks=chunks)
        assert result[-1]["content"] == "SENTINEL_USER_MSG"

    def test_chunk_reference_in_system_message(self):
        """
        SENTINEL — system message must contain doc:X#chunk:Y reference for traceability.
        Remove ref string → attribution lost → test fails.
        """
        chunks = [self._make_chunk("doc-SENTINEL", 7, "some text")]
        msgs = [{"role": "user", "content": "Q"}]
        result = build_rag_messages(msgs, context_chunks=chunks)
        system_content = result[0]["content"]
        assert "doc:doc-SENTINEL" in system_content
        assert "chunk:7" in system_content

    def test_multiple_chunks_all_included(self):
        """
        SENTINEL — all provided chunks must appear in the system message.
        Only inject first chunk → rest of context lost → test fails.
        """
        chunks = [
            self._make_chunk("doc-1", 0, "FACT_ONE"),
            self._make_chunk("doc-2", 0, "FACT_TWO"),
            self._make_chunk("doc-3", 0, "FACT_THREE"),
        ]
        msgs = [{"role": "user", "content": "Q"}]
        result = build_rag_messages(msgs, context_chunks=chunks)
        system_content = result[0]["content"]
        assert "FACT_ONE" in system_content
        assert "FACT_TWO" in system_content
        assert "FACT_THREE" in system_content

    def test_existing_system_message_is_kept(self):
        """If original messages already have a system message, it must be preserved."""
        chunks = [self._make_chunk("doc-1", 0, "context")]
        orig_system = {"role": "system", "content": "Be helpful."}
        user_msg = {"role": "user", "content": "Q"}
        result = build_rag_messages([orig_system, user_msg], context_chunks=chunks)
        # New system message prepended, original system msg still present
        assert len(result) == 3
        assert result[0]["role"] == "system"  # RAG system msg
        assert result[1] == orig_system
        assert result[2] == user_msg


# ===========================================================================
# retrieve_context — RAG retrieval (mocked)
# ===========================================================================

class TestRetrieveContext:
    @pytest.mark.asyncio
    async def test_embedding_is_called_for_query(self):
        """
        SENTINEL — retrieve_context must embed the query string.
        Remove _embed_texts_local call → query never vectorised → no results → test fails.
        """
        sentinel_vec = [0.1, 0.2, 0.3]

        with patch("app.services.knowledgebase_service._embed_texts_local",
                   new=AsyncMock(return_value=([sentinel_vec], 3))) as embed_spy, \
             patch("app.services.knowledgebase_service.repo") as mock_repo:
            mock_repo.search_chunks_by_vector = AsyncMock(return_value=[])

            await retrieve_context(query="SENTINEL_QUERY", mode="all")

        embed_spy.assert_called_once()
        call_kwargs = embed_spy.call_args[1]
        assert "SENTINEL_QUERY" in call_kwargs.get("texts", [])

    @pytest.mark.asyncio
    async def test_repo_searched_with_query_vector(self):
        """
        SENTINEL — the embedded query vector must be passed to the repo search.
        Use wrong vector → wrong results returned → test fails.
        """
        sentinel_vec = [9.9, 8.8, 7.7]

        with patch("app.services.knowledgebase_service._embed_texts_local",
                   new=AsyncMock(return_value=([sentinel_vec], 3))), \
             patch("app.services.knowledgebase_service.repo") as mock_repo:
            mock_repo.search_chunks_by_vector = AsyncMock(return_value=[])

            await retrieve_context(query="q", mode="all")

        mock_repo.search_chunks_by_vector.assert_called_once()
        call_kwargs = mock_repo.search_chunks_by_vector.call_args[1]
        assert call_kwargs["query_vec"] == sentinel_vec

    @pytest.mark.asyncio
    async def test_results_returned_from_repo(self):
        """
        SENTINEL — chunks returned by repo must appear in the result.
        Return empty list always → no context ever surfaces → test fails.
        """
        sentinel_chunk = {
            "document_id": "doc-SENTINEL",
            "chunk_index": 0,
            "text": "SENTINEL_CHUNK_TEXT",
        }

        with patch("app.services.knowledgebase_service._embed_texts_local",
                   new=AsyncMock(return_value=([[0.1]], 1))), \
             patch("app.services.knowledgebase_service.repo") as mock_repo:
            mock_repo.search_chunks_by_vector = AsyncMock(return_value=[sentinel_chunk])

            result = await retrieve_context(query="q", mode="all")

        assert len(result["chunks"]) == 1
        assert result["chunks"][0]["text"] == "SENTINEL_CHUNK_TEXT"
        assert "doc-SENTINEL" in result["doc_ids"]

    @pytest.mark.asyncio
    async def test_empty_embedding_returns_empty(self):
        """If embedding fails (returns []), retrieve_context must return empty gracefully."""
        with patch("app.services.knowledgebase_service._embed_texts_local",
                   new=AsyncMock(return_value=([], 0))):
            result = await retrieve_context(query="q", mode="all")

        assert result == {"chunks": [], "doc_ids": []}

    @pytest.mark.asyncio
    async def test_file_mode_passes_file_id_filter(self):
        """
        SENTINEL — mode='file' must pass file_id as document_filter to repo.
        Ignore mode → global search always → tenant isolation broken → test fails.
        """
        with patch("app.services.knowledgebase_service._embed_texts_local",
                   new=AsyncMock(return_value=([[0.1]], 1))), \
             patch("app.services.knowledgebase_service.repo") as mock_repo:
            mock_repo.search_chunks_by_vector = AsyncMock(return_value=[])

            await retrieve_context(query="q", mode="file", file_id="SENTINEL_FILE_ID")

        call_kwargs = mock_repo.search_chunks_by_vector.call_args[1]
        assert call_kwargs.get("document_filter") == ["SENTINEL_FILE_ID"]

    @pytest.mark.asyncio
    async def test_all_mode_passes_no_filter(self):
        """mode='all' must not restrict by document — no document_filter."""
        with patch("app.services.knowledgebase_service._embed_texts_local",
                   new=AsyncMock(return_value=([[0.1]], 1))), \
             patch("app.services.knowledgebase_service.repo") as mock_repo:
            mock_repo.search_chunks_by_vector = AsyncMock(return_value=[])

            await retrieve_context(query="q", mode="all")

        call_kwargs = mock_repo.search_chunks_by_vector.call_args[1]
        assert call_kwargs.get("document_filter") is None


# ===========================================================================
# retrieve_context_by_instance — instance-scoped retrieval
# ===========================================================================

class TestRetrieveContextByInstance:
    @pytest.mark.asyncio
    async def test_instance_name_passed_to_repo(self):
        """
        SENTINEL — retrieve_context_by_instance must pass instance_name to repo.
        Remove instance_name param → all instances can read each other's context → test fails.
        """
        with patch("app.services.knowledgebase_service._embed_texts_local",
                   new=AsyncMock(return_value=([[0.1]], 1))), \
             patch("app.services.knowledgebase_service.repo") as mock_repo:
            mock_repo.search_chunks_by_instance = AsyncMock(return_value=[])

            await retrieve_context_by_instance(
                query="q",
                instance_name="SENTINEL_INSTANCE",
            )

        mock_repo.search_chunks_by_instance.assert_called_once()
        call_kwargs = mock_repo.search_chunks_by_instance.call_args[1]
        assert call_kwargs["instance_name"] == "SENTINEL_INSTANCE"

    @pytest.mark.asyncio
    async def test_result_contains_instance_name(self):
        """instance_name must be echoed back in the result dict."""
        with patch("app.services.knowledgebase_service._embed_texts_local",
                   new=AsyncMock(return_value=([[0.1]], 1))), \
             patch("app.services.knowledgebase_service.repo") as mock_repo:
            mock_repo.search_chunks_by_instance = AsyncMock(return_value=[])

            result = await retrieve_context_by_instance(
                query="q",
                instance_name="SENTINEL_INSTANCE",
            )

        assert result["instance_name"] == "SENTINEL_INSTANCE"

    @pytest.mark.asyncio
    async def test_empty_embedding_returns_empty(self):
        """If embedding fails, result must be empty with instance_name preserved."""
        with patch("app.services.knowledgebase_service._embed_texts_local",
                   new=AsyncMock(return_value=([], 0))):
            result = await retrieve_context_by_instance(
                query="q",
                instance_name="inst-1",
            )
        assert result["chunks"] == []
        assert result["instance_name"] == "inst-1"
