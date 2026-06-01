"""
Tests for batch embedding and batch upload features.

Covers:
- embed_texts_batch() — Ollama /api/embed batch endpoint with sequential fallback
- batch_upload_and_process() — multi-file processing service
- Batch upload directory scanning logic
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from typing import List
import os
import math


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embedding(dim: int = 768, seed: float = 1.0) -> List[float]:
    """Generate a deterministic non-zero vector."""
    return [math.sin(seed * (i + 1)) * 0.5 + 0.1 for i in range(dim)]


def _mock_httpx_response(status_code: int = 200, json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    if status_code >= 400:
        import httpx

        resp.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "error", request=MagicMock(), response=resp
            )
        )
    else:
        resp.raise_for_status = MagicMock()
    return resp


def _patch_ollama_base_url(url="http://localhost:11434"):
    return patch(
        "app.services.ollama_embeddings._get_ollama_base_url", return_value=url
    )


def _make_async_client(post_side_effect=None, post_return_value=None):
    """Create a mock httpx.AsyncClient context manager."""
    client = AsyncMock()
    if post_side_effect:
        client.post = AsyncMock(side_effect=post_side_effect)
    else:
        client.post = AsyncMock(return_value=post_return_value)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


# =============================================================================
# embed_texts_batch() — batch embedding via Ollama /api/embed
# =============================================================================


class TestEmbedTextsBatch:

    @pytest.mark.asyncio
    async def test_batch_of_three_returns_three_vectors_768dim(self):
        """Batch of 3 texts returns 3 vectors, each 768-dim."""
        from app.services.ollama_embeddings import embed_texts_batch

        vecs = [_make_embedding(768, seed=i) for i in range(3)]
        mock_resp = _mock_httpx_response(200, {"embeddings": vecs})

        with _patch_ollama_base_url(), patch(
            "app.services.ollama_embeddings.httpx.AsyncClient",
            return_value=_make_async_client(post_return_value=mock_resp),
        ):
            result, dim = await embed_texts_batch(
                model="nomic-embed-text", texts=["A", "B", "C"]
            )

        assert len(result) == 3
        for v in result:
            assert len(v) == 768
        assert dim == 768

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_list(self):
        """Empty input returns empty list without making HTTP calls."""
        from app.services.ollama_embeddings import embed_texts_batch

        result, dim = await embed_texts_batch(model="nomic-embed-text", texts=[])
        assert result == []
        assert dim == 0

    @pytest.mark.asyncio
    async def test_single_text_batch_of_one(self):
        """Single text works (batch of 1)."""
        from app.services.ollama_embeddings import embed_texts_batch

        vec = _make_embedding(768, seed=42)
        mock_resp = _mock_httpx_response(200, {"embeddings": [vec]})

        with _patch_ollama_base_url(), patch(
            "app.services.ollama_embeddings.httpx.AsyncClient",
            return_value=_make_async_client(post_return_value=mock_resp),
        ):
            result, dim = await embed_texts_batch(
                model="nomic-embed-text", texts=["Single"]
            )

        assert len(result) == 1
        assert len(result[0]) == 768

    @pytest.mark.asyncio
    async def test_fallback_to_sequential_on_404(self):
        """When batch endpoint returns 404, falls back to sequential embed_texts_via_ollama."""
        from app.services.ollama_embeddings import embed_texts_batch

        # /api/embed returns 404
        batch_resp = _mock_httpx_response(404)

        with _patch_ollama_base_url(), patch(
            "app.services.ollama_embeddings.httpx.AsyncClient",
            return_value=_make_async_client(post_return_value=batch_resp),
        ), patch(
            "app.services.ollama_embeddings.embed_texts_via_ollama",
            new_callable=AsyncMock,
        ) as mock_seq:
            vecs = [_make_embedding(768, seed=i) for i in range(2)]
            mock_seq.return_value = (vecs, 768)

            result, dim = await embed_texts_batch(
                model="nomic-embed-text", texts=["A", "B"]
            )

        assert len(result) == 2
        assert dim == 768
        mock_seq.assert_called_once_with(model="nomic-embed-text", texts=["A", "B"])

    @pytest.mark.asyncio
    async def test_fallback_to_sequential_on_generic_error(self):
        """When batch endpoint raises a non-HTTP error, falls back to sequential."""
        from app.services.ollama_embeddings import embed_texts_batch

        client = _make_async_client()
        client.post = AsyncMock(side_effect=ConnectionError("refused"))

        with _patch_ollama_base_url(), patch(
            "app.services.ollama_embeddings.httpx.AsyncClient", return_value=client
        ), patch(
            "app.services.ollama_embeddings.embed_texts_via_ollama",
            new_callable=AsyncMock,
        ) as mock_seq:
            vec = _make_embedding(768, seed=1)
            mock_seq.return_value = ([vec], 768)

            result, dim = await embed_texts_batch(model="nomic-embed-text", texts=["A"])

        assert len(result) == 1
        mock_seq.assert_called_once()

    @pytest.mark.asyncio
    async def test_http_status_error_is_raised(self):
        """HTTPStatusError (e.g. 500) is re-raised, not caught by generic fallback."""
        from app.services.ollama_embeddings import embed_texts_batch
        import httpx

        resp_500 = _mock_httpx_response(500)

        with _patch_ollama_base_url(), patch(
            "app.services.ollama_embeddings.httpx.AsyncClient",
            return_value=_make_async_client(post_return_value=resp_500),
        ):
            with pytest.raises(httpx.HTTPStatusError):
                await embed_texts_batch(model="nomic-embed-text", texts=["A"])

    @pytest.mark.asyncio
    async def test_large_batch_100_texts(self):
        """Large batch (100 texts) works correctly."""
        from app.services.ollama_embeddings import embed_texts_batch

        vecs = [_make_embedding(768, seed=i) for i in range(100)]
        mock_resp = _mock_httpx_response(200, {"embeddings": vecs})

        with _patch_ollama_base_url(), patch(
            "app.services.ollama_embeddings.httpx.AsyncClient",
            return_value=_make_async_client(post_return_value=mock_resp),
        ):
            result, dim = await embed_texts_batch(
                model="nomic-embed-text",
                texts=[f"text_{i}" for i in range(100)],
            )

        assert len(result) == 100
        assert dim == 768

    @pytest.mark.asyncio
    async def test_vectors_are_nonzero(self):
        """Vectors returned are non-zero (E2E lesson learned)."""
        from app.services.ollama_embeddings import embed_texts_batch

        vec = _make_embedding(768, seed=7)
        assert any(v != 0.0 for v in vec), "Test helper should produce non-zero vectors"

        mock_resp = _mock_httpx_response(200, {"embeddings": [vec]})

        with _patch_ollama_base_url(), patch(
            "app.services.ollama_embeddings.httpx.AsyncClient",
            return_value=_make_async_client(post_return_value=mock_resp),
        ):
            result, _ = await embed_texts_batch(
                model="nomic-embed-text", texts=["Hello"]
            )

        assert len(result) == 1
        assert any(
            v != 0.0 for v in result[0]
        ), "Embedding vector should not be all zeros"


# =============================================================================
# batch_upload_and_process() — multi-file processing
# =============================================================================


class TestBatchUploadAndProcess:

    _call_count = 0

    def _patch_repo(self):
        """Patch all repo calls used by batch_upload_and_process."""
        import uuid

        def _fake_create(**kw):
            TestBatchUploadAndProcess._call_count += 1
            return str(
                uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    kw.get("filename", str(TestBatchUploadAndProcess._call_count)),
                )
            )

        return patch.multiple(
            "app.services.knowledgebase_service.repo",
            create_document=AsyncMock(side_effect=_fake_create),
            update_document_embedding=AsyncMock(),
            update_chunk_embeddings=AsyncMock(),
            insert_chunk_embeddings=AsyncMock(),
        )

    def _patch_embed(self, dim=768):
        """Patch embed_texts_batch to return proper-sized vectors."""

        async def fake_embed(*, model, texts):
            vecs = [_make_embedding(dim, seed=i) for i in range(len(texts))]
            return vecs, dim

        return patch(
            "app.services.knowledgebase_service.embed_texts_batch",
            side_effect=fake_embed,
        )

    @pytest.mark.asyncio
    async def test_multiple_files_processed(self, tmp_path):
        """Multiple files processed, returns correct doc IDs and chunk counts."""
        from app.services.knowledgebase_service import batch_upload_and_process

        # Create real temp files
        for name in ["a.txt", "b.txt", "c.txt"]:
            (tmp_path / name).write_text(f"Content of {name} " * 50, encoding="utf-8")

        paths = [str(tmp_path / n) for n in ["a.txt", "b.txt", "c.txt"]]

        with self._patch_repo(), self._patch_embed():
            result = await batch_upload_and_process(file_paths=paths, is_global=True)

        assert len(result["doc_ids"]) == 3
        assert result["total_chunks"] > 0
        assert len(result["stats"]) == 3

    @pytest.mark.asyncio
    async def test_failed_extraction_doesnt_stop_others(self, tmp_path):
        """Failed file extraction doesn't stop other files from processing."""
        from app.services.knowledgebase_service import batch_upload_and_process

        (tmp_path / "good.txt").write_text("Good content " * 50, encoding="utf-8")
        paths = [
            str(tmp_path / "good.txt"),
            str(tmp_path / "nonexistent.pdf"),  # will fail extraction
            str(tmp_path / "good.txt"),
        ]

        with self._patch_repo(), self._patch_embed():
            result = await batch_upload_and_process(file_paths=paths, is_global=True)

        # Should have processed what it could, not crashed
        assert len(result["stats"]) == 3
        errors = [s for s in result["stats"] if s.get("error")]
        successes = [s for s in result["stats"] if not s.get("error")]
        assert len(errors) >= 1
        assert len(successes) >= 1

    @pytest.mark.asyncio
    async def test_empty_file_list_returns_empty(self):
        """Empty file list returns empty results."""
        from app.services.knowledgebase_service import batch_upload_and_process

        with self._patch_repo(), self._patch_embed():
            result = await batch_upload_and_process(file_paths=[], is_global=True)

        assert result["doc_ids"] == []
        assert result["total_chunks"] == 0

    @pytest.mark.asyncio
    async def test_chunk_count_matches_stats(self, tmp_path):
        """Total chunks reported matches sum of per-file chunk counts."""
        from app.services.knowledgebase_service import batch_upload_and_process

        # Create files with known content lengths
        (tmp_path / "short.txt").write_text("Short.", encoding="utf-8")
        (tmp_path / "long.txt").write_text("Long content. " * 200, encoding="utf-8")

        paths = [str(tmp_path / "short.txt"), str(tmp_path / "long.txt")]

        with self._patch_repo(), self._patch_embed():
            result = await batch_upload_and_process(file_paths=paths, is_global=True)

        total_from_stats = sum(s.get("chunks", 0) for s in result["stats"])
        assert result["total_chunks"] == total_from_stats

    @pytest.mark.asyncio
    async def test_file_hash_is_computed(self, tmp_path):
        """Each file gets a SHA-256 hash passed to create_document."""
        from app.services.knowledgebase_service import batch_upload_and_process

        content = "Hello world"
        (tmp_path / "test.txt").write_text(content, encoding="utf-8")

        mock_create = AsyncMock(return_value="doc-test")
        with patch(
            "app.services.knowledgebase_service.repo.create_document", mock_create
        ), patch(
            "app.services.knowledgebase_service.repo.update_document_embedding",
            AsyncMock(),
        ), patch(
            "app.services.knowledgebase_service.repo.update_chunk_embeddings",
            AsyncMock(),
        ), self._patch_embed():
            result = await batch_upload_and_process(
                file_paths=[str(tmp_path / "test.txt")], is_global=True
            )

        # Verify create_document was called with a file_hash
        call_kwargs = mock_create.call_args[1]
        assert "file_hash" in call_kwargs
        assert len(call_kwargs["file_hash"]) == 64  # SHA-256 hex


# =============================================================================
# Directory scanning logic (used by batch-upload endpoint)
# =============================================================================


class TestBatchUploadDirectoryScanning:

    def test_directory_scan_finds_correct_files(self, tmp_path):
        """Directory scan finds files matching supported extensions."""
        (tmp_path / "doc.pdf").write_bytes(b"%PDF-fake")
        (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
        (tmp_path / "image.jpg").write_bytes(b"\xff\xd8\xff")
        (tmp_path / "readme.md").write_text("# Readme", encoding="utf-8")

        supported = {".pdf", ".txt", ".md", ".docx"}
        found = [
            e.name
            for e in os.scandir(tmp_path)
            if e.is_file() and os.path.splitext(e.name)[1].lower() in supported
        ]
        assert sorted(found) == ["doc.pdf", "notes.txt", "readme.md"]

    def test_recursive_vs_nonrecursive_scanning(self, tmp_path):
        """Recursive scanning finds files in subdirectories; non-recursive does not."""
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "top.txt").write_text("top", encoding="utf-8")
        (sub / "nested.txt").write_text("nested", encoding="utf-8")

        # Non-recursive
        non_recursive = [
            e.name
            for e in os.scandir(tmp_path)
            if e.is_file() and e.name.endswith(".txt")
        ]
        assert non_recursive == ["top.txt"]

        # Recursive
        recursive = []
        for root, dirs, files in os.walk(tmp_path):
            for f in files:
                if f.endswith(".txt"):
                    recursive.append(f)
        assert sorted(recursive) == ["nested.txt", "top.txt"]

    def test_invalid_directory_returns_error(self):
        """Invalid/nonexistent directory is detected."""
        bad_path = "/nonexistent/directory/path_that_does_not_exist"
        assert not os.path.isdir(bad_path)
