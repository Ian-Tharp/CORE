"""
Tests for knowledge attribution verification functionality.

Covers:
- verify_knowledge_attribution() — chunk-level attribution verification
- verify_document_knowledge_integrity() — document-level integrity checks
- batch_verify_knowledge_attribution() — batch verification across knowledge base
"""

import pytest
from unittest.mock import patch, AsyncMock
from typing import List, Dict, Any
import uuid


# =============================================================================
# Test Helpers
# =============================================================================


def _make_test_chunk(chunk_id: str = None, document_id: str = None) -> Dict[str, Any]:
    """Create a test chunk record."""
    return {
        "id": chunk_id or str(uuid.uuid4()),
        "document_id": document_id or str(uuid.uuid4()),
        "chunk_index": 0,
        "text": "Test chunk content",
        "embedding_model": "nomic-embed-text",
        "embedding_dimensions": 768,
    }


def _make_test_document(doc_id: str = None) -> Dict[str, Any]:
    """Create a test document record."""
    return {
        "id": doc_id or str(uuid.uuid4()),
        "filename": "test.txt",
        "title": "Test Document",
        "original_name": "test.txt",
        "size": 1000,
        "mime_type": "text/plain",
        "description": "Test document",
        "is_global": False,
        "source": "user_upload",
        "status": "ready",
    }


def _make_test_instance(instance_id: str = None) -> object:
    """Create a test AgentInstance object."""
    from app.repository.instance_repository import AgentInstance, InstanceStatus
    from datetime import datetime

    return AgentInstance(
        id=uuid.UUID(instance_id) if instance_id else uuid.uuid4(),
        container_id="test-container",
        agent_id="test-agent",
        agent_role="test-role",
        status=InstanceStatus.READY,
        created_at=datetime.utcnow(),
    )


def _make_test_attribution(chunk_id: str, instance_id: str) -> Dict[str, Any]:
    """Create test attribution metadata."""
    return {
        "chunk_id": chunk_id,
        "source_instance_id": instance_id,
        "created_at": "2026-02-17T14:00:00Z",
        "metadata": {"contribution_type": "original_knowledge"},
    }


# =============================================================================
# verify_knowledge_attribution() Tests
# =============================================================================


class TestVerifyKnowledgeAttribution:

    @pytest.mark.asyncio
    async def test_empty_chunk_list_returns_empty_result(self):
        """Empty chunk ID list returns empty verification result."""
        from app.services.knowledgebase_service import verify_knowledge_attribution

        result = await verify_knowledge_attribution(chunk_ids=[])

        assert result == {"verified": [], "warnings": [], "total_chunks": 0}

    @pytest.mark.asyncio
    async def test_chunk_with_valid_attribution(self):
        """Chunk with valid attribution and existing instance passes verification."""
        from app.services.knowledgebase_service import verify_knowledge_attribution

        chunk_id = str(uuid.uuid4())
        instance_id = str(uuid.uuid4())
        test_instance = _make_test_instance(instance_id)
        test_attribution = _make_test_attribution(chunk_id, instance_id)

        with patch(
            "app.services.knowledgebase_service.repo.get_chunk_attributions"
        ) as mock_get_attr, patch(
            "app.repository.instance_repository.get_instance"
        ) as mock_get_instance:

            mock_get_attr.return_value = {chunk_id: test_attribution}
            mock_get_instance.return_value = test_instance

            result = await verify_knowledge_attribution(chunk_ids=[chunk_id])

        assert len(result["verified"]) == 1
        assert len(result["warnings"]) == 0
        assert result["verified"][0]["chunk_id"] == chunk_id
        assert result["verified"][0]["source_instance"]["id"] == instance_id

    @pytest.mark.asyncio
    async def test_chunk_with_missing_attribution(self):
        """Chunk with no attribution metadata generates warning."""
        from app.services.knowledgebase_service import verify_knowledge_attribution

        chunk_id = str(uuid.uuid4())

        with patch(
            "app.services.knowledgebase_service.repo.get_chunk_attributions"
        ) as mock_get_attr:
            mock_get_attr.return_value = {chunk_id: None}

            result = await verify_knowledge_attribution(chunk_ids=[chunk_id])

        assert len(result["verified"]) == 0
        assert len(result["warnings"]) == 1
        assert result["warnings"][0]["type"] == "missing_attribution"
        assert result["warnings"][0]["chunk_id"] == chunk_id

    @pytest.mark.asyncio
    async def test_chunk_with_orphaned_attribution(self):
        """Chunk with attribution to non-existent instance generates orphaned warning."""
        from app.services.knowledgebase_service import verify_knowledge_attribution

        chunk_id = str(uuid.uuid4())
        nonexistent_instance_id = str(uuid.uuid4())
        test_attribution = _make_test_attribution(chunk_id, nonexistent_instance_id)

        with patch(
            "app.services.knowledgebase_service.repo.get_chunk_attributions"
        ) as mock_get_attr, patch(
            "app.repository.instance_repository.get_instance"
        ) as mock_get_instance:

            mock_get_attr.return_value = {chunk_id: test_attribution}
            mock_get_instance.return_value = None

            result = await verify_knowledge_attribution(chunk_ids=[chunk_id])

        assert len(result["verified"]) == 0
        assert len(result["warnings"]) == 1
        assert result["warnings"][0]["type"] == "orphaned_attribution"
        assert result["warnings"][0]["source_instance_id"] == nonexistent_instance_id

    @pytest.mark.asyncio
    async def test_chunk_with_attribution_but_no_instance_id(self):
        """Chunk with attribution metadata but missing source_instance_id generates warning."""
        from app.services.knowledgebase_service import verify_knowledge_attribution

        chunk_id = str(uuid.uuid4())
        bad_attribution = {
            "chunk_id": chunk_id,
            "source_instance_id": None,  # Missing instance ID
            "created_at": "2026-02-17T14:00:00Z",
        }

        with patch(
            "app.services.knowledgebase_service.repo.get_chunk_attributions"
        ) as mock_get_attr:
            mock_get_attr.return_value = {chunk_id: bad_attribution}

            result = await verify_knowledge_attribution(chunk_ids=[chunk_id])

        assert len(result["verified"]) == 0
        assert len(result["warnings"]) == 1
        assert result["warnings"][0]["type"] == "missing_source_instance"

    @pytest.mark.asyncio
    async def test_mixed_chunk_verification_results(self):
        """Mixed set of chunks with various attribution states."""
        from app.services.knowledgebase_service import verify_knowledge_attribution

        valid_chunk_id = str(uuid.uuid4())
        missing_chunk_id = str(uuid.uuid4())
        orphaned_chunk_id = str(uuid.uuid4())

        instance_id = str(uuid.uuid4())
        test_instance = _make_test_instance(instance_id)

        chunk_ids = [valid_chunk_id, missing_chunk_id, orphaned_chunk_id]

        with patch(
            "app.services.knowledgebase_service.repo.get_chunk_attributions"
        ) as mock_get_attr, patch(
            "app.repository.instance_repository.get_instance"
        ) as mock_get_instance:

            mock_get_attr.return_value = {
                valid_chunk_id: _make_test_attribution(valid_chunk_id, instance_id),
                missing_chunk_id: None,
                orphaned_chunk_id: _make_test_attribution(
                    orphaned_chunk_id, "nonexistent"
                ),
            }

            def mock_instance_lookup(lookup_id):
                return test_instance if lookup_id == instance_id else None

            mock_get_instance.side_effect = mock_instance_lookup

            result = await verify_knowledge_attribution(chunk_ids=chunk_ids)

        assert result["total_chunks"] == 3
        assert result["verified_count"] == 1
        assert result["warning_count"] == 2
        assert len(result["verified"]) == 1
        assert len(result["warnings"]) == 2


# =============================================================================
# verify_document_knowledge_integrity() Tests
# =============================================================================


class TestVerifyDocumentKnowledgeIntegrity:

    @pytest.mark.asyncio
    async def test_nonexistent_document_returns_error(self):
        """Non-existent document ID returns error result."""
        from app.services.knowledgebase_service import (
            verify_document_knowledge_integrity,
        )

        nonexistent_id = str(uuid.uuid4())

        with patch(
            "app.services.knowledgebase_service.repo.get_document"
        ) as mock_get_doc:
            mock_get_doc.return_value = None

            result = await verify_document_knowledge_integrity(
                document_id=nonexistent_id
            )

        assert "error" in result
        assert result["verified"] == False

    @pytest.mark.asyncio
    async def test_document_with_no_chunks(self):
        """Document with no chunks passes verification with appropriate message."""
        from app.services.knowledgebase_service import (
            verify_document_knowledge_integrity,
        )

        doc_id = str(uuid.uuid4())
        test_doc = _make_test_document(doc_id)

        with patch(
            "app.services.knowledgebase_service.repo.get_document"
        ) as mock_get_doc, patch(
            "app.services.knowledgebase_service.repo.list_chunks_for_document"
        ) as mock_list_chunks:

            mock_get_doc.return_value = test_doc
            mock_list_chunks.return_value = []

            result = await verify_document_knowledge_integrity(document_id=doc_id)

        assert result["verified"] == True
        assert result["chunks_count"] == 0
        assert "no chunks" in result["message"]

    @pytest.mark.asyncio
    async def test_document_with_verified_chunks(self):
        """Document with chunks that pass attribution verification."""
        from app.services.knowledgebase_service import (
            verify_document_knowledge_integrity,
        )

        doc_id = str(uuid.uuid4())
        chunk_id = str(uuid.uuid4())
        test_doc = _make_test_document(doc_id)
        test_chunk = _make_test_chunk(chunk_id, doc_id)

        with patch(
            "app.services.knowledgebase_service.repo.get_document"
        ) as mock_get_doc, patch(
            "app.services.knowledgebase_service.repo.list_chunks_for_document"
        ) as mock_list_chunks, patch(
            "app.services.knowledgebase_service.repo.get_document_attribution"
        ) as mock_doc_attr, patch(
            "app.services.knowledgebase_service.verify_knowledge_attribution"
        ) as mock_verify:

            mock_get_doc.return_value = test_doc
            mock_list_chunks.return_value = [test_chunk]
            mock_doc_attr.return_value = None
            mock_verify.return_value = {
                "verified": [{"chunk_id": chunk_id}],
                "warnings": [],
                "total_chunks": 1,
                "verified_count": 1,
                "warning_count": 0,
            }

            result = await verify_document_knowledge_integrity(document_id=doc_id)

        assert result["verified"] == True
        assert result["chunks_count"] == 1
        assert result["summary"]["total_issues"] == 0

    @pytest.mark.asyncio
    async def test_document_with_attribution_warnings(self):
        """Document with chunk attribution issues is marked as unverified."""
        from app.services.knowledgebase_service import (
            verify_document_knowledge_integrity,
        )

        doc_id = str(uuid.uuid4())
        chunk_id = str(uuid.uuid4())
        test_doc = _make_test_document(doc_id)
        test_chunk = _make_test_chunk(chunk_id, doc_id)

        with patch(
            "app.services.knowledgebase_service.repo.get_document"
        ) as mock_get_doc, patch(
            "app.services.knowledgebase_service.repo.list_chunks_for_document"
        ) as mock_list_chunks, patch(
            "app.services.knowledgebase_service.repo.get_document_attribution"
        ) as mock_doc_attr, patch(
            "app.services.knowledgebase_service.verify_knowledge_attribution"
        ) as mock_verify:

            mock_get_doc.return_value = test_doc
            mock_list_chunks.return_value = [test_chunk]
            mock_doc_attr.return_value = None
            mock_verify.return_value = {
                "verified": [],
                "warnings": [{"type": "missing_attribution", "chunk_id": chunk_id}],
                "total_chunks": 1,
                "verified_count": 0,
                "warning_count": 1,
            }

            result = await verify_document_knowledge_integrity(document_id=doc_id)

        assert result["verified"] == False
        assert result["summary"]["total_issues"] == 1
        assert result["summary"]["chunk_issues"] == 1


# =============================================================================
# batch_verify_knowledge_attribution() Tests
# =============================================================================


class TestBatchVerifyKnowledgeAttribution:

    @pytest.mark.asyncio
    async def test_batch_verification_with_limit(self):
        """Batch verification respects document limit."""
        from app.services.knowledgebase_service import (
            batch_verify_knowledge_attribution,
        )

        # Create 5 test documents but limit to 3
        test_docs = [_make_test_document() for _ in range(5)]

        with patch(
            "app.services.knowledgebase_service.repo.list_documents"
        ) as mock_list_docs, patch(
            "app.services.knowledgebase_service.verify_document_knowledge_integrity"
        ) as mock_verify_doc:

            mock_list_docs.return_value = test_docs
            mock_verify_doc.return_value = {
                "verified": True,
                "summary": {"total_issues": 0},
            }

            result = await batch_verify_knowledge_attribution(limit=3)

        assert result["processed_documents"] == 3  # Limited to 3
        assert mock_verify_doc.call_count == 3

    @pytest.mark.asyncio
    async def test_batch_verification_health_score(self):
        """Batch verification calculates correct health score."""
        from app.services.knowledgebase_service import (
            batch_verify_knowledge_attribution,
        )

        test_docs = [_make_test_document() for _ in range(4)]

        with patch(
            "app.services.knowledgebase_service.repo.list_documents"
        ) as mock_list_docs, patch(
            "app.services.knowledgebase_service.verify_document_knowledge_integrity"
        ) as mock_verify_doc:

            mock_list_docs.return_value = test_docs

            # 3 healthy, 1 with issues
            def mock_verify_response(document_id):
                if document_id == test_docs[0]["id"]:
                    return {"verified": False, "summary": {"total_issues": 2}}
                return {"verified": True, "summary": {"total_issues": 0}}

            mock_verify_doc.side_effect = lambda document_id: mock_verify_response(
                document_id
            )

            result = await batch_verify_knowledge_attribution()

        assert result["processed_documents"] == 4
        assert result["healthy_documents"] == 3
        assert result["documents_with_issues"] == 1
        assert result["health_score"] == 0.75  # 3/4

    @pytest.mark.asyncio
    async def test_batch_verification_handles_failures(self):
        """Batch verification handles individual document verification failures."""
        from app.services.knowledgebase_service import (
            batch_verify_knowledge_attribution,
        )

        test_docs = [_make_test_document() for _ in range(2)]

        with patch(
            "app.services.knowledgebase_service.repo.list_documents"
        ) as mock_list_docs, patch(
            "app.services.knowledgebase_service.verify_document_knowledge_integrity"
        ) as mock_verify_doc:

            mock_list_docs.return_value = test_docs

            def mock_verify_response(document_id):
                if document_id == test_docs[0]["id"]:
                    raise Exception("Verification failed")
                return {"verified": True, "summary": {"total_issues": 0}}

            mock_verify_doc.side_effect = lambda document_id: mock_verify_response(
                document_id
            )

            result = await batch_verify_knowledge_attribution()

        assert result["processed_documents"] == 2
        assert result["verified_documents"] == 1
        assert result["failed_verifications"] == 1
        assert len(result["failures"]) == 1

    @pytest.mark.asyncio
    async def test_empty_knowledge_base_returns_perfect_health(self):
        """Empty knowledge base returns 1.0 health score."""
        from app.services.knowledgebase_service import (
            batch_verify_knowledge_attribution,
        )

        with patch(
            "app.services.knowledgebase_service.repo.list_documents"
        ) as mock_list_docs:
            mock_list_docs.return_value = []

            result = await batch_verify_knowledge_attribution()

        assert result["processed_documents"] == 0
        assert result["health_score"] == 1.0
        assert result["batch_verification_complete"] == True
