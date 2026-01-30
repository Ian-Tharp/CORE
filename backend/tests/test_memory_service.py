"""
Tests for the LangMem Three-Tier Memory Service.

Tests the memory service layer including semantic, episodic, and procedural
memory operations, cross-tier retrieval, and memory management.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from uuid import uuid4, UUID
from datetime import datetime, timedelta

from app.services.memory_service import MemoryService, MemoryContext, Procedure
from app.repository.memory_repository import (
    SemanticMemory,
    EpisodicMemory,
    ProceduralMemory,
    MemoryType,
    MemoryStats,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def memory_service():
    """Create an initialized memory service with mocked dependencies."""
    service = MemoryService()
    service._initialized = True
    return service


@pytest.fixture
def sample_embedding():
    """Return a sample embedding vector (768 dimensions)."""
    return [0.01 * i for i in range(768)]


@pytest.fixture
def sample_semantic_memory():
    """Return a sample semantic memory."""
    return SemanticMemory(
        id=uuid4(),
        content="Python is a high-level programming language.",
        embedding=[0.0] * 768,
        metadata={"domain": "programming"},
        source_agent_id="researcher-001",
        confidence=0.95,
        access_count=3,
        created_at=datetime.utcnow(),
        last_accessed=datetime.utcnow(),
    )


@pytest.fixture
def sample_episodic_memory():
    """Return a sample episodic memory."""
    return EpisodicMemory(
        id=uuid4(),
        agent_id="analyst-001",
        content="Completed data analysis for project Alpha.",
        embedding=[0.0] * 768,
        memory_type="task_result",
        metadata={"project": "alpha"},
        importance=0.8,
        confidence=0.9,
        access_count=1,
        consolidated=False,
        created_at=datetime.utcnow(),
        last_accessed=None,
        expires_at=datetime.utcnow() + timedelta(days=90),
    )


@pytest.fixture
def sample_procedural_memory():
    """Return a sample procedural memory."""
    return ProceduralMemory(
        id=uuid4(),
        role="data_analyst",
        procedure_name="run_regression_analysis",
        content="run_regression_analysis: Load data -> Clean data -> Fit model -> Evaluate",
        steps=["Load data", "Clean data", "Fit model", "Evaluate"],
        embedding=[0.0] * 768,
        metadata={"framework": "scikit-learn"},
        success_rate=0.85,
        usage_count=10,
        confidence=0.9,
        access_count=5,
        created_at=datetime.utcnow(),
        last_accessed=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestInitialization:
    """Tests for memory service initialization."""

    def test_service_starts_uninitialized(self):
        """Memory service starts in an uninitialized state."""
        service = MemoryService()
        assert service._initialized is False
        assert service.langmem_manager is None
        assert service.store is None

    @pytest.mark.asyncio
    async def test_uninitialized_service_raises_on_operations(self):
        """Uninitialized service raises RuntimeError."""
        service = MemoryService()

        with pytest.raises(RuntimeError, match="not initialized"):
            await service.store_knowledge("test", {})

        with pytest.raises(RuntimeError, match="not initialized"):
            await service.search_knowledge("test")

        with pytest.raises(RuntimeError, match="not initialized"):
            await service.store_experience("agent-1", "test", {})

        with pytest.raises(RuntimeError, match="not initialized"):
            await service.store_procedure("role", "proc", ["step"])


# =============================================================================
# SEMANTIC MEMORY TESTS
# =============================================================================

class TestSemanticMemory:
    """Tests for semantic (shared knowledge) operations."""

    @pytest.mark.asyncio
    async def test_store_knowledge_success(self, memory_service, sample_embedding):
        """Successfully store a piece of knowledge."""
        expected_id = uuid4()

        with patch("app.services.memory_service.embedding_service") as mock_embed, \
             patch("app.services.memory_service.create_semantic_memory") as mock_create:
            mock_embed.generate_embedding = AsyncMock(return_value=sample_embedding)
            mock_create.return_value = expected_id

            result = await memory_service.store_knowledge(
                content="Python supports async/await.",
                metadata={"domain": "programming"},
                source_agent_id="researcher-001",
            )

            assert result == expected_id
            mock_embed.generate_embedding.assert_awaited_once_with("Python supports async/await.")
            mock_create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_store_knowledge_propagates_errors(self, memory_service):
        """Store knowledge propagates exceptions from embedding service."""
        with patch("app.services.memory_service.embedding_service") as mock_embed:
            mock_embed.generate_embedding = AsyncMock(side_effect=RuntimeError("model unavailable"))

            with pytest.raises(RuntimeError, match="model unavailable"):
                await memory_service.store_knowledge("test content", {})

    @pytest.mark.asyncio
    async def test_search_knowledge_success(self, memory_service, sample_semantic_memory, sample_embedding):
        """Search knowledge returns matching memories."""
        with patch("app.services.memory_service.embedding_service") as mock_embed, \
             patch("app.services.memory_service.search_semantic_memories") as mock_search:
            mock_embed.generate_embedding = AsyncMock(return_value=sample_embedding)
            mock_search.return_value = [sample_semantic_memory]

            results = await memory_service.search_knowledge("programming languages", limit=5, threshold=0.8)

            assert len(results) == 1
            assert results[0].content == sample_semantic_memory.content
            mock_search.assert_awaited_once_with(sample_embedding, limit=5, threshold=0.8)

    @pytest.mark.asyncio
    async def test_search_knowledge_empty_query(self, memory_service):
        """Search with empty query returns empty list."""
        results = await memory_service.search_knowledge("", limit=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_knowledge_returns_empty_on_error(self, memory_service):
        """Search returns empty list on error instead of raising."""
        with patch("app.services.memory_service.embedding_service") as mock_embed:
            mock_embed.generate_embedding = AsyncMock(side_effect=Exception("DB error"))

            results = await memory_service.search_knowledge("test query")
            assert results == []


# =============================================================================
# EPISODIC MEMORY TESTS
# =============================================================================

class TestEpisodicMemory:
    """Tests for episodic (personal experience) operations."""

    @pytest.mark.asyncio
    async def test_store_experience_success(self, memory_service, sample_embedding):
        """Successfully store an agent experience."""
        expected_id = uuid4()

        with patch("app.services.memory_service.embedding_service") as mock_embed, \
             patch("app.services.memory_service.create_episodic_memory") as mock_create:
            mock_embed.generate_embedding = AsyncMock(return_value=sample_embedding)
            mock_create.return_value = expected_id

            result = await memory_service.store_experience(
                agent_id="analyst-001",
                content="Completed analysis of dataset.",
                metadata={"type": "task_result", "importance": 0.8},
            )

            assert result == expected_id
            mock_create.assert_awaited_once()
            # Verify the memory was created with correct importance
            created_memory = mock_create.call_args[0][0]
            assert created_memory.agent_id == "analyst-001"
            assert created_memory.importance == 0.8

    @pytest.mark.asyncio
    async def test_store_experience_temporary_gets_expiration(self, memory_service, sample_embedding):
        """Temporary experiences get an expiration date."""
        expected_id = uuid4()

        with patch("app.services.memory_service.embedding_service") as mock_embed, \
             patch("app.services.memory_service.create_episodic_memory") as mock_create:
            mock_embed.generate_embedding = AsyncMock(return_value=sample_embedding)
            mock_create.return_value = expected_id

            await memory_service.store_experience(
                agent_id="agent-001",
                content="Temp data",
                metadata={"temporary": True},
            )

            created_memory = mock_create.call_args[0][0]
            assert created_memory.expires_at is not None

    @pytest.mark.asyncio
    async def test_get_agent_experiences_with_query(self, memory_service, sample_episodic_memory, sample_embedding):
        """Retrieve agent experiences filtered by query."""
        with patch("app.services.memory_service.embedding_service") as mock_embed, \
             patch("app.services.memory_service.search_episodic_memories") as mock_search:
            mock_embed.generate_embedding = AsyncMock(return_value=sample_embedding)
            mock_search.return_value = [sample_episodic_memory]

            results = await memory_service.get_agent_experiences(
                agent_id="analyst-001",
                query="data analysis",
                limit=5,
            )

            assert len(results) == 1
            mock_search.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_agent_experiences_without_query(self, memory_service, sample_episodic_memory):
        """Retrieve agent experiences without query returns recent."""
        with patch("app.services.memory_service.search_episodic_memories") as mock_search:
            mock_search.return_value = [sample_episodic_memory]

            results = await memory_service.get_agent_experiences(
                agent_id="analyst-001",
                limit=10,
            )

            assert len(results) == 1
            # Should NOT call embedding service if no query
            mock_search.assert_awaited_once_with(
                agent_id="analyst-001",
                query_embedding=None,
                limit=10,
                memory_type=None,
            )

    @pytest.mark.asyncio
    async def test_consolidate_experiences(self, memory_service):
        """Consolidation delegates to repository."""
        with patch("app.services.memory_service.consolidate_episodic_memories") as mock_consolidate:
            mock_consolidate.return_value = 5

            count = await memory_service.consolidate_experiences("agent-001")

            assert count == 5
            mock_consolidate.assert_awaited_once_with("agent-001")


# =============================================================================
# PROCEDURAL MEMORY TESTS
# =============================================================================

class TestProceduralMemory:
    """Tests for procedural (role-based) operations."""

    @pytest.mark.asyncio
    async def test_store_procedure_success(self, memory_service, sample_embedding):
        """Successfully store a procedure."""
        expected_id = uuid4()

        with patch("app.services.memory_service.embedding_service") as mock_embed, \
             patch("app.services.memory_service.create_procedural_memory") as mock_create:
            mock_embed.generate_embedding = AsyncMock(return_value=sample_embedding)
            mock_create.return_value = expected_id

            result = await memory_service.store_procedure(
                role="data_analyst",
                procedure_name="data_cleaning",
                steps=["Load CSV", "Remove nulls", "Normalize"],
                metadata={"framework": "pandas"},
            )

            assert result == expected_id
            mock_create.assert_awaited_once()
            created = mock_create.call_args[0][0]
            assert created.role == "data_analyst"
            assert created.procedure_name == "data_cleaning"
            assert len(created.steps) == 3

    @pytest.mark.asyncio
    async def test_get_role_procedures(self, memory_service, sample_procedural_memory):
        """Get procedures for a role."""
        with patch("app.services.memory_service.search_procedural_memories") as mock_search:
            mock_search.return_value = [sample_procedural_memory]

            procedures = await memory_service.get_role_procedures("data_analyst")

            assert len(procedures) == 1
            assert procedures[0].name == sample_procedural_memory.procedure_name
            assert procedures[0].success_rate == 0.85
            mock_search.assert_awaited_once_with(role="data_analyst")

    @pytest.mark.asyncio
    async def test_search_procedures_with_query(self, memory_service, sample_procedural_memory, sample_embedding):
        """Search procedures by semantic similarity."""
        with patch("app.services.memory_service.embedding_service") as mock_embed, \
             patch("app.services.memory_service.search_procedural_memories") as mock_search:
            mock_embed.generate_embedding = AsyncMock(return_value=sample_embedding)
            mock_search.return_value = [sample_procedural_memory]

            procedures = await memory_service.search_procedures(
                query="regression analysis",
                role="data_analyst",
            )

            assert len(procedures) == 1
            mock_search.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_procedures_empty_query_with_role(self, memory_service, sample_procedural_memory):
        """Empty query with role falls back to get_role_procedures."""
        with patch("app.services.memory_service.search_procedural_memories") as mock_search:
            mock_search.return_value = [sample_procedural_memory]

            procedures = await memory_service.search_procedures(query="", role="data_analyst")

            assert len(procedures) == 1

    @pytest.mark.asyncio
    async def test_update_procedure_outcome(self, memory_service):
        """Update procedure success rate."""
        proc_id = uuid4()

        with patch("app.services.memory_service.update_procedure_success_rate") as mock_update:
            mock_update.return_value = True

            result = await memory_service.update_procedure_outcome(proc_id, success=True)

            assert result is True
            mock_update.assert_awaited_once_with(proc_id, True)


# =============================================================================
# CROSS-TIER TESTS
# =============================================================================

class TestCrossTierOperations:
    """Tests for cross-tier memory retrieval."""

    @pytest.mark.asyncio
    async def test_get_relevant_context(
        self, memory_service, sample_semantic_memory, sample_episodic_memory,
        sample_procedural_memory, sample_embedding
    ):
        """Get relevant context from all tiers."""
        with patch("app.services.memory_service.embedding_service") as mock_embed, \
             patch("app.services.memory_service.get_relevant_context") as mock_ctx:
            mock_embed.generate_embedding = AsyncMock(return_value=sample_embedding)
            mock_ctx.return_value = {
                "semantic": [sample_semantic_memory],
                "episodic": [sample_episodic_memory],
                "procedural": [sample_procedural_memory],
            }

            context = await memory_service.get_relevant_context(
                query="data analysis techniques",
                agent_id="analyst-001",
                limit_per_tier=5,
                relevance_threshold=0.7,
            )

            assert isinstance(context, MemoryContext)
            assert len(context.semantic) == 1
            assert len(context.episodic) == 1
            assert len(context.procedural) == 1
            assert context.total_items == 3

    @pytest.mark.asyncio
    async def test_get_relevant_context_empty_query(self, memory_service):
        """Empty query returns empty context."""
        context = await memory_service.get_relevant_context(query="")

        assert context.total_items == 0
        assert context.semantic == []
        assert context.episodic == []
        assert context.procedural == []

    @pytest.mark.asyncio
    async def test_get_relevant_context_returns_empty_on_error(self, memory_service):
        """Errors return empty context instead of raising."""
        with patch("app.services.memory_service.embedding_service") as mock_embed:
            mock_embed.generate_embedding = AsyncMock(side_effect=Exception("fail"))

            context = await memory_service.get_relevant_context(query="test")

            assert context.total_items == 0


# =============================================================================
# MEMORY MANAGEMENT TESTS
# =============================================================================

class TestMemoryManagement:
    """Tests for memory management operations."""

    @pytest.mark.asyncio
    async def test_get_agent_memory_summary(self, memory_service):
        """Get memory stats for an agent."""
        expected_stats = MemoryStats(
            agent_id="agent-001",
            semantic_count=10,
            episodic_count=25,
            procedural_count=5,
            total_access_count=100,
            last_memory_created=datetime.utcnow(),
        )

        with patch("app.services.memory_service.get_memory_stats") as mock_stats:
            mock_stats.return_value = expected_stats

            stats = await memory_service.get_agent_memory_summary("agent-001")

            assert stats.agent_id == "agent-001"
            assert stats.semantic_count == 10
            assert stats.episodic_count == 25
            assert stats.total_access_count == 100

    @pytest.mark.asyncio
    async def test_clear_agent_memories_all_tiers(self, memory_service):
        """Clear all memory tiers for an agent."""
        with patch("app.services.memory_service.clear_agent_memories") as mock_clear:
            mock_clear.return_value = {"semantic": 5, "episodic": 10, "procedural": 0}

            counts = await memory_service.clear_agent_memories("agent-001")

            assert counts["semantic"] == 5
            assert counts["episodic"] == 10

    @pytest.mark.asyncio
    async def test_clear_agent_memories_single_tier(self, memory_service):
        """Clear a specific memory tier for an agent."""
        with patch("app.services.memory_service.clear_agent_memories") as mock_clear:
            mock_clear.return_value = {"episodic": 10}

            counts = await memory_service.clear_agent_memories("agent-001", tier="episodic")

            assert counts["episodic"] == 10
            mock_clear.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cleanup_expired_memories(self, memory_service):
        """Cleanup expired memories."""
        with patch("app.services.memory_service.expire_old_memories") as mock_expire:
            mock_expire.return_value = 15

            result = await memory_service.cleanup_expired_memories()

            assert result["expired_episodic"] == 15

    @pytest.mark.asyncio
    async def test_bulk_import_mixed_types(self, memory_service, sample_embedding):
        """Bulk import handles mixed memory types."""
        with patch("app.services.memory_service.embedding_service") as mock_embed, \
             patch("app.services.memory_service.create_semantic_memory") as mock_sem, \
             patch("app.services.memory_service.create_episodic_memory") as mock_epi, \
             patch("app.services.memory_service.create_procedural_memory") as mock_proc:
            mock_embed.generate_embedding = AsyncMock(return_value=sample_embedding)
            mock_sem.return_value = uuid4()
            mock_epi.return_value = uuid4()
            mock_proc.return_value = uuid4()

            memories = [
                {"type": "semantic", "content": "fact 1", "metadata": {}},
                {"type": "episodic", "content": "exp 1", "agent_id": "a1", "metadata": {}},
                {"type": "procedural", "content": "proc 1", "role": "r1",
                 "procedure_name": "p1", "steps": ["s1"]},
            ]

            counts = await memory_service.bulk_import_memories(memories)

            assert counts["semantic"] == 1
            assert counts["episodic"] == 1
            assert counts["procedural"] == 1
            assert counts["errors"] == 0

    @pytest.mark.asyncio
    async def test_bulk_import_handles_errors_gracefully(self, memory_service):
        """Bulk import counts errors instead of crashing."""
        with patch("app.services.memory_service.embedding_service") as mock_embed:
            mock_embed.generate_embedding = AsyncMock(side_effect=Exception("fail"))

            memories = [
                {"type": "semantic", "content": "will fail"},
            ]

            counts = await memory_service.bulk_import_memories(memories)

            assert counts["errors"] == 1
            assert counts["semantic"] == 0


# =============================================================================
# HEALTH CHECK TESTS
# =============================================================================

class TestHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self, memory_service):
        """Health check reports healthy when all components work."""
        mock_conn = MagicMock()
        mock_conn.fetchval = AsyncMock(return_value=1)

        mock_pool = MagicMock()
        # pool.acquire() must return a sync object that supports async with
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_pool.acquire.return_value = mock_cm

        with patch("app.services.memory_service.embedding_service") as mock_embed, \
             patch("app.services.memory_service.get_db_pool", new_callable=AsyncMock) as mock_get_pool:
            mock_embed.health_check = AsyncMock(return_value=True)
            mock_embed.get_model_info = AsyncMock(return_value={"model": "nomic-embed-text"})
            mock_get_pool.return_value = mock_pool

            health = await memory_service.health_check()

            assert health["initialized"] is True
            assert health["embedding_service"] is True
            assert health["database"] is True

    @pytest.mark.asyncio
    async def test_health_check_returns_error_on_failure(self):
        """Health check returns error info when components fail."""
        service = MemoryService()  # Uninitialized

        with patch("app.services.memory_service.embedding_service") as mock_embed:
            mock_embed.health_check = AsyncMock(side_effect=Exception("connection refused"))
            mock_embed.get_model_info = AsyncMock(side_effect=Exception("connection refused"))

            health = await service.health_check()

            assert health["initialized"] is False
            assert "error" in health


# =============================================================================
# REPOSITORY UNIT TESTS
# =============================================================================

class TestMemoryRepository:
    """Tests for repository-level functions."""

    def test_memory_type_enum(self):
        """MemoryType enum has all three tiers."""
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.PROCEDURAL.value == "procedural"

    def test_semantic_memory_model(self, sample_semantic_memory):
        """SemanticMemory model validates correctly."""
        assert sample_semantic_memory.source_agent_id == "researcher-001"
        assert sample_semantic_memory.confidence == 0.95

    def test_episodic_memory_model(self, sample_episodic_memory):
        """EpisodicMemory model validates correctly."""
        assert sample_episodic_memory.agent_id == "analyst-001"
        assert sample_episodic_memory.importance == 0.8
        assert sample_episodic_memory.consolidated is False

    def test_procedural_memory_model(self, sample_procedural_memory):
        """ProceduralMemory model validates correctly."""
        assert sample_procedural_memory.role == "data_analyst"
        assert len(sample_procedural_memory.steps) == 4
        assert sample_procedural_memory.success_rate == 0.85

    def test_memory_stats_model(self):
        """MemoryStats model validates correctly."""
        stats = MemoryStats(
            agent_id="test-agent",
            semantic_count=5,
            episodic_count=10,
            procedural_count=3,
            total_access_count=50,
        )
        assert stats.agent_id == "test-agent"
        assert stats.last_memory_created is None

    @pytest.mark.asyncio
    async def test_update_access_counts_validates_table_name(self):
        """_update_access_counts rejects invalid table names."""
        from app.repository.memory_repository import _update_access_counts

        with pytest.raises(ValueError, match="Invalid table name"):
            await _update_access_counts("users; DROP TABLE --", [uuid4()])
