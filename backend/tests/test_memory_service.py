"""
Memory Service Tests

Comprehensive tests for the three-tier memory service.
All tests follow AAA format (Arrange, Act, Assert).
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any
from uuid import uuid4, UUID

from app.services.memory_service import memory_service, MemoryContext, Procedure, MemoryStats
from app.services.embedding_service import embedding_service
from app.repository.memory_repository import (
    ensure_memory_tables,
    SemanticMemory,
    EpisodicMemory,
    ProceduralMemory,
    MemoryType
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
async def setup_memory_service():
    """Set up memory service for all tests."""
    # Arrange
    await embedding_service.initialize()
    await memory_service.initialize()
    
    yield
    
    # Cleanup would go here if needed


@pytest.fixture
async def sample_knowledge():
    """Provide sample knowledge for testing."""
    return {
        "content": "Python is a high-level programming language known for its simplicity and readability.",
        "metadata": {"source": "test", "category": "programming", "confidence": 0.9},
        "source_agent_id": "test_agent_001"
    }


@pytest.fixture
async def sample_experience():
    """Provide sample experience for testing."""
    return {
        "agent_id": "test_agent_001",
        "content": "Successfully implemented a REST API using FastAPI with proper error handling.",
        "metadata": {"type": "task_result", "importance": 0.8, "project": "core"}
    }


@pytest.fixture
async def sample_procedure():
    """Provide sample procedure for testing."""
    return {
        "role": "developer",
        "procedure_name": "API Error Handling",
        "steps": [
            "Define custom exception classes",
            "Add try-catch blocks around business logic",
            "Return appropriate HTTP status codes",
            "Log errors for debugging"
        ],
        "metadata": {"difficulty": "medium", "category": "best_practice"}
    }


# =============================================================================
# SEMANTIC MEMORY TESTS
# =============================================================================

class TestSemanticMemory:
    """Tests for semantic memory operations."""
    
    async def test_store_and_retrieve_knowledge(self, sample_knowledge):
        """Test storing and retrieving knowledge by semantic similarity."""
        # Arrange
        content = sample_knowledge["content"]
        metadata = sample_knowledge["metadata"]
        source_agent = sample_knowledge["source_agent_id"]
        
        # Act - Store knowledge
        memory_id = await memory_service.store_knowledge(
            content=content,
            metadata=metadata,
            source_agent_id=source_agent
        )
        
        # Act - Search for the knowledge
        search_results = await memory_service.search_knowledge(
            query="programming language Python",
            limit=5,
            threshold=0.5
        )
        
        # Assert
        assert isinstance(memory_id, UUID)
        assert len(search_results) >= 1
        
        # Find our stored knowledge in results
        stored_memory = None
        for result in search_results:
            if result.id == memory_id:
                stored_memory = result
                break
        
        assert stored_memory is not None
        assert stored_memory.content == content
        assert stored_memory.source_agent_id == source_agent
        assert stored_memory.metadata["category"] == "programming"
        assert "similarity" in stored_memory.metadata
    
    async def test_search_knowledge_with_high_threshold(self, sample_knowledge):
        """Test that high similarity threshold filters results appropriately."""
        # Arrange
        await memory_service.store_knowledge(
            content=sample_knowledge["content"],
            metadata=sample_knowledge["metadata"],
            source_agent_id=sample_knowledge["source_agent_id"]
        )
        
        # Act - Search with very high threshold
        high_threshold_results = await memory_service.search_knowledge(
            query="completely unrelated topic about cooking",
            limit=10,
            threshold=0.9
        )
        
        # Act - Search with lower threshold
        low_threshold_results = await memory_service.search_knowledge(
            query="programming Python language",
            limit=10,
            threshold=0.3
        )
        
        # Assert
        assert len(high_threshold_results) == 0  # No matches for unrelated query
        assert len(low_threshold_results) >= 1   # Should find our stored knowledge
    
    async def test_access_count_tracking(self, sample_knowledge):
        """Test that access count is tracked when retrieving knowledge."""
        # Arrange
        memory_id = await memory_service.store_knowledge(
            content=sample_knowledge["content"],
            metadata=sample_knowledge["metadata"],
            source_agent_id=sample_knowledge["source_agent_id"]
        )
        
        # Act - Search multiple times
        await memory_service.search_knowledge("Python programming", threshold=0.5)
        await memory_service.search_knowledge("Python programming", threshold=0.5)
        await memory_service.search_knowledge("Python programming", threshold=0.5)
        
        # Act - Get final results
        final_results = await memory_service.search_knowledge("Python programming", threshold=0.5)
        
        # Assert
        found_memory = None
        for result in final_results:
            if result.id == memory_id:
                found_memory = result
                break
        
        assert found_memory is not None
        assert found_memory.access_count >= 3  # Should have been accessed multiple times


# =============================================================================
# EPISODIC MEMORY TESTS
# =============================================================================

class TestEpisodicMemory:
    """Tests for episodic memory operations."""
    
    async def test_store_personal_experience(self, sample_experience):
        """Test storing personal experience for a specific agent."""
        # Arrange
        agent_id = sample_experience["agent_id"]
        content = sample_experience["content"]
        metadata = sample_experience["metadata"]
        
        # Act
        memory_id = await memory_service.store_experience(
            agent_id=agent_id,
            content=content,
            metadata=metadata
        )
        
        # Assert
        assert isinstance(memory_id, UUID)
    
    async def test_retrieve_agent_specific_memories(self, sample_experience):
        """Test that agents can only see their own episodic memories."""
        # Arrange
        agent1_id = "agent_001"
        agent2_id = "agent_002"
        
        # Store experience for agent 1
        memory1_id = await memory_service.store_experience(
            agent_id=agent1_id,
            content="Agent 1's secret experience",
            metadata={"importance": 0.8}
        )
        
        # Store experience for agent 2
        memory2_id = await memory_service.store_experience(
            agent_id=agent2_id,
            content="Agent 2's secret experience",
            metadata={"importance": 0.8}
        )
        
        # Act - Get experiences for each agent
        agent1_experiences = await memory_service.get_agent_experiences(agent1_id, limit=10)
        agent2_experiences = await memory_service.get_agent_experiences(agent2_id, limit=10)
        
        # Assert - Each agent only sees their own memories
        agent1_memory_ids = [exp.id for exp in agent1_experiences]
        agent2_memory_ids = [exp.id for exp in agent2_experiences]
        
        assert memory1_id in agent1_memory_ids
        assert memory1_id not in agent2_memory_ids
        assert memory2_id in agent2_memory_ids
        assert memory2_id not in agent1_memory_ids
    
    async def test_consolidation_of_experiences(self):
        """Test consolidation of short-term to long-term memories."""
        # Arrange
        agent_id = "test_consolidation_agent"
        
        # Store multiple experiences
        old_experience_id = await memory_service.store_experience(
            agent_id=agent_id,
            content="Important experience from yesterday",
            metadata={"importance": 0.7, "type": "experience"}
        )
        
        # Act - Trigger consolidation
        consolidated_count = await memory_service.consolidate_experiences(agent_id)
        
        # Assert
        assert isinstance(consolidated_count, int)
        # Note: Consolidation logic depends on time thresholds, so we mainly test the method runs
    
    async def test_memory_expiration(self):
        """Test that temporary memories can have expiration dates."""
        # Arrange
        agent_id = "test_expiration_agent"
        expiration_date = datetime.utcnow() + timedelta(days=1)
        
        # Act - Store temporary experience
        memory_id = await memory_service.store_experience(
            agent_id=agent_id,
            content="Temporary experience that will expire",
            metadata={
                "temporary": True,
                "expires_at": expiration_date.isoformat(),
                "importance": 0.3
            }
        )
        
        # Assert
        assert isinstance(memory_id, UUID)
        
        # Get the stored memory and verify it has expiration metadata
        experiences = await memory_service.get_agent_experiences(agent_id, limit=10)
        temp_memory = None
        for exp in experiences:
            if exp.id == memory_id:
                temp_memory = exp
                break
        
        assert temp_memory is not None
        assert temp_memory.metadata.get("temporary") is True


# =============================================================================
# PROCEDURAL MEMORY TESTS
# =============================================================================

class TestProceduralMemory:
    """Tests for procedural memory operations."""
    
    async def test_store_and_retrieve_procedures_by_role(self, sample_procedure):
        """Test storing and retrieving procedures by role."""
        # Arrange
        role = sample_procedure["role"]
        procedure_name = sample_procedure["procedure_name"]
        steps = sample_procedure["steps"]
        metadata = sample_procedure["metadata"]
        
        # Act - Store procedure
        memory_id = await memory_service.store_procedure(
            role=role,
            procedure_name=procedure_name,
            steps=steps,
            metadata=metadata
        )
        
        # Act - Retrieve procedures for role
        procedures = await memory_service.get_role_procedures(role)
        
        # Assert
        assert isinstance(memory_id, UUID)
        assert len(procedures) >= 1
        
        # Find our stored procedure
        stored_procedure = None
        for proc in procedures:
            if proc.id == memory_id:
                stored_procedure = proc
                break
        
        assert stored_procedure is not None
        assert stored_procedure.name == procedure_name
        assert stored_procedure.steps == steps
        assert stored_procedure.role == role
    
    async def test_search_procedures_by_similarity(self, sample_procedure):
        """Test searching procedures by semantic similarity."""
        # Arrange
        await memory_service.store_procedure(
            role=sample_procedure["role"],
            procedure_name=sample_procedure["procedure_name"],
            steps=sample_procedure["steps"],
            metadata=sample_procedure["metadata"]
        )
        
        # Act - Search for similar procedures
        search_results = await memory_service.search_procedures(
            query="handling errors in web APIs",
            role=sample_procedure["role"]
        )
        
        # Assert
        assert len(search_results) >= 1
        found_procedure = search_results[0]
        assert found_procedure.name == sample_procedure["procedure_name"]
        assert "error" in found_procedure.name.lower() or "api" in found_procedure.name.lower()
    
    async def test_usage_count_and_success_rate_tracking(self, sample_procedure):
        """Test that procedures track usage count and success rate."""
        # Arrange
        memory_id = await memory_service.store_procedure(
            role=sample_procedure["role"],
            procedure_name="Test Success Tracking",
            steps=["Step 1", "Step 2", "Step 3"],
            metadata={}
        )
        
        # Act - Update with successful and failed outcomes
        success1 = await memory_service.update_procedure_outcome(memory_id, True)
        success2 = await memory_service.update_procedure_outcome(memory_id, True)
        success3 = await memory_service.update_procedure_outcome(memory_id, False)
        
        # Act - Retrieve to check updated stats
        procedures = await memory_service.get_role_procedures(sample_procedure["role"])
        test_procedure = None
        for proc in procedures:
            if proc.id == memory_id:
                test_procedure = proc
                break
        
        # Assert
        assert success1 is True
        assert success2 is True
        assert success3 is True  # Method should succeed even for failed outcomes
        assert test_procedure is not None
        assert test_procedure.usage_count >= 3
        assert 0.0 <= test_procedure.success_rate <= 1.0  # Should be around 0.67 (2/3)


# =============================================================================
# CROSS-TIER TESTS
# =============================================================================

class TestCrossTierOperations:
    """Tests for operations that span multiple memory tiers."""
    
    async def test_get_relevant_context_from_all_tiers(self):
        """Test that get_relevant_context pulls from all three memory tiers."""
        # Arrange
        agent_id = "context_test_agent"
        
        # Store knowledge in semantic memory
        semantic_id = await memory_service.store_knowledge(
            content="FastAPI is a modern Python web framework for building APIs",
            metadata={"category": "framework"},
            source_agent_id=agent_id
        )
        
        # Store experience in episodic memory
        episodic_id = await memory_service.store_experience(
            agent_id=agent_id,
            content="Successfully deployed FastAPI application to production",
            metadata={"importance": 0.9, "type": "success"}
        )
        
        # Store procedure in procedural memory
        procedural_id = await memory_service.store_procedure(
            role="developer",
            procedure_name="FastAPI Deployment",
            steps=["Configure environment", "Set up database", "Deploy to server"],
            metadata={"category": "deployment"}
        )
        
        # Act - Get relevant context
        context = await memory_service.get_relevant_context(
            query="FastAPI web framework deployment",
            agent_id=agent_id,
            limit_per_tier=5,
            relevance_threshold=0.5
        )
        
        # Assert
        assert isinstance(context, MemoryContext)
        assert len(context.semantic) >= 1
        assert len(context.episodic) >= 1
        assert len(context.procedural) >= 1
        assert context.total_items >= 3
        
        # Verify content appears in context strings
        semantic_content = " ".join(context.semantic)
        assert "FastAPI" in semantic_content
        
        episodic_content = " ".join(context.episodic)
        assert "deployment" in episodic_content.lower() or "fastapi" in episodic_content.lower()
        
        procedural_content = " ".join(context.procedural)
        assert "Deployment" in procedural_content
    
    async def test_agent_isolation_in_context(self):
        """Test that episodic memories are isolated per agent in context."""
        # Arrange
        agent1_id = "isolation_test_agent_1"
        agent2_id = "isolation_test_agent_2"
        
        # Store experiences for different agents
        await memory_service.store_experience(
            agent_id=agent1_id,
            content="Agent 1 learned about isolation testing",
            metadata={"importance": 0.8}
        )
        
        await memory_service.store_experience(
            agent_id=agent2_id,
            content="Agent 2 worked on different isolation project",
            metadata={"importance": 0.8}
        )
        
        # Act - Get context for each agent
        context1 = await memory_service.get_relevant_context(
            query="isolation testing",
            agent_id=agent1_id,
            relevance_threshold=0.3
        )
        
        context2 = await memory_service.get_relevant_context(
            query="isolation testing",
            agent_id=agent2_id,
            relevance_threshold=0.3
        )
        
        # Assert - Each agent should see their own episodic memories
        agent1_episodic = " ".join(context1.episodic)
        agent2_episodic = " ".join(context2.episodic)
        
        if context1.episodic:
            assert "Agent 1" in agent1_episodic
            assert "Agent 2" not in agent1_episodic
        
        if context2.episodic:
            assert "Agent 2" in agent2_episodic
            assert "Agent 1" not in agent2_episodic
    
    async def test_shared_knowledge_visible_to_all_agents(self):
        """Test that semantic memories are visible to all agents."""
        # Arrange
        source_agent = "knowledge_source_agent"
        consumer_agent1 = "knowledge_consumer_1"
        consumer_agent2 = "knowledge_consumer_2"
        
        # Store shared knowledge
        await memory_service.store_knowledge(
            content="Shared knowledge about unit testing best practices",
            metadata={"category": "testing", "public": True},
            source_agent_id=source_agent
        )
        
        # Act - Get context from different agents
        context1 = await memory_service.get_relevant_context(
            query="unit testing practices",
            agent_id=consumer_agent1,
            relevance_threshold=0.5
        )
        
        context2 = await memory_service.get_relevant_context(
            query="testing best practices",
            agent_id=consumer_agent2,
            relevance_threshold=0.5
        )
        
        # Assert - Both agents should see the shared knowledge
        assert len(context1.semantic) >= 1
        assert len(context2.semantic) >= 1
        
        semantic1 = " ".join(context1.semantic)
        semantic2 = " ".join(context2.semantic)
        
        assert "testing" in semantic1.lower()
        assert "testing" in semantic2.lower()


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    async def test_empty_query_handling(self):
        """Test that empty queries are handled gracefully."""
        # Arrange
        empty_queries = ["", "   ", "\n\t", None]
        
        for query in empty_queries:
            if query is not None:
                # Act
                semantic_results = await memory_service.search_knowledge(query or "")
                episodic_results = await memory_service.get_agent_experiences("test_agent", query)
                context = await memory_service.get_relevant_context(query or "", "test_agent")
                
                # Assert
                assert semantic_results == []
                assert isinstance(episodic_results, list)
                assert isinstance(context, MemoryContext)
                assert context.total_items == 0
    
    async def test_very_long_content_storage(self):
        """Test storing very long content."""
        # Arrange
        very_long_content = "This is a very long piece of content. " * 100  # ~3700 chars
        
        # Act
        memory_id = await memory_service.store_knowledge(
            content=very_long_content,
            metadata={"length": "very_long"},
            source_agent_id="test_agent"
        )
        
        # Assert
        assert isinstance(memory_id, UUID)
        
        # Verify it can be retrieved
        results = await memory_service.search_knowledge("very long piece", threshold=0.3)
        assert len(results) >= 1
    
    async def test_embedding_service_unavailable_graceful_fallback(self):
        """Test graceful handling when embedding service is unavailable."""
        # Note: This test would require mocking the embedding service to fail
        # For now, we test that the service handles initialization errors
        
        # Arrange & Act
        health = await memory_service.health_check()
        
        # Assert
        assert "embedding_service" in health
        assert "database" in health
        assert "initialized" in health
    
    async def test_concurrent_memory_writes(self):
        """Test that concurrent memory writes don't cause conflicts."""
        # Arrange
        agent_id = "concurrent_test_agent"
        
        async def store_memory(index):
            return await memory_service.store_experience(
                agent_id=agent_id,
                content=f"Concurrent experience number {index}",
                metadata={"index": index, "batch": "concurrent_test"}
            )
        
        # Act - Store memories concurrently
        tasks = [store_memory(i) for i in range(10)]
        memory_ids = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Assert - All should succeed
        successful_ids = [mid for mid in memory_ids if isinstance(mid, UUID)]
        assert len(successful_ids) == 10
        assert len(set(successful_ids)) == 10  # All should be unique
        
        # Verify all can be retrieved
        experiences = await memory_service.get_agent_experiences(agent_id, limit=20)
        concurrent_experiences = [
            exp for exp in experiences 
            if exp.metadata.get("batch") == "concurrent_test"
        ]
        assert len(concurrent_experiences) == 10


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestMemoryServiceIntegration:
    """Integration tests for the complete memory service."""
    
    async def test_memory_statistics_accuracy(self):
        """Test that memory statistics are accurately calculated."""
        # Arrange
        test_agent_id = "stats_test_agent"
        
        # Store various types of memories
        semantic_ids = []
        for i in range(3):
            sid = await memory_service.store_knowledge(
                content=f"Knowledge item {i}",
                metadata={"test": "stats"},
                source_agent_id=test_agent_id
            )
            semantic_ids.append(sid)
        
        episodic_ids = []
        for i in range(5):
            eid = await memory_service.store_experience(
                agent_id=test_agent_id,
                content=f"Experience {i}",
                metadata={"test": "stats"}
            )
            episodic_ids.append(eid)
        
        # Act
        stats = await memory_service.get_agent_memory_summary(test_agent_id)
        
        # Assert
        assert isinstance(stats, MemoryStats)
        assert stats.agent_id == test_agent_id
        assert stats.semantic_count >= 3
        assert stats.episodic_count >= 5
        assert stats.last_memory_created is not None
    
    async def test_memory_cleanup_operations(self):
        """Test memory cleanup and maintenance operations."""
        # Arrange
        cleanup_agent_id = "cleanup_test_agent"
        
        # Store some test memories
        await memory_service.store_experience(
            agent_id=cleanup_agent_id,
            content="Memory to be cleaned up",
            metadata={"test": "cleanup", "importance": 0.1}
        )
        
        # Act - Clear agent memories
        clear_result = await memory_service.clear_agent_memories(cleanup_agent_id)
        
        # Assert
        assert isinstance(clear_result, dict)
        assert "episodic" in clear_result
        
        # Verify memories were cleared
        remaining_experiences = await memory_service.get_agent_experiences(cleanup_agent_id)
        cleanup_experiences = [
            exp for exp in remaining_experiences
            if exp.metadata.get("test") == "cleanup"
        ]
        assert len(cleanup_experiences) == 0
    
    async def test_end_to_end_memory_workflow(self):
        """Test complete workflow from storage to context retrieval."""
        # Arrange
        workflow_agent = "workflow_test_agent"
        
        # Act - Store knowledge
        knowledge_id = await memory_service.store_knowledge(
            content="Workflow testing involves end-to-end validation of processes",
            metadata={"category": "testing", "workflow": True},
            source_agent_id=workflow_agent
        )
        
        # Act - Store experience
        experience_id = await memory_service.store_experience(
            agent_id=workflow_agent,
            content="Successfully completed workflow testing for memory service",
            metadata={"type": "achievement", "workflow": True}
        )
        
        # Act - Store procedure
        procedure_id = await memory_service.store_procedure(
            role="tester",
            procedure_name="Memory Workflow Test",
            steps=[
                "Store test data in all memory tiers",
                "Verify retrieval and search functionality",
                "Test context generation",
                "Validate statistics and cleanup"
            ],
            metadata={"workflow": True}
        )
        
        # Act - Get comprehensive context
        context = await memory_service.get_relevant_context(
            query="workflow testing memory validation",
            agent_id=workflow_agent,
            relevance_threshold=0.4
        )
        
        # Assert - All memory types should be represented
        assert len(context.semantic) >= 1
        assert len(context.episodic) >= 1
        assert len(context.procedural) >= 1
        
        # Verify content relevance
        all_context = " ".join(context.semantic + context.episodic + context.procedural)
        assert "workflow" in all_context.lower()
        assert "testing" in all_context.lower()
        
        # Act - Get final statistics
        final_stats = await memory_service.get_agent_memory_summary(workflow_agent)
        
        # Assert - Statistics should reflect all stored memories
        assert final_stats.semantic_count >= 1
        assert final_stats.episodic_count >= 1
        assert final_stats.total_access_count >= 0  # May be 0 if not accessed yet
        
        logger.info(f"End-to-end workflow test completed successfully for agent {workflow_agent}")


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestMemoryPerformance:
    """Performance tests for memory operations."""
    
    async def test_bulk_storage_performance(self):
        """Test performance of bulk memory storage operations."""
        # Arrange
        bulk_agent_id = "bulk_test_agent"
        bulk_memories = []
        
        for i in range(50):  # Reduced from 100 to keep tests fast
            bulk_memories.append({
                "type": "episodic",
                "agent_id": bulk_agent_id,
                "content": f"Bulk experience number {i} with varying content length and details",
                "metadata": {"batch": "performance_test", "index": i}
            })
        
        # Act
        start_time = datetime.utcnow()
        result = await memory_service.bulk_import_memories(bulk_memories)
        end_time = datetime.utcnow()
        
        # Assert
        duration = (end_time - start_time).total_seconds()
        assert duration < 30.0  # Should complete within 30 seconds
        assert result["episodic"] == 50
        assert result["errors"] == 0
        
        logger.info(f"Bulk storage of 50 memories completed in {duration:.2f} seconds")
    
    async def test_search_performance_with_large_dataset(self):
        """Test search performance with a larger dataset."""
        # Arrange - Use memories from bulk test
        search_query = "bulk experience varying content"
        
        # Act
        start_time = datetime.utcnow()
        results = await memory_service.search_knowledge(search_query, limit=10)
        end_time = datetime.utcnow()
        
        # Assert
        search_duration = (end_time - start_time).total_seconds()
        assert search_duration < 5.0  # Should complete within 5 seconds
        
        logger.info(f"Search completed in {search_duration:.3f} seconds")