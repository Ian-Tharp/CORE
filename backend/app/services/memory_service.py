"""
Memory Service

Three-tier memory service using LangMem backed by Postgres + pgvector.
Integrates semantic, episodic, and procedural memory with the existing CORE architecture.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from langmem import create_memory_manager
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel

from app.dependencies import get_db_pool
from app.services.embedding_service import embedding_service
from app.repository.memory_repository import (
    ensure_memory_tables,
    create_semantic_memory,
    create_episodic_memory,
    create_procedural_memory,
    search_semantic_memories,
    search_episodic_memories,
    search_procedural_memories,
    consolidate_episodic_memories,
    get_relevant_context,
    get_memory_stats,
    clear_agent_memories,
    expire_old_memories,
    update_procedure_success_rate,
    SemanticMemory,
    EpisodicMemory,
    ProceduralMemory,
    MemoryType,
    MemoryStats
)

logger = logging.getLogger(__name__)


class MemoryContext(BaseModel):
    """Memory context returned from searches."""
    semantic: List[str] = []
    episodic: List[str] = []
    procedural: List[str] = []
    total_items: int = 0
    relevance_threshold: float = 0.7


class Procedure(BaseModel):
    """Procedure definition."""
    id: UUID
    name: str
    steps: List[str]
    role: str
    success_rate: float = 0.0
    usage_count: int = 0
    metadata: Dict[str, Any] = {}


class MemoryService:
    """
    Three-tier memory service using LangMem backed by Postgres + pgvector.
    
    Provides semantic, episodic, and procedural memory capabilities
    integrated with the existing CORE architecture.
    """
    
    def __init__(self):
        self.langmem_manager = None
        self.store = None
        self._initialized = False
        
    async def initialize(self) -> None:
        """Set up LangMem stores backed by our existing Postgres + pgvector."""
        try:
            # Initialize embedding service
            await embedding_service.initialize()
            
            # Ensure memory tables exist
            await ensure_memory_tables()
            
            # Set up LangMem PostgreSQL store
            pool = await get_db_pool()
            
            # Create LangMem store — using InMemoryStore as the LangMem coordination layer.
            # All persistent memory operations go through our custom pgvector repository,
            # so this store is only used for LangMem's internal state management.
            self.store = InMemoryStore()
            
            # Initialize LangMem memory manager
            # Note: Using a lightweight schema since we're primarily using our custom repository
            self.langmem_manager = create_memory_manager(
                model="ollama/llama3.2:3b",  # Use local Ollama model
                schemas=[],  # We'll use our custom schemas
                store=self.store,
                instructions="""
                Extract and organize information for the three-tier memory system:
                - Semantic: Facts and knowledge that are useful across agents
                - Episodic: Personal experiences and context for specific agents
                - Procedural: Step-by-step procedures and learned behaviors
                Focus on actionable information that improves future interactions.
                """
            )
            
            self._initialized = True
            logger.info("MemoryService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MemoryService: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the memory service."""
        try:
            embedding_health = await embedding_service.health_check()
            embedding_info = await embedding_service.get_model_info()
            
            # Test database connection
            pool = await get_db_pool()
            async with pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                db_healthy = result == 1
            
            return {
                "initialized": self._initialized,
                "embedding_service": embedding_health,
                "database": db_healthy,
                "embedding_model": embedding_info,
                "langmem_store": self.store is not None
            }
            
        except Exception as e:
            logger.error(f"Memory service health check failed: {e}")
            return {
                "initialized": False,
                "error": str(e)
            }
    
    # =============================================================================
    # SEMANTIC MEMORY (shared knowledge)
    # =============================================================================
    
    async def store_knowledge(
        self, 
        content: str, 
        metadata: Dict[str, Any], 
        source_agent_id: Optional[str] = None
    ) -> UUID:
        """Store a fact or piece of knowledge accessible to all agents."""
        if not self._initialized:
            raise RuntimeError("MemoryService not initialized")
        
        try:
            # Generate embedding
            embedding = await embedding_service.generate_embedding(content)
            
            # Create semantic memory
            memory = SemanticMemory(
                id=uuid4(),
                content=content,
                embedding=embedding,
                metadata=metadata,
                source_agent_id=source_agent_id,
                confidence=metadata.get('confidence', 1.0)
            )
            
            memory_id = await create_semantic_memory(memory)
            
            logger.info(f"Stored semantic memory: {memory_id} from agent {source_agent_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store knowledge: {e}")
            raise
    
    async def search_knowledge(
        self, 
        query: str, 
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[SemanticMemory]:
        """Search shared knowledge by semantic similarity."""
        if not self._initialized:
            raise RuntimeError("MemoryService not initialized")
        
        try:
            if not query.strip():
                return []
            
            # Generate query embedding
            query_embedding = await embedding_service.generate_embedding(query)
            
            # Search semantic memories
            memories = await search_semantic_memories(
                query_embedding, limit=limit, threshold=threshold
            )
            
            logger.debug(f"Found {len(memories)} semantic memories for query: {query[:50]}")
            return memories
            
        except Exception as e:
            logger.error(f"Failed to search knowledge: {e}")
            return []
    
    # =============================================================================
    # EPISODIC MEMORY (personal experience)
    # =============================================================================
    
    async def store_experience(
        self, 
        agent_id: str, 
        content: str, 
        metadata: Dict[str, Any]
    ) -> UUID:
        """Store a personal experience for a specific agent."""
        if not self._initialized:
            raise RuntimeError("MemoryService not initialized")
        
        try:
            # Generate embedding
            embedding = await embedding_service.generate_embedding(content)
            
            # Set expiration if not specified
            expires_at = metadata.get('expires_at')
            if not expires_at and metadata.get('temporary', False):
                expires_at = datetime.utcnow() + timedelta(days=30)
            
            # Create episodic memory
            memory = EpisodicMemory(
                id=uuid4(),
                agent_id=agent_id,
                content=content,
                embedding=embedding,
                memory_type=metadata.get('type', 'experience'),
                metadata=metadata,
                importance=metadata.get('importance', 0.5),
                confidence=metadata.get('confidence', 1.0),
                expires_at=expires_at
            )
            
            memory_id = await create_episodic_memory(memory)
            
            logger.debug(f"Stored episodic memory for agent {agent_id}: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store experience for agent {agent_id}: {e}")
            raise
    
    async def get_agent_experiences(
        self, 
        agent_id: str, 
        query: Optional[str] = None, 
        limit: int = 10,
        memory_type: Optional[str] = None
    ) -> List[EpisodicMemory]:
        """Retrieve an agent's personal experiences, optionally filtered by query."""
        if not self._initialized:
            raise RuntimeError("MemoryService not initialized")
        
        try:
            query_embedding = None
            if query and query.strip():
                query_embedding = await embedding_service.generate_embedding(query)
            
            memories = await search_episodic_memories(
                agent_id=agent_id,
                query_embedding=query_embedding,
                limit=limit,
                memory_type=memory_type
            )
            
            logger.debug(f"Retrieved {len(memories)} episodic memories for agent {agent_id}")
            return memories
            
        except Exception as e:
            logger.error(f"Failed to get experiences for agent {agent_id}: {e}")
            return []
    
    async def consolidate_experiences(self, agent_id: str) -> int:
        """Consolidate short-term episodic memories into long-term (periodic task)."""
        if not self._initialized:
            raise RuntimeError("MemoryService not initialized")
        
        try:
            count = await consolidate_episodic_memories(agent_id)
            
            if count > 0:
                logger.info(f"Consolidated {count} experiences for agent {agent_id}")
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to consolidate experiences for agent {agent_id}: {e}")
            return 0
    
    # =============================================================================
    # PROCEDURAL MEMORY (role definitions)
    # =============================================================================
    
    async def store_procedure(
        self, 
        role: str, 
        procedure_name: str, 
        steps: List[str], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> UUID:
        """Store a learned procedure for a role."""
        if not self._initialized:
            raise RuntimeError("MemoryService not initialized")
        
        try:
            if metadata is None:
                metadata = {}
            
            # Create content for embedding (procedure name + steps)
            content = f"{procedure_name}: " + " -> ".join(steps)
            
            # Generate embedding
            embedding = await embedding_service.generate_embedding(content)
            
            # Create procedural memory
            memory = ProceduralMemory(
                id=uuid4(),
                role=role,
                procedure_name=procedure_name,
                content=content,
                steps=steps,
                embedding=embedding,
                metadata=metadata,
                confidence=metadata.get('confidence', 1.0)
            )
            
            memory_id = await create_procedural_memory(memory)
            
            logger.info(f"Stored procedure '{procedure_name}' for role {role}: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store procedure for role {role}: {e}")
            raise
    
    async def get_role_procedures(self, role: str) -> List[Procedure]:
        """Get all procedures defined for a role."""
        if not self._initialized:
            raise RuntimeError("MemoryService not initialized")
        
        try:
            memories = await search_procedural_memories(role=role)
            
            procedures = []
            for memory in memories:
                procedure = Procedure(
                    id=memory.id,
                    name=memory.procedure_name,
                    steps=memory.steps,
                    role=memory.role,
                    success_rate=memory.success_rate,
                    usage_count=memory.usage_count,
                    metadata=memory.metadata
                )
                procedures.append(procedure)
            
            logger.debug(f"Retrieved {len(procedures)} procedures for role {role}")
            return procedures
            
        except Exception as e:
            logger.error(f"Failed to get procedures for role {role}: {e}")
            return []
    
    async def search_procedures(
        self, 
        query: str, 
        role: Optional[str] = None
    ) -> List[Procedure]:
        """Search procedures by semantic similarity."""
        if not self._initialized:
            raise RuntimeError("MemoryService not initialized")
        
        try:
            if not query.strip():
                # Return all procedures for the role
                return await self.get_role_procedures(role) if role else []
            
            # Generate query embedding
            query_embedding = await embedding_service.generate_embedding(query)
            
            # Search procedural memories
            memories = await search_procedural_memories(
                role=role,
                query_embedding=query_embedding
            )
            
            procedures = []
            for memory in memories:
                procedure = Procedure(
                    id=memory.id,
                    name=memory.procedure_name,
                    steps=memory.steps,
                    role=memory.role,
                    success_rate=memory.success_rate,
                    usage_count=memory.usage_count,
                    metadata=memory.metadata
                )
                procedures.append(procedure)
            
            logger.debug(f"Found {len(procedures)} procedures matching query: {query[:50]}")
            return procedures
            
        except Exception as e:
            logger.error(f"Failed to search procedures: {e}")
            return []
    
    async def update_procedure_outcome(
        self, 
        procedure_id: UUID, 
        success: bool
    ) -> bool:
        """Update procedure success rate based on execution outcome."""
        if not self._initialized:
            raise RuntimeError("MemoryService not initialized")
        
        try:
            result = await update_procedure_success_rate(procedure_id, success)
            
            if result:
                logger.debug(f"Updated procedure {procedure_id} with success={success}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to update procedure outcome: {e}")
            return False
    
    # =============================================================================
    # CROSS-TIER OPERATIONS
    # =============================================================================
    
    async def get_relevant_context(
        self, 
        query: str, 
        agent_id: Optional[str] = None,
        limit_per_tier: int = 5,
        relevance_threshold: float = 0.7
    ) -> MemoryContext:
        """Get relevant context from all memory tiers for a given query."""
        if not self._initialized:
            raise RuntimeError("MemoryService not initialized")
        
        try:
            if not query.strip():
                return MemoryContext()
            
            # Generate query embedding
            query_embedding = await embedding_service.generate_embedding(query)
            
            # Get memories from all tiers
            context_memories = await get_relevant_context(
                query_embedding=query_embedding,
                agent_id=agent_id,
                limit_per_tier=limit_per_tier,
                threshold=relevance_threshold
            )
            
            # Convert to context strings
            semantic_context = [
                f"Knowledge: {mem.content} (confidence: {mem.confidence:.2f})"
                for mem in context_memories['semantic']
            ]
            
            episodic_context = [
                f"Experience: {mem.content} (importance: {mem.importance:.2f})"
                for mem in context_memories['episodic']
            ]
            
            procedural_context = [
                f"Procedure '{mem.procedure_name}': {' -> '.join(mem.steps)} (success: {mem.success_rate:.1%})"
                for mem in context_memories['procedural']
            ]
            
            total_items = len(semantic_context) + len(episodic_context) + len(procedural_context)
            
            context = MemoryContext(
                semantic=semantic_context,
                episodic=episodic_context,
                procedural=procedural_context,
                total_items=total_items,
                relevance_threshold=relevance_threshold
            )
            
            logger.debug(f"Generated context with {total_items} items for query: {query[:50]}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to get relevant context: {e}")
            return MemoryContext()
    
    # =============================================================================
    # MEMORY MANAGEMENT
    # =============================================================================
    
    async def get_agent_memory_summary(self, agent_id: str) -> MemoryStats:
        """Get stats about an agent's memory usage across all tiers."""
        if not self._initialized:
            raise RuntimeError("MemoryService not initialized")
        
        try:
            stats = await get_memory_stats(agent_id)
            logger.debug(f"Retrieved memory stats for agent {agent_id}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get memory summary for agent {agent_id}: {e}")
            return MemoryStats(agent_id=agent_id)
    
    async def clear_agent_memories(
        self, 
        agent_id: str, 
        tier: Optional[str] = None
    ) -> Dict[str, int]:
        """Clear memories for an agent (optionally by tier)."""
        if not self._initialized:
            raise RuntimeError("MemoryService not initialized")
        
        try:
            memory_tier = None
            if tier:
                memory_tier = MemoryType(tier.lower())
            
            counts = await clear_agent_memories(agent_id, memory_tier)
            
            total_cleared = sum(counts.values())
            logger.info(f"Cleared {total_cleared} memories for agent {agent_id} (tier: {tier or 'all'})")
            
            return counts
            
        except Exception as e:
            logger.error(f"Failed to clear memories for agent {agent_id}: {e}")
            return {}
    
    async def cleanup_expired_memories(self) -> Dict[str, int]:
        """Clean up expired memories (periodic maintenance task)."""
        if not self._initialized:
            raise RuntimeError("MemoryService not initialized")
        
        try:
            expired_count = await expire_old_memories()
            
            # Also consolidate memories for all active agents
            # This would need integration with the agent registry to get active agents
            # For now, we'll just return the expired count
            
            result = {
                "expired_episodic": expired_count,
                "consolidated": 0  # Could be expanded to track consolidation across agents
            }
            
            if expired_count > 0:
                logger.info(f"Cleaned up {expired_count} expired memories")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired memories: {e}")
            return {"error": str(e)}
    
    async def bulk_import_memories(
        self,
        memories: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Import memories in bulk (useful for migrations or seeding)."""
        if not self._initialized:
            raise RuntimeError("MemoryService not initialized")
        
        counts = {"semantic": 0, "episodic": 0, "procedural": 0, "errors": 0}
        
        for memory_data in memories:
            try:
                memory_type = memory_data.get("type", "semantic")
                
                if memory_type == "semantic":
                    await self.store_knowledge(
                        content=memory_data["content"],
                        metadata=memory_data.get("metadata", {}),
                        source_agent_id=memory_data.get("source_agent_id")
                    )
                    counts["semantic"] += 1
                    
                elif memory_type == "episodic":
                    await self.store_experience(
                        agent_id=memory_data["agent_id"],
                        content=memory_data["content"],
                        metadata=memory_data.get("metadata", {})
                    )
                    counts["episodic"] += 1
                    
                elif memory_type == "procedural":
                    await self.store_procedure(
                        role=memory_data["role"],
                        procedure_name=memory_data["procedure_name"],
                        steps=memory_data["steps"],
                        metadata=memory_data.get("metadata", {})
                    )
                    counts["procedural"] += 1
                    
            except Exception as e:
                logger.error(f"Failed to import memory: {e}")
                counts["errors"] += 1
        
        logger.info(f"Bulk import completed: {counts}")
        return counts


# Global memory service instance
memory_service = MemoryService()