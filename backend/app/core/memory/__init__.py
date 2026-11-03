"""
CORE's knowledge self-model: what the system KNOWS (spec Phase 3, §4.5).

The Capability Registry (`app.core.registry`) answers "what can I do?"; this
package answers "what do I know that this request relies on?". Comprehension
uses it to notice knowledge gaps before spending on the O→R→E loop.
"""

from app.core.memory.grounding import (
    KnowledgeAssessment,
    MemoryGrounder,
    MemoryHit,
    NullMemoryGrounder,
    StoreMemoryGrounder,
    assess_knowledge,
    build_memory_store,
    detect_knowledge_dependencies,
    memory_grounding_enabled,
)

__all__ = [
    "KnowledgeAssessment",
    "MemoryGrounder",
    "MemoryHit",
    "NullMemoryGrounder",
    "StoreMemoryGrounder",
    "assess_knowledge",
    "build_memory_store",
    "detect_knowledge_dependencies",
    "memory_grounding_enabled",
]
