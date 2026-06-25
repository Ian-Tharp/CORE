"""
Capability Registry — the single source of truth for what CORE can do.

The dispatcher and (later) MCP servers register their capabilities here, so the
description the system reasons over can never drift from what actually runs.
Comprehension queries it for capability *presence*; orchestration sequences plans
from it.
"""

from app.core.registry.capability import CapabilityEntry, CapabilityRegistry

__all__ = ["CapabilityEntry", "CapabilityRegistry"]
