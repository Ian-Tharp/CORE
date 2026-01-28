"""
Sandbox module for secure agent execution.

This module provides:
- ContainerManager: Manages Docker containers for sandboxed execution
- Security configurations: TrustLevel-based security presets
- State management: Cross-container state sharing and coordination
"""

from .security import (
    TrustLevel,
    AgentSecurityConfig,
    TRUST_PRESETS,
    get_security_config,
    validate_security_config
)

from .container_manager import (
    ContainerManager,
    ContainerConfig,
    ContainerInfo,
    ContainerStatus,
    ExecutionResult
)

from .state_manager import (
    StateManager,
    StateScope,
    AgentState,
    EventType,
    StateEvent
)

__all__ = [
    # Security
    "TrustLevel",
    "AgentSecurityConfig", 
    "TRUST_PRESETS",
    "get_security_config",
    "validate_security_config",
    
    # Container Management
    "ContainerManager",
    "ContainerConfig",
    "ContainerInfo",
    "ContainerStatus",
    "ExecutionResult",
    
    # State Management
    "StateManager",
    "StateScope",
    "AgentState",
    "EventType",
    "StateEvent",
]
