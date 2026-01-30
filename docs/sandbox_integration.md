# Sandbox Integration Architecture

## Overview

This document describes the sandbox module and its integration with the CORE loop and MCP registry. The sandbox provides secure, isolated execution environments for AI agents.

## Components

### 1. Security Configuration (`app/sandbox/security.py`)

Defines security presets based on trust levels:

```python
TrustLevel.TRUSTED     # Full access - internal/verified agents
TrustLevel.SANDBOXED   # Limited access - user-facing agents
TrustLevel.UNTRUSTED   # Isolated - external/untrusted agents
```

**Security Config Options:**
- `network_enabled` - Allow network access
- `allowed_hosts` - Whitelist of allowed network destinations
- `allow_shell` - Permit shell command execution
- `allow_subprocess` - Allow spawning subprocesses
- `allow_network_tools` - Allow HTTP/API tools
- `memory_limit_mb` - Container memory limit
- `cpu_limit` - CPU quota (1.0 = 1 core)
- `timeout_seconds` - Execution timeout
- `read_only_root` - Make filesystem read-only
- `log_all_commands` - Audit logging

### 2. Container Manager (`app/sandbox/container_manager.py`)

Manages Docker containers for agent execution:

```python
manager = ContainerManager(pool_size=5, container_ttl_minutes=30)
await manager.initialize()

# Get a container for execution
config = ContainerConfig(
    name="agent-task",
    security=TRUST_PRESETS[TrustLevel.SANDBOXED],
    mcp_capabilities=["web_search", "file_read"]
)
container = await manager.get_container(config)

# Execute code
result = await manager.execute_code(container.id, code)

# Release when done
await manager.release_container(container.id, return_to_pool=True)
```

**Features:**
- Container pooling by trust level
- Automatic cleanup of expired containers
- Security configuration enforcement
- Network isolation and filtering
- Resource limits (memory, CPU)
- Tmpfs mounts for writable scratch space

### 3. State Manager (`app/sandbox/state_manager.py`)

Provides state persistence and coordination between containers:

```python
state_manager = StateManager(data_dir="./state_data")
await state_manager.initialize()

# Set state for an agent
await state_manager.set_state(
    agent_id="reasoning-agent",
    task_id="task-123",
    scope=StateScope.TASK,
    data={"step": 1, "result": "..."}
)

# Get state
state = await state_manager.get_state("reasoning-agent", "task-123")

# Save artifacts
artifact_id = await state_manager.save_artifact(
    agent_id="agent",
    task_id="task",
    name="output.json",
    content=b"...",
    content_type="application/json"
)
```

**State Scopes:**
- `TASK` - Single task execution (default TTL)
- `SESSION` - User session lifetime
- `AGENT` - Agent-specific persistent state
- `GLOBAL` - Shared across all agents

**Event System:**
```python
# Subscribe to events
queue = await state_manager.subscribe_events("my-agent")

# Publish events
await state_manager.publish_event(
    EventType.TASK_COMPLETED,
    source_agent="reasoning-agent",
    target_agent="evaluation-agent",
    task_id="task-123",
    payload={"result": "success"}
)
```

### 4. Dynamic Tool Selector (`mcp/dynamic_tool_selector.py`)

Selects appropriate MCP tools based on task requirements:

```python
selector = DynamicToolSelector(registry_url="http://localhost:8000")

# Analyze task and select tools
selection = await selector.select_tools_for_task(
    task_description="Search for Python docs and summarize",
    explicit_capabilities=["web:search"],
    constraints={"max_latency_ms": 5000}
)

print(selection.selected_tools)  # List of ToolDefinition
print(selection.confidence)       # 0.0 to 1.0
print(selection.missing_capabilities)  # Unmet requirements
```

**Capability Inference:**
The selector infers required capabilities from natural language:
- "Read the file" → `file:read`
- "Search the web" → `web:search`
- "Execute Python" → `code:python`

### 5. Agent Orchestrator (`app/core/agent_orchestrator.py`)

Coordinates the CORE loop with sandboxed execution:

```python
orchestrator = AgentOrchestrator(
    container_manager=container_manager,
    state_manager=state_manager,
    tool_selector=tool_selector
)
await orchestrator.initialize()

# Execute the full CORE loop
result = await orchestrator.execute_core_loop(
    user_input="What's the weather in NYC?",
    session_id="session-123",
    max_iterations=5
)
```

**CORE Loop Phases:**
1. **Comprehension** - Understand user intent
2. **Orchestration** - Plan tool usage
3. **Reasoning** - Execute with tools (sandboxed)
4. **Evaluation** - Assess results
5. **Conversation** - Generate response

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent Orchestrator                        │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐│
│  │Comprehen-│  │Orchestr- │  │Reasoning │  │    Evaluation    ││
│  │  sion    │─▶│  ation   │─▶│          │─▶│                  ││
│  └──────────┘  └──────────┘  └────┬─────┘  └────────┬─────────┘│
│                                   │                  │          │
│                                   ▼                  ▼          │
│                           ┌──────────────────────────────┐      │
│                           │      Container Manager       │      │
│                           │                              │      │
│                           │  ┌────────┐  ┌────────┐     │      │
│                           │  │TRUSTED │  │SANDBOX │     │      │
│                           │  │ Pool   │  │ Pool   │     │      │
│                           │  └────────┘  └────────┘     │      │
│                           └──────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MCP Registry Service                        │
│                                                                 │
│  ┌─────────────────────┐      ┌──────────────────────────────┐ │
│  │Dynamic Tool Selector│◀────▶│     MCP Client Manager       │ │
│  │                     │      │                              │ │
│  │ • Capability Match  │      │  ┌────────┐  ┌────────┐     │ │
│  │ • Constraint Check  │      │  │Server 1│  │Server 2│     │ │
│  │ • Priority Ranking  │      │  └────────┘  └────────┘     │ │
│  └─────────────────────┘      └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Security Model

### Trust Levels

| Level     | Network | Shell | Subprocess | Read-Only | Memory | Use Case                |
|-----------|---------|-------|------------|-----------|--------|-------------------------|
| TRUSTED   | ✅       | ✅     | ✅          | ❌         | 2GB    | Internal agents         |
| SANDBOXED | Limited | ❌     | ❌          | ✅         | 512MB  | User-facing agents      |
| UNTRUSTED | ❌       | ❌     | ❌          | ✅         | 256MB  | External/unknown agents |

### Container Isolation

```
UNTRUSTED Container
├── network_mode: none         (no network)
├── read_only: true            (immutable filesystem)
├── security_opt: no-new-privileges
├── mem_limit: 256MB
├── cpu_quota: 0.5
├── capabilities: DROP ALL
└── tmpfs: /tmp, /workspace    (writable scratch only)
```

## Usage Example

```python
import asyncio
from app.sandbox import (
    ContainerManager, StateManager, TrustLevel, TRUST_PRESETS
)
from mcp.dynamic_tool_selector import DynamicToolSelector
from app.core.agent_orchestrator import AgentOrchestrator

async def main():
    # Initialize components
    container_manager = ContainerManager(pool_size=5)
    state_manager = StateManager(data_dir="./agent_state")
    tool_selector = DynamicToolSelector(registry_url="http://localhost:8000")
    
    orchestrator = AgentOrchestrator(
        container_manager=container_manager,
        state_manager=state_manager,
        tool_selector=tool_selector
    )
    
    await orchestrator.initialize()
    
    try:
        # Run the CORE loop
        result = await orchestrator.execute_core_loop(
            user_input="Find the top 3 Python web frameworks",
            session_id="user-session-123"
        )
        
        print(f"Success: {result['success']}")
        print(f"Response: {result['response']}")
        print(f"Tools used: {result.get('tools_used', [])}")
        
    finally:
        await orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

## Testing

Run the test suite:

```bash
# Run sandbox tests
pytest backend/tests/test_sandbox.py -v

# Run orchestrator tests
pytest backend/tests/test_agent_orchestrator.py -v

# Run all tests
pytest backend/tests/ -v
```

## Future Enhancements

1. **Redis Backend** - Replace in-memory state with Redis for distributed coordination
2. **Kubernetes Support** - Add K8s pod-based sandboxing as alternative to Docker
3. **Fine-grained Permissions** - Per-tool permission model
4. **Cost Tracking** - Track and limit tool usage costs
5. **Observability** - OpenTelemetry integration for distributed tracing
