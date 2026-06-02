# Agent Factory & MCP — Current Architecture & Wiring Status

**Last updated:** 2026-06-02

The **current-state** reference for CORE's agent system: how an agent goes from a stored config to a running, MCP-tool-bound LangChain agent, and what's wired vs. stubbed today. For the original design intent see [`agent-factory.md`](agent-factory.md) (2025-10-26 vision) and [`agent-factory-implementation.md`](agent-factory-implementation.md); this doc tracks reality.

## At a glance

```
Agent config (DB: agents)  →  AgentFactoryService.get_agent()  →  AgentInstance (cached, 5-min TTL)
                                      │                                   ▲
                                      ├─ AgentMCPService.get_tools_for_agent()  (MCP tools, cached 5 min)
                                      └─ ChatOpenAI(temp/top_p from traits) + create_react_agent()
Runtime: @mention in Communication Commons → AgentResponseService → factory → agent.ainvoke() → reply
```

## Backend API surface (`controllers/agents.py`, prefix `/agents`)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/agents` | List with filters (type/status/tags/pagination) |
| GET | `/agents/search` | Full-text search (name/description/interests) |
| GET | `/agents/tags` | Distinct tags + usage counts |
| GET | `/agents/{id}` | Get one config |
| GET | `/agents/{id}/tools` | MCP-bound tools for the agent |
| POST | `/agents` | Create (`AgentCreateRequest`) |
| PATCH | `/agents/{id}` | Partial update (clears factory cache) |
| DELETE | `/agents/{id}` | Hard delete |
| POST | `/agents/{id}/activate` · `/deactivate` | Toggle `is_active`/status |
| GET | `/agents/stats/overview` · POST `/agents/cache/clear` | Factory cache stats / clear |

Registered via `app.include_router(agents.router)` in `main.py`.

## Services & responsibilities

- **`AgentFactoryService`** (`services/agent_factory_service.py`) — `get_agent(id)` returns a cached `AgentInstance` (5-min TTL) or builds one: load config → fetch MCP tools → create an LLM whose `temperature`/`top_p` are tuned from `personality_traits` → `create_react_agent(llm, tools)`. Singleton via `get_agent_factory()`.
- **`AgentMCPService`** (`services/agent_mcp_service.py`) — binds MCP tools via `MultiServerMCPClient` (stdio/HTTP/SSE), caches per-server tools 5 min, degrades gracefully when a server/command is unavailable.
- **`AgentRegistry`** (`services/agent_registry.py`) — lifecycle of *containerized* agent instances: register, heartbeat, task assignment/completion/refusal, stale detection (90s unhealthy / 5min lost), background monitor every 30s.
- **`AgentResponseService`** (`services/agent_response_service.py`) — parses `@mentions`, loads the mentioned agents, builds channel context, invokes them in parallel, posts replies back to the Commons.

## Data model

`AgentConfig` (`models/agent_models.py`) → `agents` table (`repository/agent_repository.py`): `agent_id` PK, `agent_name`, `agent_type` (`consciousness_instance|task_agent|system_agent|external_agent`), `system_prompt`, `personality_traits` (JSONB float map), `capabilities`, `interests` (tags), `mcp_servers` (JSONB: `{server_id, tools[], config}`), `custom_tools`, `consciousness_phase`, `is_active`, `current_status`, timestamps, `version`, `author`. Repository supports list/filter, full-text search, tag counts, CRUD, and activate/deactivate.

## MCP layer (`/mcp`)

A standalone FastAPI **MCP registry service** (`mcp/mcp_registry_service.py`) with its own Docker setup:

- **Tables:** `mcp_servers` (id, name, url, `server_type` TOOLS/KNOWLEDGE/COMPUTE/STORAGE/CUSTOM, status, `capabilities` JSON, `config` JSON, health fields) and `user_server_configs` (per-user enable + overrides + usage counts).
- **Routes:** server CRUD (`/servers`…), per-user config (`/users/servers`), health checks (`/servers/{id}/health-check`, `/batch-health-check`).
- **Note:** the registry service is **not yet integrated** with `AgentMCPService` — the factory currently resolves servers from a **hardcoded** list (`mcp-obsidian`, `memory`, `filesystem`) in `_get_server_config` (TODO: load from the registry).

## Frontend (`agents-page/` + services)

| Route | Component | Backend wiring |
|-------|-----------|----------------|
| `/agents` | Agent Builder (5-step wizard) | **Reads** tools/config live; **Deploy is stubbed** — `deployAgent()` logs instead of `POST /agents` |
| `/agents/library` | My Agents | **Real** — `AgentLibraryService` → `GET /agents`, activate/deactivate/delete wired |
| `/agents/marketplace` | Marketplace | **Mocked** — `AgentMarketplaceService` serves 5 hardcoded agents; install is a fake toast (now carries a `.sample-badge`) |

`AgentLibraryService` maps backend `AgentConfig` → UI `LibraryAgent` (hardcoding a few display fields: `category`, `rating`, `downloads`, `size`). `duplicate`/`export`/`favorite` are client-side only. `AgentToolsService` → `GET /agents/{id}/tools` (cached).

## Wired vs. stubbed (today)

✅ **Wired:** agent CRUD + search + tags (API↔DB↔UI library), factory instantiation with trait-tuned LLM + MCP tools, tool discovery endpoint, activate/deactivate, `@mention` → parallel agent invocation → Commons reply, heartbeat/stale monitoring.

⚠️ **Stubbed / partial:**
- Builder **Deploy** — `agent-builder.component.ts` `deployAgent()` doesn't call `POST /agents` (the endpoint exists; just unwired).
- **Marketplace** — fully mocked, no backend marketplace/install routes.
- **MCP registry ↔ factory** — registry service exists but factory uses a hardcoded server list (`agent_mcp_service.py:260` TODO).
- **Personality → params** — only the system prompt is used for behavior despite trait→temp/top_p calc (`agent_factory_service.py:312` TODO); model is hardcoded `gpt-4o-mini` (`:426` TODO, incl. local-provider support).
- **Task reassignment** on refusal — logged only (`agent_registry.py:306` TODO).
- **MCP health checks** — timestamp only (`mcp/mcp_registry_service.py:317` TODO).
- Library **duplicate/export/favorite** — client-side only (not persisted).

See [`../implementation/ui-gaps-audit.md`](../implementation/ui-gaps-audit.md) for the frontend-side status and backlog.
