# Persistence Audit

> Generated: 2026-02-03
> Branch: `feature/persistence-audit`
> Purpose: Identify in-memory state that does not survive container restarts, and verify database-backed persistence.

## In-Memory State Found

| File | Pattern | Risk | Recommendation |
|------|---------|------|----------------|
| `services/agent_registry.py` | `AgentRegistry.active_agents: Dict` — tracks all registered agents in memory | **High** — all agent registrations lost on restart; agents must re-register | Acceptable if agents re-register on reconnect (current design). Consider DB-backed registry for faster recovery. |
| `services/agent_registry.py` | `AgentRegistry.agent_websockets: Dict` — maps agent_id → websocket | **Low** — WebSocket connections are inherently ephemeral | No change needed; connections must reconnect anyway. |
| `services/agent_registry.py` | `AgentRegistry.pending_tasks: Dict` — queued task assignments per agent | **High** — pending tasks lost on restart | Move pending task queue to PostgreSQL or Redis. Tasks silently disappear. |
| `services/agent_registry.py` | `heartbeat_monitor_task` — global asyncio task reference | **Low** — recreated on startup | No change needed. |
| `services/agent_factory_service.py` | `AgentFactoryService._instance_cache: Dict` — cached agent instances (5-min TTL) | **Low** — cache, repopulated on demand from DB | No change needed; intentional performance cache. |
| `services/agent_factory_service.py` | `_agent_factory` — module-level singleton | **Low** — recreated on startup | No change needed. |
| `services/agent_mcp_service.py` | `AgentMCPService._tool_cache: Dict` — cached MCP tools (5-min TTL) | **Low** — cache, repopulated on demand | No change needed; intentional performance cache. |
| `services/agent_mcp_service.py` | `_agent_mcp_service` — module-level singleton | **Low** — recreated on startup | No change needed. |
| `services/agent_response_service.py` | `DUAL_WS_MESSAGE_EVENT = True` — feature flag | **Low** — constant, not mutable state | Move to env var or config file for easier toggling. |
| `services/agent_response_service.py` | `AgentResponseService._mention_pattern` — compiled regex | **None** — immutable | No change needed. |
| `services/agent_response_service.py` | `_resolve_agent_id()` hardcoded agent name map | **Medium** — new agents won't be resolved until code changes | Move to DB query (noted as TODO in code). |
| `services/bus_triggers.py` | `BusTriggerService._triggers: Dict` — all trigger rules stored in-memory | **High** — custom triggers lost on restart; only defaults reload | Move to PostgreSQL. Default triggers should seed, custom triggers should persist. |
| `services/catalyst_service.py` | `CatalystService._sessions: Dict` — all creative pipeline sessions | **High** — active sessions and results lost on restart | Move to PostgreSQL. Sessions are valuable intermediate state. |
| `services/comprehension_service.py` | `ComprehensionService._initialized: bool` — init flag | **None** — startup state | No change needed. |
| `services/embedding_service.py` | `EmbeddingService._model_cache: Dict` — model dimension cache | **Low** — small cache, repopulated via test embedding | No change needed. |
| `services/embedding_service.py` | `EmbeddingService.client` — Ollama client reference | **Low** — recreated on initialize() | No change needed. |
| `services/model_router.py` | `ModelRouter._clients: Dict` — provider client instances | **Low** — recreated on demand | No change needed. |
| `services/model_router.py` | `ModelRouter._usage_stats: Dict` — per-model usage tracking | **Medium** — usage analytics lost on restart | Move to PostgreSQL or Redis for durable analytics. |
| `services/model_router.py` | `ModelRouter._total_cost: float` — cumulative cost tracker | **Medium** — cost tracking resets on restart | Persist to DB for accurate billing/reporting. |
| `services/model_router.py` | `ModelRouter.fallback_chain: List` — model fallback order | **Low** — configuration, loaded from defaults | Move to config file or env var. |
| `services/spawn_template_service.py` | `SpawnTemplateService._templates: Dict` — all spawn templates in-memory | **High** — custom templates lost on restart; only builtins reload | Move to PostgreSQL. Builtin templates should seed, custom ones should persist. |
| `services/webhook_service.py` | `WebhookService.webhooks: Dict` — registered webhook endpoints | **High** — all webhook registrations lost on restart | Move to PostgreSQL. Noted as TODO in code. |
| `services/webhook_service.py` | `WebhookService.deliveries: List` — delivery attempt history | **Medium** — delivery audit trail lost on restart | Move to PostgreSQL for compliance/debugging. |
| `services/webhook_service.py` | `WebhookService._delivery_queue: asyncio.Queue` — pending deliveries | **Medium** — in-flight deliveries lost on restart | Consider Redis-backed queue for reliability. |
| `services/task_router.py` | `TaskRouter._routing_analytics: Dict` — routing analytics cache | **Low** — analytics cache, can be rebuilt from DB | No change needed. |
| `services/task_router.py` | `TaskRouter._task_type_performance: Dict` — performance score cache (1hr TTL) | **Low** — cache, repopulated from DB metrics | No change needed. |
| `services/council/voice_registry.py` | `VOICE_REGISTRY: dict` — all voice definitions at module level | **None** — immutable configuration data | No change needed; these are hardcoded definitions. |
| `services/consciousness_council_bridge.py` | `ConsciousnessCouncilBridge._blackboard`, `_context_builder` | **Low** — lazy-initialized references to file-based blackboard | No change needed; re-initializes on demand. |
| `services/event_publisher.py` | `EventPublisher._manager` — reference to ws_manager | **None** — reference to singleton | No change needed. |
| `services/health_aggregator.py` | `_SERVICE_START_TIME: float` — process start timestamp | **None** — inherently per-process | No change needed. |
| `services/memory_service.py` | `MemoryService.store: InMemoryStore` — LangMem coordination layer | **Low** — internal LangMem state; all actual memory ops use pgvector repo | Monitor for correctness; actual data is in PostgreSQL. |
| `services/memory_service.py` | `MemoryService.langmem_manager` — LangMem manager instance | **Low** — recreated on initialize() | No change needed. |
| `controllers/engine.py` | `_active_runs: Dict[str, COREState]` — all active CORE engine runs | **High** — in-flight cognitive runs lost on restart; no way to resume | Consider persisting run state to DB (partially done via run_repository). |
| `controllers/agent_ws.py` | `AgentWebSocketManager.agent_connections: Dict` — agent websockets | **Low** — WebSocket connections are inherently ephemeral | No change needed. |
| `websocket_manager.py` | `ConnectionManager.active_connections: Dict` | **Low** — ephemeral websocket state | No change needed. |
| `websocket_manager.py` | `ConnectionManager.channel_subscribers: Dict` | **Low** — ephemeral subscription state, clients resubscribe on reconnect | No change needed. |
| `core/middleware.py` | `RequestMetrics` (global `_metrics`) — request counters, timing, per-path stats | **Medium** — request analytics lost on restart | Export to Prometheus/StatsD (noted as TODO in code). |

## Database-Backed State (Verified)

| Service | Table(s) | Status |
|---------|----------|--------|
| `memory_service` / `memory_repository` | `memories_semantic`, `memories_episodic`, `memories_procedural` | ✅ Persisted (pgvector embeddings) |
| `agent_repository` | `agents` | ✅ Persisted (agent configs, personalities, capabilities) |
| `communication_repository` | `channels`, `messages` | ✅ Persisted |
| `instance_repository` | `agent_instances`, trust metrics | ✅ Persisted |
| `task_repository` | `core_tasks`, `task_assignments`, `task_results` | ✅ Persisted |
| `council_repository` | `council_sessions`, `council_perspectives` | ✅ Persisted |
| `comprehension_repository` | `comprehension_results`, `comprehension_feedback` | ✅ Persisted |
| `evaluation_repository` | `evaluation_results`, `evaluation_feedback` | ✅ Persisted |
| `bus_repository` | `bus_messages`, `bus_subscriptions`, `bus_delivery_receipts`, `bus_external_agents`, `bus_offline_queue` | ✅ Persisted |
| `mmcnc_repository` | MMCNC hierarchy tables | ✅ Persisted |
| `knowledgebase_repository` | `kb_documents`, `kb_chunks` (with pgvector embeddings) | ✅ Persisted |
| `run_repository` | `core_runs`, `core_run_events` | ✅ Persisted |
| `conversations` (controller/repo) | `conversations` | ✅ Persisted |

## Summary

### Critical In-Memory State (High Risk)

These **will** cause data loss or broken state on container restart:

1. **`bus_triggers._triggers`** — Custom trigger rules vanish. Only defaults reload.
2. **`catalyst_service._sessions`** — Active creative sessions and all results gone.
3. **`spawn_template_service._templates`** — Custom spawn templates vanish.
4. **`webhook_service.webhooks`** — All webhook registrations gone.
5. **`agent_registry.pending_tasks`** — Queued task assignments silently dropped.
6. **`controllers/engine._active_runs`** — In-flight CORE cognitive runs unrecoverable.

### Recommended Migration Priority

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| P1 | Webhook registrations → PostgreSQL | Small | Prevents silent integration breakage |
| P1 | Bus trigger rules → PostgreSQL | Small | Preserves custom automation rules |
| P1 | Catalyst sessions → PostgreSQL | Medium | Preserves creative pipeline state |
| P1 | Spawn templates → PostgreSQL | Small | Preserves custom agent templates |
| P2 | Pending task queue → Redis/PostgreSQL | Medium | Prevents task loss during restarts |
| P2 | Model router usage stats → PostgreSQL | Small | Accurate cost tracking |
| P2 | Webhook deliveries → PostgreSQL | Small | Audit trail for integrations |
| P3 | Request metrics → Prometheus/StatsD | Medium | Operational observability |
| P3 | Agent name resolution → DB query | Small | Dynamic agent @mention support |
