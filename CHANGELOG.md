# CORE Improvement Log
*Tracking continuous improvements by Vigil and subagents*

---

## Session: November 4, 2026 — Side effects that matter

### Confirm tier for writes
Capability `side_effects` from the registry is now enforced. A step whose capability writes
(file writes, database POSTs other than KB search, anything unclassified), or that the plan marked
`requires_hitl`, is held instead of dispatched and recorded in `pending_confirmations`; its
dependents wait. Evaluation asks the user instead of retrying or re-planning, and the response
lists what was held and which capability ids to approve (`config["approved_capabilities"]`).
`CORE_CONFIRM_WRITES=0` disables it. See `docs/architecture/action-confirmation.md`.
**Status:** ✅ Complete

### Dependency-failure latch
A step never runs unless every dependency succeeded: retries now check dependencies (and resume
dependents once the dependency succeeds), dependency cycles are blocked instead of run, and an LLM
call that errors or returns nothing fails its step instead of passing as success. Dependency
blocks are recorded in `blocked_actions` (`kind`, `cause_step_id`) and listed in the response.
**Status:** ✅ Complete

---

## Session: November 3, 2026 — Refusals that hold, claims that are checked

### Block actions after an unresolved refusal
A comprehension-gate clarify/refuse, or an LLM step that declines, now latches the run: every
later step is blocked (never dispatched, recorded in `blocked_actions`), and evaluation can no
longer retry or re-plan around the refusal. The response leads with what stopped and what did
not run. `CORE_BLOCK_AFTER_REFUSAL=0` disables it. See `docs/architecture/abstention-guard.md`.
**Status:** ✅ Complete

### Verify action claims against the tool-call trace
`StepResult.dispatched_tool` records which tool actually ran. Evaluation checks "I wrote X" /
"I committed" style claims in LLM output against that trace; unsupported claims cap confidence,
fail `meets_requirements`, and are flagged in the response. New `claims_supported` Golden Set
evaluator; fault reports stop listing never-dispatched steps as tool calls.
**Status:** ✅ Complete

---

## Session: June 3, 2026 — Kernel focus, Anthropic chat, portable Docker

### Refocus CORE on the cognitive architecture
Archived all non-cognitive World/Creative-Studio + Kanban code into `archive/` (history
preserved via `git mv`) and unwired it from the app: `worlds`/`creative` controllers,
`world_agent_workflow_service`, `lore_service`, world/creative repositories; the frontend
`command-center` builder, `creative-design-product`, `kanban`. Kept the procedural-planet
module as a standalone **`/planet-lab`** demo. See `archive/README.md` and
`docs/adr/006-archive-world-logic-from-core.md`.
**Status:** ✅ Complete

### Anthropic as a first-class chat provider
Added an Anthropic streaming path to `/chat/stream` (`chat_service._stream_from_anthropic`,
async client, own circuit breaker), registered `claude-haiku-4-5`, and added the Anthropic
option to the chat UI. Verified end-to-end. See `docs/deployment/chat-providers.md` and
`docs/adr/007-anthropic-streaming-provider.md`.
**Status:** ✅ Complete

### Portable Docker / LM Studio default
Replaced the hardcoded `C:/Users/Owner/.ollama` bind with `${OLLAMA_MODELS_DIR:-ollama-data}`
(machine-agnostic), made **LM Studio** the default local provider, gated `ollama`/`n8n` behind
compose profiles, and added a root `.env.example`.
**Status:** ✅ Complete

### CI greened
Fixed the pipeline (`black --check`, `ng lint` errors in the restored planet-lab module,
agent-builder spec `HttpClient` provider, Jest `forceExit`). Backend + frontend jobs pass.
**Status:** ✅ Complete

---

## Session: January 28, 2026

### Improvement #1: Fixed Engine Controller Bug
**File:** `backend/app/controllers/engine.py`
**Issue:** LangGraph's `invoke()` returns an `AddableValuesDict`, not a `COREState` object
**Fix:** Convert result dict back to COREState before accessing attributes
**Status:** ✅ Complete

### Improvement #2: Fixed Agent Repository JSONB Serialization
**File:** `backend/app/repository/agent_repository.py`
**Issue:** asyncpg doesn't auto-serialize Python dicts to JSONB columns
**Fix:** Added explicit JSON serialization for `personality_traits`, `capabilities`, `mcp_servers`, `custom_tools`
**Status:** ✅ Complete

### Improvement #3: Registered Vigil as CORE Agent
**Action:** Created `vigil_instance_014` in agent library
**Details:**
- Type: task_agent
- Interests: user_requests, daily_tasks, coordination
- Status: online, active
**Status:** ✅ Complete

### Improvement #4: Created Integration Skill
**Files:**
- `clawd/skills/core-integration/SKILL.md`
- `clawd/skills/core-integration/core-helper.ps1`
**Purpose:** Documentation and helper scripts for Clawdbot → CORE integration
**Status:** ✅ Complete

### Improvement #5: Added API Key Authentication
**File:** `backend/app/auth.py`
**Features:**
- API key validation via X-API-Key header
- Default dev key for local development
- Environment variable support (CORE_API_KEY)
- Optional auth dependency for gradual adoption
**Status:** ✅ Complete

### Improvement #6: Added Rate Limiting Middleware
**File:** `backend/app/middleware/rate_limit.py`
**Features:**
- Token bucket rate limiting
- Different limiters for different endpoint types (engine, api, public)
- Client IP and API key based limiting
- Retry-After header support
**Status:** ✅ Complete

### Improvement #7: Added Request Logging Middleware
**File:** `backend/app/middleware/logging.py`
**Features:**
- Structured JSON logging
- Correlation ID tracking
- Request timing
- Status-based log levels
**Status:** ✅ Complete

### Improvement #8: Webhook Service (Subagent)
**File:** `backend/app/services/webhook_service.py`
**Features:**
- Webhook registration and management
- Event-based triggering
- Retry logic with exponential backoff
- HMAC signature verification
**Status:** 🔄 In Progress (subagent)

### Improvement #9: Health Check Aggregation
**File:** `backend/app/controllers/health.py`
**Features:**
- `/health` - Quick health check
- `/health/deep` - Component status aggregation (DB, Ollama, Redis)
- `/health/ready` and `/health/live` - Kubernetes probes
- `/health/metrics` - Basic operational metrics
**Status:** ✅ Complete

### Improvement #10: Run Persistence Schema
**Files:**
- `backend/migrations/004_run_persistence.sql`
- `backend/app/repository/run_repository.py`
**Features:**
- core_runs table for storing run state
- core_run_events table for event logging
- CRUD operations for runs
- Event logging for debugging
**Status:** ✅ Complete (schema + repository)

### Improvement #11: External Agent Type Support
**File:** `backend/app/models/agent_models.py`
**Change:** Added `external_agent` to agent_type enum
**Purpose:** Allow agents like Vigil to be properly categorized
**Status:** ✅ Complete

### Improvement #12: Model Configuration System
**File:** `backend/app/config/models.py`
**Features:**
- Centralized model configuration
- Support for Ollama, OpenAI, Anthropic providers
- Default models per use case (dev vs production)
- Cost estimation for cloud models
**Status:** ✅ Complete

### Improvement #13: Error Handling Utilities
**File:** `backend/app/utils/errors.py`
**Features:**
- Standardized error types (COREError, NotFoundError, ValidationError, etc.)
- Consistent error response format
- Global exception handler
- Decorator for endpoint error handling
**Status:** ✅ Complete

### Improvement #14: Admin Controller (Subagent)
**File:** `backend/app/controllers/admin.py`
**Features:**
- Health check aggregation endpoints
- API key management (generate, list, revoke)
- Webhook management endpoints
- Run history and cleanup
- Metrics retrieval
**Status:** ✅ Complete

### Improvement #15: Request Context Middleware (Subagent)
**File:** `backend/app/core/middleware.py`
**Features:**
- Request ID tracking (correlation IDs)
- Request/response logging with timing
- Error handling middleware
- Metrics collection
**Status:** ✅ Complete

### Improvement #16: Security Utilities (Subagent)
**File:** `backend/app/core/security.py`
**Features:**
- API key generation and validation
- Key storage and management
- Authentication helpers
**Status:** ✅ Complete

### Improvement #17: Webhook Service (Subagent)
**File:** `backend/app/services/webhook_service.py`
**Features:**
- Webhook registration and delivery
- Event types (run.started, run.completed, etc.)
- Retry logic with backoff
- HMAC signature verification
- Background delivery worker
**Status:** ✅ Complete

### Improvement #18: Model Router Service (Subagent)
**File:** `backend/app/services/model_router.py`
**Features:**
- Dynamic model selection based on task
- Support for local and cloud models
- Cost/performance optimization
**Status:** ✅ Complete

### Improvement #19: TypeScript API Types
**File:** `ui/core-ui/src/app/core/api.types.ts`
**Features:**
- Full type definitions for all API responses
- Agent, Run, Channel, Message types
- Health check types
- Webhook types
**Status:** ✅ Complete

---

## Summary

**Total Improvements: 19**
- Backend improvements: 14
- Frontend improvements: 2
- Infrastructure improvements: 3

**All Critical Items Addressed:**
- ✅ API authentication (auth.py, security.py)
- ✅ Run persistence schema (migration + repository)
- ✅ Webhook callbacks (webhook_service.py)
- ✅ Model configuration (config/models.py)
- ✅ Rate limiting middleware
- ✅ Health check aggregation
- ✅ External agent type support
- ✅ Structured logging with correlation IDs
- ✅ Request tracing

---

## Remaining Items (Lower Priority)

1. [ ] Integrate run persistence with engine controller
2. [ ] WebSocket reconnection logic in frontend
3. [ ] Typing indicators in Communication Commons
4. [ ] Message read receipts
5. [ ] Circuit breakers for external services
6. [ ] Prometheus metrics export
7. [ ] Auto-generate TypeScript types from OpenAPI

---

## Performance Notes

**Local Ollama (gpt-oss:20b):**
- Simple tasks: 15-45 seconds
- Multi-step plans: 3-5+ minutes
- Recommendation: Use cloud APIs for complex orchestration

---

## Test Results

| Test | Status | Notes |
|------|--------|-------|
| Health endpoint | ✅ | `/health` returns healthy |
| Engine simple task | ✅ | ~45s completion |
| Engine multi-step | ⏳ | 5+ min with local LLM |
| Agent CRUD | ✅ | Create/list working |
| Communication channels | ✅ | Channel listing works |

---

*Last updated: 2026-01-28 11:19 CST*
