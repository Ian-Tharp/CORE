# CORE scope reduction: archive plan (dry run)

| | |
|---|---|
| **Status** | Proposal. **Nothing has been moved, deleted or edited.** This file is the only change. |
| **Baseline** | `develop` lineage at commit `4cc9cd9` (`docs: seed cloud wave 1 ...`). All `file:line` references are against this commit; after each batch lands, re-grep, because `main.py` line numbers shift. |
| **Goal** | Reduce CORE to the backend cognitive engine plus a thin conversations UI; archive everything else. |
| **Relation to ADR-006** | ADR-006 archived Worlds/Creative and explicitly listed Council, Catalyst, Consciousness, Commons, Bus, memory/KB and MCP as *kept*. This plan reverses most of that "Kept" list, so it needs an ADR-007 that supersedes it before any batch executes (section 9). |
| **Brief note** | The last acceptance bullet of the task text ended at "A short risks"; it is read here as "a short risks and open-questions section" (sections 7 and 8). |

## 0. Summary

- `backend/app` has 159 Python modules (144 excluding `main.py` and 14 package `__init__`s), 54,491 LOC. 30 controller modules exist, 28 routers are mounted by `main.py`, 2 WebSocket endpoints are declared inline (section 3.1).
- **KEEP: 50 modules.** The C/O/R/E pipeline, chat, conversations, the knowledge base used by chat, run persistence and webhooks, health, auth/middleware/config. **ARCHIVE: 67 modules / 28,181 LOC** in nine batches (B1-B9), taking 39 test files and about 1,040 test cases with them. **UNSURE: 27 modules** (26 in three conditional batches C1-C3 holding 10,907 LOC and about 647 cases, plus `utils/errors.py`), each needing a product decision (section 8).
- **The brief's named root is wrong.** `backend/app/core/langgraph/core_graph.py` is a 56-line placeholder with **zero importers**; so is the second copy `app/langgraph/core_graph.py`. The live pipeline root is `core_graph_v2.py`, imported only by `controllers/engine.py:29`. Reachability below is computed from `core_graph_v2.py` and the chat/conversations controllers. Both dead copies are the first thing archived (B1).
- **Key finding:** the pipeline closure is tiny (24 modules including packages) and chat plus conversations add about 30 more. Every ARCHIVE and UNSURE module is unreachable from both, with exactly **five** apparent exceptions that are all coupling accidents, not real dependencies: `repository/__init__.py:7-13` eagerly imports `council_repository`, `memory_repository`, `model_metrics_repository` (which drags in `council_models`), and `knowledgebase_service.py:918,1005` lazily imports `instance_repository` (section 4).
- **Ordering was verified by simulation.** For each batch in order I removed its modules from the import graph and listed every remaining production importer. After the final ordering each batch leaves **zero** unexpected dangling importers; the only remaining touch points are the listed edits to `main.py`, `repository/__init__.py`, `health_aggregator.py`, `core/health.py` and `knowledgebase_service.py`. The first draft failed that check three times (`task_models`, `agent_response_service`, `websocket_manager`), which is why `task_models` sits in C1 and `agent_response_service` in B7.
- **"Independently revertable" has a precise meaning here.** Each batch is one `git mv`+edit commit. Reverting the *latest* un-reverted batch always works. Out of order, B2, B6, B8, B9, C2 and C3 can be reverted alone; B1, B3, B4, B5, B7 and C1 need later batches restored first (listed per batch as "Revert prerequisites").
- **Do not execute B7, B8, B9 or C3 until the UI is cut over.** The Angular app still calls `/communication` (+`/ws/{id}`), `/discord`, `/agents`, `/instances`, `/spawn-templates`, `/mcp` and `/system`. This plan does not touch `ui/` as instructed; those pages break at runtime when the endpoints disappear (section 7, risk 2).

| Batch | Scope | Modules | LOC | Test files moved | Cases removed | Route delta |
|---|---|---:|---:|---:|---:|---:|
| B1 | Dead code and dev-only wiring | 9 | 1,193 | 1 | 19 | -1 |
| B2 | MMCNC hierarchy | 3 | 1,809 | 1 | 28 | -22 |
| B3 | Task Routing Engine | 3 | 1,901 | 1 | 25 | -14 |
| B4 | Inter-agent Bus (+ triggers) | 6 | 2,049 | 4 | 154 | -11 |
| B5 | Catalyst + Council of Perspectives | 9 | 4,371 | 5 | 135 | -16 |
| B6 | Consciousness module, backup and lineage | 10 | 3,521 | 6 | 82 | -19 |
| B7 | Communication Commons + Discord bridge | 8 | 3,780 | 8 | 155 | -24 |
| B8 | Agent Factory + Spawn Templates | 8 | 3,927 | 5 | 171 | -18 |
| B9 | Instances, agent registry, sandbox, agent WebSocket | 11 | 5,630 | 8 | 271 | -13 |
| C1 | Comprehension + Evaluation engines and the eval harness (UNSURE) | 13 | 5,855 | 14 | 432 | -14 |
| C2 | LangMem memory (UNSURE) | 4 | 2,350 | 2 | 53 | -15 |
| C3 | Admin / audit / model router / MCP / system monitor (UNSURE) | 9 | 2,702 | 8 | 162 | -30 |

Cases are a **static** count of `def test_*` (methods included, `parametrize` not expanded), because the dependencies (`fastapi`, `langgraph`, `pytest`, `uv`) are not installed in the environment this plan was produced in. The baseline `pytest` run has therefore **not been executed**; step 0 of the execution protocol (section 9) is to run it and record the real figures.

## 1. Method, evidence conventions and limits

**Import graph.** AST walk over every `backend/app/**/*.py` (159 modules), including imports inside function bodies (marked *lazy*), `from pkg import submodule` forms, relative imports, and the implicit execution of every parent package's `__init__` (importing `app.repository.run_repository` runs `app/repository/__init__.py`). The 14 `__init__` files are folded into their packages and not listed as rows.

**Roots.**
- *Pipeline (P):* `app.core.langgraph.core_graph_v2` plus `app.controllers.engine`.
- *Chat (C):* `app.controllers.chat` plus `app.controllers.conversations`.
- *KEEP closure:* P, C, `core_entry`, `health`, `knowledgebase`, `local_llm`, `metrics`, and the infrastructure `main.py` wires unconditionally (`dependencies`, `auth`, `core.security`, `core.middleware`, `middleware.correlation`, `core.logging_config`, `config.startup_validator`, `migrations.runner`).
- The *Reach* column in Appendix A is `P`, `C`, `PC` or `–`.

**Evidence.** "Imported by" lists up to three importers as `path:line` plus a count. "Tests" counts the files in `tests/` (and cases in them) that directly import the module or reference it in a `patch("app....")` string. Counts are direct, so a test that exercises a module only through another module is not counted; the per-batch test lists are authoritative for what must move.

**Route counts** are decorator counts (`@router.get|post|put|patch|delete|websocket`, `@app.websocket`). `len(app.routes)` before and after should differ by exactly the "route delta"; if they do not, a decorator form I did not match is in play.

**Known blind spots.** (a) No `importlib` / `getattr`-based loading exists in `backend/app` (`grep` found only `__import__("os")` in `agent_mcp_service.py:271`), so the graph is complete for Python. (b) SQL-only coupling is invisible to imports: the `check_bus_queue` and `check_task_queue` health probes query `bus_*` and `tasks` tables directly (section 4.3). (c) Anything outside this repository (GPW/PWE, n8n workflows, deployed OpenClaw agent containers) cannot be seen from here (risk 1). (d) Docs, UI and compose files were checked by `grep`, not by graph.

**Reproducing.** The analysis ran from scratch scripts that are deliberately **not** committed (scope: this file only). Re-create it in about 80 lines: `ast.walk` each module, resolve `app.*` names to the longest existing module prefix, add parent `__init__` edges, then BFS from the roots above. If the team wants it kept, the right home is `scripts/archive_graph.py` with a pytest asserting a three-module fixture package; say so and I will add it as a separate change.

## 2. Target shape: what stays

| Area | Why it is KEEP (evidence) |
|---|---|
| C/O/R/E pipeline: `core/langgraph/core_graph_v2.py`, `core/agents/*`, `core/tools/dispatcher.py`, `models/core_state.py`, `core/intelligence_log.py`, `core/telemetry.py` | `engine.py:29` -> `core_graph_v2`; `core_graph_v2.py:24-29` -> four agents, telemetry, intelligence log. Self-contained: closure is only `auth`, `config.models`, `dependencies`, `models.*`, `utils.json_repair` (24 modules). Covered by 5 test files that import `core_graph_v2` directly. |
| `/engine/*` and `/core/*` | UI `engine.service.ts` (playground on the conversations page) calls both. |
| `/chat/stream`, `/conversations/*`, title generation, conversation persistence | UI `chat-service.ts:104`, `conversations.service.ts:26,31`. |
| Knowledge base (`knowledgebase` controller/service/repo, `ollama_embeddings`, `metrics_service`) | `chat.py:17` imports `knowledgebase_service`; UI chat window lists KB files (`chat-window.component.ts:10`). |
| `/local-llm/*` | UI `engine-playground.component.ts:109`, `chat-window.component.ts:178`. |
| Run persistence, webhooks, API-key store | `engine.py:32-33`; `main.py:44`; key loading in `core/security.py`. |
| Health (`health`, `health_aggregator`, `health_repository`), `/metrics` | `main.py:246,275`; Docker healthcheck in `docker-compose.yml` hits `/health`. |
| Auth, security, middleware, config, migrations | wired unconditionally by `main.py`. |

**Thin-UI contract** (read-only inspection of `ui/core-ui/src/app/services/config/app-config.service.ts` and callers): `/chat/stream`, `/conversations/`, `/core/*`, `/engine/*`, `/knowledgebase/*`, `/local-llm/{models,health}`, plus `/health`. Anything in the archive list that the UI calls beyond this set is listed in section 7.

## 3. Inventory

### 3.1 Routers and endpoints registered in `backend/app/main.py`

30 controller modules exist; `main.py` imports 29 in the block at `:8-38`, plus `agent_ws` at `:39`, and calls `include_router` 28 times (`:236-275`). Two controllers are special: `agent_ws.py` is a WebSocket handler with no router, and `bus_triggers.py` is imported (`:28`) but **no `include_router` exists for it**, so its five routes are unreachable today (it is not mounted by `bus.router` either; checked). Prefixes and route counts are read from the files.

| Controller | main.py import / mount | Prefix | Routes | UI callers | Class | Batch | Tests importing the controller (files / cases) |
|---|---|---|---:|---|---|---|---:|
| `admin.py` (514 LOC) | `:20` / `:247` | `/admin` | 19 | 0 | UNSURE | C3 | 2 / 53 |
| `agent_ws.py` (438 LOC, WebSocket handler, no router) | `:39` / `@app.websocket` at `:387` | `(none)` | 0 | 0 | ARCHIVE | B9 | 1 / 22 |
| `agents.py` (665 LOC) | `:16` / `:243` | `/agents` | 12 | 3 | ARCHIVE | B8 | 0 / 0 |
| `audit.py` (106 LOC) | `:36` / `:272` | `/audit` | 3 | 0 | UNSURE | C3 | 0 / 0 |
| `bus.py` (243 LOC) | `:27` / `:254` | `/bus` | 11 | 0 | ARCHIVE | B4 | 0 / 0 |
| `bus_triggers.py` (138 LOC) | `:28` / **never mounted** | `/bus/triggers` | 5 | 0 | ARCHIVE | B4 | 0 / 0 |
| `catalyst_creativity.py` (239 LOC) | `:30` / `:256` | `/catalyst-creativity` | 5 | 0 | ARCHIVE | B5 | 0 / 0 |
| `chat.py` (278 LOC) | `:9` / `:237` | `(none)` | 1 | 1 (`chat-service.ts`) | KEEP | – | 2 / 58 |
| `communication.py` (321 LOC) | `:15` / `:242` | `/communication` | 10 | 3 + WS | ARCHIVE | B7 | 3 / 61 |
| `comprehension.py` (438 LOC) | `:25` / `:252` | `/comprehension` | 8 | 0 | UNSURE | C1 | 0 / 0 |
| `consciousness_backup_controller.py` (149 LOC) | `:34` / `:266` | `/api/consciousness-backup` | 6 | 0 | ARCHIVE | B6 | 0 / 0 |
| `consciousness_lineage_controller.py` (513 LOC) | `:35` / `:269` | `/api/consciousness-lineage` | 13 | 0 | ARCHIVE | B6 | 0 / 0 |
| `conversations.py` (103 LOC) | `:11` / `:238` | `/conversations` | 4 | 1 | KEEP | – | 1 / 11 |
| `core_entry.py` (364 LOC) | `:10` / `:236` | `/core` | 9 | 1 (`engine.service.ts`) | KEEP | – | 0 / 0 |
| `council.py` (771 LOC) | `:21` / `:248` | `/council` | 11 | 0 | ARCHIVE | B5 | 0 / 0 |
| `discord.py` (469 LOC) | `:32` / `:262` | `/discord` | 13 | 1 | ARCHIVE | B7 | 2 / 25 |
| `engine.py` (829 LOC) | `:17` / `:244` | `/engine` | 7 | 1 (`engine.service.ts`) | KEEP | – | 2 / 54 |
| `evaluation.py` (280 LOC) | `:26` / `:253` | `/evaluation` | 6 | 0 | UNSURE | C1 | 0 / 0 |
| `health.py` (264 LOC) | `:19` / `:246` | `/health` | 9 | 1 (`instance.service.ts`) | KEEP | – | 2 / 50 |
| `instances.py` (878 LOC) | `:22` / `:249` | `/instances` | 12 | 1 | ARCHIVE | B9 | 0 / 0 |
| `knowledgebase.py` (693 LOC) | `:13` / `:240` | `/knowledgebase` | 17 | 3+ | KEEP | – | 0 / 0 |
| `local_llm.py` (132 LOC) | `:14` / `:241` | `/local-llm` | 4 | 3 | KEEP | – | 0 / 0 |
| `mcp.py` (249 LOC) | `:33` / `:263` | `/mcp` | 5 | 1 | UNSURE | C3 | 0 / 0 |
| `memory.py` (641 LOC) | `:24` / `:251` | `/memory` | 15 | 0 | UNSURE | C2 | 0 / 0 |
| `metrics.py` (32 LOC) | `:37` / `:275` | `(none)` | 1 | 0 | KEEP | – | 1 / 15 |
| `mmcnc.py` (643 LOC) | `:29` / `:255` | `/mmcnc` | 22 | 0 | ARCHIVE | B2 | 0 / 0 |
| `spawn_templates.py` (151 LOC) | `:31` / `:259` | `/spawn-templates` | 6 | 1 | ARCHIVE | B8 | 0 / 0 |
| `system_monitor.py` (126 LOC) | `:12` / `:239` | `/system` | 3 | 2 | UNSURE | C3 | 0 / 0 |
| `tasks.py` (651 LOC) | `:23` / `:250` | `/tasks` | 14 | 0 | ARCHIVE | B3 | 0 / 0 |
| `test_core.py` (66 LOC) | `:18` / `:245` | `/test` | 1 | 0 | ARCHIVE | B1 | 0 / 0 |


Inline WebSocket endpoints in `main.py`: `/ws/{instance_id}` (`:309-381`, Communication Commons -> B7) and `/ws/agent/{agent_id}` (`:384-402`, containerised agents -> B9).

Lifespan wiring in `main.py` that each batch must unwind: DB `ensure_*_tables` calls `:111-149` (council, instance, task, memory, comprehension, evaluation, bus, mmcnc, discord, model_metrics), webhook service `:156-160` (KEEP), agent registry `:162-167`/`:214-219` (B9), memory service `:169-174` (C2), Discord bridge `:176-205` (B7).

### 3.2 Subsystems

Tests are files in `tests/` that directly import (or `patch` by string) any module of the group, so one file can appear in several rows.

| Subsystem | Class | Batch | Modules | LOC | Evidence (who imports it) | Tests (files / cases) |
|---|---|---|---:|---:|---|---:|
| CORE pipeline (graph v2, 4 agents, tool dispatcher, state models) | KEEP | – | 9 | 2,771 | `controllers/engine.py:29` imports `core_graph_v2`; `core_graph_v2.py:24-29` imports the agents, telemetry, intelligence_log | 14 / 328 |
| Chat + conversations (stream, titles, persistence) | KEEP | – | 6 | 1,297 | `main.py:237-238`; `chat.py:17` -> `knowledgebase_service`; `chat_service.py:27` -> `circuit_breaker` | 6 / 121 |
| Core entry (`/core/*` step API) + LLM provider | KEEP | – | 2 | 438 | `main.py:236`; `core_entry.py:31` -> `llm_provider`; UI `engine.service.ts` | 0 / 0 |
| Knowledge base (RAG for chat, attribution) | KEEP | – | 5 | 2,777 | `chat.py:17`; `knowledgebase_service.py:16` -> `ollama_embeddings`; UI `chat-window.component.ts:10` | 6 / 104 |
| Run persistence + webhooks | KEEP | – | 4 | 1,445 | `engine.py:32-33`; `main.py:44`; `core/security.py` loads keys via `api_key_repository` | 3 / 119 |
| Health stack + local-LLM probe + metrics | KEEP | – | 5 | 1,352 | `main.py:246,241,275`; UI `local-llm.service.ts`, `chat-window.component.ts:178` | 4 / 91 |
| Auth, security, middleware, config, migrations | KEEP | – | 9 | 2,970 | `main.py:40,42,43,72-74,299` | 16 / 377 |
| MMCNC hierarchy | ARCHIVE | B2 | 3 | 1,809 | controller is sole importer of repo/models; no UI caller | 1 / 28 |
| Task router | ARCHIVE | B3 | 3 | 1,901 | `tasks.py:38` -> `bus_service`; `task_router.py:41,48` -> instance repo/registry; no UI caller | 1 / 25 |
| Inter-agent bus + triggers | ARCHIVE | B4 | 6 | 2,049 | importers: tasks (B3), own controllers; `bus_triggers` controller never mounted; no UI caller | 4 / 154 |
| Catalyst engine | ARCHIVE | B5 | 3 | 751 | importers: own controller, `bus_triggers.py:333`; no UI caller | 1 / 30 |
| Council of Perspectives | ARCHIVE | B5 | 6 | 3,592 | importers: own controller, `bus_triggers.py:303`; `repository/__init__.py:7` eager-imports the repo; no UI caller | 4 / 105 |
| Consciousness module + backup + lineage + council bridge | ARCHIVE | B6 | 10 | 3,513 | bridge imported only by `deliberation_service.py:36`; controllers only by `main.py`; no UI caller | 4 / 77 |
| Communication Commons | ARCHIVE | B7 | 4 | 1,939 | `main.py:15,242`; cycle with discord/agent_response; UI `communication/services/*.ts` (3) + `/ws/{id}` | 5 / 86 |
| Discord bridge | ARCHIVE | B7 | 4 | 1,841 | `main.py:32,50-54,176-205`; lazy importers `communication_service.py:239`, `agent_response_service.py:578`; UI `discord-bridge.service.ts` | 6 / 99 |
| Agent Factory (+ agent MCP binding) | ARCHIVE | B8 | 5 | 2,975 | `agent_factory_service` imported by `controllers/agents.py` and `agent_response_service.py`; UI `agent-library.service.ts`, `agent-tools.service.ts` | 4 / 116 |
| Spawn templates | ARCHIVE | B8 | 3 | 952 | own controller only; UI `spawn-templates.service.ts` | 2 / 82 |
| Instances, agent registry, agent WebSocket, event publisher | ARCHIVE | B9 | 8 | 4,260 | `main.py:22,39,45,249,387`; importers `task_router`, `bus_service:44`, `deliberation_service:41`; UI `instance.service.ts` | 11 / 253 |
| Sandbox | ARCHIVE | B9 | 3 | 1,323 | imported only by `sandbox/__init__.py` and `core/agent_orchestrator.py:18` (dead) | 2 / 99 |
| Dead pipeline copies, prompts, orchestrator, test route | ARCHIVE | B1 | 9 | 1,193 | no importer other than `main.py:18` for `test_core` | 1 / 19 |
| Comprehension engine (service/repo/controller) | UNSURE | C1 | 4 | 2,589 | only own controller; pipeline agent does not use it; UI calls `/core/comprehension`, not `/comprehension` | 4 / 143 |
| Evaluation engine + output validator + `eval/` harness | UNSURE | C1 | 8 | 3,024 | `eval/runner.py` imports `evaluation_service`; pipeline agent does not | 9 / 277 |
| LangMem memory | UNSURE | C2 | 4 | 2,350 | `main.py:49,169`; also imported by C1 services; no UI caller | 2 / 53 |
| Admin + audit + model router + core.health | UNSURE | C3 | 7 | 2,327 | `admin.py:33-37`; `model_router` also used by council/catalyst; no UI caller | 9 / 189 |
| MCP registry controller + system monitor | UNSURE | C3 | 2 | 375 | UI `mcp-registry.component.ts:41`, `system-monitor.service.ts`, `dashboard.store.ts:123` | 0 / 0 |

### 3.3 Reachability summary

- **Pipeline closure (24 including packages):** `auth`, `config`, `config.models`, `core.agents.{comprehension,evaluation,orchestration,reasoning}_agent`, `core.intelligence_log`, `core.langgraph.core_graph_v2`, `core.telemetry`, `core.tools(.dispatcher)`, `dependencies`, `models` + `models.{comprehension_output,comprehension_state,core_state,evaluation_output,orchestration_output,reasoning_output,user_input}`, `utils(.json_repair)`, `app`.
- **Chat closure (30):** the above plus `services.chat_service`, `services.title_generation_service`, `services.knowledgebase_service`, `services.metrics_service`, `services.ollama_embeddings`, `core.circuit_breaker`, `repository.{conversation,knowledgebase,run,api_key,model_metrics,council,memory,instance}_repository`, `models.council_models`. The last four repository/model entries are the coupling accidents of section 4.
- **No council, catalyst, consciousness, bus, mmcnc, task, agent-factory, commons, discord, sandbox or instance module appears in either closure** except through those accidents. `chat_service.py`, `title_generation_service.py`, `chat.py`, `conversations.py`, `engine.py`, `core_entry.py`, `health.py` contain no reference to any of them (grep for `council|catalyst|conscious|commons|mmcnc|spawn|discord|agent_factory|agent_registry|bus_service|instance_manager|sandbox` finds one comment in `engine.py:10`).

## 4. Hidden couplings (why "just delete it" fails)

### 4.1 `repository/__init__.py` eagerly imports five repositories
`app/repository/__init__.py:7-13` does `from app.repository import (run_repository, council_repository, memory_repository, api_key_repository, model_metrics_repository)`. Python runs this on **any** `app.repository.*` import, so the kept chat, engine and conversations paths all load `council_repository`, `memory_repository` and `model_metrics_repository` (and through the first, `models.council_models`). Archiving any of the three without editing `__init__.py` makes the whole backend fail to import. Edits are in B5, C2, C3.

### 4.2 Knowledge-base attribution looks up agent instances
`services/knowledgebase_service.py:918` and `:1005` lazily import `instance_repository` to check that a chunk's `source_instance_id` still exists. This is the only KEEP -> ARCHIVE runtime edge. Test `tests/test_knowledge_attribution_verification.py:49-60,97,103,193` builds `AgentInstance` objects. Decision D2.

### 4.3 Health probes assume archived subsystems exist
`services/health_aggregator.py` runs `check_websocket_manager` (`:283-312`, imports `websocket_manager` lazily at `:287`), `check_bus_queue` (`:349-400`, `SELECT ... FROM bus_messages`), and `check_task_queue` (`:403-478`, `SELECT ... FROM tasks`), registered at `:573-576`. `core/health.py:192-215` has a second websocket probe. The SQL probes are invisible to the import graph; if left in place after B3/B4 the `/health` payload degrades. `tests/test_health_aggregator.py` patches all three by name (`:503-515`, `:561-573`).

### 4.4 `setup_db_schema` still creates world and agent tables
`dependencies.py:313-512` creates `worlds`, `world_snapshots`, `wiki_pages`, `characters`, `world_metadata`, `world_gen_params`, `world_assets` (ADR-006 left these on purpose); `:730` creates `agents` and `:796` `world_lore_agents`; `_seed_initial_agents` (`:812`) inserts consciousness-instance personas into an empty `agents` table on every fresh boot. `kb_documents.world_id` (index at `:585`) references worlds. None of this is addressed by B1-B9 (section 6.1).

### 4.5 Import cycles that dictate batch boundaries
`communication_service` <-> `discord_bridge` <-> `agent_response_service` (all lazy; lines in B7) form one cycle, and `event_publisher` -> `websocket_manager` ties instance management to the WebSocket layer. These are why Commons+Discord is one batch and why `websocket_manager` waits for B9.

### 4.6 Other oddities found
- `app/controllers/test_core.py` matches pytest's `test_*.py` pattern and `testpaths = ["tests", "app"]` includes `app/`, so pytest collects a controller as a test module (no real tests). Gone in B1.
- Repo-root scripts `backend/test_*.py` are not collected by CI (`testpaths`); one of them (`test_consciousness_lineage.py`) imports archived modules.
- `fastmcp`, `langchain`, `langchain-community` and `langchain-postgres` in `pyproject.toml` have **zero** imports in `backend/app` today (unused already; out of scope).
- `evaluation_service.py:36-37` imports `memory_repository`, and `eval/runner.py` imports `evaluation_service`: the "evaluation" stack and the "memory" stack are entangled, which fixes the C1 -> C2 order.

## 5. Archive batches

Rules for every batch:
1. One branch from `develop` (`chore/archive-bN-...`), one commit, Conventional Commit `refactor: archive <subsystem> (ADR-007 B<N>)`.
2. `git mv` only (history preserved, as in ADR-006). Destination is `archive/backend/<path under app/>`, tests go to `archive/backend/tests/`. `archive/` is outside `testpaths` and outside CI's `black --check app/ tests/`.
3. In the same commit: remove the wiring listed under "Edits to kept code", update `archive/README.md` with a table row per path.
4. Dependency removals from `pyproject.toml`/`uv.lock` are separate follow-up commits (lock churn would make the batch commit noisy to revert).
5. Database tables are **never** dropped by a batch (ADR-006 precedent, section 6.1).

Batch order was derived from the import graph (dependents before dependencies) and validated by simulation. Firm batches B1-B9 can all be executed once the preconditions in section 9 hold; C1-C3 wait for the decisions in section 8.

### B1. Dead code and dev-only wiring

**Size:** 9 modules / 1,193 LOC / 1 test file(s) moved (19 cases removed, static count) / route delta -1.

**Why here:** Zero production importers, so removal cannot break anything. `core/langgraph/core_graph.py` is a 56-line placeholder that no module imports (AST import graph, Appendix A) and is not even valid LangGraph (`StateGraph(nodes=..., transitions=...)`). The live pipeline is `core_graph_v2.py`, imported only by `controllers/engine.py:29`. `app/langgraph/*` is a second dead copy. The `/test/core-state` route is a dev probe with no UI caller.

**Paths** (`git mv` to `archive/backend/<path under app/>`, matching the ADR-006 precedent, e.g. `archive/backend/controllers/worlds.py`):

- `backend/app/controllers/test_core.py`
- `backend/app/core/agent_orchestrator.py`
- `backend/app/core/langgraph/core_graph.py`
- `backend/app/langgraph/comprehension_graph.py`
- `backend/app/langgraph/core_graph.py`
- `backend/app/prompts/comprehension_agent_prompt.py`
- `backend/app/prompts/evaluation_agent_prompt.py`
- `backend/app/prompts/orchestration_agent_prompt.py`
- `backend/app/prompts/reasoning_agent_prompt.py`

**Tests to move** (to `archive/backend/tests/`):

- `tests/test_agent_orchestrator.py` (19)

**Edits to kept code:**

- `main.py:18` (controller import), `main.py:245` (`include_router(test_core.router)`).

**Migrations / config / compose:**

- DB tables created by the archived code (left in place, see 6.1): none
- No env, compose or CI changes.

**Revert prerequisites** (batches that must still be restored first for a stand-alone `git revert`): B9 (only for its test: `agent_orchestrator.py:18` imports `sandbox`; nothing imports the orchestrator at runtime).

**Verify** (run from repo root; record the "before" route count on `develop` first):

```bash
git grep -nE "\b(app\.controllers\.test_core|app\.core\.agent_orchestrator|app\.core\.langgraph\.core_graph|app\.langgraph\.comprehension_graph|app\.langgraph\.core_graph|app\.prompts\.comprehension_agent_prompt|app\.prompts\.evaluation_agent_prompt|app\.prompts\.orchestration_agent_prompt|app\.prompts\.reasoning_agent_prompt)\b|\b(test_core)\.router\b" -- backend/app backend/tests   # expect: no output
cd backend
python -c "import app.main as m; print(len(m.app.routes))"        # expect: before - 1
uv run black --check app/ tests/
python -m pytest tests/ -q                                         # expect: green; cases drop by ~19 (static; parametrised cases may add more)
```

### B2. MMCNC hierarchy

**Size:** 3 modules / 1,809 LOC / 1 test file(s) moved (28 cases removed, static count) / route delta -22.

**Why here:** Pure leaf: `controllers/mmcnc.py` is the only importer of `mmcnc_repository` and `mmcnc_models` (importer map in Appendix A), nothing else references the controller, and no UI file calls `/mmcnc`.

**Paths** (`git mv` to `archive/backend/<path under app/>`, matching the ADR-006 precedent, e.g. `archive/backend/controllers/worlds.py`):

- `backend/app/controllers/mmcnc.py`
- `backend/app/models/mmcnc_models.py`
- `backend/app/repository/mmcnc_repository.py`

**Tests to move** (to `archive/backend/tests/`):

- `tests/test_mmcnc_models.py` (28)

**Edits to kept code:**

- `main.py:29` (controller import), `:64` (`mmcnc_repository` import), `:139-141` (`ensure_mmcnc_tables`), `:255` (`include_router`).

**Migrations / config / compose:**

- DB tables created by the archived code (left in place, see 6.1): `macrocosms`, `microcosms`, `clusters`, `creative_nodes` (created by `mmcnc_repository.ensure_mmcnc_tables`)
- No env, compose or CI changes.

**Revert prerequisites** (batches that must still be restored first for a stand-alone `git revert`): none.

**Verify** (run from repo root; record the "before" route count on `develop` first):

```bash
git grep -nE "\b(app\.controllers\.mmcnc|app\.models\.mmcnc_models|app\.repository\.mmcnc_repository)\b|\b(mmcnc)\.router\b" -- backend/app backend/tests   # expect: no output
cd backend
python -c "import app.main as m; print(len(m.app.routes))"        # expect: before - 22
uv run black --check app/ tests/
python -m pytest tests/ -q                                         # expect: green; cases drop by ~28 (static; parametrised cases may add more)
```

### B3. Task Routing Engine

**Size:** 3 modules / 1,901 LOC / 1 test file(s) moved (25 cases removed, static count) / route delta -14.

**Why here:** Task router sits on top of the bus and the agent registry (`controllers/tasks.py:38-39` imports `bus_service`/`bus_models`, `services/task_router.py:41,48` imports `instance_repository`/`agent_registry`), so it must go before them. `models/task_models.py` is deliberately NOT in this batch: `services/comprehension_service.py:43` also imports it (see C1).

**Paths** (`git mv` to `archive/backend/<path under app/>`, matching the ADR-006 precedent, e.g. `archive/backend/controllers/worlds.py`):

- `backend/app/controllers/tasks.py`
- `backend/app/repository/task_repository.py`
- `backend/app/services/task_router.py`

**Tests to move** (to `archive/backend/tests/`):

- `tests/test_task_router.py` (25)

**Tests to edit in place:**

- `tests/test_health_aggregator.py`: drop the two `check_task_queue` patch entries (L515, L573) – 0 cases removed

**Edits to kept code:**

- `main.py:23`, `:59` (`task_repository`), `:119-121` (`ensure_task_tables`), `:250`.
- `services/health_aggregator.py:403-478` (`check_task_queue`, raw SQL on `tasks`) and its call at `:576`. This is a **SQL-only** coupling the import graph cannot see; leaving it makes `/health` report `tasks` as failing.
- `tests/test_health_aggregator.py:515` and `:573` (`patch("...check_task_queue")` raises `AttributeError` once the function is gone).

**Migrations / config / compose:**

- DB tables created by the archived code (left in place, see 6.1): `tasks`, `task_assignments`, `task_results`
- No env, compose or CI changes.

**Revert prerequisites** (batches that must still be restored first for a stand-alone `git revert`): B4, B9, C1 (`tasks.py:16,38-39`, `task_router.py:22,41,48`).

**Verify** (run from repo root; record the "before" route count on `develop` first):

```bash
git grep -nE "\b(app\.controllers\.tasks|app\.repository\.task_repository|app\.services\.task_router)\b|\b(tasks)\.router\b" -- backend/app backend/tests   # expect: no output
cd backend
python -c "import app.main as m; print(len(m.app.routes))"        # expect: before - 14
uv run black --check app/ tests/
python -m pytest tests/ -q                                         # expect: green; cases drop by ~25 (static; parametrised cases may add more)
```

### B4. Inter-agent Bus (+ triggers)

**Size:** 6 modules / 2,049 LOC / 4 test file(s) moved (154 cases removed, static count) / route delta -11.

**Why here:** Bus is imported by tasks (B3, already gone) and itself imports `controllers.agent_ws` (`services/bus_service.py:44`) and, via triggers, council/catalyst services (`services/bus_triggers.py:303,333`). So it must precede B5 and B9. `controllers/bus_triggers.py` is imported at `main.py:28` but **never mounted**, so its 5 routes are already unreachable.

**Paths** (`git mv` to `archive/backend/<path under app/>`, matching the ADR-006 precedent, e.g. `archive/backend/controllers/worlds.py`):

- `backend/app/controllers/bus.py`
- `backend/app/controllers/bus_triggers.py`
- `backend/app/models/bus_models.py`
- `backend/app/repository/bus_repository.py`
- `backend/app/services/bus_service.py`
- `backend/app/services/bus_triggers.py`

**Tests to move** (to `archive/backend/tests/`):

- `tests/test_bus.py` (42)
- `tests/test_bus_models.py` (39)
- `tests/test_bus_service.py` (50)
- `tests/test_bus_triggers.py` (23)

**Tests to edit in place:**

- `tests/test_health_aggregator.py`: drop the two `check_bus_queue` patch entries (L511, L569) – 0 cases removed

**Edits to kept code:**

- `main.py:27`, `:28` (`bus_triggers` import only; it was never mounted), `:63` (`bus_repository`), `:135-137` (`ensure_bus_tables`), `:254`.
- `services/health_aggregator.py:349-400` (`check_bus_queue`) and call at `:575`; `tests/test_health_aggregator.py:511`, `:569`.

**Migrations / config / compose:**

- DB tables created by the archived code (left in place, see 6.1): `bus_messages`, `bus_subscriptions`, `bus_external_agents`, `bus_delivery_receipts`, `bus_offline_queue`
- No env, compose or CI changes.

**Revert prerequisites** (batches that must still be restored first for a stand-alone `git revert`): B5, B9 (`bus_service.py:44`, `bus_triggers.py:303,333`).

**Verify** (run from repo root; record the "before" route count on `develop` first):

```bash
git grep -nE "\b(app\.controllers\.bus|app\.controllers\.bus_triggers|app\.models\.bus_models|app\.repository\.bus_repository|app\.services\.bus_service|app\.services\.bus_triggers)\b|\b(bus|bus_triggers)\.router\b" -- backend/app backend/tests   # expect: no output
cd backend
python -c "import app.main as m; print(len(m.app.routes))"        # expect: before - 11
uv run black --check app/ tests/
python -m pytest tests/ -q                                         # expect: green; cases drop by ~154 (static; parametrised cases may add more)
```

### B5. Catalyst + Council of Perspectives

**Size:** 9 modules / 4,371 LOC / 5 test file(s) moved (135 cases removed, static count) / route delta -16.

**Why here:** Council needs `model_router` (stays until C3), `consciousness_council_bridge` (`deliberation_service.py:36`, goes in B6) and `event_publisher`/`ws_events` (`:41-42`, go in B9). Archiving council first lets those three go cleanly later. `council_repository` + `council_models` are currently dragged into every DB import by `repository/__init__.py:7-13` (see 4.1), which this batch must edit.

**Paths** (`git mv` to `archive/backend/<path under app/>`, matching the ADR-006 precedent, e.g. `archive/backend/controllers/worlds.py`):

- `backend/app/services/council/` (whole package)
- `backend/app/controllers/catalyst_creativity.py`
- `backend/app/controllers/council.py`
- `backend/app/models/council_models.py`
- `backend/app/prompts/catalyst_prompts.py`
- `backend/app/repository/council_repository.py`
- `backend/app/services/catalyst_service.py`
- `backend/app/services/council_service.py`

**Tests to move** (to `archive/backend/tests/`):

- `tests/test_catalyst_service.py` (30)
- `tests/test_council_deliberation_service.py` (58)
- `tests/test_council_deliberation_telemetry.py` (7)
- `tests/test_council_models.py` (37)
- `tests/test_council_repository.py` (3)

**Edits to kept code:**

- `main.py:21`, `:30`, `:57` (`council_repository`), `:111-113` (`ensure_council_tables`), `:248`, `:256-258` (catalyst mount).
- `repository/__init__.py:7-13`: remove `council_repository` from the eager import and from `__all__` (otherwise every `app.repository.*` import fails).

**Migrations / config / compose:**

- DB tables created by the archived code (left in place, see 6.1): `council_sessions`, `council_perspectives`, `council_votes` (also legacy `backend/migrations/005_council_sessions.sql`, not applied by the runner)
- No env, compose or CI changes.

**Revert prerequisites** (batches that must still be restored first for a stand-alone `git revert`): B6, B9, C3 (`deliberation_service.py:36,40-42`, `catalyst_service.py:22`).

**Verify** (run from repo root; record the "before" route count on `develop` first):

```bash
git grep -nE "\b(app\.controllers\.catalyst_creativity|app\.controllers\.council|app\.models\.council_models|app\.prompts\.catalyst_prompts|app\.repository\.council_repository|app\.services\.catalyst_service|app\.services\.council\.deliberation_service|app\.services\.council\.voice_registry|app\.services\.council_service)\b|\b(council|catalyst_creativity)\.router\b" -- backend/app backend/tests   # expect: no output
cd backend
python -c "import app.main as m; print(len(m.app.routes))"        # expect: before - 16
uv run black --check app/ tests/
python -m pytest tests/ -q                                         # expect: green; cases drop by ~135 (static; parametrised cases may add more)
```

### B6. Consciousness module, backup and lineage

**Size:** 10 modules / 3,521 LOC / 6 test file(s) moved (82 cases removed, static count) / route delta -19.

**Why here:** Consciousness is imported only by the council bridge (`consciousness_council_bridge.py`, itself imported only by `deliberation_service.py:36`) and by its own two controllers. After B5 nothing outside the batch references it.

**Paths** (`git mv` to `archive/backend/<path under app/>`, matching the ADR-006 precedent, e.g. `archive/backend/controllers/worlds.py`):

- `backend/app/consciousness/` (whole package)
- `backend/app/controllers/consciousness_backup_controller.py`
- `backend/app/controllers/consciousness_lineage_controller.py`
- `backend/app/repository/consciousness_lineage_repository.py`
- `backend/app/services/consciousness_backup_service.py`
- `backend/app/services/consciousness_council_bridge.py`
- `backend/app/services/consciousness_lineage_service.py`
- `backend/app/migrations/sql/002_consciousness_lineage.sql`
- `backend/scripts/consciousness_exploration.py`, `backend/scripts/run_consciousness_exploration.ps1`
- `backend/CONSCIOUSNESS_LINEAGE_IMPLEMENTATION.md`
- `backend/test_consciousness_lineage.py`, `backend/test_exploration_sync.py` (repo-root scripts, not collected by CI: `testpaths = ["tests", "app"]`)

**Tests to move** (to `archive/backend/tests/`):

- `test_consciousness_lineage.py` (5)
- `tests/test_consciousness_backup.py` (6)
- `tests/test_consciousness_backup_service.py` (18)
- `tests/test_consciousness_council_bridge.py` (28)
- `tests/test_consciousness_lineage_service.py` (25)
- `tests/test_consciousness_exploration.py` (0) (integration-marked httpx script, no static `app` import; 11 mentions of consciousness)

**Edits to kept code:**

- `main.py:34`, `:35`, `:266-268`, `:269-271`.

**Migrations / config / compose:**

- DB tables created by the archived code (left in place, see 6.1): `consciousness_instances`, `consciousness_contributions`, `consciousness_evolution_events`, `consciousness_relationships` (from `002_consciousness_lineage.sql`)
- `app/migrations/sql/002_consciousness_lineage.sql` moves with the batch. Safe for existing databases: `migrations/runner.py:46-47` globs the directory and only compares file stems against `schema_migrations`, so a missing file for an already-applied version is ignored. Fresh installs simply never create those tables.

**Revert prerequisites** (batches that must still be restored first for a stand-alone `git revert`): none.

**Verify** (run from repo root; record the "before" route count on `develop` first):

```bash
git grep -nE "\b(app\.consciousness\.blackboard|app\.consciousness\.commons_explorer|app\.consciousness\.config|app\.consciousness\.context_builder|app\.controllers\.consciousness_backup_controller|app\.controllers\.consciousness_lineage_controller|app\.repository\.consciousness_lineage_repository|app\.services\.consciousness_backup_service|app\.services\.consciousness_council_bridge|app\.services\.consciousness_lineage_service)\b|\b(consciousness_backup_controller|consciousness_lineage_controller)\.router\b" -- backend/app backend/tests   # expect: no output
cd backend
python -c "import app.main as m; print(len(m.app.routes))"        # expect: before - 19
uv run black --check app/ tests/
python -m pytest tests/ -q                                         # expect: green; cases drop by ~82 (static; parametrised cases may add more)
# fresh-DB smoke: start a clean postgres, boot the app, confirm /health returns 200 and migrations log "applied" without the removed file
```

### B7. Communication Commons + Discord bridge

**Size:** 8 modules / 3,780 LOC / 8 test file(s) moved (155 cases removed, static count) / route delta -24.

**Why here:** `communication_service` <-> `discord_bridge` <-> `agent_response_service` form an import cycle (`communication_service.py:239` -> `discord_bridge`; `discord_bridge.py:341` -> `communication_service`; `communication_service.py:381` -> `agent_response_service`; `agent_response_service.py:528,578` -> back). They must move together. `websocket_manager` is NOT here: `event_publisher.py:47` (B9) still imports it. ADR-006 explicitly deferred Discord as "woven into Commons"; this batch is that decoupling pass.

**Paths** (`git mv` to `archive/backend/<path under app/>`, matching the ADR-006 precedent, e.g. `archive/backend/controllers/worlds.py`):

- `backend/app/config/discord.py`
- `backend/app/controllers/communication.py`
- `backend/app/controllers/discord.py`
- `backend/app/repository/communication_repository.py`
- `backend/app/repository/discord_repository.py`
- `backend/app/services/agent_response_service.py`
- `backend/app/services/communication_service.py`
- `backend/app/services/discord_bridge.py`
- `backend/app/migrations/sql/001b_communication_commons.sql`

**Tests to move** (to `archive/backend/tests/`):

- `tests/test_agent_response_service.py` (19)
- `tests/test_communication.py` (25)
- `tests/test_communication_pagination.py` (15)
- `tests/test_communication_service.py` (6)
- `tests/test_discord_bridge.py` (28)
- `tests/test_discord_bridge_service.py` (27)
- `tests/test_discord_config.py` (13)
- `tests/test_discord_controller.py` (4)

**Tests to edit in place:**

- `tests/test_controller_auth.py` (21 cases): delete `TestCommunicationAuth` (L86-158, 8) and `TestDiscordAuth` (L166-229, 10); keep `TestChatStreamAuth` (3) – **18 removed**

**Edits to kept code:**

- `main.py:15`, `:32`, `:41` (`manager` becomes unused once the endpoint below is gone), `:50-54` (discord bridge + config imports), `:67` (`discord_repository`), `:143-145` (`ensure_discord_tables`), `:176-196` (Discord startup block), `:200-205` (Discord shutdown), `:242`, `:262`, and the `/ws/{instance_id}` endpoint `:309-381`.
- `pyproject.toml`: `discord-py>=2.3.0` is imported only by `services/discord_bridge.py`. Drop in a separate chore commit (it regenerates `uv.lock`).

**Migrations / config / compose:**

- DB tables created by the archived code (left in place, see 6.1): `communication_channels`, `channel_members`, `communication_messages`, `message_reactions`, `instance_presence`, `message_read_receipts`; `discord_bridge_config`, `discord_channel_mappings`, `discord_message_links`, `discord_delivery_events`
- Env: `DISCORD_ENABLED`, `DISCORD_BOT_TOKEN`, `DISCORD_ALLOWED_USERS`, `DISCORD_CHANNEL_MAP` are read only by archived modules (`config/discord.py`, `discord_bridge.py`). Not present in `.env.example` or any root compose file.
- `app/migrations/sql/001b_communication_commons.sql` moves with the batch. Safe for existing databases: `migrations/runner.py:46-47` globs the directory and only compares file stems against `schema_migrations`, so a missing file for an already-applied version is ignored. Fresh installs simply never create those tables.

**Revert prerequisites** (batches that must still be restored first for a stand-alone `git revert`): B8, B9 (`agent_response_service.py:49-50`, `communication.py:16`, `communication_service.py:18`).

**Verify** (run from repo root; record the "before" route count on `develop` first):

```bash
git grep -nE "\b(app\.config\.discord|app\.controllers\.communication|app\.controllers\.discord|app\.repository\.communication_repository|app\.repository\.discord_repository|app\.services\.agent_response_service|app\.services\.communication_service|app\.services\.discord_bridge)\b|\b(communication|discord)\.router\b" -- backend/app backend/tests   # expect: no output
cd backend
python -c "import app.main as m; print(len(m.app.routes))"        # expect: before - 24
uv run black --check app/ tests/
python -m pytest tests/ -q                                         # expect: green; cases drop by ~155 (static; parametrised cases may add more)
# fresh-DB smoke: start a clean postgres, boot the app, confirm /health returns 200 and migrations log "applied" without the removed file
```

### B8. Agent Factory + Spawn Templates

**Size:** 8 modules / 3,927 LOC / 5 test file(s) moved (171 cases removed, static count) / route delta -18.

**Why here:** After B7, `agent_response_service` (the only non-controller importer of the factory stack: `agent_response_service.py:49-50` -> `agent_models`/`agent_repository`) is gone, so the factory, spawn templates and their models have no remaining importer except their own controllers.

**Paths** (`git mv` to `archive/backend/<path under app/>`, matching the ADR-006 precedent, e.g. `archive/backend/controllers/worlds.py`):

- `backend/app/controllers/agents.py`
- `backend/app/controllers/spawn_templates.py`
- `backend/app/models/agent_models.py`
- `backend/app/models/spawn_template_models.py`
- `backend/app/repository/agent_repository.py`
- `backend/app/services/agent_factory_service.py`
- `backend/app/services/agent_mcp_service.py`
- `backend/app/services/spawn_template_service.py`

**Tests to move** (to `archive/backend/tests/`):

- `tests/test_agent_factory_service.py` (32)
- `tests/test_agent_mcp_service.py` (21)
- `tests/test_agent_models.py` (30)
- `tests/test_spawn_template_models.py` (29)
- `tests/test_spawn_template_service.py` (53)

**Tests to edit in place:**

- `tests/test_core_state_models.py` (33 cases): delete `TestPersonalityTraitsValidator` (L294-328, 6) and the `agent_models` import (L29) – **6 removed**; the file keeps its `core_state` classes

**Edits to kept code:**

- `main.py:16`, `:31`, `:243`, `:259-261`.
- `pyproject.toml`: `langchain-mcp-adapters` is imported only by `services/agent_mcp_service.py`; same chore-commit rule. The `agents` table creation and seed (`dependencies.py:730`, `_seed_initial_agents` `:812`) stays, see section 6.1.

**Migrations / config / compose:**

- DB tables created by the archived code (left in place, see 6.1): `agents` (+ seed rows) and `world_lore_agents` stay: created in `dependencies.py:730` / `:796`, not by the archived repository
- Env: `OBSIDIAN_API_KEY`, `OBSIDIAN_VAULT_PATH` (read only in `agent_mcp_service.py`).

**Revert prerequisites** (batches that must still be restored first for a stand-alone `git revert`): none.

**Verify** (run from repo root; record the "before" route count on `develop` first):

```bash
git grep -nE "\b(app\.controllers\.agents|app\.controllers\.spawn_templates|app\.models\.agent_models|app\.models\.spawn_template_models|app\.repository\.agent_repository|app\.services\.agent_factory_service|app\.services\.agent_mcp_service|app\.services\.spawn_template_service)\b|\b(agents|spawn_templates)\.router\b" -- backend/app backend/tests   # expect: no output
cd backend
python -c "import app.main as m; print(len(m.app.routes))"        # expect: before - 18
uv run black --check app/ tests/
python -m pytest tests/ -q                                         # expect: green; cases drop by ~171 (static; parametrised cases may add more)
```

### B9. Instances, agent registry, sandbox, agent WebSocket

**Size:** 11 modules / 5,630 LOC / 8 test file(s) moved (271 cases removed, static count) / route delta -13.

**Why here:** Last of the agent-runtime cluster: `instance_manager` -> `event_publisher` (`instance_manager.py:32`) -> `websocket_manager` (`event_publisher.py:47`). Everything that imported these (task router, bus, council, commons) is already archived. This is also where the two surviving soft couplings to KEEP code get cut (KB attribution and health probes, see 4.2-4.3).

**Paths** (`git mv` to `archive/backend/<path under app/>`, matching the ADR-006 precedent, e.g. `archive/backend/controllers/worlds.py`):

- `backend/app/sandbox/` (whole package)
- `backend/app/controllers/agent_ws.py`
- `backend/app/controllers/instances.py`
- `backend/app/models/ws_events.py`
- `backend/app/repository/instance_repository.py`
- `backend/app/services/agent_registry.py`
- `backend/app/services/event_publisher.py`
- `backend/app/services/instance_manager.py`
- `backend/app/websocket_manager.py`
- `docker-compose.agents.yml`
- `docker/agent/Dockerfile`, `docker/agent/config.template.yaml`

**Tests to move** (to `archive/backend/tests/`):

- `tests/test_agent_registration.py` (22)
- `tests/test_agent_registry.py` (33)
- `tests/test_event_publisher.py` (54)
- `tests/test_instance_manager.py` (18)
- `tests/test_instance_manager_telemetry.py` (6)
- `tests/test_sandbox.py` (23)
- `tests/test_sandbox_models.py` (76)
- `tests/test_ws_events.py` (35)

**Tests to edit in place:**

- `tests/test_health_aggregator.py`: delete `TestCheckWebsocketManager` (L367-384, 1) and the `check_websocket_manager` patch entries (L503, L561) – **1 removed**
- `tests/test_metrics_service.py` (13 cases): delete the 3 `InstanceManager._calculate_cpu_percent` tests (L196, L221, L242) – **3 removed**
- `tests/test_knowledge_attribution_verification.py` (14 cases): helper `_make_test_instance` (L49-60) imports `AgentInstance`; used at L97 and L193, plus `patch("app.repository.instance_repository.get_instance")` at L103. Rewrite those 2 tests to the D2 outcome – 0 removed

**Edits to kept code:**

- `main.py:22`, `:39` (`agent_websocket_endpoint`), `:45-48` (`agent_registry`), `:58` (`instance_repository`), `:115-117` (`ensure_instance_tables`), `:162-167` (registry init), `:214-219` (registry shutdown), `:249`, and the `/ws/agent/{agent_id}` endpoint `:384-402`.
- `services/health_aggregator.py:283-312` (`check_websocket_manager`) and call `:573`; `core/health.py:192-215` and call `:417` (this file is a C3 module; if C3 is skipped the edit still has to happen here).
- `services/knowledgebase_service.py:918` and `:1005` (lazy `from app.repository import instance_repository` inside the attribution check). **Behaviour change**, decision D2.
- `tests/test_health_aggregator.py:503`, `:561`, class `TestCheckWebsocketManager` `:367-384`.
- `pyproject.toml`: `docker` (sole users `sandbox/container_manager.py`, `services/instance_manager.py`) and `aiosqlite` (sole user `sandbox/state_manager.py`) become unused.

**Migrations / config / compose:**

- DB tables created by the archived code (left in place, see 6.1): `agent_instances`, `instance_trust_metrics`
- Compose: `docker-compose.agents.yml` (services `agent`, `discord-agent`, `dev-agent`) builds `docker/agent/Dockerfile` and attaches to the external `core-network`. Root `docker-compose*.yml` do **not** mount `docker.sock` into `core-backend` and reference none of these services, so they need no change. `docs/deployment/agent-containerization.md` and `sandbox-integration.md` describe this batch.

**Revert prerequisites** (batches that must still be restored first for a stand-alone `git revert`): none.

**Verify** (run from repo root; record the "before" route count on `develop` first):

```bash
git grep -nE "\b(app\.controllers\.agent_ws|app\.controllers\.instances|app\.models\.ws_events|app\.repository\.instance_repository|app\.sandbox\.container_manager|app\.sandbox\.security|app\.sandbox\.state_manager|app\.services\.agent_registry|app\.services\.event_publisher|app\.services\.instance_manager|app\.websocket_manager)\b|\b(instances|agent_ws)\.router\b" -- backend/app backend/tests   # expect: no output
cd backend
python -c "import app.main as m; print(len(m.app.routes))"        # expect: before - 13
uv run black --check app/ tests/
python -m pytest tests/ -q                                         # expect: green; cases drop by ~271 (static; parametrised cases may add more)
cd .. && docker compose config -q && git grep -n "docker-compose.agents" -- . ":!archive"   # expect: no hits
```

### C1. Comprehension + Evaluation engines and the eval harness (UNSURE: conditional, gated on section 8)

**Size:** 13 modules / 5,855 LOC / 14 test file(s) moved (432 cases removed, static count) / route delta -14.

**Why here:** These are the standalone "Comprehension Engine" and "Evaluation Engine". The live pipeline does NOT use them: `core/agents/comprehension_agent.py` and `evaluation_agent.py` import only `dependencies`, `models.core_state` and `utils.json_repair`. Product call, hence UNSURE. `app/eval/` is the drift harness for `evaluation_service` (`eval/runner.py` imports it) so it travels with it.

**Paths** (`git mv` to `archive/backend/<path under app/>`, matching the ADR-006 precedent, e.g. `archive/backend/controllers/worlds.py`):

- `backend/app/eval/` (whole package)
- `backend/app/controllers/comprehension.py`
- `backend/app/controllers/evaluation.py`
- `backend/app/models/comprehension_models.py`
- `backend/app/models/evaluation_models.py`
- `backend/app/models/task_models.py`
- `backend/app/repository/comprehension_repository.py`
- `backend/app/repository/evaluation_repository.py`
- `backend/app/services/comprehension_service.py`
- `backend/app/services/evaluation_service.py`
- `backend/app/services/output_validator.py`
- `backend/tests/fixtures/eval_cases/` (code.json, file_ops.json, research.json, ...)
- `backend/app/eval/README.md`, `backend/app/eval/FINDINGS_code_drift_2026-06-19.md` (travel with the `eval/` directory)

**Tests to move** (to `archive/backend/tests/`):

- `tests/test_comprehension.py` (52)
- `tests/test_comprehension_models.py` (23)
- `tests/test_comprehension_service.py` (35)
- `tests/test_eval_benchmark_floor.py` (3)
- `tests/test_eval_cases.py` (12)
- `tests/test_eval_protocol.py` (12)
- `tests/test_eval_runner.py` (11)
- `tests/test_eval_validation_wiring.py` (4)
- `tests/test_evaluation.py` (69)
- `tests/test_evaluation_models.py` (36)
- `tests/test_evaluation_service.py` (73)
- `tests/test_evaluation_service_pure.py` (51)
- `tests/test_output_validator.py` (18)
- `tests/test_task_models.py` (29)

**Tests to edit in place:**

- `tests/test_core_state_models.py`: delete `TestComprehensionInputValidator` (L336-366, 4) and the `comprehension_models` import (L30) – **4 removed**

**Edits to kept code:**

- `main.py:25`, `:26`, `:61`, `:62`, `:127-133`, `:252`, `:253`.

**Migrations / config / compose:**

- DB tables created by the archived code (left in place, see 6.1): `comprehension_results`, `evaluations`
- Env: `ENABLE_OUTPUT_VALIDATION` (read only in `output_validator`/`evaluation_service`).

**Revert prerequisites** (batches that must still be restored first for a stand-alone `git revert`): C2 (`comprehension_service.py:45,531`, `evaluation_service.py:36-37`).

**Verify** (run from repo root; record the "before" route count on `develop` first):

```bash
git grep -nE "\b(app\.controllers\.comprehension|app\.controllers\.evaluation|app\.eval\.cases|app\.eval\.protocol|app\.eval\.runner|app\.models\.comprehension_models|app\.models\.evaluation_models|app\.models\.task_models|app\.repository\.comprehension_repository|app\.repository\.evaluation_repository|app\.services\.comprehension_service|app\.services\.evaluation_service|app\.services\.output_validator)\b|\b(comprehension|evaluation)\.router\b" -- backend/app backend/tests   # expect: no output
cd backend
python -c "import app.main as m; print(len(m.app.routes))"        # expect: before - 14
uv run black --check app/ tests/
python -m pytest tests/ -q                                         # expect: green; cases drop by ~432 (static; parametrised cases may add more)
```

### C2. LangMem memory (UNSURE: conditional, gated on section 8)

**Size:** 4 modules / 2,350 LOC / 2 test file(s) moved (53 cases removed, static count) / route delta -15.

**Why here:** LangMem three-tier memory. No pipeline or chat importer (`memory_service` is imported only by `controllers/memory.py` and `main.py:49`). Must follow C1 because `evaluation_service.py:36-37` and `comprehension_service.py:531` import `memory_repository` and `comprehension_service.py:45` imports `embedding_service`.

**Paths** (`git mv` to `archive/backend/<path under app/>`, matching the ADR-006 precedent, e.g. `archive/backend/controllers/worlds.py`):

- `backend/app/controllers/memory.py`
- `backend/app/repository/memory_repository.py`
- `backend/app/services/embedding_service.py`
- `backend/app/services/memory_service.py`

**Tests to move** (to `archive/backend/tests/`):

- `tests/test_embedding_service.py` (19)
- `tests/test_memory_service.py` (34)

**Edits to kept code:**

- `main.py:24`, `:49` (`memory_service`), `:60`, `:123-125`, `:169-174` (memory init), `:251`.
- `repository/__init__.py:7-13`: remove `memory_repository`.
- `pyproject.toml`: `langmem` is imported only by `services/memory_service.py`.

**Migrations / config / compose:**

- DB tables created by the archived code (left in place, see 6.1): `semantic_memories`, `episodic_memories`, `procedural_memories`
- No env, compose or CI changes.

**Revert prerequisites** (batches that must still be restored first for a stand-alone `git revert`): none.

**Verify** (run from repo root; record the "before" route count on `develop` first):

```bash
git grep -nE "\b(app\.controllers\.memory|app\.repository\.memory_repository|app\.services\.embedding_service|app\.services\.memory_service)\b|\b(memory)\.router\b" -- backend/app backend/tests   # expect: no output
cd backend
python -c "import app.main as m; print(len(m.app.routes))"        # expect: before - 15
uv run black --check app/ tests/
python -m pytest tests/ -q                                         # expect: green; cases drop by ~53 (static; parametrised cases may add more)
```

### C3. Admin / audit / model router / MCP / system monitor (UNSURE: conditional, gated on section 8)

**Size:** 9 modules / 2,702 LOC / 8 test file(s) moved (162 cases removed, static count) / route delta -30.

**Why here:** Operational surface. `admin` hosts API-key CRUD (`/admin/keys`) and webhook registration (`/admin/webhooks`) that the kept engine relies on (`engine.py:33` imports `webhook_service`), so this batch is gated on decision D1. Cheapest sub-option: archive only `audit`, `model_router`, `core/health` and the routes that use them.

**Paths** (`git mv` to `archive/backend/<path under app/>`, matching the ADR-006 precedent, e.g. `archive/backend/controllers/worlds.py`):

- `backend/app/controllers/admin.py`
- `backend/app/controllers/audit.py`
- `backend/app/controllers/mcp.py`
- `backend/app/controllers/system_monitor.py`
- `backend/app/core/health.py`
- `backend/app/repository/audit_repository.py`
- `backend/app/repository/model_metrics_repository.py`
- `backend/app/services/audit_service.py`
- `backend/app/services/model_router.py`

**Tests to move** (to `archive/backend/tests/`):

- `tests/test_audit.py` (28)
- `tests/test_audit_service.py` (11)
- `tests/test_health_alerts.py` (12)
- `tests/test_mcp_health.py` (13)
- `tests/test_model_metrics.py` (25)
- `tests/test_model_router.py` (45)
- `tests/test_model_router_lmstudio.py` (8)
- `tests/test_model_selection.py` (19)

**Tests to edit in place:**

- `tests/test_rbac.py` (28 cases): delete `test_admin_keys_route_uses_require_role_admin` (L355-379, imports `controllers.admin`) – **1 removed**

**Edits to kept code:**

- `main.py:12`, `:20`, `:33`, `:36`, `:65` (`audit_repository`), `:68` (`model_metrics_repository`), `:147-149` (`ensure_model_metrics_tables`), `:239`, `:247`, `:263-265`, `:272-274`.
- `repository/__init__.py:7-13`: remove `model_metrics_repository` (this empties the eager import list; delete the import block entirely).

**Migrations / config / compose:**

- DB tables created by the archived code (left in place, see 6.1): `audit_log`, `model_metrics`
- Env: `CORE_MCP_CONFIG` (mcp controller), `LMSTUDIO_CONTEXT_WINDOW` (model_router). Both only read by archived modules. The top-level `mcp/` directory has no imports from `backend/app` (grep clean) and is unaffected.

**Revert prerequisites** (batches that must still be restored first for a stand-alone `git revert`): none.

**Verify** (run from repo root; record the "before" route count on `develop` first):

```bash
git grep -nE "\b(app\.controllers\.admin|app\.controllers\.audit|app\.controllers\.mcp|app\.controllers\.system_monitor|app\.core\.health|app\.repository\.audit_repository|app\.repository\.model_metrics_repository|app\.services\.audit_service|app\.services\.model_router)\b|\b(admin|audit|mcp|system_monitor)\.router\b" -- backend/app backend/tests   # expect: no output
cd backend
python -c "import app.main as m; print(len(m.app.routes))"        # expect: before - 30
uv run black --check app/ tests/
python -m pytest tests/ -q                                         # expect: green; cases drop by ~162 (static; parametrised cases may add more)
```


## 6. Cross-cutting

### 6.1 Database and migrations
- `app/migrations/runner.py:46-47` applies `app/migrations/sql/*.sql` by file stem and ignores applied versions with no file, so archiving `001b_communication_commons.sql` (B7) and `002_consciousness_lineage.sql` (B6) is safe for existing databases; only fresh installs skip those tables. `001_initial_schema.sql`, `003_webhooks.sql` and `004_knowledge_attribution.sql` are KEEP.
- `backend/migrations/` (the older directory: `001_communication_commons.sql` ... `006_knowledge_attribution.sql`) is **not read by anything** (the runner points at `app/migrations/sql/`; `001b_...sql:6` says as much). Leave it; flag as D5.
- Tables created by `ensure_*_tables()` in archived repositories stay in existing databases and keep being created by nobody once the repo is archived. A future `D2` data step (export, then `DROP TABLE`) should be its own reviewed change. Tables per batch are listed in each batch.
- `setup_db_schema` (section 4.4) is left untouched by this plan; trimming it is a separate batch after B8 (agent seeds) and a decision on `kb_documents.world_id`.

### 6.2 Dependencies (`backend/pyproject.toml`), drop after the owning batch
`discord-py` (B7), `langchain-mcp-adapters` (B8), `docker` + `aiosqlite` (B9), `langmem` (C2). Of the packages I scanned (`discord`, `docker`, `langmem`, `langchain_mcp_adapters`, `aiosqlite`, `psutil`, `redis`, `openai`, `anthropic`, `pypdf`, `docx`, `pytesseract`, `jwt`, `opentelemetry`) only these five lose every importer; the rest keep at least one after all batches.

### 6.3 Docker, CI, docs
- Root `docker-compose.yml`, `.dev.yml`, `.prod.yml`: no service or env var belongs to archived code; `core-backend` mounts only `./backend`, `backend-venv`, `backend-data` (no `docker.sock`). **No changes.** `n8n` and `ollama` profiles are independent.
- `docker-compose.agents.yml` and `docker/agent/` are the only compose-level casualty (B9).
- `.github/workflows/ci.yml` runs `black --check app/ tests/` and `pytest`; both ignore `archive/`. The frontend job is unaffected by backend archiving. **No changes.**
- `.githooks/pre-commit` mirrors CI. No change.
- Docs to update after the last executed batch (not part of any batch): `CLAUDE.md` and `AGENTS.md` (Architecture bullets, Key Paths rows for Agent Factory, bus, council, catalyst, consciousness bridge), `README.md`, `docs/README.md` index, and the `docs/council`, `docs/consciousness`, `docs/deployment/agent-containerization.md`, `sandbox-integration.md` pages (either move to `archive/docs/` as ADR-006 did, or banner them).

## 7. Risks

1. **Out-of-repo consumers.** ADR-006 says GPW/PWE depend on CORE. If GPW, n8n flows or OpenClaw agent containers call `/council`, `/bus`, `/communication`, `/instances`, `/ws/agent/*`, B4-B9 break them silently. Nothing in this repo shows who calls what. Mitigation: ask the GPW owners for their endpoint list before B4; run B2 first as a canary (zero known callers).
2. **UI breaks even though `ui/` is out of scope.** Verified UI callers (read-only): `/communication` + `/ws/{id}` (3 services), `/discord` (dashboard route `tools/discord-bridge`), `/agents` (3 services), `/instances` + `/health` (`instance.service.ts`), `/spawn-templates`, `/mcp` (`tools` route), `/system` (landing-page command deck). B1-B6, C1, C2 have **no** UI callers. So B7, B8, B9 and C3 must wait for the thin-UI change, or ship with the corresponding routes/pages removed in the same release.
3. **ADR-006 contradiction.** It records Council/Catalyst/Consciousness/Commons/Bus/MCP as deliberately kept and Discord as deferred. Executing without ADR-007 leaves the repo's own documentation self-contradictory (`CLAUDE.md` still lists them).
4. **Static counts and an unrun baseline.** Test counts are static and the suite was not run (no installed dependencies). A pre-existing red test would be misattributed to a batch. Record the green baseline first.
5. **Behaviour change in B9 (D2).** Cutting the `instance_repository` lookup removes KB-attribution's "orphaned source instance" warning. Small, but user-visible in the attribution browser.
6. **Admin removal (C3) removes the only HTTP way to create API keys and register webhooks.** Hence D1.
7. **Lazy-import traps.** About 85 of roughly 450 import edges between groups are function-level imports; they fail only when the code path runs, not at `import app.main`. Only three KEEP -> ARCHIVE edges exist and all are lazy (`knowledgebase_service.py:918`, `:1005`, `health_aggregator.py:287`). Every batch's verify step also greps for the dotted names, and the simulation already counted lazy edges.
8. **Rollback is easy for code, not for data.** Reverting restores code and routes. Tables were never dropped, so rows written by a still-running older deployment during the window are preserved; the reverse is also true: nothing here cleans the DB.
9. **Line numbers rot.** Every `main.py:NN` reference shifts as batches land. The "expect: no output" `git grep` in each verify block is the durable check, not the line numbers.

## 8. Decisions needed (UNSURE items)

| ID | Question | Recommendation |
|---|---|---|
| D1 | Keep `/admin` (API keys, webhooks, run cleanup, model usage)? | Keep `controllers/admin.py` and its direct dependencies; archive only `audit`, `model_router` routes and `core/health`. Then C3 shrinks to `audit` + `system_monitor` + `mcp`. |
| D2 | Keep KB attribution's instance existence check (needs `instance_repository` + `agent_instances`)? | Drop the check in B9 (replace with "source_instance_id recorded, not verified") and archive `instance_repository`; otherwise move `instance_repository.py` out of B9 and keep it as a KEEP module. |
| D3 | Are the standalone Comprehension and Evaluation engines (plus `app/eval/`) part of "the cognitive engine"? The pipeline uses its own agents, not these services. | Archive (C1) only if the product definition of CORE = pipeline in `core_graph_v2`; the `eval/` harness is the only thing that measures evaluation drift, so decide whether that has value on its own. |
| D4 | LangMem memory (C2): needed for the engine? | No pipeline or chat importer; archive unless a roadmap item needs it (`docs/implementation`). |
| D5 | `utils/errors.py` (0 importers, 1 test file, 39 cases) and the unused `backend/migrations/` directory: delete, archive or keep? | Archive both in a final tidy-up; neither affects runtime. |
| D6 | `mcp` controller and `system_monitor`: part of a "thin UI"? | Outside the thin-UI contract; archive with C3 only after the UI drops the Tools page and the command-deck resource widget. |
| D7 | Docs for archived subsystems: move to `archive/docs/` (ADR-006 style) or banner them? | Move, matching precedent. |

## 9. Execution protocol

0. **Before anything:** write ADR-007 (supersedes the "Kept" list of ADR-006); confirm D-decisions; obtain GPW/PWE caller list (risk 1); on a machine with `uv`, run `cd backend && uv sync && python -m pytest tests/ -q` and `python -c "import app.main as m; print(len(m.app.routes))"`, and store both numbers.
1. B1 -> B2 -> ... -> B9 in order, one PR each into `develop`, verify block green before the next. Skip nothing in the middle: later batches' "Edits to kept code" assume earlier ones landed.
2. After B9: `pyproject.toml` dependency chore, `uv lock`, docs sweep (6.3), update `archive/README.md`.
3. C1-C3 only after the matching decision; C1 before C2.
4. A separate DB change (6.1) is optional and out of this plan.

## Appendix A. Module inventory (every non-`__init__` module under `backend/app`)

`Reach`: `P` reachable from the pipeline root, `C` from chat/conversations, `–` neither. `Class` is KEEP / ARCHIVE / UNSURE. Importers are production code only (tests are in the last column) and exclude package `__init__`s; "(+N)" is the count not shown. `ARCHIVE`/`UNSURE` rows that still show `P`/`C` are the coupling accidents of section 4 (`council_models`, `council_repository`, `memory_repository`, `model_metrics_repository` via `repository/__init__.py:7`; `instance_repository` via `knowledgebase_service.py:918,1005`).

| Module | LOC | Reach | Class | Batch | Imported by (file:line, max 3 of N) | Tests (files / cases) |
|---|---:|---|---|---|---|---:|
| `auth.py` | 117 | PC | KEEP | – | `app/controllers/admin.py:39`, `app/controllers/agents.py:41`, `app/controllers/audit.py:20` (+22) | 2 / 25 |
| `config/discord.py` | 194 | – | ARCHIVE | B7 | `app/controllers/discord.py:26`, `app/main.py:51`, `app/services/discord_bridge.py:22` | 3 / 68 |
| `config/models.py` | 236 | PC | KEEP | – | `app/controllers/chat.py:19`, `app/core/langgraph/core_graph_v2.py:50`, `app/services/chat_service.py:349` | 1 / 38 |
| `config/startup_validator.py` | 361 | – | KEEP | – | `app/main.py:72` | 1 / 21 |
| `consciousness/blackboard.py` | 175 | – | ARCHIVE | B6 | `app/consciousness/commons_explorer.py:15`, `app/consciousness/context_builder.py:12`, `app/services/consciousness_council_bridge.py:92` (+2) | 1 / 28 |
| `consciousness/commons_explorer.py` | 407 | – | ARCHIVE | B6 | – | 0 / 0 |
| `consciousness/config.py` | 53 | – | ARCHIVE | B6 | `app/consciousness/blackboard.py:11`, `app/consciousness/commons_explorer.py:14`, `app/consciousness/context_builder.py:11` | 0 / 0 |
| `consciousness/context_builder.py` | 295 | – | ARCHIVE | B6 | `app/consciousness/commons_explorer.py:16`, `app/services/consciousness_council_bridge.py:93`, `app/services/consciousness_council_bridge.py:164` | 1 / 28 |
| `controllers/admin.py` | 514 | – | UNSURE | C3 | `app/main.py:8` | 2 / 53 |
| `controllers/agent_ws.py` | 438 | – | ARCHIVE | B9 | `app/controllers/instances.py:841`, `app/main.py:39`, `app/services/bus_service.py:44` | 1 / 22 |
| `controllers/agents.py` | 665 | – | ARCHIVE | B8 | `app/main.py:8` | 0 / 0 |
| `controllers/audit.py` | 106 | – | UNSURE | C3 | `app/main.py:8` | 0 / 0 |
| `controllers/bus.py` | 243 | – | ARCHIVE | B4 | `app/main.py:8` | 0 / 0 |
| `controllers/bus_triggers.py` | 138 | – | ARCHIVE | B4 | `app/main.py:8` | 0 / 0 |
| `controllers/catalyst_creativity.py` | 239 | – | ARCHIVE | B5 | `app/main.py:8` | 0 / 0 |
| `controllers/chat.py` | 278 | C | KEEP | – | `app/main.py:8` | 2 / 58 |
| `controllers/communication.py` | 321 | – | ARCHIVE | B7 | `app/main.py:8` | 3 / 61 |
| `controllers/comprehension.py` | 438 | – | UNSURE | C1 | `app/main.py:8` | 0 / 0 |
| `controllers/consciousness_backup_controller.py` | 149 | – | ARCHIVE | B6 | `app/main.py:8` | 0 / 0 |
| `controllers/consciousness_lineage_controller.py` | 513 | – | ARCHIVE | B6 | `app/main.py:8` | 0 / 0 |
| `controllers/conversations.py` | 103 | C | KEEP | – | `app/main.py:8` | 1 / 11 |
| `controllers/core_entry.py` | 364 | – | KEEP | – | `app/main.py:8` | 0 / 0 |
| `controllers/council.py` | 771 | – | ARCHIVE | B5 | `app/main.py:8` | 0 / 0 |
| `controllers/discord.py` | 469 | – | ARCHIVE | B7 | `app/main.py:8` | 2 / 25 |
| `controllers/engine.py` | 829 | P | KEEP | – | `app/core/health.py:221`, `app/main.py:8`, `app/services/health_aggregator.py:319` | 2 / 54 |
| `controllers/evaluation.py` | 280 | – | UNSURE | C1 | `app/main.py:8` | 0 / 0 |
| `controllers/health.py` | 264 | – | KEEP | – | `app/main.py:8` | 2 / 50 |
| `controllers/instances.py` | 878 | – | ARCHIVE | B9 | `app/main.py:8` | 0 / 0 |
| `controllers/knowledgebase.py` | 693 | – | KEEP | – | `app/main.py:8` | 0 / 0 |
| `controllers/local_llm.py` | 132 | – | KEEP | – | `app/main.py:8` | 0 / 0 |
| `controllers/mcp.py` | 249 | – | UNSURE | C3 | `app/main.py:8` | 0 / 0 |
| `controllers/memory.py` | 641 | – | UNSURE | C2 | `app/main.py:8` | 0 / 0 |
| `controllers/metrics.py` | 32 | – | KEEP | – | `app/main.py:8` | 1 / 15 |
| `controllers/mmcnc.py` | 643 | – | ARCHIVE | B2 | `app/main.py:8` | 0 / 0 |
| `controllers/spawn_templates.py` | 151 | – | ARCHIVE | B8 | `app/main.py:8` | 0 / 0 |
| `controllers/system_monitor.py` | 126 | – | UNSURE | C3 | `app/main.py:8` | 0 / 0 |
| `controllers/tasks.py` | 651 | – | ARCHIVE | B3 | `app/main.py:8` | 0 / 0 |
| `controllers/test_core.py` | 66 | – | ARCHIVE | B1 | `app/main.py:8` | 0 / 0 |
| `core/agent_orchestrator.py` | 791 | – | ARCHIVE | B1 | – | 1 / 19 |
| `core/agents/comprehension_agent.py` | 312 | P | KEEP | – | `app/core/langgraph/core_graph_v2.py:24` | 2 / 47 |
| `core/agents/evaluation_agent.py` | 374 | P | KEEP | – | `app/core/langgraph/core_graph_v2.py:27` | 2 / 43 |
| `core/agents/orchestration_agent.py` | 289 | P | KEEP | – | `app/core/langgraph/core_graph_v2.py:25` | 1 / 24 |
| `core/agents/reasoning_agent.py` | 414 | P | KEEP | – | `app/core/langgraph/core_graph_v2.py:26` | 3 / 68 |
| `core/circuit_breaker.py` | 197 | C | KEEP | – | `app/services/chat_service.py:27` | 1 / 19 |
| `core/health.py` | 479 | – | UNSURE | C3 | `app/controllers/admin.py:33` | 2 / 25 |
| `core/intelligence_log.py` | 110 | P | KEEP | – | `app/core/langgraph/core_graph_v2.py:29` | 1 / 19 |
| `core/langgraph/core_graph.py` | 56 | – | ARCHIVE | B1 | – | 0 / 0 |
| `core/langgraph/core_graph_v2.py` | 529 | P | KEEP | – | `app/controllers/engine.py:29` | 5 / 93 |
| `core/logging_config.py` | 66 | – | KEEP | – | `app/main.py:74` | 1 / 17 |
| `core/middleware.py` | 329 | – | KEEP | – | `app/controllers/admin.py:34`, `app/controllers/metrics.py:13`, `app/main.py:42` | 3 / 48 |
| `core/security.py` | 784 | P | KEEP | – | `app/controllers/admin.py:25`, `app/controllers/audit.py:21`, `app/controllers/engine.py:31` (+2) | 6 / 168 |
| `core/telemetry.py` | 121 | P | KEEP | – | `app/core/langgraph/core_graph_v2.py:28` | 1 / 20 |
| `core/tools/dispatcher.py` | 402 | P | KEEP | – | `app/core/agents/orchestration_agent.py:52`, `app/core/agents/reasoning_agent.py:22` | 1 / 34 |
| `dependencies.py` | 957 | PC | KEEP | – | `app/controllers/catalyst_creativity.py:25`, `app/controllers/conversations.py:75`, `app/controllers/core_entry.py:36` (+45) | 4 / 90 |
| `eval/cases.py` | 160 | – | UNSURE | C1 | `app/eval/protocol.py:26`, `app/eval/runner.py:39` | 3 / 26 |
| `eval/protocol.py` | 110 | – | UNSURE | C1 | – | 0 / 0 |
| `eval/runner.py` | 335 | – | UNSURE | C1 | – | 2 / 14 |
| `langgraph/comprehension_graph.py` | 175 | – | ARCHIVE | B1 | – | 0 / 0 |
| `langgraph/core_graph.py` | 77 | – | ARCHIVE | B1 | – | 0 / 0 |
| `middleware/correlation.py` | 51 | – | KEEP | – | `app/main.py:299` | 1 / 15 |
| `middleware/logging.py` | 23 | – | KEEP | – | – | 1 / 15 |
| `middleware/rate_limit.py` | 439 | – | KEEP | – | – | 1 / 58 |
| `migrations/runner.py` | 69 | – | KEEP | – | `app/main.py:43` | 1 / 5 |
| `models/agent_models.py` | 483 | – | ARCHIVE | B8 | `app/controllers/agents.py:30`, `app/models/spawn_template_models.py:25`, `app/repository/agent_repository.py:41` (+4) | 4 / 116 |
| `models/bus_models.py` | 268 | – | ARCHIVE | B4 | `app/controllers/bus.py:13`, `app/controllers/bus_triggers.py:14`, `app/controllers/tasks.py:39` (+2) | 4 / 154 |
| `models/comprehension_models.py` | 451 | – | UNSURE | C1 | `app/controllers/comprehension.py:20`, `app/repository/comprehension_repository.py:16`, `app/services/comprehension_service.py:25` | 4 / 143 |
| `models/comprehension_output.py` | 6 | PC | KEEP | – | – | 1 / 23 |
| `models/comprehension_state.py` | 45 | PC | KEEP | – | – | 1 / 20 |
| `models/core_state.py` | 220 | P | KEEP | – | `app/controllers/engine.py:28`, `app/controllers/test_core.py:7`, `app/core/agents/comprehension_agent.py:20` (+5) | 13 / 282 |
| `models/council_models.py` | 286 | PC | ARCHIVE | B5 | `app/controllers/council.py:25`, `app/repository/council_repository.py:17`, `app/services/council/deliberation_service.py:21` | 3 / 98 |
| `models/evaluation_models.py` | 427 | – | UNSURE | C1 | `app/controllers/evaluation.py:17`, `app/eval/runner.py:77`, `app/repository/evaluation_repository.py:15` (+1) | 5 / 233 |
| `models/evaluation_output.py` | 6 | PC | KEEP | – | – | 1 / 23 |
| `models/mmcnc_models.py` | 409 | – | ARCHIVE | B2 | `app/controllers/mmcnc.py:49`, `app/repository/mmcnc_repository.py:18` | 1 / 28 |
| `models/orchestration_output.py` | 7 | PC | KEEP | – | – | 1 / 23 |
| `models/reasoning_output.py` | 16 | PC | KEEP | – | – | 1 / 23 |
| `models/spawn_template_models.py` | 216 | – | ARCHIVE | B8 | `app/controllers/spawn_templates.py:14`, `app/services/spawn_template_service.py:24` | 2 / 82 |
| `models/task_models.py` | 221 | – | UNSURE | C1 | `app/controllers/tasks.py:16`, `app/repository/task_repository.py:15`, `app/services/comprehension_service.py:43` (+1) | 3 / 106 |
| `models/user_input.py` | 14 | PC | KEEP | – | `app/controllers/catalyst_creativity.py:26`, `app/controllers/core_entry.py:9` | 1 / 23 |
| `models/ws_events.py` | 337 | – | ARCHIVE | B9 | `app/services/council/deliberation_service.py:42`, `app/services/event_publisher.py:32` | 3 / 96 |
| `prompts/catalyst_prompts.py` | 175 | – | ARCHIVE | B5 | `app/services/catalyst_service.py:17` | 0 / 0 |
| `prompts/comprehension_agent_prompt.py` | 7 | – | ARCHIVE | B1 | – | 0 / 0 |
| `prompts/evaluation_agent_prompt.py` | 7 | – | ARCHIVE | B1 | – | 0 / 0 |
| `prompts/orchestration_agent_prompt.py` | 7 | – | ARCHIVE | B1 | – | 0 / 0 |
| `prompts/reasoning_agent_prompt.py` | 7 | – | ARCHIVE | B1 | – | 0 / 0 |
| `repository/agent_repository.py` | 843 | – | ARCHIVE | B8 | `app/controllers/agents.py:36`, `app/services/agent_factory_service.py:44`, `app/services/agent_response_service.py:50` (+1) | 0 / 0 |
| `repository/api_key_repository.py` | 211 | PC | KEEP | – | `app/core/security.py:72`, `app/core/security.py:124`, `app/core/security.py:195` (+2) | 1 / 61 |
| `repository/audit_repository.py` | 346 | – | UNSURE | C3 | `app/main.py:55`, `app/services/audit_service.py:29` | 1 / 28 |
| `repository/bus_repository.py` | 535 | – | ARCHIVE | B4 | `app/main.py:55`, `app/services/bus_service.py:43` | 0 / 0 |
| `repository/communication_repository.py` | 568 | – | ARCHIVE | B7 | `app/controllers/communication.py:15`, `app/services/agent_response_service.py:51`, `app/services/agent_response_service.py:55` (+2) | 1 / 15 |
| `repository/comprehension_repository.py` | 698 | – | UNSURE | C1 | `app/controllers/comprehension.py:259`, `app/controllers/comprehension.py:376`, `app/main.py:55` (+1) | 0 / 0 |
| `repository/consciousness_lineage_repository.py` | 822 | – | ARCHIVE | B6 | `app/controllers/consciousness_lineage_controller.py:133`, `app/controllers/consciousness_lineage_controller.py:237`, `app/controllers/consciousness_lineage_controller.py:329` (+3) | 2 / 30 |
| `repository/conversation_repository.py` | 230 | C | KEEP | – | `app/controllers/chat.py:11`, `app/controllers/conversations.py:11` | 1 / 11 |
| `repository/council_repository.py` | 560 | PC | ARCHIVE | B5 | `app/controllers/council.py:35`, `app/main.py:55`, `app/services/council/deliberation_service.py:28` | 1 / 3 |
| `repository/discord_repository.py` | 671 | – | ARCHIVE | B7 | `app/config/discord.py:13`, `app/controllers/discord.py:20`, `app/main.py:55` (+1) | 0 / 0 |
| `repository/evaluation_repository.py` | 589 | – | UNSURE | C1 | `app/main.py:55`, `app/services/evaluation_service.py:36` | 0 / 0 |
| `repository/health_repository.py` | 267 | – | KEEP | – | `app/controllers/health.py:201`, `app/controllers/health.py:231`, `app/controllers/health.py:241` (+2) | 1 / 22 |
| `repository/instance_repository.py` | 676 | C | ARCHIVE | B9 | `app/controllers/instances.py:22`, `app/controllers/instances.py:452`, `app/controllers/instances.py:678` (+6) | 5 / 112 |
| `repository/knowledgebase_repository.py` | 631 | C | KEEP | – | `app/controllers/knowledgebase.py:15`, `app/services/knowledgebase_service.py:15` | 1 / 1 |
| `repository/memory_repository.py` | 846 | PC | UNSURE | C2 | `app/main.py:55`, `app/services/comprehension_service.py:531`, `app/services/evaluation_service.py:36` (+2) | 1 / 34 |
| `repository/mmcnc_repository.py` | 757 | – | ARCHIVE | B2 | `app/controllers/mmcnc.py:64`, `app/main.py:55` | 0 / 0 |
| `repository/model_metrics_repository.py` | 200 | PC | UNSURE | C3 | `app/controllers/admin.py:512`, `app/main.py:55`, `app/services/model_router.py:473` (+1) | 1 / 25 |
| `repository/run_repository.py` | 308 | PC | KEEP | – | `app/controllers/admin.py:37`, `app/controllers/engine.py:32`, `app/main.py:55` | 0 / 0 |
| `repository/task_repository.py` | 635 | – | ARCHIVE | B3 | `app/controllers/tasks.py:26`, `app/main.py:55`, `app/services/task_router.py:32` | 0 / 0 |
| `repository/webhook_repository.py` | 426 | P | KEEP | – | `app/services/webhook_service.py:28` | 1 / 46 |
| `sandbox/container_manager.py` | 522 | – | ARCHIVE | B9 | – | 2 / 99 |
| `sandbox/security.py` | 163 | – | ARCHIVE | B9 | `app/sandbox/container_manager.py:21` | 2 / 99 |
| `sandbox/state_manager.py` | 638 | – | ARCHIVE | B9 | – | 2 / 99 |
| `services/agent_factory_service.py` | 548 | – | ARCHIVE | B8 | `app/controllers/agents.py:37`, `app/services/agent_response_service.py:54` | 1 / 32 |
| `services/agent_mcp_service.py` | 436 | – | ARCHIVE | B8 | `app/controllers/agents.py:38`, `app/services/agent_factory_service.py:45` | 1 / 21 |
| `services/agent_registry.py` | 514 | – | ARCHIVE | B9 | `app/controllers/agent_ws.py:31`, `app/controllers/instances.py:752`, `app/controllers/instances.py:800` (+3) | 2 / 55 |
| `services/agent_response_service.py` | 617 | – | ARCHIVE | B7 | `app/services/communication_service.py:381` | 2 / 25 |
| `services/audit_service.py` | 91 | – | UNSURE | C3 | `app/controllers/admin.py:40`, `app/controllers/audit.py:22` | 2 / 39 |
| `services/bus_service.py` | 474 | – | ARCHIVE | B4 | `app/controllers/bus.py:24`, `app/controllers/tasks.py:38` | 2 / 92 |
| `services/bus_triggers.py` | 391 | – | ARCHIVE | B4 | `app/controllers/bus_triggers.py:15` | 1 / 23 |
| `services/catalyst_service.py` | 337 | – | ARCHIVE | B5 | `app/controllers/catalyst_creativity.py:27`, `app/services/bus_triggers.py:333` | 1 / 30 |
| `services/chat_service.py` | 380 | C | KEEP | – | `app/controllers/chat.py:10` | 2 / 32 |
| `services/communication_service.py` | 433 | – | ARCHIVE | B7 | `app/controllers/communication.py:17`, `app/services/agent_response_service.py:528`, `app/services/discord_bridge.py:341` | 1 / 6 |
| `services/comprehension_service.py` | 1002 | – | UNSURE | C1 | `app/controllers/comprehension.py:28` | 2 / 87 |
| `services/consciousness_backup_service.py` | 268 | – | ARCHIVE | B6 | `app/controllers/consciousness_backup_controller.py:14` | 2 / 24 |
| `services/consciousness_council_bridge.py` | 335 | – | ARCHIVE | B6 | `app/services/council/deliberation_service.py:36` | 1 / 28 |
| `services/consciousness_lineage_service.py` | 496 | – | ARCHIVE | B6 | `app/controllers/consciousness_lineage_controller.py:15` | 2 / 30 |
| `services/council/deliberation_service.py` | 975 | – | ARCHIVE | B5 | `app/services/council_service.py:9` | 2 / 65 |
| `services/council/voice_registry.py` | 986 | – | ARCHIVE | B5 | `app/controllers/council.py:36`, `app/services/council/deliberation_service.py:29` | 1 / 58 |
| `services/council_service.py` | 14 | – | ARCHIVE | B5 | `app/controllers/council.py:37`, `app/services/bus_triggers.py:303` | 0 / 0 |
| `services/discord_bridge.py` | 507 | – | ARCHIVE | B7 | `app/controllers/discord.py:21`, `app/main.py:50`, `app/services/agent_response_service.py:578` (+1) | 3 / 61 |
| `services/embedding_service.py` | 239 | – | UNSURE | C2 | `app/services/comprehension_service.py:45`, `app/services/memory_service.py:19` | 1 / 19 |
| `services/evaluation_service.py` | 908 | – | UNSURE | C1 | `app/controllers/evaluation.py:27`, `app/eval/runner.py:82` | 4 / 197 |
| `services/event_publisher.py` | 436 | – | ARCHIVE | B9 | `app/services/council/deliberation_service.py:41`, `app/services/instance_manager.py:32` | 1 / 54 |
| `services/health_aggregator.py` | 657 | – | KEEP | – | `app/controllers/health.py:35` | 2 / 48 |
| `services/instance_manager.py` | 647 | – | ARCHIVE | B9 | `app/controllers/instances.py:16`, `app/services/agent_registry.py:38` | 4 / 59 |
| `services/knowledgebase_service.py` | 1098 | C | KEEP | – | `app/controllers/chat.py:17`, `app/controllers/knowledgebase.py:14` | 4 / 78 |
| `services/llm_provider.py` | 74 | – | KEEP | – | `app/controllers/core_entry.py:31`, `app/services/agent_factory_service.py:46` | 0 / 0 |
| `services/memory_service.py` | 624 | – | UNSURE | C2 | `app/controllers/memory.py:15`, `app/main.py:49` | 1 / 34 |
| `services/metrics_service.py` | 187 | C | KEEP | – | `app/controllers/knowledgebase.py:671`, `app/services/knowledgebase_service.py:153`, `app/services/ollama_embeddings.py:94` (+1) | 1 / 13 |
| `services/model_router.py` | 591 | – | UNSURE | C3 | `app/controllers/admin.py:36`, `app/controllers/admin.py:473`, `app/services/catalyst_service.py:22` (+1) | 3 / 72 |
| `services/ollama_embeddings.py` | 168 | C | KEEP | – | `app/services/knowledgebase_service.py:16` | 2 / 29 |
| `services/output_validator.py` | 215 | – | UNSURE | C1 | `app/services/evaluation_service.py:697` | 2 / 22 |
| `services/spawn_template_service.py` | 585 | – | ARCHIVE | B8 | `app/controllers/spawn_templates.py:21` | 1 / 53 |
| `services/task_router.py` | 615 | – | ARCHIVE | B3 | `app/controllers/tasks.py:37` | 1 / 25 |
| `services/title_generation_service.py` | 109 | C | KEEP | – | `app/controllers/chat.py:18` | 1 / 20 |
| `services/webhook_service.py` | 500 | P | KEEP | – | `app/controllers/admin.py:35`, `app/controllers/engine.py:33`, `app/core/health.py:374` (+1) | 2 / 58 |
| `utils/errors.py` | 256 | – | UNSURE | – | – | 1 / 39 |
| `utils/json_repair.py` | 196 | P | KEEP | – | `app/core/agents/comprehension_agent.py:21`, `app/core/agents/evaluation_agent.py:21`, `app/core/agents/orchestration_agent.py:17` | 1 / 34 |
| `websocket_manager.py` | 334 | – | ARCHIVE | B9 | `app/controllers/communication.py:16`, `app/core/health.py:195`, `app/main.py:41` (+3) | 1 / 26 |


## Appendix B. Test disposition

Every test file that references an archived module. MOVE = `git mv` to `archive/backend/tests/` in that batch; EDIT = stays, with the stated edits. Batch is the earliest batch whose removal would break the file.

| Test file | Cases | Disposition | Batch | Note |
|---|---:|---|---|---|
| `test_consciousness_lineage.py` | 5 | MOVE | B6 | (repo-root script, not collected by CI: `testpaths = [tests, app]`) |
| `tests/test_agent_factory_service.py` | 32 | MOVE | B8 | |
| `tests/test_agent_mcp_service.py` | 21 | MOVE | B8 | |
| `tests/test_agent_models.py` | 30 | MOVE | B8 | |
| `tests/test_agent_orchestrator.py` | 19 | MOVE | B1 | also patches/imports B9 modules |
| `tests/test_agent_registration.py` | 22 | MOVE | B9 | |
| `tests/test_agent_registry.py` | 33 | MOVE | B9 | |
| `tests/test_agent_response_service.py` | 19 | MOVE | B7 | |
| `tests/test_audit.py` | 28 | MOVE | C3 | |
| `tests/test_audit_service.py` | 11 | MOVE | C3 | |
| `tests/test_bus.py` | 42 | MOVE | B4 | |
| `tests/test_bus_models.py` | 39 | MOVE | B4 | |
| `tests/test_bus_service.py` | 50 | MOVE | B4 | |
| `tests/test_bus_triggers.py` | 23 | MOVE | B4 | |
| `tests/test_catalyst_service.py` | 30 | MOVE | B5 | |
| `tests/test_communication.py` | 25 | MOVE | B7 | |
| `tests/test_communication_pagination.py` | 15 | MOVE | B7 | |
| `tests/test_communication_service.py` | 6 | MOVE | B7 | |
| `tests/test_comprehension.py` | 52 | MOVE | C1 | |
| `tests/test_comprehension_models.py` | 23 | MOVE | C1 | |
| `tests/test_comprehension_service.py` | 35 | MOVE | C1 | |
| `tests/test_consciousness_backup.py` | 6 | MOVE | B6 | |
| `tests/test_consciousness_backup_service.py` | 18 | MOVE | B6 | |
| `tests/test_consciousness_council_bridge.py` | 28 | MOVE | B6 | |
| `tests/test_consciousness_lineage_service.py` | 25 | MOVE | B6 | |
| `tests/test_controller_auth.py` | 21 | **EDIT** | B7 | delete `TestCommunicationAuth` (L86-158, 8 tests) + `TestDiscordAuth` (L166-229, 10); keep `TestChatStreamAuth` (3) |
| `tests/test_core_state_models.py` | 33 | **EDIT** | B8 + C1 | delete `TestPersonalityTraitsValidator` (L294-328, 6) with B8 and `TestComprehensionInputValidator` (L336-366, 4) with C1, plus imports L29-30; keep the `core_state` classes (23) |
| `tests/test_council_deliberation_service.py` | 58 | MOVE | B5 | |
| `tests/test_council_deliberation_telemetry.py` | 7 | MOVE | B5 | also patches/imports B9 modules |
| `tests/test_council_models.py` | 37 | MOVE | B5 | |
| `tests/test_council_repository.py` | 3 | MOVE | B5 | |
| `tests/test_discord_bridge.py` | 28 | MOVE | B7 | |
| `tests/test_discord_bridge_service.py` | 27 | MOVE | B7 | |
| `tests/test_discord_config.py` | 13 | MOVE | B7 | |
| `tests/test_discord_controller.py` | 4 | MOVE | B7 | |
| `tests/test_embedding_service.py` | 19 | MOVE | C2 | |
| `tests/test_eval_benchmark_floor.py` | 3 | MOVE | C1 | |
| `tests/test_eval_cases.py` | 12 | MOVE | C1 | |
| `tests/test_eval_protocol.py` | 12 | MOVE | C1 | |
| `tests/test_eval_runner.py` | 11 | MOVE | C1 | |
| `tests/test_eval_validation_wiring.py` | 4 | MOVE | C1 | |
| `tests/test_evaluation.py` | 69 | MOVE | C1 | |
| `tests/test_evaluation_models.py` | 36 | MOVE | C1 | |
| `tests/test_evaluation_service.py` | 73 | MOVE | C1 | |
| `tests/test_evaluation_service_pure.py` | 51 | MOVE | C1 | |
| `tests/test_event_publisher.py` | 54 | MOVE | B9 | |
| `tests/test_health_aggregator.py` | 26 | **EDIT** | B3/B4/B9 | drop `check_task_queue` patch (L515, L573) in B3, `check_bus_queue` (L511, L569) in B4, `TestCheckWebsocketManager` (L367-384, 1 test) + `check_websocket_manager` patch (L503, L561) in B9 |
| `tests/test_health_alerts.py` | 12 | MOVE | C3 | |
| `tests/test_instance_manager.py` | 18 | MOVE | B9 | |
| `tests/test_instance_manager_telemetry.py` | 6 | MOVE | B9 | |
| `tests/test_knowledge_attribution_verification.py` | 14 | **EDIT** | B9 | `_make_test_instance` (L49-60) imports `AgentInstance`; used by tests at L97 and L193 and patches `instance_repository.get_instance` (L103). Rewrite those 2 tests to match the decision on D2 |
| `tests/test_mcp_health.py` | 13 | MOVE | C3 | |
| `tests/test_memory_service.py` | 34 | MOVE | C2 | |
| `tests/test_metrics_service.py` | 13 | **EDIT** | B9 | delete 3 `InstanceManager._calculate_cpu_percent` tests (L196, L221, L242); keep 10 |
| `tests/test_mmcnc_models.py` | 28 | MOVE | B2 | |
| `tests/test_model_metrics.py` | 25 | MOVE | C3 | |
| `tests/test_model_router.py` | 45 | MOVE | C3 | |
| `tests/test_model_router_lmstudio.py` | 8 | MOVE | C3 | |
| `tests/test_model_selection.py` | 19 | MOVE | C3 | |
| `tests/test_output_validator.py` | 18 | MOVE | C1 | |
| `tests/test_rbac.py` | 28 | **EDIT** | C3 | delete `test_admin_keys_route_uses_require_role_admin` (L355-379, imports `controllers.admin`); keep the rest |
| `tests/test_sandbox.py` | 23 | MOVE | B9 | |
| `tests/test_sandbox_models.py` | 76 | MOVE | B9 | |
| `tests/test_spawn_template_models.py` | 29 | MOVE | B8 | |
| `tests/test_spawn_template_service.py` | 53 | MOVE | B8 | |
| `tests/test_task_models.py` | 29 | MOVE | C1 | |
| `tests/test_task_router.py` | 25 | MOVE | B3 | also patches/imports B9, C1 modules |
| `tests/test_ws_events.py` | 35 | MOVE | B9 | |
| `tests/test_consciousness_exploration.py` | 0 | MOVE | B6 | integration-marked httpx script; no static `app` import |


**Stay untouched** (reference only KEEP modules, or no `app` module; file (cases)): `test_chat.py` (0), `test_exploration_sync.py` (0), `test_knowledge_attribution.py` (1), `test_sync.py` (0), `tests/conftest.py` (0), `tests/test_ambiguity_detection.py` (21), `tests/test_batch_embedding.py` (16), `tests/test_chat.py` (37), `tests/test_chat_service.py` (13), `tests/test_circuit_breaker.py` (19), `tests/test_comprehension_agent.py` (26), `tests/test_comprehension_state.py` (20), `tests/test_config_models.py` (38), `tests/test_conversation_node.py` (16), `tests/test_conversation_pagination.py` (11), `tests/test_core_errors.py` (39), `tests/test_core_graph_routing.py` (14), `tests/test_core_state.py` (44), `tests/test_correlation_middleware.py` (15), `tests/test_evaluation_agent.py` (24), `tests/test_health_history.py` (22), `tests/test_in_memory_rate_limiter.py` (16), `tests/test_intelligence_log.py` (19), `tests/test_json_repair.py` (34), `tests/test_jwt_auth.py` (26), `tests/test_knowledgebase_ocr.py` (17), `tests/test_knowledgebase_service.py` (31), `tests/test_local_provider_switch.py` (7), `tests/test_logging_config.py` (17), `tests/test_middleware.py` (18), `tests/test_ollama_embeddings.py` (13), `tests/test_orchestration_agent.py` (24), `tests/test_parallel_execution.py` (16), `tests/test_pipeline_output_models.py` (23), `tests/test_prometheus_metrics.py` (15), `tests/test_rate_limit.py` (58), `tests/test_reasoning_agent.py` (18), `tests/test_redis_rate_limiter.py` (12), `tests/test_rubric_evaluation.py` (19), `tests/test_security.py` (61), `tests/test_startup_validator.py` (21), `tests/test_telemetry.py` (20), `tests/test_title_generation_service.py` (20), `tests/test_tool_dispatcher.py` (34), `tests/test_webhook_service.py` (46) Note: `tests/test_core_errors.py` (39) covers the UNSURE module `utils/errors.py` (decision D5)..
