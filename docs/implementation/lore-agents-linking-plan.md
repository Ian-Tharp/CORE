# Linking Modular Agents to World Lore Generation — Implementation Plan

**Date:** 2026-06-02
**Status:** Phases 1–4 (foundation) **implemented** in `d21c54d`; Phases 5–8 pending (coordinate with the parallel worker). Produced by a mapping workflow (understand → design → critique).

**Decisions locked:** per-**world** binding · backend **`role` filter** for the lore-agent list · model-override via a **one-off uncached** factory instance (no cache mutation).

**Progress:**
- ✅ **Phase 1** — `model` field on `AgentConfig`/create/update; `agents.model` column + idempotent ALTER; `world_lore_agents` binding table; `model` through repo SELECTs + INSERT.
- ✅ **Phase 2** — shared `llm_provider.build_chat_model` (provider-aware); `core_entry` repointed with thin wrappers (no behavior change).
- ✅ **Phase 3** — factory builds each agent's LLM from `config.model` (→ env → fallback) via `build_chat_model`.
- ✅ **Phase 4** — `lore_service.persist_lore_page()` + `lore_user_prompt()` extracted; `generate_lore_page` unchanged (fallback).
- ⏳ **Phase 5** (wire `generate_lore`) — **blocked on coordination** (parallel worker owns the file).
- ⏳ **Phases 6–8** — binding repo/routes + role filter, frontend (ModelsService, pickers), tests.

## Goal

Make the modular **Agent Factory** agents act as the **world lore agents** in the Generative Procedural Worlds command center, with a **selectable, provider-aware model per agent**, and surface all of that **configuration in the UI** (pick a lore agent + its model per world; configure an agent's model in the builder).

## Current state

- **Lore is generated inline**, not by modular agents: `lore_service.generate_lore_page()` calls `_llm_or_stub` (core_entry) with model = `gpt-4o-mini` (if `OPENAI_API_KEY`) else `CORE_DEFAULT_MODEL`. It persists a `wiki_pages` row tagged `{template: kind, source: 'ai'}`.
- The **Agent Factory** builds each agent's LLM with a **hardcoded `gpt-4o-mini`** (`agent_factory_service._create_llm_for_agent`). `AgentConfig` has **no `model` field**; agents aren't model-selectable.
- A **new world-agent workflow is in progress** (owned by a parallel worker): `world_agent_workflow_service.py` (`build_context`, `generate_lore`, `audit_lore`, `WorldAgentLoreResponse`), `spawn_template_service.py`, `spawn-templates.service.ts`, and the `/worlds/{id}/agents/lore` route. `generated_by` is currently a hardcoded `'world_lore_architect'` label with **no backing `AgentConfig`**.
- Provider-agnostic model routing already exists (`_is_openai_model` / `_local_endpoint` in core_entry; `CORE_LOCAL_PROVIDER` switch) but is **not shared** with the factory.
- The frontend already has a **per-step model dropdown** in the engine playground (loads models from the backend) — the pattern to reuse. The world-detail-panel lore section hardcodes `agent_id: 'world_lore_architect'`. The agent builder's model list is hardcoded and **Deploy is stubbed**.

## Target architecture

```
World Detail Panel: [lore agent ▼] [model ▼] → generateAgentLore({agent_id, model})
        │ (binding persisted per world: GET/PUT /worlds/{id}/lore/agent)
        ▼
worlds.py /agents/lore → WorldAgentWorkflowService.generate_lore
        │  build_context (existing) → resolve model precedence
        │  request.model > per-world binding > agent.config.model > CORE_DEFAULT_MODEL
        ▼
AgentFactoryService.get_agent(agent_id)  →  llm_provider.build_chat_model(model, …)   ← shared provider-aware builder
        │  agent.ainvoke(system=agent.system_prompt|LORE_SYSTEM, user=lore_prompt(context))
        ▼
lore_service.persist_lore_page(world_id, kind, text)  → wiki_pages (unchanged shape)
```

Key idea: **don't build a second pipeline.** Finish wiring the in-progress world-agent workflow to a real modular agent, extract a shared `llm_provider`, add a `model` field through the agent stack, and reuse the existing model-dropdown UI pattern.

## Phased task plan (de-risked build order)

Ordered so the low-conflict foundation lands first and the high-collision file (`world_agent_workflow_service.py`, owned by the parallel worker) is touched last, as a single coordinated edit.

**Phase 1 — Data plumbing (low conflict, unblocks everything)**
1. `agent_models.py` — add `model: Optional[str]` to `AgentConfig`, `AgentCreateRequest`, `AgentUpdateRequest`.
2. `dependencies.py` — add `model VARCHAR(128)` to the `agents` CREATE + idempotent `ALTER TABLE agents ADD COLUMN IF NOT EXISTS model`; create the `world_lore_agents` binding table (`world_id` PK, `agent_id`, `model`, `updated_at`).
3. `agent_repository.py` — add `model` to the SELECT lists (get/list/search) and to `create_agent` INSERT (the 17→18 param change must update column list + placeholder + args tuple in lockstep).

**Phase 2 — Shared provider-aware LLM builder (highest blast radius — ship as no-behavior-change refactor with tests green)**
4. NEW `services/llm_provider.py` — port `_is_openai_model`/`_local_endpoint` + a `build_chat_model(model, temperature, top_p)` (incl. the gpt-5 temperature exception + one-retry-without-temp fallback, verbatim).
5. `core_entry.py` — repoint `_llm_or_stub`/`_is_openai_model`/`_local_endpoint` to import from `llm_provider`, keeping thin wrappers (lore_service imports `_llm_or_stub` from here — preserve the symbol).

**Phase 3 — Factory provider-aware + model override**
6. `agent_factory_service.py` — `_create_llm_for_agent` resolves `model = config.model or CORE_DEFAULT_MODEL or 'gpt-4o-mini'` via `build_chat_model`. **Decide the override mechanism:** build a one-off **uncached** instance from a cloned config with overridden `.model` (never mutate the cached config — avoids cache poisoning).

**Phase 4 — Lore service seam**
7. `lore_service.py` — extract `persist_lore_page(world_id, kind, text)` (title extraction + `create_wiki_page` tail) and export `LORE_SYSTEM` + a `lore_user_prompt(...)`. Keep `generate_lore_page` byte-identical for the fallback path.

**Phase 5 — Wire the workflow (COORDINATE — parallel worker owns this file)**
8. `world_agent_workflow_service.py` — in `generate_lore`: resolve model precedence; if `agent_id` resolves, load the factory agent, `ainvoke` on the `build_context` summary, persist via `persist_lore_page`; fall back to `generate_lore_page` when `agent_id` is unknown. Add `model` to `WorldAgentLoreResponse`. Set `generated_by` to the real `agent_id`/`agent_name`.

**Phase 6 — Binding repo + routes + safety**
9. `world_repository.py` — `get_world_lore_agent` / `set_world_lore_agent` (UPSERT).
10. `worlds.py` — `GET`/`PUT /worlds/{id}/lore/agent` (with `require_api_key`, matching the agents controller); fill missing `agent_id`/`model` from the stored binding in the agents-lore route. Add a **timeout/cancellation wrapper** around the LLM call before exposing slow local models.
11. `main.py` (lifespan) — **one owner** seeds a real `world_lore_architect` agent (spawn `tmpl-world-lore-architect`, idempotent). Recommend the parallel worker (who owns `spawn_template_service`) owns this; this work only consumes it.

**Phase 7 — Frontend, bottom-up**
12. NEW `services/models.service.ts` — `listModels()` merging `GET /local-llm/models` + `GET /admin/models` (extract the engine-playground fetch logic).
13. `engine-playground.component.ts` — refactor `_loadModels` to use `ModelsService` (pure refactor).
14. `agent-library.service.ts` — `createAgent`/`updateAgent` (incl. `model`, **with the API-key header**); map `model` in `mapBackendAgentToLibrary`.
15. `models/agent.models.ts` — add `llmModel?: string` to `LibraryAgent` (distinct from the existing pricing-tier field).
16. `agent-builder.component.ts` — model dropdown from `ModelsService`; implement `deployAgent()` → `createAgent`/`updateAgent` (defer until backend write-auth confirmed via curl).
17. `worlds.service.ts` — `getLoreAgentBinding`/`setLoreAgentBinding`; `generateAgentLore` already carries `agent_id`+`model`.
18–19. `world-detail-panel.component.ts/.html` — add an **agent picker** (lore-capable agents) + **model dropdown** above the three lore buttons; load + persist the per-world binding; show the active model + `generated_by`.

**Phase 8 — Tests**
20. `test_world_agent_workflow_service.py` — agent-loaded-and-invoked, model-precedence, unknown-agent fallback; `llm_provider.build_chat_model` routing tests. (Coordinate — parallel worker owns this file.)

## Decisions to make before coding (gating)

1. **Binding granularity** — one lore agent+model **per world**, or **per lore-kind** (Overview/History/Peoples → composite key)? Plan assumes per-world.
2. **Direct `/lore/generate` path** — keep as an explicit no-agent fallback, or deprecate now that the agent path is primary?
3. **Model-override caching** — one-off **uncached** instance (recommended; avoids poisoning the 5-min agent cache) vs. cache-key-by-model.
4. **Seed owner** — who creates the `world_lore_architect` agent row (this work vs. the parallel spawn worker) to avoid double-seed.
5. **Lore-agent list contract** — filter `getAgents()` by role/tag/interests client-side, or add a backend `/agents?role=world_lore` filter (cleaner contract)?
6. **`audit_lore`** — keep heuristic for v1, or route through a `tmpl-canon-continuity-auditor` agent.
7. **Offline default model** — does `CORE_DEFAULT_MODEL` point at an available local model, so the seeded agent isn't stuck on `gpt-4o-mini` without a key?

## Risks (from the critique)

- **Merge collision (highest):** the parallel worker owns `generate_lore`, `spawn_template_service`, and `test_world_agent_workflow_service.py`; both works edit `worlds.py`. Serialize the `worlds.py` route additions and pair on the `generate_lore` edit.
- **Cache poisoning:** implementing model override by mutating `config.model` then calling `get_agent` leaks the override into the cached instance → use a one-off uncached path.
- **Blast radius:** consolidating provider routing into one module means a misclassified model id breaks **both** lore and every factory agent. Port the routing verbatim; add tests.
- **Slow-model UX:** a 70B local model with no timeout/streaming will appear to hang the panel; users retry → concurrent multi-minute generations against a serialized local server. Add timeout + feedback before shipping the dropdown.
- **Schema/repo drift:** the 17→18 INSERT param change must update column list + placeholder + args tuple together; the `ALTER ADD COLUMN` must run before any seed/create.
- **Dangling bindings:** nothing clears `world_lore_agents` when a world or bound agent is deleted; handle a stale persisted `agent_id` at generate time (not just an unknown request-time id).
- **Response shape:** `WorldAgentLoreResponse.generated_by` is a single string with no `model` field — add `model` or the UI can't show the resolved model.

## Coordination

This work overlaps the in-progress world-agent subsystem. Agree up front on: (a) who edits `generate_lore`, (b) the seed owner, (c) the factory override signature, (d) test-file ownership, and serialize `worlds.py` route additions through one person. Build a thin end-to-end slice first (one agent, one world, default model, no override) to validate the `create_react_agent` `ainvoke` shape before adding model selection, bindings, and overrides.
