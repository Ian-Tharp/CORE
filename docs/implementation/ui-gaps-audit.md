# UI Gaps & Improvements Audit

**Date:** 2026-06-01
**Scope:** Every route in `ui/core-ui/src/app` (16 routes across 5 feature clusters).
**Method:** Parallel code-reading audit of each component's `.ts` / `.html` / `.scss`. Findings are grounded in source; claims about *backend* endpoints existing/not existing are inferred from the UI side and marked **(unverified)** — confirm against `backend/app/controllers/` before acting.

> This is a living backlog. When you fix an item, strike it through or move it to a "Done" note with the commit hash. The landing-page **command deck** is the reference implementation and the styling gold standard — it is intentionally excluded from the "fix" list.

## Resolved (log)

| Date | Item | Commit |
|------|------|--------|
| 2026-06-02 | **World Detail P0** — `MatButtonModule` import, not-found state, back nav, themed SCSS | `f5f1a8c` |
| 2026-06-02 | Missing-Material-module bug sweep — clean (World Detail was the only one) | — |
| 2026-06-02 | **Communication reactions** now persist via existing endpoints (optimistic + rollback) | `4668751` |
| 2026-06-02 | **Engine Playground** SSE subscription leak fixed (`OnDestroy`); confirmed CORE step-stream is real (stale comment removed) | `4668751` |
| 2026-06-02 | **Worlds Grid** — surface saved-world load errors + loading state; `takeUntilDestroyed` | `bb6ca53` |
| 2026-06-02 | **Docs:** Ollama → LM Studio in README + CLAUDE.md | — |
| 2026-06-02 | **Hygiene:** removed dead typing-sim code; engine.service URLs → AppConfigService | — |
| 2026-06-02 | **Docs:** new `architecture/worlds-architecture.md`; indexed; historical logs flagged; `core_graph.py` stub claim verified | `58c894c` |
| 2026-06-02 | **Hygiene:** presence/creative/system-monitor URLs → AppConfigService | `293e1d2` |
| 2026-06-02 | **Hygiene:** deleted ~300 lines dead mock in MessageService; collapsed duplicate `isConnected` | `102a655` |
| 2026-06-02 | **Hygiene:** `.sample-badge` on Agent Marketplace + landing Boards (mock-data screens) | `0219b38` |
| 2026-06-02 | **Docs:** new `architecture/agent-factory-mcp.md` (current-state wiring reference); indexed | — |

## Status legend

| Icon | Meaning |
|------|---------|
| ✅ | Working — wired end-to-end, only polish opportunities |
| 🟡 | Partial — renders and partly works, but has dead actions / missing wiring / mock data |
| 🟦 | Stub / placeholder — UI shell present, data hardcoded or feature not implemented |
| 🔴 | Broken / unwired — fails, or is entirely mocked with no backend path |

## Status at a glance

| Route | Component | Status | Headline gap |
|-------|-----------|--------|--------------|
| `/` | Landing command deck | ✅ | Reference impl — polish only |
| `/command-center` | Command Center (3D galaxy) | 🟡 | "Enter Tile" + AI Observation are stubs; silent metadata-save failures |
| `/boards` (landing) | Boards (calendar) | 🟡 | Mock-only data; menu/action buttons are no-ops |
| `/kanban` | Kanban | ✅ | Updates "local only"; `N` shortcut unwired; local SCSS vars |
| `/conversations` | Conversations page | 🟡 | Duplicate connection logic; hardcoded colors; TODOs to move to service/signals |
| `/conversations` (engine playground) | Engine Playground | 🟦 | Unified CORE step-stream may not exist on backend **(unverified)**; no unsub cleanup |
| `/communication` | Communication | 🟡 | Hardcoded `human_ian` user; ~8 stubbed presence actions; reactions not persisted |
| `/agents` | Agent Builder | 🟡 | File upload, Deploy, Save-as-Template, MCP connect, "Enhance" all unimplemented; test chat is faked |
| `/agents/library` | My Agents | ✅ | Import / Export-All buttons unwired |
| `/agents/marketplace` | Agent Marketplace | 🔴 | Service fully mocked; install is a fake toast |
| `/knowledge` | Knowledgebase | 🟡 | Global tab is empty; "View Details" unwired; inline color hexes |
| `/knowledge-attribution` | Attribution Browser | 🟡 | "View Source" unwired; **all** colors hardcoded; FontAwesome vs Material mismatch |
| `/analytics` | Analytics | 🟦 | All telemetry hardcoded; no analytics service (self-labeled "Sample telemetry") |
| `/tools` | MCP Registry | ✅ | Modal focus trap + Escape; some hardcoded colors |
| `/tools/discord-bridge` | Discord Bridge Dashboard | ✅ | ARIA labels on status pills; div-tables need grid semantics |
| `/creative` | Creative Landing | 🟡 | No hover/focus states; hardcoded colors |
| `/creative/worlds` | Worlds Grid | 🟡 | Subscription leak; remote-load errors swallowed silently |
| `/creative/world/:id` | World Detail | 🟦 | Missing `MatButtonModule` import → unstyled buttons; blank screen on bad id |
| `/creative/wiki` | Wiki | ✅ | Character-gen endpoints may be unwired **(unverified)**; localStorage-only; no undo |
| `/creative/boards` | Creative Boards | 🟡 | Board cards read-only; no detail view; fragile CSS-column masonry |
| `/creative/marketplace` | Creative Marketplace | ✅ | Client-side filtering only; no loading/error state |

---

## Backend reality-check (verified 2026-06-02)

The first pass marked backend claims **(unverified)**. A follow-up audit of `backend/app/controllers/` resolves them. This is the difference between "wire up an existing endpoint" (hours) and "build a backend feature" (days).

### Frontend-wiring-only — backend already exists ✅
| UI gap | Existing endpoint(s) |
|--------|----------------------|
| Communication **reactions not persisted** | `POST /communication/messages/{id}/reactions`, `DELETE …/reactions/{type}` — **just call them** |
| Communication **presence actions** | `GET/PATCH /communication/presence[/{instance_id}]` (presence exists; the dialogs are the missing part) |
| Conversations "move to service / signals" | `GET /conversations/`, `PATCH /conversations/{id}` exist |
| Wiki **character generation** (was "may be unwired") | `POST /creative/characters`, `POST /creative/characters/{id}/image` — **exist exactly as the UI calls them** |
| Engine Playground **step-stream** (was "may not exist") | `POST /engine/run/stream` — **SSE with `thinking/intent/plan/step/result` events already implemented**; the playground just isn't consuming it correctly |
| Knowledgebase Global tab / search | `POST /knowledgebase/semantic-search` + `upload`/`batch-upload` + `files` CRUD exist |

### Needs backend work first ❗
| UI gap | Backend status |
|--------|----------------|
| **Agent Marketplace** (🔴 mocked) | No marketplace/install routes at all — the mock is "honest". Needs backend, or keep behind a clear "preview". |
| **Analytics** (🟦 hardcoded) | Only `GET /system/resources(/stream)` + Prometheus `/metrics` exist — **no cognition-KPI/throughput/insights route**. Either add one or wire the dashboard to system metrics for partial real data. |
| Communication **mark-as-read** | WebSocket-only (`mark_read` handler); no REST endpoint — wire the WS path or add REST. |
| Communication **channel update/delete** | Only create/read exist. |
| **Kanban persist** ("local only") | Task routes exist but no lightweight `PATCH /tasks/{id}` for drag/status edits — needs one. |

> Net: the highest-ROI next fixes are **frontend-only** — reactions persistence and the engine step-stream are both "call the endpoint that already exists."

### Missing-Material-module bug sweep — clean ✅
A full sweep of all standalone components for the World-Detail bug class (Material directives used without importing the module) found **no remaining instances**. World Detail (`f5f1a8c`) was the only one.

---

## Cross-cutting themes

These recur across nearly every cluster and are the highest-leverage fixes because one pattern repays many screens.

### 1. Design-token debt (highest breadth)
Hardcoded hex/rgba instead of `solarpunk-theme.scss` tokens (`--core-*`, `--spacing-*`, `--radius-*`, `--font-*`) in: Boards (landing), Kanban (local SCSS vars), Conversations, Communication, Attribution Browser (**every** color), Knowledgebase (inline `style="color:#…"`), Agent Builder, and all of `creative-design-product/*`. Consequence: theme changes don't propagate; two different "ambers" already exist (Kanban `#f59e0b` vs deck `--core-amber`). **A token sweep is the single biggest consistency win.**

### 2. Accessibility floor (AA) not yet met broadly
- Non-semantic clickable `div`s without `role="button"` / `tabindex` (Agent Builder agent-type cards, Knowledgebase file cards).
- Emoji-as-icon with no `aria-label` (Wiki, Creative Boards, view-mode toggles).
- Missing `focus-visible` rings on interactive elements across Kanban, creative/*, Agent filter inputs.
- Modal focus management: MCP Registry modal doesn't trap focus or close on Escape.
- Form inputs without associated `<label>` (Creative Boards, Marketplace, Wiki tag editor).
- Status badges/pills as styled `div`s with no `role="status"`/`aria-label` (Communication presence, Discord bridge).

### 3. Responsiveness below ~1100px
Fixed multi-column grids and fixed-width sidebars with no (or one) breakpoint: Boards landing (`340px auto 280px`), Command Center detail panel <800px, Wiki 22rem sidebar, creative landing auto-fit collapse. No hamburger/drawer pattern anywhere in creative/*.

### 4. Mock data / unwired backend (feature-blocking)
- **Agent Marketplace** — service is 5 hardcoded agents; `installAgent()` is a fake 1s toast. 🔴
- **Analytics** — KPIs/throughput/insights are literal arrays; no service. 🟦
- **Boards (landing)** — hardcoded tasks/events; no API. 🟡
- **Communication MessageService** — `getMockMessages()` per channel id; reactions and `markAsRead` not persisted.
- **Agent Library mapping** — `mapBackendAgentToLibrary()` hardcodes `category`, `rating: 4.5`, `downloads/size: 0`, `releaseDate: now`.
- **Creative `*`** — persistence is `CreativeDataService` (localStorage) only; lost on browser clear.

### 5. Missing async states & silent failures
Loading spinners/skeletons and error toasts are inconsistently present. Notable silent failures: Command Center metadata save (console only), Worlds Grid remote-load `catchError(() => of(null))`, Communication send error (TODO to surface), Channel/Marketplace API errors.

### 6. Subscription / lifecycle leaks
Worlds Grid (constructor subscribe, no `OnDestroy`), Engine Playground (`_subs` never torn down), PresenceService heartbeat interval never unsubscribed.

---

## Cluster detail

### A. Command surface — landing deck, command-center, boards, kanban

**Command Center (`/command-center`) 🟡**
- 🔴/stub: `onEnterWorld()` only sets `isLoading` for 2s with no navigation.
- Stub: AI Observation returns a hardcoded "placeholder response … connect to CORE cognitive engine" string — not wired.
- Silent fail: `worlds.saveMetadata` error is console-only; no user feedback.
- `Ctrl+K` search hint shown but no `@HostListener` for it; connection-type fixed to `'alliance'` with no selector UI.
- No WebGL/engine init error boundary; no loading state during `loadWorldById`; world-label `div` is `aria-hidden` yet semantic.
- Improvements: undo/redo for grid edits; metadata autosave; richer tile context menu (Edit/Delete/Duplicate); grid-config presets; visual affordance during connection creation.

**Boards (landing `/boards`) 🟡**
- Mock-only tasks/events (no API, unlike Kanban). "Add Task", "Edit", "Reschedule", "Delete", "Export", and calendar menu items (Today/Week/Month, "Kanban View") have **no handlers**.
- `toggleTaskStatus` mutates local state only; priority colors hardcoded hex; no responsive breakpoints; type hack `'maintenance' as any`.

**Kanban (`/kanban`) ✅**
- Updates are "local only" (snackbar says so) — not persisted **(verify backend)**; `N` shortcut shown but unwired; hardcoded `projects` list may never match service data; local SCSS variables instead of shared tokens; drag handles lack ARIA.

### B. Conversations & communication

**Conversations page (`/conversations`) 🟡** — duplicate `isConnected` logic; hardcoded `localhost:8001` assumption in error copy; hardcoded status colors; in-file TODOs to (a) use `AppConfigService`, (b) move fetching into `ConversationsService`, (c) convert to signals, (d) add retry/backoff + diagnostics.

**Engine Playground 🟦** — hardcoded model list (`gpt-5`, `gpt-4.1`…) misleading on first render before real models load; `_subs` never unsubscribed (leak); blindly reads `coreState.intent/plan/step_results/...` with no null guards; in-code TODO: "Backend needs to implement true step-by-step graph execution with streaming" — **step-stream likely not implemented (unverified)**; typing simulation commented out while indicators still show.

**Communication (`/communication`) 🟡** — `human_ian` hardcoded across Message/Presence/Channel services (no auth); presence actions `viewProfile/viewConsciousnessState/viewAgentCapabilities/requestAgentTask/inviteToChannel/viewSharedContext/initiateCollaboration` are all `console.log` stubs with live buttons; reactions optimistic-only (no `addReaction`/`removeReaction` call); `markAsRead` backend call commented out; search-scroll matches by text substring (fragile — needs message ids); send errors not surfaced; threads reconstructed from cache; SCSS hardcoded colors and no `prefers-reduced-motion`. ✅ sub-parts: WebSocketService (reconnect/ping solid), MessageRenderer (markdown/sanitize) — minor: no code-block line numbers / copy button.

### C. Agents

**Agent Builder (`/agents`) 🟡** — `onDrop()` file upload is a placeholder log; `deployAgent()` / `saveAsTemplate()` only `console.log` (cannot save agents); test chat uses keyword-matched canned responses (not real inference); `toggleMcpServer()` flips UI only (no connect/validate); "Enhance with AI" / "Add Safety Guidelines" buttons have no handlers; agent-type cards are non-semantic divs; extensive hardcoded colors; thin form validation.

**My Agents (`/agents/library`) ✅** — Import and Export-All buttons unwired; empty state fixed sizing; client-side filter re-fetches all (no pagination).

**Agent Marketplace (`/agents/marketplace`) 🔴** — `agent-marketplace.service` loads 5 mock agents; `installAgent()` returns a fake toast; "Copy container image" has no handler; missing `.status-*` badge CSS; no retry on load error; filters not persisted.

**Shared/services** — Agent Card health divides by hardcoded `50`; "View Logs" action has no handler; Library service mapping hardcodes rating/category/downloads/size/releaseDate; Detail Drawer is read-only (no edit mode).

### D. Knowledge & analytics

**Knowledgebase (`/knowledge`) 🟡** — **Global tab is structurally empty** (controls/grid are placeholder comments); list-view "View Details" menu item unwired; inline `style="color:#…"` on action buttons; Material `white` info-card background off-theme; file cards lack `role="button"`/`aria-pressed`; semantic-search dialog lacks focus management.

**Attribution Browser (`/knowledge-attribution`) 🟡** — "View Source" unwired; **entire SCSS hardcoded** (~50 colors, light-theme `#1a202c`/`white` — clashes with dark solarpunk); uses **FontAwesome** while siblings use Material Icons; `setTimeout(…,500)` mock delay; partial date-range filters silently ignored; checkboxes/date inputs unlabeled.

**Analytics (`/analytics`) 🟦** — production-quality layout/animation but **all data hardcoded signals**; no analytics service, polling, or streams; self-labeled preview tag "Sample telemetry — live analytics feed not yet wired"; sparkline needs a data-table alternative; trend arrows have no text fallback. **Highest-value net-new wiring target.**

### E. Tools & creative

**MCP Registry (`/tools`) ✅** — modal `tabindex="-1"` blocks return focus + no Escape close; possible missing `MatIconModule` for template icon names **(verify)**; some hardcoded colors; single breakpoint.

**Discord Bridge (`/tools/discord-bridge`) ✅** — solid polling/validation; needs ARIA labels on status pills + tab panels; div-based tables should be real `<table>`/`role="grid"`; no error overlay on mid-stream API failure.

**Creative Landing (`/creative`) 🟡** — cards lack hover/focus states + aria-labels; hardcoded colors; no responsive breakpoints; no skip-nav.

**Worlds Grid (`/creative/worlds`) 🟡** — constructor subscription leak (no `OnDestroy`); remote errors swallowed (`of(null)`); no loading indicator; fixed limit 24, no pagination; remote worlds unsorted.

**World Detail (`/creative/world/:id`) 🟦** — **missing `MatButtonModule` import** → `mat-*-button` render unstyled; no null-check on `projects.load(id)` → blank screen on bad id; empty SCSS; unsafe `w.layers.terrain.length` access; no "Back to Worlds".

**Wiki (`/creative/wiki`) ✅(gaps)** — rich features; `createCharacter()`/`generateCharacterImage()` may be unwired **(unverified)**; localStorage sync writes (no debounce); drag-drop missing `dragover` binding; no undo/redo; emoji icons + div context-menu need ARIA/`role="menu"`; fixed 22rem sidebar not responsive; hardcoded colors.

**Creative Boards (`/creative/boards`) 🟡** — board cards read-only (no detail/edit/delete); input not in a `<form>` (Enter doesn't submit) and unlabeled; fragile `column-count` masonry; no empty state; no button hover/focus.

**Creative Marketplace (`/creative/marketplace`) ✅** — no loading/error state; client-side filtering of full set; fixed limit 60; `::before { content:'Marketplace' }` heading is inaccessible; generic delete confirm lacking world name; hardcoded colors.

---

## Prioritized backlog

### P0 — Breaking / data-loss / "looks broken"
1. **World Detail** — add `MatButtonModule` import; null-check the loaded world and redirect on miss; guard `w.layers?.terrain?.length`.
2. **Command Center** — surface metadata-save failures (toast); replace the "Enter Tile" 2s stub with real behavior or hide it; remove/disable the AI-Observation placeholder until wired.
3. **Lifecycle leaks** — add `OnDestroy`+unsubscribe to Worlds Grid, Engine Playground (`_subs`), PresenceService heartbeat.
4. **Agent Builder** — wire `deployAgent()` to a real POST (or clearly disable + label "coming soon"); stop presenting faked test-chat as real inference.
5. **Knowledgebase** — fill the empty Global tab (extract shared controls/grid sub-component) or hide the tab until built.

### P1 — High-value (unblock features / meet AA floor)
1. **Auth/user context** — replace hardcoded `human_ian` with a shared `UserContextService` (Message/Presence/Channel).
2. **Reaction persistence** — ✅ backend exists; just call `POST/DELETE /communication/messages/{id}/reactions`. (`markAsRead` is WS-only — wire the WS path or add a REST route.) **Top frontend-only quick win.**
3. **Engine Playground step-stream** — ✅ `POST /engine/run/stream` (SSE) exists; fix the client to consume `thinking/intent/plan/step/result` events instead of showing mock/typing. Frontend-only.
4. **Analytics service** — ❗ needs a backend cognition-KPI route (only `system/resources` + Prometheus exist today); or wire the dashboard to system metrics for partial real data. Then add loading/error/poll and remove the sample tag.
5. **Agent Marketplace** — ❗ no backend marketplace/install routes exist; either build them or gate the UI behind a clear "preview" banner instead of a fake install toast.
6. **Accessibility pass** — `role="button"`+`tabindex`+`focus-visible` on clickable divs/cards; `aria-label` on emoji/icon buttons and status pills; focus-trap + Escape on MCP modal; labels on creative form inputs.
7. **Wire dead buttons** — My Agents Import/Export-All, Knowledgebase "View Details", Attribution "View Source", Agent Builder "Enhance"/file-upload, Creative Boards detail nav.

### P2 — Polish & consistency
1. **Design-token sweep** — Boards, Kanban, Conversations, Communication, Attribution (full), Knowledgebase inline styles, Agent Builder, creative/* → `--core-*`/`--spacing-*`/`--radius-*`. Unify the two ambers.
2. **Responsiveness** — breakpoints + drawer pattern for Boards, Command Center detail panel, Wiki sidebar, creative landing.
3. **Iconography** — migrate Attribution Browser FontAwesome → Material Icons; confirm `MatIconModule` everywhere icons are template-driven.
4. **Async UX** — skeletons/spinners + retry actions on Worlds Grid, Marketplace(s), Conversations, Communication; debounce Wiki localStorage writes.
5. **Loose ends** — Kanban `N` shortcut + persisted updates; masonry → CSS Grid (Creative Boards/Marketplace); message-renderer copy button + line numbers; semantic tables for Discord bridge; reduced-motion guards where animations lack them.

---

## Code & docs hygiene backlog (added 2026-06-02)

A dedicated sweep for TODOs, stale comments, dead code, mock data, and out-of-date docs. ✅ = fixed in this pass.

### Docs staleness
- ✅ **Ollama → LM Studio** — `README.md` (tech-stack table + ASCII diagram) and `CLAUDE.md` said "Ollama for local LLM" only; now "Ollama or LM Studio (`CORE_LOCAL_PROVIDER`)".
- **Missing feature docs** — no docs for the world-creation studio (worlds/lore/art/character gen, `world_metadata`/`world_assets`, `lore_service`), the command-center 3D galaxy, or world-scoped RAG. Add a `docs/architecture/worlds-architecture.md` + a `## Creative Design & Worlds` section in `docs/README.md`.
- **Historical logs not marked** — `docs/council/outputs/*` (implementation_roadmap dated 2026-01-28, vision_session, dockerization_deliberation) read as current guidance; add "historical RSI session output" headers pointing to `docs/roadmap/` for current priorities.
- ✅ **Verified (claim stands)** — `docs/roadmap/command-deck-cognition-next-steps.md` calls `core_graph.py` "a non-functional stub"; confirmed still a 56-line skeleton (5 nodes wired but stub bodies, `compile()` result unused, `intialize_graph` typo). No doc change needed — the graph itself is the real backlog item. Note: the `/engine/run/stream` path streams real graph *traversal/events*, but the nodes it traverses are these stubs.
- **Out-of-scope content** — `docs/deployment/docker.md` has a "Consciousness-Hosting Capabilities" digression; move to a consciousness doc.
- **Index integrity** — `docs/CORE/README.md` is a near-empty entry; expand or drop from the index.

### Code hygiene
- ✅ **Stale "not implemented" comment** — engine-playground claimed the CORE run "simulates execution / backend needs to implement streaming"; it already streams real graph execution. Comment corrected (`4668751`).
- ✅ **Dead code** — removed the no-op `simulateTypingResponse()` + its misleading caller in `communication.component.ts` (commented-out body, "simulate a random instance typing back").
- ✅ **Hardcoded URLs (engine)** — `engine.service.ts` had 6 hardcoded `http://localhost:8001/...`; now routed through `AppConfigService` (`api`/`engineApi` getters).
- ✅ **Hardcoded URLs (presence/creative/system-monitor)** — routed through `AppConfigService` (`293e1d2`).
- **Hardcoded URLs (remaining)** — `worlds.service.ts`, `spawn-templates.service.ts`, `chat-window.component.ts:97` still hardcode `http://localhost:8001`. Left intentionally — under active edit by a parallel worker; sweep once those land.
- **Mock data presented as real** — ✅ `message.service.ts` mock methods were **dead** (no caller) → deleted; ✅ Agent Marketplace + landing Boards now carry a `.sample-badge` "preview" indicator. ✅ `instance.service.ts` `getTaskSummary()` returns **all-zeros** (an honest empty state, not misleading sample data) — no badge needed; wiring it to the real `GET /tasks` aggregation is a future *enhancement*, not hygiene.
- ✅ **Duplicate logic** — `conversations-page.component.ts` `isConnected` computed once now (was twice, with a buggy `data.length > 0` clause).
- **`mark-as-read` TODO** — `channel.service.ts:57` "Call backend API to mark messages as read": REST route doesn't exist (WS-only) — wire the WS path or add the route.
- **8 stubbed presence actions** — `communication.component.ts:672–707` (`viewProfile`, `viewConsciousnessState`, `requestAgentTask`, `inviteToChannel`, …) are `console.log` placeholders with live buttons; implement or hide.

### TODO triage (keep vs resolve)
- **Backend TODOs are mostly valid & blocked** (not stale): `agent_factory_service` personality→params, `agent_mcp_service` registry loading, `task_router` reassign/cache, `comprehension_service` tool-registry integration, `main.py` `/api/v1/` prefix. Leave in place; track in issues.
- **`voice_registry.py` `TODO_GENERATOR`/`TODO_EVALUATOR`** are **not** TODO markers — they're "to-do list" council voice definitions. Ignore (false positive).
- **Frontend RSI TODOs** (config externalization, signals migration, per-conversation stream maps, OpenAPI-typed clients) are valid hardening items, not stale.

## How this was produced
Five parallel read-only audits (one per cluster) over the live component source on branch `develop`, 2026-06-01. Findings reflect code at that point; re-run before a big push since several files are under active iteration. Backend-existence notes are **UI-inferred** — confirm against `backend/app/controllers/` and the OpenAPI surface before building against them.
