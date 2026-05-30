# Command Deck & Cognition — Next Steps

_Last updated: 2026-05-30_

Status and prioritized backlog following the command-deck rebuild and the
autonomous UI-polish pass. This is the "what to do next" companion to the
[UI Polish Log](../ui-polish-log.md).

## Where we are now

- **Command Deck** (`landing-page`) rebuilt as the home surface (top tabs removed;
  Tasks/Agents/Boards moved to sidebar + routes). It now has a cognition-driven
  reactor, a live LangGraph pipeline graph, vitals rings, an activity stream, and
  count-up metrics — entrance choreography, status-reactive animation, all
  reduced-motion safe.
- **CognitionStore** (`command-deck/cognition.store.ts`) + **CognitionGraph**
  consume the WebSocket cognition events (`agent_activity` / `task_progress` /
  `council`) and project `cognitionLoad`, `activityFeed`, and per-stage state.
- **Reusable primitives:** `CountUpDirective`, `reactor-core`, `vitals-ring`,
  `cognition-graph`.
- **Section polish (autonomous loop, now stopped):** Analytics (built from a stub),
  Tools/MCP Registry, Conversations — each token-aligned, animated, build-verified.
- **Containerized UI:** `core-ui` service runs `ng serve` over the bind-mounted
  source on `:4200` (hot reload).

> All of the above is currently **uncommitted** on `develop`. See §6 before this
> grows further.

---

## P0 — Make cognition actually live (backend)

**This is the single biggest unlock.** The deck's reactor, activity stream, and
pipeline graph are fully wired to the WebSocket — but **no backend product code
emits the cognition events**, so the feed connects and stays empty (graceful empty
state). Lighting this up makes the whole deck come alive with real data.

- [ ] Emit `agent_activity` / `task_progress` from the cognitive graph
      (`backend/app/core/langgraph/core_graph.py`) — one event per stage
      transition (Comprehension → Orchestration → Reasoning → Evaluation) and per
      active agent, via `event_publisher.publish(...)`.
- [ ] Emit `council` events from `backend/app/services/council/deliberation_service.py`
      (`session_started` / `perspective_added` / `vote_cast` / `debate_round` /
      `synthesis_ready` / `session_complete`).
- [ ] Emit agent lifecycle from `agent_factory_service.py`.
- [ ] (Optional) Add a REST snapshot/history endpoint so the store can backfill on
      connect — cognition events are fire-and-forget broadcast today, which is why
      `CognitionStore` intentionally has **no** polling fallback.
- **Acceptance:** open the deck → reactor energy tracks real cognition load,
  pipeline nodes light up by stage, the activity stream populates live.
- **Refs:** `backend/app/services/event_publisher.py`,
  `backend/app/models/ws_events.py`,
  `ui/core-ui/src/app/landing-page/command-deck/cognition.store.ts` (see the
  backend-wiring TODO block), [`docs/api/websocket-events.md`](../api/websocket-events.md).

---

## P1 — Finish the cognition surface

- [ ] **Stage detail panel.** `cognition-graph` already emits `stageSelected`;
      build the side panel it feeds — show the selected stage's active agents,
      prompt, model, token usage, and latest output (Conductor-style node inspector).
- [ ] **Verify the per-stage mapping** (`_buildPipelineStages` in
      `landing-page.component.ts`) once real `agent_activity`/`task_progress` frames
      flow; tune the agent→stage and state mapping against live data.
- [ ] **Council Chamber view.** The store already aggregates `CouncilSessionState`
      (perspectives, votes, round, synthesis). Render it as an amphitheater/timeline:
      perspectives around a ring, debate rounds on a timeline, a vote tally, and the
      synthesis "crystallizing" in the center. Most cinematic thing the system can show.
- [ ] **Command palette (Cmd-K)** as the primary "steer" input, plus inline
      human-approval gates for human-in-the-loop steps.

## P1 — Analytics: wire real data

The Analytics page (`analytics-page`) is a polished shell on **sample telemetry**
(flagged with a "Preview" pill).

- [ ] Wire a real metrics endpoint for the KPIs + throughput series.
- [ ] Decide charting approach (keep the token-based SVG/CSS sparkline, or adopt a
      lightweight lib) for richer series.
- [ ] Remove the "Preview · sample telemetry" pill once values are live.

---

## P2 — Continue the section polish loop

Re-arm with `/loop 30m <the polish prompt>` (see the prompt at the top of
[`docs/ui-polish-log.md`](../ui-polish-log.md)). Remaining rotation, each needs the
same treatment (token migration, deck-consistent animation, empty/loading/error
states, `:focus-visible`, responsive):

- [ ] `communication` (large — 46 KB SCSS; scope carefully)
- [ ] `kanban`
- [ ] `command-center`
- [ ] `agents` (builder / library / marketplace)
- [ ] `knowledgebase`
- [ ] `knowledge-attribution`
- [ ] `discord-bridge-dashboard`
- [ ] `boards`
- [ ] `creative/*`

## P2 — Design-system hardening

- [ ] Promote the deck's recurring patterns into a shared **"FUI kit"** (glass panel,
      status pill/dot, HUD frame/corners, entrance keyframes, count-up) so each section
      composes consistently instead of re-deriving them.
- [ ] Token audit: sweep remaining hardcoded hex/rgba/rem across components → tokens
      (per-section follow-ups are noted in the polish log, e.g. mcp-registry spacing).
- [ ] Document the token taxonomy (`solarpunk-theme.scss`) as a short reference.

---

## P1 — Tech debt & housekeeping (from this session)

- [ ] **Branch hygiene:** move the uncommitted UI work off `develop` onto a
      `feature/*` branch; commit in logical chunks (deck, directive, per-section).
- [ ] **Revert** `ui/core-ui/package-lock.json` — an unrelated 803-line
      `@electron/windows-sign` churn (not from this work).
- [ ] **Remove dead code:** `recentActivities` + the `instanceService.activities$`
      subscription in `landing-page.component.ts` (the activity panel now reads
      `cognitionStore.activityFeed()`).
- [ ] **Dockerfile:** `ui/core-ui/Dockerfile` uses `npm ci --legacy-peer-deps` to work
      around `@angular/material@19.2.17` pinning `@angular/cdk@19.2.17` while the
      lockfile resolves `cdk@19.2.19`. Proper fix: bump material to `^19.2.19` and
      regenerate the lockfile, then drop the flag.
- [ ] **Bundle budget:** production build's initial bundle (~3.99 MB) exceeds the
      3 MB `angular.json` budget (pre-existing; three.js + Angular Material). Decide:
      raise the budget, route-level lazy-load, or drop `three.js` if it stays unused.
- [ ] **Separate non-UI changes:** `backend/pyproject.toml`, `backend/uv.lock`,
      `docker-compose.dev.yml` are unrelated local-dev fixes — commit them separately.

## P2 — Testing

- [ ] Add specs for `CountUpDirective` and the Analytics component's pure logic
      (`barHeight`, projections). The cognition store + graph already have
      sentinel-style specs.
- [ ] Keep `npm run build:ng -- --configuration development` green as the gate
      (the polish loop already enforced this per iteration).

---

## Key files

| What | Where |
|------|-------|
| Command deck | `ui/core-ui/src/app/landing-page/landing-page.component.*` |
| Cognition store | `ui/core-ui/src/app/landing-page/command-deck/cognition.store.ts` |
| Cognition graph | `ui/core-ui/src/app/landing-page/command-deck/cognition-graph/` |
| Reactor / vitals | `ui/core-ui/src/app/landing-page/command-deck/{reactor-core,vitals-ring}/` |
| Count-up directive | `ui/core-ui/src/app/shared/directives/count-up.directive.ts` |
| Design tokens | `ui/core-ui/src/solarpunk-theme.scss` |
| Event publisher (backend) | `backend/app/services/event_publisher.py` |
| WS event models (backend) | `backend/app/models/ws_events.py` |
| Polish log | [`docs/ui-polish-log.md`](../ui-polish-log.md) |
| WS event contract | [`docs/api/websocket-events.md`](../api/websocket-events.md) |
