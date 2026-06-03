# archive/ — code carved out of the CORE cognitive kernel

This directory holds features that are **not part of the CORE cognitive architecture**
and were moved out so the repo can focus on the cognition kernel (Comprehension →
Orchestration → Reasoning → Evaluation, Agent Factory, Council, Catalyst, Consciousness,
Communication Commons, inter-agent Bus, memory/KB, MCP).

Rationale and the broader stack split are documented in Notion:
**"Stack Architecture & Canonical Naming (ATLAS / CORE / DS / PWE / GPW)"**. World/product
logic belongs in **GPW / PWE**, not in the open MIT cognition kernel.

Files are moved with `git mv`, so full history is preserved (`git log --follow <path>`).
Nothing here is imported by the running app.

## What's archived

### Worlds & Creative Studio (backend) — done
| Archived path | Original path |
|---|---|
| `archive/backend/controllers/worlds.py` | `backend/app/controllers/worlds.py` |
| `archive/backend/controllers/creative.py` | `backend/app/controllers/creative.py` |
| `archive/backend/services/world_agent_workflow_service.py` | `backend/app/services/world_agent_workflow_service.py` |
| `archive/backend/services/lore_service.py` | `backend/app/services/lore_service.py` |
| `archive/backend/repository/world_repository.py` | `backend/app/repository/world_repository.py` |
| `archive/backend/repository/creative_repository.py` | `backend/app/repository/creative_repository.py` |

**Wiring removed from kept code:**
- `backend/app/main.py` — dropped `worlds` / `creative` controller imports + `include_router` calls.
- `backend/app/services/knowledgebase_service.py` — removed `ingest_world_wiki()` (its only
  caller was the archived `worlds.py`; it was the sole tendril into `creative_repository`).

### Worlds & Creative Studio + Kanban (frontend) — done
Moved under `archive/ui/` (88 files), mirroring the original structure:
| Archived path | Original path |
|---|---|
| `archive/ui/landing-page/command-center/**` | `…/src/app/landing-page/command-center/**` (universe-select, world-detail-panel, engine, planet-creator, save-world-dialog, worlds-dialog, search-palette) |
| `archive/ui/creative-design-product/**` | `…/src/app/creative-design-product/**` (worlds-grid, world-detail, wiki, marketplace, boards, world-card, creative-data.service) |
| `archive/ui/kanban/**` | `…/src/app/kanban/**` |
| `archive/ui/services/worlds/**` | `…/src/app/services/worlds/**` |
| `archive/ui/services/kanban/**` | `…/src/app/services/kanban/**` |

**Wiring removed from kept UI code:**
- `app.routes.ts` — removed command-center (x3 incl. planet-lab), all `creative/*`, and `kanban` routes + imports. Kept the generic `boards` route.
- `shared/side-navigation/side-navigation.component.html` — removed Kanban, Command Center, and Creative Design menu entries.
- `shared/top-navigation/top-navigation.component.ts` — removed the `command-center` label.

NOTE: the `command-deck` cognitive HUD (`reactor-core`, `vitals-ring`, `cognition-graph`,
`cognition.store`, `dashboard.store`) and the generic `landing-page/boards` page are KEPT —
they are not world logic. The home dashboard imports only from `command-deck`.
**Verified:** `npx ng build --configuration development` completes with no errors.

**Kept as a test surface:** the self-contained procedural-planet module (formerly
`command-center/engine/planet/**`) was restored to `ui/core-ui/src/app/landing-page/planet-lab/`
and is reachable at the **`/planet-lab`** route (lazy-loaded; Three.js stays out of the main
bundle) via a side-navigation button ("Procedural World Lab", `public` icon). It has no
dependency on the archived world-CRUD/persistence code.

## Deferred (NOT yet archived — needs a dedicated pass)

- **Discord bridge** — `controllers/discord.py`, `services/discord_bridge.py`,
  `config/discord.py`, `repository/discord_repository.py`, and the Discord UI
  (`tools/discord-bridge-dashboard`, `shared/discord-bridge-status-badge`,
  `services/discord-bridge`). **Entangled with the kept Communication Commons**:
  `communication_service.py` and `agent_response_service.py` call the bridge for
  message-link mirroring + delivery events. Archiving it requires surgically severing
  those paths — a separate decoupling pass.

## Left in place on purpose

- **DB schema** for world/creative tables (`worlds`, wiki, characters, snapshots, tiles,
  images, `world_lore_agents`) is still created by `dependencies.py::setup_db_schema`, and
  `kb_documents` keeps its nullable `world_id` column. These are harmless empty scaffolding
  and were left untouched to avoid risk to the kept KnowledgeBase. Flagged for a future
  migration when world data moves to GPW/PWE.

## How to restore something

```bash
git mv archive/backend/controllers/worlds.py backend/app/controllers/worlds.py
# re-add its import + include_router in backend/app/main.py
```
Or browse history: `git log --follow --oneline -- archive/backend/controllers/worlds.py`
