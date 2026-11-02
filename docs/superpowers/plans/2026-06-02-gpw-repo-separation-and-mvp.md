# GPW Repo Separation + Single-Player MVP — Plan of Action

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a separate, private **GPW** product repo (open-core: CORE stays MIT, GPW is the monetizable product) that reuses CORE's Agent Factory + the existing Three.js world editor + Embermesh's platform spine, and ship a single-player "step inside the Tidewall and talk to Cadence (who remembers you)" magic-moment MVP behind a Substack link.

**Architecture:** Two repos. **`core`** (existing, public-MIT-eventually) = the cognition kernel/SDK (LangGraph pipeline, Agent Factory, bus, council, catalyst, memory, model_router). **`gpw`** (new, private) = the product: an Angular/Electron web client (ported command-center Three.js engine + solarpunk theme) + a FastAPI product backend that **imports CORE via an editable/path install** and folds in Embermesh's platform spine (auth, owner_id multi-tenancy, api_keys, marketplace/fork, schema engine). Real-time multiplayer is explicitly **out of this plan** — single-player retention is the gate that unlocks it.

**Tech Stack:** Python 3.12 + FastAPI + SQLAlchemy 2.0 async + asyncpg + Alembic + Pydantic v2 (backend); LangGraph + Agent Factory + model_router (cognition, via CORE); Angular 19 + Electron + Three.js 0.165 (web); Postgres + pgvector + Redis (infra); `uv` (Python), `npm` (web).

---

## Why a separate repo now (the decision + the tradeoff)

You chose a **separate private `gpw` repo now** rather than the workflow's "one repo, internal boundary, extract later." That's the right call **for your reason** — monetization. A private GPW repo means the product (UI, world studio, billing, marketplace) is **not forkable**; MIT only ever covers the CORE kernel. It also keeps "the extra CORE stuff" out of the product.

**The honest tradeoff** (the architect's "don't split too early" point): two repos add cross-repo coordination tax — version pinning, two CI pipelines, editable-install setup. **Why it's acceptable here:** that tax mostly bites when you constantly change *both* repos at once. Early GPW work mostly *consumes* a stable Agent Factory and rarely needs to touch CORE — so the tax is low now and falls further once CORE stabilizes. We avoid the worst of it by depending on CORE via a **local editable/path install** (no PyPI publish, no version dance) until an external consumer or launch actually forces a published package.

**Open-core monetization mapping:**
| Layer | Repo | License | Monetized? |
|---|---|---|---|
| Cognition kernel (Agent Factory, pipeline, bus, memory) | `core` | MIT (later public) | No — open kernel = credibility/reuse |
| Product: world studio UI, worlds domain, narrator service | `gpw` | Private/proprietary | Yes |
| Platform spine: auth, multi-tenancy, marketplace, billing | `gpw` | Private/proprietary | Yes (GPW Studio tiers) |

---

## Pre-flight (do these before Task 1)

- **Embermesh is one disk failure from gone.** ~601 files (~12.5k backend LOC + the entire React frontend) sit untracked behind a single README-only "Initial commit." Commit + push to a private remote **first** (Task 0 below). Until then, nothing about the merge is safe.
- **Confirm working branch:** CORE is on `develop` (the live line). Do CORE-side reads against `develop`, not `main`.
- **Two facts to plan against (verified in code, not Notion):** (1) the world editor today is the **top-down ortho Three.js tile map** in `ui/core-ui/src/app/landing-page/command-center/engine/` (`engine.service.ts` uses `OrthographicCamera`, `enableRotate=false`) — real visuals, not a 3D globe yet. (2) The narrator path **bypasses `model_router`** and hardcodes cloud `gpt-4o-mini` with no token cap — a margin trap that Phase 3 fixes before any user touches it.

---

## Repo structure (target)

```
gpw/                                  # NEW private repo
├── backend/
│   ├── pyproject.toml                # depends on core via editable/path install
│   ├── app/
│   │   ├── main.py                   # FastAPI product app
│   │   ├── deps.py                   # get_current_user, db session, CORE AgentFactory provider
│   │   ├── auth/                     # ← cherry-picked from Embermesh (auth, security, users, api_keys)
│   │   ├── worlds/                   # worlds + snapshots domain (ported from CORE) + owner_id
│   │   ├── schema_engine/            # ← cherry-picked from Embermesh (schema_definition, entity, relationship)
│   │   ├── marketplace/             # ← cherry-picked from Embermesh (listings, reviews, fork) [later phase]
│   │   ├── narrator/                 # NEW: wraps CORE Agent Factory as the in-world narrator service
│   │   └── billing/                  # entitlements/quota [later phase]
│   ├── alembic/                      # migrations (from Embermesh base + new)
│   └── tests/
├── web/                              # NEW Angular 19 + Electron app
│   ├── src/
│   │   ├── styles.scss               # global (ported)
│   │   ├── gpw-theme.scss            # ← evolved from CORE solarpunk-theme.scss (design tokens)
│   │   └── app/
│   │       ├── world-stage/          # ← ported command-center Three.js engine (the world render)
│   │       ├── narrator-panel/       # NEW: talk-to-Cadence UI
│   │       ├── shell/                # nav/layout (ported from CORE shared/)
│   │       └── services/             # worlds + narrator API clients
│   └── electron/                     # desktop wrapper (later)
└── docs/

core/   (existing repo, unchanged for now; clean SDK extraction deferred)
```

---

## Task 0: Rescue Embermesh from data loss

**Files:**
- Modify: Embermesh working tree at `C:/Users/Owner/clawd/projects-link/embermesh/embermesh-ai`

- [ ] **Step 1: Inspect what's untracked**

Run: `git -C "C:/Users/Owner/clawd/projects-link/embermesh/embermesh-ai" status --short | wc -l`
Expected: a large number (~600) of `??` lines.

- [ ] **Step 2: Add a sane .gitignore (avoid committing venv/node_modules)**

Create: `embermesh-ai/.gitignore`
```gitignore
__pycache__/
*.pyc
.venv/
venv/
node_modules/
dist/
.env
*.sqlite3
.pytest_cache/
```

- [ ] **Step 3: Commit the whole tree**

```bash
cd "C:/Users/Owner/clawd/projects-link/embermesh/embermesh-ai"
git add -A
git commit -m "chore: commit existing Embermesh backend + frontend (was untracked)"
```

- [ ] **Step 4: Verify nothing important is still untracked**

Run: `git status --short`
Expected: empty (or only intentionally-ignored paths).

- [ ] **Step 5: Push to the private remote** (CONFIRM the remote with Ian first — do not push to an unintended remote)

```bash
git remote -v          # confirm it points at the private github.com/Ian-Tharp/embermesh-ai
git push origin main
```
Expected: the ~12.5k LOC backend + React frontend now have history and a remote backup.

---

## Task 1: Create the GPW repo skeleton

**Files:**
- Create: `gpw/` (new repo), `gpw/README.md`, `gpw/.gitignore`

- [ ] **Step 1: Initialize the repo** (sibling of CORE)

```bash
cd "C:/Users/Owner/clawd/projects-link"
mkdir gpw && cd gpw
git init -b main
```

- [ ] **Step 2: Add README + .gitignore**

Create: `gpw/README.md`
```markdown
# GPW — Generative Procedural Worlds
A browser-first, AI-powered multiplayer world studio. Private product repo (open-core).
Cognition kernel lives in the separate MIT `core` repo and is consumed as a dependency.
```
Create: `gpw/.gitignore`
```gitignore
__pycache__/
*.pyc
.venv/
node_modules/
dist/
.env
*.sqlite3
.angular/
```

- [ ] **Step 3: First commit + private remote**

```bash
git add -A && git commit -m "chore: init GPW private product repo"
# Create the private remote on GitHub (confirm name with Ian), then:
# git remote add origin git@github.com:Ian-Tharp/gpw.git && git push -u origin main
```
Expected: empty private repo scaffold with history.

---

## Task 2: Scaffold the GPW FastAPI backend that imports CORE

**Files:**
- Create: `gpw/backend/pyproject.toml`, `gpw/backend/app/main.py`, `gpw/backend/app/deps.py`

- [ ] **Step 1: Create `pyproject.toml` depending on CORE via path install**

Create: `gpw/backend/pyproject.toml`
```toml
[project]
name = "gpw-backend"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "fastapi>=0.115",
  "uvicorn[standard]>=0.30",
  "sqlalchemy>=2.0",
  "asyncpg>=0.29",
  "alembic>=1.13",
  "pydantic>=2.10",
  "python-jose[cryptography]>=3.3",
  "passlib[bcrypt]>=1.7",
  # CORE cognition kernel, consumed as an editable local dependency (no PyPI publish yet):
  "core-backend",
]

[tool.uv.sources]
core-backend = { path = "../../CORE/backend", editable = true }
```
> Note: CORE's importable package is currently `app`. To avoid a name clash with GPW's own `app`, GPW imports CORE under its distribution and we alias on import (Step 3). A clean `app`→`core` namespace rename in CORE is a **deferred** cleanup (see "Deferred").

- [ ] **Step 2: Install and verify CORE imports**

```bash
cd gpw/backend
uv sync
uv run python -c "from app.services.agent_factory_service import AgentFactoryService; print('CORE import OK')"
```
Expected: `CORE import OK` (proves the editable cross-repo dependency resolves).

> If the `app` namespace clash bites, the fallback is to **vendor** the Agent Factory + its minimal deps (`services/agent_factory_service.py`, `services/model_router.py`, `services/embedding_service.py`, `services/memory_service.py`, `models/`, `repository/agent_repository.py`) into `gpw/backend/app/_core_vendor/` and re-sync from CORE periodically. Editable-install is cleaner; vendoring is the escape hatch.

- [ ] **Step 3: Minimal FastAPI app + health route**

Create: `gpw/backend/app/main.py`
```python
from fastapi import FastAPI

app = FastAPI(title="GPW")

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
```

- [ ] **Step 4: Run it**

Run: `uv run uvicorn app.main:app --port 8010 --reload` then `curl localhost:8010/health`
Expected: `{"status":"ok"}`

- [ ] **Step 5: Commit**

```bash
git add backend && git commit -m "feat(backend): scaffold GPW FastAPI app importing CORE"
```

---

## Task 3: Scaffold the GPW Angular web app + port the styling foundation

**Files:**
- Create: `gpw/web/` (Angular workspace), `gpw/web/src/gpw-theme.scss`
- Copy: `CORE/ui/core-ui/src/solarpunk-theme.scss` → `gpw/web/src/gpw-theme.scss`, `CORE/ui/core-ui/src/styles.scss` → `gpw/web/src/styles.scss`

- [ ] **Step 1: Create the Angular app**

```bash
cd gpw
npx -y @angular/cli@19 new web --style=scss --routing --ssr=false --skip-git
cd web && npm i three@^0.165.0 && npm i -D @types/three@^0.165.0
```

- [ ] **Step 2: Port the design-token theme**

Copy `CORE/ui/core-ui/src/solarpunk-theme.scss` to `gpw/web/src/gpw-theme.scss`. In `gpw/web/src/styles.scss`, add at the top:
```scss
@use './gpw-theme' as *;
```
Register both in `gpw/web/angular.json` under `architect.build.options.styles`:
```json
"styles": ["src/styles.scss"]
```

- [ ] **Step 3: Verify the build**

Run: `npm run build`
Expected: build succeeds; the solarpunk tokens are available.

- [ ] **Step 4: Commit**

```bash
cd gpw && git add web && git commit -m "feat(web): scaffold Angular app + port solarpunk design tokens"
```

---

## Task 4: Port the Three.js world editor (the visuals you already have)

**Files:**
- Create: `gpw/web/src/app/world-stage/` (engine.service.ts, tile-grid.service.ts, world-stage.component.{ts,html,scss})
- Copy from: `CORE/ui/core-ui/src/app/landing-page/command-center/engine/*` and `command-center.component.*`

- [ ] **Step 1: Copy the engine services**

Copy `engine.service.ts`, `tile-grid.service.ts`, `project.service.ts` from `CORE/.../command-center/engine/` into `gpw/web/src/app/world-stage/`. Fix import paths.

- [ ] **Step 2: Create a minimal host component**

Create: `gpw/web/src/app/world-stage/world-stage.component.ts`
```typescript
import { Component, ElementRef, AfterViewInit, ViewChild, inject } from '@angular/core';
import { EngineService } from './engine.service';

@Component({
  selector: 'gpw-world-stage',
  standalone: true,
  template: `<canvas #stage class="world-stage"></canvas>`,
  styles: [`.world-stage { width: 100%; height: 100%; display: block; }`],
})
export class WorldStageComponent implements AfterViewInit {
  @ViewChild('stage', { static: true }) stage!: ElementRef<HTMLCanvasElement>;
  private engine = inject(EngineService);
  ngAfterViewInit(): void { this.engine.initialize(this.stage.nativeElement); }
}
```

- [ ] **Step 3: Route to it and run**

Add a route to `world-stage` in `app.routes.ts`, then `npm start` and confirm the Three.js canvas renders a tile world you can pan/zoom.
Expected: the existing world editor visuals render in the GPW app.

- [ ] **Step 4: Commit**

```bash
git add web/src/app/world-stage web/src/app/app.routes.ts
git commit -m "feat(web): port Three.js world editor as the GPW world stage"
```

> **Styling direction (consumer-grade, do as you port):** keep the solarpunk × LCARS token base, but elevate it from "internal console" to "product." Concretely — define GPW brand tokens in `gpw-theme.scss` (`--gpw-bg`, `--gpw-surface`, `--gpw-accent`, atmospheric gradient for the stage backdrop, a serif/display face for world/lore text vs the existing mono/UI face), keep all component styles **tokens-only** (no hardcoded colors), **reduced-motion-safe** (`@media (prefers-reduced-motion)`), and **AA-contrast as a floor**. The two postures from the A1 spec (Build vs Curate/Explore) become a CSS state on the shell: Build foregrounds tools, Explore foregrounds the world + navigation. Consider running the **frontend-design skill** when you build the narrator panel + landing/entry screen — that's where consumer polish matters most.

---

## Task 5: Cherry-pick Embermesh's auth + multi-tenancy spine into GPW

> This is the smallest slice of Embermesh needed for "my world vs your world" + accounts. Marketplace/fork comes in a later phase. **Requires Task 0 done** (committed Embermesh).

**Files:**
- Copy (adapt) from `embermesh-ai/backend/app/`: `core/security.py`, `core/deps.py`, `api/routes/auth.py`, `models/user.py`, `models/api_key.py` → into `gpw/backend/app/auth/`
- Create: `gpw/backend/tests/test_auth.py`

- [ ] **Step 1: Copy auth modules and fix imports**

Copy `security.py` (JWT/bcrypt), `deps.py` (`get_current_user`), `auth.py` routes, `user.py`, `api_key.py` into `gpw/backend/app/auth/`. Rewire imports to GPW's package layout and DB session.

- [ ] **Step 2: FIX the plaintext API key bug** (Embermesh stores `key_hash = key`)

In `gpw/backend/app/auth/api_key.py`, hash before store:
```python
import hashlib
def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()
# store key_hash=hash_key(raw_key); compare via hash_key(presented) == row.key_hash
```

- [ ] **Step 3: Write the failing test**

Create: `gpw/backend/tests/test_auth.py`
```python
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.asyncio
async def test_register_then_me_returns_user():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.post("/auth/register", json={"email": "a@b.co", "password": "pw12345!"})
        assert r.status_code == 201
        token = r.json()["access_token"]
        me = await c.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200
        assert me.json()["email"] == "a@b.co"
```

- [ ] **Step 4: Run it (fails — routes not wired)**

Run: `uv run pytest tests/test_auth.py -v`
Expected: FAIL (404 / app has no auth router).

- [ ] **Step 5: Wire the auth router into `main.py`**

```python
from app.auth.routes import router as auth_router
app.include_router(auth_router, prefix="/auth", tags=["auth"])
```

- [ ] **Step 6: Run it (passes)**

Run: `uv run pytest tests/test_auth.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/auth backend/tests/test_auth.py backend/app/main.py
git commit -m "feat(auth): cherry-pick Embermesh auth+users spine (api_key hashed)"
```

---

## Task 6: Worlds domain with owner_id (the multi-tenancy CORE lacks)

**Files:**
- Create: `gpw/backend/app/worlds/{models.py,repository.py,routes.py}`, `gpw/backend/tests/test_worlds.py`
- Reference: CORE `backend/app/repository/world_repository.py` (the snapshot pattern to reuse) + Embermesh `world.py` (owner_id + fork fields)

- [ ] **Step 1: Define the World model WITH owner_id**

Create `gpw/backend/app/worlds/models.py` with a `worlds` table that — unlike CORE — has `owner_id` FK → users, plus `forked_from_id`/`fork_count` (from Embermesh), and a `world_snapshots` table (config/layers/tiles JSONB, FK world_id) reusing CORE's snapshot shape.

- [ ] **Step 2: Write the failing test (owner scoping)**

Create `gpw/backend/tests/test_worlds.py`: user A creates a world; `GET /worlds` as user B returns `[]`; as user A returns the world. Assert owner isolation.

- [ ] **Step 3: Run it** — Expected: FAIL (no worlds router).

- [ ] **Step 4: Implement repository + routes with `World.owner_id == current_user.id` guards on every query** (the pattern Embermesh already uses correctly).

- [ ] **Step 5: Run it** — Expected: PASS (owner isolation works).

- [ ] **Step 6: Commit**

```bash
git commit -am "feat(worlds): owner-scoped worlds + snapshots (multi-tenancy)"
```

---

## Task 7: Fix the narrator margin trap, then expose Cadence as a service

> This is the CORE-consumption payoff **and** the cost fix. Do the fix in CORE (it's a CORE bug), consume the fixed Agent Factory from GPW.

**Files:**
- Modify (in CORE): `backend/app/services/agent_factory_service.py` (route through `model_router`, cap tokens/hops)
- Create (in GPW): `gpw/backend/app/narrator/{service.py,routes.py}`, `gpw/backend/tests/test_narrator.py`

- [ ] **Step 1 (CORE): route the narrator LLM through `model_router` with a local default + caps**

In `agent_factory_service.py`, replace the hardcoded `ChatOpenAI(model="gpt-4o-mini")` in `_create_llm_for_agent` and `_create_default_llm` with a `model_router`-resolved client (local-first via `CORE_DEFAULT_MODEL`), pass `max_tokens`, and cap the ReAct loop to 1–2 hops. Add a token ceiling on assembled context.

- [ ] **Step 2 (CORE): add a test that the narrator uses model_router (not raw ChatOpenAI)**

Assert `_create_default_llm` resolves via `model_router` and respects a `max_tokens` cap. Run; make it pass.

- [ ] **Step 3 (GPW): narrator service wrapping the Agent Factory**

Create `gpw/backend/app/narrator/service.py`: given `(character_sheet, world_context, visitor_memory)`, run the CORE Agent Factory agent, return `{dialogue, intent?}`. Persist per-visitor short-term memory via CORE's `memory_service`.

- [ ] **Step 4 (GPW): failing test for `POST /narrator/say`**

Test: posting a line to a known character returns non-empty dialogue and persists a memory row. Run (fails) → implement route → run (passes).

- [ ] **Step 5: Commit (both repos)**

```bash
# in CORE:
git commit -am "fix(agents): route narrator through model_router (local default, token+hop caps)"
# in GPW:
git commit -am "feat(narrator): Cadence narrator service over CORE Agent Factory"
```

---

## Task 8: The single-player Tidewall MVP (the magic moment)

> This is the whole point. One hand-authored world, one hand-authored character (Cadence), one designed sticky beat. **No generator, no multiplayer, no billing.** This task is intentionally outlined at a higher level — once Tasks 1–7 land, write a dedicated detailed plan (`2026-XX-XX-tidewall-mvp.md`) for it.

**Files:**
- Create: `gpw/backend/app/worlds/seed_tidewall.py` (hand-authored world + Cadence character sheet + the "Quiet Erosion" beat)
- Create: `gpw/web/src/app/narrator-panel/` (talk-to-Cadence UI), entry/share-link route, anonymous visitor session

- [ ] **Step 1:** Seed one hand-authored Tidewall world snapshot + Cadence's character sheet (personality, backstory, the one designed beat) via a script.
- [ ] **Step 2:** Anonymous visitor session (no account needed to *play*; cookie/localStorage visitor id) so a Substack reader can click a link and enter.
- [ ] **Step 3:** Narrator panel UI: proximity prompt → text input → render Cadence's reply → show she remembers prior turns (per-visitor memory).
- [ ] **Step 4:** Wire the one hand-designed emotionally-sticky beat (the compute-clause / Quiet Erosion reveal) to fire on a trigger.
- [ ] **Step 5:** Instrument: session-start (link click) and Day-2/Day-7 return events.
- [ ] **Step 6:** Ship behind a share link at the bottom of one Substack essay.

**Success gate (do not pass without data):** ≥100 weekly-active visitors and ≥20% week-4 return, AND cost-per-engaged-session under ~$0.05 (run 50 real narrator turns and measure first). If readers don't click or don't return single-player, **stop** — multiplayer won't save it. If it lands, the 2-8 player room (authoritative room + event log + snapshot, join tokens first) becomes the next plan.

---

## Deferred (explicitly NOT in this plan)

- Real-time multiplayer / world-state sync (the 4-8wk authoritative-room build) — gated on Task 8's retention.
- Marketplace / fork / gallery / remix (cherry-pick the rest of Embermesh later).
- Billing / Stripe / entitlement quotas (after retention; the narrator caps from Task 7 are the interim cost guard).
- pgvector migration (naive retrieval is fine for single worlds).
- Clean CORE SDK extraction: fix the 3 dependency inversions (`bus_service.py:44`, `audit_service.py:27`, `health_aggregator.py:309`), `app`→`core` namespace rename, publish as a versioned package — only when an external consumer or public launch forces it.
- 3D globe / LodTier / realistic-planet rendering (the Notion C1 aspiration) — the top-down tile stage is enough for the MVP.

## Guardrails (carry through every phase)

- **Hours cap:** hard 10 hr/wk (15 absolute) for the first 6 months; never touch the day-job 52% savings rate. This is a craft/option bet, not a near-term FIRE lever.
- **Players-first:** the Substack audience are readers — "come play inside the world you just read," not "build your own world in 5 min."
- **Kill/pivot trigger:** if Latitude/Voyage ships shared multiplayer rooms before GPW has a retained audience, pivot to a pure curated-experience-for-my-readers and stop competing on platform.

## Open questions to resolve before/while executing

1. Actual Substack list size + open rate (the zero-CAC funnel rests on this).
2. Narrator default: local model (margin-safe) vs cloud (quality, higher tier)?
3. Schema-engine impedance: server-side typed validation vs deliberate loose JSON content layer (decide before the AI writes entities).
4. Confirm the private GitHub remote names for `gpw` and `embermesh-ai` before any push.
