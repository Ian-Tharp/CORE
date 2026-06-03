# ADR-006: Archive World/Creative-Studio logic out of the CORE kernel

**Status:** ✅ Implemented
**Date:** 2026-06-03
**Author:** Ian + Claude Code

## Context

CORE is meant to be a **neutral cognitive kernel** — Comprehension, Orchestration,
Reasoning, Evaluation (CLAUDE.md: "CORE is a neutral cognitive kernel, not a persona").
In practice the repo had grown a substantial **Worlds & Creative Studio** product surface:
backend `worlds`/`creative` controllers, `world_agent_workflow_service`, `lore_service`,
`world`/`creative` repositories, and a large frontend (the `command-center` world/universe/
planet builder, `creative-design-product`, and Kanban).

This product/world logic does not belong in the open cognition kernel. Per the cross-project
canonical naming/architecture note (Notion: *"Stack Architecture & Canonical Naming
(ATLAS / CORE / DS / PWE / GPW)"*), world generation and the product experience live in
**GPW / PWE**, and **GPW depends on CORE, never the reverse**. Keeping world logic in CORE
blurred that boundary and was open-decision #5 in that note.

## Decision

Archive everything that is **not** part of the cognitive architecture into an in-repo
`archive/` directory (history preserved via `git mv`), and unwire it from the running app.

**Archived (see [`archive/README.md`](../../archive/README.md)):**
- Backend: `worlds.py`, `creative.py` controllers; `world_agent_workflow_service.py`,
  `lore_service.py`; `world_repository.py`, `creative_repository.py`.
- Frontend: `command-center/**` (world/universe builder), `creative-design-product/**`,
  `kanban/**`, and the `services/worlds` + `services/kanban` services.
- Docs: `worlds-architecture.md`, `planet-world-creator.md`, `lore-agents-linking-plan.md`
  (moved under `archive/docs/`).

**Kept (cognitive architecture + test surface):** the C/O/R/E pipeline, Agent Factory,
Council, Catalyst, Consciousness, Communication Commons, inter-agent Bus, memory/KB, MCP,
and the `command-deck` cognition HUD. The self-contained **procedural-planet module** was
retained as a standalone visual test surface at the **`/planet-lab`** route.

**Scope notes:** the Discord bridge was **not** archived (it is woven into the kept
Communication Commons; deferred to a separate decoupling pass). World/creative **DB tables**
created by `setup_db_schema` are intentionally left in place (harmless empty scaffolding) to
avoid risk to the kept KnowledgeBase; flagged for a future migration to GPW/PWE.

## Consequences

**Positive:** CORE matches its stated identity (clean cognition kernel); the open-core
boundary with GPW is restored; smaller surface to maintain and test.

**Negative / follow-ups:** world data model + Creative Studio must eventually be reimplemented
in GPW/PWE; the retained world DB tables are now unused (see `persistence-audit.md`); Discord
still needs its own decoupling pass.
