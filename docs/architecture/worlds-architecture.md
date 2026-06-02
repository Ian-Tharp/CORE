# Worlds & Creative Studio Architecture

**Last updated:** 2026-06-02

How CORE's "explorable universe of worlds" works end to end — the 3D command center, the per-world creation studio (AI art, inhabitants, lore, knowledge), and how all of it persists and reloads. This complements the per-route status in [`../implementation/ui-gaps-audit.md`](../implementation/ui-gaps-audit.md).

## Concept

The **command center** (`/command-center`) is a Three.js galaxy where each orb/tile is a **world** — a product of the procedural world generator that the user authors into a populated place. A world accrues:

- a **name, biome, terrain, tags, description** (tile metadata),
- AI-generated **world art** (gpt-image-1),
- **inhabitants** — characters with AI portraits,
- **lore** — schema-tagged wiki pages (Overview / History / Peoples & Culture…),
- **knowledge** — the world's wiki ingested into a world-scoped vector store for RAG.

Everything is persisted immediately to the database and reloads independently of an explicit "save".

## Data model

All world-scoped content hangs off a backend `worlds.id` (the command-center's backing world). Tables (created idempotently in `backend/app/dependencies.py`):

| Table | Shape | Purpose |
|-------|-------|---------|
| `worlds` | `id`, `name`, … | The world record. |
| `world_metadata` | `world_id` PK, `snapshot` JSONB, `updated_at`, FK→`worlds` CASCADE | The tile-grid snapshot blob written on **Save** (tile names/tags/notes/links, connections). |
| `world_assets` | `id`, `world_id`, `tile_index`, `kind` (`'art'`…), `title`, `image_b64` TEXT, `created_at`, FK CASCADE | Generated images, per tile. Persisted on generation (not on save). |
| `wiki_pages` | `id`, `world_id`, `title`, `content`, `metadata` JSONB | Lore pages. Generated lore is tagged `metadata.template` (Overview/History/…) + `source: 'ai'`. |
| `characters` | `id`, `world_id`, `name`, `traits` JSONB, `image_b64` | Inhabitants + portraits. |
| `kb_documents` | … + `world_id` (added), `file_hash` (dedupe), `doc_embedding_vec vector(768)` | World wiki ingested as knowledge. |
| `kb_chunks` | … `embedding_vec vector(768)`, HNSW cosine index | Chunked + embedded lore for RAG. The legacy JSONB `embedding` is superseded by `embedding_vec`. |

**Reload model:** content (art, inhabitants, lore, knowledge) is queried directly by `world_id` (+ `tile_index` for art), so it reloads regardless of whether the grid snapshot was saved. The **tile↔content links** (which wiki pages belong to which tile) live in the tile metadata, which is snapshotted into `world_metadata` on Save.

## Backend API surface

**Worlds** (`backend/app/controllers/worlds.py`, prefix `/worlds`):

- `POST /` create · `GET /` list · `DELETE /{id}` · `GET /by-name/{name}`
- `POST /{id}/snapshots` · `GET /{id}/snapshots/latest` · `GET /{id}/snapshots` · `DELETE /{id}/snapshots/{snapshot_id}`
- `PUT /{id}/metadata` · `GET /{id}/metadata` — the `world_metadata` snapshot blob
- `POST /{id}/assets` · `GET /{id}/assets` · `DELETE /{id}/assets/{asset_id}` — world art
- `POST /{id}/knowledge/ingest-wiki` · `GET /{id}/knowledge` · `POST /{id}/knowledge/search` — world-scoped RAG
- `POST /{id}/lore/generate` — schema-aware lore (see below)
- `POST /{id}/agents/lore` — agent-driven lore workflow

**Creative** (`backend/app/controllers/creative.py`, prefix `/creative`):

- `POST /wiki` · `PUT /wiki/{id}` · `GET /wiki` — wiki pages (metadata stored as JSONB)
- `POST /characters` · `GET /characters` — inhabitants
- `POST /image` — generate an image, returns `{ b64 }`
- `POST /characters/{id}/image` — generate + persist a portrait

## Generation pipeline

- **Images** — `creative.py::_generate_image_b64()` calls the OpenAI Images API. Model is `OPENAI_IMAGE_MODEL` (default `gpt-image-1`); `gpt-image-1` returns base64 by default and rejects `response_format`, so that param is only set for older models. Used by both `/creative/image` (world art) and `/creative/characters/{id}/image` (portraits).
- **Lore** — `backend/app/services/lore_service.py::generate_lore_page(world_id, *, kind, focus, world_name, context)` builds a loremaster prompt grounded in the world's known details and calls the provider-agnostic `_llm_or_stub` off the event loop. It prefers `gpt-4o-mini` when `OPENAI_API_KEY` is present (local models are too slow for a full page), else `CORE_DEFAULT_MODEL`. The first `# ` heading becomes the title; the page is persisted via `creative_repository.create_wiki_page(...)` tagged `{template: kind, source: 'ai'}`.
- **Knowledge** — `knowledgebase_service.ingest_world_wiki(world_id)` turns a world's wiki pages into `kb_documents`/`kb_chunks` (chunk + embed, deduped by `wiki:<page_id>` hash); `retrieve_context_by_world(query, world_id)` does vector search scoped via a document filter. Embeddings are provider-aware (LM Studio `/v1/embeddings` or Ollama) — see [`../deployment/local-llm-providers.md`](../deployment/local-llm-providers.md).

## Frontend

- **3D galaxy** — `ui/core-ui/src/app/landing-page/command-center/engine/tile-grid.service.ts` (spiral-galaxy placement, world-orbs, atmosphere/beacons, connection arcs, selection ping, documented-world emphasis, Next/Back stepper) on top of `engine.service.ts` (orthographic MapControls, camera tweens, `projectPoint` for screen-space labels).
- **Creation studio** — `world-detail-panel/` renders the selected world: a large **World Plate** hero of the newest art, a clickable filmstrip of older art, a hover "loupe" preview, a focused **lightbox** (`image-lightbox-dialog/`), the **inhabitants** grid, **lore** generation buttons (Overview/History/Peoples), and **knowledge** ingest + search.
- **First-load universe picker** — `universe-picker-dialog/` greets a fresh session with "Chart a New Universe" vs. load a previous one (reuses `WorldsService.listWorlds`).
- **Services** — `services/worlds/worlds.service.ts` (metadata, assets, knowledge, lore) and `services/creative/creative.service.ts` (wiki, characters, image). Both should resolve their base URL through `AppConfigService` (in progress — see the hygiene backlog).

## Persistence & reload flow

1. **Author** a tile (name/biome/tags) → tile metadata in `TileMetadataService`.
2. **Generate** art/inhabitants/lore → persisted immediately to `world_assets` / `characters` / `wiki_pages` by `world_id`.
3. **Ingest** wiki → `kb_documents`/`kb_chunks` (world-scoped RAG).
4. **Save** → grid + tile-metadata snapshot written to `world_metadata`.
5. **Reload** → `getLatestSnapshot` restores the grid; `getMetadata` rehydrates tile metadata; per-tile `listAssets`/`listCharacters`/`listKnowledge` lazily reload generated content by `world_id`.

## Vision & current gaps

The north-star ("The Forge Atlas" — galaxy as both map and easel) and a phased roadmap live with the team; the concrete per-feature status (what's wired vs stubbed) is tracked in [`../implementation/ui-gaps-audit.md`](../implementation/ui-gaps-audit.md). Notable: Agent Marketplace and the Analytics dashboard have **no backend** yet, whereas reactions, the engine step-stream, character generation, and knowledge search are all backed and (now) wired.
