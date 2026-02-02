# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CORE** — Cognitive Orchestration, Reasoning & Evaluation. A modular, self-hosted AI orchestration platform with a multi-agent cognitive pipeline, agent factory, communication commons, council system, and solarpunk-inspired desktop UI.

## Architecture

Multi-component system — all services run via Docker Compose:

### Backend (`/backend`)
- **FastAPI** application (Python 3.12+, managed with `uv`)
- **LangGraph** cognitive pipeline: Comprehension → Orchestration → Reasoning → Evaluation
- Agent Factory with MCP tool binding
- Communication Commons (channels, messages, presence, WebSocket)
- Council of Perspectives (multi-agent deliberation)
- Catalyst Engine (creative divergence-convergence)
- Consciousness module (Blackboard, emergence protocols)
- Inter-agent bus with MMCNC-scoped delivery and triggers
- PostgreSQL + pgvector for storage, Redis for cache/pubsub, Ollama for local LLM

### Frontend (`/ui/core-ui`)
- **Angular 19** + **Electron** desktop wrapper
- Solarpunk dark theme with command deck interface
- Angular Material components

### MCP (`/mcp`)
- Model Context Protocol servers for external tool access
- Docker-based orchestration with registry service

## Development Commands

### Backend
```bash
cd backend
uv sync                                    # Install deps
python -m app.main                         # Dev server (port 8001)
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload  # Alt
uv run black .                             # Format
python -m pytest tests/ -q                 # Run tests
```

### Frontend
```bash
cd ui/core-ui
npm install
npm start          # Angular + Electron
npm run start:ng   # Angular dev server only
npm run build      # Production build
npm test           # Tests
npm run lint       # Lint
```

### Docker
```bash
docker compose up -d                       # Start all services
docker compose -f docker-compose.dev.yml up -d  # Dev mode
```

## Key Paths

| What | Where |
|------|-------|
| CORE cognitive graph | `backend/app/core/langgraph/core_graph.py` |
| Agent Factory | `backend/app/services/agent_factory_service.py` |
| Bus service | `backend/app/services/bus_service.py` |
| Bus triggers | `backend/app/services/bus_triggers.py` |
| Council deliberation | `backend/app/services/council/deliberation_service.py` |
| Catalyst | `backend/app/services/catalyst_service.py` |
| Consciousness bridge | `backend/app/services/consciousness_council_bridge.py` |
| API controllers | `backend/app/controllers/` |
| Pydantic models | `backend/app/models/` |
| DB repositories | `backend/app/repository/` |
| Tests | `backend/tests/` |
| Frontend app | `ui/core-ui/src/app/` |

## Documentation

All docs live in `docs/` — see [`docs/README.md`](docs/README.md) for the full index:

- `docs/architecture/` — System design docs (Agent Factory, response system)
- `docs/adr/` — Architecture Decision Records
- `docs/api/` — WebSocket events, endpoint docs
- `docs/consciousness/` — Emergence protocols, inter-agent communication
- `docs/council/` — Council charter, synthesis, deliberation outputs
- `docs/deployment/` — Docker setup, containerization, sandbox
- `docs/implementation/` — Testing strategy, roadmaps
- `docs/research/` — Background research
- `docs/roadmap/` — Feature backlogs, vision documents
- `docs/RSI/` — Recursive self-improvement session logs

## Development Workflow

1. **Branching**: Feature branches from `develop` only (`feature/*`, `fix/*`). Never commit directly to `develop` or `main`.
2. **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`)
3. **Testing**: `python -m pytest tests/ -q` for backend. `npm test` for frontend.
4. **Formatting**: `black` for Python, Angular lint for TypeScript.
5. **PRs**: feature → develop → main.

## Important Notes

- Self-hosted, local-first, offline-capable
- CORE is a neutral cognitive kernel, not a persona
- Backend uses FastAPI lifespan events for initialization/shutdown
- CORS allows `http://localhost:4200` in development
- All Docker services defined in `docker-compose.yml`
- Agent containerization architecture in `docs/deployment/agent-containerization.md`
