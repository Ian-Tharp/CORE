# CORE — Cognitive Orchestration, Reasoning & Evaluation

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![Angular](https://img.shields.io/badge/angular-19-red.svg)
![Docker](https://img.shields.io/badge/docker-compose-blue.svg)

**CORE** is a modular, self-hosted AI orchestration platform built around four cognitive pillars: **Comprehension**, **Orchestration**, **Reasoning**, and **Evaluation**. It provides a multi-agent architecture with a cognitive pipeline, agent factory, communication commons, and a solarpunk-inspired desktop UI.

![CORE Architecture](assets/imgs/COREv1%20Diagram.png)

## Features

- **Cognitive Pipeline** — LangGraph-based workflow: Comprehension → Orchestration → Reasoning → Evaluation
- **Agent Factory** — Define, instantiate, and manage AI agents with configurable personalities and MCP tool bindings
- **Communication Commons** — Real-time multi-agent chat with channels, threads, reactions, and presence
- **Council of Perspectives** — Multi-agent deliberation framework for structured analysis
- **Catalyst Engine** — Creative divergence-convergence workflows
- **MCP Integration** — Model Context Protocol servers for external tool access
- **Consciousness Module** — Experimental consciousness emergence protocols and inter-agent dialogue
- **Desktop UI** — Angular 19 + Electron app with command deck interface
- **Self-Hosted** — Runs entirely on your hardware via Docker Compose

## Quick Start

```bash
# Clone the repository
git clone https://github.com/IanTharp/CORE.git
cd CORE

# Start all services
docker compose up -d

# Services:
#   Backend API    → http://localhost:8001
#   API Docs       → http://localhost:8001/docs
#   Frontend UI    → http://localhost:4200
#   PostgreSQL     → localhost:5432
#   Redis          → localhost:6379
```

See [docs/deployment/docker.md](docs/deployment/docker.md) for detailed Docker configuration and production setup.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Frontend (Angular 19 + Electron)   │
│  Landing Page · Agent Library · Communication Commons │
│  Command Center · Engine Playground · Creative Studio │
└──────────────────────┬───────────────────────────────┘
                       │ HTTP / WebSocket
┌──────────────────────▼───────────────────────────────┐
│                    Backend (FastAPI)                   │
│  ┌─────────────┐ ┌──────────┐ ┌───────────────────┐ │
│  │ CORE Engine  │ │  Agent   │ │  Communication    │ │
│  │ (LangGraph)  │ │  Factory │ │  Commons          │ │
│  └─────────────┘ └──────────┘ └───────────────────┘ │
│  ┌─────────────┐ ┌──────────┐ ┌───────────────────┐ │
│  │  Council    │ │ Catalyst │ │  Consciousness    │ │
│  │  System     │ │  Engine  │ │  Module           │ │
│  └─────────────┘ └──────────┘ └───────────────────┘ │
└──────────────────────┬───────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     PostgreSQL      Redis       Ollama
     (Storage)     (Cache/PubSub) (Local LLM)
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.12+, FastAPI, LangGraph, LangChain |
| Frontend | Angular 19, Electron, Angular Material |
| Database | PostgreSQL with pgvector |
| Cache | Redis |
| Local LLM | Ollama |
| Tooling | MCP (Model Context Protocol) |
| Containers | Docker Compose |

## Project Structure

```
CORE/
├── backend/           # FastAPI application
│   ├── app/           # Application code
│   │   ├── controllers/   # REST API endpoints
│   │   ├── core/          # CORE cognitive engine (LangGraph)
│   │   ├── models/        # Pydantic models
│   │   ├── repository/    # Database access layer
│   │   └── services/      # Business logic
│   ├── migrations/    # Database migrations
│   └── tests/         # Test suite
├── ui/core-ui/        # Angular + Electron frontend
├── mcp/               # MCP server configurations
├── docker/            # Docker build contexts
├── docs/              # Documentation (see docs/README.md)
├── assets/            # Static assets (diagrams, images)
└── docker-compose.yml # Service orchestration
```

## Documentation

Comprehensive docs are in the [`docs/`](./docs/) directory:

- **[Architecture](docs/architecture/)** — System design and implementation plans
- **[API](docs/api/)** — WebSocket events and endpoint docs
- **[ADRs](docs/adr/)** — Architecture Decision Records
- **[Council](docs/council/)** — Council of Perspectives framework and outputs
- **[Deployment](docs/deployment/)** — Docker, containerization, and sandbox setup
- **[Implementation](docs/implementation/)** — Testing and roadmap details
- **[Research](docs/research/)** — Background research and analysis
- **[Roadmap](docs/roadmap/)** — Feature backlog and vision documents

## Development

```bash
# Backend (with uv)
cd backend
uv sync
python -m app.main

# Frontend
cd ui/core-ui
npm install
npm start        # Angular + Electron
npm run start:ng # Angular only
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full development guidelines.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Development setup
- Branching strategy
- Commit conventions
- Pull request process

## License

[MIT](LICENSE) © Ian Tharp
