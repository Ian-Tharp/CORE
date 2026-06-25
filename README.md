# CORE — Cognitive Orchestration, Reasoning & Evaluation

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![Angular](https://img.shields.io/badge/angular-19-red.svg)
![Docker](https://img.shields.io/badge/docker-compose-blue.svg)

**CORE** is a modular, self-hosted AI orchestration platform built around four cognitive pillars: **Comprehension**, **Orchestration**, **Reasoning**, and **Evaluation**. It provides a multi-agent architecture with a cognitive pipeline, agent factory, communication commons, and a solarpunk-inspired desktop UI.

My personal definition of AGI is already satisfied. The different form factor of intelligence that is enabled through LLM technology is vastly more intelligent than I could hope to become in my lifetime (disregarding possibilities of longevity escape velocity). LLMs hold more expertise in so many different fields and domains than I could truly and deeply understand at that same level of comprehension. The industry's definition will always be a moving goalpost.

It stands for Comprehension, Orchestration, Reasoning, and Evaluation — the four pillars that provide a modular and scalable foundation for eventual autonomous task execution and decision-making using natural language inputs.

The system is structured around the CORE principles, with each component serving a specific function in the task resolution process:

- **Comprehension**: Interprets user inputs and transforms them into structured tasks.
- **Orchestration**: Coordinates task flows between system components and manages the lifecycle of tasks.
- **Reasoning**: Applies logic and decision-making to process tasks and derive solutions.
- **Evaluation**: Assesses the outcomes of tasks for quality assurance and relevance to the original input.

![CORE Architecture](assets/imgs/COREv1%20Diagram.png)

## The deliberation loop (two verifiers)

CORE is not a single forward pass — it is a **deliberation loop bracketed by two verifiers**.
**Comprehension** verifies *up front* that the system understands the request and actually has the
capabilities to attempt it. **Evaluation** verifies *after* that the result achieved the intent,
looping back to retry a step or revise the plan when it didn't. The loop generalizes to the degree
those two verifiers are good.

```
   user input
       |
       v
  +----------------+  proceed   +----------------+    +-----------+    +----------------+  finalize
  | COMPREHENSION  |----------->| ORCHESTRATION  |--->| REASONING |--->|  EVALUATION    |---------> response
  | front verifier |            | sequencer      |    | executor  |    | back verifier  |
  +----------------+            +----------------+    +-----------+    +----------------+
       |    |                         ^                     ^                  |    |
       |    |          revise plan    |                     |   retry step     |    |
       |    |    (goal not achieved)  +---------------------|------------------+    |
       |    |                                               |  (execution flaked)   |
       |    |                                               +-----------------------+
       |    |
       |    +-- clarify --> ask the user: "do you mean X / Y / Z?  (here's what I can do)"
       +------- refuse  --> honest:       "I can't do that -- but here's what I CAN do"
                            clarify & refuse answer directly -- the O->R->E loop never runs,
                            which makes the front gate the cheapest place to run a verifier.
```

**Comprehension is grounded in a self-model** — it reasons about intent using *what it can do* (a
Capability Registry) and *what it knows* (a memory summary), not a blind classifier:

```
       what it CAN do                          what it KNOWS
  +--------------------------+        +----------------------------+
  |   Capability Registry    |        |      Memory summary        |
  |  tools + MCP, with        |        |  recent asks, session      |
  |  descriptions + schemas   |        |  history, tool activity    |
  +------------+-------------+        +-------------+--------------+
               | presence retrieval                | injected context
               v                                   v
            +-------------------------------------------+
            |               COMPREHENSION               |
            |     intent + feasibility + ambiguity      |
            +---------------------+---------------------+
                                  v
                  +--------------------------------+
                  |          tri-state gate        |
                  |   proceed | clarify | refuse   |
                  +--------------------------------+
```

Each node owns exactly one question — keeping them separate is what stops Comprehension from
becoming a second Evaluator (which causes over-conservative refusals):

| Node | The one question it answers | Verifies |
|------|------------------------------|----------|
| **Comprehension** | "Do I understand this, and do the *primitives* exist to attempt it?" | capability **presence** |
| **Orchestration** | "Can I *wire* those primitives into an ordered plan?" | **composability** |
| **Reasoning** | "Execute each step." | — |
| **Evaluation** | "Did the result actually *satisfy the intent*?" | **achievement** |

> Design detail and the phased build live in
> [`docs/architecture/comprehension-grounded-gate.md`](docs/architecture/comprehension-grounded-gate.md).

## Features

- **Cognitive Pipeline** — LangGraph-based workflow: Comprehension → Orchestration → Reasoning → Evaluation
- **Agent Factory** — Define, instantiate, and manage AI agents with configurable personalities and MCP tool bindings
- **Communication Commons** — Real-time multi-agent chat with channels, threads, reactions, and presence
- **Council of Perspectives** — Multi-agent deliberation framework for structured analysis
- **Catalyst Engine** — Creative divergence-convergence workflows
- **MCP Integration** — Model Context Protocol servers for external tool access
- **Multi-Provider Chat** — Stream from OpenAI, Anthropic (incl. Claude Haiku 4.5), or local models (Ollama / LM Studio)
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

> **Local models & optional services:** CORE defaults to **LM Studio** as its local provider
> (`CORE_LOCAL_PROVIDER`, reached at `host.docker.internal:1234`). The `ollama` and `n8n`
> containers are **opt-in** via compose profiles — e.g. `docker compose --profile ollama up -d`.
> Copy `.env.example` → `.env` for per-machine values (model dirs, provider, keys).

See [docs/deployment/docker.md](docs/deployment/docker.md) for detailed Docker configuration and production setup,
[docs/deployment/local-llm-providers.md](docs/deployment/local-llm-providers.md) for local models, and
[docs/deployment/chat-providers.md](docs/deployment/chat-providers.md) for OpenAI/Anthropic/local chat selection.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Frontend (Angular 19 + Electron)   │
│  Landing Page · Agent Library · Communication Commons │
│  Command Deck · Engine Playground · Planet Lab        │
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
     PostgreSQL      Redis     Ollama / LM Studio
     (Storage)     (Cache/PubSub)  (Local LLM)
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.12+, FastAPI, LangGraph, LangChain |
| Frontend | Angular 19, Electron, Angular Material |
| Database | PostgreSQL with pgvector |
| Cache | Redis |
| Local LLM | Ollama or LM Studio (via `CORE_LOCAL_PROVIDER`) |
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
