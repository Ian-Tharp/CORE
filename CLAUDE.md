# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Cognitive CORE is a personal research project for building an AI assistant using a modular cognitive architecture based on the CORE principles: **Comprehension**, **Orchestration**, **Reasoning**, and **Evaluation**. The system is designed as a self-hosted, offline-capable platform with a solarpunk-inspired interface.

## Architecture

This is a multi-component system with the following structure:

### Backend (`/backend`)
- **FastAPI** application serving REST endpoints
- **LangGraph** implementation of the CORE cognitive workflow
- **Python 3.12+** with dependencies managed via `uv`
- Agent-based architecture with individual agents for each CORE component
- System monitoring capabilities using `psutil`

### Frontend (`/ui/core-ui`)
- **Angular 19** application with **Electron** desktop wrapper
- Solarpunk-inspired UI with command deck interface
- **Angular Material** components for consistent design
- Real-time communication with backend services

### MCP Integration (`/mcp`)
- Model Context Protocol servers for external integrations
- Docker-based MCP server orchestration
- Registry service for managing MCP connections

## Development Commands

### Backend Development
```bash
cd backend

# Install dependencies (using uv)
uv sync

# Run development server
python -m app.main
# OR
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload

# Format code
uv run black .
```

### Frontend Development
```bash
cd ui/core-ui

# Install dependencies
npm install

# Start development server (Angular + Electron)
npm start

# Angular development server only
npm run start:ng

# Run Electron only (requires Angular dev server running)
npm run electron

# Build for production
npm run build

# Run tests
npm test

# Lint code
npm run lint
npm run lint:fix
```

## Key Components

### CORE Cognitive Graph
Located in `backend/app/core/langgraph/core_graph.py`, this implements the main workflow:
- **Comprehension**: Interprets user inputs into structured tasks
- **Orchestration**: Coordinates task flows and manages lifecycles  
- **Reasoning**: Applies logic and decision-making to process tasks
- **Evaluation**: Assesses outcomes for quality and relevance

### Agent Controllers
Individual controllers in `backend/app/controllers/` handle:
- `chat.py`: Chat interface endpoints
- `conversations.py`: Conversation management
- `system_monitor.py`: System resource monitoring
- `core_entry.py`: Main CORE system entry points

### Frontend Components
Key Angular components in `ui/core-ui/src/app/`:
- `landing-page/`: Main dashboard and command center
- `agents-page/`: Agent builder and marketplace
- `conversations-page/`: Chat and conversation history
- `shared/`: Reusable components (chat window, navigation)

## Development Workflow

1. **Backend Setup**: Use `uv` for Python dependency management
2. **Frontend Setup**: Standard npm workflow with Angular CLI
3. **CORS Configuration**: Backend allows `http://localhost:4200` for development
4. **API Communication**: Frontend communicates with backend on port 8001
5. **Testing**: Use `npm test` for Angular tests, no Python test framework currently configured

## Important Notes

- The system is designed for local-first, offline operation
- CORE is intended as a neutral cognitive kernel, not a persona
- The architecture supports containerized agent deployment (planned)
- UI follows solarpunk design principles with command deck metaphors
- Backend uses FastAPI lifespan events for proper initialization/shutdown