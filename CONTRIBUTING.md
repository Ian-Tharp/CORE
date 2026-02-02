# Contributing to CORE

Thank you for your interest in contributing to CORE! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites
- Docker & Docker Compose
- Python 3.12+ with [uv](https://github.com/astral-sh/uv)
- Node.js 22+ with npm
- Git

### Quick Start

```bash
# Clone and start all services
git clone https://github.com/IanTharp/CORE.git
cd CORE
docker compose up -d

# Backend available at http://localhost:8001
# Frontend available at http://localhost:4200
# API docs at http://localhost:8001/docs
```

### Local Development (without Docker)

**Backend:**
```bash
cd backend
uv sync
python -m app.main
```

**Frontend:**
```bash
cd ui/core-ui
npm install
npm start
```

## Branching Strategy

- `main` — stable releases
- `develop` — integration branch
- `feature/*` — new features branch from `develop`
- `fix/*` — bug fixes branch from `develop`

**Never commit directly to `main` or `develop`.** Always use feature branches and pull requests.

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation changes
- `refactor:` — code restructuring
- `test:` — adding or updating tests
- `chore:` — maintenance tasks

## Pull Request Process

1. Create a feature branch from `develop`
2. Make your changes with clear, atomic commits
3. Ensure tests pass
4. Open a PR against `develop`
5. Describe what changed and why

## Code Style

**Python (backend):** Format with `black`, type hints encouraged
**TypeScript (frontend):** Follow Angular style guide, use `npm run lint`

## Architecture

See [docs/](./docs/) for architecture documentation, ADRs, and design docs.

## Questions?

Open an issue or start a discussion. We're happy to help!
