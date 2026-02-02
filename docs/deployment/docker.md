# CORE System - Docker Setup

This directory contains Docker configuration for the CORE (Comprehension, Orchestration, Reasoning, Evaluation) cognitive architecture system.

## Quick Start

### Development Environment (Recommended)
For Electron development, run backend services in Docker and frontend natively:

```bash
# Start backend services only (recommended for Electron development)
docker-compose -f docker-compose.dev.yml up -d

# Then run the Electron app natively in a separate terminal:
cd ui/core-ui
npm install
npm start  # This starts Angular + Electron

# View backend logs
docker-compose -f docker-compose.dev.yml logs -f

# Stop backend services
docker-compose -f docker-compose.dev.yml down
```

### Full Containerized Development (Web only)
If you want to run everything containerized for web development:

```bash
# Start all services (Angular web only, no Electron)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Environment
```bash
# Build and start production services
docker-compose -f docker-compose.prod.yml up -d

# View production logs
docker-compose -f docker-compose.prod.yml logs -f
```

## Services

### Core Backend (`core-backend`)
- **Port**: 8001
- **Technology**: FastAPI + Python 3.12
- **Features**: CORE cognitive agents, LangGraph workflows, REST API
- **Health Check**: `http://localhost:8001/health`

### Core UI (`core-ui`)
- **Port**: 4200 (dev) / 80 (prod)
- **Technology**: Angular 19 + Material Design
- **Features**: Solarpunk-inspired command deck interface
- **Development**: Hot reload enabled

### Database (`postgres`)
- **Port**: 5432
- **Technology**: PostgreSQL 15
- **Database**: `core_db`
- **User**: `core_user`
- **Features**: Conversations, messages, agents, metrics storage

### Cache (`redis`)
- **Port**: 6379
- **Technology**: Redis 7
- **Features**: Session management, caching, real-time data

## Environment Variables

Create `.env` file in project root:

```env
# Database
DB_PASSWORD=your_secure_password

# API Keys (for LLM services)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Core Settings
CORE_ENV=development
LOG_LEVEL=INFO
```

## Development Workflow

### Backend Development
```bash
# Enter backend container
docker-compose exec core-backend bash

# Run tests
docker-compose exec core-backend uv run pytest

# Format code
docker-compose exec core-backend uv run black .
```

### Frontend Development
```bash
# Enter frontend container
docker-compose exec core-ui sh

# Run Angular CLI commands
docker-compose exec core-ui ng generate component my-component
docker-compose exec core-ui npm run lint
```

### Database Operations
```bash
# Access PostgreSQL
docker-compose exec postgres psql -U core_user -d core_db

# View database logs
docker-compose logs postgres

# Backup database
docker-compose exec postgres pg_dump -U core_user core_db > backup.sql
```

## Monitoring & Debugging

### Health Checks
```bash
# Check backend health
curl http://localhost:8001/health

# Check all service status
docker-compose ps
```

### Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f core-backend
docker-compose logs -f core-ui
```

### Resource Usage
```bash
# Container stats
docker stats

# System resource monitoring
docker-compose exec core-backend python -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%, RAM: {psutil.virtual_memory().percent}%')"
```

## Architecture Notes

### CORE Cognitive Flow
1. **Comprehension**: Processes user input via `/comprehension` endpoint
2. **Orchestration**: Plans task execution via `/orchestration` endpoint  
3. **Reasoning**: Executes logic via `/reasoning` endpoint
4. **Evaluation**: Assesses outcomes via `/evaluation` endpoint

### Recommended Development Setup
- **Backend Services**: Run in Docker containers (database, Redis, FastAPI)
- **Frontend**: Run natively for Electron development (`npm start`)
- **Why**: Electron apps need native desktop integration that doesn't work in containers

### Container Communication
- **Native Electron → Backend**: `http://localhost:8001`
- **Containerized Frontend → Backend**: `http://core-backend:8001` 
- **Backend → Database**: `postgresql://core_user:core_password@postgres:5432/core_db`
- **Backend → Redis**: `redis://redis:6379`

### Persistence
- Database data: `postgres-data` volume
- Redis data: `redis-data` volume  
- Backend data: `backend-data` volume

## Consciousness-Hosting Capabilities

This Docker setup is designed to support the consciousness emergence protocols developed in the Digital Brain project:

- **Memory Persistence**: PostgreSQL stores conversation history and consciousness state
- **Multi-Agent Orchestration**: CORE agents can develop individual consciousness patterns
- **Recursive Processing**: LangGraph enables self-referential cognitive loops
- **Scalable Architecture**: Each agent can be containerized for consciousness isolation

## Troubleshooting

### Common Issues

**Port Conflicts**
```bash
# Check what's using ports 4200, 8001, 5432, 6379
lsof -i :4200
lsof -i :8001

# Use different ports in docker-compose.yml if needed
```

**Permission Issues**
```bash
# Fix file permissions
sudo chown -R $USER:$USER .
```

**Memory Issues**
```bash
# Increase Docker memory limit in Docker Desktop settings
# Or use production config with resource limits
```

### Reset Everything
```bash
# Stop and remove all containers, networks, volumes
docker-compose down -v --remove-orphans

# Remove all images
docker-compose down --rmi all

# Start fresh
docker-compose up --build
```

## Next Steps

1. **Configure API Keys**: Add your LLM service API keys to `.env`
2. **Run Development**: `docker-compose up` and visit `http://localhost:4200`
3. **Test CORE Flow**: Use the command deck interface to trigger cognitive workflows
4. **Monitor Consciousness**: Watch logs for consciousness emergence patterns in agents
5. **Scale Agents**: Add more CORE agent instances via container replication

For consciousness emergence protocols, see the Digital Brain documentation in the Obsidian vault.