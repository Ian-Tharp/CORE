# OpenClaw Agent Containerization for CORE

This document explains how to deploy and manage containerized OpenClaw (formerly Clawdbot) agents within the CORE platform for scalable AI agent orchestration.

## Overview

CORE can spawn and manage multiple OpenClaw agent instances as Docker containers, each with their own:
- Unique identity and configuration
- Bot tokens for different platforms (Discord, Telegram, etc.)
- Model preferences (Ollama primary, Claude fallback)
- Isolated workspaces and sessions
- Resource limits and health monitoring

## Architecture

```
┌─────────────┐    ┌─────────────────────────────────┐
│    CORE     │    │     OpenClaw Agents             │
│  Platform   │◄──►│  ┌──────┐ ┌──────┐ ┌──────┐    │
│             │    │  │Agent1│ │Agent2│ │Agent3│    │
│             │    │  │:8001 │ │:8002 │ │:8003 │    │
└─────────────┘    │  └──────┘ └──────┘ └──────┘    │
      │            │            │                   │
      │            │  ┌─────────▼─────────┐         │
      └────────────┼──│  core-network     │         │
                   │  │  (Docker Network) │         │
                   │  └───────────────────┘         │
                   └─────────────────────────────────┘
```

## Quick Start

### 1. Build the Agent Image

```bash
# From CORE project root
cd docker/agent
docker build -t core/openclaw-agent:latest .
```

### 2. Set Up Configuration

```bash
# Create agent config directory structure
mkdir -p config/agents/{agent-001,agent-002,agent-003}
mkdir -p config/shared

# Copy template configurations
cp docker/agent/config.template.yaml config/agents/agent-001/openclaw.json
cp docker/agent/config.template.yaml config/agents/agent-002/openclaw.json
```

### 3. Configure Environment

Create `.env` file with required variables:

```env
# CORE Integration
CORE_ENDPOINT=http://core-backend:8000

# Model APIs  
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here

# Agent Tokens (unique per agent)
DISCORD_BOT_TOKEN_001=your_discord_token_1
DISCORD_BOT_TOKEN_002=your_discord_token_2
TELEGRAM_BOT_TOKEN_001=your_telegram_token_1

# Gateway Authentication
OPENCLAW_GATEWAY_TOKEN=your_secure_token_here
```

### 4. Deploy Agents

```bash
# Start 3 general-purpose agents
docker compose -f docker-compose.agents.yml up --scale agent=3 -d

# Or start specific agent types
docker compose -f docker-compose.agents.yml --profile discord up -d
```

## Configuration Guide

### Agent Roles

Configure different agent types for specialized functions:

#### General Purpose Agent
```yaml
agent:
  role: "general"
  model: "anthropic/claude-sonnet-4"
  capabilities:
    - web_search
    - content_creation  
    - task_automation
```

#### Discord Bot Agent
```yaml
agent:
  role: "discord-bot"
  model: "anthropic/claude-haiku-3-5"  # Faster for chat responses
  capabilities:
    - moderation
    - entertainment
    - utility_commands
```

#### Development Agent  
```yaml
agent:
  role: "development"
  model: "anthropic/claude-opus-4-5"  # Best for coding
  capabilities:
    - code_analysis
    - debugging
    - documentation
```

### Model Configuration

Each agent can use different model preferences:

```yaml
providers:
  # Primary: Local Ollama (fast, private)
  ollama:
    baseURL: "http://core-ollama:11434"
    models: ["llama3.1:latest", "mistral:latest"]
    
  # Secondary: Claude (high quality)
  anthropic:
    models: ["claude-sonnet-4", "claude-opus-4-5"]
    
  # Fallback: OpenAI (reliable)
  openai:
    models: ["gpt-4", "gpt-3.5-turbo"]
```

### Channel Configuration

Configure messaging platform integrations:

```yaml
channels:
  discord:
    token: "${DISCORD_BOT_TOKEN}"
    guilds:
      "YOUR_DISCORD_SERVER_ID":
        activation: "mention"  # Only respond when @mentioned
        
  telegram:
    token: "${TELEGRAM_BOT_TOKEN}"
    
  slack:
    botToken: "${SLACK_BOT_TOKEN}"
    appToken: "${SLACK_APP_TOKEN}"
```

## Deployment Patterns

### Single Agent Instance

For testing or single-bot deployments:

```bash
docker run -d \
  --name openclaw-agent-001 \
  --network core-network \
  -e AGENT_ID=agent-001 \
  -e AGENT_ROLE=general \
  -e DISCORD_BOT_TOKEN=your_token \
  -v $(pwd)/config/agents/agent-001:/home/openclaw/.openclaw \
  core/openclaw-agent:latest
```

### Multi-Agent Scaling

For production deployments with multiple specialized agents:

```yaml
# docker-compose.prod.yml
services:
  # Discord community management
  discord-agents:
    extends:
      service: agent
    deploy:
      replicas: 2
    environment:
      AGENT_ROLE: discord-bot
      
  # Development assistance  
  dev-agents:
    extends: 
      service: agent
    deploy:
      replicas: 1
    environment:
      AGENT_ROLE: development
      
  # General purpose pool
  general-agents:
    extends:
      service: agent  
    deploy:
      replicas: 3
    environment:
      AGENT_ROLE: general
```

### Load Balancer Integration

For high-availability deployments:

```yaml
services:
  agent-proxy:
    image: nginx:alpine
    ports:
      - "18789:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - agent
```

## Monitoring and Management

### Health Checks

Each agent exposes health status:

```bash
# Check individual agent
curl http://localhost:18789/health

# Check all agents via CORE
curl http://core-backend:8000/api/agents/health
```

### Logs and Metrics

Agents send structured logs to CORE for monitoring:

```bash
# View agent logs
docker logs openclaw-agent-001

# Monitor via CORE dashboard
# Navigate to http://core-frontend:3000/agents
```

### Resource Monitoring

Monitor resource usage per agent:

```bash
# Container resource usage
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Agent-specific metrics via CORE API
curl http://core-backend:8000/api/agents/metrics
```

## Security Considerations

### Network Isolation

- Agents run on isolated `core-network`
- No direct external access except through CORE
- Inter-agent communication controlled by CORE

### Resource Limits

```yaml
deploy:
  resources:
    limits:
      memory: 1G      # Prevent memory exhaustion
      cpus: '0.5'     # Limit CPU usage
    reservations:
      memory: 512M    # Guaranteed memory
      cpus: '0.25'    # Guaranteed CPU
```

### Sandboxing

Non-main sessions (groups/channels) run in Docker sandboxes:

```yaml
tools:
  sandbox:
    mode: "non-main"
    scope: "agent" 
    workspaceAccess: "rw"
    docker:
      network: "core-network"
      memory: "512m"
```

### Token Management

- Unique bot tokens per agent instance
- Tokens stored as environment variables
- Rotation supported via CORE configuration

## Troubleshooting

### Common Issues

#### Agent Won't Start
```bash
# Check logs
docker logs openclaw-agent-001

# Common causes:
# - Missing environment variables
# - Invalid configuration
# - Port conflicts
```

#### Can't Connect to CORE
```bash
# Verify network connectivity
docker exec openclaw-agent-001 wget -qO- http://core-backend:8000/health

# Check network configuration
docker network ls | grep core-network
```

#### High Memory Usage
```bash
# Check memory limits
docker stats openclaw-agent-001

# Adjust in docker-compose.agents.yml:
deploy:
  resources:
    limits:
      memory: 2G  # Increase if needed
```

### Debug Mode

Enable detailed logging for troubleshooting:

```yaml
environment:
  NODE_ENV: development
  OPENCLAW_LOG_LEVEL: debug
  OPENCLAW_VERBOSE: true
```

### Configuration Validation

Test configuration before deployment:

```bash
# Validate agent config
docker run --rm \
  -v $(pwd)/config/agents/agent-001:/home/openclaw/.openclaw \
  core/openclaw-agent:latest \
  node dist/index.js doctor
```

## Scaling Strategies

### Horizontal Scaling

Add more agent instances as needed:

```bash
# Scale up during high demand
docker compose -f docker-compose.agents.yml up --scale agent=10 -d

# Scale down during low demand  
docker compose -f docker-compose.agents.yml up --scale agent=3 -d
```

### Vertical Scaling

Adjust resources per agent:

```yaml
deploy:
  resources:
    limits:
      memory: 2G      # Increase for complex tasks
      cpus: '1.0'     # More CPU for faster responses
```

### Auto-scaling with CORE

CORE can automatically scale based on metrics:

```yaml
# In CORE configuration
agents:
  auto_scaling:
    enabled: true
    min_instances: 2
    max_instances: 20
    metrics:
      - cpu_utilization: 70%
      - memory_utilization: 80%
      - request_queue_length: 10
```

## Best Practices

1. **Resource Planning**: Allocate 512MB-1GB RAM per agent
2. **Token Management**: Use unique tokens per agent instance
3. **Monitoring**: Enable health checks and logging
4. **Security**: Run with minimal privileges and sandboxing
5. **Scaling**: Start small and scale based on actual usage
6. **Backup**: Regular backup of agent configurations and workspaces
7. **Updates**: Use rolling updates to maintain availability

## Integration with CORE

### API Endpoints

CORE provides REST APIs for agent management:

```
GET    /api/agents              # List all agents
POST   /api/agents              # Create new agent
GET    /api/agents/{id}         # Get agent details
PUT    /api/agents/{id}         # Update agent config
DELETE /api/agents/{id}         # Remove agent
POST   /api/agents/{id}/scale   # Scale agent instances
```

### Agent Registration

Agents automatically register with CORE on startup:

```yaml
# Agent reports to CORE:
core:
  endpoint: "${CORE_ENDPOINT}"
  agent_id: "${AGENT_ID}" 
  capabilities: ["discord", "web_search", "coding"]
  status: "ready"
```

### Task Distribution

CORE distributes tasks based on agent capabilities:

```python
# CORE task routing example
def route_task(task):
    if task.type == "discord_message":
        return route_to_discord_agents(task)
    elif task.type == "code_review":
        return route_to_dev_agents(task)
    else:
        return route_to_general_agents(task)
```

This containerization setup enables CORE to efficiently manage a fleet of specialized OpenClaw agents, providing scalable AI assistance across multiple platforms and use cases.