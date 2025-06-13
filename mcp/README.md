# MCP Server Registry

A FastAPI-based registry and router for managing multiple Model Context Protocol (MCP) servers. This service acts as a central hub for discovering, managing, and routing requests to various MCP servers in your infrastructure.

## Architecture Overview

The MCP Registry provides:
- **Service Discovery**: Central registry of all available MCP servers
- **CRUD Operations**: Add, update, delete, and list MCP servers
- **User-specific Configurations**: Per-user server access and preferences
- **Connection Pooling**: Efficient connection management to MCP servers
- **Health Monitoring**: Regular health checks on registered servers
- **Load Balancing**: Distribute requests across multiple server instances
- **Tool Discovery**: Aggregate available tools from all registered servers

### Important: MCP Servers vs REST APIs

MCP servers are **not** REST APIs. They use the Model Context Protocol, which is specifically designed for LLM interactions. Key differences:

- **MCP Servers**: Use FastMCP decorators (`@mcp.tool()`, `@mcp.resource()`, `@mcp.prompt()`)
- **Transport**: Communicate via stdio, SSE, or streamable-http (not traditional REST)
- **Discovery**: Tools and resources are automatically discoverable by LLMs
- **Context**: Can access MCP context for logging, progress reporting, and LLM sampling

The registry acts as a broker between MCP clients (like Claude Desktop or Cursor) and actual MCP servers, managing connections and routing requests appropriately.

## Components

### 1. MCP Registry Service (`mcp_registry_service.py`)
The main FastAPI application that provides:
- REST API for server management
- Database models for persistence
- Health monitoring endpoints

### 2. MCP Client Manager (`mcp_client_manager.py`)
Handles:
- Connection pooling to MCP servers
- Request routing and load balancing
- Tool discovery and execution
- Caching and performance optimization

### 3. Integration Routes (`mcp_integration_routes.py`)
FastAPI routes that can be mounted in your main application to provide:
- Tool discovery across all servers
- Tool execution with user context
- Batch operations for efficiency
- WebSocket support for streaming

## Quick Start

### 1. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the registry service
uvicorn mcp.mcp_registry_service:app --reload

# The service will be available at http://localhost:8000
```

### 2. Docker Deployment

```bash
# Start all services
docker-compose up -d

# Services:
# - MCP Registry: http://localhost:8000
# - PostgreSQL: localhost:5432
# - Redis: localhost:6379
# - pgAdmin: http://localhost:5050
```

### 3. Environment Configuration

Create a `.env` file based on the example:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## API Usage

### Register a New MCP Server

```bash
curl -X POST http://localhost:8000/servers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Weather MCP Server",
    "description": "Provides weather information",
    "url": "http://localhost:8001",
    "server_type": "tools",
    "capabilities": {
      "tools": [
        {
          "name": "get_weather",
          "description": "Get current weather"
        }
      ]
    }
  }'
```

### List All Servers

```bash
curl http://localhost:8000/servers
```

### Configure User Access

```bash
curl -X POST http://localhost:8000/users/servers?user_id=user123 \
  -H "Content-Type: application/json" \
  -d '{
    "server_id": "uuid-here",
    "enabled": true
  }'
```

## Integration with Main Application

In your main CORE backend, integrate the MCP functionality:

```python
from fastapi import FastAPI
from mcp.mcp_integration_routes import router as mcp_router

app = FastAPI()

# Mount MCP routes
app.include_router(mcp_router)
```

Then from your Angular frontend, you can:

1. **Discover available tools**:
```typescript
const response = await fetch('/mcp/discover');
const tools = await response.json();
```

2. **Execute a tool**:
```typescript
const response = await fetch('/mcp/call', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    server_id: 'server-uuid',
    tool_name: 'get_weather',
    parameters: { city: 'New York' }
  })
});
```

## Creating MCP Servers

See `mcp/servers/example_weather_server.py` for a complete example using FastMCP. Key components:

1. Use `@mcp.tool()` decorator to expose functions as tools
2. Use `@mcp.resource()` decorator to expose data resources
3. Use `@mcp.prompt()` decorator to define prompt templates
4. Run with `mcp.run()` using appropriate transport (stdio, sse, streamable-http)

Example:
```python
from fastmcp import FastMCP

mcp = FastMCP("My Server")

@mcp.tool()
async def my_tool(param: str) -> str:
    """Tool description for LLM"""
    return f"Result: {param}"

if __name__ == "__main__":
    mcp.run()  # Default stdio transport
```

## Scaling Considerations

### For Hundreds of MCP Servers:

1. **Database**: Switch from SQLite to PostgreSQL
2. **Caching**: Use Redis for server metadata and tool lists
3. **Connection Pooling**: Adjust `max_connections_per_server` based on load
4. **Load Balancing**: Deploy multiple registry instances behind a load balancer
5. **Health Checks**: Use background tasks to avoid blocking requests
6. **Rate Limiting**: Implement per-user rate limits

### Per-User Configuration:

- Each user can enable/disable specific servers
- Custom configurations per user-server combination
- Usage tracking and quotas
- Role-based access control (implement in authentication layer)

## Monitoring and Maintenance

### Health Monitoring

```bash
# Check all server health
curl -X POST http://localhost:8000/servers/batch-health-check

# Check specific server
curl -X POST http://localhost:8000/servers/{server_id}/health-check
```

### Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head
```

## Security Considerations

1. **Authentication**: Implement proper JWT/session authentication
2. **Authorization**: Validate user access to servers
3. **Input Validation**: All inputs are validated with Pydantic
4. **Rate Limiting**: Configure based on your needs
5. **HTTPS**: Use proper certificates in production
6. **Secrets**: Store sensitive data in environment variables

## Future Enhancements

1. **GraphQL API**: For more flexible querying
2. **gRPC Support**: For high-performance scenarios
3. **Metrics Collection**: Prometheus integration
4. **Admin Dashboard**: Web UI for management
5. **Auto-discovery**: Automatic MCP server detection
6. **Circuit Breakers**: For better fault tolerance 