"""
MCP (Model Context Protocol) Tool Registry Controller

Provides endpoints for discovering and managing MCP tools available to agents.
"""

from typing import Dict, List, Optional, Any
import json
import os
import asyncio
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.auth import require_auth


router = APIRouter(prefix="/mcp", tags=["MCP Tools"])


class MCPTool(BaseModel):
    """Model for an individual MCP tool."""
    name: str
    description: str
    parameters_schema: Optional[Dict[str, Any]] = None
    server: str


class MCPServer(BaseModel):
    """Model for an MCP server configuration."""
    name: str
    type: str
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    status: str = "unknown"
    tools: List[MCPTool] = []


class MCPRegistry(BaseModel):
    """Model for the complete MCP tool registry."""
    servers: List[MCPServer] = []
    total_tools: int = 0


def load_mcp_config() -> Dict[str, Any]:
    """Load MCP configuration from .mcp.json file."""
    config_path = Path(__file__).parent.parent.parent.parent / ".mcp.json"
    
    if not config_path.exists():
        return {"mcpServers": {}}
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading MCP config: {e}")
        return {"mcpServers": {}}


async def discover_server_tools(server_name: str, config: Dict[str, Any]) -> List[MCPTool]:
    """
    Discover available tools for a specific MCP server.
    This is a simplified implementation that would need to be expanded
    to actually communicate with MCP servers using the protocol.
    """
    # Placeholder implementation - in a real scenario, this would:
    # 1. Start the MCP server process
    # 2. Send list_tools request via stdio/websocket
    # 3. Parse the response and return actual tools
    
    # For now, return mock data based on known server types
    mock_tools = {
        "skill-orchestrator": [
            MCPTool(
                name="create_skill",
                description="Create a new skill with code and metadata",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "code": {"type": "string"}
                    }
                },
                server=server_name
            ),
            MCPTool(
                name="execute_skill",
                description="Execute an existing skill",
                parameters_schema={
                    "type": "object", 
                    "properties": {
                        "skill_name": {"type": "string"},
                        "parameters": {"type": "object"}
                    }
                },
                server=server_name
            )
        ],
        "memory": [
            MCPTool(
                name="store_memory",
                description="Store a memory in the vector database",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}}
                    }
                },
                server=server_name
            ),
            MCPTool(
                name="search_memory",
                description="Search stored memories",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"}
                    }
                },
                server=server_name
            )
        ],
        "mcp-obsidian": [
            MCPTool(
                name="create_note",
                description="Create a new note in Obsidian",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                        "path": {"type": "string"}
                    }
                },
                server=server_name
            ),
            MCPTool(
                name="search_notes", 
                description="Search notes in Obsidian vault",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                },
                server=server_name
            )
        ]
    }
    
    return mock_tools.get(server_name, [])


@router.get("/registry", response_model=MCPRegistry)
async def get_mcp_registry(_auth: Dict = Depends(require_auth)) -> MCPRegistry:
    """
    Get the complete MCP tool registry with all available servers and tools.
    """
    config = load_mcp_config()
    servers = []
    total_tools = 0
    
    for server_name, server_config in config.get("mcpServers", {}).items():
        # Discover tools for this server
        tools = await discover_server_tools(server_name, server_config)
        
        server = MCPServer(
            name=server_name,
            type=server_config.get("type", "unknown"),
            command=server_config.get("command"),
            args=server_config.get("args", []),
            env=server_config.get("env", {}),
            status="configured",  # In real implementation, check if server is running
            tools=tools
        )
        
        servers.append(server)
        total_tools += len(tools)
    
    return MCPRegistry(servers=servers, total_tools=total_tools)


@router.get("/servers", response_model=List[MCPServer])
async def list_mcp_servers(_auth: Dict = Depends(require_auth)) -> List[MCPServer]:
    """List all configured MCP servers."""
    registry = await get_mcp_registry()
    return registry.servers


@router.get("/servers/{server_name}/tools", response_model=List[MCPTool])
async def get_server_tools(server_name: str, _auth: Dict = Depends(require_auth)) -> List[MCPTool]:
    """Get all tools available from a specific MCP server."""
    config = load_mcp_config()
    
    if server_name not in config.get("mcpServers", {}):
        raise HTTPException(status_code=404, detail=f"MCP server '{server_name}' not found")
    
    server_config = config["mcpServers"][server_name]
    tools = await discover_server_tools(server_name, server_config)
    
    return tools


@router.get("/tools", response_model=List[MCPTool])
async def list_all_tools(_auth: Dict = Depends(require_auth)) -> List[MCPTool]:
    """Get all tools from all MCP servers."""
    registry = await get_mcp_registry()
    
    all_tools = []
    for server in registry.servers:
        all_tools.extend(server.tools)
    
    return all_tools


@router.get("/tools/search")
async def search_tools(
    q: str = "", 
    server: Optional[str] = None,
    _auth: Dict = Depends(require_auth)
) -> List[MCPTool]:
    """Search tools by name or description."""
    all_tools = await list_all_tools()
    
    # Filter by server if specified
    if server:
        all_tools = [tool for tool in all_tools if tool.server == server]
    
    # Filter by search query if provided
    if q:
        q_lower = q.lower()
        all_tools = [
            tool for tool in all_tools
            if q_lower in tool.name.lower() or q_lower in tool.description.lower()
        ]
    
    return all_tools