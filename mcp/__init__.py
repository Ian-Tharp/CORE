"""
MCP Server Registry Package

A comprehensive system for managing and routing to multiple MCP servers.
"""

from .config import settings
from .mcp_client_manager import MCPClientManager, MCPRequest, MCPResponse
from .mcp_integration_routes import router as mcp_router

__version__ = "1.0.0"
__all__ = [
    "settings",
    "MCPClientManager",
    "MCPRequest",
    "MCPResponse",
    "mcp_router"
] 