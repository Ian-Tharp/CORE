"""
MCP Server Registry Package

A comprehensive system for managing and routing to multiple MCP servers.
"""

from .config import settings
from .mcp_client_manager import MCPClientManager, MCPRequest, MCPResponse
from .mcp_integration_routes import router as mcp_router
from .dynamic_tool_selector import (
    DynamicToolSelector,
    ToolSelection,
    ToolDefinition,
    TaskAnalysis,
    Capability
)

__version__ = "1.0.0"
__all__ = [
    "settings",
    "MCPClientManager",
    "MCPRequest",
    "MCPResponse",
    "mcp_router",
    # Tool Selection
    "DynamicToolSelector",
    "ToolSelection",
    "ToolDefinition",
    "TaskAnalysis",
    "Capability",
] 