"""
Agent MCP Service

Bridges MCP servers with LangChain agents using langchain-mcp-adapters.

This service wraps MultiServerMCPClient to handle both stdio and HTTP transports,
loading MCP tools and making them available to LangGraph agents.

Architecture:
  - Uses MultiServerMCPClient for all transport types (stdio, HTTP, SSE)
  - Loads tools from configured MCP servers
  - Filters tools based on agent configuration
  - Caches tool conversions for performance

For junior developers:
  - MCP servers can run as subprocesses (stdio) or HTTP services
  - MultiServerMCPClient handles the connection details
  - We just configure server params and get tools back
  - Tools are LangChain-compatible and ready for create_react_agent

Performance considerations:
  - Tool conversion cached for 5 minutes
  - Client connections managed by MultiServerMCPClient
  - Selective tool binding (only load what agent needs)
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from langchain_core.tools import BaseTool

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    _HAS_MCP_ADAPTERS = True
except ImportError:
    _HAS_MCP_ADAPTERS = False
    MultiServerMCPClient = None  # type: ignore

from app.models.agent_models import MCPServerConfig

logger = logging.getLogger(__name__)


class AgentMCPService:
    """
    Service for binding MCP tools to LangChain agents.

    Uses MultiServerMCPClient to connect to MCP servers with any transport:
      - stdio: Local subprocess (e.g., Python scripts, Node.js servers)
      - sse: Server-sent events (HTTP streaming)
      - http: Standard HTTP

    Example usage:
        service = AgentMCPService()

        # Get tools for an agent
        agent_mcp_configs = [
            MCPServerConfig(
                server_id="mcp-obsidian",
                tools=["search_nodes", "get_file_contents"],
                config={}
            )
        ]

        tools = await service.get_tools_for_agent(agent_mcp_configs)

        # Use with create_react_agent
        from langgraph.prebuilt import create_react_agent
        agent = create_react_agent(llm, tools)
    """

    def __init__(self):
        """Initialize the MCP service with empty cache."""

        if not _HAS_MCP_ADAPTERS:
            logger.warning(
                "langchain-mcp-adapters not installed. "
                "MCP tool binding will not work. "
                "Install with: uv add langchain-mcp-adapters"
            )

        # Cache structure: {server_id: {tool_name: BaseTool}}
        self._tool_cache: Dict[str, Dict[str, BaseTool]] = {}

        # Cache TTL (5 minutes)
        self._cache_ttl = timedelta(minutes=5)
        self._last_cache_update: Dict[str, datetime] = {}

        # MCP client instance (created on-demand)
        self._mcp_client: Optional[MultiServerMCPClient] = None

        logger.info("AgentMCPService initialized")

    async def get_tools_for_agent(
        self,
        mcp_configs: List[MCPServerConfig]
    ) -> List[BaseTool]:
        """
        Get filtered LangChain tools for an agent.

        Main method for Agent Factory to call. Takes agent's MCP server
        configurations and returns LangChain BaseTool objects.

        Args:
            mcp_configs: List of MCP server configurations from agent

        Returns:
            List of LangChain BaseTool objects

        Example:
            configs = [
                MCPServerConfig(
                    server_id="mcp-obsidian",
                    tools=["search_nodes"],
                    config={}
                )
            ]

            tools = await service.get_tools_for_agent(configs)
        """

        if not _HAS_MCP_ADAPTERS:
            logger.warning("MCP adapters not available, returning empty tool list")
            return []

        all_tools: List[BaseTool] = []

        for mcp_config in mcp_configs:
            server_id = mcp_config.server_id
            allowed_tools = mcp_config.tools

            try:
                # Get all tools from this server (cached or fresh)
                server_tools = await self._get_server_tools(
                    server_id=server_id,
                    config=mcp_config.config
                )

                # Filter to only allowed tools
                filtered_tools = [
                    tool for tool in server_tools
                    if tool.name in allowed_tools
                ]

                all_tools.extend(filtered_tools)

                logger.debug(
                    f"Added {len(filtered_tools)} tools from {server_id} "
                    f"(allowed: {allowed_tools})"
                )

            except Exception as e:
                logger.error(
                    f"Failed to get tools from {server_id}: {e}",
                    exc_info=True
                )
                # Continue with other servers

        logger.info(
            f"Retrieved {len(all_tools)} total tools from "
            f"{len(mcp_configs)} MCP servers"
        )

        return all_tools

    async def _get_server_tools(
        self,
        server_id: str,
        config: Dict[str, Any]
    ) -> List[BaseTool]:
        """
        Get all tools from a specific MCP server.

        Uses cache if available and fresh, otherwise connects to server.

        Args:
            server_id: MCP server identifier
            config: Server-specific configuration

        Returns:
            List of all tools available from this server
        """

        # Check cache first
        if self._is_cache_valid(server_id):
            cached_tools = self._tool_cache[server_id]
            logger.debug(f"Using cached tools for {server_id}")
            return list(cached_tools.values())

        # Cache miss - connect to server
        logger.debug(f"Cache miss for {server_id}, connecting to MCP server...")

        try:
            # Get server configuration
            server_config = self._get_server_config(server_id, config)

            # Check if npx is available (required for stdio MCP servers)
            import shutil
            if server_config.get("transport") == "stdio" and not shutil.which(server_config.get("command", "npx")):
                logger.warning(
                    f"Command '{server_config.get('command', 'npx')}' not found. "
                    f"MCP server {server_id} will not be available. "
                    f"Install Node.js in container to enable MCP tools."
                )
                # Cache empty result to avoid repeated checks
                self._tool_cache[server_id] = {}
                self._last_cache_update[server_id] = datetime.utcnow()
                return []

            # Create client with this server
            client = MultiServerMCPClient({
                server_id: server_config
            })

            # Get tools from client
            tools = await client.get_tools()

            # Update cache
            self._tool_cache[server_id] = {
                tool.name: tool for tool in tools
            }
            self._last_cache_update[server_id] = datetime.utcnow()

            logger.info(f"Discovered {len(tools)} tools from {server_id}")

            return tools

        except Exception as e:
            logger.error(f"Failed to connect to {server_id}: {e}", exc_info=True)
            # Return empty list instead of raising (graceful degradation)
            return []

    def _get_server_config(
        self,
        server_id: str,
        agent_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get MultiServerMCPClient configuration for a server.

        Maps server_id to actual server configuration with transport details.

        Args:
            server_id: Server identifier (e.g., 'mcp-obsidian', 'memory')
            agent_config: Agent-provided configuration overrides

        Returns:
            Configuration dict for MultiServerMCPClient

        Configuration format:
            {
                "command": "python",           # For stdio
                "args": ["/path/to/server.py"],
                "transport": "stdio",
                # OR
                "url": "http://localhost:8100",  # For HTTP/SSE
                "transport": "sse"
            }
        """

        # Hardcoded configurations for common MCP servers
        # TODO: Load from MCP registry or config file

        server_configs = {
            # Obsidian MCP server (stdio transport)
            "mcp-obsidian": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-obsidian",
                    agent_config.get("vault_path", "/mnt/c/Users/Owner/Desktop/Ian's Personal Digital Brain/Digital-Brain/Digital Brain")
                ],
                "transport": "stdio"
            },

            # Memory MCP server (stdio transport)
            "memory": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-memory"
                ],
                "transport": "stdio"
            },

            # Filesystem MCP server (stdio transport)
            "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    agent_config.get("allowed_directories", "/mnt/c/Users/Owner/Desktop/Projects/CORE")
                ],
                "transport": "stdio"
            },
        }

        config = server_configs.get(server_id)

        if not config:
            # Fallback: try to construct from agent config
            logger.warning(
                f"Unknown server_id '{server_id}', using agent config"
            )
            config = agent_config

        return config

    def _is_cache_valid(self, server_id: str) -> bool:
        """
        Check if cached tools for a server are still valid.

        Args:
            server_id: Server to check

        Returns:
            True if cache is valid, False if expired or missing
        """

        if server_id not in self._tool_cache:
            return False

        if server_id not in self._last_cache_update:
            return False

        age = datetime.utcnow() - self._last_cache_update[server_id]
        return age < self._cache_ttl

    def clear_cache(self, server_id: Optional[str] = None):
        """
        Clear tool cache for a specific server or all servers.

        Args:
            server_id: Server to clear, or None for all servers
        """

        if server_id:
            self._tool_cache.pop(server_id, None)
            self._last_cache_update.pop(server_id, None)
            logger.info(f"Cleared tool cache for {server_id}")
        else:
            self._tool_cache.clear()
            self._last_cache_update.clear()
            logger.info("Cleared all tool caches")

    async def list_available_servers(self) -> List[Dict[str, Any]]:
        """
        List all available MCP servers.

        Returns:
            List of server information dictionaries

        Example return:
            [
                {
                    "server_id": "mcp-obsidian",
                    "name": "Obsidian Vault",
                    "description": "Access Obsidian knowledge base",
                    "transport": "stdio"
                },
                ...
            ]
        """

        # TODO: Query actual MCP registry
        # For now return common servers

        common_servers = [
            {
                "server_id": "mcp-obsidian",
                "name": "Obsidian Vault",
                "description": "Access to Digital Brain Obsidian vault",
                "transport": "stdio",
                "available_tools": [
                    "obsidian_search_nodes",
                    "obsidian_get_file_contents",
                    "obsidian_patch_content",
                    "obsidian_list_files_in_vault",
                    "obsidian_simple_search"
                ]
            },
            {
                "server_id": "memory",
                "name": "Memory Graph",
                "description": "Knowledge graph for storing observations",
                "transport": "stdio",
                "available_tools": [
                    "read_graph",
                    "search_nodes",
                    "create_entities",
                    "create_relations",
                    "add_observations"
                ]
            },
            {
                "server_id": "filesystem",
                "name": "Filesystem",
                "description": "Read and write files",
                "transport": "stdio",
                "available_tools": [
                    "read_file",
                    "write_file",
                    "list_directory"
                ]
            }
        ]

        logger.debug(f"Listed {len(common_servers)} available servers")

        return common_servers


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_agent_mcp_service: Optional[AgentMCPService] = None


def get_agent_mcp_service() -> AgentMCPService:
    """
    Get the singleton AgentMCPService instance.

    Returns:
        The singleton AgentMCPService instance
    """

    global _agent_mcp_service

    if _agent_mcp_service is None:
        _agent_mcp_service = AgentMCPService()
        logger.info("Created singleton AgentMCPService instance")

    return _agent_mcp_service
