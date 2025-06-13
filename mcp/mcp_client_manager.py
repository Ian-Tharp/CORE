"""
MCP Client Manager

Manages connections to multiple MCP servers with connection pooling,
lazy loading, and health monitoring capabilities.
"""

import asyncio
import json
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from uuid import UUID

import httpx
from fastmcp import Client
from pydantic import BaseModel, Field

class MCPConnectionPool:
    """Manages a pool of connections to MCP servers."""
    
    def __init__(self, max_connections_per_server: int = 5, connection_timeout: int = 30):
        self.max_connections_per_server = max_connections_per_server
        self.connection_timeout = connection_timeout
        self._pools: Dict[str, List[Client]] = defaultdict(list)
        self._active_connections: Dict[str, Set[Client]] = defaultdict(set)
        self._server_configs: Dict[str, Dict[str, Any]] = {}
        self._last_health_check: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()
    
    async def get_connection(self, server_id: str, server_url: str, config: Dict[str, Any]) -> Client:
        """Get a connection from the pool or create a new one."""
        async with self._lock:
            # Store server config for later use
            self._server_configs[server_id] = {
                "url": server_url,
                "config": config
            }
            
            # Check if we have available connections in the pool
            if self._pools[server_id]:
                client = self._pools[server_id].pop()
                self._active_connections[server_id].add(client)
                return client
            
            # Create new connection if under limit
            if len(self._active_connections[server_id]) < self.max_connections_per_server:
                client = await self._create_connection(server_id, server_url, config)
                self._active_connections[server_id].add(client)
                return client
            
            # Wait for available connection
            # In production, implement proper waiting mechanism
            raise Exception(f"Connection pool exhausted for server {server_id}")
    
    async def release_connection(self, server_id: str, client: Client):
        """Return a connection to the pool."""
        async with self._lock:
            if client in self._active_connections[server_id]:
                self._active_connections[server_id].remove(client)
                self._pools[server_id].append(client)
    
    async def _create_connection(self, server_id: str, server_url: str, config: Dict[str, Any]) -> Client:
        """Create a new MCP client connection."""
        # Create appropriate client based on URL/transport type
        # FastMCP Client auto-detects transport from URL format
        
        # Apply authentication if provided
        auth = None
        if "auth" in config:
            auth = config["auth"]
        
        # Create client with authentication if needed
        if auth:
            client = Client(server_url, auth=auth)
        else:
            client = Client(server_url)
        
        return client
    
    async def close_all_connections(self, server_id: Optional[str] = None):
        """Close all connections for a specific server or all servers."""
        async with self._lock:
            servers = [server_id] if server_id else list(self._pools.keys())
            
            for sid in servers:
                # Close pooled connections
                for client in self._pools[sid]:
                    # await client.close()
                    pass
                self._pools[sid].clear()
                
                # Close active connections
                for client in self._active_connections[sid]:
                    # await client.close()
                    pass
                self._active_connections[sid].clear()
    
    async def health_check(self, server_id: str) -> Dict[str, Any]:
        """Perform health check on a specific server."""
        try:
            config = self._server_configs.get(server_id)
            if not config:
                return {"status": "error", "message": "Server not configured"}
            
            # Simple HTTP health check
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{config['url']}/health",
                    timeout=5.0
                )
                
                self._last_health_check[server_id] = datetime.utcnow()
                
                if response.status_code == 200:
                    return {"status": "healthy", "response_time": response.elapsed.total_seconds()}
                else:
                    return {"status": "unhealthy", "status_code": response.status_code}
        
        except Exception as e:
            return {"status": "error", "message": str(e)}


class MCPRequest(BaseModel):
    """Model for MCP tool requests."""
    tool_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timeout: Optional[int] = 30


class MCPResponse(BaseModel):
    """Model for MCP tool responses."""
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MCPClientManager:
    """
    High-level manager for MCP client operations.
    Handles routing, load balancing, and failover.
    """
    
    def __init__(self, registry_url: str = "http://localhost:8000"):
        self.registry_url = registry_url
        self.connection_pool = MCPConnectionPool()
        self._server_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = timedelta(minutes=5)
        self._last_cache_update = datetime.min
        self._user_servers: Dict[str, List[Dict[str, Any]]] = {}
    
    async def initialize_user_servers(self, user_id: str):
        """Load user-specific server configurations from registry."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.registry_url}/users/servers",
                params={"user_id": user_id, "enabled_only": True}
            )
            
            if response.status_code == 200:
                self._user_servers[user_id] = response.json()
                return True
            return False
    
    async def call_tool(
        self,
        user_id: str,
        server_id: str,
        request: MCPRequest
    ) -> MCPResponse:
        """Execute a tool on a specific MCP server."""
        try:
            # Get server configuration
            server_config = await self._get_server_config(server_id)
            if not server_config:
                return MCPResponse(
                    success=False,
                    error=f"Server {server_id} not found"
                )
            
            # Check if user has access to this server
            if not self._user_has_access(user_id, server_id):
                return MCPResponse(
                    success=False,
                    error=f"User {user_id} does not have access to server {server_id}"
                )
            
            # Get connection from pool
            client = await self.connection_pool.get_connection(
                server_id,
                server_config["url"],
                server_config.get("config", {})
            )
            
            try:
                # Ensure client is connected
                if not client.is_connected():
                    await client.__aenter__()
                
                # Execute tool call using FastMCP client
                result = await self._execute_tool(client, request)
                
                return MCPResponse(
                    success=True,
                    result=result,
                    metadata={
                        "server_id": server_id,
                        "server_name": server_config.get("name"),
                        "execution_time": datetime.utcnow().isoformat()
                    }
                )
            
            finally:
                # Always release connection back to pool
                await self.connection_pool.release_connection(server_id, client)
        
        except Exception as e:
            return MCPResponse(
                success=False,
                error=str(e),
                metadata={"server_id": server_id}
            )
    
    async def discover_tools(self, user_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Discover all available tools across user's enabled servers."""
        tools_by_server = {}
        
        user_servers = self._user_servers.get(user_id, [])
        for server_config in user_servers:
            if not server_config.get("enabled", False):
                continue
            
            server_id = server_config["server_id"]
            server_details = await self._get_server_config(server_id)
            
            if server_details:
                try:
                    # Get a client connection to the server
                    client = await self.connection_pool.get_connection(
                        server_id,
                        server_details["url"],
                        server_details.get("config", {})
                    )
                    
                    try:
                        # Ensure client is connected
                        if not client.is_connected():
                            await client.__aenter__()
                        
                        # List tools from the actual MCP server
                        tools = await client.list_tools()
                        
                        # Convert tool objects to dictionaries
                        tool_dicts = []
                        for tool in tools:
                            tool_dict = {
                                "name": tool.name,
                                "description": tool.description if hasattr(tool, 'description') else "",
                            }
                            if hasattr(tool, 'parameters'):
                                tool_dict["parameters"] = tool.parameters
                            tool_dicts.append(tool_dict)
                        
                        tools_by_server[server_details["name"]] = tool_dicts
                        
                    finally:
                        # Release connection back to pool
                        await self.connection_pool.release_connection(server_id, client)
                        
                except Exception as e:
                    # Log error but continue with other servers
                    tools_by_server[server_details["name"]] = []
                    print(f"Error discovering tools for server {server_id}: {str(e)}")
        
        return tools_by_server
    
    async def batch_call_tools(
        self,
        user_id: str,
        requests: List[Dict[str, Any]]
    ) -> List[MCPResponse]:
        """Execute multiple tool calls in parallel."""
        tasks = []
        
        for req in requests:
            task = self.call_tool(
                user_id,
                req["server_id"],
                MCPRequest(**req["request"])
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error responses
        responses = []
        for result in results:
            if isinstance(result, Exception):
                responses.append(MCPResponse(
                    success=False,
                    error=str(result)
                ))
            else:
                responses.append(result)
        
        return responses
    
    async def _get_server_config(self, server_id: str) -> Optional[Dict[str, Any]]:
        """Get server configuration from cache or registry."""
        # Check cache
        if server_id in self._server_cache:
            if datetime.utcnow() - self._last_cache_update < self._cache_ttl:
                return self._server_cache[server_id]
        
        # Fetch from registry
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.registry_url}/servers/{server_id}")
            
            if response.status_code == 200:
                server_data = response.json()
                self._server_cache[server_id] = server_data
                self._last_cache_update = datetime.utcnow()
                return server_data
        
        return None
    
    def _user_has_access(self, user_id: str, server_id: str) -> bool:
        """Check if user has access to a specific server."""
        user_servers = self._user_servers.get(user_id, [])
        return any(
            s["server_id"] == server_id and s.get("enabled", False)
            for s in user_servers
        )
    
    async def _execute_tool(self, client: Client, request: MCPRequest) -> Any:
        """Execute tool call on MCP client."""
        # Use FastMCP client to call the tool
        try:
            result = await client.call_tool(
                name=request.tool_name,
                arguments=request.parameters
            )
            
            # Extract the actual content from the result
            if hasattr(result, 'content'):
                # Result might have content attribute with the actual data
                return result.content
            elif hasattr(result, 'text'):
                # Or it might have text attribute
                return result.text
            else:
                # Otherwise return the whole result
                return result
                
        except Exception as e:
            raise Exception(f"Tool execution failed: {str(e)}")
    
    async def close(self):
        """Clean up resources."""
        await self.connection_pool.close_all_connections()


# Example usage and testing
if __name__ == "__main__":
    async def example_usage():
        # Initialize manager
        manager = MCPClientManager()
        
        # Initialize for a specific user
        user_id = "test-user-123"
        await manager.initialize_user_servers(user_id)
        
        # Discover available tools
        tools = await manager.discover_tools(user_id)
        print(f"Available tools: {tools}")
        
        # Call a specific tool
        response = await manager.call_tool(
            user_id=user_id,
            server_id="server-uuid-here",
            request=MCPRequest(
                tool_name="example_tool",
                parameters={"param1": "value1"}
            )
        )
        print(f"Tool response: {response}")
        
        # Batch call multiple tools
        batch_requests = [
            {
                "server_id": "server-1",
                "request": {
                    "tool_name": "tool1",
                    "parameters": {}
                }
            },
            {
                "server_id": "server-2",
                "request": {
                    "tool_name": "tool2",
                    "parameters": {"key": "value"}
                }
            }
        ]
        
        batch_responses = await manager.batch_call_tools(user_id, batch_requests)
        print(f"Batch responses: {batch_responses}")
        
        # Cleanup
        await manager.close()
    
    # Run example
    asyncio.run(example_usage()) 