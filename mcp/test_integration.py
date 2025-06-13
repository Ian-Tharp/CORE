"""
Test Integration Script

Demonstrates how the MCP registry integrates with actual MCP servers.
This script shows the complete flow from server registration to tool execution.
"""

import asyncio
import json
from datetime import datetime
from uuid import uuid4

import httpx
from fastmcp import Client


async def test_registry_integration():
    """Test the complete MCP registry integration flow."""
    
    registry_url = "http://localhost:8000"
    
    # Step 1: Register an MCP server with the registry
    print("1. Registering MCP server with registry...")
    
    async with httpx.AsyncClient() as http_client:
        # Register our example weather server
        server_data = {
            "name": "Weather MCP Server",
            "description": "Provides weather information and forecasts",
            "url": "http://localhost:8001/sse",  # SSE transport
            "server_type": "tools",
            "capabilities": {
                "tools": [
                    {
                        "name": "get_current_weather",
                        "description": "Get current weather for a city"
                    },
                    {
                        "name": "get_weather_forecast", 
                        "description": "Get weather forecast"
                    },
                    {
                        "name": "compare_weather",
                        "description": "Compare weather between cities"
                    }
                ]
            },
            "health_check_url": "http://localhost:8001/health"
        }
        
        response = await http_client.post(
            f"{registry_url}/servers",
            json=server_data
        )
        
        if response.status_code == 200:
            server_info = response.json()
            server_id = server_info["id"]
            print(f"✓ Server registered with ID: {server_id}")
        else:
            print(f"✗ Failed to register server: {response.text}")
            return
    
    # Step 2: Configure user access to the server
    print("\n2. Configuring user access...")
    
    user_id = "test-user-123"
    
    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            f"{registry_url}/users/servers?user_id={user_id}",
            json={
                "server_id": server_id,
                "enabled": True,
                "custom_config": {}
            }
        )
        
        if response.status_code == 200:
            print(f"✓ User {user_id} configured for server access")
        else:
            print(f"✗ Failed to configure user: {response.text}")
    
    # Step 3: Use the MCP integration endpoints to discover and call tools
    print("\n3. Testing MCP integration endpoints...")
    
    # Initialize user session
    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            f"{registry_url}/mcp/initialize",
            params={"user_id": user_id}
        )
        print(f"✓ User session initialized")
    
    # Discover available tools
    async with httpx.AsyncClient() as http_client:
        response = await http_client.get(
            f"{registry_url}/mcp/discover",
            params={"user_id": user_id}
        )
        
        if response.status_code == 200:
            tools_data = response.json()
            print(f"✓ Discovered {tools_data['total_tools']} tools")
            print(f"  Servers: {list(tools_data['servers'].keys())}")
    
    # Call a tool through the registry
    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            f"{registry_url}/mcp/call",
            params={"user_id": user_id},
            json={
                "server_id": server_id,
                "tool_name": "get_current_weather",
                "parameters": {"city": "New York"}
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Tool call successful:")
            print(f"  Result: {json.dumps(result['result'], indent=2)}")
        else:
            print(f"✗ Tool call failed: {response.text}")
    
    # Step 4: Direct MCP client connection (bypassing registry)
    print("\n4. Testing direct MCP client connection...")
    
    # This demonstrates how the registry's client manager works internally
    weather_server_url = "http://localhost:8001/sse"
    
    try:
        # Create a direct client to the MCP server
        client = Client(weather_server_url)
        
        async with client:
            # List available tools
            tools = await client.list_tools()
            print(f"✓ Connected directly to MCP server")
            print(f"  Available tools: {[tool.name for tool in tools]}")
            
            # Call a tool directly
            result = await client.call_tool(
                "get_current_weather",
                {"city": "London"}
            )
            print(f"✓ Direct tool call result: {result}")
            
    except Exception as e:
        print(f"✗ Direct connection failed: {str(e)}")
        print("  (Make sure the weather server is running on port 8001)")
    
    # Step 5: Test batch operations
    print("\n5. Testing batch tool calls...")
    
    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            f"{registry_url}/mcp/batch-call",
            params={"user_id": user_id},
            json={
                "calls": [
                    {
                        "server_id": server_id,
                        "tool_name": "get_current_weather",
                        "parameters": {"city": "Paris"}
                    },
                    {
                        "server_id": server_id,
                        "tool_name": "get_current_weather",
                        "parameters": {"city": "Tokyo"}
                    }
                ]
            }
        )
        
        if response.status_code == 200:
            results = response.json()
            print(f"✓ Batch call successful: {len(results)} results")
        else:
            print(f"✗ Batch call failed: {response.text}")


async def run_weather_server():
    """Run the example weather server for testing."""
    import sys
    import subprocess
    
    print("Starting example weather server...")
    
    # Run the weather server with SSE transport
    process = subprocess.Popen([
        sys.executable,
        "-c",
        """
import sys
sys.path.insert(0, '.')
from mcp.servers.example_weather_server import mcp
mcp.run(transport='sse', port=8001)
"""
    ])
    
    # Give the server time to start
    await asyncio.sleep(2)
    
    return process


async def main():
    """Main test function."""
    print("MCP Registry Integration Test")
    print("=" * 50)
    
    # Start the weather server
    weather_process = await run_weather_server()
    
    try:
        # Run the integration test
        await test_registry_integration()
        
    finally:
        # Clean up
        print("\nCleaning up...")
        weather_process.terminate()
        weather_process.wait()
    
    print("\nTest complete!")


if __name__ == "__main__":
    asyncio.run(main()) 