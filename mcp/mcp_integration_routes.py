"""
MCP Integration Routes

FastAPI routes that can be mounted in the main CORE backend
to provide MCP functionality through the registry.
"""

from typing import List, Dict, Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from .mcp_client_manager import MCPClientManager, MCPRequest, MCPResponse

# Initialize router
router = APIRouter(prefix="/mcp", tags=["MCP Integration"])

# Global client manager instance
# In production, this should be properly initialized with dependency injection
client_manager = MCPClientManager(registry_url="http://localhost:8000")


# Request/Response Models
class ToolCallRequest(BaseModel):
    """Request model for calling an MCP tool."""
    server_id: UUID
    tool_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timeout: Optional[int] = 30


class BatchToolCallRequest(BaseModel):
    """Request model for batch tool calls."""
    calls: List[ToolCallRequest]


class ToolDiscoveryResponse(BaseModel):
    """Response model for tool discovery."""
    servers: Dict[str, List[Dict[str, Any]]]
    total_tools: int


class ServerCapability(BaseModel):
    """Model for server capabilities."""
    server_id: UUID
    server_name: str
    server_type: str
    tools: List[Dict[str, Any]]
    status: str


# Dependency to get current user
# This should be replaced with your actual authentication dependency
async def get_current_user():
    """Get the current authenticated user."""
    # Placeholder - replace with actual auth logic
    return {"user_id": "test-user-123"}


# Routes
@router.post("/initialize")
async def initialize_user_session(
    background_tasks: BackgroundTasks,
    user: Dict = Depends(get_current_user)
):
    """
    Initialize MCP session for the current user.
    Loads user's enabled servers and prepares connections.
    """
    user_id = user["user_id"]
    
    # Initialize in background to avoid blocking
    background_tasks.add_task(
        client_manager.initialize_user_servers,
        user_id
    )
    
    return {
        "message": "MCP session initialization started",
        "user_id": user_id
    }


@router.get("/discover", response_model=ToolDiscoveryResponse)
async def discover_available_tools(
    user: Dict = Depends(get_current_user)
):
    """
    Discover all available tools across user's enabled MCP servers.
    """
    user_id = user["user_id"]
    
    # Ensure user servers are initialized
    if user_id not in client_manager._user_servers:
        await client_manager.initialize_user_servers(user_id)
    
    tools_by_server = await client_manager.discover_tools(user_id)
    
    # Count total tools
    total_tools = sum(len(tools) for tools in tools_by_server.values())
    
    return ToolDiscoveryResponse(
        servers=tools_by_server,
        total_tools=total_tools
    )


@router.post("/call", response_model=MCPResponse)
async def call_mcp_tool(
    request: ToolCallRequest,
    user: Dict = Depends(get_current_user)
):
    """
    Call a specific tool on an MCP server.
    """
    user_id = user["user_id"]
    
    # Ensure user servers are initialized
    if user_id not in client_manager._user_servers:
        await client_manager.initialize_user_servers(user_id)
    
    # Create MCP request
    mcp_request = MCPRequest(
        tool_name=request.tool_name,
        parameters=request.parameters,
        timeout=request.timeout
    )
    
    # Execute tool call
    response = await client_manager.call_tool(
        user_id=user_id,
        server_id=str(request.server_id),
        request=mcp_request
    )
    
    if not response.success:
        raise HTTPException(status_code=400, detail=response.error)
    
    return response


@router.post("/batch-call", response_model=List[MCPResponse])
async def batch_call_tools(
    request: BatchToolCallRequest,
    user: Dict = Depends(get_current_user)
):
    """
    Execute multiple tool calls in parallel across different MCP servers.
    """
    user_id = user["user_id"]
    
    # Ensure user servers are initialized
    if user_id not in client_manager._user_servers:
        await client_manager.initialize_user_servers(user_id)
    
    # Prepare batch requests
    batch_requests = [
        {
            "server_id": str(call.server_id),
            "request": {
                "tool_name": call.tool_name,
                "parameters": call.parameters,
                "timeout": call.timeout
            }
        }
        for call in request.calls
    ]
    
    # Execute batch calls
    responses = await client_manager.batch_call_tools(user_id, batch_requests)
    
    return responses


@router.get("/capabilities", response_model=List[ServerCapability])
async def get_server_capabilities(
    user: Dict = Depends(get_current_user)
):
    """
    Get detailed capabilities of all user's enabled MCP servers.
    """
    user_id = user["user_id"]
    
    # Ensure user servers are initialized
    if user_id not in client_manager._user_servers:
        await client_manager.initialize_user_servers(user_id)
    
    capabilities = []
    user_servers = client_manager._user_servers.get(user_id, [])
    
    for server_config in user_servers:
        if not server_config.get("enabled", False):
            continue
        
        server_details = await client_manager._get_server_config(
            server_config["server_id"]
        )
        
        if server_details:
            capabilities.append(ServerCapability(
                server_id=server_config["server_id"],
                server_name=server_details.get("name", "Unknown"),
                server_type=server_details.get("server_type", "custom"),
                tools=server_details.get("capabilities", {}).get("tools", []),
                status=server_details.get("status", "unknown")
            ))
    
    return capabilities


@router.post("/health-check/{server_id}")
async def check_server_health(
    server_id: UUID,
    user: Dict = Depends(get_current_user)
):
    """
    Perform health check on a specific MCP server.
    """
    user_id = user["user_id"]
    
    # Check if user has access to this server
    if not client_manager._user_has_access(user_id, str(server_id)):
        raise HTTPException(
            status_code=403,
            detail="User does not have access to this server"
        )
    
    health_status = await client_manager.connection_pool.health_check(str(server_id))
    
    return {
        "server_id": server_id,
        "health_status": health_status,
        "timestamp": health_status.get("last_check")
    }


# WebSocket endpoint for real-time tool streaming (if needed)
from fastapi import WebSocket, WebSocketDisconnect
import json

@router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time MCP tool streaming.
    Useful for long-running tools or tools that produce streaming output.
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message["type"] == "authenticate":
                # Handle authentication
                user_id = message.get("user_id")
                await client_manager.initialize_user_servers(user_id)
                await websocket.send_json({
                    "type": "authenticated",
                    "user_id": user_id
                })
            
            elif message["type"] == "tool_call":
                # Handle tool call
                user_id = message.get("user_id")
                server_id = message.get("server_id")
                tool_request = MCPRequest(**message.get("request", {}))
                
                # Execute tool call
                response = await client_manager.call_tool(
                    user_id=user_id,
                    server_id=server_id,
                    request=tool_request
                )
                
                # Send response
                await websocket.send_json({
                    "type": "tool_response",
                    "response": response.dict()
                })
            
            elif message["type"] == "ping":
                # Handle ping
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()


# Example of how to mount these routes in your main FastAPI app:
"""
from fastapi import FastAPI
from mcp.mcp_integration_routes import router as mcp_router

app = FastAPI()

# Mount MCP routes
app.include_router(mcp_router)

# Your other routes...
""" 