"""
Agent WebSocket Controller

WebSocket endpoint for real-time agent communication.
Separate from the Communication Commons WebSocket - this is specifically 
for containerized agents to register, send heartbeats, and receive tasks.

Endpoint: /ws/agent/{agent_id}

Agent → CORE Messages:
- register: Initial registration with capabilities and version
- heartbeat: Periodic status and resource usage updates  
- task_complete: Task completion with results and duration
- task_refused: Task refusal with reason and suggestions
- deregister: Clean shutdown with final state

CORE → Agent Messages:
- registered: Registration confirmation with config
- task_assigned: New task assignment with payload and priority
- config_update: Configuration changes
- shutdown: Graceful shutdown request with grace period
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime

from app.services.agent_registry import (
    agent_registry,
    AgentRegistrationPayload,
    AgentHeartbeatData,
    TaskCompletion,
    TaskRefusal,
    AgentDeregistration
)

logger = logging.getLogger(__name__)


class AgentWebSocketManager:
    """
    Manages WebSocket connections for agent communication.
    
    Handles agent lifecycle events through WebSocket messages.
    """
    
    def __init__(self):
        # agent_id -> WebSocket connection
        self.agent_connections: Dict[str, WebSocket] = {}
        # Track connection metadata
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, agent_id: str, websocket: WebSocket):
        """Accept agent WebSocket connection."""
        await websocket.accept()
        self.agent_connections[agent_id] = websocket
        self.connection_metadata[agent_id] = {
            "connected_at": datetime.utcnow(),
            "registered": False,
            "last_activity": datetime.utcnow()
        }
        
        logger.info(f"Agent WebSocket connected: {agent_id}")
        
        # Send connection acknowledgment
        await self.send_message(agent_id, {
            "type": "connected",
            "message": "Connected to CORE Agent Registry",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def disconnect(self, agent_id: str):
        """Remove agent connection."""
        if agent_id in self.agent_connections:
            del self.agent_connections[agent_id]
        
        if agent_id in self.connection_metadata:
            del self.connection_metadata[agent_id]
        
        logger.info(f"Agent WebSocket disconnected: {agent_id}")
    
    async def send_message(self, agent_id: str, message: Dict[str, Any]) -> bool:
        """
        Send message to specific agent.
        
        Args:
            agent_id: Target agent identifier
            message: Message payload
            
        Returns:
            True if sent successfully, False otherwise
        """
        websocket = self.agent_connections.get(agent_id)
        if websocket:
            try:
                await websocket.send_json(message)
                
                # Update last activity
                if agent_id in self.connection_metadata:
                    self.connection_metadata[agent_id]["last_activity"] = datetime.utcnow()
                
                return True
            except Exception as e:
                logger.error(f"Error sending message to agent {agent_id}: {e}")
                self.disconnect(agent_id)
                return False
        
        logger.warning(f"Attempted to send message to disconnected agent: {agent_id}")
        return False
    
    async def broadcast_to_role(self, role: str, message: Dict[str, Any]):
        """Broadcast message to all agents of a specific role."""
        # Get agents by role from registry
        agent_ids = agent_registry.get_agents_by_role(role)
        
        for agent_id in agent_ids:
            if agent_id in self.agent_connections:
                await self.send_message(agent_id, message)
    
    async def handle_message(self, agent_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle incoming message from agent.
        
        Args:
            agent_id: Source agent identifier
            data: Message payload
            
        Returns:
            Response message if any, None otherwise
        """
        message_type = data.get("type")
        
        if not message_type:
            logger.warning(f"Received message without type from {agent_id}")
            return {"type": "error", "error": "Message type required"}
        
        try:
            # Update last activity
            if agent_id in self.connection_metadata:
                self.connection_metadata[agent_id]["last_activity"] = datetime.utcnow()
            
            if message_type == "register":
                return await self._handle_registration(agent_id, data)
            
            elif message_type == "heartbeat":
                return await self._handle_heartbeat(agent_id, data)
            
            elif message_type == "task_complete":
                return await self._handle_task_completion(agent_id, data)
            
            elif message_type == "task_refused":
                return await self._handle_task_refusal(agent_id, data)
            
            elif message_type == "deregister":
                return await self._handle_deregistration(agent_id, data)
            
            else:
                logger.warning(f"Unknown message type '{message_type}' from {agent_id}")
                return {"type": "error", "error": f"Unknown message type: {message_type}"}
        
        except Exception as e:
            logger.error(f"Error handling message from {agent_id}: {e}")
            return {"type": "error", "error": f"Internal error: {str(e)}"}
    
    async def _handle_registration(self, agent_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent registration."""
        try:
            registration = AgentRegistrationPayload(**data)
            
            # Register with agent registry
            config = await agent_registry.register_agent(registration)
            
            # Update connection metadata
            if agent_id in self.connection_metadata:
                self.connection_metadata[agent_id]["registered"] = True
                self.connection_metadata[agent_id]["container_id"] = registration.container_id
                self.connection_metadata[agent_id]["role"] = registration.role
            
            logger.info(f"Agent {agent_id} registered successfully")
            
            return {
                "type": "registered",
                "agent_id": config.agent_id,
                "config": {
                    "model": config.model,
                    "tools": config.tools,
                    "memory_config": config.memory_config
                }
            }
        
        except ValueError as e:
            logger.error(f"Registration failed for {agent_id}: {e}")
            return {"type": "error", "error": f"Registration failed: {str(e)}"}
        
        except Exception as e:
            logger.error(f"Unexpected error during registration for {agent_id}: {e}")
            return {"type": "error", "error": "Registration failed"}
    
    async def _handle_heartbeat(self, agent_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent heartbeat."""
        try:
            heartbeat_data = AgentHeartbeatData(
                status=data.get("status", "unknown"),
                current_task=data.get("current_task"),
                resource_usage=data.get("resource_usage", {})
            )
            
            # Process heartbeat with agent registry
            response = await agent_registry.handle_heartbeat(agent_id, heartbeat_data)
            
            return response
        
        except ValueError as e:
            logger.error(f"Heartbeat failed for {agent_id}: {e}")
            return {"type": "error", "error": f"Heartbeat failed: {str(e)}"}
        
        except Exception as e:
            logger.error(f"Unexpected error during heartbeat for {agent_id}: {e}")
            return {"type": "heartbeat_ack"}  # Send basic ack even if processing failed
    
    async def _handle_task_completion(self, agent_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task completion."""
        try:
            completion = TaskCompletion(
                task_id=data["task_id"],
                result=data.get("result", {}),
                duration_ms=data["duration_ms"]
            )
            
            await agent_registry.handle_task_completion(agent_id, completion)
            
            return {"type": "task_ack", "task_id": completion.task_id}
        
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid task completion from {agent_id}: {e}")
            return {"type": "error", "error": f"Invalid task completion: {str(e)}"}
        
        except Exception as e:
            logger.error(f"Error processing task completion from {agent_id}: {e}")
            return {"type": "error", "error": "Failed to process task completion"}
    
    async def _handle_task_refusal(self, agent_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task refusal."""
        try:
            refusal = TaskRefusal(
                task_id=data["task_id"],
                reason=data["reason"],
                suggested_agent=data.get("suggested_agent")
            )
            
            await agent_registry.handle_task_refusal(agent_id, refusal)
            
            return {"type": "task_ack", "task_id": refusal.task_id}
        
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid task refusal from {agent_id}: {e}")
            return {"type": "error", "error": f"Invalid task refusal: {str(e)}"}
        
        except Exception as e:
            logger.error(f"Error processing task refusal from {agent_id}: {e}")
            return {"type": "error", "error": "Failed to process task refusal"}
    
    async def _handle_deregistration(self, agent_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent deregistration."""
        try:
            deregistration = AgentDeregistration(
                reason=data.get("reason", "unknown"),
                final_state=data.get("final_state", {})
            )
            
            await agent_registry.deregister_agent(agent_id, deregistration)
            
            # Mark connection as deregistered
            if agent_id in self.connection_metadata:
                self.connection_metadata[agent_id]["registered"] = False
            
            return {"type": "deregister_ack", "message": "Deregistration acknowledged"}
        
        except Exception as e:
            logger.error(f"Error processing deregistration from {agent_id}: {e}")
            return {"type": "error", "error": "Failed to process deregistration"}
    
    def get_connected_agents(self) -> List[str]:
        """Get list of connected agent IDs."""
        return list(self.agent_connections.keys())
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        registered_count = sum(
            1 for metadata in self.connection_metadata.values()
            if metadata.get("registered", False)
        )
        
        return {
            "connected_agents": len(self.agent_connections),
            "registered_agents": registered_count,
            "connection_metadata": len(self.connection_metadata)
        }


# Global agent WebSocket manager
agent_ws_manager = AgentWebSocketManager()


async def agent_websocket_endpoint(websocket: WebSocket, agent_id: str):
    """
    WebSocket endpoint for agent communication.
    
    Handles the complete agent lifecycle through WebSocket messages.
    
    Args:
        websocket: WebSocket connection
        agent_id: Agent identifier (typically container ID or assigned ID)
    """
    await agent_ws_manager.connect(agent_id, websocket)
    
    try:
        while True:
            # Receive message from agent
            try:
                data = await websocket.receive_json()
            except Exception as e:
                logger.error(f"Error receiving message from {agent_id}: {e}")
                break
            
            # Handle message and get response
            response = await agent_ws_manager.handle_message(agent_id, data)
            
            # Send response if any
            if response:
                await agent_ws_manager.send_message(agent_id, response)
    
    except WebSocketDisconnect:
        logger.info(f"Agent WebSocket disconnected: {agent_id}")
    
    except Exception as e:
        logger.error(f"Agent WebSocket error for {agent_id}: {e}")
    
    finally:
        # Clean up connection
        agent_ws_manager.disconnect(agent_id)
        
        # Attempt to deregister agent if it was registered
        try:
            if agent_id in agent_registry.active_agents:
                deregistration = AgentDeregistration(
                    reason="websocket_disconnect",
                    final_state={}
                )
                await agent_registry.deregister_agent(agent_id, deregistration)
        except Exception as e:
            logger.error(f"Error deregistering agent {agent_id} on disconnect: {e}")


# Helper functions for task management
async def send_task_to_agent(agent_id: str, task_assignment: dict) -> bool:
    """
    Send a task assignment directly to an agent via WebSocket.
    
    Args:
        agent_id: Target agent identifier
        task_assignment: Task assignment data
        
    Returns:
        True if sent successfully, False otherwise
    """
    message = {
        "type": "task_assigned",
        **task_assignment
    }
    
    return await agent_ws_manager.send_message(agent_id, message)


async def send_config_update(agent_id: str, changes: dict) -> bool:
    """
    Send configuration update to an agent.
    
    Args:
        agent_id: Target agent identifier
        changes: Configuration changes
        
    Returns:
        True if sent successfully, False otherwise
    """
    message = {
        "type": "config_update", 
        "changes": changes,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return await agent_ws_manager.send_message(agent_id, message)


async def send_shutdown_request(agent_id: str, reason: str, grace_period_seconds: int = 30) -> bool:
    """
    Send graceful shutdown request to an agent.
    
    Args:
        agent_id: Target agent identifier
        reason: Reason for shutdown
        grace_period_seconds: Time to allow for graceful shutdown
        
    Returns:
        True if sent successfully, False otherwise
    """
    message = {
        "type": "shutdown",
        "reason": reason,
        "grace_period_seconds": grace_period_seconds,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return await agent_ws_manager.send_message(agent_id, message)