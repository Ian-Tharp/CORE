"""
Agent Registry Service

Manages agent registration, heartbeats, and lifecycle for containerized agents.
This is part of the Agent Registry layer in the CORE architecture.

The Agent Registry handles:
1. Agent registration when containers start up
2. Heartbeat processing to track agent health
3. Task assignment and refusal management
4. Stale agent detection and cleanup
5. Agent deregistration on shutdown

Architecture Context: 
Comprehension → Orchestration → Reasoning → Evaluation
Agent Library → Agent Factory → Agent Registry → Communication Commons
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from uuid import uuid4, UUID
from pydantic import BaseModel, Field

from app.repository.instance_repository import (
    get_instance_by_container_id,
    update_instance_status,
    update_heartbeat,
    list_instances_by_status,
    update_instance,
    InstanceStatus,
    increment_task_completed,
    increment_task_failed,
    increment_task_refused
)
from app.services.instance_manager import instance_manager

logger = logging.getLogger(__name__)


class AgentRegistrationPayload(BaseModel):
    """Payload for agent registration."""
    container_id: str = Field(..., description="Docker container ID")
    role: str = Field(..., description="Agent role (e.g., 'researcher', 'analyst')")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    version: str = Field(..., description="Agent version")


class AgentHeartbeatData(BaseModel):
    """Heartbeat data from agent."""
    status: str = Field(..., description="Current agent status")
    current_task: Optional[str] = Field(None, description="Current task ID")
    resource_usage: Dict[str, Any] = Field(default_factory=dict, description="Resource usage metrics")


class AgentConfig(BaseModel):
    """Configuration returned to agent on registration."""
    agent_id: str = Field(..., description="Assigned agent ID")
    model: str = Field(..., description="LLM model to use")
    tools: List[str] = Field(default_factory=list, description="Available tools")
    memory_config: Dict[str, Any] = Field(default_factory=dict, description="Memory configuration")


class TaskAssignment(BaseModel):
    """Task assignment for agent."""
    task_id: str = Field(..., description="Unique task ID")
    task_type: str = Field(..., description="Type of task")
    payload: Dict[str, Any] = Field(..., description="Task payload")
    priority: int = Field(default=5, description="Task priority (1-10)")


class TaskCompletion(BaseModel):
    """Task completion data from agent."""
    task_id: str = Field(..., description="Task ID")
    result: Dict[str, Any] = Field(..., description="Task result")
    duration_ms: int = Field(..., description="Task duration in milliseconds")


class TaskRefusal(BaseModel):
    """Task refusal data from agent."""
    task_id: str = Field(..., description="Task ID")
    reason: str = Field(..., description="Reason for refusal")
    suggested_agent: Optional[str] = Field(None, description="Suggested alternative agent")


class AgentDeregistration(BaseModel):
    """Agent deregistration data."""
    reason: str = Field(..., description="Reason for deregistration")
    final_state: Dict[str, Any] = Field(default_factory=dict, description="Final agent state")


class AgentRegistry:
    """
    Central registry for managing containerized agent instances.
    
    Handles the complete agent lifecycle from registration to deregistration,
    including heartbeat monitoring and task management.
    """
    
    def __init__(self):
        self.active_agents: Dict[str, Dict[str, Any]] = {}  # agent_id -> agent_info
        self.agent_websockets: Dict[str, Any] = {}  # agent_id -> websocket connection
        self.pending_tasks: Dict[str, List[TaskAssignment]] = {}  # agent_id -> tasks
        self._shutdown = False
    
    async def register_agent(self, registration_payload: AgentRegistrationPayload) -> AgentConfig:
        """
        Register an agent instance.
        
        Validates that the container_id exists in instance_repository,
        updates status to 'ready', stores capabilities and version info,
        and returns configuration for the agent.
        
        Args:
            registration_payload: Registration data from agent
            
        Returns:
            AgentConfig with assigned configuration
            
        Raises:
            ValueError: If container_id is not found or invalid
        """
        container_id = registration_payload.container_id
        
        # Verify container exists in database
        instance = await get_instance_by_container_id(container_id)
        if not instance:
            raise ValueError(f"Container {container_id} not found in database")
        
        # Generate agent_id if not already set
        agent_id = instance.agent_id or f"{registration_payload.role}-{str(uuid4())[:8]}"
        
        # Update instance with registration data
        await update_instance(instance.id, {
            "agent_id": agent_id,
            "status": InstanceStatus.READY.value,
            "capabilities": registration_payload.capabilities,
            "last_heartbeat": datetime.utcnow()
        })
        
        # Store agent info in memory
        self.active_agents[agent_id] = {
            "container_id": container_id,
            "instance_id": instance.id,
            "role": registration_payload.role,
            "capabilities": registration_payload.capabilities,
            "version": registration_payload.version,
            "registered_at": datetime.utcnow(),
            "last_heartbeat": datetime.utcnow(),
            "current_status": "ready",
            "current_task": None
        }
        
        logger.info(f"Registered agent: {agent_id} (container: {container_id[:12]})")
        
        # Return configuration
        return AgentConfig(
            agent_id=agent_id,
            model="ollama/llama3.2",  # Default model - could be configurable
            tools=self._get_tools_for_role(registration_payload.role),
            memory_config={
                "max_context_length": 4000,
                "conversation_memory": True
            }
        )
    
    def _get_tools_for_role(self, role: str) -> List[str]:
        """Get available tools for a given agent role."""
        # This could be configurable or pulled from database
        role_tools = {
            "researcher": ["web_search", "web_fetch", "file_read", "file_write"],
            "analyst": ["web_search", "file_read", "image_analysis"],
            "writer": ["file_read", "file_write", "web_search"],
            "monitor": ["web_fetch", "file_read"],
            "coordinator": ["message", "file_read", "file_write"]
        }
        return role_tools.get(role, ["web_search", "file_read"])
    
    async def handle_heartbeat(self, agent_id: str, heartbeat_data: AgentHeartbeatData) -> Dict[str, Any]:
        """
        Process heartbeat from agent.
        
        Updates last_heartbeat timestamp, current status and resource usage.
        Checks for assigned tasks to push to agent.
        
        Args:
            agent_id: Agent identifier
            heartbeat_data: Heartbeat payload from agent
            
        Returns:
            Dict with any pending tasks or configuration updates
        """
        if agent_id not in self.active_agents:
            raise ValueError(f"Agent {agent_id} not registered")
        
        agent_info = self.active_agents[agent_id]
        current_time = datetime.utcnow()
        
        # Update agent info
        agent_info["last_heartbeat"] = current_time
        agent_info["current_status"] = heartbeat_data.status
        agent_info["current_task"] = heartbeat_data.current_task
        agent_info["resource_usage"] = heartbeat_data.resource_usage
        
        # Update database
        await update_heartbeat(agent_info["container_id"], current_time)
        
        # Check for pending tasks
        response = {"type": "heartbeat_ack"}
        
        if agent_id in self.pending_tasks and self.pending_tasks[agent_id]:
            # Send next pending task
            task = self.pending_tasks[agent_id].pop(0)
            response = {
                "type": "task_assigned",
                "task_id": task.task_id,
                "task_type": task.task_type,
                "payload": task.payload,
                "priority": task.priority
            }
            
            logger.info(f"Assigned task {task.task_id} to agent {agent_id}")
        
        logger.debug(f"Processed heartbeat from {agent_id} (status: {heartbeat_data.status})")
        
        return response
    
    async def handle_task_completion(self, agent_id: str, completion: TaskCompletion) -> None:
        """
        Handle task completion from agent.
        
        Updates trust metrics and logs task completion.
        
        Args:
            agent_id: Agent identifier
            completion: Task completion data
        """
        if agent_id not in self.active_agents:
            raise ValueError(f"Agent {agent_id} not registered")
        
        agent_info = self.active_agents[agent_id]
        
        # Update trust metrics
        await increment_task_completed(
            agent_info["instance_id"],
            completion.duration_ms
        )
        
        # Clear current task
        agent_info["current_task"] = None
        
        logger.info(f"Agent {agent_id} completed task {completion.task_id} in {completion.duration_ms}ms")
    
    async def handle_task_refusal(self, agent_id: str, refusal: TaskRefusal) -> None:
        """
        Handle task refusal from agent.
        
        Records refusal in trust metrics and potentially reassigns task.
        
        Args:
            agent_id: Agent identifier
            refusal: Task refusal data
        """
        if agent_id not in self.active_agents:
            raise ValueError(f"Agent {agent_id} not registered")
        
        agent_info = self.active_agents[agent_id]
        
        # Update trust metrics
        await increment_task_refused(agent_info["instance_id"])
        
        logger.info(f"Agent {agent_id} refused task {refusal.task_id}: {refusal.reason}")
        
        # TODO: Implement task reassignment logic
        # For now, just log the refusal
        if refusal.suggested_agent:
            logger.info(f"Agent {agent_id} suggested reassigning to {refusal.suggested_agent}")
    
    async def deregister_agent(self, agent_id: str, deregistration: AgentDeregistration) -> None:
        """
        Clean deregistration of agent.
        
        Updates status to 'stopped' and saves final state.
        
        Args:
            agent_id: Agent identifier
            deregistration: Deregistration data
        """
        if agent_id not in self.active_agents:
            logger.warning(f"Attempted to deregister unknown agent: {agent_id}")
            return
        
        agent_info = self.active_agents[agent_id]
        
        # Update database status
        await update_instance_status(
            agent_info["container_id"],
            InstanceStatus.STOPPED,
            datetime.utcnow()
        )
        
        # Remove from active agents
        del self.active_agents[agent_id]
        
        # Clean up pending tasks
        if agent_id in self.pending_tasks:
            del self.pending_tasks[agent_id]
        
        logger.info(f"Deregistered agent {agent_id}: {deregistration.reason}")
    
    async def assign_task(self, agent_id: str, task: TaskAssignment) -> bool:
        """
        Assign a task to an agent.
        
        Adds task to pending queue to be sent on next heartbeat.
        
        Args:
            agent_id: Target agent identifier
            task: Task assignment data
            
        Returns:
            True if task was queued, False if agent not found
        """
        if agent_id not in self.active_agents:
            return False
        
        if agent_id not in self.pending_tasks:
            self.pending_tasks[agent_id] = []
        
        self.pending_tasks[agent_id].append(task)
        
        logger.info(f"Queued task {task.task_id} for agent {agent_id}")
        return True
    
    async def check_stale_agents(self) -> List[str]:
        """
        Background task to find and mark stale agents.
        
        Finds agents with heartbeat older than 90s and marks as 'unhealthy'.
        After 5 minutes no heartbeat, marks as 'lost'.
        Optionally triggers restart via InstanceManager.
        
        Returns:
            List of agent IDs that were marked as stale
        """
        if self._shutdown:
            return []
        
        current_time = datetime.utcnow()
        stale_agents = []
        lost_agents = []
        
        for agent_id, agent_info in list(self.active_agents.items()):
            last_heartbeat = agent_info["last_heartbeat"]
            time_since_heartbeat = current_time - last_heartbeat
            
            if time_since_heartbeat > timedelta(minutes=5):
                # Lost agent - mark as lost and remove
                lost_agents.append(agent_id)
                await update_instance_status(
                    agent_info["container_id"],
                    InstanceStatus.FAILED
                )
                del self.active_agents[agent_id]
                
                logger.warning(f"Agent {agent_id} marked as lost (no heartbeat for 5+ minutes)")
                
                # Trigger restart if InstanceManager is available
                try:
                    await instance_manager.restart_instance(agent_info["container_id"])
                except Exception as e:
                    logger.error(f"Failed to restart lost agent {agent_id}: {e}")
            
            elif time_since_heartbeat > timedelta(seconds=90):
                # Stale agent - mark as unhealthy but keep tracking
                stale_agents.append(agent_id)
                agent_info["current_status"] = "unhealthy"
                
                # Update database status
                try:
                    instance = await get_instance_by_container_id(agent_info["container_id"])
                    if instance and instance.status != InstanceStatus.FAILED:
                        await update_instance_status(
                            agent_info["container_id"],
                            InstanceStatus.FAILED  # Mark as failed for stale heartbeat
                        )
                except Exception as e:
                    logger.error(f"Failed to update stale agent status: {e}")
                
                logger.warning(f"Agent {agent_id} marked as stale (no heartbeat for 90+ seconds)")
        
        return stale_agents + lost_agents
    
    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered agent."""
        return self.active_agents.get(agent_id)
    
    def list_active_agents(self) -> List[str]:
        """Get list of active agent IDs."""
        return list(self.active_agents.keys())
    
    def get_agents_by_role(self, role: str) -> List[str]:
        """Get agents by role."""
        return [
            agent_id for agent_id, info in self.active_agents.items()
            if info.get("role") == role
        ]
    
    def get_healthy_agents(self) -> List[str]:
        """Get agents that are healthy (recent heartbeat, ready status)."""
        current_time = datetime.utcnow()
        healthy = []
        
        for agent_id, info in self.active_agents.items():
            time_since_heartbeat = current_time - info["last_heartbeat"]
            if (time_since_heartbeat < timedelta(seconds=60) and 
                info["current_status"] in ["ready", "busy"]):
                healthy.append(agent_id)
        
        return healthy
    
    async def shutdown(self) -> None:
        """Shutdown the agent registry."""
        self._shutdown = True
        logger.info("Agent registry shutting down")


# Global agent registry instance
agent_registry = AgentRegistry()


async def start_heartbeat_monitor():
    """Start the background heartbeat monitoring task."""
    while not agent_registry._shutdown:
        try:
            await agent_registry.check_stale_agents()
        except Exception as e:
            logger.error(f"Error in heartbeat monitor: {e}")
        
        # Check every 30 seconds
        await asyncio.sleep(30)


# Background task reference for lifecycle management
heartbeat_monitor_task = None


async def initialize_agent_registry():
    """Initialize the agent registry and start background tasks."""
    global heartbeat_monitor_task
    
    if heartbeat_monitor_task is None:
        heartbeat_monitor_task = asyncio.create_task(start_heartbeat_monitor())
        logger.info("Agent registry initialized with heartbeat monitor")


async def shutdown_agent_registry():
    """Shutdown the agent registry and background tasks."""
    global heartbeat_monitor_task
    
    await agent_registry.shutdown()
    
    if heartbeat_monitor_task:
        heartbeat_monitor_task.cancel()
        try:
            await heartbeat_monitor_task
        except asyncio.CancelledError:
            pass
        heartbeat_monitor_task = None
    
    logger.info("Agent registry shutdown complete")