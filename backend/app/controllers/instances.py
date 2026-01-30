"""
Instances API Controller

REST endpoints for managing agent instances and container orchestration.
Provides the API layer for the InstanceManager service.
"""

import logging
from typing import Dict, List, Any, Optional
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Query, Depends
from pydantic import BaseModel, Field

from app.services.instance_manager import (
    instance_manager,
    InstanceConfig,
    InstanceInfo,
    ScaleRequest
)
from app.repository.instance_repository import (
    get_instance,
    list_instances,
    delete_instance,
    get_trust_metrics,
    get_instances_with_trust_metrics,
    InstanceStatus
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/instances", tags=["instances"])


class SpawnInstanceRequest(BaseModel):
    """Request to spawn a new agent instance."""
    agent_id: str = Field(..., description="Unique agent identifier")
    agent_role: str = Field(..., description="Role of the agent (e.g., 'reasoning', 'comprehension')")
    device_id: Optional[str] = Field(None, description="Device ID where instance should run")
    resource_profile: Dict[str, Any] = Field(default_factory=dict, description="Resource configuration")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    environment_vars: Dict[str, str] = Field(default_factory=dict, description="Additional environment variables")
    memory_limit: str = Field("512m", description="Memory limit (e.g., '512m', '1g')")
    cpu_limit: float = Field(0.5, description="CPU limit (fraction of one CPU)")
    network: str = Field("core-network", description="Docker network to connect to")


class InstanceResponse(BaseModel):
    """Response containing instance information."""
    container_id: str
    agent_id: str
    agent_role: str
    status: str
    health_status: Optional[str] = None
    uptime_seconds: Optional[int] = None
    memory_usage: Optional[Dict[str, Any]] = None
    cpu_usage: Optional[float] = None
    created_at: str
    last_heartbeat: Optional[str] = None


class ListInstancesResponse(BaseModel):
    """Response for listing instances."""
    instances: List[InstanceResponse]
    total_count: int
    page: int
    page_size: int


class ScaleResponse(BaseModel):
    """Response for scaling operation."""
    role: str
    target_count: int
    initial_count: int
    final_count: int
    actions_taken: List[Dict[str, Any]]
    success: bool


class TrustMetricsResponse(BaseModel):
    """Response containing trust metrics."""
    tasks_completed: int
    tasks_refused: int
    tasks_failed: int
    overrides_received: int
    override_success_rate: float
    avg_task_duration_ms: Optional[float]
    trust_score: Optional[float] = None


class InstanceWithTrustResponse(BaseModel):
    """Response containing instance with trust metrics."""
    instance: InstanceResponse
    trust_metrics: TrustMetricsResponse


def _convert_instance_info(info: InstanceInfo) -> InstanceResponse:
    """Convert InstanceInfo to API response format."""
    return InstanceResponse(
        container_id=info.container_id,
        agent_id=info.agent_id,
        agent_role=info.agent_role,
        status=info.status,
        health_status=info.health_status,
        uptime_seconds=info.uptime_seconds,
        memory_usage=info.memory_usage,
        cpu_usage=info.cpu_usage,
        created_at=info.created_at.isoformat(),
        last_heartbeat=info.last_heartbeat.isoformat() if info.last_heartbeat else None
    )


@router.post("/", status_code=status.HTTP_201_CREATED, response_model=InstanceResponse)
async def spawn_instance(request: SpawnInstanceRequest) -> InstanceResponse:
    """
    Spawn a new agent container.
    
    Creates and starts a new agent container with the specified configuration.
    The container will be connected to the core-network and configured with
    the appropriate environment variables for integration with CORE.
    """
    try:
        # Validate that instance manager is initialized
        if not instance_manager.docker_client:
            await instance_manager.initialize()
        
        # Create instance configuration
        config = InstanceConfig(
            agent_id=request.agent_id,
            agent_role=request.agent_role,
            device_id=request.device_id,
            resource_profile=request.resource_profile,
            capabilities=request.capabilities,
            environment_vars=request.environment_vars,
            memory_limit=request.memory_limit,
            cpu_limit=request.cpu_limit,
            network=request.network
        )
        
        # Spawn the instance
        instance_info = await instance_manager.spawn_instance(config)
        
        logger.info(f"Spawned instance: {instance_info.container_id[:12]} for role {request.agent_role}")
        
        return _convert_instance_info(instance_info)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid configuration: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to spawn instance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to spawn instance: {str(e)}"
        )


@router.get("/", status_code=status.HTTP_200_OK, response_model=ListInstancesResponse)
async def list_instances_endpoint(
    agent_role: Optional[str] = Query(None, description="Filter by agent role"),
    status_filter: Optional[str] = Query(None, description="Filter by status", alias="status"),
    device_id: Optional[str] = Query(None, description="Filter by device ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Page size"),
    include_docker: bool = Query(True, description="Include live Docker container data")
) -> ListInstancesResponse:
    """
    List all agent instances with optional filtering.
    
    Returns instances from both the database and live Docker containers.
    Supports filtering by role, status, and device ID with pagination.
    """
    try:
        if include_docker:
            # Get live data from Docker
            if not instance_manager.docker_client:
                await instance_manager.initialize()
            
            docker_instances = await instance_manager.list_instances()
            
            # Apply filters
            filtered_instances = docker_instances
            
            if agent_role:
                filtered_instances = [i for i in filtered_instances if i.agent_role == agent_role]
            
            if status_filter:
                filtered_instances = [i for i in filtered_instances if i.status == status_filter]
            
            if device_id:
                # Note: Docker instances might not have device_id, but we can filter by container labels
                # For now, we'll skip this filter for Docker instances
                pass
            
            # Apply pagination
            total_count = len(filtered_instances)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated_instances = filtered_instances[start_idx:end_idx]
            
            response_instances = [_convert_instance_info(info) for info in paginated_instances]
        else:
            # Get data from database only
            status_enum = None
            if status_filter:
                try:
                    status_enum = InstanceStatus(status_filter)
                except ValueError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid status: {status_filter}"
                    )
            
            offset = (page - 1) * page_size
            db_instances = await list_instances(
                agent_role=agent_role,
                status=status_enum,
                device_id=device_id,
                limit=page_size,
                offset=offset
            )
            
            # Get total count (simplified - in production would need separate count query)
            total_count = len(db_instances)
            
            response_instances = []
            for instance in db_instances:
                response_instances.append(InstanceResponse(
                    container_id=instance.container_id,
                    agent_id=instance.agent_id,
                    agent_role=instance.agent_role,
                    status=instance.status.value,
                    created_at=instance.created_at.isoformat(),
                    last_heartbeat=instance.last_heartbeat.isoformat() if instance.last_heartbeat else None
                ))
        
        return ListInstancesResponse(
            instances=response_instances,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Failed to list instances: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list instances: {str(e)}"
        )


@router.get("/{instance_id}", status_code=status.HTTP_200_OK, response_model=InstanceResponse)
async def get_instance_details(instance_id: str) -> InstanceResponse:
    """
    Get detailed information about a specific instance.
    
    Can be called with either database UUID or Docker container ID.
    Returns detailed status including resource usage if the container is running.
    """
    try:
        # Try to parse as UUID first (database ID)
        try:
            uuid_id = UUID(instance_id)
            db_instance = await get_instance(uuid_id)
            if db_instance:
                container_id = db_instance.container_id
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Instance {instance_id} not found"
                )
        except ValueError:
            # Not a UUID, assume it's a container ID
            container_id = instance_id
        
        # Get detailed status from Docker
        if not instance_manager.docker_client:
            await instance_manager.initialize()
        
        instance_info = await instance_manager.get_instance_status(container_id)
        
        if not instance_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Instance {instance_id} not found"
            )
        
        return _convert_instance_info(instance_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get instance details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get instance details: {str(e)}"
        )


@router.post("/{instance_id}/stop", status_code=status.HTTP_200_OK)
async def stop_instance(instance_id: str) -> Dict[str, Any]:
    """
    Gracefully stop an instance.
    
    Sends SIGTERM to the container, waits 10 seconds, then sends SIGKILL if needed.
    Updates the instance status in the database.
    """
    try:
        # Resolve to container ID
        container_id = await _resolve_to_container_id(instance_id)
        
        if not instance_manager.docker_client:
            await instance_manager.initialize()
        
        success = await instance_manager.stop_instance(container_id)
        
        if success:
            return {
                "message": f"Instance {instance_id} stopped successfully",
                "container_id": container_id,
                "success": True
            }
        else:
            return {
                "message": f"Failed to stop instance {instance_id}",
                "container_id": container_id,
                "success": False
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop instance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop instance: {str(e)}"
        )


@router.post("/{instance_id}/restart", status_code=status.HTTP_200_OK)
async def restart_instance(instance_id: str) -> Dict[str, Any]:
    """
    Restart an instance.
    
    Restarts the Docker container and waits for it to become healthy.
    Updates the instance status in the database.
    """
    try:
        # Resolve to container ID
        container_id = await _resolve_to_container_id(instance_id)
        
        if not instance_manager.docker_client:
            await instance_manager.initialize()
        
        success = await instance_manager.restart_instance(container_id)
        
        if success:
            return {
                "message": f"Instance {instance_id} restarted successfully",
                "container_id": container_id,
                "success": True
            }
        else:
            return {
                "message": f"Failed to restart instance {instance_id}",
                "container_id": container_id,
                "success": False
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restart instance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restart instance: {str(e)}"
        )


@router.delete("/{instance_id}", status_code=status.HTTP_200_OK)
async def remove_instance(instance_id: str, force: bool = Query(False, description="Force remove running container")) -> Dict[str, Any]:
    """
    Remove an instance.
    
    Stops and removes the Docker container, then deletes the database record.
    Use force=true to remove running containers.
    """
    try:
        # Resolve to container ID and get database record
        container_id = await _resolve_to_container_id(instance_id)
        
        # Get database record
        db_instance = None
        try:
            uuid_id = UUID(instance_id)
            db_instance = await get_instance(uuid_id)
        except ValueError:
            # Not a UUID, look up by container ID
            from app.repository.instance_repository import get_instance_by_container_id
            db_instance = await get_instance_by_container_id(container_id)
        
        if not instance_manager.docker_client:
            await instance_manager.initialize()
        
        # Check if container is running
        instance_info = await instance_manager.get_instance_status(container_id)
        if instance_info and instance_info.status == "running" and not force:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot remove running instance. Stop it first or use force=true"
            )
        
        # Stop and remove container
        try:
            container = instance_manager.docker_client.containers.get(container_id)
            if container.status == "running":
                container.stop(timeout=10)
            container.remove()
            logger.info(f"Removed container {container_id[:12]}")
        except Exception as e:
            logger.warning(f"Failed to remove container {container_id}: {e}")
        
        # Remove from database
        if db_instance:
            await delete_instance(db_instance.id)
            logger.info(f"Removed instance {db_instance.id} from database")
        
        return {
            "message": f"Instance {instance_id} removed successfully",
            "container_id": container_id,
            "success": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove instance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove instance: {str(e)}"
        )


@router.post("/scale", status_code=status.HTTP_200_OK, response_model=ScaleResponse)
async def scale_instances(request: ScaleRequest) -> ScaleResponse:
    """
    Scale instances of a specific role to target count.
    
    Spawns or stops containers to match the target count for the given role.
    Returns detailed information about actions taken.
    """
    try:
        if not instance_manager.docker_client:
            await instance_manager.initialize()
        
        result = await instance_manager.scale_instances(request.role, request.target_count)
        
        return ScaleResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to scale instances: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to scale instances: {str(e)}"
        )


@router.get("/{instance_id}/trust", status_code=status.HTTP_200_OK, response_model=TrustMetricsResponse)
async def get_instance_trust_metrics(instance_id: str) -> TrustMetricsResponse:
    """
    Get trust metrics for a specific instance.
    
    Returns task completion statistics, override history, and trust scores.
    """
    try:
        # Resolve to database ID
        db_id = await _resolve_to_database_id(instance_id)
        
        trust_metrics = await get_trust_metrics(db_id)
        
        if not trust_metrics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Trust metrics not found for instance {instance_id}"
            )
        
        # Calculate trust score
        total_tasks = trust_metrics.tasks_completed + trust_metrics.tasks_failed
        trust_score = (trust_metrics.tasks_completed + 1.0) / (total_tasks + 1.0) if total_tasks >= 0 else 0.5
        
        return TrustMetricsResponse(
            tasks_completed=trust_metrics.tasks_completed,
            tasks_refused=trust_metrics.tasks_refused,
            tasks_failed=trust_metrics.tasks_failed,
            overrides_received=trust_metrics.overrides_received,
            override_success_rate=trust_metrics.override_success_rate,
            avg_task_duration_ms=trust_metrics.avg_task_duration_ms,
            trust_score=trust_score
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get trust metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get trust metrics: {str(e)}"
        )


@router.get("/analytics/trust", status_code=status.HTTP_200_OK)
async def get_trust_analytics(
    agent_role: Optional[str] = Query(None, description="Filter by agent role"),
    min_trust_score: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum trust score")
) -> List[InstanceWithTrustResponse]:
    """
    Get instances with trust analytics.
    
    Returns instances sorted by trust score with their metrics.
    Useful for identifying the most reliable instances.
    """
    try:
        results = await get_instances_with_trust_metrics(
            agent_role=agent_role,
            min_trust_score=min_trust_score
        )
        
        response_data = []
        for item in results:
            instance = item["instance"]
            metrics = item["trust_metrics"]
            
            instance_response = InstanceResponse(
                container_id=instance.container_id,
                agent_id=instance.agent_id,
                agent_role=instance.agent_role,
                status=instance.status.value,
                created_at=instance.created_at.isoformat(),
                last_heartbeat=instance.last_heartbeat.isoformat() if instance.last_heartbeat else None
            )
            
            trust_response = TrustMetricsResponse(
                tasks_completed=metrics["tasks_completed"],
                tasks_refused=metrics["tasks_refused"],
                tasks_failed=metrics["tasks_failed"],
                overrides_received=metrics["overrides_received"],
                override_success_rate=metrics["override_success_rate"],
                avg_task_duration_ms=metrics["avg_task_duration_ms"],
                trust_score=metrics["trust_score"]
            )
            
            response_data.append(InstanceWithTrustResponse(
                instance=instance_response,
                trust_metrics=trust_response
            ))
        
        return response_data
        
    except Exception as e:
        logger.error(f"Failed to get trust analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get trust analytics: {str(e)}"
        )


async def _resolve_to_container_id(instance_id: str) -> str:
    """Resolve instance ID (UUID or container ID) to container ID."""
    try:
        # Try to parse as UUID first
        uuid_id = UUID(instance_id)
        db_instance = await get_instance(uuid_id)
        if db_instance:
            return db_instance.container_id
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Instance {instance_id} not found"
            )
    except ValueError:
        # Not a UUID, assume it's already a container ID
        return instance_id


async def _resolve_to_database_id(instance_id: str) -> UUID:
    """Resolve instance ID (UUID or container ID) to database UUID."""
    try:
        # Try to parse as UUID first
        uuid_id = UUID(instance_id)
        # Verify it exists
        db_instance = await get_instance(uuid_id)
        if db_instance:
            return uuid_id
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Instance {instance_id} not found"
            )
    except ValueError:
        # Not a UUID, assume it's a container ID - look up database record
        from app.repository.instance_repository import get_instance_by_container_id
        db_instance = await get_instance_by_container_id(instance_id)
        if db_instance:
            return db_instance.id
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Instance with container ID {instance_id} not found"
            )


# =============================================================================
# AGENT REGISTRATION ENDPOINTS (REST Fallbacks)
# =============================================================================

class AgentRegistrationRequest(BaseModel):
    """REST request for agent registration (fallback when WebSocket unavailable)."""
    container_id: str = Field(..., description="Docker container ID")
    role: str = Field(..., description="Agent role")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    version: str = Field(..., description="Agent version")


class AgentConfigResponse(BaseModel):
    """Response containing agent configuration."""
    agent_id: str
    model: str
    tools: List[str]
    memory_config: Dict[str, Any]


class AgentHeartbeatRequest(BaseModel):
    """REST request for agent heartbeat (fallback when WebSocket unavailable)."""
    status: str = Field(..., description="Current agent status")
    current_task: Optional[str] = Field(None, description="Current task ID")
    resource_usage: Dict[str, Any] = Field(default_factory=dict, description="Resource usage metrics")


class ConnectedAgentInfo(BaseModel):
    """Information about a connected agent."""
    agent_id: str
    container_id: str
    role: str
    status: str
    capabilities: List[str]
    last_heartbeat: str
    websocket_connected: bool


@router.post("/agents/register", status_code=status.HTTP_200_OK, response_model=AgentConfigResponse)
async def register_agent_rest(request: AgentRegistrationRequest) -> AgentConfigResponse:
    """
    REST fallback for agent registration when WebSocket is unavailable.
    
    Agents should prefer WebSocket registration for real-time communication,
    but can use this endpoint if WebSocket connection fails.
    """
    try:
        from app.services.agent_registry import agent_registry, AgentRegistrationPayload
        
        registration = AgentRegistrationPayload(
            container_id=request.container_id,
            role=request.role,
            capabilities=request.capabilities,
            version=request.version
        )
        
        config = await agent_registry.register_agent(registration)
        
        logger.info(f"Agent registered via REST: {config.agent_id} (container: {request.container_id[:12]})")
        
        return AgentConfigResponse(
            agent_id=config.agent_id,
            model=config.model,
            tools=config.tools,
            memory_config=config.memory_config
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Registration failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to register agent via REST: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )


@router.post("/agents/{agent_id}/heartbeat", status_code=status.HTTP_200_OK)
async def send_heartbeat_rest(agent_id: str, request: AgentHeartbeatRequest) -> Dict[str, Any]:
    """
    REST fallback for agent heartbeat when WebSocket is unavailable.
    
    Agents should prefer WebSocket heartbeats for real-time communication,
    but can use this endpoint if WebSocket connection is lost.
    """
    try:
        from app.services.agent_registry import agent_registry, AgentHeartbeatData
        
        heartbeat_data = AgentHeartbeatData(
            status=request.status,
            current_task=request.current_task,
            resource_usage=request.resource_usage
        )
        
        response = await agent_registry.handle_heartbeat(agent_id, heartbeat_data)
        
        return {
            "message": "Heartbeat processed",
            "response": response,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Heartbeat failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to process heartbeat via REST: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Heartbeat failed: {str(e)}"
        )


@router.get("/agents/connected", status_code=status.HTTP_200_OK)
async def list_connected_agents() -> Dict[str, Any]:
    """
    List currently connected agents with WebSocket status.
    
    Shows both agents connected via WebSocket and those registered via REST,
    along with their current status and capabilities.
    """
    try:
        from app.services.agent_registry import agent_registry
        from app.controllers.agent_ws import agent_ws_manager
        
        # Get active agents from registry
        active_agents = []
        websocket_connections = agent_ws_manager.get_connected_agents()
        
        for agent_id in agent_registry.list_active_agents():
            agent_info = agent_registry.get_agent_info(agent_id)
            if agent_info:
                active_agents.append(ConnectedAgentInfo(
                    agent_id=agent_id,
                    container_id=agent_info["container_id"],
                    role=agent_info["role"],
                    status=agent_info["current_status"],
                    capabilities=agent_info["capabilities"],
                    last_heartbeat=agent_info["last_heartbeat"].isoformat(),
                    websocket_connected=agent_id in websocket_connections
                ))
        
        # Get connection stats
        ws_stats = agent_ws_manager.get_connection_stats()
        
        return {
            "agents": active_agents,
            "total_agents": len(active_agents),
            "websocket_stats": ws_stats,
            "healthy_agents": len(agent_registry.get_healthy_agents()),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to list connected agents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list connected agents: {str(e)}"
        )