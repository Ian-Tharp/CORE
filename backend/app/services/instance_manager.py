"""
Instance Manager Service

Manages the Docker container lifecycle for CORE agent instances.
This sits below the Agent Registry, providing the infrastructure layer that agents run on.

The InstanceManager orchestrates containers without replacing the existing Agent Factory/Registry architecture.
It provides the foundation for containerized agent execution within the CORE loop.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from uuid import uuid4, UUID

import docker
from docker import DockerClient
from docker.errors import DockerException, NotFound, APIError
from pydantic import BaseModel

from app.repository.instance_repository import (
    create_instance,
    update_instance_status,
    get_instance_by_container_id,
    list_instances_by_status,
    delete_instance,
    AgentInstance,
    InstanceStatus
)

logger = logging.getLogger(__name__)


class InstanceConfig(BaseModel):
    """Configuration for spawning a new agent instance."""
    agent_id: str
    agent_role: str
    device_id: Optional[str] = None
    resource_profile: Dict[str, Any] = {}
    capabilities: List[str] = []
    environment_vars: Dict[str, str] = {}
    memory_limit: str = "512m"
    cpu_limit: float = 0.5
    network: str = "core-network"


class InstanceInfo(BaseModel):
    """Information about a running instance."""
    container_id: str
    agent_id: str
    agent_role: str
    status: str
    health_status: Optional[str] = None
    uptime_seconds: Optional[int] = None
    memory_usage: Optional[Dict[str, Any]] = None
    cpu_usage: Optional[float] = None
    created_at: datetime
    last_heartbeat: Optional[datetime] = None


class ScaleRequest(BaseModel):
    """Request to scale instances of a specific role."""
    role: str
    target_count: int


class InstanceManager:
    """
    Manages Docker containers for CORE agent instances.
    
    Responsibilities:
    - Spawn and manage agent containers using docker/agent/Dockerfile
    - Provide infrastructure layer for Agent Registry
    - Handle container lifecycle (start, stop, restart, health checks)
    - Scale instances by role
    - Track instance metrics and trust scores
    """
    
    def __init__(self):
        """Initialize the InstanceManager."""
        self.docker_client: Optional[DockerClient] = None
        self._shutdown = False
        
    async def initialize(self) -> None:
        """Initialize Docker client and ensure network exists."""
        try:
            self.docker_client = docker.from_env()
            await self._ensure_core_network()
            logger.info("InstanceManager initialized successfully")
        except DockerException as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise
            
    async def shutdown(self) -> None:
        """Shutdown the InstanceManager."""
        self._shutdown = True
        if self.docker_client:
            self.docker_client.close()
        logger.info("InstanceManager shutdown complete")
        
    async def _ensure_core_network(self) -> None:
        """Ensure the core-network Docker network exists."""
        try:
            self.docker_client.networks.get("core-network")
            logger.debug("core-network already exists")
        except NotFound:
            logger.info("Creating core-network...")
            self.docker_client.networks.create(
                "core-network",
                driver="bridge",
                labels={"project": "core", "component": "agent-network"}
            )
            logger.info("Created core-network")
    
    async def spawn_instance(self, config: InstanceConfig) -> InstanceInfo:
        """
        Create and start a new agent container.
        
        Args:
            config: Instance configuration
            
        Returns:
            InstanceInfo with container details
            
        Raises:
            DockerException: If container creation fails
        """
        if not self.docker_client:
            raise RuntimeError("InstanceManager not initialized")
            
        container_name = f"core-agent-{config.agent_role}-{str(uuid4())[:8]}"
        
        # Build environment variables
        env_vars = {
            "AGENT_ID": config.agent_id,
            "AGENT_ROLE": config.agent_role,
            "CORE_ENDPOINT": "http://core-backend:8001",
            "DEVICE_ID": config.device_id or "unknown",
            **config.environment_vars
        }
        
        # Container configuration
        container_config = {
            "image": "core-agent:latest",  # Build from docker/agent/Dockerfile
            "name": container_name,
            "environment": env_vars,
            "labels": {
                "core.agent": "true",
                "core.role": config.agent_role,
                "core.agent_id": config.agent_id,
                "core.version": "1.0"
            },
            "networks": [config.network],
            "mem_limit": config.memory_limit,
            "cpu_quota": int(config.cpu_limit * 100000),  # Docker CPU quota in microseconds
            "cpu_period": 100000,
            "restart_policy": {"Name": "unless-stopped"},
            "detach": True,
            "remove": False
        }
        
        try:
            # Create and start container
            logger.info(f"Creating container: {container_name}")
            container = self.docker_client.containers.run(**container_config)
            
            # Store instance in database
            instance = AgentInstance(
                id=uuid4(),
                container_id=container.id,
                agent_id=config.agent_id,
                agent_role=config.agent_role,
                status=InstanceStatus.STARTING,
                device_id=config.device_id,
                resource_profile=config.resource_profile,
                capabilities=config.capabilities,
                created_at=datetime.utcnow()
            )
            
            await create_instance(instance)
            
            # Wait for container to be ready
            await self._wait_for_healthy(container.id)
            
            # Update status to ready
            await update_instance_status(container.id, InstanceStatus.READY)
            
            logger.info(f"Successfully spawned instance: {container_name} ({container.id[:12]})")
            
            return InstanceInfo(
                container_id=container.id,
                agent_id=config.agent_id,
                agent_role=config.agent_role,
                status="ready",
                created_at=instance.created_at
            )
            
        except DockerException as e:
            logger.error(f"Failed to spawn instance {container_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error spawning instance: {e}")
            # Clean up container if it was created
            try:
                container = self.docker_client.containers.get(container_name)
                container.remove(force=True)
            except NotFound:
                pass
            raise
    
    async def _wait_for_healthy(self, container_id: str, timeout: int = 30) -> None:
        """Wait for container to become healthy."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                container = self.docker_client.containers.get(container_id)
                if container.status == "running":
                    # Additional health checks could go here
                    return
                elif container.status in ["exited", "dead"]:
                    logs = container.logs().decode('utf-8')
                    raise RuntimeError(f"Container failed to start. Logs: {logs}")
                    
                await asyncio.sleep(1)
            except NotFound:
                raise RuntimeError(f"Container {container_id} disappeared during startup")
        
        raise TimeoutError(f"Container {container_id} did not become healthy within {timeout}s")
    
    async def stop_instance(self, container_id: str) -> bool:
        """
        Gracefully stop a container.
        
        Sends SIGTERM, waits 10s, then SIGKILL if needed.
        
        Args:
            container_id: Container ID to stop
            
        Returns:
            True if successful, False otherwise
        """
        if not self.docker_client:
            raise RuntimeError("InstanceManager not initialized")
            
        try:
            container = self.docker_client.containers.get(container_id)
            
            # Update database status
            await update_instance_status(container_id, InstanceStatus.STOPPING)
            
            # Graceful stop (SIGTERM)
            logger.info(f"Gracefully stopping container {container_id[:12]}...")
            container.stop(timeout=10)
            
            # Update database status
            await update_instance_status(container_id, InstanceStatus.STOPPED)
            
            logger.info(f"Successfully stopped container {container_id[:12]}")
            return True
            
        except NotFound:
            logger.warning(f"Container {container_id} not found")
            # Update database anyway - container might have crashed
            await update_instance_status(container_id, InstanceStatus.STOPPED)
            return False
        except Exception as e:
            logger.error(f"Failed to stop container {container_id}: {e}")
            return False
    
    async def restart_instance(self, container_id: str) -> bool:
        """
        Restart a container.
        
        Args:
            container_id: Container ID to restart
            
        Returns:
            True if successful, False otherwise
        """
        if not self.docker_client:
            raise RuntimeError("InstanceManager not initialized")
            
        try:
            container = self.docker_client.containers.get(container_id)
            
            logger.info(f"Restarting container {container_id[:12]}...")
            
            # Update status
            await update_instance_status(container_id, InstanceStatus.RESTARTING)
            
            # Restart container
            container.restart(timeout=10)
            
            # Wait for healthy
            await self._wait_for_healthy(container_id)
            
            # Update status
            await update_instance_status(container_id, InstanceStatus.READY)
            
            logger.info(f"Successfully restarted container {container_id[:12]}")
            return True
            
        except NotFound:
            logger.error(f"Container {container_id} not found")
            return False
        except Exception as e:
            logger.error(f"Failed to restart container {container_id}: {e}")
            await update_instance_status(container_id, InstanceStatus.FAILED)
            return False
    
    async def list_instances(self) -> List[InstanceInfo]:
        """
        List all agent containers with status.
        
        Filters by label core.agent=true
        
        Returns:
            List of InstanceInfo objects
        """
        if not self.docker_client:
            raise RuntimeError("InstanceManager not initialized")
            
        instances = []
        
        try:
            # Get containers with core.agent=true label
            containers = self.docker_client.containers.list(
                all=True,  # Include stopped containers
                filters={"label": "core.agent=true"}
            )
            
            for container in containers:
                try:
                    # Get database record
                    instance_record = await get_instance_by_container_id(container.id)
                    
                    # Get container stats if running
                    uptime = None
                    memory_usage = None
                    cpu_usage = None
                    
                    if container.status == "running":
                        created_time = container.attrs['Created']
                        created_dt = datetime.fromisoformat(created_time.rstrip('Z'))
                        uptime = int((datetime.utcnow() - created_dt).total_seconds())
                        
                        # Get memory/CPU stats
                        try:
                            stats = container.stats(stream=False)
                            memory_usage = stats.get('memory_stats', {})
                            cpu_stats = stats.get('cpu_stats', {})
                            cpu_usage = self._calculate_cpu_percent(cpu_stats)
                        except Exception:
                            # Stats might not be available
                            pass
                    
                    instance_info = InstanceInfo(
                        container_id=container.id,
                        agent_id=container.labels.get('core.agent_id', 'unknown'),
                        agent_role=container.labels.get('core.role', 'unknown'),
                        status=container.status,
                        uptime_seconds=uptime,
                        memory_usage=memory_usage,
                        cpu_usage=cpu_usage,
                        created_at=instance_record.created_at if instance_record else datetime.utcnow(),
                        last_heartbeat=instance_record.last_heartbeat if instance_record else None
                    )
                    
                    instances.append(instance_info)
                    
                except Exception as e:
                    logger.warning(f"Error processing container {container.id[:12]}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error listing containers: {e}")
            raise
        
        return instances
    
    def _calculate_cpu_percent(self, cpu_stats: Dict[str, Any]) -> Optional[float]:
        """Calculate CPU usage percentage from Docker stats."""
        try:
            cpu_delta = cpu_stats['cpu_usage']['total_usage'] - \
                       cpu_stats.get('precpu_stats', {}).get('cpu_usage', {}).get('total_usage', 0)
            system_delta = cpu_stats['system_cpu_usage'] - \
                          cpu_stats.get('precpu_stats', {}).get('system_cpu_usage', 0)
            
            if system_delta > 0 and cpu_delta > 0:
                num_cpus = len(cpu_stats['cpu_usage']['percpu_usage'])
                return (cpu_delta / system_delta) * num_cpus * 100
        except (KeyError, ZeroDivisionError, TypeError):
            pass
        return None
    
    async def get_instance_status(self, container_id: str) -> Optional[InstanceInfo]:
        """
        Get detailed status for a specific instance.
        
        Args:
            container_id: Container ID to query
            
        Returns:
            InstanceInfo with detailed status, or None if not found
        """
        if not self.docker_client:
            raise RuntimeError("InstanceManager not initialized")
            
        try:
            container = self.docker_client.containers.get(container_id)
            instance_record = await get_instance_by_container_id(container_id)
            
            # Get detailed stats
            uptime = None
            memory_usage = None
            cpu_usage = None
            health_status = None
            
            if container.status == "running":
                created_time = container.attrs['Created']
                created_dt = datetime.fromisoformat(created_time.rstrip('Z'))
                uptime = int((datetime.utcnow() - created_dt).total_seconds())
                
                # Health check
                health_status = container.attrs.get('State', {}).get('Health', {}).get('Status')
                
                # Resource usage
                try:
                    stats = container.stats(stream=False)
                    memory_usage = stats.get('memory_stats', {})
                    cpu_stats = stats.get('cpu_stats', {})
                    cpu_usage = self._calculate_cpu_percent(cpu_stats)
                except Exception:
                    pass
            
            return InstanceInfo(
                container_id=container.id,
                agent_id=container.labels.get('core.agent_id', 'unknown'),
                agent_role=container.labels.get('core.role', 'unknown'),
                status=container.status,
                health_status=health_status,
                uptime_seconds=uptime,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                created_at=instance_record.created_at if instance_record else datetime.utcnow(),
                last_heartbeat=instance_record.last_heartbeat if instance_record else None
            )
            
        except NotFound:
            return None
        except Exception as e:
            logger.error(f"Error getting instance status {container_id}: {e}")
            raise
    
    async def scale_instances(self, role: str, target_count: int) -> Dict[str, Any]:
        """
        Scale instances of a role to target count.
        
        Spawns or stops containers to match target count.
        
        Args:
            role: Agent role to scale
            target_count: Target number of instances
            
        Returns:
            Dict with scaling results
        """
        if not self.docker_client:
            raise RuntimeError("InstanceManager not initialized")
            
        if target_count < 0:
            raise ValueError("Target count cannot be negative")
        
        # Get current instances for this role
        current_instances = []
        try:
            containers = self.docker_client.containers.list(
                all=True,
                filters={"label": f"core.role={role}"}
            )
            
            for container in containers:
                if container.status in ["running", "restarting"]:
                    current_instances.append({
                        "container_id": container.id,
                        "agent_id": container.labels.get('core.agent_id'),
                        "status": container.status
                    })
                        
        except Exception as e:
            logger.error(f"Error listing instances for role {role}: {e}")
            raise
        
        current_count = len(current_instances)
        actions_taken = []
        
        logger.info(f"Scaling {role}: {current_count} -> {target_count}")
        
        if current_count < target_count:
            # Need to spawn more instances
            spawn_count = target_count - current_count
            
            for i in range(spawn_count):
                try:
                    config = InstanceConfig(
                        agent_id=f"{role}-{str(uuid4())[:8]}",
                        agent_role=role,
                        resource_profile={"tier": "standard"},
                        capabilities=[]
                    )
                    
                    instance = await self.spawn_instance(config)
                    actions_taken.append({
                        "action": "spawn",
                        "container_id": instance.container_id,
                        "agent_id": instance.agent_id
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to spawn instance {i+1}/{spawn_count} for role {role}: {e}")
                    actions_taken.append({
                        "action": "spawn_failed",
                        "error": str(e)
                    })
        
        elif current_count > target_count:
            # Need to stop some instances
            stop_count = current_count - target_count
            instances_to_stop = current_instances[-stop_count:]  # Remove newest first
            
            for instance in instances_to_stop:
                try:
                    success = await self.stop_instance(instance["container_id"])
                    actions_taken.append({
                        "action": "stop" if success else "stop_failed",
                        "container_id": instance["container_id"],
                        "agent_id": instance["agent_id"]
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to stop instance {instance['container_id']}: {e}")
                    actions_taken.append({
                        "action": "stop_failed",
                        "container_id": instance["container_id"],
                        "error": str(e)
                    })
        
        # Get final count
        final_containers = self.docker_client.containers.list(
            filters={"label": f"core.role={role}"}
        )
        final_count = len([c for c in final_containers if c.status in ["running", "restarting"]])
        
        return {
            "role": role,
            "target_count": target_count,
            "initial_count": current_count,
            "final_count": final_count,
            "actions_taken": actions_taken,
            "success": final_count == target_count
        }


# Global instance manager
instance_manager = InstanceManager()