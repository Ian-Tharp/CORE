"""
Container Manager for Sandboxed Agent Execution

Manages Docker containers for isolated agent task execution.
Supports connection pooling, resource limits, and secure execution.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from uuid import uuid4
from enum import Enum

import docker
from docker.errors import ContainerError, ImageNotFound, APIError
from docker.models.containers import Container
from pydantic import BaseModel, Field

from .security import TrustLevel, AgentSecurityConfig, TRUST_PRESETS, get_security_config

logger = logging.getLogger(__name__)


class ContainerStatus(str, Enum):
    """Status of a managed container."""
    CREATING = "creating"
    RUNNING = "running"
    IDLE = "idle"
    STOPPED = "stopped"
    ERROR = "error"


class ContainerConfig(BaseModel):
    """Configuration for creating an agent container."""
    
    name: str = Field(default_factory=lambda: f"core-agent-{uuid4().hex[:8]}")
    image: str = "python:3.12-slim"
    
    # Security configuration
    security: AgentSecurityConfig = Field(default_factory=lambda: TRUST_PRESETS[TrustLevel.TRUSTED])
    
    # Environment
    environment: Dict[str, str] = Field(default_factory=dict)
    working_dir: str = "/workspace"
    
    # Volumes
    volumes: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    
    # MCP Configuration
    mcp_capabilities: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True


class ContainerInfo(BaseModel):
    """Information about a managed container."""
    
    id: str
    name: str
    status: ContainerStatus
    created_at: datetime
    last_activity: datetime
    config: ContainerConfig
    execution_count: int = 0
    total_execution_time_ms: int = 0
    
    class Config:
        arbitrary_types_allowed = True


class ExecutionResult(BaseModel):
    """Result of executing a command in a container."""
    
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    execution_time_ms: int
    container_id: str
    
    class Config:
        arbitrary_types_allowed = True


class ContainerManager:
    """
    Manages Docker containers for sandboxed agent execution.
    
    Features:
    - Container pooling for reduced startup latency
    - Configurable security levels
    - Resource monitoring and limits
    - Automatic cleanup
    """
    
    def __init__(
        self,
        docker_url: Optional[str] = None,
        pool_size: int = 3,
        container_ttl_minutes: int = 30,
        default_image: str = "python:3.12-slim"
    ):
        """
        Initialize the container manager.
        
        Args:
            docker_url: Docker daemon URL (None for default)
            pool_size: Number of warm containers to maintain per trust level
            container_ttl_minutes: Container lifetime before cleanup
            default_image: Default Docker image for containers
        """
        self.docker_url = docker_url
        self.pool_size = pool_size
        self.container_ttl = timedelta(minutes=container_ttl_minutes)
        self.default_image = default_image
        
        # Docker client (lazily initialized)
        self._client: Optional[docker.DockerClient] = None
        
        # Container pools by trust level
        self._pools: Dict[TrustLevel, List[ContainerInfo]] = {
            level: [] for level in TrustLevel
        }
        
        # Active containers
        self._active_containers: Dict[str, ContainerInfo] = {}
        
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
        
        # Cleanup task handle
        self._cleanup_task: Optional[asyncio.Task] = None
    
    @property
    def client(self) -> docker.DockerClient:
        """Lazily initialize Docker client."""
        if self._client is None:
            if self.docker_url:
                self._client = docker.DockerClient(base_url=self.docker_url)
            else:
                self._client = docker.from_env()
        return self._client
    
    async def initialize(self):
        """Initialize the container manager and warm pools."""
        logger.info("Initializing ContainerManager...")
        
        # Verify Docker connection
        try:
            self.client.ping()
            logger.info("Docker connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Docker: {e}")
            raise
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Optionally warm pools (can be done lazily instead)
        # await self._warm_pools()
        
        logger.info("ContainerManager initialized")
    
    async def shutdown(self):
        """Shutdown the container manager and cleanup."""
        logger.info("Shutting down ContainerManager...")
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Stop all containers
        await self._cleanup_all_containers()
        
        # Close Docker client
        if self._client:
            self._client.close()
            self._client = None
        
        logger.info("ContainerManager shutdown complete")
    
    async def get_container(
        self,
        config: ContainerConfig,
        reuse: bool = True
    ) -> ContainerInfo:
        """
        Get a container for execution.
        
        Args:
            config: Container configuration
            reuse: Whether to reuse existing containers from pool
            
        Returns:
            ContainerInfo for the allocated container
        """
        async with self._lock:
            trust_level = config.security.trust_level
            
            # Try to get from pool if reuse is enabled
            if reuse and self._pools[trust_level]:
                container_info = self._pools[trust_level].pop()
                container_info.status = ContainerStatus.RUNNING
                container_info.last_activity = datetime.utcnow()
                self._active_containers[container_info.id] = container_info
                logger.debug(f"Reusing container {container_info.id} from pool")
                return container_info
            
            # Create new container
            container_info = await self._create_container(config)
            self._active_containers[container_info.id] = container_info
            return container_info
    
    async def release_container(self, container_id: str, return_to_pool: bool = True):
        """
        Release a container back to the pool or destroy it.
        
        Args:
            container_id: ID of the container to release
            return_to_pool: Whether to return to pool for reuse
        """
        async with self._lock:
            if container_id not in self._active_containers:
                logger.warning(f"Attempted to release unknown container: {container_id}")
                return
            
            container_info = self._active_containers.pop(container_id)
            
            if return_to_pool:
                container_info.status = ContainerStatus.IDLE
                container_info.last_activity = datetime.utcnow()
                trust_level = container_info.config.security.trust_level
                
                # Check pool size limit
                if len(self._pools[trust_level]) < self.pool_size:
                    self._pools[trust_level].append(container_info)
                    logger.debug(f"Returned container {container_id} to pool")
                    return
            
            # Destroy if not returning to pool or pool is full
            await self._destroy_container(container_info)
    
    async def execute(
        self,
        container_id: str,
        command: str,
        timeout: Optional[int] = None,
        environment: Optional[Dict[str, str]] = None,
        working_dir: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute a command in a container.
        
        Args:
            container_id: ID of the container
            command: Command to execute
            timeout: Execution timeout in seconds
            environment: Additional environment variables
            working_dir: Working directory for command
            
        Returns:
            ExecutionResult with stdout, stderr, and exit code
        """
        if container_id not in self._active_containers:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Container {container_id} not found",
                execution_time_ms=0,
                container_id=container_id
            )
        
        container_info = self._active_containers[container_id]
        security = container_info.config.security
        
        # Use security timeout if not specified
        if timeout is None:
            timeout = security.timeout_seconds
        
        start_time = datetime.utcnow()
        
        try:
            # Get Docker container object
            container = self.client.containers.get(container_id)
            
            # Build exec environment
            exec_env = container_info.config.environment.copy()
            if environment:
                exec_env.update(environment)
            
            # Execute command
            exec_result = container.exec_run(
                cmd=command,
                environment=exec_env,
                workdir=working_dir or container_info.config.working_dir,
                stdout=security.capture_stdout,
                stderr=security.capture_stderr,
                demux=True
            )
            
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Update container stats
            container_info.execution_count += 1
            container_info.total_execution_time_ms += execution_time_ms
            container_info.last_activity = datetime.utcnow()
            
            stdout = exec_result.output[0].decode() if exec_result.output[0] else ""
            stderr = exec_result.output[1].decode() if exec_result.output[1] else ""
            
            # Log command if required
            if security.log_all_commands:
                logger.info(f"Container {container_id} executed: {command[:100]}...")
            
            return ExecutionResult(
                success=exec_result.exit_code == 0,
                exit_code=exec_result.exit_code,
                stdout=stdout,
                stderr=stderr,
                execution_time_ms=execution_time_ms,
                container_id=container_id
            )
        
        except Exception as e:
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            logger.error(f"Execution failed in container {container_id}: {e}")
            
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time_ms=execution_time_ms,
                container_id=container_id
            )
    
    async def execute_python(
        self,
        container_id: str,
        code: str,
        timeout: Optional[int] = None
    ) -> ExecutionResult:
        """
        Execute Python code in a container.
        
        Args:
            container_id: ID of the container
            code: Python code to execute
            timeout: Execution timeout in seconds
            
        Returns:
            ExecutionResult with stdout, stderr, and exit code
        """
        # Escape the code for shell
        escaped_code = code.replace("'", "'\"'\"'")
        command = f"python -c '{escaped_code}'"
        
        return await self.execute(container_id, command, timeout)
    
    async def _create_container(self, config: ContainerConfig) -> ContainerInfo:
        """Create a new Docker container."""
        security = config.security
        
        # Build Docker container config
        docker_config = self._build_docker_config(config)
        
        try:
            # Create container
            container = self.client.containers.create(**docker_config)
            
            # Start container
            container.start()
            
            container_info = ContainerInfo(
                id=container.id,
                name=config.name,
                status=ContainerStatus.RUNNING,
                created_at=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                config=config
            )
            
            logger.info(f"Created container {container.id} with trust level {security.trust_level}")
            return container_info
        
        except ImageNotFound:
            logger.info(f"Pulling image {config.image}...")
            self.client.images.pull(config.image)
            return await self._create_container(config)
        
        except Exception as e:
            logger.error(f"Failed to create container: {e}")
            raise
    
    def _build_docker_config(self, config: ContainerConfig) -> Dict[str, Any]:
        """Build Docker container configuration from ContainerConfig."""
        security = config.security
        
        docker_config: Dict[str, Any] = {
            "name": config.name,
            "image": config.image,
            "detach": True,
            "tty": True,
            "stdin_open": True,
            "working_dir": config.working_dir,
            "environment": config.environment,
        }
        
        # Resource limits
        docker_config["mem_limit"] = f"{security.memory_limit_mb}m"
        docker_config["cpu_period"] = 100000
        docker_config["cpu_quota"] = int(security.cpu_limit * 100000)
        docker_config["pids_limit"] = security.max_processes
        
        # Network configuration
        if not security.network_enabled:
            docker_config["network_mode"] = "none"
        elif security.allowed_hosts:
            # Use custom network with restrictions
            # For now, use bridge mode (implement network policies separately)
            docker_config["network_mode"] = "bridge"
        
        # Filesystem configuration
        if security.read_only_root:
            docker_config["read_only"] = True
            # Add tmpfs for writable paths
            tmpfs_config = {}
            for path in security.writable_paths:
                tmpfs_config[path] = f"size={security.max_file_size_mb}m"
            if tmpfs_config:
                docker_config["tmpfs"] = tmpfs_config
        
        # Volumes
        if config.volumes:
            docker_config["volumes"] = config.volumes
        
        # Security options for untrusted
        if security.trust_level == TrustLevel.UNTRUSTED:
            docker_config["security_opt"] = ["no-new-privileges"]
            # Could add gVisor runtime here if available
            # docker_config["runtime"] = "runsc"
        
        return docker_config
    
    async def _destroy_container(self, container_info: ContainerInfo):
        """Destroy a container."""
        try:
            container = self.client.containers.get(container_info.id)
            container.stop(timeout=5)
            container.remove(force=True)
            logger.debug(f"Destroyed container {container_info.id}")
        except Exception as e:
            logger.warning(f"Error destroying container {container_info.id}: {e}")
    
    async def _cleanup_loop(self):
        """Background task to cleanup expired containers."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_expired_containers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_expired_containers(self):
        """Cleanup containers that have exceeded their TTL."""
        async with self._lock:
            now = datetime.utcnow()
            
            for trust_level in TrustLevel:
                pool = self._pools[trust_level]
                expired = [
                    c for c in pool
                    if now - c.last_activity > self.container_ttl
                ]
                
                for container_info in expired:
                    pool.remove(container_info)
                    await self._destroy_container(container_info)
                    logger.debug(f"Cleaned up expired container {container_info.id}")
    
    async def _cleanup_all_containers(self):
        """Cleanup all managed containers."""
        async with self._lock:
            # Cleanup pools
            for trust_level in TrustLevel:
                for container_info in self._pools[trust_level]:
                    await self._destroy_container(container_info)
                self._pools[trust_level].clear()
            
            # Cleanup active containers
            for container_info in self._active_containers.values():
                await self._destroy_container(container_info)
            self._active_containers.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """Get status information about the container manager."""
        return {
            "pools": {
                level.value: len(self._pools[level])
                for level in TrustLevel
            },
            "active_containers": len(self._active_containers),
            "pool_size_limit": self.pool_size,
            "container_ttl_minutes": self.container_ttl.total_seconds() / 60
        }
