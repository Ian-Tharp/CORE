"""
Tests for InstanceManager service.

Tests follow AAA format (Arrange, Act, Assert) and mock the Docker SDK
to avoid creating actual containers during testing.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from datetime import datetime, timedelta
from uuid import uuid4, UUID

from app.services.instance_manager import (
    InstanceManager,
    InstanceConfig,
    InstanceInfo,
    ScaleRequest
)
from app.repository.instance_repository import (
    AgentInstance,
    InstanceStatus,
    InstanceTrustMetrics
)


@pytest.fixture
def mock_docker_client():
    """Mock Docker client for testing."""
    client = MagicMock()
    
    # Mock networks
    client.networks.get.return_value = MagicMock()
    client.networks.create.return_value = MagicMock()
    
    # Mock containers
    client.containers.run.return_value = MagicMock(id="mock_container_id_12345")
    client.containers.get.return_value = MagicMock(
        id="mock_container_id_12345",
        status="running",
        labels={"core.agent": "true", "core.role": "test", "core.agent_id": "test-agent"}
    )
    client.containers.list.return_value = []
    
    return client


@pytest.fixture
def instance_manager():
    """Instance manager fixture."""
    return InstanceManager()


@pytest.fixture
def sample_instance_config():
    """Sample instance configuration."""
    return InstanceConfig(
        agent_id="test-agent-001",
        agent_role="reasoning",
        device_id="test-device",
        resource_profile={"tier": "standard"},
        capabilities=["file_access", "web_search"],
        environment_vars={"TEST_VAR": "test_value"},
        memory_limit="1g",
        cpu_limit=1.0,
        network="core-network"
    )


@pytest.fixture
def sample_agent_instance():
    """Sample agent instance database record."""
    return AgentInstance(
        id=uuid4(),
        container_id="mock_container_id_12345",
        agent_id="test-agent-001",
        agent_role="reasoning",
        status=InstanceStatus.READY,
        device_id="test-device",
        resource_profile={"tier": "standard"},
        capabilities=["file_access", "web_search"],
        created_at=datetime.utcnow(),
        last_heartbeat=datetime.utcnow()
    )


class TestInstanceManagerInitialization:
    """Test InstanceManager initialization and setup."""
    
    @patch('app.services.instance_manager.docker')
    async def test_initialize_creates_docker_client(self, mock_docker):
        """Test that initialize creates Docker client and network."""
        # Arrange
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.networks.get.side_effect = Exception("Network not found")
        
        manager = InstanceManager()
        
        # Act
        await manager.initialize()
        
        # Assert
        assert manager.docker_client is not None
        mock_docker.from_env.assert_called_once()
        mock_client.networks.create.assert_called_once_with(
            "core-network",
            driver="bridge",
            labels={"project": "core", "component": "agent-network"}
        )
    
    @patch('app.services.instance_manager.docker')
    async def test_initialize_uses_existing_network(self, mock_docker):
        """Test that initialize uses existing core-network if present."""
        # Arrange
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.networks.get.return_value = MagicMock()  # Network exists
        
        manager = InstanceManager()
        
        # Act
        await manager.initialize()
        
        # Assert
        mock_client.networks.get.assert_called_once_with("core-network")
        mock_client.networks.create.assert_not_called()
    
    async def test_shutdown_closes_client(self):
        """Test that shutdown properly closes Docker client."""
        # Arrange
        manager = InstanceManager()
        mock_client = MagicMock()
        manager.docker_client = mock_client
        
        # Act
        await manager.shutdown()
        
        # Assert
        assert manager._shutdown is True
        mock_client.close.assert_called_once()


class TestSpawnInstance:
    """Test instance spawning functionality."""
    
    @patch('app.repository.instance_repository.create_instance')
    @patch('app.repository.instance_repository.update_instance_status')
    async def test_spawn_instance_creates_container_with_correct_labels(
        self, mock_update_status, mock_create_instance, mock_docker_client, sample_instance_config
    ):
        """Test that spawn_instance creates container with correct labels and config."""
        # Arrange
        manager = InstanceManager()
        manager.docker_client = mock_docker_client
        
        mock_container = MagicMock()
        mock_container.id = "test_container_123"
        mock_docker_client.containers.run.return_value = mock_container
        
        mock_create_instance.return_value = uuid4()
        mock_update_status.return_value = True
        
        # Act
        result = await manager.spawn_instance(sample_instance_config)
        
        # Assert
        assert isinstance(result, InstanceInfo)
        assert result.container_id == "test_container_123"
        assert result.agent_id == sample_instance_config.agent_id
        assert result.agent_role == sample_instance_config.agent_role
        assert result.status == "ready"
        
        # Check that container was created with correct config
        mock_docker_client.containers.run.assert_called_once()
        call_kwargs = mock_docker_client.containers.run.call_args[1]
        
        assert call_kwargs["image"] == "core-agent:latest"
        assert call_kwargs["environment"]["AGENT_ID"] == sample_instance_config.agent_id
        assert call_kwargs["environment"]["AGENT_ROLE"] == sample_instance_config.agent_role
        assert call_kwargs["environment"]["TEST_VAR"] == "test_value"
        assert call_kwargs["labels"]["core.agent"] == "true"
        assert call_kwargs["labels"]["core.role"] == sample_instance_config.agent_role
        assert call_kwargs["mem_limit"] == sample_instance_config.memory_limit
        
        # Check database operations
        mock_create_instance.assert_called_once()
        mock_update_status.assert_called_once_with("test_container_123", InstanceStatus.READY)
    
    @patch('app.repository.instance_repository.create_instance')
    async def test_spawn_instance_handles_docker_failure(
        self, mock_create_instance, mock_docker_client, sample_instance_config
    ):
        """Test that spawn_instance properly handles Docker failures."""
        # Arrange
        manager = InstanceManager()
        manager.docker_client = mock_docker_client
        mock_docker_client.containers.run.side_effect = Exception("Docker error")
        
        # Act & Assert
        with pytest.raises(Exception, match="Docker error"):
            await manager.spawn_instance(sample_instance_config)
        
        # Database record should not be created if Docker fails
        mock_create_instance.assert_not_called()
    
    async def test_spawn_instance_requires_initialization(self, sample_instance_config):
        """Test that spawn_instance raises error if not initialized."""
        # Arrange
        manager = InstanceManager()
        # Don't initialize - docker_client will be None
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="InstanceManager not initialized"):
            await manager.spawn_instance(sample_instance_config)


class TestStopInstance:
    """Test instance stopping functionality."""
    
    @patch('app.repository.instance_repository.update_instance_status')
    async def test_stop_instance_gracefully_stops_container(
        self, mock_update_status, mock_docker_client
    ):
        """Test that stop_instance gracefully stops container."""
        # Arrange
        manager = InstanceManager()
        manager.docker_client = mock_docker_client
        
        mock_container = MagicMock()
        mock_docker_client.containers.get.return_value = mock_container
        mock_update_status.return_value = True
        
        container_id = "test_container_123"
        
        # Act
        result = await manager.stop_instance(container_id)
        
        # Assert
        assert result is True
        mock_docker_client.containers.get.assert_called_once_with(container_id)
        mock_container.stop.assert_called_once_with(timeout=10)
        
        # Check status updates
        assert mock_update_status.call_count == 2
        mock_update_status.assert_any_call(container_id, InstanceStatus.STOPPING)
        mock_update_status.assert_any_call(container_id, InstanceStatus.STOPPED)
    
    @patch('app.repository.instance_repository.update_instance_status')
    async def test_stop_instance_handles_missing_container(
        self, mock_update_status, mock_docker_client
    ):
        """Test that stop_instance handles missing containers gracefully."""
        # Arrange
        manager = InstanceManager()
        manager.docker_client = mock_docker_client
        
        from docker.errors import NotFound
        mock_docker_client.containers.get.side_effect = NotFound("Container not found")
        mock_update_status.return_value = True
        
        container_id = "missing_container_123"
        
        # Act
        result = await manager.stop_instance(container_id)
        
        # Assert
        assert result is False
        # Should still update database status
        mock_update_status.assert_called_once_with(container_id, InstanceStatus.STOPPED)


class TestRestartInstance:
    """Test instance restarting functionality."""
    
    @patch('app.repository.instance_repository.update_instance_status')
    async def test_restart_instance_restarts_and_waits_for_healthy(
        self, mock_update_status, mock_docker_client
    ):
        """Test that restart_instance restarts container and waits for health."""
        # Arrange
        manager = InstanceManager()
        manager.docker_client = mock_docker_client
        
        mock_container = MagicMock()
        mock_container.status = "running"  # Healthy after restart
        mock_docker_client.containers.get.return_value = mock_container
        mock_update_status.return_value = True
        
        container_id = "test_container_123"
        
        # Act
        result = await manager.restart_instance(container_id)
        
        # Assert
        assert result is True
        mock_container.restart.assert_called_once_with(timeout=10)
        
        # Check status updates
        mock_update_status.assert_any_call(container_id, InstanceStatus.RESTARTING)
        mock_update_status.assert_any_call(container_id, InstanceStatus.READY)


class TestListInstances:
    """Test instance listing functionality."""
    
    @patch('app.repository.instance_repository.get_instance_by_container_id')
    async def test_list_instances_filters_by_agent_label(
        self, mock_get_instance, mock_docker_client, sample_agent_instance
    ):
        """Test that list_instances filters containers by core.agent=true label."""
        # Arrange
        manager = InstanceManager()
        manager.docker_client = mock_docker_client
        
        mock_container1 = MagicMock()
        mock_container1.id = "container1"
        mock_container1.status = "running"
        mock_container1.labels = {
            "core.agent": "true",
            "core.role": "reasoning",
            "core.agent_id": "agent1"
        }
        
        mock_container2 = MagicMock()
        mock_container2.id = "container2"
        mock_container2.status = "stopped"
        mock_container2.labels = {
            "core.agent": "true",
            "core.role": "evaluation",
            "core.agent_id": "agent2"
        }
        
        mock_docker_client.containers.list.return_value = [mock_container1, mock_container2]
        mock_get_instance.return_value = sample_agent_instance
        
        # Act
        result = await manager.list_instances()
        
        # Assert
        assert len(result) == 2
        assert all(isinstance(instance, InstanceInfo) for instance in result)
        
        # Check filter was applied
        mock_docker_client.containers.list.assert_called_once_with(
            all=True,
            filters={"label": "core.agent=true"}
        )


class TestScaleInstances:
    """Test instance scaling functionality."""
    
    @patch('app.services.instance_manager.uuid4')
    @patch.object(InstanceManager, 'spawn_instance')
    async def test_scale_instances_spawns_additional_containers(
        self, mock_spawn, mock_uuid4, mock_docker_client
    ):
        """Test that scale_instances spawns containers to reach target count."""
        # Arrange
        manager = InstanceManager()
        manager.docker_client = mock_docker_client
        
        # No existing containers for this role
        mock_docker_client.containers.list.side_effect = [
            [],  # Initial count: 0
            [MagicMock(), MagicMock()]  # Final count: 2
        ]
        
        mock_uuid4.return_value = UUID('12345678-1234-5678-9012-123456789012')
        
        mock_spawn.return_value = InstanceInfo(
            container_id="new_container_123",
            agent_id="reasoning-12345678",
            agent_role="reasoning",
            status="ready",
            created_at=datetime.utcnow()
        )
        
        # Act
        result = await manager.scale_instances("reasoning", 2)
        
        # Assert
        assert result["success"] is True
        assert result["role"] == "reasoning"
        assert result["target_count"] == 2
        assert result["initial_count"] == 0
        assert result["final_count"] == 2
        assert len(result["actions_taken"]) == 2
        
        # Check spawn was called twice
        assert mock_spawn.call_count == 2
        
        # Verify spawn called with correct config
        for call in mock_spawn.call_args_list:
            config = call[0][0]
            assert config.agent_role == "reasoning"
            assert config.agent_id.startswith("reasoning-")
    
    @patch.object(InstanceManager, 'stop_instance')
    async def test_scale_instances_stops_excess_containers(
        self, mock_stop, mock_docker_client
    ):
        """Test that scale_instances stops containers when scaling down."""
        # Arrange
        manager = InstanceManager()
        manager.docker_client = mock_docker_client
        
        # Mock existing containers
        mock_container1 = MagicMock()
        mock_container1.id = "container1"
        mock_container1.status = "running"
        mock_container1.labels = {"core.agent_id": "agent1"}
        
        mock_container2 = MagicMock()
        mock_container2.id = "container2"
        mock_container2.status = "running"
        mock_container2.labels = {"core.agent_id": "agent2"}
        
        mock_container3 = MagicMock()
        mock_container3.id = "container3"
        mock_container3.status = "running"
        mock_container3.labels = {"core.agent_id": "agent3"}
        
        mock_docker_client.containers.list.side_effect = [
            [mock_container1, mock_container2, mock_container3],  # Initial: 3
            [mock_container1]  # Final: 1
        ]
        
        mock_stop.return_value = True
        
        # Act
        result = await manager.scale_instances("reasoning", 1)
        
        # Assert
        assert result["success"] is True
        assert result["initial_count"] == 3
        assert result["final_count"] == 1
        assert len(result["actions_taken"]) == 2
        
        # Check stop was called twice (for the last 2 containers)
        assert mock_stop.call_count == 2
    
    async def test_scale_instances_validates_negative_target(self, mock_docker_client):
        """Test that scale_instances rejects negative target count."""
        # Arrange
        manager = InstanceManager()
        manager.docker_client = mock_docker_client
        
        # Act & Assert
        with pytest.raises(ValueError, match="Target count cannot be negative"):
            await manager.scale_instances("reasoning", -1)


class TestGetInstanceStatus:
    """Test instance status retrieval."""
    
    @patch('app.repository.instance_repository.get_instance_by_container_id')
    async def test_get_instance_status_returns_detailed_info(
        self, mock_get_instance, mock_docker_client, sample_agent_instance
    ):
        """Test that get_instance_status returns detailed information."""
        # Arrange
        manager = InstanceManager()
        manager.docker_client = mock_docker_client
        
        mock_container = MagicMock()
        mock_container.id = "test_container_123"
        mock_container.status = "running"
        mock_container.labels = {
            "core.agent_id": "test-agent",
            "core.role": "reasoning"
        }
        mock_container.attrs = {
            'Created': '2025-01-27T10:00:00.000000Z',
            'State': {'Health': {'Status': 'healthy'}}
        }
        mock_container.stats.return_value = {
            'memory_stats': {'usage': 1000000},
            'cpu_stats': {
                'cpu_usage': {'total_usage': 1000000},
                'system_cpu_usage': 2000000
            }
        }
        
        mock_docker_client.containers.get.return_value = mock_container
        mock_get_instance.return_value = sample_agent_instance
        
        container_id = "test_container_123"
        
        # Act
        result = await manager.get_instance_status(container_id)
        
        # Assert
        assert result is not None
        assert isinstance(result, InstanceInfo)
        assert result.container_id == container_id
        assert result.status == "running"
        assert result.health_status == "healthy"
        assert result.uptime_seconds is not None
        assert result.memory_usage is not None
    
    async def test_get_instance_status_returns_none_for_missing_container(
        self, mock_docker_client
    ):
        """Test that get_instance_status returns None for missing containers."""
        # Arrange
        manager = InstanceManager()
        manager.docker_client = mock_docker_client
        
        from docker.errors import NotFound
        mock_docker_client.containers.get.side_effect = NotFound("Container not found")
        
        # Act
        result = await manager.get_instance_status("missing_container")
        
        # Assert
        assert result is None


class TestInstanceManagerIntegration:
    """Integration tests for InstanceManager."""
    
    @patch('app.repository.instance_repository.create_instance')
    @patch('app.repository.instance_repository.update_instance_status')
    async def test_full_lifecycle_spawn_and_stop(
        self, mock_update_status, mock_create_instance, mock_docker_client, sample_instance_config
    ):
        """Test full lifecycle: spawn, check status, then stop instance."""
        # Arrange
        manager = InstanceManager()
        manager.docker_client = mock_docker_client
        
        container_id = "lifecycle_test_123"
        mock_container = MagicMock()
        mock_container.id = container_id
        mock_docker_client.containers.run.return_value = mock_container
        mock_docker_client.containers.get.return_value = mock_container
        
        mock_create_instance.return_value = uuid4()
        mock_update_status.return_value = True
        
        # Act 1: Spawn
        spawn_result = await manager.spawn_instance(sample_instance_config)
        
        # Assert 1: Spawn successful
        assert spawn_result.container_id == container_id
        assert spawn_result.status == "ready"
        
        # Act 2: Stop
        stop_result = await manager.stop_instance(container_id)
        
        # Assert 2: Stop successful
        assert stop_result is True
        mock_container.stop.assert_called_once_with(timeout=10)
        
        # Check final status updates
        final_calls = [call for call in mock_update_status.call_args_list if call[0][1] == InstanceStatus.STOPPED]
        assert len(final_calls) >= 1  # Should have at least one STOPPED status update


class TestInstanceManagerErrorHandling:
    """Test error handling scenarios."""
    
    async def test_operations_require_initialization(self, sample_instance_config):
        """Test that operations fail gracefully when not initialized."""
        # Arrange
        manager = InstanceManager()
        # Don't initialize
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="InstanceManager not initialized"):
            await manager.spawn_instance(sample_instance_config)
        
        with pytest.raises(RuntimeError, match="InstanceManager not initialized"):
            await manager.stop_instance("test_container")
        
        with pytest.raises(RuntimeError, match="InstanceManager not initialized"):
            await manager.list_instances()
    
    @patch('app.services.instance_manager.logger')
    async def test_container_failures_are_logged(
        self, mock_logger, mock_docker_client, sample_instance_config
    ):
        """Test that container operation failures are properly logged."""
        # Arrange
        manager = InstanceManager()
        manager.docker_client = mock_docker_client
        
        mock_docker_client.containers.run.side_effect = Exception("Docker daemon error")
        
        # Act
        with pytest.raises(Exception):
            await manager.spawn_instance(sample_instance_config)
        
        # Assert
        mock_logger.error.assert_called()
        error_message = mock_logger.error.call_args[0][0]
        assert "Failed to spawn instance" in error_message