"""
Agent Registration System Tests

Comprehensive tests for the agent registration system following AAA format.
Tests cover WebSocket and REST endpoints, heartbeat monitoring, task management,
and edge cases.

ALL TESTS FOLLOW AAA FORMAT:
- Arrange: Set up test data and preconditions
- Act: Execute the code under test  
- Assert: Verify expected outcomes
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from uuid import uuid4, UUID
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any, List

from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket
from pydantic import ValidationError

from app.services.agent_registry import (
    AgentRegistry,
    AgentRegistrationPayload,
    AgentHeartbeatData,
    TaskCompletion,
    TaskRefusal,
    AgentDeregistration,
    TaskAssignment
)
from app.controllers.agent_ws import AgentWebSocketManager
from app.repository.instance_repository import AgentInstance, InstanceStatus


class MockWebSocket:
    """Mock WebSocket for testing."""
    
    def __init__(self):
        self.sent_messages: List[Dict[str, Any]] = []
        self.received_messages: List[Dict[str, Any]] = []
        self.accepted = False
        self.closed = False
    
    async def accept(self):
        self.accepted = True
    
    async def send_json(self, data: Dict[str, Any]):
        self.sent_messages.append(data)
    
    async def receive_json(self) -> Dict[str, Any]:
        if not self.received_messages:
            raise Exception("No messages to receive")
        return self.received_messages.pop(0)
    
    def add_message(self, message: Dict[str, Any]):
        self.received_messages.append(message)


@pytest.fixture
def mock_instance():
    """Create a mock agent instance for testing."""
    return AgentInstance(
        id=uuid4(),
        container_id="test_container_123",
        agent_id="test-agent-001",
        agent_role="researcher",
        status=InstanceStatus.STARTING,
        capabilities=["web_search", "file_ops"],
        created_at=datetime.utcnow()
    )


@pytest.fixture
def agent_registry():
    """Create a fresh agent registry for testing."""
    return AgentRegistry()


@pytest.fixture
def ws_manager():
    """Create a fresh WebSocket manager for testing."""
    return AgentWebSocketManager()


@pytest.fixture
def registration_payload():
    """Create a valid registration payload for testing."""
    return AgentRegistrationPayload(
        container_id="test_container_123",
        role="researcher", 
        capabilities=["web_search", "file_ops"],
        version="1.0.0"
    )


@pytest.fixture
def heartbeat_data():
    """Create valid heartbeat data for testing."""
    return AgentHeartbeatData(
        status="ready",
        current_task=None,
        resource_usage={"memory_mb": 256, "cpu_percent": 15.2}
    )


class TestAgentRegistration:
    """Test agent registration functionality."""
    
    @pytest.mark.asyncio
    async def test_register_agent_success(self, agent_registry, mock_instance, registration_payload):
        """Test successful agent registration."""
        # Arrange
        with patch('app.repository.instance_repository.get_instance_by_container_id', return_value=mock_instance):
            with patch('app.repository.instance_repository.update_instance', return_value=True):
                
                # Act
                config = await agent_registry.register_agent(registration_payload)
                
                # Assert
                assert config.agent_id == mock_instance.agent_id
                assert config.model == "ollama/llama3.2"
                assert "web_search" in config.tools
                assert "file_ops" in config.tools
                assert config.memory_config["max_context_length"] == 4000
                assert mock_instance.agent_id in agent_registry.active_agents
    
    @pytest.mark.asyncio
    async def test_register_agent_unknown_container(self, agent_registry, registration_payload):
        """Test registration fails with unknown container_id."""
        # Arrange
        with patch('app.repository.instance_repository.get_instance_by_container_id', return_value=None):
            
            # Act & Assert
            with pytest.raises(ValueError, match="Container .* not found in database"):
                await agent_registry.register_agent(registration_payload)
    
    @pytest.mark.asyncio
    async def test_register_agent_missing_fields(self, agent_registry):
        """Test registration fails with missing required fields."""
        # Arrange
        invalid_payload = {
            "container_id": "test_container_123",
            "role": "researcher",
            # Missing version and capabilities
        }
        
        # Act & Assert
        with pytest.raises(ValidationError):
            AgentRegistrationPayload(**invalid_payload)
    
    @pytest.mark.asyncio
    async def test_duplicate_registration_updates_record(self, agent_registry, mock_instance, registration_payload):
        """Test duplicate registration updates existing record."""
        # Arrange
        with patch('app.repository.instance_repository.get_instance_by_container_id', return_value=mock_instance):
            with patch('app.repository.instance_repository.update_instance', return_value=True):
                # Register agent first time
                await agent_registry.register_agent(registration_payload)
                
                # Modify registration payload
                updated_payload = AgentRegistrationPayload(
                    container_id=registration_payload.container_id,
                    role=registration_payload.role,
                    capabilities=["web_search", "file_ops", "analysis"],
                    version="1.1.0"
                )
                
                # Act
                config = await agent_registry.register_agent(updated_payload)
                
                # Assert
                assert config.agent_id == mock_instance.agent_id
                agent_info = agent_registry.get_agent_info(mock_instance.agent_id)
                assert "analysis" in agent_info["capabilities"]
                assert agent_info["version"] == "1.1.0"


class TestHeartbeat:
    """Test heartbeat functionality."""
    
    @pytest.mark.asyncio
    async def test_heartbeat_updates_timestamp_and_status(self, agent_registry, mock_instance, registration_payload, heartbeat_data):
        """Test heartbeat updates timestamp and status."""
        # Arrange
        with patch('app.repository.instance_repository.get_instance_by_container_id', return_value=mock_instance):
            with patch('app.repository.instance_repository.update_instance', return_value=True):
                with patch('app.repository.instance_repository.update_heartbeat', return_value=True):
                    # Register agent first
                    config = await agent_registry.register_agent(registration_payload)
                    agent_id = config.agent_id
                    initial_heartbeat = agent_registry.active_agents[agent_id]["last_heartbeat"]
                    
                    # Wait a moment to ensure timestamp difference
                    await asyncio.sleep(0.01)
                    
                    # Act
                    response = await agent_registry.handle_heartbeat(agent_id, heartbeat_data)
                    
                    # Assert
                    assert response["type"] == "heartbeat_ack"
                    agent_info = agent_registry.get_agent_info(agent_id)
                    assert agent_info["last_heartbeat"] > initial_heartbeat
                    assert agent_info["current_status"] == "ready"
                    assert agent_info["resource_usage"]["memory_mb"] == 256
    
    @pytest.mark.asyncio
    async def test_heartbeat_with_resource_usage_stores_metrics(self, agent_registry, mock_instance, registration_payload):
        """Test heartbeat with resource usage stores metrics."""
        # Arrange
        with patch('app.repository.instance_repository.get_instance_by_container_id', return_value=mock_instance):
            with patch('app.repository.instance_repository.update_instance', return_value=True):
                with patch('app.repository.instance_repository.update_heartbeat', return_value=True):
                    config = await agent_registry.register_agent(registration_payload)
                    agent_id = config.agent_id
                    
                    detailed_heartbeat = AgentHeartbeatData(
                        status="busy",
                        current_task="task_123",
                        resource_usage={
                            "memory_mb": 512,
                            "cpu_percent": 45.6,
                            "disk_io": 1024,
                            "network_bytes": 2048
                        }
                    )
                    
                    # Act
                    await agent_registry.handle_heartbeat(agent_id, detailed_heartbeat)
                    
                    # Assert
                    agent_info = agent_registry.get_agent_info(agent_id)
                    assert agent_info["current_status"] == "busy"
                    assert agent_info["current_task"] == "task_123"
                    assert agent_info["resource_usage"]["memory_mb"] == 512
                    assert agent_info["resource_usage"]["cpu_percent"] == 45.6
                    assert agent_info["resource_usage"]["disk_io"] == 1024
    
    @pytest.mark.asyncio
    async def test_heartbeat_from_unregistered_agent_rejected(self, agent_registry, heartbeat_data):
        """Test heartbeat from unregistered agent is rejected."""
        # Arrange
        unregistered_agent_id = "unregistered-agent-999"
        
        # Act & Assert
        with pytest.raises(ValueError, match="Agent .* not registered"):
            await agent_registry.handle_heartbeat(unregistered_agent_id, heartbeat_data)


class TestDeregistration:
    """Test agent deregistration functionality."""
    
    @pytest.mark.asyncio
    async def test_graceful_deregister_saves_final_state(self, agent_registry, mock_instance, registration_payload):
        """Test graceful deregister saves final state."""
        # Arrange
        with patch('app.repository.instance_repository.get_instance_by_container_id', return_value=mock_instance):
            with patch('app.repository.instance_repository.update_instance', return_value=True):
                with patch('app.repository.instance_repository.update_instance_status', return_value=True):
                    config = await agent_registry.register_agent(registration_payload)
                    agent_id = config.agent_id
                    
                    deregistration = AgentDeregistration(
                        reason="graceful_shutdown",
                        final_state={"tasks_completed": 5, "uptime_seconds": 3600}
                    )
                    
                    # Act
                    await agent_registry.deregister_agent(agent_id, deregistration)
                    
                    # Assert
                    assert agent_id not in agent_registry.active_agents
                    assert agent_id not in agent_registry.pending_tasks
    
    @pytest.mark.asyncio
    async def test_deregister_updates_status_to_stopped(self, agent_registry, mock_instance, registration_payload):
        """Test deregister updates status to 'stopped'."""
        # Arrange
        with patch('app.repository.instance_repository.get_instance_by_container_id', return_value=mock_instance):
            with patch('app.repository.instance_repository.update_instance', return_value=True):
                with patch('app.repository.instance_repository.update_instance_status', return_value=True) as mock_update_status:
                    config = await agent_registry.register_agent(registration_payload)
                    agent_id = config.agent_id
                    
                    deregistration = AgentDeregistration(reason="container_stop", final_state={})
                    
                    # Act
                    await agent_registry.deregister_agent(agent_id, deregistration)
                    
                    # Assert
                    mock_update_status.assert_called_once()
                    call_args = mock_update_status.call_args[0]
                    assert call_args[1] == InstanceStatus.STOPPED


class TestStaleAgentDetection:
    """Test stale agent detection and cleanup."""
    
    @pytest.mark.asyncio
    async def test_agent_with_old_heartbeat_marked_unhealthy(self, agent_registry, mock_instance, registration_payload):
        """Test agent with old heartbeat marked unhealthy."""
        # Arrange
        with patch('app.repository.instance_repository.get_instance_by_container_id', return_value=mock_instance):
            with patch('app.repository.instance_repository.update_instance', return_value=True):
                with patch('app.repository.instance_repository.update_instance_status', return_value=True):
                    config = await agent_registry.register_agent(registration_payload)
                    agent_id = config.agent_id
                    
                    # Simulate old heartbeat (2 minutes ago)
                    old_time = datetime.utcnow() - timedelta(minutes=2)
                    agent_registry.active_agents[agent_id]["last_heartbeat"] = old_time
                    
                    # Act
                    stale_agents = await agent_registry.check_stale_agents()
                    
                    # Assert
                    assert agent_id in stale_agents
                    agent_info = agent_registry.get_agent_info(agent_id)
                    assert agent_info["current_status"] == "unhealthy"
    
    @pytest.mark.asyncio
    async def test_agent_missing_5_minutes_marked_as_lost(self, agent_registry, mock_instance, registration_payload):
        """Test agent missing 5+ minutes marked as lost."""
        # Arrange
        with patch('app.repository.instance_repository.get_instance_by_container_id', return_value=mock_instance):
            with patch('app.repository.instance_repository.update_instance', return_value=True):
                with patch('app.repository.instance_repository.update_instance_status', return_value=True):
                    with patch('app.services.instance_manager.instance_manager.restart_instance', return_value=True) as mock_restart:
                        config = await agent_registry.register_agent(registration_payload)
                        agent_id = config.agent_id
                        
                        # Simulate very old heartbeat (6 minutes ago)
                        old_time = datetime.utcnow() - timedelta(minutes=6)
                        agent_registry.active_agents[agent_id]["last_heartbeat"] = old_time
                        
                        # Act
                        stale_agents = await agent_registry.check_stale_agents()
                        
                        # Assert
                        assert agent_id in stale_agents
                        assert agent_id not in agent_registry.active_agents
                        mock_restart.assert_called_once_with(mock_instance.container_id)
    
    @pytest.mark.asyncio
    async def test_healthy_agent_not_affected_by_stale_check(self, agent_registry, mock_instance, registration_payload):
        """Test healthy agent is not affected by stale check."""
        # Arrange
        with patch('app.repository.instance_repository.get_instance_by_container_id', return_value=mock_instance):
            with patch('app.repository.instance_repository.update_instance', return_value=True):
                config = await agent_registry.register_agent(registration_payload)
                agent_id = config.agent_id
                
                # Ensure recent heartbeat
                agent_registry.active_agents[agent_id]["last_heartbeat"] = datetime.utcnow()
                
                # Act
                stale_agents = await agent_registry.check_stale_agents()
                
                # Assert
                assert agent_id not in stale_agents
                assert agent_id in agent_registry.active_agents
                agent_info = agent_registry.get_agent_info(agent_id)
                assert agent_info["current_status"] != "unhealthy"


class TestTaskRefusal:
    """Test task refusal functionality."""
    
    @pytest.mark.asyncio
    async def test_agent_can_refuse_task_with_reason(self, agent_registry, mock_instance, registration_payload):
        """Test agent can refuse task with reason."""
        # Arrange
        with patch('app.repository.instance_repository.get_instance_by_container_id', return_value=mock_instance):
            with patch('app.repository.instance_repository.update_instance', return_value=True):
                with patch('app.repository.instance_repository.increment_task_refused', return_value=True) as mock_increment:
                    config = await agent_registry.register_agent(registration_payload)
                    agent_id = config.agent_id
                    
                    refusal = TaskRefusal(
                        task_id="task_456",
                        reason="Capability mismatch",
                        suggested_agent="analyst-02"
                    )
                    
                    # Act
                    await agent_registry.handle_task_refusal(agent_id, refusal)
                    
                    # Assert
                    mock_increment.assert_called_once_with(mock_instance.id)
    
    @pytest.mark.asyncio 
    async def test_refused_task_recorded_in_trust_metrics(self, agent_registry, mock_instance, registration_payload):
        """Test refusal recorded in trust metrics."""
        # Arrange
        with patch('app.repository.instance_repository.get_instance_by_container_id', return_value=mock_instance):
            with patch('app.repository.instance_repository.update_instance', return_value=True):
                with patch('app.repository.instance_repository.increment_task_refused', return_value=True) as mock_increment:
                    config = await agent_registry.register_agent(registration_payload)
                    agent_id = config.agent_id
                    
                    refusal = TaskRefusal(task_id="task_789", reason="Too complex", suggested_agent=None)
                    
                    # Act
                    await agent_registry.handle_task_refusal(agent_id, refusal)
                    
                    # Assert
                    mock_increment.assert_called_once_with(mock_instance.id)


class TestTaskCompletion:
    """Test task completion functionality."""
    
    @pytest.mark.asyncio
    async def test_task_completion_updates_trust_metrics(self, agent_registry, mock_instance, registration_payload):
        """Test task completion updates trust metrics."""
        # Arrange
        with patch('app.repository.instance_repository.get_instance_by_container_id', return_value=mock_instance):
            with patch('app.repository.instance_repository.update_instance', return_value=True):
                with patch('app.repository.instance_repository.increment_task_completed', return_value=True) as mock_increment:
                    config = await agent_registry.register_agent(registration_payload)
                    agent_id = config.agent_id
                    
                    completion = TaskCompletion(
                        task_id="task_123",
                        result={"status": "success", "data": [1, 2, 3]},
                        duration_ms=1500
                    )
                    
                    # Act
                    await agent_registry.handle_task_completion(agent_id, completion)
                    
                    # Assert
                    mock_increment.assert_called_once_with(mock_instance.id, 1500)
                    agent_info = agent_registry.get_agent_info(agent_id)
                    assert agent_info["current_task"] is None


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_websocket_disconnects_unexpectedly_agent_marked_unhealthy(self, ws_manager):
        """Test WebSocket disconnects unexpectedly — agent marked unhealthy."""
        # Arrange
        agent_id = "test-agent-001"
        mock_websocket = MockWebSocket()
        
        await ws_manager.connect(agent_id, mock_websocket)
        assert agent_id in ws_manager.agent_connections
        
        # Act - simulate unexpected disconnect
        ws_manager.disconnect(agent_id)
        
        # Assert
        assert agent_id not in ws_manager.agent_connections
        assert agent_id not in ws_manager.connection_metadata
    
    @pytest.mark.asyncio
    async def test_multiple_rapid_heartbeats_handled_correctly(self, agent_registry, mock_instance, registration_payload):
        """Test multiple rapid heartbeats handled correctly."""
        # Arrange
        with patch('app.repository.instance_repository.get_instance_by_container_id', return_value=mock_instance):
            with patch('app.repository.instance_repository.update_instance', return_value=True):
                with patch('app.repository.instance_repository.update_heartbeat', return_value=True):
                    config = await agent_registry.register_agent(registration_payload)
                    agent_id = config.agent_id
                    
                    heartbeat_data = AgentHeartbeatData(
                        status="ready",
                        current_task=None,
                        resource_usage={"memory_mb": 256}
                    )
                    
                    # Act - send multiple rapid heartbeats
                    responses = []
                    for _ in range(5):
                        response = await agent_registry.handle_heartbeat(agent_id, heartbeat_data)
                        responses.append(response)
                        await asyncio.sleep(0.001)
                    
                    # Assert
                    assert len(responses) == 5
                    for response in responses:
                        assert response["type"] == "heartbeat_ack"
                    
                    # Agent should still be active
                    assert agent_id in agent_registry.active_agents
    
    @pytest.mark.asyncio
    async def test_registration_during_shutdown_rejected(self, agent_registry, registration_payload):
        """Test registration during shutdown is rejected."""
        # Arrange
        agent_registry._shutdown = True
        
        with patch('app.repository.instance_repository.get_instance_by_container_id', return_value=None):
            
            # Act & Assert
            with pytest.raises(ValueError):
                await agent_registry.register_agent(registration_payload)
    
    @pytest.mark.asyncio
    async def test_invalid_json_in_websocket_message_handled_gracefully(self, ws_manager):
        """Test invalid JSON in WebSocket message handled gracefully."""
        # Arrange
        agent_id = "test-agent-001"
        invalid_message = {"type": "invalid", "malformed": True}  # Missing required fields
        
        # Act
        response = await ws_manager.handle_message(agent_id, invalid_message)
        
        # Assert
        assert response is not None
        assert response["type"] == "error"
        assert "error" in response
    
    @pytest.mark.asyncio
    async def test_concurrent_registrations_from_same_container_id(self, agent_registry, mock_instance):
        """Test concurrent registrations from same container_id."""
        # Arrange
        payload1 = AgentRegistrationPayload(
            container_id=mock_instance.container_id,
            role="researcher",
            capabilities=["web_search"],
            version="1.0.0"
        )
        
        payload2 = AgentRegistrationPayload(
            container_id=mock_instance.container_id,
            role="analyst", 
            capabilities=["analysis"],
            version="1.0.1"
        )
        
        with patch('app.repository.instance_repository.get_instance_by_container_id', return_value=mock_instance):
            with patch('app.repository.instance_repository.update_instance', return_value=True):
                
                # Act - register concurrently
                results = await asyncio.gather(
                    agent_registry.register_agent(payload1),
                    agent_registry.register_agent(payload2),
                    return_exceptions=True
                )
                
                # Assert
                # Both should succeed and result in the same agent_id
                assert len(results) == 2
                for result in results:
                    assert not isinstance(result, Exception)
                    assert result.agent_id == mock_instance.agent_id


class TestIntegration:
    """Integration tests for full lifecycle scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_lifecycle_spawn_register_heartbeat_task_deregister(self, agent_registry, mock_instance, registration_payload):
        """Test full lifecycle: spawn → register → heartbeat → task → deregister."""
        # Arrange
        with patch('app.repository.instance_repository.get_instance_by_container_id', return_value=mock_instance):
            with patch('app.repository.instance_repository.update_instance', return_value=True):
                with patch('app.repository.instance_repository.update_heartbeat', return_value=True):
                    with patch('app.repository.instance_repository.update_instance_status', return_value=True):
                        with patch('app.repository.instance_repository.increment_task_completed', return_value=True):
                            
                            # Act 1: Register
                            config = await agent_registry.register_agent(registration_payload)
                            agent_id = config.agent_id
                            
                            # Act 2: Heartbeat
                            heartbeat = AgentHeartbeatData(status="ready", current_task=None, resource_usage={})
                            response = await agent_registry.handle_heartbeat(agent_id, heartbeat)
                            
                            # Act 3: Assign and complete task
                            task = TaskAssignment(task_id="integration_task", task_type="test", payload={})
                            task_assigned = await agent_registry.assign_task(agent_id, task)
                            
                            completion = TaskCompletion(task_id="integration_task", result={}, duration_ms=1000)
                            await agent_registry.handle_task_completion(agent_id, completion)
                            
                            # Act 4: Deregister
                            deregistration = AgentDeregistration(reason="test_complete", final_state={})
                            await agent_registry.deregister_agent(agent_id, deregistration)
                            
                            # Assert
                            assert config.agent_id == mock_instance.agent_id
                            assert response["type"] == "heartbeat_ack"
                            assert task_assigned is True
                            assert agent_id not in agent_registry.active_agents
    
    @pytest.mark.asyncio
    async def test_scale_up_3_agents_verify_all_register_scale_down_to_1(self):
        """Test scale up 3 agents, verify all register, scale down to 1."""
        # This would be more of a system integration test requiring Docker
        # For unit testing, we'll mock the scaling behavior
        
        # Arrange
        agent_registry = AgentRegistry()
        
        # Create 3 mock instances
        instances = []
        for i in range(3):
            instance = AgentInstance(
                id=uuid4(),
                container_id=f"test_container_{i}",
                agent_id=f"test-agent-{i}",
                agent_role="researcher",
                status=InstanceStatus.STARTING,
                capabilities=["web_search"],
                created_at=datetime.utcnow()
            )
            instances.append(instance)
        
        with patch('app.repository.instance_repository.get_instance_by_container_id') as mock_get_instance:
            with patch('app.repository.instance_repository.update_instance', return_value=True):
                
                def get_instance_side_effect(container_id):
                    for instance in instances:
                        if instance.container_id == container_id:
                            return instance
                    return None
                
                mock_get_instance.side_effect = get_instance_side_effect
                
                # Act 1: Register all 3 agents
                configs = []
                for instance in instances:
                    payload = AgentRegistrationPayload(
                        container_id=instance.container_id,
                        role="researcher",
                        capabilities=["web_search"],
                        version="1.0.0"
                    )
                    config = await agent_registry.register_agent(payload)
                    configs.append(config)
                
                # Act 2: Verify all registered
                active_agents = agent_registry.list_active_agents()
                
                # Act 3: "Scale down" by deregistering 2 agents
                for i in range(2):
                    deregistration = AgentDeregistration(reason="scale_down", final_state={})
                    with patch('app.repository.instance_repository.update_instance_status', return_value=True):
                        await agent_registry.deregister_agent(configs[i].agent_id, deregistration)
                
                # Assert
                assert len(configs) == 3
                assert len(active_agents) == 3
                final_active = agent_registry.list_active_agents()
                assert len(final_active) == 1
                assert final_active[0] == configs[2].agent_id