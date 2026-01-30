"""
Unit tests for the sandbox module.

Tests security configurations, container management, and state management.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os

# Add the backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.sandbox.security import (
    TrustLevel,
    AgentSecurityConfig,
    TRUST_PRESETS,
    get_security_config,
    validate_security_config
)
from app.sandbox.container_manager import (
    ContainerManager,
    ContainerConfig,
    ContainerStatus,
    ContainerInfo,
    ExecutionResult
)
from app.sandbox.state_manager import (
    StateManager,
    StateScope,
    AgentState,
    EventType,
    StateEvent
)


# ============================================================================
# Security Configuration Tests
# ============================================================================

class TestSecurityConfig:
    """Tests for security configuration."""
    
    def test_trust_levels_exist(self):
        """Test that all trust levels are defined."""
        assert TrustLevel.TRUSTED in TRUST_PRESETS
        assert TrustLevel.SANDBOXED in TRUST_PRESETS
        assert TrustLevel.UNTRUSTED in TRUST_PRESETS
    
    def test_trusted_config_has_full_access(self):
        """Test that TRUSTED level has full access."""
        config = TRUST_PRESETS[TrustLevel.TRUSTED]
        
        assert config.network_enabled == True
        assert config.allow_shell == True
        assert config.allow_subprocess == True
        assert config.memory_limit_mb >= 1024  # At least 1GB
    
    def test_sandboxed_config_has_restrictions(self):
        """Test that SANDBOXED level has appropriate restrictions."""
        config = TRUST_PRESETS[TrustLevel.SANDBOXED]
        
        assert config.network_enabled == True  # Network allowed but limited
        assert config.allowed_hosts is not None  # Hosts should be allowlisted
        assert config.allow_shell == False
        assert config.log_all_commands == True
    
    def test_untrusted_config_is_isolated(self):
        """Test that UNTRUSTED level is fully isolated."""
        config = TRUST_PRESETS[TrustLevel.UNTRUSTED]
        
        assert config.network_enabled == False
        assert config.allow_shell == False
        assert config.allow_network_tools == False
        assert config.allow_subprocess == False
        assert config.read_only_root == True
        assert config.memory_limit_mb <= 256
    
    def test_get_security_config_returns_copy(self):
        """Test that get_security_config returns a copy, not the original."""
        config1 = get_security_config(TrustLevel.TRUSTED)
        config2 = get_security_config(TrustLevel.TRUSTED)
        
        # Modify config1
        config1.memory_limit_mb = 9999
        
        # config2 should not be affected
        assert config2.memory_limit_mb != 9999
    
    def test_get_security_config_with_overrides(self):
        """Test that overrides are applied correctly."""
        config = get_security_config(
            TrustLevel.TRUSTED,
            overrides={"memory_limit_mb": 4096, "timeout_seconds": 60}
        )
        
        assert config.memory_limit_mb == 4096
        assert config.timeout_seconds == 60
        # Other values should be from preset
        assert config.network_enabled == True
    
    def test_validate_security_config_untrusted_with_network(self):
        """Test validation catches UNTRUSTED with network enabled."""
        config = AgentSecurityConfig(
            trust_level=TrustLevel.UNTRUSTED,
            network_enabled=True  # This is bad
        )
        
        warnings = validate_security_config(config)
        assert len(warnings) > 0
        assert any("network" in w.lower() for w in warnings)
    
    def test_validate_security_config_sandboxed_shell_no_logging(self):
        """Test validation warns about shell without logging."""
        config = AgentSecurityConfig(
            trust_level=TrustLevel.SANDBOXED,
            allow_shell=True,
            log_all_commands=False  # Should warn
        )
        
        warnings = validate_security_config(config)
        assert len(warnings) > 0
        assert any("log" in w.lower() for w in warnings)
    
    def test_validate_security_config_low_memory(self):
        """Test validation warns about very low memory."""
        config = AgentSecurityConfig(
            memory_limit_mb=32  # Too low
        )
        
        warnings = validate_security_config(config)
        assert any("memory" in w.lower() for w in warnings)


# ============================================================================
# Container Manager Tests
# ============================================================================

class TestContainerManager:
    """Tests for container management."""
    
    @pytest.fixture
    def manager(self):
        """Create a container manager instance."""
        return ContainerManager(
            pool_size=2,
            container_ttl_minutes=5
        )
    
    def test_manager_initialization(self, manager):
        """Test manager initializes correctly."""
        assert manager.pool_size == 2
        assert manager.container_ttl == timedelta(minutes=5)
        assert len(manager._pools) == len(TrustLevel)
    
    def test_build_docker_config_trusted(self, manager):
        """Test Docker config generation for TRUSTED level."""
        config = ContainerConfig(
            name="test-container",
            security=TRUST_PRESETS[TrustLevel.TRUSTED]
        )
        
        docker_config = manager._build_docker_config(config)
        
        assert docker_config["name"] == "test-container"
        assert "mem_limit" in docker_config
        assert docker_config.get("network_mode") != "none"
        assert docker_config.get("read_only") != True
    
    def test_build_docker_config_untrusted(self, manager):
        """Test Docker config generation for UNTRUSTED level."""
        config = ContainerConfig(
            name="test-untrusted",
            security=TRUST_PRESETS[TrustLevel.UNTRUSTED]
        )
        
        docker_config = manager._build_docker_config(config)
        
        assert docker_config["network_mode"] == "none"
        assert docker_config["read_only"] == True
        assert "security_opt" in docker_config
        # Check memory limit is restrictive
        mem_limit = docker_config["mem_limit"]
        assert "256m" in mem_limit or int(mem_limit.rstrip('m')) <= 256
    
    def test_build_docker_config_sandboxed(self, manager):
        """Test Docker config generation for SANDBOXED level."""
        config = ContainerConfig(
            name="test-sandboxed",
            security=TRUST_PRESETS[TrustLevel.SANDBOXED]
        )
        
        docker_config = manager._build_docker_config(config)
        
        # Network should be bridge (for filtering) not none
        assert docker_config.get("network_mode") != "none"
        assert docker_config["read_only"] == True
        # Should have tmpfs for writable paths
        assert "tmpfs" in docker_config
    
    def test_get_status(self, manager):
        """Test status reporting."""
        status = manager.get_status()
        
        assert "pools" in status
        assert "active_containers" in status
        assert "pool_size_limit" in status
        assert status["pool_size_limit"] == 2


# ============================================================================
# State Manager Tests
# ============================================================================

class TestStateManager:
    """Tests for state management."""
    
    @pytest.fixture
    def state_manager(self, tmp_path):
        """Create a state manager instance with temp directory."""
        return StateManager(
            data_dir=str(tmp_path / "state_data"),
            state_ttl_minutes=5
        )
    
    @pytest.mark.asyncio
    async def test_set_and_get_state(self, state_manager):
        """Test setting and getting state."""
        await state_manager.initialize()
        
        # Set state
        state = await state_manager.set_state(
            agent_id="test-agent",
            task_id="test-task",
            data={"key": "value", "count": 42}
        )
        
        assert state.agent_id == "test-agent"
        assert state.task_id == "test-task"
        assert state.data["key"] == "value"
        
        # Get state
        retrieved = await state_manager.get_state(
            agent_id="test-agent",
            task_id="test-task"
        )
        
        assert retrieved is not None
        assert retrieved.data["key"] == "value"
        assert retrieved.data["count"] == 42
    
    @pytest.mark.asyncio
    async def test_state_scopes(self, state_manager):
        """Test different state scopes."""
        await state_manager.initialize()
        
        # Set states with different scopes
        await state_manager.set_state(
            agent_id="agent1",
            task_id="task1",
            scope=StateScope.TASK,
            data={"scope": "task"}
        )
        
        await state_manager.set_state(
            agent_id="agent1",
            task_id="task1",
            scope=StateScope.AGENT,
            data={"scope": "agent"}
        )
        
        # Retrieve by scope
        task_state = await state_manager.get_state("agent1", "task1", StateScope.TASK)
        agent_state = await state_manager.get_state("agent1", "task1", StateScope.AGENT)
        
        assert task_state.data["scope"] == "task"
        assert agent_state.data["scope"] == "agent"
    
    @pytest.mark.asyncio
    async def test_state_update(self, state_manager):
        """Test updating existing state."""
        await state_manager.initialize()
        
        # Set initial state
        await state_manager.set_state(
            agent_id="agent1",
            task_id="task1",
            data={"step": 1, "status": "running"}
        )
        
        # Update state
        await state_manager.set_state(
            agent_id="agent1",
            task_id="task1",
            data={"step": 2}  # Only update step
        )
        
        # Get state
        state = await state_manager.get_state("agent1", "task1")
        
        assert state.data["step"] == 2
        assert state.data["status"] == "running"  # Should be preserved
    
    @pytest.mark.asyncio
    async def test_state_deletion(self, state_manager):
        """Test deleting state."""
        await state_manager.initialize()
        
        # Set state
        await state_manager.set_state(
            agent_id="agent1",
            task_id="task1",
            data={"key": "value"}
        )
        
        # Delete state
        result = await state_manager.delete_state("agent1", "task1")
        assert result == True
        
        # Verify deletion
        state = await state_manager.get_state("agent1", "task1")
        assert state is None
    
    @pytest.mark.asyncio
    async def test_event_subscription(self, state_manager):
        """Test event subscription and publishing."""
        await state_manager.initialize()
        
        # Subscribe to events
        queue = await state_manager.subscribe_events("agent1")
        
        # Publish event
        await state_manager.publish_event(
            event_type=EventType.TASK_STARTED,
            source_agent="agent2",
            target_agent="agent1",
            task_id="task1",
            payload={"message": "Hello"}
        )
        
        # Check event was received
        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        
        assert event.event_type == EventType.TASK_STARTED
        assert event.source_agent == "agent2"
        assert event.payload["message"] == "Hello"
    
    @pytest.mark.asyncio
    async def test_artifact_save_and_get(self, state_manager):
        """Test saving and retrieving artifacts."""
        await state_manager.initialize()
        
        # Save artifact
        content = b"Hello, this is test content!"
        artifact_id = await state_manager.save_artifact(
            agent_id="agent1",
            task_id="task1",
            name="test.txt",
            content=content,
            content_type="text/plain",
            metadata={"description": "Test file"}
        )
        
        assert artifact_id is not None
        
        # Get artifact
        result = await state_manager.get_artifact(artifact_id)
        
        assert result is not None
        retrieved_content, content_type, metadata = result
        
        assert retrieved_content == content
        assert content_type == "text/plain"
        assert metadata["description"] == "Test file"
    
    @pytest.mark.asyncio
    async def test_list_artifacts(self, state_manager):
        """Test listing artifacts for a task."""
        await state_manager.initialize()
        
        # Save multiple artifacts
        await state_manager.save_artifact(
            agent_id="agent1",
            task_id="task1",
            name="file1.txt",
            content=b"Content 1"
        )
        
        await state_manager.save_artifact(
            agent_id="agent1",
            task_id="task1",
            name="file2.txt",
            content=b"Content 2"
        )
        
        # List artifacts
        artifacts = await state_manager.list_artifacts("task1")
        
        assert len(artifacts) == 2
        names = [a["name"] for a in artifacts]
        assert "file1.txt" in names
        assert "file2.txt" in names
    
    def test_get_status(self, state_manager):
        """Test status reporting."""
        status = state_manager.get_status()
        
        assert "state_entries" in status
        assert "subscribed_agents" in status
        assert "data_dir" in status


# ============================================================================
# Integration Tests
# ============================================================================

class TestSandboxIntegration:
    """Integration tests for sandbox components."""
    
    @pytest.mark.asyncio
    async def test_security_config_applied_to_container(self):
        """Test that security config is correctly applied to containers."""
        manager = ContainerManager(pool_size=1)
        
        # Create config with specific security settings
        security = AgentSecurityConfig(
            trust_level=TrustLevel.SANDBOXED,
            memory_limit_mb=256,
            timeout_seconds=60,
            network_enabled=True,
            allowed_hosts=["api.example.com"]
        )
        
        config = ContainerConfig(
            name="test-integration",
            security=security
        )
        
        docker_config = manager._build_docker_config(config)
        
        # Verify security settings are applied
        assert docker_config["mem_limit"] == "256m"
        assert docker_config["cpu_quota"] == 100000  # 1.0 CPU


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
