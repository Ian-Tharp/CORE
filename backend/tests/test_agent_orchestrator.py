"""
Integration tests for the Agent Orchestrator.

Tests the full CORE loop execution with mocked dependencies.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os

# Add the backend and mcp to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.sandbox import (
    ContainerManager,
    ContainerConfig,
    TrustLevel,
    TRUST_PRESETS,
    StateManager,
    StateScope,
)
from mcp.dynamic_tool_selector import (
    DynamicToolSelector,
    ToolSelection,
    ToolDefinition,
    Capability
)


# ============================================================================
# Dynamic Tool Selector Tests
# ============================================================================

class TestDynamicToolSelector:
    """Tests for the dynamic tool selector."""
    
    @pytest.fixture
    def selector(self):
        """Create a tool selector instance."""
        return DynamicToolSelector(
            registry_url="http://localhost:8000",
            cache_ttl_minutes=5
        )
    
    def test_infer_capabilities_file_operations(self, selector):
        """Test capability inference for file operations."""
        task = "Read the config file and find all settings"
        capabilities = selector._infer_capabilities(task)
        
        assert Capability.FILE_READ.value in capabilities
        assert Capability.FILE_SEARCH.value in capabilities
    
    def test_infer_capabilities_web_operations(self, selector):
        """Test capability inference for web operations."""
        task = "Search the web for Python documentation and fetch the results"
        capabilities = selector._infer_capabilities(task)
        
        assert Capability.WEB_SEARCH.value in capabilities
        assert Capability.WEB_FETCH.value in capabilities
    
    def test_infer_capabilities_code_execution(self, selector):
        """Test capability inference for code execution."""
        task = "Execute this Python script to analyze the data"
        capabilities = selector._infer_capabilities(task)
        
        assert Capability.CODE_PYTHON.value in capabilities
    
    def test_prioritize_capabilities(self, selector):
        """Test capability prioritization based on task context."""
        task = "First search the web, then save the results to a file"
        capabilities = [
            Capability.WEB_SEARCH.value,
            Capability.FILE_WRITE.value
        ]
        
        prioritized = selector._prioritize_capabilities(task, capabilities)
        
        # "search" appears before "save" in the task
        assert prioritized[0] == Capability.WEB_SEARCH.value
        assert prioritized[1] == Capability.FILE_WRITE.value
    
    @pytest.mark.asyncio
    async def test_analyze_task(self, selector):
        """Test task analysis."""
        analysis = await selector.analyze_task(
            task_description="Read the log file and search for errors",
            explicit_capabilities=[Capability.DATA_ANALYZE.value]
        )
        
        assert analysis.task_description == "Read the log file and search for errors"
        assert Capability.DATA_ANALYZE.value in analysis.explicit_capabilities
        assert len(analysis.inferred_capabilities) > 0
        assert len(analysis.priority_order) > 0
    
    def test_check_constraints_max_latency(self, selector):
        """Test constraint checking for max latency."""
        tool = ToolDefinition(
            name="slow_tool",
            description="A slow tool",
            server_id="server1",
            server_name="Test Server",
            estimated_latency_ms=5000
        )
        
        # Should fail constraint
        result = selector._check_constraints(tool, {"max_latency_ms": 1000})
        assert result == False
        
        # Should pass constraint
        result = selector._check_constraints(tool, {"max_latency_ms": 10000})
        assert result == True
    
    def test_check_constraints_blocked_tools(self, selector):
        """Test constraint checking for blocked tools."""
        tool = ToolDefinition(
            name="dangerous_tool",
            description="A dangerous tool",
            server_id="server1",
            server_name="Test Server"
        )
        
        # Should fail constraint
        result = selector._check_constraints(
            tool, 
            {"blocked_tools": ["dangerous_tool", "other_tool"]}
        )
        assert result == False
        
        # Should pass constraint
        result = selector._check_constraints(
            tool,
            {"blocked_tools": ["other_tool"]}
        )
        assert result == True
    
    def test_generate_selection_reason_no_tools(self, selector):
        """Test selection reason when no tools selected."""
        reason = selector._generate_selection_reason(
            selected_tools=[],
            matched_caps=set(),
            missing_caps=["web:search", "file:read"]
        )
        
        assert "No tools found" in reason
        assert "web:search" in reason
    
    def test_generate_selection_reason_with_tools(self, selector):
        """Test selection reason with selected tools."""
        tools = [
            ToolDefinition(
                name="tool1",
                description="Tool 1",
                server_id="s1",
                server_name="Server 1"
            ),
            ToolDefinition(
                name="tool2",
                description="Tool 2",
                server_id="s1",
                server_name="Server 1"
            )
        ]
        
        reason = selector._generate_selection_reason(
            selected_tools=tools,
            matched_caps={"web:search"},
            missing_caps=[]
        )
        
        assert "2 tools" in reason
        assert "tool1" in reason
    
    def test_get_status(self, selector):
        """Test status reporting."""
        status = selector.get_status()
        
        assert "cached_tools" in status
        assert "indexed_capabilities" in status
        assert "cache_age_seconds" in status


# ============================================================================
# Mock-based Orchestrator Tests
# ============================================================================

class TestAgentOrchestratorWithMocks:
    """Tests for agent orchestrator using mocks."""
    
    @pytest.fixture
    def mock_container_manager(self):
        """Create a mock container manager."""
        manager = Mock(spec=ContainerManager)
        manager.initialize = AsyncMock()
        manager.shutdown = AsyncMock()
        manager.get_container = AsyncMock(return_value=Mock(
            id="container-123",
            name="test-container",
            status="running"
        ))
        manager.release_container = AsyncMock()
        manager.get_status = Mock(return_value={
            "pools": {},
            "active_containers": 0
        })
        return manager
    
    @pytest.fixture
    def mock_state_manager(self, tmp_path):
        """Create a mock state manager."""
        manager = Mock(spec=StateManager)
        manager.initialize = AsyncMock()
        manager.set_state = AsyncMock(return_value=Mock(
            id="state-123",
            data={}
        ))
        manager.get_state = AsyncMock(return_value=None)
        manager.get_status = Mock(return_value={
            "state_entries": 0,
            "subscribed_agents": 0
        })
        return manager
    
    @pytest.fixture
    def mock_tool_selector(self):
        """Create a mock tool selector."""
        selector = Mock(spec=DynamicToolSelector)
        selector.refresh_tool_cache = AsyncMock()
        selector.select_tools_for_task = AsyncMock(return_value=ToolSelection(
            selected_tools=[
                ToolDefinition(
                    name="web_search",
                    description="Search the web",
                    server_id="server1",
                    server_name="Search Server",
                    capabilities=["web:search"]
                )
            ],
            required_capabilities=["web:search"],
            missing_capabilities=[],
            selection_reason="Selected 1 tool matching requirements",
            confidence=0.95
        ))
        selector.get_status = Mock(return_value={
            "cached_tools": 10,
            "indexed_capabilities": 5
        })
        return selector
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(
        self,
        mock_container_manager,
        mock_state_manager,
        mock_tool_selector
    ):
        """Test orchestrator initializes correctly."""
        # Import here to avoid import issues in the fixture
        from app.core.agent_orchestrator import AgentOrchestrator, AgentRole
        
        orchestrator = AgentOrchestrator(
            container_manager=mock_container_manager,
            state_manager=mock_state_manager,
            tool_selector=mock_tool_selector
        )
        
        await orchestrator.initialize()
        
        mock_container_manager.initialize.assert_called_once()
        mock_state_manager.initialize.assert_called_once()
        mock_tool_selector.refresh_tool_cache.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_orchestrator_shutdown(
        self,
        mock_container_manager,
        mock_state_manager,
        mock_tool_selector
    ):
        """Test orchestrator shuts down correctly."""
        from app.core.agent_orchestrator import AgentOrchestrator
        
        orchestrator = AgentOrchestrator(
            container_manager=mock_container_manager,
            state_manager=mock_state_manager,
            tool_selector=mock_tool_selector
        )
        
        await orchestrator.shutdown()
        
        mock_container_manager.shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_submit_task(
        self,
        mock_container_manager,
        mock_state_manager,
        mock_tool_selector
    ):
        """Test submitting a task."""
        from app.core.agent_orchestrator import AgentOrchestrator, AgentRole, TaskStatus
        
        orchestrator = AgentOrchestrator(
            container_manager=mock_container_manager,
            state_manager=mock_state_manager,
            tool_selector=mock_tool_selector
        )
        
        task = await orchestrator.submit_task(
            role=AgentRole.COMPREHENSION,
            description="Test task",
            input_data={"user_input": "Hello"}
        )
        
        assert task.role == AgentRole.COMPREHENSION
        assert task.description == "Test task"
        assert task.status == TaskStatus.PENDING
        assert task.input_data["user_input"] == "Hello"
    
    @pytest.mark.asyncio
    async def test_execute_task(
        self,
        mock_container_manager,
        mock_state_manager,
        mock_tool_selector
    ):
        """Test executing a task."""
        from app.core.agent_orchestrator import AgentOrchestrator, AgentRole, TaskStatus
        
        orchestrator = AgentOrchestrator(
            container_manager=mock_container_manager,
            state_manager=mock_state_manager,
            tool_selector=mock_tool_selector
        )
        
        # Submit task
        task = await orchestrator.submit_task(
            role=AgentRole.COMPREHENSION,
            description="Understand user input",
            input_data={"user_input": "What is the weather?"}
        )
        
        # Execute task
        result = await orchestrator.execute_task(task.id)
        
        assert result.success == True
        assert result.task_id == task.id
        assert "intent" in result.output
        
        # Verify container was used and released
        mock_container_manager.get_container.assert_called_once()
        mock_container_manager.release_container.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_task_not_found(
        self,
        mock_container_manager,
        mock_state_manager,
        mock_tool_selector
    ):
        """Test executing a non-existent task."""
        from app.core.agent_orchestrator import AgentOrchestrator
        
        orchestrator = AgentOrchestrator(
            container_manager=mock_container_manager,
            state_manager=mock_state_manager,
            tool_selector=mock_tool_selector
        )
        
        result = await orchestrator.execute_task("nonexistent-task-id")
        
        assert result.success == False
        assert "not found" in result.error.lower()
    
    def test_get_status(
        self,
        mock_container_manager,
        mock_state_manager,
        mock_tool_selector
    ):
        """Test getting orchestrator status."""
        from app.core.agent_orchestrator import AgentOrchestrator
        
        orchestrator = AgentOrchestrator(
            container_manager=mock_container_manager,
            state_manager=mock_state_manager,
            tool_selector=mock_tool_selector
        )
        
        status = orchestrator.get_status()
        
        assert "active_tasks" in status
        assert "total_tasks" in status
        assert "container_manager" in status
        assert "state_manager" in status
        assert "tool_selector" in status


# ============================================================================
# CORE Loop Tests
# ============================================================================

class TestCORELoop:
    """Tests for the full CORE loop execution."""
    
    @pytest.fixture
    def setup_orchestrator(self, tmp_path):
        """Setup orchestrator with mocked dependencies."""
        container_manager = Mock(spec=ContainerManager)
        container_manager.initialize = AsyncMock()
        container_manager.shutdown = AsyncMock()
        container_manager.get_container = AsyncMock(return_value=Mock(
            id="container-123",
            name="test-container",
            status="running"
        ))
        container_manager.release_container = AsyncMock()
        container_manager.get_status = Mock(return_value={})
        
        state_manager = Mock(spec=StateManager)
        state_manager.initialize = AsyncMock()
        state_manager.set_state = AsyncMock(return_value=Mock(data={}))
        state_manager.get_state = AsyncMock(return_value=None)
        state_manager.get_status = Mock(return_value={})
        
        tool_selector = Mock(spec=DynamicToolSelector)
        tool_selector.refresh_tool_cache = AsyncMock()
        tool_selector.select_tools_for_task = AsyncMock(return_value=ToolSelection(
            selected_tools=[],
            required_capabilities=[],
            missing_capabilities=[],
            selection_reason="No tools required",
            confidence=1.0
        ))
        tool_selector.get_status = Mock(return_value={})
        
        from app.core.agent_orchestrator import AgentOrchestrator
        
        return AgentOrchestrator(
            container_manager=container_manager,
            state_manager=state_manager,
            tool_selector=tool_selector
        )
    
    @pytest.mark.asyncio
    async def test_core_loop_simple_query(self, setup_orchestrator):
        """Test CORE loop for a simple query (no tool execution)."""
        orchestrator = setup_orchestrator
        
        # Note: Current implementation doesn't have a real LLM,
        # so we're testing the flow structure
        result = await orchestrator.execute_core_loop(
            user_input="Hello, how are you?",
            session_id="test-session"
        )
        
        # Should complete successfully
        assert result["success"] == True
        # Should have taken the complex path (current impl always does)
        assert result.get("iterations", 0) >= 1
    
    @pytest.mark.asyncio
    async def test_core_loop_complex_task(self, setup_orchestrator):
        """Test CORE loop for a complex task requiring tools."""
        orchestrator = setup_orchestrator
        
        result = await orchestrator.execute_core_loop(
            user_input="Search for Python documentation and summarize it",
            session_id="test-session",
            max_iterations=3
        )
        
        assert result["success"] == True
        assert "response" in result
        assert "phase" in result
        
        # Should have metadata about the execution
        if "metadata" in result:
            assert isinstance(result["metadata"], dict)
    
    @pytest.mark.asyncio
    async def test_core_loop_max_iterations(self, setup_orchestrator):
        """Test that CORE loop respects max iterations."""
        orchestrator = setup_orchestrator
        
        result = await orchestrator.execute_core_loop(
            user_input="Complex task",
            session_id="test-session",
            max_iterations=1
        )
        
        # Should complete within 1 iteration
        assert result["iterations"] <= 1


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
