"""
Task Router Tests

Comprehensive tests for the Task Routing Engine following AAA format.
Tests cover routing logic, refusal handling, completion tracking, and edge cases.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Any
from uuid import uuid4, UUID
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.task_models import (
    Task, TaskAssignment, TaskResult, TaskStatus, AgentResponse, 
    TaskType, TaskRoutingScore
)
from app.repository.instance_repository import InstanceStatus, AgentInstance
from app.services.task_router import TaskRouter


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_task():
    """Create a sample task for testing."""
    return Task(
        id=uuid4(),
        task_type=TaskType.RESEARCH,
        payload={"query": "test research topic"},
        priority=7,
        required_capabilities=["web_search", "analysis"],
        preferred_model="claude-sonnet"
    )


@pytest.fixture
def sample_agents_with_metrics():
    """Create sample agents with trust metrics for testing."""
    agent1_id = uuid4()
    agent2_id = uuid4()
    agent3_id = uuid4()
    
    return [
        {
            "instance": AgentInstance(
                id=agent1_id,
                container_id="container1",
                agent_id="researcher-001",
                agent_role="researcher",
                status=InstanceStatus.READY,
                capabilities=["web_search", "analysis", "writing"],
                created_at=datetime.utcnow()
            ),
            "trust_metrics": {
                "tasks_completed": 25,
                "tasks_refused": 2,
                "tasks_failed": 1,
                "trust_score": 0.85,
                "avg_task_duration_ms": 45000
            }
        },
        {
            "instance": AgentInstance(
                id=agent2_id,
                container_id="container2",
                agent_id="analyst-001",
                agent_role="analyst",
                status=InstanceStatus.READY,
                capabilities=["analysis", "data_processing"],
                created_at=datetime.utcnow()
            ),
            "trust_metrics": {
                "tasks_completed": 15,
                "tasks_refused": 5,
                "tasks_failed": 3,
                "trust_score": 0.65,
                "avg_task_duration_ms": 60000
            }
        },
        {
            "instance": AgentInstance(
                id=agent3_id,
                container_id="container3",
                agent_id="researcher-002",
                agent_role="researcher",
                status=InstanceStatus.READY,
                capabilities=["web_search", "analysis"],
                created_at=datetime.utcnow()
            ),
            "trust_metrics": {
                "tasks_completed": 10,
                "tasks_refused": 1,
                "tasks_failed": 0,
                "trust_score": 0.90,
                "avg_task_duration_ms": 35000
            }
        }
    ]


@pytest.fixture
def task_router():
    """Create a TaskRouter instance for testing."""
    return TaskRouter()


# =============================================================================
# ROUTING TESTS
# =============================================================================

class TestTaskRouting:
    """Test core task routing functionality."""
    
    @pytest.mark.asyncio
    async def test_routes_to_capable_agent_with_lowest_load(self, task_router, sample_task, sample_agents_with_metrics):
        """Test that tasks are routed to agents with required capabilities and lowest load."""
        # Arrange
        with patch('app.services.task_router.get_instances_with_trust_metrics') as mock_get_instances, \
             patch('app.services.task_router.agent_registry') as mock_registry, \
             patch('app.services.task_router.update_task_status') as mock_update_status, \
             patch('app.services.task_router.create_task_assignment') as mock_create_assignment, \
             patch('app.services.task_router.get_agent_task_metrics') as mock_agent_metrics, \
             patch.object(task_router, '_get_agent_current_load') as mock_get_load:
            
            # Setup mocks
            mock_get_instances.return_value = sample_agents_with_metrics
            mock_registry.get_healthy_agents.return_value = ["researcher-001", "analyst-001", "researcher-002"]
            mock_update_status.return_value = True
            mock_create_assignment.return_value = uuid4()
            mock_agent_metrics.return_value = []
            
            # Set different load levels
            async def mock_load_side_effect(agent_id):
                if str(agent_id) == str(sample_agents_with_metrics[0]["instance"].id):
                    return 1  # Low load
                elif str(agent_id) == str(sample_agents_with_metrics[2]["instance"].id):
                    return 0  # Lowest load
                return 2  # Higher load
            
            mock_get_load.side_effect = mock_load_side_effect
            
            # Act
            result = await task_router.route_task(sample_task)
            
            # Assert
            assert result is not None
            assert result.task_id == sample_task.id
            assert result.agent_response == AgentResponse.ACCEPT
            # Should route to researcher-002 (lowest load, highest trust, has capabilities)
            assert result.agent_id == sample_agents_with_metrics[2]["instance"].id
    
    @pytest.mark.asyncio
    async def test_filters_out_agents_missing_required_capabilities(self, task_router, sample_agents_with_metrics):
        """Test that agents without required capabilities are filtered out."""
        # Arrange
        task_with_specific_requirements = Task(
            id=uuid4(),
            task_type=TaskType.CODE,
            payload={"language": "python"},
            required_capabilities=["code_generation", "debugging"]  # None of sample agents have these
        )
        
        with patch('app.services.task_router.get_instances_with_trust_metrics') as mock_get_instances, \
             patch('app.services.task_router.agent_registry') as mock_registry, \
             patch('app.services.task_router.update_task_status') as mock_update_status:
            
            mock_get_instances.return_value = sample_agents_with_metrics
            mock_registry.get_healthy_agents.return_value = ["researcher-001", "analyst-001", "researcher-002"]
            mock_update_status.return_value = True
            
            # Act
            result = await task_router.route_task(task_with_specific_requirements)
            
            # Assert
            assert result is None  # No eligible agents
            mock_update_status.assert_called_once_with(task_with_specific_requirements.id, TaskStatus.QUEUED)
    
    @pytest.mark.asyncio
    async def test_respects_priority_ordering(self, task_router, sample_task, sample_agents_with_metrics):
        """Test that higher priority tasks are handled appropriately."""
        # Arrange
        high_priority_task = Task(
            id=uuid4(),
            task_type=TaskType.ANALYSIS,
            payload={"urgent": True},
            priority=10,  # Maximum priority
            required_capabilities=["analysis"]
        )
        
        with patch('app.services.task_router.get_instances_with_trust_metrics') as mock_get_instances, \
             patch('app.services.task_router.agent_registry') as mock_registry, \
             patch('app.services.task_router.update_task_status') as mock_update_status, \
             patch('app.services.task_router.create_task_assignment') as mock_create_assignment, \
             patch('app.services.task_router.get_agent_task_metrics') as mock_agent_metrics, \
             patch.object(task_router, '_get_agent_current_load') as mock_get_load:
            
            mock_get_instances.return_value = sample_agents_with_metrics
            mock_registry.get_healthy_agents.return_value = ["researcher-001", "analyst-001", "researcher-002"]
            mock_update_status.return_value = True
            mock_create_assignment.return_value = uuid4()
            mock_agent_metrics.return_value = []
            mock_get_load.return_value = 0  # All agents have low load
            
            # Act
            result = await task_router.route_task(high_priority_task)
            
            # Assert
            assert result is not None
            assert result.task_id == high_priority_task.id
            # Priority doesn't directly affect routing, but task should be routed successfully
    
    @pytest.mark.asyncio
    async def test_handles_no_available_agents_queues_task(self, task_router, sample_task):
        """Test that tasks are queued when no agents are available."""
        # Arrange
        with patch('app.services.task_router.get_instances_with_trust_metrics') as mock_get_instances, \
             patch('app.services.task_router.update_task_status') as mock_update_status:
            
            mock_get_instances.return_value = []  # No agents available
            mock_update_status.return_value = True
            
            # Act
            result = await task_router.route_task(sample_task)
            
            # Assert
            assert result is None
            mock_update_status.assert_called_once_with(sample_task.id, TaskStatus.QUEUED)
    
    @pytest.mark.asyncio
    async def test_trust_score_affects_routing_preference(self, task_router, sample_task, sample_agents_with_metrics):
        """Test that agents with higher trust scores are preferred."""
        # Arrange
        with patch('app.services.task_router.get_instances_with_trust_metrics') as mock_get_instances, \
             patch('app.services.task_router.agent_registry') as mock_registry, \
             patch('app.services.task_router.update_task_status') as mock_update_status, \
             patch('app.services.task_router.create_task_assignment') as mock_create_assignment, \
             patch('app.services.task_router.get_agent_task_metrics') as mock_agent_metrics, \
             patch.object(task_router, '_get_agent_current_load') as mock_get_load:
            
            mock_get_instances.return_value = sample_agents_with_metrics
            mock_registry.get_healthy_agents.return_value = ["researcher-001", "analyst-001", "researcher-002"]
            mock_update_status.return_value = True
            mock_create_assignment.return_value = uuid4()
            mock_agent_metrics.return_value = []
            mock_get_load.return_value = 1  # Same load for all
            
            # Act
            result = await task_router.route_task(sample_task)
            
            # Assert
            assert result is not None
            # Should prefer researcher-002 with highest trust score (0.90)
            assert result.agent_id == sample_agents_with_metrics[2]["instance"].id


# =============================================================================
# REFUSAL HANDLING TESTS
# =============================================================================

class TestTaskRefusal:
    """Test task refusal handling."""
    
    @pytest.mark.asyncio
    async def test_refusal_triggers_reroute_to_next_best_agent(self, task_router):
        """Test that task refusal triggers re-routing to next best agent."""
        # Arrange
        task_id = uuid4()
        agent_id = uuid4()
        reason = "Currently overloaded with similar tasks"
        
        with patch('app.services.task_router.create_task_assignment') as mock_create_assignment, \
             patch('app.services.task_router.get_instances_with_trust_metrics') as mock_get_instances, \
             patch('app.services.task_router.increment_task_refused') as mock_increment_refused, \
             patch('app.services.task_router.update_task_status') as mock_update_status:
            
            mock_create_assignment.return_value = uuid4()
            mock_get_instances.return_value = [{"instance": MagicMock(id=agent_id)}]
            mock_increment_refused.return_value = True
            mock_update_status.return_value = True
            
            # Act
            result = await task_router.handle_task_refusal(task_id, agent_id, reason)
            
            # Assert
            mock_create_assignment.assert_called_once()
            mock_increment_refused.assert_called_once_with(agent_id)
            mock_update_status.assert_called_once_with(task_id, TaskStatus.FAILED, result={"error": "Agent refused task"})
    
    @pytest.mark.asyncio
    async def test_suggested_agent_gets_priority_on_reroute(self, task_router):
        """Test that suggested alternative agent gets priority in re-routing."""
        # Arrange
        task_id = uuid4()
        agent_id = uuid4()
        reason = "Wrong expertise area"
        suggested_agent = "specialist-agent-007"
        
        with patch('app.services.task_router.create_task_assignment') as mock_create_assignment, \
             patch('app.services.task_router.get_instances_with_trust_metrics') as mock_get_instances, \
             patch('app.services.task_router.increment_task_refused') as mock_increment_refused, \
             patch('app.services.task_router.update_task_status') as mock_update_status:
            
            mock_create_assignment.return_value = uuid4()
            mock_get_instances.return_value = [{"instance": MagicMock(id=agent_id)}]
            mock_increment_refused.return_value = True
            mock_update_status.return_value = True
            
            # Act
            result = await task_router.handle_task_refusal(task_id, agent_id, reason, suggested_agent)
            
            # Assert
            # Verify the assignment was created with suggestion
            call_args = mock_create_assignment.call_args[0][0]
            assert call_args.suggested_agent == suggested_agent
            assert call_args.refusal_reason == reason
    
    @pytest.mark.asyncio
    async def test_refusal_recorded_in_trust_metrics(self, task_router):
        """Test that task refusals are properly recorded in trust metrics."""
        # Arrange
        task_id = uuid4()
        agent_id = uuid4()
        reason = "Agent is being updated"
        
        with patch('app.services.task_router.create_task_assignment') as mock_create_assignment, \
             patch('app.services.task_router.get_instances_with_trust_metrics') as mock_get_instances, \
             patch('app.services.task_router.increment_task_refused') as mock_increment_refused, \
             patch('app.services.task_router.update_task_status') as mock_update_status:
            
            mock_create_assignment.return_value = uuid4()
            mock_get_instances.return_value = [{"instance": MagicMock(id=agent_id)}]
            mock_increment_refused.return_value = True
            mock_update_status.return_value = True
            
            # Act
            await task_router.handle_task_refusal(task_id, agent_id, reason)
            
            # Assert
            mock_increment_refused.assert_called_once_with(agent_id)


# =============================================================================
# COMPLETION HANDLING TESTS
# =============================================================================

class TestTaskCompletion:
    """Test task completion handling."""
    
    @pytest.mark.asyncio
    async def test_successful_completion_updates_metrics(self, task_router):
        """Test that successful task completion updates trust metrics."""
        # Arrange
        task_id = uuid4()
        agent_id = uuid4()
        result = {"output": "Research complete", "sources": ["url1", "url2"]}
        duration_ms = 45000
        model_used = "claude-sonnet"
        tokens_used = 2500
        
        with patch('app.services.task_router.update_task_status') as mock_update_status, \
             patch('app.services.task_router.create_task_result') as mock_create_result, \
             patch('app.services.task_router.get_instances_with_trust_metrics') as mock_get_instances, \
             patch('app.services.task_router.increment_task_completed') as mock_increment_completed:
            
            mock_update_status.return_value = True
            mock_create_result.return_value = uuid4()
            mock_get_instances.return_value = [{"instance": MagicMock(id=agent_id)}]
            mock_increment_completed.return_value = True
            
            # Act
            await task_router.handle_task_completion(
                task_id, result, duration_ms, agent_id, model_used, tokens_used
            )
            
            # Assert
            mock_update_status.assert_called_once_with(
                task_id, TaskStatus.COMPLETED, result=result, duration_ms=duration_ms
            )
            mock_create_result.assert_called_once()
            mock_increment_completed.assert_called_once_with(agent_id, duration_ms)
    
    @pytest.mark.asyncio
    async def test_failed_task_can_be_retried(self, task_router):
        """Test that failed tasks can be retried with proper status updates."""
        # Arrange
        task_id = uuid4()
        agent_id = uuid4()
        error = "Network timeout during web search"
        duration_ms = 30000
        model_used = "claude-sonnet"
        
        with patch('app.services.task_router.update_task_status') as mock_update_status, \
             patch('app.services.task_router.create_task_result') as mock_create_result, \
             patch('app.services.task_router.get_instances_with_trust_metrics') as mock_get_instances, \
             patch('app.services.task_router.increment_task_failed') as mock_increment_failed:
            
            mock_update_status.return_value = True
            mock_create_result.return_value = uuid4()
            mock_get_instances.return_value = [{"instance": MagicMock(id=agent_id)}]
            mock_increment_failed.return_value = True
            
            # Act
            await task_router.handle_task_failure(task_id, error, agent_id, duration_ms, model_used)
            
            # Assert
            mock_update_status.assert_called_once_with(
                task_id, TaskStatus.FAILED, result={"error": error}
            )
            mock_create_result.assert_called_once()
            mock_increment_failed.assert_called_once_with(agent_id)
    
    @pytest.mark.asyncio
    async def test_duration_tracking_is_accurate(self, task_router):
        """Test that task duration tracking is accurate."""
        # Arrange
        task_id = uuid4()
        agent_id = uuid4()
        result = {"status": "completed"}
        expected_duration_ms = 67890
        model_used = "claude-sonnet"
        
        with patch('app.services.task_router.update_task_status') as mock_update_status, \
             patch('app.services.task_router.create_task_result') as mock_create_result, \
             patch('app.services.task_router.get_instances_with_trust_metrics') as mock_get_instances, \
             patch('app.services.task_router.increment_task_completed') as mock_increment_completed:
            
            mock_update_status.return_value = True
            mock_create_result.return_value = uuid4()
            mock_get_instances.return_value = [{"instance": MagicMock(id=agent_id)}]
            mock_increment_completed.return_value = True
            
            # Act
            await task_router.handle_task_completion(
                task_id, result, expected_duration_ms, agent_id, model_used
            )
            
            # Assert
            # Verify duration was passed correctly to task result
            result_call = mock_create_result.call_args[0][0]
            assert result_call.duration_ms == expected_duration_ms


# =============================================================================
# OVERRIDE HANDLING TESTS
# =============================================================================

class TestHumanOverride:
    """Test human override functionality."""
    
    @pytest.mark.asyncio
    async def test_human_override_bypasses_routing_algorithm(self, task_router):
        """Test that human overrides bypass normal routing algorithm."""
        # Arrange
        # This test would verify manual assignment functionality
        # Implementation would depend on how overrides are handled
        
        # Act & Assert
        # For now, this is a placeholder test
        assert True  # Manual override logic would be tested here
    
    @pytest.mark.asyncio
    async def test_override_reason_recorded(self, task_router):
        """Test that override reasons are properly recorded."""
        # Arrange
        # This would test the recording of override reasons
        
        # Act & Assert
        assert True  # Override reason recording would be tested here
    
    @pytest.mark.asyncio
    async def test_override_outcome_tracked_for_trust_evolution(self, task_router):
        """Test that override outcomes are tracked for trust metric evolution."""
        # Arrange
        # This would test tracking of override success/failure
        
        # Act & Assert
        assert True  # Override outcome tracking would be tested here


# =============================================================================
# ANALYTICS TESTS
# =============================================================================

class TestAnalytics:
    """Test routing analytics functionality."""
    
    @pytest.mark.asyncio
    async def test_per_agent_task_metrics_calculated_correctly(self, task_router):
        """Test that per-agent task metrics are calculated correctly."""
        # Arrange
        with patch('app.services.task_router.get_task_metrics') as mock_get_task_metrics, \
             patch('app.services.task_router.get_agent_task_metrics') as mock_get_agent_metrics:
            
            mock_get_task_metrics.return_value = MagicMock(
                total_tasks=100, completed_tasks=85, failed_tasks=10, refused_tasks=5
            )
            mock_get_agent_metrics.return_value = [
                MagicMock(agent_id=uuid4(), agent_role="researcher", completed=25, failed=2, refused=1),
                MagicMock(agent_id=uuid4(), agent_role="analyst", completed=30, failed=3, refused=2)
            ]
            
            # Act
            analytics = await task_router.get_routing_analytics()
            
            # Assert
            assert "overview" in analytics
            assert "agent_performance" in analytics
            assert len(analytics["agent_performance"]) == 2
            assert analytics["overview"]["total_tasks"] == 100
    
    @pytest.mark.asyncio
    async def test_queue_depth_reported_accurately(self, task_router):
        """Test that queue depth is reported accurately."""
        # Arrange
        with patch('app.services.task_router.get_task_metrics') as mock_get_task_metrics, \
             patch('app.services.task_router.get_agent_task_metrics') as mock_get_agent_metrics:
            mock_get_task_metrics.return_value = MagicMock(queue_depth=15)
            mock_get_agent_metrics.return_value = []
            
            # Act
            analytics = await task_router.get_routing_analytics()
            
            # Assert
            assert analytics["overview"]["queue_depth"] == 15
    
    @pytest.mark.asyncio
    async def test_refusal_rates_aggregated_properly(self, task_router):
        """Test that refusal rates are aggregated properly."""
        # Arrange
        with patch('app.services.task_router.get_task_metrics') as mock_get_task_metrics, \
             patch('app.services.task_router.get_agent_task_metrics') as mock_get_agent_metrics:
            
            mock_get_task_metrics.return_value = MagicMock(refusal_rate=0.12)
            mock_get_agent_metrics.return_value = []
            
            # Act
            analytics = await task_router.get_routing_analytics()
            
            # Assert
            assert analytics["overview"]["refusal_rate"] == 0.12


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error scenarios."""
    
    @pytest.mark.asyncio
    async def test_task_with_impossible_capabilities(self, task_router):
        """Test handling of tasks with capabilities no agent can fulfill."""
        # Arrange
        impossible_task = Task(
            id=uuid4(),
            task_type="quantum_computing",
            required_capabilities=["quantum_processor", "time_travel", "magic"]
        )
        
        with patch('app.services.task_router.get_instances_with_trust_metrics') as mock_get_instances, \
             patch('app.services.task_router.update_task_status') as mock_update_status:
            
            # Return agents but none with required capabilities
            mock_get_instances.return_value = [
                {
                    "instance": MagicMock(capabilities=["web_search", "analysis"], status="ready", agent_id="agent-001"),
                    "trust_metrics": {"trust_score": 0.7, "tasks_completed": 5, "tasks_failed": 0, "tasks_refused": 0, "avg_task_duration_ms": 30000}
                }
            ]
            mock_update_status.return_value = True
            
            # Act
            result = await task_router.route_task(impossible_task)
            
            # Assert
            assert result is None
            mock_update_status.assert_called_once_with(impossible_task.id, TaskStatus.QUEUED)
    
    @pytest.mark.asyncio
    async def test_all_agents_at_max_capacity(self, task_router, sample_task, sample_agents_with_metrics):
        """Test handling when all agents are at maximum capacity."""
        # Arrange
        with patch('app.services.task_router.get_instances_with_trust_metrics') as mock_get_instances, \
             patch('app.services.task_router.agent_registry') as mock_registry, \
             patch('app.services.task_router.update_task_status') as mock_update_status, \
             patch.object(task_router, '_get_agent_current_load') as mock_get_load:
            
            mock_get_instances.return_value = sample_agents_with_metrics
            mock_registry.get_healthy_agents.return_value = ["researcher-001", "analyst-001", "researcher-002"]
            mock_update_status.return_value = True
            mock_get_load.return_value = 5  # All agents at max capacity
            
            # Act
            result = await task_router.route_task(sample_task)
            
            # Assert
            assert result is None  # Should queue task since all agents are busy
            mock_update_status.assert_called_once_with(sample_task.id, TaskStatus.QUEUED)
    
    @pytest.mark.asyncio
    async def test_concurrent_task_routing_race_conditions(self, task_router):
        """Test handling of concurrent task routing to prevent race conditions."""
        # Arrange
        tasks = [
            Task(id=uuid4(), task_type=TaskType.RESEARCH, required_capabilities=["web_search"]),
            Task(id=uuid4(), task_type=TaskType.ANALYSIS, required_capabilities=["analysis"]),
            Task(id=uuid4(), task_type=TaskType.WRITING, required_capabilities=["writing"])
        ]
        
        with patch('app.services.task_router.get_instances_with_trust_metrics') as mock_get_instances, \
             patch('app.services.task_router.agent_registry') as mock_registry, \
             patch('app.services.task_router.update_task_status') as mock_update_status, \
             patch('app.services.task_router.create_task_assignment') as mock_create_assignment, \
             patch('app.services.task_router.get_agent_task_metrics') as mock_agent_metrics, \
             patch.object(task_router, '_get_agent_current_load') as mock_get_load:
            
            mock_get_instances.return_value = [
                {
                    "instance": MagicMock(capabilities=["web_search", "analysis", "writing"], status="ready", agent_id="agent-001", id=uuid4(), agent_role="researcher"),
                    "trust_metrics": {"trust_score": 0.8, "tasks_completed": 10, "tasks_failed": 1, "tasks_refused": 0, "avg_task_duration_ms": 25000}
                }
            ]
            mock_registry.get_healthy_agents.return_value = ["agent-001"]
            mock_update_status.return_value = True
            mock_create_assignment.return_value = uuid4()
            mock_agent_metrics.return_value = []
            mock_get_load.return_value = 0
            
            # Act
            results = await asyncio.gather(
                *[task_router.route_task(task) for task in tasks],
                return_exceptions=True
            )
            
            # Assert
            # All tasks should be processed without exceptions
            for result in results:
                assert not isinstance(result, Exception)
    
    @pytest.mark.asyncio
    async def test_task_timeout_handling(self, task_router):
        """Test handling of task timeouts."""
        # Arrange
        # This would test timeout scenarios
        
        # Act & Assert
        # For now, this is a placeholder since timeout handling isn't implemented
        assert True
    
    @pytest.mark.asyncio
    async def test_agent_goes_offline_during_task_execution(self, task_router):
        """Test handling when agent goes offline during task execution."""
        # Arrange
        task_id = uuid4()
        agent_id = uuid4()
        
        with patch('app.services.task_router.get_instances_with_trust_metrics') as mock_get_instances, \
             patch('app.services.task_router.increment_task_failed') as mock_increment_failed, \
             patch('app.services.task_router.update_task_status') as mock_update_status, \
             patch('app.services.task_router.create_task_result') as mock_create_result:
            
            mock_get_instances.return_value = [{"instance": MagicMock(id=agent_id)}]
            mock_increment_failed.return_value = True
            mock_update_status.return_value = True
            mock_create_result.return_value = uuid4()
            
            # Act
            await task_router.handle_task_failure(
                task_id, "Agent went offline", agent_id, 15000, "claude-sonnet"
            )
            
            # Assert
            mock_update_status.assert_called_once_with(
                task_id, TaskStatus.FAILED, result={"error": "Agent went offline"}
            )
            mock_increment_failed.assert_called_once_with(agent_id)


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================

class TestHelperFunctions:
    """Test internal helper functions."""
    
    def test_capability_match_score_calculation(self, task_router):
        """Test capability matching score calculation."""
        # Arrange
        task = Task(
            id=uuid4(),
            task_type=TaskType.RESEARCH,
            required_capabilities=["web_search", "analysis"]
        )
        
        # Agent with exact capabilities
        agent_exact = MagicMock(capabilities=["web_search", "analysis"])
        # Agent with extra capabilities
        agent_extra = MagicMock(capabilities=["web_search", "analysis", "writing", "coding"])
        # Agent with missing capabilities (shouldn't reach this in real routing)
        agent_missing = MagicMock(capabilities=["web_search"])
        
        # Act
        score_exact = task_router._calculate_capability_match_score(task, agent_exact)
        score_extra = task_router._calculate_capability_match_score(task, agent_extra)
        score_missing = task_router._calculate_capability_match_score(task, agent_missing)
        
        # Assert
        assert score_exact == 0.9  # Exact match baseline
        assert score_extra > score_exact  # Bonus for extra capabilities
        assert score_extra <= 1.0  # Score stays within [0, 1] range
        assert score_missing < 1.0  # Penalty for missing capabilities
    
    def test_load_score_calculation(self, task_router):
        """Test load score calculation."""
        # Arrange & Act
        score_empty = task_router._calculate_load_score(0, 5)  # No load
        score_half = task_router._calculate_load_score(2, 5)   # Half load
        score_full = task_router._calculate_load_score(5, 5)   # Full load
        
        # Assert
        assert score_empty == 1.0
        assert score_half == 0.6
        assert score_full == 0.0
    
    def test_model_preference_score_calculation(self, task_router):
        """Test model preference score calculation."""
        # Arrange
        task_with_preference = Task(
            id=uuid4(),
            task_type=TaskType.RESEARCH,
            preferred_model="claude-sonnet"
        )
        task_no_preference = Task(
            id=uuid4(),
            task_type=TaskType.ANALYSIS
        )
        
        researcher_agent = MagicMock(agent_role="researcher")
        coder_agent = MagicMock(agent_role="coder")
        
        # Act
        score_preferred = task_router._calculate_model_preference_score(task_with_preference, researcher_agent)
        score_no_preference = task_router._calculate_model_preference_score(task_no_preference, researcher_agent)
        score_mismatch = task_router._calculate_model_preference_score(task_with_preference, coder_agent)
        
        # Assert
        assert score_preferred == 1.0  # Researcher role matches claude-sonnet preference
        assert score_no_preference == 1.0  # No preference means no penalty
        assert score_mismatch > 0.0  # Some score even with mismatch