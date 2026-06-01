"""
Comprehensive unit tests for the EventPublisher service.

Tests cover:
- Core publishing methods (broadcast, channel, instance)
- All convenience methods (agent lifecycle, task lifecycle, council, system, notifications)
- Error handling and graceful degradation
- Event serialization and structure
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from datetime import datetime

from app.services.event_publisher import EventPublisher, event_publisher
from app.models.ws_events import (
    BaseEvent,
    AgentActivityEvent,
    TaskProgressEvent,
    CouncilEvent,
    SystemEvent,
    NotificationEvent,
    EventType,
    AgentStatus,
    TaskStage,
    CouncilEventType,
    SystemLevel,
    NotificationPriority,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_ws_manager():
    """Mock WebSocket manager with all broadcast/send methods."""
    mgr = AsyncMock()
    mgr.broadcast_to_all = AsyncMock()
    mgr.broadcast_to_channel = AsyncMock()
    mgr.send_personal_message = AsyncMock()
    return mgr


@pytest.fixture
def publisher(mock_ws_manager):
    """EventPublisher with injected mock manager."""
    pub = EventPublisher()
    pub._manager = mock_ws_manager
    return pub


# =============================================================================
# Core Publishing
# =============================================================================


class TestPublish:
    """Tests for the publish() method (broadcast to all)."""

    async def test_publish_broadcasts_to_all(self, publisher, mock_ws_manager):
        event = AgentActivityEvent(
            agent_id="test-agent",
            action="test",
            status=AgentStatus.ACTIVE,
        )
        result = await publisher.publish(event)

        assert result is True
        mock_ws_manager.broadcast_to_all.assert_awaited_once()
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert payload["event_type"] == "agent_activity"
        assert payload["agent_id"] == "test-agent"

    async def test_publish_returns_false_on_error(self, publisher, mock_ws_manager):
        mock_ws_manager.broadcast_to_all.side_effect = RuntimeError("ws down")
        event = AgentActivityEvent(agent_id="a", action="b", status=AgentStatus.IDLE)
        result = await publisher.publish(event)
        assert result is False

    async def test_publish_event_contains_id_and_timestamp(
        self, publisher, mock_ws_manager
    ):
        event = SystemEvent(level=SystemLevel.INFO, message="hello")
        await publisher.publish(event)

        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert "event_id" in payload
        assert "timestamp" in payload

    async def test_publish_passes_session_id_through(self, publisher, mock_ws_manager):
        event = NotificationEvent(title="t", body="b", session_id="sess-42")
        await publisher.publish(event)

        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert payload["session_id"] == "sess-42"


class TestPublishToChannel:
    """Tests for publish_to_channel()."""

    async def test_channel_publish_routes_correctly(self, publisher, mock_ws_manager):
        event = TaskProgressEvent(
            task_id="t1", progress_pct=50, stage=TaskStage.PROCESSING
        )
        result = await publisher.publish_to_channel("chan-1", event)

        assert result is True
        mock_ws_manager.broadcast_to_channel.assert_awaited_once()
        args = mock_ws_manager.broadcast_to_channel.call_args[0]
        assert args[0] == "chan-1"
        assert args[1]["task_id"] == "t1"

    async def test_channel_publish_returns_false_on_error(
        self, publisher, mock_ws_manager
    ):
        mock_ws_manager.broadcast_to_channel.side_effect = RuntimeError("fail")
        event = SystemEvent(level=SystemLevel.INFO, message="x")
        result = await publisher.publish_to_channel("ch", event)
        assert result is False


class TestPublishToInstance:
    """Tests for publish_to_instance()."""

    async def test_instance_publish_routes_correctly(self, publisher, mock_ws_manager):
        event = NotificationEvent(title="alert", body="check this")
        result = await publisher.publish_to_instance("inst-7", event)

        assert result is True
        mock_ws_manager.send_personal_message.assert_awaited_once()
        args = mock_ws_manager.send_personal_message.call_args[0]
        assert args[0] == "inst-7"
        assert args[1]["title"] == "alert"

    async def test_instance_publish_returns_false_on_error(
        self, publisher, mock_ws_manager
    ):
        mock_ws_manager.send_personal_message.side_effect = ConnectionError("gone")
        event = SystemEvent(level=SystemLevel.ERROR, message="err")
        result = await publisher.publish_to_instance("x", event)
        assert result is False


# =============================================================================
# Agent Convenience Methods
# =============================================================================


class TestAgentEvents:
    """Tests for agent lifecycle convenience methods."""

    async def test_agent_started(self, publisher, mock_ws_manager):
        result = await publisher.agent_started(
            "comp-agent", "analyzing", "Starting analysis"
        )
        assert result is True
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert payload["agent_id"] == "comp-agent"
        assert payload["action"] == "analyzing"
        assert payload["status"] == "active"
        assert payload["message"] == "Starting analysis"

    async def test_agent_started_default_message(self, publisher, mock_ws_manager):
        await publisher.agent_started("my-agent", "parsing")
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert "my-agent" in payload["message"]
        assert "parsing" in payload["message"]

    async def test_agent_started_with_session(self, publisher, mock_ws_manager):
        await publisher.agent_started("a", "b", session_id="s-1")
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert payload["session_id"] == "s-1"

    async def test_agent_thinking(self, publisher, mock_ws_manager):
        result = await publisher.agent_thinking("thinker", "Deep thought...")
        assert result is True
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert payload["agent_id"] == "thinker"
        assert payload["action"] == "thinking"
        assert payload["status"] == "thinking"
        assert payload["message"] == "Deep thought..."

    async def test_agent_thinking_default_message(self, publisher, mock_ws_manager):
        await publisher.agent_thinking("oracle")
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert "oracle" in payload["message"]
        assert "thinking" in payload["message"]

    async def test_agent_complete(self, publisher, mock_ws_manager):
        result = await publisher.agent_complete("worker", "compile", "Done!")
        assert result is True
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert payload["status"] == "complete"
        assert payload["action"] == "compile"
        assert payload["message"] == "Done!"

    async def test_agent_complete_default_message(self, publisher, mock_ws_manager):
        await publisher.agent_complete("w", "build")
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert "completed" in payload["message"]

    async def test_agent_error(self, publisher, mock_ws_manager):
        result = await publisher.agent_error("buggy", "crash", "Stack overflow")
        assert result is True
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert payload["status"] == "error"
        assert payload["message"] == "Stack overflow"

    async def test_agent_error_with_session(self, publisher, mock_ws_manager):
        await publisher.agent_error("a", "b", "err", session_id="s-99")
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert payload["session_id"] == "s-99"


# =============================================================================
# Task Convenience Methods
# =============================================================================


class TestTaskEvents:
    """Tests for task lifecycle convenience methods."""

    async def test_task_started(self, publisher, mock_ws_manager):
        result = await publisher.task_started("task-1", "Beginning work", total_steps=5)
        assert result is True
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert payload["task_id"] == "task-1"
        assert payload["progress_pct"] == 0
        assert payload["stage"] == "starting"
        assert payload["total_steps"] == 5
        assert payload["current_step_num"] == 1

    async def test_task_started_default_message(self, publisher, mock_ws_manager):
        await publisher.task_started("t-abc")
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert "t-abc" in payload["message"]

    async def test_task_started_without_steps(self, publisher, mock_ws_manager):
        await publisher.task_started("t-2")
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert payload["total_steps"] is None
        assert payload["current_step_num"] is None

    async def test_task_progress(self, publisher, mock_ws_manager):
        result = await publisher.task_progress(
            "t-1",
            45,
            message="Halfway",
            eta_seconds=30,
            current_step="embedding",
            current_step_num=3,
            total_steps=6,
        )
        assert result is True
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert payload["progress_pct"] == 45
        assert payload["stage"] == "processing"
        assert payload["eta_seconds"] == 30
        assert payload["current_step"] == "embedding"
        assert payload["current_step_num"] == 3
        assert payload["total_steps"] == 6

    async def test_task_progress_with_session(self, publisher, mock_ws_manager):
        await publisher.task_progress("t", 10, session_id="s-5")
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert payload["session_id"] == "s-5"

    async def test_task_complete(self, publisher, mock_ws_manager):
        result = await publisher.task_complete("t-done", "All finished")
        assert result is True
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert payload["progress_pct"] == 100
        assert payload["stage"] == "complete"
        assert payload["message"] == "All finished"

    async def test_task_complete_default_message(self, publisher, mock_ws_manager):
        await publisher.task_complete("t-x")
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert "t-x" in payload["message"]

    async def test_task_failed(self, publisher, mock_ws_manager):
        result = await publisher.task_failed("t-bad", "OOM killed")
        assert result is True
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert payload["progress_pct"] == 0
        assert payload["stage"] == "failed"
        assert payload["message"] == "OOM killed"


# =============================================================================
# Council Convenience Methods
# =============================================================================


class TestCouncilEvents:
    """Tests for council deliberation convenience methods."""

    async def test_council_perspective(self, publisher, mock_ws_manager):
        result = await publisher.council_perspective(
            "council-1", "ethics-agent", "This is ethically sound", confidence=0.85
        )
        assert result is True
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert payload["event_type"] == "council"
        assert payload["council_session_id"] == "council-1"
        assert payload["event"] == "perspective_added"
        assert payload["agent_id"] == "ethics-agent"
        assert payload["content"] == "This is ethically sound"
        assert payload["confidence"] == 0.85

    async def test_council_perspective_without_confidence(
        self, publisher, mock_ws_manager
    ):
        await publisher.council_perspective("c", "a", "opinion")
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert payload["confidence"] is None

    async def test_council_vote(self, publisher, mock_ws_manager):
        result = await publisher.council_vote(
            "council-2", "logic-agent", "approve", confidence=0.9
        )
        assert result is True
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert payload["event"] == "vote_cast"
        assert payload["vote"] == "approve"
        assert payload["confidence"] == 0.9

    async def test_council_synthesis(self, publisher, mock_ws_manager):
        result = await publisher.council_synthesis("council-3", "The consensus is...")
        assert result is True
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert payload["event"] == "synthesis_ready"
        assert payload["content"] == "The consensus is..."
        assert payload["agent_id"] is None

    async def test_council_synthesis_with_session(self, publisher, mock_ws_manager):
        await publisher.council_synthesis("c", "text", session_id="s-7")
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert payload["session_id"] == "s-7"


# =============================================================================
# System Event Convenience Methods
# =============================================================================


class TestSystemEvents:
    """Tests for system event convenience methods."""

    async def test_system_info(self, publisher, mock_ws_manager):
        result = await publisher.system_info(
            "Service started", source="health_monitor", details={"uptime": 3600}
        )
        assert result is True
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert payload["event_type"] == "system"
        assert payload["level"] == "info"
        assert payload["message"] == "Service started"
        assert payload["source"] == "health_monitor"
        assert payload["details"] == {"uptime": 3600}

    async def test_system_info_minimal(self, publisher, mock_ws_manager):
        await publisher.system_info("ok")
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert payload["source"] is None
        assert payload["details"] is None

    async def test_system_warning(self, publisher, mock_ws_manager):
        result = await publisher.system_warning(
            "High memory", source="monitor", details={"mem_pct": 90}
        )
        assert result is True
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert payload["level"] == "warning"
        assert payload["message"] == "High memory"

    async def test_system_error(self, publisher, mock_ws_manager):
        result = await publisher.system_error(
            "DB connection lost",
            source="postgres",
            error_code="PG_CONN_FAIL",
            details={"retries": 3},
        )
        assert result is True
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert payload["level"] == "error"
        assert payload["error_code"] == "PG_CONN_FAIL"
        assert payload["details"]["retries"] == 3

    async def test_system_error_minimal(self, publisher, mock_ws_manager):
        await publisher.system_error("boom")
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert payload["error_code"] is None


# =============================================================================
# Notification Convenience Methods
# =============================================================================


class TestNotificationEvents:
    """Tests for notification convenience methods."""

    async def test_notify_all(self, publisher, mock_ws_manager):
        result = await publisher.notify(
            title="Task Done",
            body="Your analysis is ready",
            action_url="/results/123",
            priority=NotificationPriority.HIGH,
            auto_dismiss_ms=5000,
        )
        assert result is True
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert payload["event_type"] == "notification"
        assert payload["title"] == "Task Done"
        assert payload["body"] == "Your analysis is ready"
        assert payload["action_url"] == "/results/123"
        assert payload["priority"] == "high"
        assert payload["auto_dismiss_ms"] == 5000

    async def test_notify_defaults(self, publisher, mock_ws_manager):
        await publisher.notify(title="Hey", body="Check this")
        payload = mock_ws_manager.broadcast_to_all.call_args[0][0]
        assert payload["priority"] == "normal"
        assert payload["auto_dismiss_ms"] is None
        assert payload["action_url"] is None

    async def test_notify_instance(self, publisher, mock_ws_manager):
        result = await publisher.notify_instance(
            "inst-42",
            title="Personal Alert",
            body="Just for you",
            action_url="/me",
            priority=NotificationPriority.URGENT,
        )
        assert result is True
        mock_ws_manager.send_personal_message.assert_awaited_once()
        args = mock_ws_manager.send_personal_message.call_args[0]
        assert args[0] == "inst-42"
        assert args[1]["title"] == "Personal Alert"
        assert args[1]["priority"] == "urgent"

    async def test_notify_instance_defaults(self, publisher, mock_ws_manager):
        await publisher.notify_instance("i", title="T", body="B")
        payload = mock_ws_manager.send_personal_message.call_args[0][1]
        assert payload["priority"] == "normal"


# =============================================================================
# Event Model Serialization
# =============================================================================


class TestEventSerialization:
    """Tests for event model serialization."""

    def test_agent_event_serializes_enums(self):
        event = AgentActivityEvent(
            agent_id="a", action="x", status=AgentStatus.EXECUTING
        )
        msg = event.to_ws_message()
        assert msg["status"] == "executing"
        assert msg["event_type"] == "agent_activity"

    def test_task_event_serializes_stage(self):
        event = TaskProgressEvent(
            task_id="t", progress_pct=75, stage=TaskStage.FINALIZING
        )
        msg = event.to_ws_message()
        assert msg["stage"] == "finalizing"
        assert msg["progress_pct"] == 75

    def test_council_event_serializes_type(self):
        event = CouncilEvent(
            council_session_id="c",
            event=CouncilEventType.DEBATE_ROUND,
            round_number=2,
        )
        msg = event.to_ws_message()
        assert msg["event"] == "debate_round"
        assert msg["round_number"] == 2

    def test_system_event_serializes_level(self):
        event = SystemEvent(level=SystemLevel.CRITICAL, message="panic")
        msg = event.to_ws_message()
        assert msg["level"] == "critical"

    def test_notification_event_serializes_priority(self):
        event = NotificationEvent(
            title="T", body="B", priority=NotificationPriority.LOW
        )
        msg = event.to_ws_message()
        assert msg["priority"] == "low"
        assert msg["dismissible"] is True

    def test_event_has_unique_ids(self):
        e1 = SystemEvent(level=SystemLevel.INFO, message="a")
        e2 = SystemEvent(level=SystemLevel.INFO, message="b")
        assert e1.event_id != e2.event_id

    def test_event_timestamp_is_populated(self):
        event = SystemEvent(level=SystemLevel.INFO, message="t")
        assert event.timestamp is not None
        msg = event.to_ws_message()
        assert isinstance(msg["timestamp"], str)


# =============================================================================
# Global Singleton
# =============================================================================


class TestGlobalSingleton:
    """Tests for the module-level singleton."""

    def test_singleton_is_event_publisher_instance(self):
        assert isinstance(event_publisher, EventPublisher)

    def test_singleton_has_manager(self):
        assert event_publisher._manager is not None


# =============================================================================
# Error Resilience
# =============================================================================


class TestErrorResilience:
    """Tests that errors in WS manager don't crash the publisher."""

    async def test_publish_catches_type_error(self, publisher, mock_ws_manager):
        mock_ws_manager.broadcast_to_all.side_effect = TypeError("bad arg")
        event = SystemEvent(level=SystemLevel.INFO, message="x")
        result = await publisher.publish(event)
        assert result is False

    async def test_channel_publish_catches_value_error(
        self, publisher, mock_ws_manager
    ):
        mock_ws_manager.broadcast_to_channel.side_effect = ValueError("nope")
        event = SystemEvent(level=SystemLevel.INFO, message="x")
        result = await publisher.publish_to_channel("ch", event)
        assert result is False

    async def test_instance_publish_catches_os_error(self, publisher, mock_ws_manager):
        mock_ws_manager.send_personal_message.side_effect = OSError("pipe broken")
        event = SystemEvent(level=SystemLevel.INFO, message="x")
        result = await publisher.publish_to_instance("i", event)
        assert result is False

    async def test_agent_started_resilient(self, publisher, mock_ws_manager):
        mock_ws_manager.broadcast_to_all.side_effect = Exception("boom")
        result = await publisher.agent_started("a", "b")
        assert result is False

    async def test_task_progress_resilient(self, publisher, mock_ws_manager):
        mock_ws_manager.broadcast_to_all.side_effect = Exception("boom")
        result = await publisher.task_progress("t", 50)
        assert result is False

    async def test_notify_resilient(self, publisher, mock_ws_manager):
        mock_ws_manager.broadcast_to_all.side_effect = Exception("boom")
        result = await publisher.notify(title="T", body="B")
        assert result is False
