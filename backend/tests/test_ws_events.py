"""
Tests for ws_events.py — WebSocket event models, enum values, bounds, and to_ws_message.

Sentinel-value contract:
- Remove TaskProgressEvent.progress_pct le=100 → 150% progress sent to UI → progress bar breaks → test fails
- Remove CouncilEvent.confidence le=1 → confidence=2 sent → chart exceeds 100% → test fails
- Remove NotificationEvent default priority NORMAL → None priority → UI switch falls through → test fails
- Remove BaseEvent.to_ws_message() → events never serialised → WebSocket sends nothing → test fails
- Remove EventType.AGENT_ACTIVITY default on AgentActivityEvent → wrong discriminator → routing fails → test fails
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.ws_events import (
    AgentActivityEvent,
    AgentStatus,
    BaseEvent,
    CouncilEvent,
    CouncilEventType,
    EventType,
    NotificationEvent,
    NotificationPriority,
    SystemEvent,
    SystemLevel,
    TaskProgressEvent,
    TaskStage,
)


# ===========================================================================
# EventType enum
# ===========================================================================


class TestEventType:
    def test_agent_activity_value(self):
        """
        SENTINEL — AGENT_ACTIVITY must equal 'agent_activity'.
        Change → WS client switch can't match event type → events silently dropped → test fails.
        """
        assert EventType.AGENT_ACTIVITY == "agent_activity"

    def test_task_progress_value(self):
        """TASK_PROGRESS must equal 'task_progress'."""
        assert EventType.TASK_PROGRESS == "task_progress"

    def test_council_value(self):
        """COUNCIL must equal 'council'."""
        assert EventType.COUNCIL == "council"

    def test_system_value(self):
        """SYSTEM must equal 'system'."""
        assert EventType.SYSTEM == "system"

    def test_notification_value(self):
        """NOTIFICATION must equal 'notification'."""
        assert EventType.NOTIFICATION == "notification"


# ===========================================================================
# AgentStatus enum
# ===========================================================================


class TestAgentStatus:
    def test_all_values(self):
        """
        SENTINEL — all seven AgentStatus values must be correct strings.
        Rename any → UI status badges break → test fails.
        """
        assert AgentStatus.IDLE == "idle"
        assert AgentStatus.ACTIVE == "active"
        assert AgentStatus.THINKING == "thinking"
        assert AgentStatus.EXECUTING == "executing"
        assert AgentStatus.WAITING == "waiting"
        assert AgentStatus.ERROR == "error"
        assert AgentStatus.COMPLETE == "complete"


# ===========================================================================
# TaskStage enum
# ===========================================================================


class TestTaskStage:
    def test_all_values(self):
        """
        SENTINEL — all TaskStage values must match expected strings.
        Rename → progress tracker mismatch → UI shows wrong stage → test fails.
        """
        assert TaskStage.QUEUED == "queued"
        assert TaskStage.STARTING == "starting"
        assert TaskStage.PROCESSING == "processing"
        assert TaskStage.FINALIZING == "finalizing"
        assert TaskStage.COMPLETE == "complete"
        assert TaskStage.FAILED == "failed"
        assert TaskStage.CANCELLED == "cancelled"


# ===========================================================================
# CouncilEventType enum
# ===========================================================================


class TestCouncilEventType:
    def test_all_values(self):
        """All CouncilEventType values must be correct strings."""
        assert CouncilEventType.SESSION_STARTED == "session_started"
        assert CouncilEventType.PERSPECTIVE_ADDED == "perspective_added"
        assert CouncilEventType.VOTE_CAST == "vote_cast"
        assert CouncilEventType.DEBATE_ROUND == "debate_round"
        assert CouncilEventType.SYNTHESIS_READY == "synthesis_ready"
        assert CouncilEventType.SESSION_COMPLETE == "session_complete"


# ===========================================================================
# SystemLevel + NotificationPriority enums
# ===========================================================================


class TestSystemLevel:
    def test_all_severity_levels(self):
        """All five SystemLevel severity values must be correct."""
        assert SystemLevel.DEBUG == "debug"
        assert SystemLevel.INFO == "info"
        assert SystemLevel.WARNING == "warning"
        assert SystemLevel.ERROR == "error"
        assert SystemLevel.CRITICAL == "critical"


class TestNotificationPriority:
    def test_all_priority_levels(self):
        """All four NotificationPriority values must be correct."""
        assert NotificationPriority.LOW == "low"
        assert NotificationPriority.NORMAL == "normal"
        assert NotificationPriority.HIGH == "high"
        assert NotificationPriority.URGENT == "urgent"


# ===========================================================================
# BaseEvent — event_id auto-generation and to_ws_message
# ===========================================================================


class TestBaseEvent:
    def _make_system_event(self):
        return SystemEvent(level=SystemLevel.INFO, message="SENTINEL_MSG")

    def test_event_id_auto_generated(self):
        """
        SENTINEL — event_id must be auto-generated.
        Default None → WS client can't deduplicate events → test fails.
        """
        e1 = self._make_system_event()
        e2 = self._make_system_event()
        assert e1.event_id != e2.event_id

    def test_event_id_is_string(self):
        """event_id must be a string (UUID format)."""
        ev = self._make_system_event()
        assert isinstance(ev.event_id, str)
        assert len(ev.event_id) > 0

    def test_timestamp_auto_set(self):
        """timestamp must be auto-set."""
        ev = self._make_system_event()
        assert ev.timestamp is not None

    def test_to_ws_message_returns_dict(self):
        """
        SENTINEL — to_ws_message() must return a dict.
        Return None → WS send(None) → connection error → test fails.
        """
        ev = self._make_system_event()
        result = ev.to_ws_message()
        assert isinstance(result, dict)

    def test_to_ws_message_contains_event_type(self):
        """
        SENTINEL — to_ws_message() dict must contain event_type key.
        Omit it → WS client can't route event → test fails.
        """
        ev = self._make_system_event()
        result = ev.to_ws_message()
        assert "event_type" in result

    def test_to_ws_message_event_type_is_string(self):
        """
        SENTINEL — event_type in WS message must be a string, not an enum object.
        Return enum → JSON.stringify fails in browser → test fails.
        """
        ev = self._make_system_event()
        result = ev.to_ws_message()
        assert isinstance(result["event_type"], str)


# ===========================================================================
# AgentActivityEvent — defaults
# ===========================================================================


class TestAgentActivityEvent:
    def test_default_event_type_is_agent_activity(self):
        """
        SENTINEL — event_type must default to AGENT_ACTIVITY.
        Default SYSTEM → all agent events routed to system handler → UI misroutes → test fails.
        """
        ev = AgentActivityEvent(
            agent_id="SENTINEL_AGENT",
            action="thinking",
            status=AgentStatus.ACTIVE,
        )
        assert (
            ev.event_type == EventType.AGENT_ACTIVITY
            or ev.event_type == "agent_activity"
        )

    def test_fields_stored(self):
        """Required fields must be stored correctly."""
        ev = AgentActivityEvent(
            agent_id="SENTINEL_AGENT",
            action="SENTINEL_ACTION",
            status=AgentStatus.THINKING,
        )
        assert ev.agent_id == "SENTINEL_AGENT"
        assert ev.action == "SENTINEL_ACTION"


# ===========================================================================
# TaskProgressEvent — progress_pct and other bounds
# ===========================================================================


class TestTaskProgressEvent:
    def _min_event(self, **overrides) -> dict:
        base = {
            "task_id": "task-1",
            "progress_pct": 50,
            "stage": TaskStage.PROCESSING,
        }
        base.update(overrides)
        return base

    def test_progress_pct_above_100_rejected(self):
        """
        SENTINEL — progress_pct > 100 must raise ValidationError.
        Remove le=100 → 150% sent to UI → progress bar overflows → test fails.
        """
        with pytest.raises(ValidationError):
            TaskProgressEvent(**self._min_event(progress_pct=101))

    def test_progress_pct_below_0_rejected(self):
        """progress_pct < 0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            TaskProgressEvent(**self._min_event(progress_pct=-1))

    def test_progress_pct_boundaries_accepted(self):
        """0 and 100 must be accepted as progress_pct."""
        zero = TaskProgressEvent(**self._min_event(progress_pct=0))
        hundred = TaskProgressEvent(**self._min_event(progress_pct=100))
        assert zero.progress_pct == 0
        assert hundred.progress_pct == 100

    def test_eta_seconds_negative_rejected(self):
        """eta_seconds < 0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            TaskProgressEvent(**self._min_event(eta_seconds=-1))

    def test_total_steps_below_1_rejected(self):
        """
        SENTINEL — total_steps < 1 must raise ValidationError.
        Remove ge=1 → total_steps=0 → division by zero in step-progress calc → test fails.
        """
        with pytest.raises(ValidationError):
            TaskProgressEvent(**self._min_event(total_steps=0))

    def test_current_step_num_below_1_rejected(self):
        """current_step_num < 1 must raise ValidationError."""
        with pytest.raises(ValidationError):
            TaskProgressEvent(**self._min_event(current_step_num=0))

    def test_default_event_type_is_task_progress(self):
        """event_type must default to TASK_PROGRESS."""
        ev = TaskProgressEvent(**self._min_event())
        assert (
            ev.event_type == EventType.TASK_PROGRESS or ev.event_type == "task_progress"
        )


# ===========================================================================
# CouncilEvent — confidence and round_number bounds
# ===========================================================================


class TestCouncilEvent:
    def _min_event(self, **overrides) -> dict:
        base = {
            "council_session_id": "session-1",
            "event": CouncilEventType.PERSPECTIVE_ADDED,
        }
        base.update(overrides)
        return base

    def test_confidence_above_1_rejected(self):
        """
        SENTINEL — confidence > 1.0 must raise ValidationError.
        Remove le=1 → confidence=2.0 sent → confidence bar exceeds 100% → test fails.
        """
        with pytest.raises(ValidationError):
            CouncilEvent(**self._min_event(confidence=1.1))

    def test_confidence_below_0_rejected(self):
        """confidence < 0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            CouncilEvent(**self._min_event(confidence=-0.1))

    def test_confidence_boundary_accepted(self):
        """0.0 and 1.0 must be accepted as confidence."""
        low = CouncilEvent(**self._min_event(confidence=0.0))
        high = CouncilEvent(**self._min_event(confidence=1.0))
        assert low.confidence == pytest.approx(0.0)
        assert high.confidence == pytest.approx(1.0)

    def test_round_number_below_1_rejected(self):
        """
        SENTINEL — round_number < 1 must raise ValidationError.
        Remove ge=1 → 'Round 0' displayed → test fails.
        """
        with pytest.raises(ValidationError):
            CouncilEvent(**self._min_event(round_number=0))

    def test_default_event_type_is_council(self):
        """event_type must default to COUNCIL."""
        ev = CouncilEvent(**self._min_event())
        assert ev.event_type == EventType.COUNCIL or ev.event_type == "council"


# ===========================================================================
# NotificationEvent — priority default and auto_dismiss_ms bounds
# ===========================================================================


class TestNotificationEvent:
    def _min_event(self, **overrides) -> dict:
        base = {
            "title": "SENTINEL_TITLE",
            "body": "SENTINEL_BODY",
        }
        base.update(overrides)
        return base

    def test_default_priority_is_normal(self):
        """
        SENTINEL — default priority must be NORMAL.
        Default URGENT → all new notifications interrupt user → test fails.
        """
        ev = NotificationEvent(**self._min_event())
        assert ev.priority == NotificationPriority.NORMAL or ev.priority == "normal"

    def test_default_dismissible_is_true(self):
        """
        SENTINEL — dismissible must default to True.
        Default False → all notifications permanent → UI fills up → test fails.
        """
        ev = NotificationEvent(**self._min_event())
        assert ev.dismissible is True

    def test_auto_dismiss_ms_negative_rejected(self):
        """
        SENTINEL — auto_dismiss_ms < 0 must raise ValidationError.
        Remove ge=0 → -1ms auto-dismiss → notification vanishes instantly → test fails.
        """
        with pytest.raises(ValidationError):
            NotificationEvent(**self._min_event(auto_dismiss_ms=-1))

    def test_auto_dismiss_ms_zero_accepted(self):
        """auto_dismiss_ms=0 must be accepted."""
        ev = NotificationEvent(**self._min_event(auto_dismiss_ms=0))
        assert ev.auto_dismiss_ms == 0

    def test_to_ws_message_includes_title_and_body(self):
        """
        SENTINEL — serialised message must include title and body.
        Omit fields → UI renders blank notification → test fails.
        """
        ev = NotificationEvent(title="SENTINEL_TITLE", body="SENTINEL_BODY")
        msg = ev.to_ws_message()
        assert msg["title"] == "SENTINEL_TITLE"
        assert msg["body"] == "SENTINEL_BODY"
