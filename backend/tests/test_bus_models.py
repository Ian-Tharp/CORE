"""
Tests for bus_models.py — enum values, field bounds, and defaults.

Sentinel-value contract:
- Remove BusMessage.sender_id min_length=1 → empty sender_id stored → routing breaks → test fails
- Remove BusMessage.ttl_seconds ge=1 → ttl=0 stored → messages expire immediately → test fails
- Remove WebhookConfig.max_retries le=10 → 1000 retries → flood attack → test fails
- Remove WebhookConfig.timeout_ms ge=1000 → 1ms timeout → all webhooks fail → test fails
- Remove ExternalAgentRegistration.name max_length=255 → DB column overflow → test fails
- Remove DeliveryReceipt default PENDING → untracked delivery status → receipts show None → test fails
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from uuid import uuid4

from app.models.bus_models import (
    BroadcastRequest,
    BusMessage,
    BusMetrics,
    BusScope,
    DeliveryReceipt,
    DeliveryStatus,
    ExternalAgentRegistration,
    MessagePriority,
    MessageType,
    Subscription,
    SubscriptionCreate,
    WebhookConfig,
    WebhookIncoming,
)


# ===========================================================================
# MessageType enum
# ===========================================================================


class TestMessageType:
    def test_task_request_value(self):
        """
        SENTINEL — TASK_REQUEST must equal 'task_request'.
        Change → message routing switch misroutes tasks → test fails.
        """
        assert MessageType.TASK_REQUEST == "task_request"

    def test_task_result_value(self):
        """TASK_RESULT must equal 'task_result'."""
        assert MessageType.TASK_RESULT == "task_result"

    def test_heartbeat_value(self):
        """HEARTBEAT must equal 'heartbeat'."""
        assert MessageType.HEARTBEAT == "heartbeat"

    def test_broadcast_value(self):
        """BROADCAST must equal 'broadcast'."""
        assert MessageType.BROADCAST == "broadcast"

    def test_all_eight_types_defined(self):
        """All 10 message types must be defined in the enum."""
        expected = {
            "task_request",
            "task_result",
            "help_request",
            "status_update",
            "capability_query",
            "capability_response",
            "heartbeat",
            "broadcast",
            "catalyst_phase_complete",
            "level_transition",
        }
        actual = {m.value for m in MessageType}
        assert expected == actual


# ===========================================================================
# MessagePriority enum
# ===========================================================================


class TestMessagePriority:
    def test_all_priority_values(self):
        """
        SENTINEL — all four priorities must have correct string values.
        Change 'normal' → bus routing priority maths break → test fails.
        """
        assert MessagePriority.LOW == "low"
        assert MessagePriority.NORMAL == "normal"
        assert MessagePriority.HIGH == "high"
        assert MessagePriority.URGENT == "urgent"


# ===========================================================================
# DeliveryStatus enum
# ===========================================================================


class TestDeliveryStatus:
    def test_all_delivery_status_values(self):
        """All five DeliveryStatus values must be correct strings."""
        assert DeliveryStatus.PENDING == "pending"
        assert DeliveryStatus.DELIVERED == "delivered"
        assert DeliveryStatus.READ == "read"
        assert DeliveryStatus.FAILED == "failed"
        assert DeliveryStatus.QUEUED == "queued"


# ===========================================================================
# BusMessage — sender_id, priority, ttl_seconds
# ===========================================================================


class TestBusMessage:
    def _min_msg(self, **overrides) -> dict:
        base = {
            "sender_id": "SENTINEL_SENDER",
            "message_type": MessageType.STATUS_UPDATE,
        }
        base.update(overrides)
        return base

    def test_empty_sender_id_rejected(self):
        """
        SENTINEL — empty sender_id must raise ValidationError.
        Remove min_length=1 → empty string stored → routing can't identify source → test fails.
        """
        with pytest.raises(ValidationError):
            BusMessage(**self._min_msg(sender_id=""))

    def test_default_priority_is_normal(self):
        """
        SENTINEL — default priority must be NORMAL.
        Default URGENT → all routine messages treated as urgent → system overloaded → test fails.
        """
        msg = BusMessage(**self._min_msg())
        assert msg.priority == MessagePriority.NORMAL

    def test_id_auto_generated(self):
        """id must be auto-generated string."""
        msg = BusMessage(**self._min_msg())
        assert isinstance(msg.id, str)
        assert len(msg.id) > 0

    def test_ttl_seconds_zero_rejected(self):
        """
        SENTINEL — ttl_seconds=0 must raise ValidationError.
        Remove ge=1 → zero-TTL messages expire immediately → agents never receive them → test fails.
        """
        with pytest.raises(ValidationError):
            BusMessage(**self._min_msg(ttl_seconds=0))

    def test_ttl_seconds_none_accepted(self):
        """ttl_seconds=None must be accepted (no expiry)."""
        msg = BusMessage(**self._min_msg(ttl_seconds=None))
        assert msg.ttl_seconds is None

    def test_payload_defaults_empty(self):
        """payload defaults to empty dict."""
        msg = BusMessage(**self._min_msg())
        assert msg.payload == {}

    def test_recipients_default_empty(self):
        """recipients defaults to empty list."""
        msg = BusMessage(**self._min_msg())
        assert msg.recipients == []

    def test_scope_defaults_none(self):
        """scope defaults to None (no MMCNC scoping)."""
        msg = BusMessage(**self._min_msg())
        assert msg.scope is None


# ===========================================================================
# Subscription — agent_id min_length
# ===========================================================================


class TestSubscription:
    def test_empty_agent_id_rejected(self):
        """
        SENTINEL — empty agent_id must raise ValidationError.
        Remove min_length=1 → blank agent subscribes → all messages broadcast to blank ID → test fails.
        """
        with pytest.raises(ValidationError):
            Subscription(agent_id="", message_types=[MessageType.HEARTBEAT])

    def test_valid_subscription_accepted(self):
        """Valid subscription with agent_id must be accepted."""
        sub = Subscription(agent_id="SENTINEL_AGENT")
        assert sub.agent_id == "SENTINEL_AGENT"
        assert sub.message_types == []
        assert sub.topics == []

    def test_id_auto_generated(self):
        """id must be auto-generated."""
        sub = Subscription(agent_id="agent-1")
        assert isinstance(sub.id, str)
        assert len(sub.id) > 0


# ===========================================================================
# WebhookConfig — all three numeric bounds
# ===========================================================================


class TestWebhookConfig:
    def _min_cfg(self, **overrides) -> dict:
        base = {"url": "https://example.com/webhook"}
        base.update(overrides)
        return base

    def test_max_retries_above_10_rejected(self):
        """
        SENTINEL — max_retries > 10 must raise ValidationError.
        Remove le=10 → 1000 retries flood external servers → test fails.
        """
        with pytest.raises(ValidationError):
            WebhookConfig(**self._min_cfg(max_retries=11))

    def test_max_retries_negative_rejected(self):
        """max_retries < 0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            WebhookConfig(**self._min_cfg(max_retries=-1))

    def test_default_max_retries_is_3(self):
        """
        SENTINEL — default max_retries must be 3.
        Default 0 → no retries on transient failures → messages lost → test fails.
        """
        cfg = WebhookConfig(**self._min_cfg())
        assert cfg.max_retries == 3

    def test_retry_backoff_base_ms_below_100_rejected(self):
        """
        SENTINEL — retry_backoff_base_ms < 100 must raise ValidationError.
        Remove ge=100 → 1ms backoff → thundering herd → test fails.
        """
        with pytest.raises(ValidationError):
            WebhookConfig(**self._min_cfg(retry_backoff_base_ms=99))

    def test_retry_backoff_base_ms_above_60000_rejected(self):
        """retry_backoff_base_ms > 60000 must raise ValidationError."""
        with pytest.raises(ValidationError):
            WebhookConfig(**self._min_cfg(retry_backoff_base_ms=60001))

    def test_default_retry_backoff_base_ms_is_1000(self):
        """default retry_backoff_base_ms must be 1000ms."""
        cfg = WebhookConfig(**self._min_cfg())
        assert cfg.retry_backoff_base_ms == 1000

    def test_timeout_ms_below_1000_rejected(self):
        """
        SENTINEL — timeout_ms < 1000 must raise ValidationError.
        Remove ge=1000 → 1ms timeout → all webhooks fail immediately → test fails.
        """
        with pytest.raises(ValidationError):
            WebhookConfig(**self._min_cfg(timeout_ms=999))

    def test_timeout_ms_above_30000_rejected(self):
        """timeout_ms > 30000 must raise ValidationError."""
        with pytest.raises(ValidationError):
            WebhookConfig(**self._min_cfg(timeout_ms=30001))

    def test_default_timeout_ms_is_5000(self):
        """default timeout_ms must be 5000ms (5 seconds)."""
        cfg = WebhookConfig(**self._min_cfg())
        assert cfg.timeout_ms == 5000


# ===========================================================================
# ExternalAgentRegistration — name bounds
# ===========================================================================


class TestExternalAgentRegistration:
    def _min_reg(self, **overrides) -> dict:
        base = {
            "agent_id": "SENTINEL_ID",
            "name": "SENTINEL_AGENT",
            "webhook": WebhookConfig(url="https://example.com/wh"),
        }
        base.update(overrides)
        return base

    def test_empty_name_rejected(self):
        """
        SENTINEL — empty name must raise ValidationError.
        Remove min_length=1 → blank-named agent registered → UI shows nothing → test fails.
        """
        with pytest.raises(ValidationError):
            ExternalAgentRegistration(**self._min_reg(name=""))

    def test_name_too_long_rejected(self):
        """
        SENTINEL — name > 255 chars must raise ValidationError.
        Remove max_length=255 → 500-char name stored → DB column overflow → test fails.
        """
        with pytest.raises(ValidationError):
            ExternalAgentRegistration(**self._min_reg(name="n" * 256))

    def test_name_exactly_255_accepted(self):
        """name of exactly 255 characters must be accepted (boundary)."""
        reg = ExternalAgentRegistration(**self._min_reg(name="n" * 255))
        assert len(reg.name) == 255

    def test_empty_agent_id_rejected(self):
        """agent_id min_length=1 must be enforced."""
        with pytest.raises(ValidationError):
            ExternalAgentRegistration(**self._min_reg(agent_id=""))

    def test_capabilities_default_empty(self):
        """capabilities defaults to empty list."""
        reg = ExternalAgentRegistration(**self._min_reg())
        assert reg.capabilities == []


# ===========================================================================
# DeliveryReceipt — default status
# ===========================================================================


class TestDeliveryReceipt:
    def test_default_status_is_pending(self):
        """
        SENTINEL — default status must be PENDING.
        Default DELIVERED → undelivered messages marked delivered → test fails.
        """
        receipt = DeliveryReceipt(message_id="msg-1", recipient_id="agent-1")
        assert receipt.status == DeliveryStatus.PENDING

    def test_id_auto_generated(self):
        """id must be auto-generated."""
        r1 = DeliveryReceipt(message_id="m", recipient_id="a")
        r2 = DeliveryReceipt(message_id="m", recipient_id="a")
        assert r1.id != r2.id

    def test_delivered_at_defaults_none(self):
        """delivered_at defaults to None before delivery."""
        receipt = DeliveryReceipt(message_id="m", recipient_id="a")
        assert receipt.delivered_at is None


# ===========================================================================
# BusMetrics — defaults
# ===========================================================================


class TestBusMetrics:
    def test_all_counts_default_zero(self):
        """
        SENTINEL — all integer counters must default to 0.
        Default 1 → fresh metrics show false positives → alerting fires → test fails.
        """
        metrics = BusMetrics()
        assert metrics.total_messages_published == 0
        assert metrics.total_messages_delivered == 0
        assert metrics.total_messages_failed == 0
        assert metrics.total_messages_queued == 0
        assert metrics.active_subscriptions == 0
        assert metrics.external_agents_count == 0

    def test_avg_delivery_latency_defaults_none(self):
        """avg_delivery_latency_ms defaults to None (no data yet)."""
        metrics = BusMetrics()
        assert metrics.avg_delivery_latency_ms is None


# ===========================================================================
# BusScope — all fields optional
# ===========================================================================


class TestBusScope:
    def test_all_fields_optional(self):
        """
        SENTINEL — BusScope must allow creation with no arguments.
        Make any field required → broadcast messages need scope → broadcast breaks → test fails.
        """
        scope = BusScope()
        assert scope.macrocosm_id is None
        assert scope.microcosm_id is None
        assert scope.cluster_id is None

    def test_cluster_id_stored(self):
        """cluster_id must be stored when provided."""
        scope = BusScope(cluster_id="SENTINEL_CLUSTER")
        assert scope.cluster_id == "SENTINEL_CLUSTER"
