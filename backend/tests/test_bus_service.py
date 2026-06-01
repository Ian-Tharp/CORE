"""
Comprehensive tests for bus_service — inter-agent communication bus.

Tests cover:
- publish: persistence, recipient resolution, delivery routing, self-exclusion
- broadcast: topic-based delivery via publish
- subscribe / unsubscribe / get_subscriptions
- _deliver: internal WS, external webhook, offline queue fallback
- deliver_to_external: HMAC signing, retries, backoff
- _resolve_subscription_targets: type/topic/scope matching
- _scopes_overlap: all scope combinations
- _parse_mentions: @agent extraction
- request_response: correlation-based polling
- get_metrics: aggregation from repo
- external agent CRUD
- queue_for_offline / drain_queue
"""

from __future__ import annotations

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, ANY

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
)
from app.services import bus_service


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def _patch_repo():
    """Patch the bus_repository for every test."""
    with patch.object(bus_service, "repo") as mock_repo:
        # Sensible defaults — tests override as needed
        mock_repo.store_message = AsyncMock()
        mock_repo.get_all_subscriptions = AsyncMock(return_value=[])
        mock_repo.get_external_agent = AsyncMock(return_value=None)
        mock_repo.create_delivery_receipt = AsyncMock()
        mock_repo.enqueue_offline = AsyncMock()
        mock_repo.drain_offline_queue = AsyncMock(return_value=[])
        mock_repo.create_subscription = AsyncMock()
        mock_repo.delete_subscription = AsyncMock(return_value=True)
        mock_repo.get_subscriptions_for_agent = AsyncMock(return_value=[])
        mock_repo.register_external_agent = AsyncMock(
            return_value={"agent_id": "ext-1"}
        )
        mock_repo.deregister_external_agent = AsyncMock(return_value=True)
        mock_repo.list_external_agents = AsyncMock(return_value=[])
        mock_repo.count_messages = AsyncMock(return_value=42)
        mock_repo.count_messages_by_type = AsyncMock(return_value={"broadcast": 10})
        mock_repo.count_messages_by_priority = AsyncMock(return_value={"normal": 30})
        mock_repo.count_receipts_by_status = AsyncMock(
            return_value={"delivered": 35, "failed": 2}
        )
        mock_repo.count_subscriptions = AsyncMock(return_value=5)
        mock_repo.count_external_agents = AsyncMock(return_value=1)
        mock_repo.count_offline_queued = AsyncMock(return_value=3)
        mock_repo.avg_delivery_latency_ms = AsyncMock(return_value=12.5)
        mock_repo.get_messages_by_correlation = AsyncMock(return_value=[])
        yield mock_repo


@pytest.fixture(autouse=True)
def _patch_ws():
    """Patch the agent WebSocket manager."""
    with patch.object(bus_service, "agent_ws_manager") as mock_ws:
        mock_ws.send_message = AsyncMock(return_value=False)  # offline by default
        yield mock_ws


def _msg(**overrides) -> BusMessage:
    defaults = dict(
        sender_id="agent-sender",
        recipients=["agent-target"],
        message_type=MessageType.TASK_REQUEST,
        payload={"text": "hello"},
    )
    defaults.update(overrides)
    return BusMessage(**defaults)


# =============================================================================
# PUBLISH
# =============================================================================


class TestPublish:
    @pytest.mark.asyncio
    async def test_persists_message(self, _patch_repo):
        msg = _msg()
        await bus_service.publish(msg)
        _patch_repo.store_message.assert_called_once()
        call_kw = _patch_repo.store_message.call_args[1]
        assert call_kw["message_id"] == msg.id
        assert call_kw["sender_id"] == "agent-sender"

    @pytest.mark.asyncio
    async def test_excludes_sender_from_delivery(self, _patch_repo, _patch_ws):
        msg = _msg(recipients=["agent-sender", "agent-other"])
        _patch_ws.send_message = AsyncMock(return_value=True)
        receipt = await bus_service.publish(msg)
        # sender should NOT appear in delivery targets
        ws_targets = [c[0][0] for c in _patch_ws.send_message.call_args_list]
        assert "agent-sender" not in ws_targets

    @pytest.mark.asyncio
    async def test_delivers_to_direct_recipients(self, _patch_ws):
        _patch_ws.send_message = AsyncMock(return_value=True)
        msg = _msg(recipients=["a1", "a2"])
        receipt = await bus_service.publish(msg)
        assert receipt.status == DeliveryStatus.DELIVERED

    @pytest.mark.asyncio
    async def test_includes_subscription_matches(self, _patch_repo, _patch_ws):
        _patch_repo.get_all_subscriptions = AsyncMock(
            return_value=[
                {
                    "agent_id": "sub-agent",
                    "message_types": ["task_request"],
                    "topics": [],
                    "scope": None,
                },
            ]
        )
        _patch_ws.send_message = AsyncMock(return_value=True)
        msg = _msg(recipients=[])
        await bus_service.publish(msg)
        ws_targets = [c[0][0] for c in _patch_ws.send_message.call_args_list]
        assert "sub-agent" in ws_targets

    @pytest.mark.asyncio
    async def test_includes_mentioned_agents(self, _patch_ws):
        _patch_ws.send_message = AsyncMock(return_value=True)
        msg = _msg(recipients=[], payload={"text": "Hey @agent-bob check this"})
        await bus_service.publish(msg)
        ws_targets = [c[0][0] for c in _patch_ws.send_message.call_args_list]
        assert "agent-bob" in ws_targets

    @pytest.mark.asyncio
    async def test_all_failed_returns_failed_status(self, _patch_repo):
        _patch_repo.get_external_agent = AsyncMock(
            return_value={
                "webhook_url": "http://x",
                "webhook_max_retries": 0,
                "webhook_retry_backoff_base_ms": 100,
                "webhook_timeout_ms": 1000,
            }
        )
        with patch(
            "app.services.bus_service.deliver_to_external",
            new_callable=AsyncMock,
            return_value=False,
        ):
            msg = _msg()
            receipt = await bus_service.publish(msg)
            assert receipt.status == DeliveryStatus.FAILED

    @pytest.mark.asyncio
    async def test_all_queued_returns_queued_status(self, _patch_ws):
        _patch_ws.send_message = AsyncMock(return_value=False)  # offline
        msg = _msg()
        receipt = await bus_service.publish(msg)
        assert receipt.status == DeliveryStatus.QUEUED

    @pytest.mark.asyncio
    async def test_no_targets_returns_delivered(self):
        msg = _msg(recipients=[], payload={"text": "no mentions"})
        receipt = await bus_service.publish(msg)
        assert receipt.status == DeliveryStatus.DELIVERED


# =============================================================================
# BROADCAST
# =============================================================================


class TestBroadcast:
    @pytest.mark.asyncio
    async def test_creates_message_and_publishes(self, _patch_repo):
        req = BroadcastRequest(
            sender_id="broadcaster", topic="alerts", payload={"level": "warn"}
        )
        receipt = await bus_service.broadcast(req)
        _patch_repo.store_message.assert_called_once()
        call_kw = _patch_repo.store_message.call_args[1]
        assert call_kw["sender_id"] == "broadcaster"
        assert call_kw["topic"] == "alerts"


# =============================================================================
# SUBSCRIPTIONS
# =============================================================================


class TestSubscriptions:
    @pytest.mark.asyncio
    async def test_subscribe_creates_and_returns(self, _patch_repo):
        sub_req = SubscriptionCreate(
            agent_id="agent-1",
            message_types=[MessageType.TASK_REQUEST],
            topics=["research"],
        )
        result = await bus_service.subscribe("agent-1", sub_req)
        assert isinstance(result, Subscription)
        assert result.agent_id == "agent-1"
        assert result.topics == ["research"]
        _patch_repo.create_subscription.assert_called_once()

    @pytest.mark.asyncio
    async def test_unsubscribe_delegates_to_repo(self, _patch_repo):
        ok = await bus_service.unsubscribe("agent-1", "sub-123")
        assert ok is True
        _patch_repo.delete_subscription.assert_called_once_with("sub-123")

    @pytest.mark.asyncio
    async def test_get_subscriptions_parses_rows(self, _patch_repo):
        _patch_repo.get_subscriptions_for_agent = AsyncMock(
            return_value=[
                {
                    "subscription_id": "s1",
                    "agent_id": "a1",
                    "message_types": ["broadcast"],
                    "topics": ["news"],
                    "scope": None,
                },
                {
                    "subscription_id": "s2",
                    "agent_id": "a1",
                    "message_types": [],
                    "topics": [],
                    "scope": {"macrocosm_id": "m1"},
                },
            ]
        )
        subs = await bus_service.get_subscriptions("a1")
        assert len(subs) == 2
        assert subs[1].scope.macrocosm_id == "m1"


# =============================================================================
# DELIVER INTERNAL / EXTERNAL / QUEUE
# =============================================================================


class TestDeliverInternal:
    @pytest.mark.asyncio
    async def test_sends_via_ws_manager(self, _patch_ws):
        _patch_ws.send_message = AsyncMock(return_value=True)
        ok = await bus_service.deliver_to_internal("agent-1", _msg())
        assert ok is True
        _patch_ws.send_message.assert_called_once()


class TestDeliverExternal:
    @pytest.mark.asyncio
    async def test_no_registration_returns_false(self, _patch_repo):
        _patch_repo.get_external_agent = AsyncMock(return_value=None)
        ok = await bus_service.deliver_to_external("ghost", _msg())
        assert ok is False

    @pytest.mark.asyncio
    async def test_successful_webhook(self, _patch_repo):
        _patch_repo.get_external_agent = AsyncMock(
            return_value={
                "webhook_url": "https://example.com/hook",
                "webhook_secret": None,
                "webhook_max_retries": 0,
                "webhook_retry_backoff_base_ms": 100,
                "webhook_timeout_ms": 5000,
            }
        )
        mock_resp = MagicMock(status_code=200)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = AsyncMock(return_value=mock_resp)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            ok = await bus_service.deliver_to_external("ext-1", _msg())
            assert ok is True

    @pytest.mark.asyncio
    async def test_hmac_signature_added_when_secret(self, _patch_repo):
        _patch_repo.get_external_agent = AsyncMock(
            return_value={
                "webhook_url": "https://example.com/hook",
                "webhook_secret": "my-secret",
                "webhook_max_retries": 0,
                "webhook_retry_backoff_base_ms": 100,
                "webhook_timeout_ms": 5000,
            }
        )
        mock_resp = MagicMock(status_code=200)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = AsyncMock(return_value=mock_resp)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            await bus_service.deliver_to_external("ext-1", _msg())
            call_kwargs = instance.post.call_args[1]
            assert "X-Bus-Signature" in call_kwargs["headers"]


class TestQueueForOffline:
    @pytest.mark.asyncio
    async def test_enqueues_message(self, _patch_repo):
        msg = _msg()
        await bus_service.queue_for_offline("agent-offline", msg)
        _patch_repo.enqueue_offline.assert_called_once_with("agent-offline", msg.id)

    @pytest.mark.asyncio
    async def test_drain_queue_delegates(self, _patch_repo):
        _patch_repo.drain_offline_queue = AsyncMock(return_value=[{"message_id": "m1"}])
        result = await bus_service.drain_queue("agent-1")
        assert len(result) == 1


# =============================================================================
# _deliver routing
# =============================================================================


class TestDeliverRouting:
    @pytest.mark.asyncio
    async def test_routes_to_external_first(self, _patch_repo):
        _patch_repo.get_external_agent = AsyncMock(
            return_value={
                "webhook_url": "https://x.com",
                "webhook_max_retries": 0,
                "webhook_retry_backoff_base_ms": 100,
                "webhook_timeout_ms": 1000,
            }
        )
        with patch(
            "app.services.bus_service.deliver_to_external",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_ext:
            receipt = await bus_service._deliver("ext-agent", _msg())
            assert receipt.status == DeliveryStatus.DELIVERED
            mock_ext.assert_called_once()

    @pytest.mark.asyncio
    async def test_falls_back_to_ws(self, _patch_repo, _patch_ws):
        _patch_repo.get_external_agent = AsyncMock(return_value=None)
        _patch_ws.send_message = AsyncMock(return_value=True)
        receipt = await bus_service._deliver("internal-agent", _msg())
        assert receipt.status == DeliveryStatus.DELIVERED

    @pytest.mark.asyncio
    async def test_queues_when_offline(self, _patch_repo, _patch_ws):
        _patch_repo.get_external_agent = AsyncMock(return_value=None)
        _patch_ws.send_message = AsyncMock(return_value=False)
        receipt = await bus_service._deliver("offline-agent", _msg())
        assert receipt.status == DeliveryStatus.QUEUED
        _patch_repo.enqueue_offline.assert_called_once()


# =============================================================================
# SCOPE OVERLAP
# =============================================================================


class TestScopesOverlap:
    def test_no_msg_scope_matches_all(self):
        assert bus_service._scopes_overlap(None, BusScope(macrocosm_id="m1")) is True

    def test_no_sub_scope_matches_all(self):
        assert bus_service._scopes_overlap(BusScope(macrocosm_id="m1"), None) is True

    def test_both_none(self):
        assert bus_service._scopes_overlap(None, None) is True

    def test_matching_macrocosm(self):
        msg = BusScope(macrocosm_id="m1", microcosm_id="mic1")
        sub = BusScope(macrocosm_id="m1")
        assert bus_service._scopes_overlap(msg, sub) is True

    def test_mismatched_macrocosm(self):
        msg = BusScope(macrocosm_id="m1")
        sub = BusScope(macrocosm_id="m2")
        assert bus_service._scopes_overlap(msg, sub) is False

    def test_subscriber_cluster_matches_exact(self):
        msg = BusScope(macrocosm_id="m1", microcosm_id="mic1", cluster_id="c1")
        sub = BusScope(macrocosm_id="m1", microcosm_id="mic1", cluster_id="c1")
        assert bus_service._scopes_overlap(msg, sub) is True

    def test_subscriber_cluster_rejects_different(self):
        msg = BusScope(macrocosm_id="m1", microcosm_id="mic1", cluster_id="c1")
        sub = BusScope(macrocosm_id="m1", microcosm_id="mic1", cluster_id="c2")
        assert bus_service._scopes_overlap(msg, sub) is False

    def test_subscriber_broader_scope_matches(self):
        """Subscriber at macrocosm level should see cluster-scoped messages."""
        msg = BusScope(macrocosm_id="m1", microcosm_id="mic1", cluster_id="c1")
        sub = BusScope(macrocosm_id="m1")
        assert bus_service._scopes_overlap(msg, sub) is True

    def test_msg_without_cluster_vs_sub_with_cluster(self):
        """Msg scoped to macrocosm, sub scoped to cluster — cluster field doesn't match."""
        msg = BusScope(macrocosm_id="m1")
        sub = BusScope(macrocosm_id="m1", cluster_id="c1")
        # msg has no cluster_id (None), sub wants c1 → None != "c1" → False
        assert bus_service._scopes_overlap(msg, sub) is False


# =============================================================================
# PARSE MENTIONS
# =============================================================================


class TestParseMentions:
    def test_basic_mention(self):
        assert bus_service._parse_mentions("Hello @agent-bob") == {"agent-bob"}

    def test_multiple_mentions(self):
        result = bus_service._parse_mentions("@alice and @bob discuss")
        assert result == {"alice", "bob"}

    def test_no_mentions(self):
        assert bus_service._parse_mentions("no mentions here") == set()

    def test_empty_string(self):
        assert bus_service._parse_mentions("") == set()

    def test_none_input(self):
        assert bus_service._parse_mentions(None) == set()

    def test_underscore_in_name(self):
        assert bus_service._parse_mentions("@agent_v2") == {"agent_v2"}


# =============================================================================
# RESOLVE SUBSCRIPTION TARGETS
# =============================================================================


class TestResolveSubscriptionTargets:
    @pytest.mark.asyncio
    async def test_type_match(self, _patch_repo):
        _patch_repo.get_all_subscriptions = AsyncMock(
            return_value=[
                {
                    "agent_id": "listener",
                    "message_types": ["task_request"],
                    "topics": [],
                    "scope": None,
                },
            ]
        )
        msg = _msg(message_type=MessageType.TASK_REQUEST)
        result = await bus_service._resolve_subscription_targets(msg)
        assert "listener" in result

    @pytest.mark.asyncio
    async def test_type_mismatch_excluded(self, _patch_repo):
        _patch_repo.get_all_subscriptions = AsyncMock(
            return_value=[
                {
                    "agent_id": "listener",
                    "message_types": ["heartbeat"],
                    "topics": [],
                    "scope": None,
                },
            ]
        )
        msg = _msg(message_type=MessageType.TASK_REQUEST)
        result = await bus_service._resolve_subscription_targets(msg)
        assert "listener" not in result

    @pytest.mark.asyncio
    async def test_topic_match(self, _patch_repo):
        _patch_repo.get_all_subscriptions = AsyncMock(
            return_value=[
                {
                    "agent_id": "listener",
                    "message_types": [],
                    "topics": ["research"],
                    "scope": None,
                },
            ]
        )
        msg = _msg(topic="research")
        result = await bus_service._resolve_subscription_targets(msg)
        assert "listener" in result

    @pytest.mark.asyncio
    async def test_sender_excluded_from_own_subscriptions(self, _patch_repo):
        _patch_repo.get_all_subscriptions = AsyncMock(
            return_value=[
                {
                    "agent_id": "agent-sender",
                    "message_types": [],
                    "topics": [],
                    "scope": None,
                },
            ]
        )
        msg = _msg(sender_id="agent-sender")
        result = await bus_service._resolve_subscription_targets(msg)
        assert "agent-sender" not in result

    @pytest.mark.asyncio
    async def test_scope_filtering(self, _patch_repo):
        _patch_repo.get_all_subscriptions = AsyncMock(
            return_value=[
                {
                    "agent_id": "scoped",
                    "message_types": [],
                    "topics": [],
                    "scope": {"macrocosm_id": "m1"},
                },
                {
                    "agent_id": "wrong-scope",
                    "message_types": [],
                    "topics": [],
                    "scope": {"macrocosm_id": "m2"},
                },
            ]
        )
        msg = _msg(scope=BusScope(macrocosm_id="m1"))
        # Need to set scope on the message object
        msg.scope = BusScope(macrocosm_id="m1")
        result = await bus_service._resolve_subscription_targets(msg)
        assert "scoped" in result
        assert "wrong-scope" not in result

    @pytest.mark.asyncio
    async def test_empty_types_and_topics_matches_all(self, _patch_repo):
        _patch_repo.get_all_subscriptions = AsyncMock(
            return_value=[
                {
                    "agent_id": "wildcard",
                    "message_types": [],
                    "topics": [],
                    "scope": None,
                },
            ]
        )
        msg = _msg()
        result = await bus_service._resolve_subscription_targets(msg)
        assert "wildcard" in result


# =============================================================================
# EXTERNAL AGENT CRUD
# =============================================================================


class TestExternalAgentCrud:
    @pytest.mark.asyncio
    async def test_register(self, _patch_repo):
        reg = ExternalAgentRegistration(
            agent_id="vigil",
            name="Vigil",
            capabilities=["chat"],
            webhook=WebhookConfig(url="https://example.com/hook"),
        )
        result = await bus_service.register_external_agent(reg)
        _patch_repo.register_external_agent.assert_called_once()
        assert result["agent_id"] == "ext-1"

    @pytest.mark.asyncio
    async def test_get(self, _patch_repo):
        _patch_repo.get_external_agent = AsyncMock(return_value={"agent_id": "vigil"})
        result = await bus_service.get_external_agent("vigil")
        assert result["agent_id"] == "vigil"

    @pytest.mark.asyncio
    async def test_deregister(self, _patch_repo):
        ok = await bus_service.deregister_external_agent("vigil")
        assert ok is True

    @pytest.mark.asyncio
    async def test_list(self, _patch_repo):
        _patch_repo.list_external_agents = AsyncMock(return_value=[{"agent_id": "v1"}])
        result = await bus_service.list_external_agents()
        assert len(result) == 1


# =============================================================================
# METRICS
# =============================================================================


class TestMetrics:
    @pytest.mark.asyncio
    async def test_aggregates_from_repo(self, _patch_repo):
        metrics = await bus_service.get_metrics()
        assert isinstance(metrics, BusMetrics)
        assert metrics.total_messages_published == 42
        assert metrics.total_messages_delivered == 35
        assert metrics.total_messages_failed == 2
        assert metrics.total_messages_queued == 3
        assert metrics.active_subscriptions == 5
        assert metrics.external_agents_count == 1
        assert metrics.avg_delivery_latency_ms == 12.5
        assert metrics.messages_by_type == {"broadcast": 10}
        assert metrics.messages_by_priority == {"normal": 30}


# =============================================================================
# REQUEST-RESPONSE
# =============================================================================


class TestRequestResponse:
    @pytest.mark.asyncio
    async def test_returns_none_on_timeout(self, _patch_repo):
        msg = _msg(correlation_id="corr-1")
        result = await bus_service.request_response(msg, timeout_ms=300)
        assert result is None

    @pytest.mark.asyncio
    async def test_finds_correlated_reply(self, _patch_repo):
        msg = _msg(correlation_id="corr-1")
        _patch_repo.get_messages_by_correlation = AsyncMock(
            return_value=[
                {
                    "message_id": "reply-1",
                    "sender_id": "responder",
                    "recipients": ["agent-sender"],
                    "message_type": "task_result",
                    "topic": None,
                    "payload": {"answer": 42},
                    "priority": "normal",
                    "correlation_id": "corr-1",
                    "reply_to": msg.id,
                }
            ]
        )
        result = await bus_service.request_response(msg, timeout_ms=500)
        assert result is not None
        assert result.sender_id == "responder"
        assert result.payload["answer"] == 42

    @pytest.mark.asyncio
    async def test_generates_correlation_id_if_missing(self, _patch_repo):
        msg = _msg(correlation_id=None)
        await bus_service.request_response(msg, timeout_ms=300)
        assert msg.correlation_id is not None
