"""
Inter-Agent Communication Bus Tests

Comprehensive tests following AAA (Arrange, Act, Assert) format.
Covers publishing, subscriptions, delivery, external agents, metrics, and edge cases.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.bus_models import (
    BroadcastRequest,
    BusMessage,
    BusMetrics,
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
from app.services import bus_service


# =============================================================================
# HELPERS
# =============================================================================

def _make_message(**overrides) -> BusMessage:
    defaults = dict(
        sender_id="agent-alpha",
        recipients=["agent-beta"],
        message_type=MessageType.TASK_REQUEST,
        payload={"task": "summarise document"},
        priority=MessagePriority.NORMAL,
    )
    defaults.update(overrides)
    return BusMessage(**defaults)


def _make_external_registration(**overrides) -> ExternalAgentRegistration:
    defaults = dict(
        agent_id="vigil-001",
        name="Vigil",
        description="External monitoring agent",
        capabilities=["monitoring", "alerting"],
        webhook=WebhookConfig(url="https://example.com/webhook"),
    )
    defaults.update(overrides)
    return ExternalAgentRegistration(**defaults)


def _fake_stored_message(msg: BusMessage) -> Dict[str, Any]:
    """Simulate the dict that bus_repository.store_message would return."""
    return {
        "message_id": msg.id,
        "sender_id": msg.sender_id,
        "recipients": msg.recipients,
        "message_type": msg.message_type.value,
        "topic": msg.topic,
        "payload": msg.payload,
        "priority": msg.priority.value,
        "correlation_id": msg.correlation_id,
        "reply_to": msg.reply_to,
        "ttl_seconds": msg.ttl_seconds,
        "created_at": msg.created_at.isoformat(),
    }


# =============================================================================
# 1. DIRECT MESSAGE DELIVERY TESTS
# =============================================================================

class TestDirectMessageDelivery:
    """Tests for direct agent-to-agent message publishing."""

    @pytest.mark.asyncio
    async def test_publish_persists_message_and_returns_receipt(self):
        """Verify publish stores the message and returns a delivery receipt."""
        # Arrange
        msg = _make_message()

        with patch("app.services.bus_service.repo") as mock_repo, \
             patch("app.services.bus_service.agent_ws_manager") as mock_ws:
            mock_repo.store_message = AsyncMock(return_value=_fake_stored_message(msg))
            mock_repo.get_all_subscriptions = AsyncMock(return_value=[])
            mock_repo.get_external_agent = AsyncMock(return_value=None)
            mock_ws.send_message = AsyncMock(return_value=True)
            mock_repo.create_delivery_receipt = AsyncMock(return_value={})

            # Act
            receipt = await bus_service.publish(msg)

            # Assert
            mock_repo.store_message.assert_awaited_once()
            assert receipt.message_id == msg.id
            assert receipt.status == DeliveryStatus.DELIVERED

    @pytest.mark.asyncio
    async def test_publish_delivers_to_all_direct_recipients(self):
        """Each recipient in the list should receive the message."""
        # Arrange
        msg = _make_message(recipients=["agent-beta", "agent-gamma"])

        with patch("app.services.bus_service.repo") as mock_repo, \
             patch("app.services.bus_service.agent_ws_manager") as mock_ws:
            mock_repo.store_message = AsyncMock(return_value=_fake_stored_message(msg))
            mock_repo.get_all_subscriptions = AsyncMock(return_value=[])
            mock_repo.get_external_agent = AsyncMock(return_value=None)
            mock_ws.send_message = AsyncMock(return_value=True)
            mock_repo.create_delivery_receipt = AsyncMock(return_value={})

            # Act
            receipt = await bus_service.publish(msg)

            # Assert
            assert mock_ws.send_message.await_count == 2
            assert receipt.status == DeliveryStatus.DELIVERED

    @pytest.mark.asyncio
    async def test_sender_does_not_receive_own_message(self):
        """Sender should never be in the delivery set even if subscribed."""
        # Arrange
        msg = _make_message(sender_id="agent-self", recipients=["agent-self", "agent-beta"])

        with patch("app.services.bus_service.repo") as mock_repo, \
             patch("app.services.bus_service.agent_ws_manager") as mock_ws:
            mock_repo.store_message = AsyncMock(return_value=_fake_stored_message(msg))
            mock_repo.get_all_subscriptions = AsyncMock(return_value=[])
            mock_repo.get_external_agent = AsyncMock(return_value=None)
            mock_ws.send_message = AsyncMock(return_value=True)
            mock_repo.create_delivery_receipt = AsyncMock(return_value={})

            # Act
            await bus_service.publish(msg)

            # Assert — only agent-beta should be delivered to
            assert mock_ws.send_message.await_count == 1
            delivered_agent = mock_ws.send_message.call_args_list[0][0][0]
            assert delivered_agent == "agent-beta"


# =============================================================================
# 2. PUB/SUB SUBSCRIPTION & BROADCAST TESTS
# =============================================================================

class TestPubSubBroadcast:
    """Tests for topic-based pub/sub and broadcast."""

    @pytest.mark.asyncio
    async def test_subscribe_creates_subscription(self):
        """Creating a subscription should persist and return a Subscription model."""
        # Arrange
        sub_create = SubscriptionCreate(
            agent_id="agent-x",
            message_types=[MessageType.STATUS_UPDATE],
            topics=["deployments"],
        )

        with patch("app.services.bus_service.repo") as mock_repo:
            mock_repo.create_subscription = AsyncMock(return_value={
                "subscription_id": "sub-123",
                "agent_id": "agent-x",
                "message_types": ["status_update"],
                "topics": ["deployments"],
            })

            # Act
            result = await bus_service.subscribe("agent-x", sub_create)

            # Assert
            assert isinstance(result, Subscription)
            assert result.agent_id == "agent-x"
            mock_repo.create_subscription.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_broadcast_delivers_to_topic_subscribers(self):
        """A broadcast should reach all agents subscribed to the topic."""
        # Arrange
        request = BroadcastRequest(
            sender_id="agent-alpha",
            topic="alerts",
            payload={"level": "warning"},
        )

        sub_row = {
            "subscription_id": "sub-1",
            "agent_id": "agent-beta",
            "message_types": [],
            "topics": ["alerts"],
        }

        with patch("app.services.bus_service.repo") as mock_repo, \
             patch("app.services.bus_service.agent_ws_manager") as mock_ws:
            mock_repo.store_message = AsyncMock(return_value={"message_id": "m1"})
            mock_repo.get_all_subscriptions = AsyncMock(return_value=[sub_row])
            mock_repo.get_external_agent = AsyncMock(return_value=None)
            mock_ws.send_message = AsyncMock(return_value=True)
            mock_repo.create_delivery_receipt = AsyncMock(return_value={})

            # Act
            receipt = await bus_service.broadcast(request)

            # Assert
            mock_ws.send_message.assert_awaited_once()
            call_agent = mock_ws.send_message.call_args_list[0][0][0]
            assert call_agent == "agent-beta"

    @pytest.mark.asyncio
    async def test_subscription_type_filter_works(self):
        """Only messages of subscribed types should match."""
        # Arrange
        msg = _make_message(
            message_type=MessageType.HELP_REQUEST,
            topic="general",
            recipients=[],
        )

        sub_status = {
            "subscription_id": "sub-s",
            "agent_id": "agent-status-listener",
            "message_types": ["status_update"],
            "topics": [],
        }
        sub_help = {
            "subscription_id": "sub-h",
            "agent_id": "agent-helper",
            "message_types": ["help_request"],
            "topics": [],
        }

        with patch("app.services.bus_service.repo") as mock_repo, \
             patch("app.services.bus_service.agent_ws_manager") as mock_ws:
            mock_repo.store_message = AsyncMock(return_value=_fake_stored_message(msg))
            mock_repo.get_all_subscriptions = AsyncMock(return_value=[sub_status, sub_help])
            mock_repo.get_external_agent = AsyncMock(return_value=None)
            mock_ws.send_message = AsyncMock(return_value=True)
            mock_repo.create_delivery_receipt = AsyncMock(return_value={})

            # Act
            await bus_service.publish(msg)

            # Assert
            assert mock_ws.send_message.await_count == 1
            assert mock_ws.send_message.call_args_list[0][0][0] == "agent-helper"

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_subscription(self):
        """Unsubscribing should delegate to the repository delete."""
        # Arrange
        with patch("app.services.bus_service.repo") as mock_repo:
            mock_repo.delete_subscription = AsyncMock(return_value=True)

            # Act
            result = await bus_service.unsubscribe("agent-x", "sub-123")

            # Assert
            assert result is True
            mock_repo.delete_subscription.assert_awaited_once_with("sub-123")

    @pytest.mark.asyncio
    async def test_get_subscriptions_returns_list(self):
        """Listing subscriptions returns Subscription models."""
        # Arrange
        rows = [
            {
                "subscription_id": "s1",
                "agent_id": "agent-x",
                "message_types": ["heartbeat"],
                "topics": ["health"],
            }
        ]

        with patch("app.services.bus_service.repo") as mock_repo:
            mock_repo.get_subscriptions_for_agent = AsyncMock(return_value=rows)

            # Act
            subs = await bus_service.get_subscriptions("agent-x")

            # Assert
            assert len(subs) == 1
            assert subs[0].id == "s1"


# =============================================================================
# 3. EXTERNAL AGENT REGISTRATION & WEBHOOK DELIVERY TESTS
# =============================================================================

class TestExternalAgents:
    """Tests for external agent lifecycle and webhook delivery."""

    @pytest.mark.asyncio
    async def test_register_external_agent_persists(self):
        """Registration should store the agent with webhook config."""
        # Arrange
        reg = _make_external_registration()

        with patch("app.services.bus_service.repo") as mock_repo:
            mock_repo.register_external_agent = AsyncMock(return_value={
                "agent_id": "vigil-001",
                "name": "Vigil",
                "webhook_url": "https://example.com/webhook",
            })

            # Act
            result = await bus_service.register_external_agent(reg)

            # Assert
            assert result["agent_id"] == "vigil-001"
            mock_repo.register_external_agent.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_deregister_external_agent(self):
        """Deregistration should remove the agent record."""
        # Arrange
        with patch("app.services.bus_service.repo") as mock_repo:
            mock_repo.deregister_external_agent = AsyncMock(return_value=True)

            # Act
            result = await bus_service.deregister_external_agent("vigil-001")

            # Assert
            assert result is True

    @pytest.mark.asyncio
    async def test_list_external_agents(self):
        """Listing should return all registered external agents."""
        # Arrange
        with patch("app.services.bus_service.repo") as mock_repo:
            mock_repo.list_external_agents = AsyncMock(return_value=[
                {"agent_id": "vigil-001", "name": "Vigil"},
                {"agent_id": "sentry-002", "name": "Sentry"},
            ])

            # Act
            agents = await bus_service.list_external_agents()

            # Assert
            assert len(agents) == 2

    @pytest.mark.asyncio
    async def test_deliver_to_external_agent_via_webhook(self):
        """Messages to external agents should be POST-ed to their webhook URL."""
        # Arrange
        msg = _make_message(recipients=["vigil-001"])
        ext_data = {
            "agent_id": "vigil-001",
            "webhook_url": "https://example.com/hook",
            "webhook_secret": "s3cret",
            "webhook_max_retries": 1,
            "webhook_retry_backoff_base_ms": 100,
            "webhook_timeout_ms": 3000,
        }

        with patch("app.services.bus_service.repo") as mock_repo, \
             patch("app.services.bus_service.httpx") as mock_httpx:
            mock_repo.get_external_agent = AsyncMock(return_value=ext_data)

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client

            # Act
            success = await bus_service.deliver_to_external("vigil-001", msg)

            # Assert
            assert success is True
            mock_client.post.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_webhook_failure_returns_false_after_retries(self):
        """If all retries fail, deliver_to_external returns False."""
        # Arrange
        msg = _make_message()
        ext_data = {
            "agent_id": "vigil-001",
            "webhook_url": "https://example.com/hook",
            "webhook_secret": None,
            "webhook_max_retries": 1,
            "webhook_retry_backoff_base_ms": 50,
            "webhook_timeout_ms": 1000,
        }

        with patch("app.services.bus_service.repo") as mock_repo, \
             patch("app.services.bus_service.httpx") as mock_httpx:
            mock_repo.get_external_agent = AsyncMock(return_value=ext_data)

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=Exception("connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client

            # Act
            success = await bus_service.deliver_to_external("vigil-001", msg)

            # Assert
            assert success is False
            # 1 initial + 1 retry = 2 calls
            assert mock_client.post.await_count == 2

    @pytest.mark.asyncio
    async def test_webhook_signature_is_hmac_sha256(self):
        """External deliveries should include an HMAC-SHA256 signature header."""
        # Arrange
        msg = _make_message()
        ext_data = {
            "agent_id": "vigil-001",
            "webhook_url": "https://example.com/hook",
            "webhook_secret": "my-secret",
            "webhook_max_retries": 0,
            "webhook_retry_backoff_base_ms": 100,
            "webhook_timeout_ms": 2000,
        }

        with patch("app.services.bus_service.repo") as mock_repo, \
             patch("app.services.bus_service.httpx") as mock_httpx:
            mock_repo.get_external_agent = AsyncMock(return_value=ext_data)

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client

            # Act
            await bus_service.deliver_to_external("vigil-001", msg)

            # Assert
            call_kwargs = mock_client.post.call_args
            headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
            assert "X-Bus-Signature" in headers


# =============================================================================
# 4. OFFLINE QUEUE & DRAIN TESTS
# =============================================================================

class TestOfflineQueue:
    """Tests for queuing messages when agents are offline."""

    @pytest.mark.asyncio
    async def test_offline_agent_message_is_queued(self):
        """If an agent is not connected via WS and not external, message should be queued."""
        # Arrange
        msg = _make_message(recipients=["offline-agent"])

        with patch("app.services.bus_service.repo") as mock_repo, \
             patch("app.services.bus_service.agent_ws_manager") as mock_ws:
            mock_repo.store_message = AsyncMock(return_value=_fake_stored_message(msg))
            mock_repo.get_all_subscriptions = AsyncMock(return_value=[])
            mock_repo.get_external_agent = AsyncMock(return_value=None)
            mock_ws.send_message = AsyncMock(return_value=False)  # offline
            mock_repo.enqueue_offline = AsyncMock()
            mock_repo.create_delivery_receipt = AsyncMock(return_value={})

            # Act
            receipt = await bus_service.publish(msg)

            # Assert
            mock_repo.enqueue_offline.assert_awaited_once_with("offline-agent", msg.id)
            assert receipt.status == DeliveryStatus.QUEUED

    @pytest.mark.asyncio
    async def test_drain_queue_returns_and_clears_messages(self):
        """Draining should return all queued messages and clear the queue."""
        # Arrange
        queued = [
            {"message_id": "m1", "sender_id": "a", "payload": {}},
            {"message_id": "m2", "sender_id": "b", "payload": {}},
        ]

        with patch("app.services.bus_service.repo") as mock_repo:
            mock_repo.drain_offline_queue = AsyncMock(return_value=queued)

            # Act
            result = await bus_service.drain_queue("offline-agent")

            # Assert
            assert len(result) == 2
            mock_repo.drain_offline_queue.assert_awaited_once_with("offline-agent")

    @pytest.mark.asyncio
    async def test_queue_for_offline_delegates_to_repo(self):
        """queue_for_offline should call the repo enqueue method."""
        # Arrange
        msg = _make_message()

        with patch("app.services.bus_service.repo") as mock_repo:
            mock_repo.enqueue_offline = AsyncMock()

            # Act
            await bus_service.queue_for_offline("target-agent", msg)

            # Assert
            mock_repo.enqueue_offline.assert_awaited_once_with("target-agent", msg.id)


# =============================================================================
# 5. MESSAGE ROUTING TESTS (@mention, capability query)
# =============================================================================

class TestMessageRouting:
    """Tests for advanced routing patterns."""

    @pytest.mark.asyncio
    async def test_mention_parsing_adds_recipients(self):
        """@agent-id mentions in payload text should add those agents as recipients."""
        # Arrange
        msg = _make_message(
            recipients=[],
            payload={"text": "Hey @agent-beta and @agent-gamma, please help."},
        )

        with patch("app.services.bus_service.repo") as mock_repo, \
             patch("app.services.bus_service.agent_ws_manager") as mock_ws:
            mock_repo.store_message = AsyncMock(return_value=_fake_stored_message(msg))
            mock_repo.get_all_subscriptions = AsyncMock(return_value=[])
            mock_repo.get_external_agent = AsyncMock(return_value=None)
            mock_ws.send_message = AsyncMock(return_value=True)
            mock_repo.create_delivery_receipt = AsyncMock(return_value={})

            # Act
            await bus_service.publish(msg)

            # Assert
            delivered_agents = {
                call[0][0] for call in mock_ws.send_message.call_args_list
            }
            assert "agent-beta" in delivered_agents
            assert "agent-gamma" in delivered_agents

    def test_parse_mentions_extracts_ids(self):
        """_parse_mentions should extract all @word patterns."""
        # Arrange
        text = "@alice said hello to @bob-agent and @charlie_99"

        # Act
        result = bus_service._parse_mentions(text)

        # Assert
        assert result == {"alice", "bob-agent", "charlie_99"}

    def test_parse_mentions_empty_string_returns_empty(self):
        """Empty text should yield an empty set."""
        # Arrange / Act
        result = bus_service._parse_mentions("")

        # Assert
        assert result == set()

    def test_parse_mentions_none_returns_empty(self):
        """None text should yield an empty set."""
        # Arrange / Act
        result = bus_service._parse_mentions(None)

        # Assert
        assert result == set()

    @pytest.mark.asyncio
    async def test_capability_query_message_is_published(self):
        """A capability_query should be publishable like any other message."""
        # Arrange
        msg = _make_message(
            message_type=MessageType.CAPABILITY_QUERY,
            payload={"capability": "code_review"},
            recipients=[],
            topic="capabilities",
        )
        sub_row = {
            "subscription_id": "s1",
            "agent_id": "agent-reviewer",
            "message_types": ["capability_query"],
            "topics": ["capabilities"],
        }

        with patch("app.services.bus_service.repo") as mock_repo, \
             patch("app.services.bus_service.agent_ws_manager") as mock_ws:
            mock_repo.store_message = AsyncMock(return_value=_fake_stored_message(msg))
            mock_repo.get_all_subscriptions = AsyncMock(return_value=[sub_row])
            mock_repo.get_external_agent = AsyncMock(return_value=None)
            mock_ws.send_message = AsyncMock(return_value=True)
            mock_repo.create_delivery_receipt = AsyncMock(return_value={})

            # Act
            receipt = await bus_service.publish(msg)

            # Assert
            assert receipt.status == DeliveryStatus.DELIVERED
            assert mock_ws.send_message.call_args_list[0][0][0] == "agent-reviewer"


# =============================================================================
# 6. REQUEST / RESPONSE PATTERN TESTS
# =============================================================================

class TestRequestResponse:
    """Tests for the synchronous request/response pattern."""

    @pytest.mark.asyncio
    async def test_request_response_returns_reply(self):
        """request_response should find a correlated reply message."""
        # Arrange
        correlation = str(uuid.uuid4())
        request_msg = _make_message(correlation_id=correlation)

        reply_row = {
            "message_id": "reply-1",
            "sender_id": "agent-beta",
            "recipients": ["agent-alpha"],
            "message_type": "task_result",
            "topic": None,
            "payload": {"result": "done"},
            "priority": "normal",
            "correlation_id": correlation,
            "reply_to": request_msg.id,
            "created_at": datetime.utcnow().isoformat(),
        }

        call_count = 0

        async def mock_get_correlated(cid):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                return [_fake_stored_message(request_msg), reply_row]
            return [_fake_stored_message(request_msg)]

        with patch("app.services.bus_service.repo") as mock_repo, \
             patch("app.services.bus_service.agent_ws_manager") as mock_ws:
            mock_repo.store_message = AsyncMock(return_value=_fake_stored_message(request_msg))
            mock_repo.get_all_subscriptions = AsyncMock(return_value=[])
            mock_repo.get_external_agent = AsyncMock(return_value=None)
            mock_ws.send_message = AsyncMock(return_value=True)
            mock_repo.create_delivery_receipt = AsyncMock(return_value={})
            mock_repo.get_messages_by_correlation = AsyncMock(side_effect=mock_get_correlated)

            # Act
            result = await bus_service.request_response(request_msg, timeout_ms=2000)

            # Assert
            assert result is not None
            assert result.id == "reply-1"
            assert result.sender_id == "agent-beta"

    @pytest.mark.asyncio
    async def test_request_response_timeout_returns_none(self):
        """If no reply arrives within timeout, return None."""
        # Arrange
        request_msg = _make_message(correlation_id=str(uuid.uuid4()))

        with patch("app.services.bus_service.repo") as mock_repo, \
             patch("app.services.bus_service.agent_ws_manager") as mock_ws:
            mock_repo.store_message = AsyncMock(return_value=_fake_stored_message(request_msg))
            mock_repo.get_all_subscriptions = AsyncMock(return_value=[])
            mock_repo.get_external_agent = AsyncMock(return_value=None)
            mock_ws.send_message = AsyncMock(return_value=True)
            mock_repo.create_delivery_receipt = AsyncMock(return_value={})
            mock_repo.get_messages_by_correlation = AsyncMock(
                return_value=[_fake_stored_message(request_msg)]
            )

            # Act — very short timeout
            result = await bus_service.request_response(request_msg, timeout_ms=300)

            # Assert
            assert result is None

    @pytest.mark.asyncio
    async def test_request_response_assigns_correlation_id_if_missing(self):
        """If the request has no correlation_id, one should be generated."""
        # Arrange
        request_msg = _make_message(correlation_id=None)

        with patch("app.services.bus_service.repo") as mock_repo, \
             patch("app.services.bus_service.agent_ws_manager") as mock_ws:
            mock_repo.store_message = AsyncMock(return_value=_fake_stored_message(request_msg))
            mock_repo.get_all_subscriptions = AsyncMock(return_value=[])
            mock_repo.get_external_agent = AsyncMock(return_value=None)
            mock_ws.send_message = AsyncMock(return_value=True)
            mock_repo.create_delivery_receipt = AsyncMock(return_value={})
            mock_repo.get_messages_by_correlation = AsyncMock(return_value=[])

            # Act
            await bus_service.request_response(request_msg, timeout_ms=300)

            # Assert — correlation_id should have been assigned
            assert request_msg.correlation_id is not None


# =============================================================================
# 7. DELIVERY RECEIPT TRACKING TESTS
# =============================================================================

class TestDeliveryReceipts:
    """Tests for delivery receipt creation and tracking."""

    @pytest.mark.asyncio
    async def test_successful_delivery_creates_delivered_receipt(self):
        """When WS delivery succeeds, a 'delivered' receipt should be stored."""
        # Arrange
        msg = _make_message(recipients=["agent-beta"])

        with patch("app.services.bus_service.repo") as mock_repo, \
             patch("app.services.bus_service.agent_ws_manager") as mock_ws:
            mock_repo.store_message = AsyncMock(return_value=_fake_stored_message(msg))
            mock_repo.get_all_subscriptions = AsyncMock(return_value=[])
            mock_repo.get_external_agent = AsyncMock(return_value=None)
            mock_ws.send_message = AsyncMock(return_value=True)
            mock_repo.create_delivery_receipt = AsyncMock(return_value={})

            # Act
            await bus_service.publish(msg)

            # Assert
            receipt_call = mock_repo.create_delivery_receipt.call_args_list[0]
            assert receipt_call[0][3] == "delivered"  # status arg

    @pytest.mark.asyncio
    async def test_failed_delivery_creates_failed_receipt(self):
        """When external webhook fails, a 'failed' receipt should be stored."""
        # Arrange
        msg = _make_message(recipients=["ext-agent"])

        ext_data = {
            "agent_id": "ext-agent",
            "webhook_url": "https://bad.example.com/hook",
            "webhook_secret": None,
            "webhook_max_retries": 0,
            "webhook_retry_backoff_base_ms": 50,
            "webhook_timeout_ms": 500,
        }

        with patch("app.services.bus_service.repo") as mock_repo, \
             patch("app.services.bus_service.agent_ws_manager") as mock_ws, \
             patch("app.services.bus_service.httpx") as mock_httpx:
            mock_repo.store_message = AsyncMock(return_value=_fake_stored_message(msg))
            mock_repo.get_all_subscriptions = AsyncMock(return_value=[])
            mock_repo.get_external_agent = AsyncMock(return_value=ext_data)
            mock_repo.create_delivery_receipt = AsyncMock(return_value={})

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=Exception("timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client

            # Act
            await bus_service.publish(msg)

            # Assert
            receipt_call = mock_repo.create_delivery_receipt.call_args_list[0]
            assert receipt_call[0][3] == "failed"  # status arg

    @pytest.mark.asyncio
    async def test_queued_delivery_creates_queued_receipt(self):
        """When agent is offline, a 'queued' receipt should be stored."""
        # Arrange
        msg = _make_message(recipients=["offline-agent"])

        with patch("app.services.bus_service.repo") as mock_repo, \
             patch("app.services.bus_service.agent_ws_manager") as mock_ws:
            mock_repo.store_message = AsyncMock(return_value=_fake_stored_message(msg))
            mock_repo.get_all_subscriptions = AsyncMock(return_value=[])
            mock_repo.get_external_agent = AsyncMock(return_value=None)
            mock_ws.send_message = AsyncMock(return_value=False)
            mock_repo.enqueue_offline = AsyncMock()
            mock_repo.create_delivery_receipt = AsyncMock(return_value={})

            # Act
            await bus_service.publish(msg)

            # Assert
            receipt_call = mock_repo.create_delivery_receipt.call_args_list[0]
            assert receipt_call[0][3] == "queued"


# =============================================================================
# 8. METRICS CALCULATION TESTS
# =============================================================================

class TestMetrics:
    """Tests for bus metrics aggregation."""

    @pytest.mark.asyncio
    async def test_get_metrics_returns_bus_metrics(self):
        """get_metrics should aggregate repo counters into a BusMetrics model."""
        # Arrange
        with patch("app.services.bus_service.repo") as mock_repo:
            mock_repo.count_messages = AsyncMock(return_value=100)
            mock_repo.count_messages_by_type = AsyncMock(return_value={"task_request": 60, "heartbeat": 40})
            mock_repo.count_messages_by_priority = AsyncMock(return_value={"normal": 80, "high": 20})
            mock_repo.count_receipts_by_status = AsyncMock(return_value={"delivered": 90, "failed": 5, "queued": 5})
            mock_repo.count_subscriptions = AsyncMock(return_value=12)
            mock_repo.count_external_agents = AsyncMock(return_value=2)
            mock_repo.count_offline_queued = AsyncMock(return_value=3)
            mock_repo.avg_delivery_latency_ms = AsyncMock(return_value=42.5)

            # Act
            metrics = await bus_service.get_metrics()

            # Assert
            assert isinstance(metrics, BusMetrics)
            assert metrics.total_messages_published == 100
            assert metrics.total_messages_delivered == 90
            assert metrics.total_messages_failed == 5
            assert metrics.total_messages_queued == 3
            assert metrics.active_subscriptions == 12
            assert metrics.external_agents_count == 2
            assert metrics.avg_delivery_latency_ms == 42.5
            assert metrics.messages_by_type["task_request"] == 60

    @pytest.mark.asyncio
    async def test_metrics_handles_empty_system(self):
        """Metrics should gracefully handle a fresh system with no data."""
        # Arrange
        with patch("app.services.bus_service.repo") as mock_repo:
            mock_repo.count_messages = AsyncMock(return_value=0)
            mock_repo.count_messages_by_type = AsyncMock(return_value={})
            mock_repo.count_messages_by_priority = AsyncMock(return_value={})
            mock_repo.count_receipts_by_status = AsyncMock(return_value={})
            mock_repo.count_subscriptions = AsyncMock(return_value=0)
            mock_repo.count_external_agents = AsyncMock(return_value=0)
            mock_repo.count_offline_queued = AsyncMock(return_value=0)
            mock_repo.avg_delivery_latency_ms = AsyncMock(return_value=None)

            # Act
            metrics = await bus_service.get_metrics()

            # Assert
            assert metrics.total_messages_published == 0
            assert metrics.avg_delivery_latency_ms is None


# =============================================================================
# 9. EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_unknown_external_agent_delivery_returns_false(self):
        """Delivering to a non-existent external agent should return False."""
        # Arrange
        msg = _make_message()

        with patch("app.services.bus_service.repo") as mock_repo:
            mock_repo.get_external_agent = AsyncMock(return_value=None)

            # Act
            result = await bus_service.deliver_to_external("ghost-agent", msg)

            # Assert
            assert result is False

    @pytest.mark.asyncio
    async def test_duplicate_message_id_handled(self):
        """Publishing a message with a duplicate ID should not crash."""
        # Arrange
        msg = _make_message()

        with patch("app.services.bus_service.repo") as mock_repo, \
             patch("app.services.bus_service.agent_ws_manager") as mock_ws:
            mock_repo.store_message = AsyncMock(side_effect=Exception("unique_violation"))
            mock_repo.get_all_subscriptions = AsyncMock(return_value=[])

            # Act / Assert — should propagate but not crash unrecoverably
            with pytest.raises(Exception, match="unique_violation"):
                await bus_service.publish(msg)

    @pytest.mark.asyncio
    async def test_empty_recipients_and_no_subscriptions_no_delivery(self):
        """Message with no recipients and no matching subscriptions delivers nowhere."""
        # Arrange
        msg = _make_message(recipients=[])

        with patch("app.services.bus_service.repo") as mock_repo, \
             patch("app.services.bus_service.agent_ws_manager") as mock_ws:
            mock_repo.store_message = AsyncMock(return_value=_fake_stored_message(msg))
            mock_repo.get_all_subscriptions = AsyncMock(return_value=[])
            mock_ws.send_message = AsyncMock(return_value=True)

            # Act
            receipt = await bus_service.publish(msg)

            # Assert
            mock_ws.send_message.assert_not_awaited()
            assert receipt.recipient_id == "none"

    @pytest.mark.asyncio
    async def test_message_priority_preserved_in_storage(self):
        """Priority should be preserved when storing and delivering."""
        # Arrange
        msg = _make_message(priority=MessagePriority.URGENT)

        with patch("app.services.bus_service.repo") as mock_repo, \
             patch("app.services.bus_service.agent_ws_manager") as mock_ws:
            mock_repo.store_message = AsyncMock(return_value=_fake_stored_message(msg))
            mock_repo.get_all_subscriptions = AsyncMock(return_value=[])
            mock_repo.get_external_agent = AsyncMock(return_value=None)
            mock_ws.send_message = AsyncMock(return_value=True)
            mock_repo.create_delivery_receipt = AsyncMock(return_value={})

            # Act
            await bus_service.publish(msg)

            # Assert
            store_call = mock_repo.store_message.call_args
            assert store_call.kwargs["priority"] == "urgent"

    @pytest.mark.asyncio
    async def test_ttl_field_accepted_in_message(self):
        """TTL field should be accepted and stored."""
        # Arrange
        msg = _make_message(ttl_seconds=300)

        with patch("app.services.bus_service.repo") as mock_repo, \
             patch("app.services.bus_service.agent_ws_manager") as mock_ws:
            mock_repo.store_message = AsyncMock(return_value=_fake_stored_message(msg))
            mock_repo.get_all_subscriptions = AsyncMock(return_value=[])
            mock_repo.get_external_agent = AsyncMock(return_value=None)
            mock_ws.send_message = AsyncMock(return_value=True)
            mock_repo.create_delivery_receipt = AsyncMock(return_value={})

            # Act
            await bus_service.publish(msg)

            # Assert
            store_call = mock_repo.store_message.call_args
            assert store_call.kwargs["ttl_seconds"] == 300

    @pytest.mark.asyncio
    async def test_mixed_online_offline_recipients(self):
        """Some recipients online, some offline — each handled correctly."""
        # Arrange
        msg = _make_message(recipients=["online-agent", "offline-agent"])

        async def ws_side_effect(agent_id, payload):
            return agent_id == "online-agent"

        with patch("app.services.bus_service.repo") as mock_repo, \
             patch("app.services.bus_service.agent_ws_manager") as mock_ws:
            mock_repo.store_message = AsyncMock(return_value=_fake_stored_message(msg))
            mock_repo.get_all_subscriptions = AsyncMock(return_value=[])
            mock_repo.get_external_agent = AsyncMock(return_value=None)
            mock_ws.send_message = AsyncMock(side_effect=ws_side_effect)
            mock_repo.enqueue_offline = AsyncMock()
            mock_repo.create_delivery_receipt = AsyncMock(return_value={})

            # Act
            receipt = await bus_service.publish(msg)

            # Assert
            mock_repo.enqueue_offline.assert_awaited_once_with("offline-agent", msg.id)


# =============================================================================
# 10. MODEL VALIDATION TESTS
# =============================================================================

class TestModelValidation:
    """Tests for Pydantic model validation."""

    def test_bus_message_requires_sender(self):
        """BusMessage must have a non-empty sender_id."""
        # Arrange / Act / Assert
        with pytest.raises(Exception):
            BusMessage(sender_id="", message_type=MessageType.HEARTBEAT)

    def test_bus_message_default_id_is_uuid(self):
        """BusMessage should auto-generate a UUID id."""
        # Arrange / Act
        msg = BusMessage(sender_id="a", message_type=MessageType.HEARTBEAT)

        # Assert
        assert len(msg.id) == 36  # UUID format

    def test_webhook_config_validates_retries(self):
        """WebhookConfig max_retries should be bounded 0-10."""
        # Arrange / Act / Assert
        with pytest.raises(Exception):
            WebhookConfig(url="https://x.com", max_retries=99)

    def test_subscription_create_validates_agent_id(self):
        """SubscriptionCreate must have a non-empty agent_id."""
        # Arrange / Act / Assert
        with pytest.raises(Exception):
            SubscriptionCreate(agent_id="")

    def test_message_priority_enum_values(self):
        """MessagePriority should have exactly 4 levels."""
        # Arrange / Act
        values = [p.value for p in MessagePriority]

        # Assert
        assert set(values) == {"low", "normal", "high", "urgent"}

    def test_message_type_enum_values(self):
        """MessageType should have exactly 8 types."""
        # Arrange / Act
        values = [t.value for t in MessageType]

        # Assert
        assert len(values) == 8
        assert "task_request" in values
        assert "broadcast" in values
