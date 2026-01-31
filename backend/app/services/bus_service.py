"""
Inter-Agent Communication Bus Service

Core logic for the programmatic agent-to-agent messaging backbone.

Responsibilities:
- Publish messages to direct recipients or topics
- Manage pub/sub subscriptions
- Deliver to internal agents via WebSocket and external agents via webhooks
- Queue messages for offline agents
- Register / deregister external agents (e.g. Vigil)
- Provide bus-level metrics
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

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
    WebhookIncoming,
)
from app.repository import bus_repository as repo
from app.controllers.agent_ws import agent_ws_manager

logger = logging.getLogger(__name__)


# =============================================================================
# PUBLISHING
# =============================================================================

async def publish(message: BusMessage) -> DeliveryReceipt:
    """
    Publish a message on the bus.

    1. Persist the message.
    2. Resolve all target recipients (direct + subscription matches).
    3. Deliver to each recipient (internal WS or external webhook).
    4. Queue for any offline internal agents.
    5. Return a summary delivery receipt.
    """
    # Persist
    await repo.store_message(
        message_id=message.id,
        sender_id=message.sender_id,
        recipients=message.recipients,
        message_type=message.message_type.value,
        topic=message.topic,
        payload=message.payload,
        priority=message.priority.value,
        correlation_id=message.correlation_id,
        reply_to=message.reply_to,
        ttl_seconds=message.ttl_seconds,
        created_at=message.created_at,
    )

    # Collect all recipients ---------------------------------------------------
    targets: set[str] = set(message.recipients)

    # Add subscription-matched agents
    matched = await _resolve_subscription_targets(message)
    targets.update(matched)

    # Add @mention-parsed agents
    mentioned = _parse_mentions(message.payload.get("text", ""))
    targets.update(mentioned)

    # Never send to self
    targets.discard(message.sender_id)

    # Deliver ------------------------------------------------------------------
    receipts: List[DeliveryReceipt] = []
    for target_id in targets:
        receipt = await _deliver(target_id, message)
        receipts.append(receipt)

    # Return aggregate receipt
    delivered = sum(1 for r in receipts if r.status == DeliveryStatus.DELIVERED)
    failed = sum(1 for r in receipts if r.status == DeliveryStatus.FAILED)
    queued = sum(1 for r in receipts if r.status == DeliveryStatus.QUEUED)

    summary_status = DeliveryStatus.DELIVERED
    if failed and not delivered:
        summary_status = DeliveryStatus.FAILED
    elif queued and not delivered:
        summary_status = DeliveryStatus.QUEUED

    return DeliveryReceipt(
        message_id=message.id,
        recipient_id=",".join(targets) if targets else "none",
        status=summary_status,
        delivered_at=datetime.utcnow() if delivered else None,
        error=f"{failed} failed, {queued} queued" if (failed or queued) else None,
    )


async def broadcast(request: BroadcastRequest) -> DeliveryReceipt:
    """Broadcast a message to all agents subscribed to a topic."""
    message = BusMessage(
        sender_id=request.sender_id,
        recipients=[],
        message_type=request.message_type,
        topic=request.topic,
        payload=request.payload,
        priority=request.priority,
    )
    return await publish(message)


async def request_response(
    request: BusMessage, timeout_ms: int = 10000
) -> Optional[BusMessage]:
    """
    Synchronous request/response pattern.

    Publishes *request* with a ``correlation_id`` and waits up to *timeout_ms*
    for a reply message sharing the same ``correlation_id``.
    """
    if not request.correlation_id:
        request.correlation_id = str(uuid.uuid4())

    await publish(request)

    # Poll for response
    deadline = asyncio.get_event_loop().time() + timeout_ms / 1000
    while asyncio.get_event_loop().time() < deadline:
        rows = await repo.get_messages_by_correlation(request.correlation_id)
        for row in rows:
            if row["message_id"] != request.id and row["reply_to"] == request.id:
                return BusMessage(
                    id=row["message_id"],
                    sender_id=row["sender_id"],
                    recipients=row.get("recipients", []),
                    message_type=row["message_type"],
                    topic=row.get("topic"),
                    payload=row.get("payload", {}),
                    priority=row.get("priority", "normal"),
                    correlation_id=row.get("correlation_id"),
                    reply_to=row.get("reply_to"),
                )
        await asyncio.sleep(0.25)
    return None


# =============================================================================
# SUBSCRIPTIONS
# =============================================================================

async def subscribe(agent_id: str, sub: SubscriptionCreate) -> Subscription:
    subscription_id = str(uuid.uuid4())
    await repo.create_subscription(
        subscription_id=subscription_id,
        agent_id=agent_id,
        message_types=[mt.value for mt in sub.message_types],
        topics=sub.topics,
        scope=sub.scope.model_dump() if sub.scope else None,
    )
    return Subscription(
        id=subscription_id,
        agent_id=agent_id,
        message_types=sub.message_types,
        topics=sub.topics,
        scope=sub.scope,
    )


async def unsubscribe(agent_id: str, subscription_id: str) -> bool:
    return await repo.delete_subscription(subscription_id)


async def get_subscriptions(agent_id: str) -> List[Subscription]:
    rows = await repo.get_subscriptions_for_agent(agent_id)
    subs: List[Subscription] = []
    for r in rows:
        scope_data = r.get("scope")
        scope = BusScope(**scope_data) if isinstance(scope_data, dict) else None
        subs.append(
            Subscription(
                id=r["subscription_id"],
                agent_id=r["agent_id"],
                message_types=r.get("message_types", []),
                topics=r.get("topics", []),
                scope=scope,
            )
        )
    return subs


# =============================================================================
# DELIVERY (INTERNAL / EXTERNAL / QUEUE)
# =============================================================================

async def deliver_to_internal(agent_id: str, message: BusMessage) -> bool:
    """Deliver via the Agent WebSocket manager."""
    payload = {
        "type": "bus_message",
        "bus_message": message.model_dump(mode="json"),
    }
    return await agent_ws_manager.send_message(agent_id, payload)


async def deliver_to_external(agent_id: str, message: BusMessage) -> bool:
    """Deliver via HTTP webhook to an external agent."""
    ext = await repo.get_external_agent(agent_id)
    if not ext:
        logger.warning(f"External agent {agent_id} not found for delivery")
        return False

    url = ext["webhook_url"]
    secret = ext.get("webhook_secret")
    max_retries = ext.get("webhook_max_retries", 3)
    backoff_base = ext.get("webhook_retry_backoff_base_ms", 1000)
    timeout = ext.get("webhook_timeout_ms", 5000)

    body = json.dumps(message.model_dump(mode="json"))
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if secret:
        sig = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
        headers["X-Bus-Signature"] = sig

    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout / 1000) as client:
                resp = await client.post(url, content=body, headers=headers)
                if resp.status_code < 300:
                    return True
                logger.warning(
                    f"Webhook to {agent_id} returned {resp.status_code} (attempt {attempt+1})"
                )
        except Exception as exc:
            logger.warning(f"Webhook to {agent_id} failed (attempt {attempt+1}): {exc}")

        if attempt < max_retries:
            await asyncio.sleep(backoff_base * (2 ** attempt) / 1000)

    return False


async def queue_for_offline(agent_id: str, message: BusMessage) -> None:
    """Queue a message for later delivery when the agent comes back online."""
    await repo.enqueue_offline(agent_id, message.id)


async def drain_queue(agent_id: str) -> List[Dict[str, Any]]:
    """Return all queued messages for *agent_id* and clear the queue."""
    return await repo.drain_offline_queue(agent_id)


# =============================================================================
# EXTERNAL AGENT MANAGEMENT
# =============================================================================

async def register_external_agent(
    registration: ExternalAgentRegistration,
) -> Dict[str, Any]:
    return await repo.register_external_agent(
        agent_id=registration.agent_id,
        name=registration.name,
        description=registration.description,
        capabilities=registration.capabilities,
        webhook_url=registration.webhook.url,
        webhook_secret=registration.webhook.secret,
        webhook_max_retries=registration.webhook.max_retries,
        webhook_retry_backoff_base_ms=registration.webhook.retry_backoff_base_ms,
        webhook_timeout_ms=registration.webhook.timeout_ms,
    )


async def get_external_agent(agent_id: str) -> Optional[Dict[str, Any]]:
    return await repo.get_external_agent(agent_id)


async def deregister_external_agent(agent_id: str) -> bool:
    return await repo.deregister_external_agent(agent_id)


async def list_external_agents() -> List[Dict[str, Any]]:
    return await repo.list_external_agents()


# =============================================================================
# METRICS
# =============================================================================

async def get_metrics() -> BusMetrics:
    total_published = await repo.count_messages()
    by_type = await repo.count_messages_by_type()
    by_priority = await repo.count_messages_by_priority()
    receipt_counts = await repo.count_receipts_by_status()
    subs_count = await repo.count_subscriptions()
    ext_count = await repo.count_external_agents()
    queued = await repo.count_offline_queued()
    latency = await repo.avg_delivery_latency_ms()

    return BusMetrics(
        total_messages_published=total_published,
        total_messages_delivered=receipt_counts.get("delivered", 0),
        total_messages_failed=receipt_counts.get("failed", 0),
        total_messages_queued=queued,
        active_subscriptions=subs_count,
        external_agents_count=ext_count,
        avg_delivery_latency_ms=latency,
        messages_by_type=by_type,
        messages_by_priority=by_priority,
    )


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

async def _deliver(target_id: str, message: BusMessage) -> DeliveryReceipt:
    """Route delivery to the right channel and persist a receipt."""
    receipt_id = str(uuid.uuid4())

    # Try external agent first
    ext = await repo.get_external_agent(target_id)
    if ext:
        success = await deliver_to_external(target_id, message)
        status = DeliveryStatus.DELIVERED if success else DeliveryStatus.FAILED
        error = None if success else "Webhook delivery failed after retries"
        await repo.create_delivery_receipt(
            receipt_id, message.id, target_id, status.value, error
        )
        return DeliveryReceipt(
            id=receipt_id,
            message_id=message.id,
            recipient_id=target_id,
            status=status,
            delivered_at=datetime.utcnow() if success else None,
            error=error,
        )

    # Try internal WebSocket
    ws_ok = await deliver_to_internal(target_id, message)
    if ws_ok:
        await repo.create_delivery_receipt(
            receipt_id, message.id, target_id, DeliveryStatus.DELIVERED.value
        )
        return DeliveryReceipt(
            id=receipt_id,
            message_id=message.id,
            recipient_id=target_id,
            status=DeliveryStatus.DELIVERED,
            delivered_at=datetime.utcnow(),
        )

    # Agent offline — queue
    await queue_for_offline(target_id, message)
    await repo.create_delivery_receipt(
        receipt_id, message.id, target_id, DeliveryStatus.QUEUED.value
    )
    return DeliveryReceipt(
        id=receipt_id,
        message_id=message.id,
        recipient_id=target_id,
        status=DeliveryStatus.QUEUED,
    )


async def _resolve_subscription_targets(message: BusMessage) -> set[str]:
    """Find agents whose subscriptions match the message.

    Matching considers message type, topic, **and** MMCNC scope.

    Scope rules:
    - Message with **no scope** → delivered to every matching subscriber
      (backward compatible).
    - Subscriber with **no scope** → receives all messages regardless of
      the message's scope (backward compatible).
    - Both have scopes → the subscriber's scope must *overlap* the
      message's scope.  A subscriber's scope overlaps when every non-None
      field in the subscriber's scope equals the corresponding field in
      the message's scope.  In other words a subscriber scoped to a
      macrocosm sees everything inside that macrocosm (microcosms &
      clusters), while a subscriber scoped to a cluster only sees that
      exact cluster.
    """
    all_subs = await repo.get_all_subscriptions()
    matched: set[str] = set()
    for sub in all_subs:
        # Skip sender's own subscriptions
        if sub["agent_id"] == message.sender_id:
            continue

        types = sub.get("message_types", [])
        topics = sub.get("topics", [])

        type_match = (not types) or (message.message_type.value in types)
        topic_match = (not topics) or (message.topic and message.topic in topics)

        if not (type_match and topic_match):
            continue

        # --- Scope filtering --------------------------------------------------
        sub_scope_data = sub.get("scope")
        sub_scope: Optional[BusScope] = None
        if sub_scope_data:
            if isinstance(sub_scope_data, BusScope):
                sub_scope = sub_scope_data
            elif isinstance(sub_scope_data, dict):
                sub_scope = BusScope(**sub_scope_data)

        if not _scopes_overlap(message.scope, sub_scope):
            continue

        matched.add(sub["agent_id"])
    return matched


def _scopes_overlap(
    msg_scope: Optional[BusScope],
    sub_scope: Optional[BusScope],
) -> bool:
    """Return True when a subscriber should receive a message given their scopes.

    Rules:
    1. Message has no scope → all subscribers match (broadcast).
    2. Subscriber has no scope → matches every message (wildcard listener).
    3. Both scoped → the subscriber's specified fields must equal the
       message's corresponding fields.  A subscriber may specify *fewer*
       fields (e.g. only macrocosm_id) to listen at a higher level;
       unset fields on the subscriber side are treated as wildcards.
    """
    # Rule 1 — unscoped message reaches everyone
    if msg_scope is None:
        return True

    # Rule 2 — unscoped subscriber hears everything
    if sub_scope is None:
        return True

    # Rule 3 — field-by-field comparison for subscriber's non-None fields
    for field in ("macrocosm_id", "microcosm_id", "cluster_id"):
        sub_val = getattr(sub_scope, field)
        if sub_val is not None:
            msg_val = getattr(msg_scope, field)
            if msg_val != sub_val:
                return False
    return True


def _parse_mentions(text: str) -> set[str]:
    """Extract @agent_id mentions from text."""
    if not text:
        return set()
    return set(re.findall(r"@([\w\-]+)", text))
