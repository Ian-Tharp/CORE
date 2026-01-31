"""
REST endpoints for the Inter-Agent Communication Bus.

Provides API for publishing messages, managing subscriptions,
registering external agents, receiving webhooks, and observing metrics.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from typing import Any, Dict, List

from app.models.bus_models import (
    BroadcastRequest,
    BusMessage,
    BusMetrics,
    DeliveryReceipt,
    ExternalAgentRegistration,
    Subscription,
    SubscriptionCreate,
    WebhookIncoming,
    MessageType,
)
from app.services import bus_service

router = APIRouter(prefix="/bus", tags=["bus"])


# =============================================================================
# PUBLISHING
# =============================================================================

@router.post("/publish", status_code=status.HTTP_201_CREATED, response_model=DeliveryReceipt)
async def publish_message(message: BusMessage) -> DeliveryReceipt:
    """Publish a message on the Communication Bus."""
    try:
        return await bus_service.publish(message)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to publish message: {exc}",
        )


@router.post("/broadcast", status_code=status.HTTP_201_CREATED, response_model=DeliveryReceipt)
async def broadcast_message(request: BroadcastRequest) -> DeliveryReceipt:
    """Broadcast a message to all subscribers of a topic."""
    try:
        return await bus_service.broadcast(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to broadcast message: {exc}",
        )


# =============================================================================
# MESSAGE RETRIEVAL (queued / offline)
# =============================================================================

@router.get("/messages/{agent_id}", status_code=status.HTTP_200_OK)
async def get_queued_messages(agent_id: str) -> Dict[str, Any]:
    """Get queued (offline) messages for an agent and drain the queue."""
    try:
        messages = await bus_service.drain_queue(agent_id)
        return {"agent_id": agent_id, "messages": messages, "count": len(messages)}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to drain queue: {exc}",
        )


# =============================================================================
# SUBSCRIPTIONS
# =============================================================================

@router.post("/subscribe", status_code=status.HTTP_201_CREATED, response_model=Subscription)
async def create_subscription(sub: SubscriptionCreate) -> Subscription:
    """Create a subscription for an agent."""
    try:
        return await bus_service.subscribe(sub.agent_id, sub)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create subscription: {exc}",
        )


@router.delete("/subscribe/{subscription_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_subscription(subscription_id: str) -> None:
    """Remove a subscription."""
    deleted = await bus_service.unsubscribe("", subscription_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Subscription {subscription_id} not found",
        )


@router.get("/subscriptions/{agent_id}", status_code=status.HTTP_200_OK)
async def list_subscriptions(agent_id: str) -> Dict[str, Any]:
    """List all subscriptions for an agent."""
    subs = await bus_service.get_subscriptions(agent_id)
    return {"agent_id": agent_id, "subscriptions": [s.model_dump() for s in subs]}


# =============================================================================
# EXTERNAL AGENTS
# =============================================================================

@router.post("/external-agents", status_code=status.HTTP_201_CREATED)
async def register_external_agent(
    registration: ExternalAgentRegistration,
) -> Dict[str, Any]:
    """Register an external agent (e.g. Vigil) with a webhook URL."""
    try:
        result = await bus_service.register_external_agent(registration)
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register external agent: {exc}",
        )


@router.get("/external-agents", status_code=status.HTTP_200_OK)
async def list_external_agents() -> Dict[str, Any]:
    """List all registered external agents."""
    agents = await bus_service.list_external_agents()
    return {"external_agents": agents, "count": len(agents)}


@router.delete("/external-agents/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def deregister_external_agent(agent_id: str) -> None:
    """Deregister an external agent."""
    deleted = await bus_service.deregister_external_agent(agent_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"External agent {agent_id} not found",
        )


# =============================================================================
# METRICS
# =============================================================================

@router.get("/metrics", status_code=status.HTTP_200_OK, response_model=BusMetrics)
async def get_metrics() -> BusMetrics:
    """Get Communication Bus throughput and health metrics."""
    try:
        return await bus_service.get_metrics()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve metrics: {exc}",
        )


# =============================================================================
# WEBHOOK RECEIVE (external agents → bus)
# =============================================================================

@router.post("/webhook/receive", status_code=status.HTTP_201_CREATED)
async def webhook_receive(incoming: WebhookIncoming) -> Dict[str, Any]:
    """
    Receive a message from an external agent into the bus.

    This is the counterpart to webhook delivery — external agents POST here
    to inject messages into the bus.  Only registered external agents may
    send messages; the ``sender_id`` must match an existing registration.
    """
    # Validate sender is a registered external agent
    sender = await bus_service.get_external_agent(incoming.sender_id)
    if not sender:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Sender '{incoming.sender_id}' is not a registered external agent",
        )

    message = BusMessage(
        sender_id=incoming.sender_id,
        recipients=incoming.recipients,
        message_type=incoming.message_type,
        topic=incoming.topic,
        payload=incoming.payload,
        priority=incoming.priority,
        correlation_id=incoming.correlation_id,
        reply_to=incoming.reply_to,
    )
    try:
        receipt = await bus_service.publish(message)
        return {
            "status": "accepted",
            "message_id": message.id,
            "delivery": receipt.model_dump(),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process incoming webhook: {exc}",
        )
