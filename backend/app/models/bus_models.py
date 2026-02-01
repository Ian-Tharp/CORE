"""
Inter-Agent Communication Bus Models

Pydantic models for the programmatic agent-to-agent messaging backbone.
The Bus sits alongside the Communication Commons (UI chat) and provides
structured messaging, pub/sub subscriptions, external agent webhooks,
and delivery tracking.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# ENUMS
# =============================================================================

class MessageType(str, Enum):
    """Types of messages that can be sent on the bus."""
    TASK_REQUEST = "task_request"
    TASK_RESULT = "task_result"
    HELP_REQUEST = "help_request"
    STATUS_UPDATE = "status_update"
    CAPABILITY_QUERY = "capability_query"
    CAPABILITY_RESPONSE = "capability_response"
    HEARTBEAT = "heartbeat"
    BROADCAST = "broadcast"
    CATALYST_PHASE_COMPLETE = "catalyst_phase_complete"
    LEVEL_TRANSITION = "level_transition"


class MessagePriority(str, Enum):
    """Priority levels for bus messages."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class DeliveryStatus(str, Enum):
    """Status of message delivery."""
    PENDING = "pending"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"
    QUEUED = "queued"


# =============================================================================
# SCOPING
# =============================================================================

class BusScope(BaseModel):
    """
    MMCNC hierarchy scope for targeted message delivery.

    Messages can be addressed to a specific level of the Macrocosm / Microcosm /
    Cluster hierarchy.  When a scope is attached to a message, only subscribers
    whose own scope *overlaps* will receive it.  Overlap rules:

    - A subscriber scoped to a **macrocosm** receives messages for that
      macrocosm and any of its child microcosms / clusters.
    - A subscriber scoped to a **microcosm** receives messages for that
      microcosm and any of its child clusters.
    - A subscriber scoped to a **cluster** only receives messages for that
      exact cluster.
    - A subscriber with **no scope** (None) receives all messages (backward
      compatible).

    Fields are cumulative — setting ``cluster_id`` implies the message lives
    inside a specific microcosm inside a specific macrocosm, so callers should
    populate the parent IDs as well for correct hierarchical matching.
    """
    macrocosm_id: Optional[str] = None
    microcosm_id: Optional[str] = None
    cluster_id: Optional[str] = None


# =============================================================================
# CORE MODELS
# =============================================================================

class BusMessage(BaseModel):
    """
    A structured message sent via the Communication Bus.

    Supports direct agent-to-agent messaging (via ``recipients``), topic-based
    pub/sub (via ``topic``), and broadcast (empty recipients + topic).
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = Field(..., min_length=1, description="Agent ID of the sender")
    recipients: List[str] = Field(
        default_factory=list,
        description="Target agent IDs. Empty for topic-only / broadcast."
    )
    message_type: MessageType = Field(..., description="Semantic type of the message")
    topic: Optional[str] = Field(
        None,
        description="Optional topic for pub/sub routing"
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary structured payload"
    )
    priority: MessagePriority = Field(
        default=MessagePriority.NORMAL,
        description="Delivery priority"
    )
    correlation_id: Optional[str] = Field(
        None,
        description="Links request/response pairs together"
    )
    reply_to: Optional[str] = Field(
        None,
        description="Message ID this is a reply to"
    )
    scope: Optional[BusScope] = Field(
        None,
        description="MMCNC hierarchy scope for targeted delivery (None = all subscribers)"
    )
    ttl_seconds: Optional[int] = Field(
        None,
        ge=1,
        description="Time-to-live in seconds; expired messages are dropped"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# SUBSCRIPTION
# =============================================================================

class Subscription(BaseModel):
    """An agent's subscription to a set of message types and/or topics."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = Field(..., min_length=1)
    message_types: List[MessageType] = Field(
        default_factory=list,
        description="Message types to subscribe to (empty = all)"
    )
    topics: List[str] = Field(
        default_factory=list,
        description="Topics to subscribe to (empty = all for matched types)"
    )
    scope: Optional[BusScope] = Field(
        None,
        description="MMCNC scope filter — only receive messages whose scope overlaps"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SubscriptionCreate(BaseModel):
    """Request body for creating a subscription."""
    agent_id: str = Field(..., min_length=1)
    message_types: List[MessageType] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    scope: Optional[BusScope] = None


# =============================================================================
# EXTERNAL AGENTS (webhooks)
# =============================================================================

class WebhookConfig(BaseModel):
    """Configuration for an external agent's webhook endpoint."""
    url: str = Field(..., description="HTTPS URL to POST messages to")
    secret: Optional[str] = Field(
        None,
        description="Shared secret for HMAC signature verification"
    )
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_backoff_base_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Base delay (ms) for exponential back-off"
    )
    timeout_ms: int = Field(
        default=5000,
        ge=1000,
        le=30000,
        description="HTTP request timeout in milliseconds"
    )


class ExternalAgentRegistration(BaseModel):
    """Registration payload for an external agent (e.g. Vigil on Clawdbot)."""
    agent_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    webhook: WebhookConfig
    created_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# DELIVERY
# =============================================================================

class DeliveryReceipt(BaseModel):
    """Confirmation of message delivery."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_id: str
    recipient_id: str
    status: DeliveryStatus = DeliveryStatus.PENDING
    delivered_at: Optional[datetime] = None
    error: Optional[str] = None


# =============================================================================
# METRICS
# =============================================================================

class BusMetrics(BaseModel):
    """Aggregate health / throughput metrics for the bus."""
    total_messages_published: int = 0
    total_messages_delivered: int = 0
    total_messages_failed: int = 0
    total_messages_queued: int = 0
    active_subscriptions: int = 0
    external_agents_count: int = 0
    avg_delivery_latency_ms: Optional[float] = None
    messages_by_type: Dict[str, int] = Field(default_factory=dict)
    messages_by_priority: Dict[str, int] = Field(default_factory=dict)


# =============================================================================
# WEBHOOK INCOMING
# =============================================================================

class WebhookIncoming(BaseModel):
    """Payload an external agent sends INTO the bus via the webhook receive endpoint."""
    sender_id: str = Field(..., min_length=1)
    recipients: List[str] = Field(default_factory=list)
    message_type: MessageType
    topic: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    priority: MessagePriority = Field(default=MessagePriority.NORMAL)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None


# =============================================================================
# BROADCAST REQUEST
# =============================================================================

class BroadcastRequest(BaseModel):
    """Request body for broadcasting a message to a topic."""
    sender_id: str = Field(..., min_length=1)
    topic: str = Field(..., min_length=1)
    message_type: MessageType = Field(default=MessageType.BROADCAST)
    payload: Dict[str, Any] = Field(default_factory=dict)
    priority: MessagePriority = Field(default=MessagePriority.NORMAL)
