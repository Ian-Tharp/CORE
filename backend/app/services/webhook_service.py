"""
CORE Webhook Service - Async callbacks for task completion.

Provides:
- Register webhook endpoints (persisted to database)
- Fire webhooks on events with event-type filtering
- Retry logic with exponential backoff
- Webhook history and status tracking
- HMAC signature verification
- Delivery pruning for old records
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from enum import Enum
from uuid import uuid4

import httpx
import logging

from app.repository import webhook_repository

logger = logging.getLogger(__name__)


class WebhookEvent(str, Enum):
    """Webhook event types."""
    WILDCARD = "*"
    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    NODE_STARTED = "node.started"
    NODE_COMPLETED = "node.completed"
    STEP_EXECUTED = "step.executed"
    AGENT_STATUS_CHANGED = "agent.status_changed"


class WebhookRegistration:
    """A registered webhook endpoint."""

    def __init__(
        self,
        url: str,
        events: List[WebhookEvent],
        secret: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        max_retries: int = 3,
        webhook_id: Optional[str] = None,
    ):
        self.id = webhook_id or str(uuid4())[:8]
        self.url = url
        self.events = events
        self.secret = secret
        self.headers = headers or {}
        self.name = name or f"webhook-{self.id}"
        self.max_retries = max_retries
        self.created_at = datetime.now(timezone.utc)
        self.is_active = True
        self.delivery_count = 0
        self.failure_count = 0
        self.last_delivery = None
        self.last_error = None

    def matches_event(self, event: WebhookEvent) -> bool:
        """Check if this webhook should receive the event."""
        if not self.is_active:
            return False
        # Match if subscribed to wildcard or the specific event
        return WebhookEvent.WILDCARD in self.events or event in self.events

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "events": [e.value for e in self.events],
            "is_active": self.is_active,
            "delivery_count": self.delivery_count,
            "failure_count": self.failure_count,
            "created_at": self.created_at.isoformat(),
            "last_delivery": self.last_delivery.isoformat() if self.last_delivery else None,
            "last_error": self.last_error,
        }

    @classmethod
    def from_db_dict(cls, data: Dict[str, Any]) -> "WebhookRegistration":
        """Reconstruct from a database row dict."""
        events = []
        for e in data.get("events", []):
            try:
                events.append(WebhookEvent(e))
            except ValueError:
                logger.warning(f"Unknown webhook event type: {e}")
        reg = cls(
            url=data["url"],
            events=events,
            secret=data.get("secret"),
            headers=data.get("headers", {}),
            name=data.get("name"),
            max_retries=data.get("max_retries", 3),
            webhook_id=data["id"],
        )
        reg.is_active = data.get("is_active", True)
        reg.delivery_count = data.get("delivery_count", 0)
        reg.failure_count = data.get("failure_count", 0)
        if data.get("last_delivery_at"):
            reg.last_delivery = (
                datetime.fromisoformat(data["last_delivery_at"])
                if isinstance(data["last_delivery_at"], str)
                else data["last_delivery_at"]
            )
        reg.last_error = data.get("last_error")
        if data.get("created_at"):
            reg.created_at = (
                datetime.fromisoformat(data["created_at"])
                if isinstance(data["created_at"], str)
                else data["created_at"]
            )
        return reg


class WebhookDelivery:
    """Record of a webhook delivery attempt."""

    def __init__(
        self,
        webhook_id: str,
        event: WebhookEvent,
        payload: Dict[str, Any],
    ):
        self.id = str(uuid4())
        self.webhook_id = webhook_id
        self.event = event
        self.payload = payload
        self.attempts = 0
        self.status_code: Optional[int] = None
        self.response_body: Optional[str] = None
        self.error: Optional[str] = None
        self.delivered_at: Optional[datetime] = None
        self.created_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "webhook_id": self.webhook_id,
            "event": self.event.value,
            "attempts": self.attempts,
            "status_code": self.status_code,
            "error": self.error,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "created_at": self.created_at.isoformat(),
        }


class WebhookService:
    """
    Manages webhook registrations and deliveries.

    Registrations are persisted to the database and loaded on startup.
    Deliveries are recorded asynchronously for history/debugging.
    """

    def __init__(self):
        self.webhooks: Dict[str, WebhookRegistration] = {}
        self._delivery_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._http_client: Optional[httpx.AsyncClient] = None

    async def start(self):
        """Start the webhook delivery worker and load registrations from DB."""
        self._http_client = httpx.AsyncClient(timeout=30.0)
        self._worker_task = asyncio.create_task(self._delivery_worker())

        # Load persisted registrations
        try:
            rows = await webhook_repository.list_registrations(active_only=False)
            for row in rows:
                reg = WebhookRegistration.from_db_dict(row)
                self.webhooks[reg.id] = reg
            logger.info(f"Webhook service started, loaded {len(rows)} registrations")
        except Exception as e:
            logger.warning(f"Webhook service started (DB load failed: {e})")

    async def stop(self):
        """Stop the webhook delivery worker."""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        if self._http_client:
            await self._http_client.aclose()

        logger.info("Webhook service stopped")

    async def register(
        self,
        url: str,
        events: List[str],
        secret: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
    ) -> WebhookRegistration:
        """
        Register a new webhook and persist it to the database.

        Args:
            url: The endpoint URL to call
            events: List of event types to receive (use "*" for all)
            secret: Optional HMAC secret for signature verification
            headers: Optional additional headers to send
            name: Optional human-readable name

        Returns:
            The registered webhook
        """
        webhook_events = [WebhookEvent(e) for e in events]

        webhook = WebhookRegistration(
            url=url,
            events=webhook_events,
            secret=secret,
            headers=headers,
            name=name,
        )

        self.webhooks[webhook.id] = webhook

        # Persist to database (best-effort; service works in-memory if DB fails)
        await webhook_repository.create_registration(
            webhook_id=webhook.id,
            name=webhook.name,
            url=url,
            events=events,
            secret=secret,
            headers=headers,
            max_retries=webhook.max_retries,
        )

        logger.info(f"Registered webhook {webhook.name} ({webhook.id}) for {url}")
        return webhook

    async def unregister(self, webhook_id: str) -> bool:
        """Unregister a webhook and remove from database."""
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            await webhook_repository.delete_registration(webhook_id)
            logger.info(f"Unregistered webhook {webhook_id}")
            return True
        return False

    def get_webhook(self, webhook_id: str) -> Optional[WebhookRegistration]:
        """Get a webhook by ID."""
        return self.webhooks.get(webhook_id)

    def list_webhooks(self) -> List[WebhookRegistration]:
        """List all registered webhooks."""
        return list(self.webhooks.values())

    async def update_webhook(
        self,
        webhook_id: str,
        **fields,
    ) -> Optional[WebhookRegistration]:
        """Update a webhook's configuration."""
        webhook = self.webhooks.get(webhook_id)
        if not webhook:
            return None

        # Update in-memory object
        if "url" in fields:
            webhook.url = fields["url"]
        if "name" in fields:
            webhook.name = fields["name"]
        if "events" in fields:
            webhook.events = [WebhookEvent(e) for e in fields["events"]]
        if "secret" in fields:
            webhook.secret = fields["secret"]
        if "headers" in fields:
            webhook.headers = fields["headers"]
        if "is_active" in fields:
            webhook.is_active = fields["is_active"]
        if "max_retries" in fields:
            webhook.max_retries = fields["max_retries"]

        # Persist changes
        await webhook_repository.update_registration(webhook_id, **fields)
        return webhook

    async def fire(
        self,
        event: WebhookEvent,
        payload: Dict[str, Any],
        run_id: Optional[str] = None,
    ):
        """
        Fire a webhook event to all matching registrations.

        Args:
            event: The event type
            payload: The event payload
            run_id: Optional run ID for context
        """
        full_payload = {
            "event": event.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            **payload,
        }

        for webhook in self.webhooks.values():
            if webhook.matches_event(event):
                delivery = WebhookDelivery(
                    webhook_id=webhook.id,
                    event=event,
                    payload=full_payload,
                )
                await self._delivery_queue.put((webhook, delivery))
                logger.debug(f"Queued webhook delivery {delivery.id} for {webhook.name}")

    async def _delivery_worker(self):
        """Background worker that processes webhook deliveries."""
        logger.info("Webhook delivery worker started")

        while True:
            try:
                webhook, delivery = await self._delivery_queue.get()
                await self._deliver(webhook, delivery)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Webhook worker error: {e}")

    async def _deliver(self, webhook: WebhookRegistration, delivery: WebhookDelivery):
        """Attempt to deliver a webhook with retries."""
        retry_delays = [1, 5, 30]

        for attempt in range(webhook.max_retries):
            delivery.attempts = attempt + 1

            try:
                headers = {
                    "Content-Type": "application/json",
                    "X-Webhook-ID": webhook.id,
                    "X-Delivery-ID": delivery.id,
                    "X-Event-Type": delivery.event.value,
                    **webhook.headers,
                }

                # HMAC signature for payload verification
                if webhook.secret:
                    payload_bytes = json.dumps(delivery.payload).encode()
                    signature = hmac.new(
                        webhook.secret.encode(),
                        payload_bytes,
                        hashlib.sha256,
                    ).hexdigest()
                    headers["X-Webhook-Signature"] = f"sha256={signature}"

                response = await self._http_client.post(
                    webhook.url,
                    json=delivery.payload,
                    headers=headers,
                )

                delivery.status_code = response.status_code
                delivery.response_body = response.text[:500]

                if response.is_success:
                    delivery.delivered_at = datetime.now(timezone.utc)
                    webhook.delivery_count += 1
                    webhook.last_delivery = delivery.delivered_at

                    # Persist success
                    await webhook_repository.increment_delivery_count(webhook.id)
                    await webhook_repository.record_delivery(
                        delivery_id=delivery.id,
                        webhook_id=webhook.id,
                        event=delivery.event.value,
                        payload=delivery.payload,
                        attempts=delivery.attempts,
                        status_code=delivery.status_code,
                        response_body=delivery.response_body,
                        delivered_at=delivery.delivered_at,
                    )

                    logger.info(
                        f"Webhook {webhook.name} delivered: {delivery.event.value} "
                        f"(attempt {attempt + 1}, status {response.status_code})"
                    )
                    return
                else:
                    delivery.error = f"HTTP {response.status_code}"
                    logger.warning(
                        f"Webhook {webhook.name} failed with {response.status_code} "
                        f"(attempt {attempt + 1})"
                    )

            except Exception as e:
                delivery.error = str(e)
                logger.warning(
                    f"Webhook {webhook.name} error: {e} (attempt {attempt + 1})"
                )

            # Retry with backoff (except last attempt)
            if attempt < webhook.max_retries - 1:
                delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                await asyncio.sleep(delay)

        # All retries exhausted
        webhook.failure_count += 1
        webhook.last_error = delivery.error

        # Persist failure
        await webhook_repository.increment_failure_count(webhook.id, delivery.error or "unknown")
        await webhook_repository.record_delivery(
            delivery_id=delivery.id,
            webhook_id=webhook.id,
            event=delivery.event.value,
            payload=delivery.payload,
            attempts=delivery.attempts,
            status_code=delivery.status_code,
            response_body=delivery.response_body,
            error=delivery.error,
        )

        logger.error(
            f"Webhook {webhook.name} failed after {webhook.max_retries} attempts"
        )

    async def get_recent_deliveries(
        self,
        webhook_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get recent delivery attempts from the database."""
        return await webhook_repository.get_deliveries(
            webhook_id=webhook_id,
            limit=limit,
        )

    async def get_stats(self) -> Dict[str, Any]:
        """Get webhook service statistics."""
        db_stats = await webhook_repository.get_delivery_stats(hours=24)
        return {
            "registered_webhooks": len(self.webhooks),
            "active_webhooks": sum(1 for w in self.webhooks.values() if w.is_active),
            "queue_size": self._delivery_queue.qsize(),
            "last_24h": db_stats,
        }

    async def prune_deliveries(self, keep_days: int = 30) -> int:
        """Prune old delivery records from the database."""
        return await webhook_repository.prune_old_deliveries(keep_days)


# Global webhook service instance
_webhook_service: Optional[WebhookService] = None


def get_webhook_service() -> WebhookService:
    """Get the global webhook service instance."""
    global _webhook_service
    if _webhook_service is None:
        _webhook_service = WebhookService()
    return _webhook_service


async def init_webhook_service():
    """Initialize and start the webhook service."""
    service = get_webhook_service()
    await service.start()


async def shutdown_webhook_service():
    """Shutdown the webhook service."""
    if _webhook_service:
        await _webhook_service.stop()