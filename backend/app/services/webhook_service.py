"""
CORE Webhook Service - Async callbacks for task completion.

Provides:
- Register webhook endpoints
- Fire webhooks on events
- Retry logic with exponential backoff
- Webhook history and status tracking

RSI TODO: Add webhook signature verification (HMAC)
RSI TODO: Add webhook filtering by event type
RSI TODO: Persist webhook registrations to database
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum
from uuid import uuid4

import httpx
import logging

logger = logging.getLogger(__name__)


class WebhookEvent(str, Enum):
    """Webhook event types."""
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
        max_retries: int = 3
    ):
        self.id = str(uuid4())[:8]
        self.url = url
        self.events = events
        self.secret = secret
        self.headers = headers or {}
        self.name = name or f"webhook-{self.id}"
        self.max_retries = max_retries
        self.created_at = datetime.utcnow()
        self.is_active = True
        self.delivery_count = 0
        self.failure_count = 0
        self.last_delivery = None
        self.last_error = None
    
    def matches_event(self, event: WebhookEvent) -> bool:
        """Check if this webhook should receive the event."""
        if not self.is_active:
            return False
        return event in self.events or WebhookEvent.RUN_COMPLETED in self.events  # wildcard
    
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
            "last_error": self.last_error
        }


class WebhookDelivery:
    """Record of a webhook delivery attempt."""
    
    def __init__(
        self,
        webhook_id: str,
        event: WebhookEvent,
        payload: Dict[str, Any]
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
        self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "webhook_id": self.webhook_id,
            "event": self.event.value,
            "attempts": self.attempts,
            "status_code": self.status_code,
            "error": self.error,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "created_at": self.created_at.isoformat()
        }


class WebhookService:
    """
    Manages webhook registrations and deliveries.
    """
    
    def __init__(self):
        self.webhooks: Dict[str, WebhookRegistration] = {}
        self.deliveries: List[WebhookDelivery] = []
        self._delivery_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._http_client: Optional[httpx.AsyncClient] = None
    
    async def start(self):
        """Start the webhook delivery worker."""
        self._http_client = httpx.AsyncClient(timeout=30.0)
        self._worker_task = asyncio.create_task(self._delivery_worker())
        logger.info("Webhook service started")
    
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
    
    def register(
        self,
        url: str,
        events: List[str],
        secret: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        name: Optional[str] = None
    ) -> WebhookRegistration:
        """
        Register a new webhook.
        
        Args:
            url: The endpoint URL to call
            events: List of event types to receive
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
            name=name
        )
        
        self.webhooks[webhook.id] = webhook
        logger.info(f"Registered webhook {webhook.name} ({webhook.id}) for {url}")
        
        return webhook
    
    def unregister(self, webhook_id: str) -> bool:
        """Unregister a webhook."""
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            logger.info(f"Unregistered webhook {webhook_id}")
            return True
        return False
    
    def get_webhook(self, webhook_id: str) -> Optional[WebhookRegistration]:
        """Get a webhook by ID."""
        return self.webhooks.get(webhook_id)
    
    def list_webhooks(self) -> List[WebhookRegistration]:
        """List all registered webhooks."""
        return list(self.webhooks.values())
    
    async def fire(
        self,
        event: WebhookEvent,
        payload: Dict[str, Any],
        run_id: Optional[str] = None
    ):
        """
        Fire a webhook event.
        
        Args:
            event: The event type
            payload: The event payload
            run_id: Optional run ID for filtering
        """
        # Add metadata to payload
        full_payload = {
            "event": event.value,
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": run_id,
            **payload
        }
        
        # Queue delivery for each matching webhook
        for webhook in self.webhooks.values():
            if webhook.matches_event(event):
                delivery = WebhookDelivery(
                    webhook_id=webhook.id,
                    event=event,
                    payload=full_payload
                )
                self.deliveries.append(delivery)
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
        retry_delays = [1, 5, 30]  # Seconds between retries
        
        for attempt in range(webhook.max_retries):
            delivery.attempts = attempt + 1
            
            try:
                # Build headers
                headers = {
                    "Content-Type": "application/json",
                    "X-Webhook-ID": webhook.id,
                    "X-Delivery-ID": delivery.id,
                    "X-Event-Type": delivery.event.value,
                    **webhook.headers
                }
                
                # Add signature if secret configured
                if webhook.secret:
                    payload_bytes = json.dumps(delivery.payload).encode()
                    signature = hmac.new(
                        webhook.secret.encode(),
                        payload_bytes,
                        hashlib.sha256
                    ).hexdigest()
                    headers["X-Webhook-Signature"] = f"sha256={signature}"
                
                # Make the request
                response = await self._http_client.post(
                    webhook.url,
                    json=delivery.payload,
                    headers=headers
                )
                
                delivery.status_code = response.status_code
                delivery.response_body = response.text[:500]  # Truncate
                
                if response.is_success:
                    delivery.delivered_at = datetime.utcnow()
                    webhook.delivery_count += 1
                    webhook.last_delivery = delivery.delivered_at
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
        logger.error(
            f"Webhook {webhook.name} failed after {webhook.max_retries} attempts"
        )
    
    def get_recent_deliveries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent delivery attempts."""
        return [d.to_dict() for d in self.deliveries[-limit:]]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get webhook service statistics."""
        total_deliveries = len(self.deliveries)
        successful = sum(1 for d in self.deliveries if d.delivered_at)
        
        return {
            "registered_webhooks": len(self.webhooks),
            "active_webhooks": sum(1 for w in self.webhooks.values() if w.is_active),
            "total_deliveries": total_deliveries,
            "successful_deliveries": successful,
            "failed_deliveries": total_deliveries - successful,
            "success_rate": successful / total_deliveries if total_deliveries > 0 else 0,
            "queue_size": self._delivery_queue.qsize()
        }


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
