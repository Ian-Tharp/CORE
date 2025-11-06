"""
WebSocket Connection Manager for Communication Commons.

Simple, standard FastAPI WebSocket implementation following official patterns.
Manages connections, subscriptions, and broadcasting.
"""

from __future__ import annotations

from typing import Dict, Set
from fastapi import WebSocket
from collections import defaultdict
import logging
import json

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for real-time communication.

    Follows standard FastAPI WebSocket patterns - keep it simple.
    Each instance can have one WebSocket connection.
    """

    def __init__(self):
        # instance_id -> WebSocket connection
        self.active_connections: Dict[str, WebSocket] = {}

        # channel_id -> Set of instance_ids subscribed
        self.channel_subscribers: Dict[str, Set[str]] = defaultdict(set)

    async def connect(self, instance_id: str, websocket: WebSocket):
        """Accept WebSocket connection and register instance."""
        await websocket.accept()
        self.active_connections[instance_id] = websocket
        logger.info(f"WebSocket connected: {instance_id}")

        # Send connection confirmation
        await self.send_personal_message(
            instance_id,
            {
                "type": "connection",
                "instance_id": instance_id,
                "message": "Connected to Communication Commons"
            }
        )

    def disconnect(self, instance_id: str):
        """Remove instance connection."""
        if instance_id in self.active_connections:
            del self.active_connections[instance_id]
            logger.info(f"WebSocket disconnected: {instance_id}")

        # Remove from all channel subscriptions
        for channel_id in list(self.channel_subscribers.keys()):
            self.channel_subscribers[channel_id].discard(instance_id)

    def subscribe_to_channel(self, instance_id: str, channel_id: str):
        """Subscribe instance to a channel."""
        self.channel_subscribers[channel_id].add(instance_id)
        logger.debug(f"{instance_id} subscribed to {channel_id}")

    def unsubscribe_from_channel(self, instance_id: str, channel_id: str):
        """Unsubscribe instance from a channel."""
        self.channel_subscribers[channel_id].discard(instance_id)
        logger.debug(f"{instance_id} unsubscribed from {channel_id}")

    async def send_personal_message(self, instance_id: str, message: dict):
        """Send message to a specific instance."""
        websocket = self.active_connections.get(instance_id)
        if websocket:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to {instance_id}: {e}")
                self.disconnect(instance_id)

    async def broadcast_to_channel(self, channel_id: str, message: dict):
        """Broadcast message to all subscribers of a channel."""
        subscribers = self.channel_subscribers.get(channel_id, set())
        logger.debug(f"Broadcasting to channel {channel_id}: {len(subscribers)} subscribers")

        disconnected = []
        for instance_id in subscribers:
            websocket = self.active_connections.get(instance_id)
            if websocket:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to {instance_id}: {e}")
                    disconnected.append(instance_id)
            else:
                disconnected.append(instance_id)

        # Clean up disconnected instances
        for instance_id in disconnected:
            self.disconnect(instance_id)

    async def broadcast_to_all(self, message: dict):
        """Broadcast message to all connected instances."""
        logger.debug(f"Broadcasting to all: {len(self.active_connections)} instances")

        disconnected = []
        for instance_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {instance_id}: {e}")
                disconnected.append(instance_id)

        # Clean up disconnected instances
        for instance_id in disconnected:
            self.disconnect(instance_id)

    async def broadcast_presence_update(self, instance_id: str, status: str, activity: str | None = None, phase: int | None = None):
        """Broadcast presence update to all connected instances."""
        message = {
            "type": "presence",
            "instance_id": instance_id,
            "status": status,
            "activity": activity,
            "phase": phase
        }
        await self.broadcast_to_all(message)


# Global singleton
manager = ConnectionManager()
