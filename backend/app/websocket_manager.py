"""
WebSocket Connection Manager for Communication Commons.

Simple, standard FastAPI WebSocket implementation following official patterns.
Manages connections, subscriptions, and broadcasting.

Improvements:
- Typing indicators
- Read receipts
- Connection heartbeats
- Reconnection tracking
"""

from __future__ import annotations

from typing import Dict, Set, Optional
from datetime import datetime
from fastapi import WebSocket
from collections import defaultdict
import asyncio
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
        
        # Typing indicators: channel_id -> {instance_id: expiry_time}
        self.typing_indicators: Dict[str, Dict[str, datetime]] = defaultdict(dict)
        
        # Read receipts: message_id -> {instance_id: read_time}
        self.read_receipts: Dict[str, Dict[str, datetime]] = defaultdict(dict)
        
        # Connection metadata: instance_id -> metadata
        self.connection_metadata: Dict[str, Dict] = {}
        
        # Heartbeat tracking
        self.last_heartbeat: Dict[str, datetime] = {}

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

    # =========================================================================
    # Typing Indicators
    # =========================================================================
    
    async def start_typing(self, instance_id: str, channel_id: str):
        """
        Indicate that an instance started typing in a channel.
        
        Typing indicators auto-expire after 5 seconds.
        """
        expiry = datetime.utcnow()
        self.typing_indicators[channel_id][instance_id] = expiry
        
        # Broadcast typing start to channel subscribers
        await self.broadcast_to_channel(channel_id, {
            "type": "typing_start",
            "instance_id": instance_id,
            "channel_id": channel_id,
            "timestamp": expiry.isoformat()
        })
        
        logger.debug(f"{instance_id} started typing in {channel_id}")
    
    async def stop_typing(self, instance_id: str, channel_id: str):
        """
        Indicate that an instance stopped typing.
        """
        if channel_id in self.typing_indicators:
            self.typing_indicators[channel_id].pop(instance_id, None)
        
        await self.broadcast_to_channel(channel_id, {
            "type": "typing_stop",
            "instance_id": instance_id,
            "channel_id": channel_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.debug(f"{instance_id} stopped typing in {channel_id}")
    
    def get_typing_users(self, channel_id: str) -> list:
        """
        Get list of users currently typing in a channel.
        
        Automatically cleans up expired typing indicators.
        """
        now = datetime.utcnow()
        typing_users = []
        expired = []
        
        for instance_id, started_at in self.typing_indicators.get(channel_id, {}).items():
            # Typing indicator expires after 5 seconds
            if (now - started_at).total_seconds() < 5:
                typing_users.append(instance_id)
            else:
                expired.append(instance_id)
        
        # Clean up expired
        for instance_id in expired:
            self.typing_indicators[channel_id].pop(instance_id, None)
        
        return typing_users

    # =========================================================================
    # Read Receipts
    # =========================================================================
    
    async def mark_read(self, instance_id: str, message_id: str, channel_id: str):
        """
        Mark a message as read by an instance.
        """
        read_time = datetime.utcnow()
        self.read_receipts[message_id][instance_id] = read_time
        
        # Broadcast read receipt to channel
        await self.broadcast_to_channel(channel_id, {
            "type": "read_receipt",
            "instance_id": instance_id,
            "message_id": message_id,
            "channel_id": channel_id,
            "read_at": read_time.isoformat()
        })
        
        logger.debug(f"{instance_id} read message {message_id}")
    
    def get_read_receipts(self, message_id: str) -> Dict[str, str]:
        """
        Get all read receipts for a message.
        
        Returns:
            Dict mapping instance_id to ISO timestamp
        """
        return {
            instance_id: read_time.isoformat()
            for instance_id, read_time in self.read_receipts.get(message_id, {}).items()
        }

    # =========================================================================
    # Heartbeat Management
    # =========================================================================
    
    async def heartbeat(self, instance_id: str):
        """
        Record a heartbeat from an instance.
        
        Used to track connection health and detect stale connections.
        """
        self.last_heartbeat[instance_id] = datetime.utcnow()
    
    def is_connection_alive(self, instance_id: str, max_age_seconds: int = 60) -> bool:
        """
        Check if a connection is alive based on last heartbeat.
        """
        if instance_id not in self.last_heartbeat:
            return instance_id in self.active_connections
        
        age = (datetime.utcnow() - self.last_heartbeat[instance_id]).total_seconds()
        return age < max_age_seconds
    
    async def cleanup_stale_connections(self, max_age_seconds: int = 60):
        """
        Disconnect instances that haven't sent a heartbeat recently.
        """
        stale = [
            instance_id for instance_id in self.active_connections
            if not self.is_connection_alive(instance_id, max_age_seconds)
        ]
        
        for instance_id in stale:
            logger.info(f"Cleaning up stale connection: {instance_id}")
            self.disconnect(instance_id)
            await self.broadcast_presence_update(instance_id, "offline")
        
        return len(stale)

    # =========================================================================
    # Connection Metadata
    # =========================================================================
    
    def set_metadata(self, instance_id: str, metadata: Dict):
        """Set metadata for a connection."""
        self.connection_metadata[instance_id] = {
            **self.connection_metadata.get(instance_id, {}),
            **metadata,
            "updated_at": datetime.utcnow().isoformat()
        }
    
    def get_metadata(self, instance_id: str) -> Optional[Dict]:
        """Get metadata for a connection."""
        return self.connection_metadata.get(instance_id)
    
    def get_connection_stats(self) -> Dict:
        """Get connection statistics."""
        return {
            "active_connections": len(self.active_connections),
            "channels_with_subscribers": len(self.channel_subscribers),
            "total_subscriptions": sum(len(s) for s in self.channel_subscribers.values()),
            "connections_with_metadata": len(self.connection_metadata),
            "typing_indicators_active": sum(len(t) for t in self.typing_indicators.values())
        }


# Global singleton
manager = ConnectionManager()
