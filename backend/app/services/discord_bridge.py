"""
Discord Bridge Service

Connects Discord to CORE's Communication Commons, enabling bidirectional
message flow between Discord channels and CORE's internal communication system.

This is CORE's native Discord gateway - no external dependencies like OpenClaw needed.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from enum import Enum

import discord
from discord import Client, Intents, Message, TextChannel, DMChannel, Thread
from discord.errors import LoginFailure, ConnectionClosed

from app.config.discord import (
    DiscordConfig,
    DiscordChannelMapping,
    get_config,
    load_config_from_store,
    persist_channel_mapping,
)
from app.repository import communication_repository as comm_repo

logger = logging.getLogger(__name__)


class BridgeStatus(str, Enum):
    """Status of the Discord bridge."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class DiscordBridgeService:
    """
    Bridges Discord to CORE's Communication Commons.

    Features:
    - Bidirectional message routing
    - Channel mapping (Discord ↔ CORE)
    - User allowlist support
    - Automatic reconnection
    - Presence synchronization
    """

    def __init__(self, config: Optional[DiscordConfig] = None):
        """
        Initialize the Discord bridge.

        Args:
            config: Discord configuration. If None, loads from environment.
        """
        self.config = config or get_config()
        self._client: Optional[Client] = None
        self._status = BridgeStatus.DISCONNECTED
        self._last_error: Optional[str] = None
        self._connected_at: Optional[datetime] = None
        self._reconnect_attempts = 0
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # WebSocket subscription for outbound messages
        self._ws_subscription_id: Optional[str] = None

        # Message handlers that external code can register
        self._message_handlers: List[Callable] = []

        # Track which CORE channels we're bridging
        self._bridged_core_channels: set = set()

    @property
    def status(self) -> BridgeStatus:
        return self._status

    @property
    def is_connected(self) -> bool:
        return self._status == BridgeStatus.CONNECTED and self._client is not None

    def get_status_info(self) -> Dict[str, Any]:
        """Get detailed status information."""
        return {
            "status": self._status.value,
            "connected": self.is_connected,
            "connected_at": (
                self._connected_at.isoformat() if self._connected_at else None
            ),
            "last_error": self._last_error,
            "reconnect_attempts": self._reconnect_attempts,
            "bot_user": (
                self._client.user.name if self._client and self._client.user else None
            ),
            "guilds": len(self._client.guilds) if self._client else 0,
            "channel_mappings": len(self.config.channel_mappings),
            "bridged_core_channels": list(self._bridged_core_channels),
        }

    async def start(self) -> bool:
        """
        Start the Discord bridge.

        Returns:
            True if started successfully, False otherwise.
        """
        self.config = await load_config_from_store()

        if not self.config.enabled:
            logger.info("Discord bridge is disabled")
            return False

        if not self.config.bot_token:
            logger.error("Discord bot token not configured")
            self._status = BridgeStatus.ERROR
            self._last_error = "Bot token not configured"
            return False

        if self._running:
            logger.warning("Discord bridge is already running")
            return True

        self._running = True
        self._status = BridgeStatus.CONNECTING

        # Create Discord client with intents
        intents = Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.members = True

        self._client = Client(intents=intents)
        self._setup_event_handlers()

        # Start the client in a background task
        self._task = asyncio.create_task(self._run_client())

        return True

    async def stop(self) -> None:
        """Stop the Discord bridge."""
        self._running = False

        if self._client:
            await self._client.close()
            self._client = None

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        self._status = BridgeStatus.DISCONNECTED
        self._connected_at = None
        logger.info("Discord bridge stopped")

    async def _run_client(self) -> None:
        """Run the Discord client with reconnection logic."""
        while self._running:
            try:
                await self._client.start(self.config.bot_token)
            except LoginFailure as e:
                self._status = BridgeStatus.ERROR
                self._last_error = f"Login failed: {e}"
                logger.error(f"Discord login failed: {e}")
                break
            except ConnectionClosed:
                if not self._running:
                    break
                self._status = BridgeStatus.RECONNECTING
                self._reconnect_attempts += 1

                if self._reconnect_attempts > self.config.max_reconnect_attempts:
                    self._status = BridgeStatus.ERROR
                    self._last_error = "Max reconnection attempts exceeded"
                    logger.error("Discord bridge exceeded max reconnection attempts")
                    break

                logger.warning(
                    f"Discord connection closed, reconnecting in {self.config.reconnect_delay_seconds}s..."
                )
                await asyncio.sleep(self.config.reconnect_delay_seconds)
            except Exception as e:
                self._status = BridgeStatus.ERROR
                self._last_error = str(e)
                logger.error(f"Discord bridge error: {e}")
                if self._running:
                    await asyncio.sleep(self.config.reconnect_delay_seconds)

    def _setup_event_handlers(self) -> None:
        """Set up Discord event handlers."""

        @self._client.event
        async def on_ready():
            self._status = BridgeStatus.CONNECTED
            self._connected_at = datetime.utcnow()
            self._reconnect_attempts = 0
            logger.info(f"Discord bridge connected as {self._client.user}")
            logger.info(f"Connected to {len(self._client.guilds)} guilds")

            # Initialize channel mappings
            await self._initialize_channel_mappings()

            # Subscribe to CORE WebSocket for outbound messages
            await self._subscribe_to_core_channels()

        @self._client.event
        async def on_message(message: Message):
            # Ignore messages from the bot itself
            if message.author == self._client.user:
                return

            # Ignore bots (optional, configurable later)
            if message.author.bot:
                return

            await self._handle_discord_message(message)

        @self._client.event
        async def on_disconnect():
            if self._status == BridgeStatus.CONNECTED:
                self._status = BridgeStatus.RECONNECTING
                logger.warning("Discord bridge disconnected")

    async def _initialize_channel_mappings(self) -> None:
        """Initialize channel mappings and create CORE channels as needed."""
        for discord_channel_id, mapping in self.config.channel_mappings.items():
            try:
                # Try to get Discord channel info
                channel = self._client.get_channel(int(discord_channel_id))
                if channel:
                    mapping.discord_channel_name = getattr(channel, "name", None)
                    if hasattr(channel, "guild") and channel.guild:
                        mapping.discord_guild_id = str(channel.guild.id)
                        mapping.discord_guild_name = channel.guild.name

                # Ensure CORE channel exists
                core_channel = await comm_repo.get_channel(mapping.core_channel_id)
                if not core_channel and self.config.auto_create_channels:
                    await comm_repo.create_channel(
                        channel_id=mapping.core_channel_id,
                        channel_type="context",
                        name=mapping.core_channel_name
                        or f"discord_{mapping.discord_channel_name or discord_channel_id}",
                        description=f"Bridged from Discord: {mapping.discord_channel_name or discord_channel_id}",
                        is_persistent=True,
                        is_public=True,
                        created_by=self.config.bot_instance_id,
                        initial_members=[],
                    )
                    logger.info(
                        f"Created CORE channel {mapping.core_channel_id} for Discord channel {discord_channel_id}"
                    )

                self._bridged_core_channels.add(mapping.core_channel_id)
                await persist_channel_mapping(mapping)

            except Exception as e:
                logger.error(
                    f"Error initializing channel mapping for {discord_channel_id}: {e}"
                )

    async def _subscribe_to_core_channels(self) -> None:
        """Subscribe to CORE channels for outbound messages."""
        # Note: This would integrate with WebSocket manager
        # For now, we'll poll or use the existing broadcast mechanism
        pass

    async def _handle_discord_message(self, message: Message) -> None:
        """
        Handle an inbound Discord message.

        Args:
            message: The Discord message to process.
        """
        # Check user allowlist
        if (
            self.config.allowed_users
            and str(message.author.id) not in self.config.allowed_users
        ):
            logger.debug(f"Ignoring message from non-allowed user {message.author.id}")
            return

        # Get channel mapping
        channel_id = str(message.channel.id)
        mapping = self.config.channel_mappings.get(channel_id)

        if not mapping:
            # Check if we should auto-create mapping
            if self.config.auto_create_channels:
                core_channel_id = f"discord_{channel_id}"
                mapping = DiscordChannelMapping(
                    discord_channel_id=channel_id,
                    discord_channel_name=getattr(message.channel, "name", None),
                    core_channel_id=core_channel_id,
                )
                self.config.channel_mappings[channel_id] = mapping

                # Create the CORE channel
                await comm_repo.create_channel(
                    channel_id=core_channel_id,
                    channel_type="context",
                    name=f"discord_{getattr(message.channel, 'name', channel_id)}",
                    description=f"Auto-bridged from Discord",
                    is_persistent=True,
                    is_public=True,
                    created_by=self.config.bot_instance_id,
                    initial_members=[],
                )
                self._bridged_core_channels.add(core_channel_id)
                await persist_channel_mapping(mapping)
            else:
                logger.debug(f"No mapping for Discord channel {channel_id}")
                return

        if not mapping.enabled:
            return

        # Check require_mention
        if mapping.require_mention and self._client.user not in message.mentions:
            return

        # Create message in CORE Communication Commons
        try:
            # Clean content (remove bot mention if present)
            content = message.content
            if self._client.user in message.mentions:
                content = content.replace(f"<@{self._client.user.id}>", "").strip()
                content = content.replace(f"<@!{self._client.user.id}>", "").strip()

            from app.services.communication_service import get_communication_service

            core_message = await get_communication_service().ingest_discord_message(
                mapping=mapping,
                discord_message=message,
                cleaned_content=content,
                message_prefix=self.config.message_prefix,
            )

            if core_message:
                logger.debug(
                    "Bridged Discord message %s to CORE channel %s",
                    message.id,
                    mapping.core_channel_id,
                )

        except Exception as e:
            logger.error(f"Error bridging Discord message: {e}")

    async def send_to_discord(
        self,
        discord_channel_id: str,
        content: str,
        reply_to_message_id: Optional[str] = None,
    ) -> List[str]:
        """
        Send a message to a Discord channel.

        Args:
            discord_channel_id: The Discord channel ID to send to.
            content: The message content.
            reply_to_message_id: Optional message ID to reply to.

        Returns:
            List of created Discord message IDs. Empty list indicates failure.
        """
        if not self.is_connected:
            logger.warning("Cannot send to Discord: bridge not connected")
            return []

        try:
            channel = self._client.get_channel(int(discord_channel_id))
            if not channel:
                logger.error(f"Discord channel {discord_channel_id} not found")
                return []

            if not isinstance(channel, (TextChannel, DMChannel, Thread)):
                logger.error(
                    f"Discord channel {discord_channel_id} is not a text channel"
                )
                return []

            # Add response prefix if configured
            if self.config.response_prefix:
                content = f"{self.config.response_prefix}{content}"

            # Split long messages (Discord limit is 2000 chars)
            chunks = self._split_message(content, 2000)
            created_message_ids: List[str] = []

            for chunk in chunks:
                if reply_to_message_id:
                    try:
                        reply_msg = await channel.fetch_message(
                            int(reply_to_message_id)
                        )
                        sent_message = await reply_msg.reply(chunk)
                    except:
                        sent_message = await channel.send(chunk)
                    reply_to_message_id = None  # Only reply to first chunk
                else:
                    sent_message = await channel.send(chunk)

                created_message_ids.append(str(sent_message.id))

            return created_message_ids

        except Exception as e:
            logger.error(f"Error sending to Discord: {e}")
            return []

    async def start_typing(self, discord_channel_id: str) -> None:
        """
        Show typing indicator in a Discord channel.

        Call this when an agent starts processing a request.
        The typing indicator will show for ~10 seconds or until a message is sent.

        Args:
            discord_channel_id: The Discord channel ID to show typing in.
        """
        if not self.is_connected:
            return

        try:
            channel = self._client.get_channel(int(discord_channel_id))
            if channel and isinstance(channel, (TextChannel, DMChannel, Thread)):
                await channel.typing()
                logger.debug(
                    f"Started typing indicator in Discord channel {discord_channel_id}"
                )
        except Exception as e:
            logger.debug(f"Failed to start typing indicator: {e}")

    def _split_message(self, content: str, max_length: int) -> List[str]:
        """Split a message into chunks that fit Discord's limit."""
        if len(content) <= max_length:
            return [content]

        chunks = []
        while content:
            if len(content) <= max_length:
                chunks.append(content)
                break

            # Try to split at a newline
            split_idx = content.rfind("\n", 0, max_length)
            if split_idx == -1:
                # Try to split at a space
                split_idx = content.rfind(" ", 0, max_length)
            if split_idx == -1:
                # Hard split
                split_idx = max_length

            chunks.append(content[:split_idx])
            content = content[split_idx:].lstrip()

        return chunks

    def add_channel_mapping(self, mapping: DiscordChannelMapping) -> None:
        """Add a channel mapping."""
        self.config.channel_mappings[mapping.discord_channel_id] = mapping
        self._bridged_core_channels.add(mapping.core_channel_id)

    def remove_channel_mapping(self, discord_channel_id: str) -> bool:
        """Remove a channel mapping."""
        if discord_channel_id in self.config.channel_mappings:
            mapping = self.config.channel_mappings.pop(discord_channel_id)
            # Don't remove from bridged_core_channels as other mappings may use it
            return True
        return False

    def get_channel_mappings(self) -> Dict[str, DiscordChannelMapping]:
        """Get all channel mappings."""
        return self.config.channel_mappings.copy()


# Singleton instance
_bridge_service: Optional[DiscordBridgeService] = None


def get_discord_bridge() -> DiscordBridgeService:
    """Get or create the Discord bridge singleton."""
    global _bridge_service
    if _bridge_service is None:
        _bridge_service = DiscordBridgeService()
    return _bridge_service


async def start_discord_bridge() -> bool:
    """Start the Discord bridge service."""
    return await get_discord_bridge().start()


async def stop_discord_bridge() -> None:
    """Stop the Discord bridge service."""
    await get_discord_bridge().stop()
