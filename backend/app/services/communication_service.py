"""
Shared Communication Commons message flow service.

Centralizes message creation, broadcast, Discord forwarding, and agent mention
triggering so HTTP controllers, agents, and gateway adapters all follow the
same path.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional

from app.repository import communication_repository as comm_repo
from app.repository import discord_repository
from app.websocket_manager import manager as websocket_manager

logger = logging.getLogger(__name__)

# Keep emitting the legacy event during the UI migration window.
DUAL_WS_MESSAGE_EVENT = True


class CommunicationService:
    """Coordinates message persistence and fan-out side effects."""

    async def create_and_dispatch_message(
        self,
        *,
        channel_id: str,
        sender_id: str,
        sender_name: str,
        sender_type: str,
        content: str,
        message_type: str = "text",
        parent_message_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        message_id: Optional[str] = None,
        process_mentions: bool = True,
        allow_discord_forward: bool = True,
    ) -> Dict[str, Any]:
        """Create a message, broadcast it, and trigger any side effects."""
        resolved_thread_id = await self._resolve_thread_id(
            parent_message_id=parent_message_id,
            thread_id=thread_id,
        )
        created_message_id = message_id or str(uuid.uuid4())

        message = await comm_repo.create_message(
            message_id=created_message_id,
            channel_id=channel_id,
            sender_id=sender_id,
            sender_name=sender_name,
            sender_type=sender_type,
            content=content,
            message_type=message_type,
            parent_message_id=parent_message_id,
            thread_id=resolved_thread_id,
            metadata=metadata,
        )
        message["reactions"] = []

        await self.broadcast_message(message)

        if allow_discord_forward:
            await self._forward_message_to_discord(message)

        if process_mentions:
            self._trigger_agent_processing(message)

        return message

    async def ingest_discord_message(
        self,
        *,
        mapping,
        discord_message,
        cleaned_content: str,
        message_prefix: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Ingest a Discord message into Communication Commons with dedupe."""
        discord_message_id = str(discord_message.id)
        discord_channel_id = str(discord_message.channel.id)

        existing_link = await discord_repository.get_message_link_by_discord_message(
            discord_message_id=discord_message_id,
            discord_channel_id=discord_channel_id,
        )
        if existing_link:
            logger.debug(
                "Skipping duplicate Discord message %s already linked to CORE message %s",
                discord_message_id,
                existing_link.get("core_message_id"),
            )
            await self._record_delivery_event(
                status="skipped",
                direction="discord_to_core",
                core_message_id=existing_link.get("core_message_id"),
                core_channel_id=mapping.core_channel_id,
                discord_message_id=discord_message_id,
                discord_channel_id=discord_channel_id,
                discord_guild_id=(
                    str(discord_message.guild.id) if discord_message.guild else None
                ),
                metadata={"reason": "duplicate_link"},
            )
            return None

        content = cleaned_content.strip()
        if message_prefix:
            content = f"{message_prefix}{content}"

        parent_core_message_id = None
        referenced_message = getattr(discord_message, "reference", None)
        if referenced_message and getattr(referenced_message, "message_id", None):
            parent_link = await discord_repository.get_message_link_by_discord_message(
                discord_message_id=str(referenced_message.message_id),
                discord_channel_id=discord_channel_id,
            )
            if parent_link:
                parent_core_message_id = parent_link.get("core_message_id")

        metadata = {
            "source": "discord",
            "discord_message_id": discord_message_id,
            "discord_channel_id": discord_channel_id,
            "discord_author_id": str(discord_message.author.id),
            "discord_guild_id": (
                str(discord_message.guild.id) if discord_message.guild else None
            ),
            "discord_reference_message_id": (
                str(referenced_message.message_id)
                if referenced_message
                and getattr(referenced_message, "message_id", None)
                else None
            ),
            "discord_thread_id": (
                str(discord_message.channel.id)
                if discord_message.channel.__class__.__name__ == "Thread"
                else None
            ),
        }

        created = await self.create_and_dispatch_message(
            channel_id=mapping.core_channel_id,
            sender_id=f"discord_{discord_message.author.id}",
            sender_name=discord_message.author.display_name,
            sender_type="human",
            content=content,
            message_type="text",
            parent_message_id=parent_core_message_id,
            metadata=metadata,
            process_mentions=True,
            allow_discord_forward=False,
        )

        await discord_repository.create_message_link(
            core_message_id=created["message_id"],
            core_channel_id=created["channel_id"],
            discord_message_id=discord_message_id,
            discord_channel_id=discord_channel_id,
            discord_guild_id=metadata["discord_guild_id"],
            discord_author_id=metadata["discord_author_id"],
            direction="discord_to_core",
            metadata={
                "discord_reference_message_id": metadata[
                    "discord_reference_message_id"
                ],
                "discord_thread_id": metadata["discord_thread_id"],
            },
        )

        await self._record_delivery_event(
            status="success",
            direction="discord_to_core",
            core_message_id=created["message_id"],
            core_channel_id=created["channel_id"],
            discord_message_id=discord_message_id,
            discord_channel_id=discord_channel_id,
            discord_guild_id=metadata["discord_guild_id"],
            metadata={
                "parent_core_message_id": parent_core_message_id,
                "discord_reference_message_id": metadata[
                    "discord_reference_message_id"
                ],
            },
        )

        return created

    async def broadcast_message(self, message: Dict[str, Any]) -> None:
        """Broadcast a Communication Commons message over WebSocket."""
        payload = {
            "channel_id": message["channel_id"],
            "message": message,
        }

        await websocket_manager.broadcast_to_channel(
            channel_id=message["channel_id"],
            message={
                "type": "message",
                **payload,
            },
        )

        if DUAL_WS_MESSAGE_EVENT:
            await websocket_manager.broadcast_to_channel(
                channel_id=message["channel_id"],
                message={
                    "type": "new_message",
                    **payload,
                },
            )

    async def _resolve_thread_id(
        self,
        *,
        parent_message_id: Optional[str],
        thread_id: Optional[str],
    ) -> Optional[str]:
        """Resolve the effective thread root for a new message."""
        if thread_id or not parent_message_id:
            return thread_id

        parent_message = await comm_repo.get_message(parent_message_id)
        if parent_message and parent_message.get("thread_id"):
            return parent_message["thread_id"]
        return parent_message_id

    async def _forward_message_to_discord(self, message: Dict[str, Any]) -> None:
        """Forward a CORE message to any bridged Discord channels."""
        metadata = message.get("metadata") or {}
        if metadata.get("source") == "discord":
            return

        from app.services.discord_bridge import get_discord_bridge

        bridge = get_discord_bridge()
        mappings = [
            mapping
            for mapping in bridge.get_channel_mappings().values()
            if mapping.enabled and mapping.core_channel_id == message["channel_id"]
        ]
        if not mappings:
            return

        if not bridge.is_connected:
            for mapping in mappings:
                await self._record_delivery_event(
                    status="failed",
                    direction="core_to_discord",
                    core_message_id=message["message_id"],
                    core_channel_id=message["channel_id"],
                    discord_channel_id=mapping.discord_channel_id,
                    discord_guild_id=mapping.discord_guild_id,
                    error="Discord bridge not connected",
                )
            return

        for mapping in mappings:

            existing_link = (
                await discord_repository.get_primary_message_link_for_core_message(
                    core_message_id=message["message_id"],
                    discord_channel_id=mapping.discord_channel_id,
                )
            )
            if existing_link:
                logger.debug(
                    "Skipping duplicate CORE→Discord forward for message %s to channel %s",
                    message["message_id"],
                    mapping.discord_channel_id,
                )
                await self._record_delivery_event(
                    status="skipped",
                    direction="core_to_discord",
                    core_message_id=message["message_id"],
                    core_channel_id=message["channel_id"],
                    discord_message_id=existing_link.get("discord_message_id"),
                    discord_channel_id=mapping.discord_channel_id,
                    discord_guild_id=mapping.discord_guild_id,
                    metadata={"reason": "duplicate_link"},
                )
                continue

            reply_to_message_id = await self._get_discord_reply_target(
                parent_message_id=message.get("parent_message_id"),
                discord_channel_id=mapping.discord_channel_id,
            )
            formatted_content = f"**{message['sender_name']}**: {message['content']}"
            sent_message_ids = await bridge.send_to_discord(
                discord_channel_id=mapping.discord_channel_id,
                content=formatted_content,
                reply_to_message_id=reply_to_message_id,
            )

            if not sent_message_ids:
                logger.warning(
                    "Failed to forward CORE message %s to Discord channel %s",
                    message["message_id"],
                    mapping.discord_channel_id,
                )
                await self._record_delivery_event(
                    status="failed",
                    direction="core_to_discord",
                    core_message_id=message["message_id"],
                    core_channel_id=message["channel_id"],
                    discord_channel_id=mapping.discord_channel_id,
                    discord_guild_id=mapping.discord_guild_id,
                    error="Discord send returned no message IDs",
                    metadata={"reply_to_discord_message_id": reply_to_message_id},
                )
                continue

            total_chunks = len(sent_message_ids)
            for chunk_index, discord_message_id in enumerate(sent_message_ids):
                await discord_repository.create_message_link(
                    core_message_id=message["message_id"],
                    core_channel_id=message["channel_id"],
                    discord_message_id=discord_message_id,
                    discord_channel_id=mapping.discord_channel_id,
                    discord_guild_id=mapping.discord_guild_id,
                    direction="core_to_discord",
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    metadata={
                        "reply_to_core_message_id": message.get("parent_message_id"),
                        "reply_to_discord_message_id": reply_to_message_id,
                    },
                )

            await self._record_delivery_event(
                status="success",
                direction="core_to_discord",
                core_message_id=message["message_id"],
                core_channel_id=message["channel_id"],
                discord_message_id=sent_message_ids[0],
                discord_channel_id=mapping.discord_channel_id,
                discord_guild_id=mapping.discord_guild_id,
                metadata={
                    "chunk_count": total_chunks,
                    "reply_to_discord_message_id": reply_to_message_id,
                    "sent_message_ids": sent_message_ids,
                },
            )

    async def _get_discord_reply_target(
        self,
        *,
        parent_message_id: Optional[str],
        discord_channel_id: str,
    ) -> Optional[str]:
        """Resolve the Discord message that a CORE reply should target."""
        if not parent_message_id:
            return None

        parent_link = (
            await discord_repository.get_primary_message_link_for_core_message(
                core_message_id=parent_message_id,
                discord_channel_id=discord_channel_id,
            )
        )
        if parent_link:
            return parent_link.get("discord_message_id")

        parent_message = await comm_repo.get_message(parent_message_id)
        if not parent_message:
            return None

        parent_metadata = parent_message.get("metadata") or {}
        if parent_metadata.get("source") == "discord":
            return parent_metadata.get("discord_message_id")

        return None

    def _trigger_agent_processing(self, message: Dict[str, Any]) -> None:
        """Schedule agent mention processing for a newly created message."""
        from app.services.agent_response_service import get_agent_response_service

        asyncio.create_task(
            get_agent_response_service().process_message(
                message_id=message["message_id"],
                channel_id=message["channel_id"],
                content=message["content"],
                sender_id=message["sender_id"],
            )
        )

    async def _record_delivery_event(
        self,
        *,
        status: str,
        direction: str,
        core_message_id: Optional[str] = None,
        core_channel_id: Optional[str] = None,
        discord_message_id: Optional[str] = None,
        discord_channel_id: Optional[str] = None,
        discord_guild_id: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist a best-effort Discord delivery event for observability."""
        try:
            await discord_repository.create_delivery_event(
                event_id=str(uuid.uuid4()),
                status=status,
                direction=direction,
                core_message_id=core_message_id,
                core_channel_id=core_channel_id,
                discord_message_id=discord_message_id,
                discord_channel_id=discord_channel_id,
                discord_guild_id=discord_guild_id,
                error=error,
                metadata=metadata,
            )
        except Exception as exc:
            logger.warning("Failed to record Discord delivery event: %s", exc)


_communication_service: Optional[CommunicationService] = None


def get_communication_service() -> CommunicationService:
    """Get the shared CommunicationService singleton."""
    global _communication_service

    if _communication_service is None:
        _communication_service = CommunicationService()

    return _communication_service
