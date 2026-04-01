from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.communication_service import CommunicationService


def _make_message(message_id: str, channel_id: str = "ch_1", **overrides):
    message = {
        "message_id": message_id,
        "channel_id": channel_id,
        "sender_id": "user_1",
        "sender_name": "Test User",
        "sender_type": "human",
        "content": "hello",
        "message_type": "text",
        "parent_message_id": None,
        "thread_id": None,
        "created_at": "2026-02-09T00:00:00",
        "edited_at": None,
        "metadata": None,
    }
    message.update(overrides)
    return message


def _make_mapping(
    *,
    discord_channel_id: str = "discord-1",
    core_channel_id: str = "ch_1",
    discord_guild_id: str = "guild-1",
):
    return SimpleNamespace(
        discord_channel_id=discord_channel_id,
        core_channel_id=core_channel_id,
        discord_guild_id=discord_guild_id,
        enabled=True,
    )


def _make_discord_message(
    *,
    message_id: str = "discord-msg-1",
    channel_id: str = "discord-chan-1",
    author_id: str = "discord-user-1",
    author_name: str = "Discord User",
    guild_id: str = "discord-guild-1",
    reference_message_id: str | None = None,
):
    reference = (
        SimpleNamespace(message_id=reference_message_id)
        if reference_message_id is not None
        else None
    )
    return SimpleNamespace(
        id=message_id,
        channel=SimpleNamespace(id=channel_id),
        author=SimpleNamespace(id=author_id, display_name=author_name),
        guild=SimpleNamespace(id=guild_id),
        reference=reference,
    )


@pytest.mark.asyncio
class TestCommunicationServiceThreadResolution:
    async def test_reply_to_top_level_creates_thread_from_parent(self):
        # Arrange
        service = CommunicationService()
        parent_message = _make_message("parent-1", thread_id=None)

        with patch(
            "app.services.communication_service.comm_repo.get_message",
            new=AsyncMock(return_value=parent_message),
        ), patch(
            "app.services.communication_service.comm_repo.create_message",
            new=AsyncMock(return_value=_make_message("child-1", parent_message_id="parent-1", thread_id="parent-1")),
        ) as mock_create, patch(
            "app.services.communication_service.websocket_manager.broadcast_to_channel",
            new=AsyncMock(),
        ):
            # Act
            await service.create_and_dispatch_message(
                channel_id="ch_1",
                sender_id="user_1",
                sender_name="Test User",
                sender_type="human",
                content="Reply",
                parent_message_id="parent-1",
                process_mentions=False,
                allow_discord_forward=False,
                message_id="child-1",
            )

        # Assert
        assert mock_create.await_args.kwargs["thread_id"] == "parent-1"

    async def test_reply_to_reply_inherits_existing_thread_root(self):
        # Arrange
        service = CommunicationService()
        parent_message = _make_message("reply-1", thread_id="root-1")

        with patch(
            "app.services.communication_service.comm_repo.get_message",
            new=AsyncMock(return_value=parent_message),
        ), patch(
            "app.services.communication_service.comm_repo.create_message",
            new=AsyncMock(return_value=_make_message("child-1", parent_message_id="reply-1", thread_id="root-1")),
        ) as mock_create, patch(
            "app.services.communication_service.websocket_manager.broadcast_to_channel",
            new=AsyncMock(),
        ):
            # Act
            await service.create_and_dispatch_message(
                channel_id="ch_1",
                sender_id="user_1",
                sender_name="Test User",
                sender_type="human",
                content="Nested reply",
                parent_message_id="reply-1",
                process_mentions=False,
                allow_discord_forward=False,
                message_id="child-1",
            )

        # Assert
        assert mock_create.await_args.kwargs["thread_id"] == "root-1"


@pytest.mark.asyncio
class TestCommunicationServiceDiscordForwarding:
    async def test_forwarding_uses_parent_link_and_records_chunks(self):
        # Arrange
        service = CommunicationService()
        mapping = _make_mapping()
        parent_message = _make_message("parent-1", thread_id="root-1")
        created_message = _make_message(
            "core-msg-1",
            parent_message_id="parent-1",
            thread_id="root-1",
            sender_name="Continuum",
            content="Here is the answer",
        )
        bridge = MagicMock()
        bridge.is_connected = True
        bridge.get_channel_mappings.return_value = {mapping.discord_channel_id: mapping}
        bridge.send_to_discord = AsyncMock(return_value=["discord-msg-1", "discord-msg-2"])

        def _link_lookup(*, core_message_id: str, discord_channel_id: str | None = None):
            if core_message_id == "core-msg-1":
                return None
            if core_message_id == "parent-1":
                return {
                    "core_message_id": "parent-1",
                    "discord_message_id": "discord-parent",
                    "discord_channel_id": mapping.discord_channel_id,
                }
            return None

        with patch(
            "app.services.communication_service.comm_repo.get_message",
            new=AsyncMock(return_value=parent_message),
        ), patch(
            "app.services.communication_service.comm_repo.create_message",
            new=AsyncMock(return_value=created_message),
        ), patch(
            "app.services.communication_service.websocket_manager.broadcast_to_channel",
            new=AsyncMock(),
        ), patch(
            "app.services.communication_service.discord_repository.get_primary_message_link_for_core_message",
            new=AsyncMock(side_effect=_link_lookup),
        ), patch(
            "app.services.communication_service.discord_repository.create_message_link",
            new=AsyncMock(),
        ) as mock_create_link, patch(
            "app.services.communication_service.discord_repository.create_delivery_event",
            new=AsyncMock(),
        ) as mock_create_event, patch(
            "app.services.discord_bridge.get_discord_bridge",
            return_value=bridge,
        ):
            # Act
            await service.create_and_dispatch_message(
                channel_id="ch_1",
                sender_id="agent_1",
                sender_name="Continuum",
                sender_type="agent",
                content="Here is the answer",
                parent_message_id="parent-1",
                process_mentions=False,
                allow_discord_forward=True,
                message_id="core-msg-1",
            )

        # Assert
        bridge.send_to_discord.assert_awaited_once_with(
            discord_channel_id="discord-1",
            content="**Continuum**: Here is the answer",
            reply_to_message_id="discord-parent",
        )
        assert mock_create_link.await_count == 2
        first_call = mock_create_link.await_args_list[0].kwargs
        second_call = mock_create_link.await_args_list[1].kwargs
        assert first_call["chunk_index"] == 0
        assert first_call["total_chunks"] == 2
        assert second_call["chunk_index"] == 1
        assert second_call["discord_message_id"] == "discord-msg-2"
        assert mock_create_event.await_count == 1
        assert mock_create_event.await_args.kwargs["status"] == "success"
        assert mock_create_event.await_args.kwargs["direction"] == "core_to_discord"

    async def test_forwarding_records_failure_when_bridge_is_disconnected(self):
        # Arrange
        service = CommunicationService()
        mapping = _make_mapping()
        created_message = _make_message(
            "core-msg-1",
            sender_name="Continuum",
            content="Bridge is offline",
        )
        bridge = MagicMock()
        bridge.is_connected = False
        bridge.get_channel_mappings.return_value = {mapping.discord_channel_id: mapping}

        with patch(
            "app.services.communication_service.comm_repo.create_message",
            new=AsyncMock(return_value=created_message),
        ), patch(
            "app.services.communication_service.websocket_manager.broadcast_to_channel",
            new=AsyncMock(),
        ), patch(
            "app.services.communication_service.discord_repository.create_delivery_event",
            new=AsyncMock(),
        ) as mock_create_event, patch(
            "app.services.discord_bridge.get_discord_bridge",
            return_value=bridge,
        ):
            # Act
            await service.create_and_dispatch_message(
                channel_id="ch_1",
                sender_id="agent_1",
                sender_name="Continuum",
                sender_type="agent",
                content="Bridge is offline",
                process_mentions=False,
                allow_discord_forward=True,
                message_id="core-msg-1",
            )

        # Assert
        mock_create_event.assert_awaited_once()
        assert mock_create_event.await_args.kwargs["status"] == "failed"
        assert mock_create_event.await_args.kwargs["error"] == "Discord bridge not connected"


@pytest.mark.asyncio
class TestCommunicationServiceDiscordIngress:
    async def test_ingest_discord_message_skips_duplicate_links(self):
        # Arrange
        service = CommunicationService()
        mapping = _make_mapping()
        discord_message = _make_discord_message(message_id="discord-msg-1")

        with patch(
            "app.services.communication_service.discord_repository.get_message_link_by_discord_message",
            new=AsyncMock(return_value={"core_message_id": "existing-core-msg"}),
        ), patch(
            "app.services.communication_service.discord_repository.create_delivery_event",
            new=AsyncMock(),
        ) as mock_create_event, patch(
            "app.services.communication_service.discord_repository.create_message_link",
            new=AsyncMock(),
        ), patch(
            "app.services.communication_service.comm_repo.create_message",
            new=AsyncMock(),
        ) as mock_create_message:
            # Act
            result = await service.ingest_discord_message(
                mapping=mapping,
                discord_message=discord_message,
                cleaned_content="Hello from Discord",
            )

        # Assert
        assert result is None
        mock_create_message.assert_not_called()
        mock_create_event.assert_awaited_once()
        assert mock_create_event.await_args.kwargs["status"] == "skipped"
        assert mock_create_event.await_args.kwargs["direction"] == "discord_to_core"

    async def test_ingest_discord_reply_maps_back_to_core_parent(self):
        # Arrange
        service = CommunicationService()
        mapping = _make_mapping()
        discord_message = _make_discord_message(
            message_id="discord-msg-2",
            reference_message_id="discord-parent",
        )
        parent_message = _make_message("core-parent", thread_id="core-root")
        created_message = _make_message(
            "core-msg-2",
            parent_message_id="core-parent",
            thread_id="core-root",
            metadata={"source": "discord"},
        )

        def _discord_lookup(*, discord_message_id: str, discord_channel_id: str):
            if discord_message_id == "discord-msg-2":
                return None
            if discord_message_id == "discord-parent":
                return {
                    "core_message_id": "core-parent",
                    "discord_message_id": "discord-parent",
                    "discord_channel_id": discord_channel_id,
                }
            return None

        agent_service = MagicMock()
        agent_service.process_message = AsyncMock()

        def _drop_task(coroutine):
            coroutine.close()
            return None

        with patch(
            "app.services.communication_service.discord_repository.get_message_link_by_discord_message",
            new=AsyncMock(side_effect=_discord_lookup),
        ), patch(
            "app.services.communication_service.comm_repo.get_message",
            new=AsyncMock(return_value=parent_message),
        ), patch(
            "app.services.communication_service.comm_repo.create_message",
            new=AsyncMock(return_value=created_message),
        ) as mock_create_message, patch(
            "app.services.communication_service.websocket_manager.broadcast_to_channel",
            new=AsyncMock(),
        ), patch(
            "app.services.communication_service.discord_repository.create_message_link",
            new=AsyncMock(),
        ) as mock_create_link, patch(
            "app.services.communication_service.discord_repository.create_delivery_event",
            new=AsyncMock(),
        ) as mock_create_event, patch(
            "app.services.agent_response_service.get_agent_response_service",
            return_value=agent_service,
        ), patch(
            "app.services.communication_service.asyncio.create_task",
            side_effect=_drop_task,
        ):
            # Act
            result = await service.ingest_discord_message(
                mapping=mapping,
                discord_message=discord_message,
                cleaned_content="Discord reply",
            )

        # Assert
        assert result["message_id"] == "core-msg-2"
        create_kwargs = mock_create_message.await_args.kwargs
        assert create_kwargs["parent_message_id"] == "core-parent"
        assert create_kwargs["thread_id"] == "core-root"
        assert create_kwargs["metadata"]["discord_reference_message_id"] == "discord-parent"
        mock_create_link.assert_awaited_once()
        mock_create_event.assert_awaited_once()
        assert mock_create_event.await_args.kwargs["status"] == "success"
        assert mock_create_event.await_args.kwargs["direction"] == "discord_to_core"
