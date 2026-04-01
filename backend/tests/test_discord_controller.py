from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.controllers.discord import (
    SendMessageRequest,
    get_delivery_events,
    get_message_links,
    get_metrics,
    send_message as send_discord_message,
)


@pytest.mark.asyncio
class TestDiscordControllerSendMessage:
    @patch("app.controllers.discord.discord_repository.create_delivery_event")
    @patch("app.controllers.discord.get_discord_bridge")
    async def test_send_message_returns_message_ids_and_logs_event(self, mock_get_bridge, mock_create_event):
        # Arrange
        bridge = MagicMock()
        bridge.is_connected = True
        bridge.send_to_discord = AsyncMock(return_value=["discord-msg-1", "discord-msg-2"])
        mock_get_bridge.return_value = bridge
        mock_create_event.return_value = {}

        request = SendMessageRequest(
            discord_channel_id="discord-chan-1",
            content="Hello Discord",
            reply_to_message_id="discord-parent",
        )

        # Act
        result = await send_discord_message(request)

        # Assert
        assert result["success"] is True
        assert result["message_count"] == 2
        assert result["message_ids"] == ["discord-msg-1", "discord-msg-2"]
        mock_create_event.assert_awaited_once()
        assert mock_create_event.await_args.kwargs["status"] == "success"


@pytest.mark.asyncio
class TestDiscordControllerInspectionEndpoints:
    @patch("app.controllers.discord.discord_repository.list_message_links")
    async def test_get_message_links_forwards_filters(self, mock_list_links):
        # Arrange
        mock_list_links.return_value = [{"core_message_id": "core-msg-1"}]

        # Act
        result = await get_message_links(
            limit=25,
            core_message_id="core-msg-1",
            core_channel_id="ch_1",
            discord_channel_id="discord-chan-1",
            direction="core_to_discord",
        )

        # Assert
        assert result["count"] == 1
        assert result["links"][0]["core_message_id"] == "core-msg-1"
        mock_list_links.assert_awaited_once_with(
            limit=25,
            core_message_id="core-msg-1",
            core_channel_id="ch_1",
            discord_channel_id="discord-chan-1",
            direction="core_to_discord",
        )

    @patch("app.controllers.discord.discord_repository.list_delivery_events")
    async def test_get_delivery_events_forwards_filters(self, mock_list_events):
        # Arrange
        mock_list_events.return_value = [{"status": "failed"}]

        # Act
        result = await get_delivery_events(
            limit=10,
            status_filter="failed",
            direction="core_to_discord",
            core_channel_id="ch_1",
            discord_channel_id="discord-chan-1",
            core_message_id="core-msg-1",
        )

        # Assert
        assert result["count"] == 1
        assert result["events"][0]["status"] == "failed"
        mock_list_events.assert_awaited_once_with(
            limit=10,
            status="failed",
            direction="core_to_discord",
            core_channel_id="ch_1",
            discord_channel_id="discord-chan-1",
            core_message_id="core-msg-1",
        )

    @patch("app.controllers.discord.discord_repository.list_delivery_events")
    @patch("app.controllers.discord.discord_repository.count_delivery_events_by_direction")
    @patch("app.controllers.discord.discord_repository.count_delivery_events_by_status")
    @patch("app.controllers.discord.discord_repository.count_delivery_events")
    @patch("app.controllers.discord.discord_repository.count_message_links_by_direction")
    @patch("app.controllers.discord.discord_repository.count_message_links")
    @patch("app.controllers.discord.discord_repository.count_channel_mappings")
    @patch("app.controllers.discord.get_discord_bridge")
    async def test_get_metrics_aggregates_bridge_and_repository_data(
        self,
        mock_get_bridge,
        mock_count_mappings,
        mock_count_message_links,
        mock_count_links_by_direction,
        mock_count_delivery_events,
        mock_count_delivery_by_status,
        mock_count_delivery_by_direction,
        mock_list_delivery_events,
    ):
        # Arrange
        bridge = MagicMock()
        bridge.get_status_info.return_value = {
            "status": "connected",
            "connected": True,
            "connected_at": "2026-02-09T00:00:00",
            "last_error": None,
            "reconnect_attempts": 0,
            "bot_user": "COREBot",
            "guilds": 1,
            "channel_mappings": 2,
            "bridged_core_channels": ["discord_general"],
        }
        mock_get_bridge.return_value = bridge
        mock_count_mappings.return_value = 2
        mock_count_message_links.return_value = 12
        mock_count_links_by_direction.return_value = {"discord_to_core": 5, "core_to_discord": 7}
        mock_count_delivery_events.return_value = 9
        mock_count_delivery_by_status.return_value = {"success": 7, "failed": 1, "skipped": 1}
        mock_count_delivery_by_direction.return_value = {"discord_to_core": 4, "core_to_discord": 5}
        mock_list_delivery_events.return_value = [{"status": "failed", "error": "Bridge offline"}]

        # Act
        result = await get_metrics(recent_failures_limit=5)

        # Assert
        assert result.mappings_count == 2
        assert result.message_links_count == 12
        assert result.delivery_events_count == 9
        assert result.delivery_events_by_status["failed"] == 1
        assert result.recent_failures[0]["error"] == "Bridge offline"
        mock_list_delivery_events.assert_awaited_once_with(limit=5, status="failed")
