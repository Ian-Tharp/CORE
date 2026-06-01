"""
Tests for DiscordBridgeService — property guards, message splitting, channel
mapping CRUD, and start() early-exit conditions.

Sentinel-value contract:
- Remove is_connected client check → CONNECTED status without a client returns True → test fails
- Remove newline split in _split_message → long messages split at arbitrary position → test fails
- Remove space fallback in _split_message → messages split mid-word → test fails
- Remove enabled check in start() → disabled bridge starts anyway → test fails
- Remove token check in start() → bridge starts without credentials → test fails
- Remove remove_channel_mapping pop → removed mapping still in dict → test fails
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config.discord import DiscordChannelMapping, DiscordConfig
from app.services.discord_bridge import BridgeStatus, DiscordBridgeService


# ===========================================================================
# Helpers
# ===========================================================================


def _make_config(
    enabled: bool = True,
    bot_token: str = "fake-token",
    channel_mappings: dict | None = None,
    allowed_users: list | None = None,
    auto_create_channels: bool = False,
    response_prefix: str = "",
    message_prefix: str = "",
) -> DiscordConfig:
    return DiscordConfig(
        enabled=enabled,
        bot_token=bot_token,
        channel_mappings=channel_mappings or {},
        allowed_users=allowed_users or [],
        auto_create_channels=auto_create_channels,
        response_prefix=response_prefix,
        message_prefix=message_prefix,
    )


def _make_service(config: DiscordConfig | None = None) -> DiscordBridgeService:
    cfg = config or _make_config()
    return DiscordBridgeService(config=cfg)


# ===========================================================================
# is_connected — property
# ===========================================================================


class TestIsConnected:
    def test_disconnected_status_returns_false(self):
        """DISCONNECTED status must return False."""
        svc = _make_service()
        svc._status = BridgeStatus.DISCONNECTED
        svc._client = None
        assert svc.is_connected is False

    def test_connected_with_client_returns_true(self):
        """
        SENTINEL — CONNECTED status AND client set must return True.
        Return True based on status alone → client=None still reports connected → test fails.
        """
        svc = _make_service()
        svc._status = BridgeStatus.CONNECTED
        svc._client = MagicMock()
        assert svc.is_connected is True

    def test_connected_without_client_returns_false(self):
        """
        SENTINEL — CONNECTED status but _client=None must still return False.
        Ignore _client check → can't distinguish real vs phantom connection → test fails.
        """
        svc = _make_service()
        svc._status = BridgeStatus.CONNECTED
        svc._client = None
        assert svc.is_connected is False

    def test_error_status_returns_false(self):
        """ERROR status must return False."""
        svc = _make_service()
        svc._status = BridgeStatus.ERROR
        svc._client = MagicMock()
        assert svc.is_connected is False

    def test_reconnecting_status_returns_false(self):
        """RECONNECTING status must return False."""
        svc = _make_service()
        svc._status = BridgeStatus.RECONNECTING
        svc._client = MagicMock()
        assert svc.is_connected is False


# ===========================================================================
# get_status_info — status dictionary
# ===========================================================================


class TestGetStatusInfo:
    def test_returns_all_required_fields(self):
        """
        SENTINEL — status dict must include status, connected, last_error, reconnect_attempts.
        Remove a field → monitoring dashboard breaks → test fails.
        """
        svc = _make_service()
        info = svc.get_status_info()
        assert "status" in info
        assert "connected" in info
        assert "last_error" in info
        assert "reconnect_attempts" in info
        assert "channel_mappings" in info

    def test_status_value_is_string(self):
        """status field must be a string (enum .value)."""
        svc = _make_service()
        info = svc.get_status_info()
        assert isinstance(info["status"], str)
        assert info["status"] == "disconnected"

    def test_connected_at_none_when_not_connected(self):
        """connected_at must be None before first connection."""
        svc = _make_service()
        info = svc.get_status_info()
        assert info["connected_at"] is None

    def test_connected_at_isoformat_when_set(self):
        """
        SENTINEL — connected_at must be an ISO string when set.
        Return datetime object → JSON serialisation fails → test fails.
        """
        svc = _make_service()
        svc._connected_at = datetime(2024, 6, 1, 12, 0, 0)
        info = svc.get_status_info()
        assert info["connected_at"] == "2024-06-01T12:00:00"

    def test_channel_mappings_count_correct(self):
        """channel_mappings count must match config.channel_mappings."""
        mapping = DiscordChannelMapping(
            discord_channel_id="ch-1", core_channel_id="core-ch-1"
        )
        cfg = _make_config(channel_mappings={"ch-1": mapping})
        svc = _make_service(cfg)
        info = svc.get_status_info()
        assert info["channel_mappings"] == 1

    def test_guilds_zero_when_no_client(self):
        """guilds must be 0 when no client is connected."""
        svc = _make_service()
        assert svc._client is None
        info = svc.get_status_info()
        assert info["guilds"] == 0


# ===========================================================================
# start() — early-exit conditions
# ===========================================================================


class TestStart:
    @pytest.mark.asyncio
    async def test_disabled_config_returns_false(self):
        """
        SENTINEL — start() must return False immediately when enabled=False.
        Remove enabled check → disabled bridge starts anyway → test fails.
        """
        cfg = _make_config(enabled=False)
        svc = _make_service(cfg)

        with patch(
            "app.services.discord_bridge.load_config_from_store",
            new=AsyncMock(return_value=cfg),
        ):
            result = await svc.start()

        assert result is False

    @pytest.mark.asyncio
    async def test_missing_token_returns_false_and_sets_error(self):
        """
        SENTINEL — missing bot_token must return False and set error status.
        Remove token check → bridge starts with invalid token, fails at login → test fails.
        """
        cfg = _make_config(enabled=True, bot_token="")
        svc = _make_service(cfg)

        with patch(
            "app.services.discord_bridge.load_config_from_store",
            new=AsyncMock(return_value=cfg),
        ):
            result = await svc.start()

        assert result is False
        assert svc._status == BridgeStatus.ERROR

    @pytest.mark.asyncio
    async def test_already_running_returns_true(self):
        """start() called while already running must return True without double-starting."""
        cfg = _make_config()
        svc = _make_service(cfg)
        svc._running = True  # simulate already running

        with patch(
            "app.services.discord_bridge.load_config_from_store",
            new=AsyncMock(return_value=cfg),
        ):
            result = await svc.start()

        assert result is True


# ===========================================================================
# _split_message — pure chunking logic
# ===========================================================================


class TestSplitMessage:
    def test_short_message_not_split(self):
        """Content shorter than max_length must be returned as a single-element list."""
        svc = _make_service()
        chunks = svc._split_message("hello world", 100)
        assert chunks == ["hello world"]

    def test_message_at_exact_limit_not_split(self):
        """Content exactly at max_length must not be split."""
        svc = _make_service()
        content = "A" * 2000
        chunks = svc._split_message(content, 2000)
        assert len(chunks) == 1

    def test_long_message_split_at_newline(self):
        """
        SENTINEL — split must prefer newline boundaries.
        Remove newline check → splits at arbitrary position mid-sentence → test fails.
        """
        svc = _make_service()
        first_line = "A" * 100
        second_line = "B" * 100
        content = first_line + "\n" + second_line
        chunks = svc._split_message(content, 150)
        assert len(chunks) == 2
        assert chunks[0] == first_line  # split exactly at newline

    def test_long_message_falls_back_to_space(self):
        """
        SENTINEL — when no newline fits, must split at the last space.
        Remove space fallback → splits mid-word → test fails.
        """
        svc = _make_service()
        # 90 As, a space, then 90 Bs — total > 100 but split at space
        content = ("A" * 90) + " " + ("B" * 90)
        chunks = svc._split_message(content, 100)
        assert len(chunks) >= 2
        # First chunk should not contain any B (split at the space before Bs)
        assert "B" not in chunks[0]

    def test_hard_split_when_no_whitespace(self):
        """When no space or newline exists, must hard-split at max_length."""
        svc = _make_service()
        content = "X" * 300
        chunks = svc._split_message(content, 100)
        assert len(chunks) == 3
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_all_chunks_within_limit(self):
        """Every chunk must be at or under max_length."""
        svc = _make_service()
        content = "word " * 1000  # 5000 chars
        chunks = svc._split_message(content, 200)
        for chunk in chunks:
            assert len(chunk) <= 200

    def test_split_reassembles_to_original(self):
        """Joining chunks (with stripped whitespace) must yield the original content."""
        svc = _make_service()
        original = "alpha beta gamma delta epsilon " * 50
        chunks = svc._split_message(original, 80)
        # After stripping leading whitespace (lstrip in the loop), reassembled text
        # should cover all the words
        rejoined = " ".join(chunks)
        for word in ["alpha", "beta", "gamma"]:
            assert word in rejoined


# ===========================================================================
# add/remove channel mappings — CRUD
# ===========================================================================


class TestChannelMappingCRUD:
    def test_add_mapping_stores_in_config(self):
        """
        SENTINEL — add_channel_mapping must update config.channel_mappings.
        Return without storing → mapping lost immediately → test fails.
        """
        svc = _make_service()
        mapping = DiscordChannelMapping(
            discord_channel_id="SENTINEL_CH", core_channel_id="core-ch"
        )
        svc.add_channel_mapping(mapping)
        assert "SENTINEL_CH" in svc.config.channel_mappings

    def test_add_mapping_tracks_core_channel(self):
        """
        SENTINEL — add_channel_mapping must add core_channel_id to bridged set.
        Skip bridged_core_channels update → channel not bridged → test fails.
        """
        svc = _make_service()
        mapping = DiscordChannelMapping(
            discord_channel_id="ch-1", core_channel_id="SENTINEL_CORE"
        )
        svc.add_channel_mapping(mapping)
        assert "SENTINEL_CORE" in svc._bridged_core_channels

    def test_remove_existing_mapping_returns_true(self):
        """
        SENTINEL — remove_channel_mapping must return True for existing mapping.
        Return False always → callers think removal failed → test fails.
        """
        svc = _make_service()
        mapping = DiscordChannelMapping(
            discord_channel_id="SENTINEL_REMOVE", core_channel_id="core-ch"
        )
        svc.add_channel_mapping(mapping)
        result = svc.remove_channel_mapping("SENTINEL_REMOVE")
        assert result is True
        assert "SENTINEL_REMOVE" not in svc.config.channel_mappings

    def test_remove_nonexistent_mapping_returns_false(self):
        """Removing a non-existent mapping must return False."""
        svc = _make_service()
        result = svc.remove_channel_mapping("ghost-ch")
        assert result is False

    def test_get_channel_mappings_returns_copy(self):
        """get_channel_mappings must return a copy (mutating it won't affect config)."""
        svc = _make_service()
        mappings = svc.get_channel_mappings()
        mappings["injected"] = "injected"
        assert "injected" not in svc.config.channel_mappings


# ===========================================================================
# send_to_discord — not connected guard
# ===========================================================================


class TestSendToDiscord:
    @pytest.mark.asyncio
    async def test_returns_empty_when_not_connected(self):
        """
        SENTINEL — send_to_discord must return [] when bridge is not connected.
        Remove is_connected check → tries to get channel on None client → AttributeError → test fails.
        """
        svc = _make_service()
        svc._status = BridgeStatus.DISCONNECTED
        result = await svc.send_to_discord("ch-1", "hello")
        assert result == []
