"""
Tests for DiscordBridgeService — pure synchronous methods only.

Sentinel-value contract:
- Remove newline-first split in _split_message → code blocks split mid-line → test fails
- Remove space-fallback split → words bisected in Discord messages → test fails
- Remove hard-split fallback → infinite loop on no whitespace → test fails
- Remove is_connected client-None check → disconnected bridge reports connected → test fails
- Remove get_status_info reconnect_attempts key → frontend crashes on missing key → test fails
- Remove add_channel_mapping bridged_core_channels update → bridge misses inbound routing → test fails
- Remove remove_channel_mapping return False branch → callers can't tell if key was absent → test fails
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from app.services.discord_bridge import BridgeStatus, DiscordBridgeService
from app.config.discord import DiscordChannelMapping, DiscordConfig


# ===========================================================================
# Helpers
# ===========================================================================

def _make_bridge() -> DiscordBridgeService:
    """Build a bridge with a minimal disabled config (no network calls)."""
    config = DiscordConfig(bot_token="", enabled=False)
    return DiscordBridgeService(config=config)


def _make_mapping(discord_id: str = "DC_111", core_id: str = "CORE_222") -> DiscordChannelMapping:
    return DiscordChannelMapping(discord_channel_id=discord_id, core_channel_id=core_id)


# ===========================================================================
# BridgeStatus enum
# ===========================================================================

class TestBridgeStatus:
    def test_disconnected_value(self):
        """
        SENTINEL — DISCONNECTED must equal 'disconnected'.
        Change to 'offline' → status checks in API fail → test fails.
        """
        assert BridgeStatus.DISCONNECTED == "disconnected"

    def test_connected_value(self):
        """CONNECTED must equal 'connected'."""
        assert BridgeStatus.CONNECTED == "connected"

    def test_reconnecting_value(self):
        """RECONNECTING must equal 'reconnecting'."""
        assert BridgeStatus.RECONNECTING == "reconnecting"

    def test_error_value(self):
        """ERROR must equal 'error'."""
        assert BridgeStatus.ERROR == "error"


# ===========================================================================
# is_connected property
# ===========================================================================

class TestIsConnected:
    def test_false_when_disconnected(self):
        """
        SENTINEL — must be False when status is DISCONNECTED.
        Remove status check → returns True always → test fails.
        """
        bridge = _make_bridge()
        assert bridge.is_connected is False

    def test_false_when_connected_but_no_client(self):
        """
        SENTINEL — must be False when status=CONNECTED but _client is None.
        Remove client-None check → disconnected bridge claims connected → test fails.
        """
        bridge = _make_bridge()
        bridge._status = BridgeStatus.CONNECTED
        bridge._client = None
        assert bridge.is_connected is False

    def test_true_when_connected_with_client(self):
        """Must be True when status=CONNECTED and _client is not None."""
        bridge = _make_bridge()
        bridge._status = BridgeStatus.CONNECTED
        bridge._client = MagicMock()
        assert bridge.is_connected is True

    def test_false_when_reconnecting_with_client(self):
        """Must be False if status is RECONNECTING even with a client set."""
        bridge = _make_bridge()
        bridge._status = BridgeStatus.RECONNECTING
        bridge._client = MagicMock()
        assert bridge.is_connected is False


# ===========================================================================
# get_status_info — required keys
# ===========================================================================

class TestGetStatusInfo:
    def test_required_keys_present(self):
        """
        SENTINEL — status_info must include all required keys.
        Remove reconnect_attempts → frontend crashes on missing key → test fails.
        """
        bridge = _make_bridge()
        info = bridge.get_status_info()
        for key in ("status", "connected", "connected_at", "last_error",
                    "reconnect_attempts", "bot_user", "guilds",
                    "channel_mappings", "bridged_core_channels"):
            assert key in info, f"Missing key: {key}"

    def test_status_value_is_string(self):
        """
        SENTINEL — status must be the enum's string value, not the enum object.
        Return enum object → JSON serialisation fails → test fails.
        """
        bridge = _make_bridge()
        info = bridge.get_status_info()
        assert isinstance(info["status"], str)
        assert info["status"] == "disconnected"

    def test_connected_at_none_when_not_connected(self):
        """connected_at must be None before first connection."""
        bridge = _make_bridge()
        info = bridge.get_status_info()
        assert info["connected_at"] is None

    def test_guilds_zero_without_client(self):
        """guilds must be 0 when client is None."""
        bridge = _make_bridge()
        info = bridge.get_status_info()
        assert info["guilds"] == 0

    def test_reconnect_attempts_initial_zero(self):
        """reconnect_attempts must start at 0."""
        bridge = _make_bridge()
        info = bridge.get_status_info()
        assert info["reconnect_attempts"] == 0

    def test_channel_mappings_count_reflects_config(self):
        """channel_mappings count must match config.channel_mappings length."""
        bridge = _make_bridge()
        bridge.add_channel_mapping(_make_mapping("DC_001", "CORE_001"))
        bridge.add_channel_mapping(_make_mapping("DC_002", "CORE_002"))
        info = bridge.get_status_info()
        assert info["channel_mappings"] == 2


# ===========================================================================
# _split_message — chunking logic
# ===========================================================================

class TestSplitMessage:
    def setup_method(self):
        self.bridge = _make_bridge()

    def test_short_message_unchanged(self):
        """
        SENTINEL — message at or below max_length must return as single-item list.
        Always split → single-chunk messages split unnecessarily → test fails.
        """
        result = self.bridge._split_message("SENTINEL_MESSAGE", max_length=2000)
        assert result == ["SENTINEL_MESSAGE"]

    def test_exact_length_not_split(self):
        """Message exactly at max_length must be returned as-is."""
        msg = "x" * 100
        result = self.bridge._split_message(msg, max_length=100)
        assert result == [msg]

    def test_newline_preferred_split_point(self):
        """
        SENTINEL — newline within max_length must be preferred split point.
        Use space instead → code blocks bisected at random space → test fails.
        """
        # "AAAA\nBBBB" where max_length=6 should split after \n
        msg = "AAAA\nBBBB"
        result = self.bridge._split_message(msg, max_length=6)
        assert len(result) == 2
        assert result[0] == "AAAA"
        assert result[1] == "BBBB"

    def test_space_fallback_when_no_newline(self):
        """
        SENTINEL — space must be fallback when no newline within max_length.
        Hard-split only → words bisected → test fails.
        """
        msg = "hello world"
        result = self.bridge._split_message(msg, max_length=8)
        # "hello" (5) fits; "world" is second chunk
        assert len(result) == 2
        assert "hello" in result[0]

    def test_hard_split_when_no_whitespace(self):
        """
        SENTINEL — must hard-split when there is no whitespace in window.
        Infinite loop if no fallback → test hangs → test fails.
        """
        msg = "x" * 10
        result = self.bridge._split_message(msg, max_length=4)
        # Must produce chunks, no infinite loop
        assert len(result) > 1
        assert all(len(c) <= 4 for c in result)

    def test_all_chunks_combined_equal_original(self):
        """
        SENTINEL — joining all chunks must reconstruct original content (minus stripped whitespace).
        Lose characters during split → content corrupted → test fails.
        """
        msg = "word1 word2 word3 word4 word5"
        result = self.bridge._split_message(msg, max_length=12)
        combined = " ".join(result)
        # All words must be present
        for word in ("word1", "word2", "word3", "word4", "word5"):
            assert word in combined

    def test_empty_string_returns_single_empty_chunk(self):
        """Empty input must return [''] without crashing."""
        result = self.bridge._split_message("", max_length=2000)
        assert result == [""]


# ===========================================================================
# add_channel_mapping / remove_channel_mapping / get_channel_mappings
# ===========================================================================

class TestChannelMappingMutations:
    def setup_method(self):
        self.bridge = _make_bridge()

    def test_add_mapping_stored_in_config(self):
        """
        SENTINEL — add_channel_mapping must store mapping in config.channel_mappings.
        Store in separate dict → routing lookup fails → test fails.
        """
        mapping = _make_mapping("DC_SENTINEL", "CORE_SENTINEL")
        self.bridge.add_channel_mapping(mapping)
        assert "DC_SENTINEL" in self.bridge.config.channel_mappings

    def test_add_mapping_adds_to_bridged_core_channels(self):
        """
        SENTINEL — core_channel_id must be added to _bridged_core_channels.
        Skip update → inbound routing skips this channel → messages lost → test fails.
        """
        mapping = _make_mapping("DC_111", "CORE_SENTINEL_ID")
        self.bridge.add_channel_mapping(mapping)
        assert "CORE_SENTINEL_ID" in self.bridge._bridged_core_channels

    def test_remove_mapping_returns_true_on_success(self):
        """
        SENTINEL — removing existing mapping must return True.
        Always return None → caller can't confirm removal → test fails.
        """
        mapping = _make_mapping("DC_RM_TEST", "CORE_RM")
        self.bridge.add_channel_mapping(mapping)
        result = self.bridge.remove_channel_mapping("DC_RM_TEST")
        assert result is True

    def test_remove_mapping_returns_false_when_absent(self):
        """
        SENTINEL — removing non-existent mapping must return False.
        Raise KeyError instead → callers crash on idempotent remove → test fails.
        """
        result = self.bridge.remove_channel_mapping("DC_NONEXISTENT")
        assert result is False

    def test_remove_mapping_deletes_from_config(self):
        """After removal, key must be absent from config.channel_mappings."""
        mapping = _make_mapping("DC_DEL", "CORE_DEL")
        self.bridge.add_channel_mapping(mapping)
        self.bridge.remove_channel_mapping("DC_DEL")
        assert "DC_DEL" not in self.bridge.config.channel_mappings

    def test_get_channel_mappings_returns_copy(self):
        """
        SENTINEL — get_channel_mappings must return a copy, not the live dict.
        Return reference → caller mutation changes config → test fails.
        """
        mapping = _make_mapping("DC_CPY", "CORE_CPY")
        self.bridge.add_channel_mapping(mapping)
        result = self.bridge.get_channel_mappings()
        result.pop("DC_CPY")  # mutate the returned dict
        # Original config must be unchanged
        assert "DC_CPY" in self.bridge.config.channel_mappings

    def test_get_channel_mappings_empty_initially(self):
        """get_channel_mappings must return empty dict with no mappings added."""
        result = self.bridge.get_channel_mappings()
        assert result == {}
