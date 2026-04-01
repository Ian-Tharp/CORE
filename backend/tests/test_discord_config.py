from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest

from app.config import discord as discord_config_module


@pytest.fixture(autouse=True)
def reset_discord_config_singleton():
    discord_config_module._discord_config = None
    yield
    discord_config_module._discord_config = None


class TestLoadConfigFromStore:
    @pytest.mark.asyncio
    async def test_loads_persisted_values_and_preserves_env_token(self):
        stored_config = {
            "enabled": True,
            "allowed_users": ["user-1", "user-2"],
            "default_core_channel": "discord_general",
            "auto_create_channels": False,
            "message_prefix": "[discord] ",
            "response_prefix": "[core] ",
            "reconnect_delay_seconds": 9,
            "max_reconnect_attempts": 42,
        }
        stored_mapping = {
            "discord_channel_id": "123",
            "discord_channel_name": "general",
            "discord_guild_id": "456",
            "discord_guild_name": "CORE Guild",
            "core_channel_id": "discord_general",
            "core_channel_name": "Discord General",
            "require_mention": True,
            "enabled": True,
        }

        with patch.dict(
            os.environ,
            {
                "DISCORD_BOT_TOKEN": "env-secret-token",
                "DISCORD_ENABLED": "false",
            },
            clear=False,
        ):
            with patch(
                "app.config.discord.discord_repository.get_bridge_config",
                new=AsyncMock(return_value=stored_config),
            ), patch(
                "app.config.discord.discord_repository.list_channel_mappings",
                new=AsyncMock(return_value=[stored_mapping]),
            ):
                # Act
                config = await discord_config_module.load_config_from_store()

        # Assert
        assert config.bot_token == "env-secret-token"
        assert config.enabled is True
        assert config.allowed_users == ["user-1", "user-2"]
        assert config.default_core_channel == "discord_general"
        assert config.auto_create_channels is False
        assert config.message_prefix == "[discord] "
        assert config.response_prefix == "[core] "
        assert config.reconnect_delay_seconds == 9
        assert config.max_reconnect_attempts == 42
        assert "123" in config.channel_mappings
        assert config.channel_mappings["123"].core_channel_id == "discord_general"
        assert config.channel_mappings["123"].require_mention is True

    @pytest.mark.asyncio
    async def test_keeps_environment_channel_mappings_when_store_is_empty(self):
        with patch.dict(
            os.environ,
            {
                "DISCORD_BOT_TOKEN": "env-secret-token",
                "DISCORD_CHANNEL_MAP": "111:discord_general",
            },
            clear=False,
        ):
            with patch(
                "app.config.discord.discord_repository.get_bridge_config",
                new=AsyncMock(return_value=None),
            ), patch(
                "app.config.discord.discord_repository.list_channel_mappings",
                new=AsyncMock(return_value=[]),
            ):
                # Act
                config = await discord_config_module.load_config_from_store()

        # Assert
        assert config.bot_token == "env-secret-token"
        assert "111" in config.channel_mappings
        assert config.channel_mappings["111"].core_channel_id == "discord_general"


class TestPersistConfig:
    @pytest.mark.asyncio
    async def test_persists_only_non_sensitive_settings(self):
        config = discord_config_module.DiscordConfig(
            bot_token="do-not-persist-me",
            enabled=True,
            allowed_users=["user-1"],
            default_core_channel="discord_general",
            auto_create_channels=False,
            message_prefix="[discord] ",
            response_prefix="[core] ",
            reconnect_delay_seconds=7,
            max_reconnect_attempts=12,
        )

        with patch(
            "app.config.discord.discord_repository.upsert_bridge_config",
            new=AsyncMock(return_value={}),
        ) as mock_upsert:
            # Act
            await discord_config_module.persist_config(config)

        # Assert
        call_kwargs = mock_upsert.await_args.kwargs
        assert call_kwargs["enabled"] is True
        assert call_kwargs["allowed_users"] == ["user-1"]
        assert call_kwargs["default_core_channel"] == "discord_general"
        assert call_kwargs["auto_create_channels"] is False
        assert call_kwargs["message_prefix"] == "[discord] "
        assert call_kwargs["response_prefix"] == "[core] "
        assert call_kwargs["reconnect_delay_seconds"] == 7
        assert call_kwargs["max_reconnect_attempts"] == 12
        assert "bot_token" not in call_kwargs


class TestPersistChannelMapping:
    @pytest.mark.asyncio
    async def test_persist_channel_mapping_updates_singleton(self):
        discord_config_module.update_config(
            discord_config_module.DiscordConfig(bot_token="env-token")
        )
        mapping = discord_config_module.DiscordChannelMapping(
            discord_channel_id="123",
            discord_channel_name="general",
            core_channel_id="discord_general",
            core_channel_name="Discord General",
            require_mention=False,
            enabled=True,
        )

        with patch(
            "app.config.discord.discord_repository.upsert_channel_mapping",
            new=AsyncMock(
                return_value={
                    "discord_channel_id": "123",
                    "discord_channel_name": "general",
                    "discord_guild_id": "456",
                    "discord_guild_name": "CORE Guild",
                    "core_channel_id": "discord_general",
                    "core_channel_name": "Discord General",
                    "require_mention": False,
                    "enabled": True,
                }
            ),
        ):
            # Act
            persisted = await discord_config_module.persist_channel_mapping(mapping)

        # Assert
        assert persisted.discord_channel_id == "123"
        assert persisted.discord_guild_name == "CORE Guild"
        assert discord_config_module.get_config().channel_mappings["123"].discord_guild_id == "456"


class TestDeleteChannelMapping:
    @pytest.mark.asyncio
    async def test_delete_channel_mapping_removes_singleton_entry(self):
        config = discord_config_module.DiscordConfig(bot_token="env-token")
        config.channel_mappings["123"] = discord_config_module.DiscordChannelMapping(
            discord_channel_id="123",
            discord_channel_name="general",
            core_channel_id="discord_general",
        )
        discord_config_module.update_config(config)

        with patch(
            "app.config.discord.discord_repository.delete_channel_mapping",
            new=AsyncMock(return_value=True),
        ):
            # Act
            deleted = await discord_config_module.delete_channel_mapping("123")

        # Assert
        assert deleted is True
        assert "123" not in discord_config_module.get_config().channel_mappings
