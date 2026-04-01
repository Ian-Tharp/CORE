"""
Discord Bridge Configuration

Manages Discord bot settings, channel mappings, and user allowlists.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from app.repository import discord_repository


class DiscordChannelMapping(BaseModel):
    """Maps a Discord channel to a CORE Communication Commons channel."""
    
    discord_channel_id: str
    discord_channel_name: Optional[str] = None
    discord_guild_id: Optional[str] = None
    discord_guild_name: Optional[str] = None
    core_channel_id: str
    core_channel_name: Optional[str] = None
    require_mention: bool = False
    enabled: bool = True


class DiscordConfig(BaseModel):
    """Configuration for the Discord bridge."""
    
    # Bot authentication
    bot_token: str = Field(default_factory=lambda: os.getenv("DISCORD_BOT_TOKEN", ""))
    
    # Feature flags
    enabled: bool = Field(default_factory=lambda: os.getenv("DISCORD_ENABLED", "false").lower() == "true")
    
    # Channel mappings (Discord channel ID → CORE channel ID)
    channel_mappings: Dict[str, DiscordChannelMapping] = Field(default_factory=dict)
    
    # User allowlist (Discord user IDs that can interact)
    # Empty list means all users allowed
    allowed_users: List[str] = Field(default_factory=list)
    
    # Default CORE channel for unmapped Discord channels
    default_core_channel: Optional[str] = None
    
    # Whether to create CORE channels automatically for unmapped Discord channels
    auto_create_channels: bool = True
    
    # Message settings
    message_prefix: str = ""  # Prefix to add to messages from Discord
    response_prefix: str = ""  # Prefix to add to responses going to Discord
    
    # Bot identity in CORE
    bot_instance_id: str = "discord_bridge"
    bot_display_name: str = "Discord Bridge"
    
    # Reconnection settings
    reconnect_delay_seconds: int = 5
    max_reconnect_attempts: int = 10


def get_discord_config() -> DiscordConfig:
    """Get Discord configuration from environment and defaults."""
    
    config = DiscordConfig()
    
    # Load channel mappings from environment if provided
    # Format: DISCORD_CHANNEL_MAP="discord_id:core_id,discord_id2:core_id2"
    channel_map_env = os.getenv("DISCORD_CHANNEL_MAP", "")
    if channel_map_env:
        for mapping in channel_map_env.split(","):
            if ":" in mapping:
                discord_id, core_id = mapping.strip().split(":", 1)
                config.channel_mappings[discord_id] = DiscordChannelMapping(
                    discord_channel_id=discord_id,
                    core_channel_id=core_id
                )
    
    # Load allowed users from environment
    # Format: DISCORD_ALLOWED_USERS="user_id1,user_id2"
    allowed_users_env = os.getenv("DISCORD_ALLOWED_USERS", "")
    if allowed_users_env:
        config.allowed_users = [u.strip() for u in allowed_users_env.split(",") if u.strip()]
    
    return config


async def load_config_from_store() -> DiscordConfig:
    """
    Load persisted Discord bridge configuration and merge it with environment
    values for sensitive settings like the bot token.
    """
    config = get_discord_config()

    stored_config = await discord_repository.get_bridge_config()
    if stored_config:
        config.enabled = stored_config.get("enabled", config.enabled)
        config.allowed_users = stored_config.get("allowed_users") or []
        config.default_core_channel = stored_config.get("default_core_channel")
        config.auto_create_channels = stored_config.get(
            "auto_create_channels",
            config.auto_create_channels,
        )
        config.message_prefix = stored_config.get("message_prefix") or ""
        config.response_prefix = stored_config.get("response_prefix") or ""
        config.reconnect_delay_seconds = stored_config.get(
            "reconnect_delay_seconds",
            config.reconnect_delay_seconds,
        )
        config.max_reconnect_attempts = stored_config.get(
            "max_reconnect_attempts",
            config.max_reconnect_attempts,
        )

    stored_mappings = await discord_repository.list_channel_mappings()
    if stored_mappings:
        config.channel_mappings = {
            mapping["discord_channel_id"]: DiscordChannelMapping(**mapping)
            for mapping in stored_mappings
        }

    update_config(config)
    return config


async def persist_config(config: DiscordConfig) -> DiscordConfig:
    """Persist the non-sensitive Discord bridge configuration."""
    await discord_repository.upsert_bridge_config(
        enabled=config.enabled,
        allowed_users=config.allowed_users,
        default_core_channel=config.default_core_channel,
        auto_create_channels=config.auto_create_channels,
        message_prefix=config.message_prefix,
        response_prefix=config.response_prefix,
        reconnect_delay_seconds=config.reconnect_delay_seconds,
        max_reconnect_attempts=config.max_reconnect_attempts,
    )
    update_config(config)
    return config


async def persist_channel_mapping(mapping: DiscordChannelMapping) -> DiscordChannelMapping:
    """Persist a Discord channel mapping and update the config singleton."""
    stored_mapping = await discord_repository.upsert_channel_mapping(
        discord_channel_id=mapping.discord_channel_id,
        discord_channel_name=mapping.discord_channel_name,
        discord_guild_id=mapping.discord_guild_id,
        discord_guild_name=mapping.discord_guild_name,
        core_channel_id=mapping.core_channel_id,
        core_channel_name=mapping.core_channel_name,
        require_mention=mapping.require_mention,
        enabled=mapping.enabled,
    )
    persisted = DiscordChannelMapping(**stored_mapping)

    config = get_config()
    config.channel_mappings[persisted.discord_channel_id] = persisted
    update_config(config)
    return persisted


async def delete_channel_mapping(discord_channel_id: str) -> bool:
    """Delete a Discord channel mapping from storage and the config singleton."""
    deleted = await discord_repository.delete_channel_mapping(discord_channel_id)
    if deleted:
        config = get_config()
        config.channel_mappings.pop(discord_channel_id, None)
        update_config(config)
    return deleted


# Singleton instance
_discord_config: Optional[DiscordConfig] = None


def get_config() -> DiscordConfig:
    """Get or create the Discord config singleton."""
    global _discord_config
    if _discord_config is None:
        _discord_config = get_discord_config()
    return _discord_config


def update_config(config: DiscordConfig) -> None:
    """Update the Discord config singleton."""
    global _discord_config
    _discord_config = config
