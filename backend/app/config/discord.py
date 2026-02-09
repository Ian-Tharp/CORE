"""
Discord Bridge Configuration

Manages Discord bot settings, channel mappings, and user allowlists.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


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
