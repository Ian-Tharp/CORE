"""
Discord Bridge REST API Controller

Provides endpoints for managing the Discord bridge:
- Status and health
- Channel mapping management
- Configuration updates
- Manual message sending
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status, Query, Body
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

from app.services.discord_bridge import (
    get_discord_bridge,
    start_discord_bridge,
    stop_discord_bridge,
    BridgeStatus,
)
from app.config.discord import (
    DiscordChannelMapping,
    DiscordConfig,
    get_config,
    update_config,
)

router = APIRouter(prefix="/discord", tags=["discord"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ChannelMappingRequest(BaseModel):
    """Request to create or update a channel mapping."""
    discord_channel_id: str = Field(..., description="Discord channel ID")
    core_channel_id: str = Field(..., description="CORE Communication Commons channel ID")
    discord_channel_name: Optional[str] = Field(None, description="Discord channel name (informational)")
    core_channel_name: Optional[str] = Field(None, description="CORE channel name (for auto-creation)")
    require_mention: bool = Field(False, description="Require @mention to trigger")
    enabled: bool = Field(True, description="Whether the mapping is active")


class SendMessageRequest(BaseModel):
    """Request to send a message to Discord."""
    discord_channel_id: str = Field(..., description="Discord channel ID to send to")
    content: str = Field(..., min_length=1, description="Message content")
    reply_to_message_id: Optional[str] = Field(None, description="Message ID to reply to")


class ConfigUpdateRequest(BaseModel):
    """Request to update Discord bridge configuration."""
    enabled: Optional[bool] = None
    allowed_users: Optional[List[str]] = None
    default_core_channel: Optional[str] = None
    auto_create_channels: Optional[bool] = None
    message_prefix: Optional[str] = None
    response_prefix: Optional[str] = None


class StatusResponse(BaseModel):
    """Response containing bridge status."""
    status: str
    connected: bool
    connected_at: Optional[str]
    last_error: Optional[str]
    reconnect_attempts: int
    bot_user: Optional[str]
    guilds: int
    channel_mappings: int
    bridged_core_channels: List[str]


class ChannelMappingResponse(BaseModel):
    """Response containing a channel mapping."""
    discord_channel_id: str
    discord_channel_name: Optional[str]
    discord_guild_id: Optional[str]
    discord_guild_name: Optional[str]
    core_channel_id: str
    core_channel_name: Optional[str]
    require_mention: bool
    enabled: bool


# =============================================================================
# STATUS ENDPOINTS
# =============================================================================

@router.get("/status", status_code=status.HTTP_200_OK, response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """Get Discord bridge status."""
    bridge = get_discord_bridge()
    info = bridge.get_status_info()
    return StatusResponse(**info)


@router.post("/start", status_code=status.HTTP_200_OK)
async def start_bridge() -> Dict[str, Any]:
    """Start the Discord bridge."""
    success = await start_discord_bridge()
    if success:
        return {"message": "Discord bridge starting", "status": "connecting"}
    else:
        config = get_config()
        if not config.enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Discord bridge is disabled in configuration"
            )
        if not config.bot_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Discord bot token not configured"
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start Discord bridge"
        )


@router.post("/stop", status_code=status.HTTP_200_OK)
async def stop_bridge() -> Dict[str, str]:
    """Stop the Discord bridge."""
    await stop_discord_bridge()
    return {"message": "Discord bridge stopped", "status": "disconnected"}


@router.post("/restart", status_code=status.HTTP_200_OK)
async def restart_bridge() -> Dict[str, str]:
    """Restart the Discord bridge."""
    await stop_discord_bridge()
    success = await start_discord_bridge()
    if success:
        return {"message": "Discord bridge restarting", "status": "connecting"}
    else:
        return {"message": "Discord bridge stopped, failed to restart", "status": "error"}


# =============================================================================
# CHANNEL MAPPING ENDPOINTS
# =============================================================================

@router.get("/channels", status_code=status.HTTP_200_OK)
async def get_channel_mappings() -> Dict[str, Any]:
    """Get all channel mappings."""
    bridge = get_discord_bridge()
    mappings = bridge.get_channel_mappings()
    return {
        "mappings": [
            ChannelMappingResponse(
                discord_channel_id=m.discord_channel_id,
                discord_channel_name=m.discord_channel_name,
                discord_guild_id=m.discord_guild_id,
                discord_guild_name=m.discord_guild_name,
                core_channel_id=m.core_channel_id,
                core_channel_name=m.core_channel_name,
                require_mention=m.require_mention,
                enabled=m.enabled,
            )
            for m in mappings.values()
        ],
        "count": len(mappings)
    }


@router.post("/channels", status_code=status.HTTP_201_CREATED)
async def add_channel_mapping(request: ChannelMappingRequest) -> ChannelMappingResponse:
    """Add or update a channel mapping."""
    bridge = get_discord_bridge()
    
    mapping = DiscordChannelMapping(
        discord_channel_id=request.discord_channel_id,
        discord_channel_name=request.discord_channel_name,
        core_channel_id=request.core_channel_id,
        core_channel_name=request.core_channel_name,
        require_mention=request.require_mention,
        enabled=request.enabled,
    )
    
    bridge.add_channel_mapping(mapping)
    
    return ChannelMappingResponse(
        discord_channel_id=mapping.discord_channel_id,
        discord_channel_name=mapping.discord_channel_name,
        discord_guild_id=mapping.discord_guild_id,
        discord_guild_name=mapping.discord_guild_name,
        core_channel_id=mapping.core_channel_id,
        core_channel_name=mapping.core_channel_name,
        require_mention=mapping.require_mention,
        enabled=mapping.enabled,
    )


@router.delete("/channels/{discord_channel_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_channel_mapping(discord_channel_id: str):
    """Remove a channel mapping."""
    bridge = get_discord_bridge()
    
    if not bridge.remove_channel_mapping(discord_channel_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No mapping found for Discord channel {discord_channel_id}"
        )
    
    return None


# =============================================================================
# MESSAGE ENDPOINTS
# =============================================================================

@router.post("/send", status_code=status.HTTP_200_OK)
async def send_message(request: SendMessageRequest) -> Dict[str, Any]:
    """Send a message to a Discord channel."""
    bridge = get_discord_bridge()
    
    if not bridge.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Discord bridge is not connected"
        )
    
    success = await bridge.send_to_discord(
        discord_channel_id=request.discord_channel_id,
        content=request.content,
        reply_to_message_id=request.reply_to_message_id,
    )
    
    if success:
        return {"message": "Message sent", "success": True}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send message to Discord"
        )


# =============================================================================
# CONFIGURATION ENDPOINTS
# =============================================================================

@router.get("/config", status_code=status.HTTP_200_OK)
async def get_configuration() -> Dict[str, Any]:
    """Get Discord bridge configuration (excluding sensitive data)."""
    config = get_config()
    return {
        "enabled": config.enabled,
        "has_token": bool(config.bot_token),
        "allowed_users": config.allowed_users,
        "default_core_channel": config.default_core_channel,
        "auto_create_channels": config.auto_create_channels,
        "message_prefix": config.message_prefix,
        "response_prefix": config.response_prefix,
        "bot_instance_id": config.bot_instance_id,
        "bot_display_name": config.bot_display_name,
        "channel_mappings_count": len(config.channel_mappings),
    }


@router.patch("/config", status_code=status.HTTP_200_OK)
async def update_configuration(request: ConfigUpdateRequest) -> Dict[str, Any]:
    """Update Discord bridge configuration."""
    config = get_config()
    
    if request.enabled is not None:
        config.enabled = request.enabled
    if request.allowed_users is not None:
        config.allowed_users = request.allowed_users
    if request.default_core_channel is not None:
        config.default_core_channel = request.default_core_channel
    if request.auto_create_channels is not None:
        config.auto_create_channels = request.auto_create_channels
    if request.message_prefix is not None:
        config.message_prefix = request.message_prefix
    if request.response_prefix is not None:
        config.response_prefix = request.response_prefix
    
    update_config(config)
    
    return {
        "message": "Configuration updated",
        "note": "Restart bridge for changes to take effect"
    }
