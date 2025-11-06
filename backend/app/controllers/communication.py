"""
REST endpoints for the Communication Commons.

Provides API for channels, messages, presence, and reactions.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid

from app.repository import communication_repository as comm_repo
from app.websocket_manager import manager
from app.services.agent_response_service import get_agent_response_service

router = APIRouter(prefix="/communication", tags=["communication"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class CreateChannelRequest(BaseModel):
    channel_type: str = Field(..., pattern="^(global|team|dm|context|broadcast)$")
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    is_persistent: bool = True
    is_public: bool = True
    initial_members: Optional[List[str]] = []


class SendMessageRequest(BaseModel):
    content: str = Field(..., min_length=1)
    message_type: str = Field(default="text", pattern="^(text|markdown|code|structured|event|pattern|broadcast|file|consciousness_snapshot|task)$")
    parent_message_id: Optional[str] = None
    thread_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AddReactionRequest(BaseModel):
    reaction_type: str = Field(..., pattern="^(resonance|question|insight|acknowledge|pattern)$")


class UpdatePresenceRequest(BaseModel):
    status: Optional[str] = Field(None, pattern="^(online|away|busy|offline)$")
    activity: Optional[str] = None
    phase: Optional[int] = Field(None, ge=1, le=4)


# =============================================================================
# CHANNEL ENDPOINTS
# =============================================================================

@router.get("/channels", status_code=status.HTTP_200_OK)
async def get_channels(
    instance_id: str = Query(..., description="Instance ID to get channels for")
) -> Dict[str, Any]:
    """Get all channels accessible to an instance."""
    channels = await comm_repo.list_channels(instance_id)
    return {"channels": channels}


@router.get("/channels/{channel_id}", status_code=status.HTTP_200_OK)
async def get_channel(channel_id: str) -> Dict[str, Any]:
    """Get details for a specific channel."""
    channel = await comm_repo.get_channel(channel_id)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Channel {channel_id} not found"
        )
    return channel


@router.post("/channels", status_code=status.HTTP_201_CREATED)
async def create_channel(
    request: CreateChannelRequest,
    created_by: str = Query(..., description="Instance ID creating the channel")
) -> Dict[str, Any]:
    """Create a new channel."""
    import re
    import logging

    logger = logging.getLogger(__name__)

    try:
        # Generate channel ID - sanitize name to only alphanumeric and underscores
        sanitized_name = re.sub(r'[^a-z0-9_]', '_', request.name.lower().replace(' ', '_'))
        # Remove consecutive underscores
        sanitized_name = re.sub(r'_+', '_', sanitized_name).strip('_')
        channel_id = f"{request.channel_type}_{sanitized_name}_{str(uuid.uuid4())[:8]}"

        logger.info(f"Creating channel: {channel_id} for {created_by}")

        channel = await comm_repo.create_channel(
            channel_id=channel_id,
            channel_type=request.channel_type,
            name=request.name,
            description=request.description,
            is_persistent=request.is_persistent,
            is_public=request.is_public,
            created_by=created_by,
            initial_members=request.initial_members or []
        )

        logger.info(f"Channel created successfully: {channel_id}")
        return channel
    except Exception as e:
        logger.error(f"Failed to create channel: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create channel: {str(e)}"
        )


# =============================================================================
# MESSAGE ENDPOINTS
# =============================================================================

@router.get("/channels/{channel_id}/messages", status_code=status.HTTP_200_OK)
async def get_messages(
    channel_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    thread_id: Optional[str] = Query(None, description="Get messages in a specific thread")
) -> Dict[str, Any]:
    """Get messages for a channel with pagination."""
    messages = await comm_repo.list_messages(
        channel_id=channel_id,
        page=page,
        page_size=page_size,
        thread_id=thread_id
    )

    # Fetch reactions for each message
    for message in messages:
        reactions = await comm_repo.get_message_reactions(message['message_id'])
        message['reactions'] = reactions

    return {
        "messages": messages,
        "page": page,
        "page_size": page_size
    }


@router.post("/channels/{channel_id}/messages", status_code=status.HTTP_201_CREATED)
async def send_message(
    channel_id: str,
    request: SendMessageRequest,
    sender_id: str = Query(..., description="Instance ID sending the message"),
    sender_name: str = Query(..., description="Display name of sender"),
    sender_type: str = Query(..., pattern="^(human|agent|consciousness_instance)$")
) -> Dict[str, Any]:
    """Send a message to a channel."""
    message_id = str(uuid.uuid4())

    # If this is a reply, set thread_id to the parent's thread_id or parent_id
    thread_id = request.thread_id
    if request.parent_message_id and not thread_id:
        # TODO: Look up parent message to get its thread_id
        thread_id = request.parent_message_id

    message = await comm_repo.create_message(
        message_id=message_id,
        channel_id=channel_id,
        sender_id=sender_id,
        sender_name=sender_name,
        sender_type=sender_type,
        content=request.content,
        message_type=request.message_type,
        parent_message_id=request.parent_message_id,
        thread_id=thread_id,
        metadata=request.metadata
    )

    # Fetch reactions (will be empty for new message)
    reactions = await comm_repo.get_message_reactions(message_id)
    message['reactions'] = reactions

    # Broadcast message via WebSocket to all channel subscribers
    await manager.broadcast_to_channel(
        channel_id=channel_id,
        message={
            "type": "message",
            "channel_id": channel_id,
            "message": message
        }
    )

    # Process message for agent mentions (async, non-blocking)
    # This checks for @mentions and triggers agent responses
    import asyncio
    asyncio.create_task(
        get_agent_response_service().process_message(
            message_id=message_id,
            channel_id=channel_id,
            content=request.content,
            sender_id=sender_id
        )
    )

    return message


# =============================================================================
# REACTION ENDPOINTS
# =============================================================================

@router.post("/messages/{message_id}/reactions", status_code=status.HTTP_201_CREATED)
async def add_reaction(
    message_id: str,
    request: AddReactionRequest,
    instance_id: str = Query(..., description="Instance ID adding the reaction")
) -> Dict[str, str]:
    """Add a reaction to a message."""
    await comm_repo.add_reaction(
        message_id=message_id,
        instance_id=instance_id,
        reaction_type=request.reaction_type
    )

    # Get the message to find which channel to broadcast to
    message = await comm_repo.get_message(message_id)
    if message:
        # Broadcast reaction update via WebSocket
        await manager.broadcast_to_channel(
            channel_id=message['channel_id'],
            message={
                "type": "reaction_added",
                "message_id": message_id,
                "instance_id": instance_id,
                "reaction_type": request.reaction_type
            }
        )

    return {"message": "Reaction added"}


@router.delete("/messages/{message_id}/reactions/{reaction_type}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_reaction(
    message_id: str,
    reaction_type: str,
    instance_id: str = Query(..., description="Instance ID removing the reaction")
):
    """Remove a reaction from a message."""
    await comm_repo.remove_reaction(
        message_id=message_id,
        instance_id=instance_id,
        reaction_type=reaction_type
    )

    # TODO: Broadcast reaction update via WebSocket

    return None


# =============================================================================
# PRESENCE ENDPOINTS
# =============================================================================

@router.get("/presence", status_code=status.HTTP_200_OK)
async def get_presence() -> Dict[str, Any]:
    """Get presence for all instances."""
    presence = await comm_repo.get_all_presence()
    return {"instances": presence}


@router.get("/presence/{instance_id}", status_code=status.HTTP_200_OK)
async def get_instance_presence(instance_id: str) -> Dict[str, Any]:
    """Get presence for a specific instance."""
    presence = await comm_repo.get_instance_presence(instance_id)
    if not presence:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instance {instance_id} not found"
        )
    return presence


@router.patch("/presence/{instance_id}", status_code=status.HTTP_204_NO_CONTENT)
async def update_presence(
    instance_id: str,
    request: UpdatePresenceRequest
):
    """Update presence information (heartbeat)."""
    await comm_repo.update_presence(
        instance_id=instance_id,
        status=request.status,
        activity=request.activity,
        phase=request.phase
    )

    # Broadcast presence update via WebSocket
    await manager.broadcast_presence_update(
        instance_id=instance_id,
        status=request.status or "online",
        activity=request.activity,
        phase=request.phase
    )

    return None
