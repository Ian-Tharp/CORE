"""
Tests for communication controller and repository.

Covers: channel CRUD, message sending (threading, metadata),
reactions, presence updates.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.controllers.communication import (
    CreateChannelRequest,
    SendMessageRequest,
    AddReactionRequest,
    UpdatePresenceRequest,
    create_channel,
    get_channels,
    get_channel,
    get_messages,
    send_message,
    add_reaction,
    remove_reaction,
    get_presence,
    get_instance_presence,
    update_presence,
)


# =============================================================================
# FIXTURES
# =============================================================================

def _make_channel(channel_id="ch_1", name="test", **overrides):
    base = {
        "channel_id": channel_id,
        "channel_type": "global",
        "name": name,
        "description": None,
        "is_persistent": True,
        "is_public": True,
        "created_by": "user_1",
        "created_at": "2026-01-01T00:00:00",
        "metadata": None,
    }
    base.update(overrides)
    return base


def _make_message(message_id=None, channel_id="ch_1", **overrides):
    mid = message_id or str(uuid.uuid4())
    base = {
        "message_id": mid,
        "channel_id": channel_id,
        "sender_id": "user_1",
        "sender_name": "Test User",
        "sender_type": "human",
        "content": "hello",
        "message_type": "text",
        "parent_message_id": None,
        "thread_id": None,
        "created_at": "2026-01-01T00:00:00",
        "edited_at": None,
        "metadata": None,
        "reactions": [],
    }
    base.update(overrides)
    return base


def _mock_comm_repo():
    """Create a mock comm_repo where all methods are AsyncMock."""
    repo = MagicMock()
    repo.list_channels = AsyncMock(return_value=[])
    repo.get_channel = AsyncMock(return_value=None)
    repo.create_channel = AsyncMock(return_value={})
    repo.list_messages = AsyncMock(return_value=[])
    repo.create_message = AsyncMock(return_value={})
    repo.get_message = AsyncMock(return_value=None)
    repo.get_message_reactions = AsyncMock(return_value=[])
    repo.add_reaction = AsyncMock()
    repo.remove_reaction = AsyncMock()
    repo.get_all_presence = AsyncMock(return_value=[])
    repo.get_instance_presence = AsyncMock(return_value=None)
    repo.update_presence = AsyncMock()
    return repo


def _mock_ws_manager():
    """Create a mock websocket manager."""
    ws = MagicMock()
    ws.broadcast_to_channel = AsyncMock()
    ws.broadcast_presence_update = AsyncMock()
    return ws


def _mock_discord_bridge(connected=False):
    """Create a mock Discord bridge."""
    bridge_inst = MagicMock()
    bridge_inst.is_connected = connected
    bridge_fn = MagicMock(return_value=bridge_inst)
    return bridge_fn


def _mock_agent_response_service():
    """Create a mock agent response service."""
    svc = MagicMock()
    svc.process_message = AsyncMock()
    fn = MagicMock(return_value=svc)
    return fn


# =============================================================================
# REQUEST MODEL VALIDATION
# =============================================================================

class TestCreateChannelRequest:
    def test_valid_channel_types(self):
        for ctype in ("global", "team", "dm", "context", "broadcast"):
            req = CreateChannelRequest(channel_type=ctype, name="test")
            assert req.channel_type == ctype

    def test_invalid_channel_type_rejected(self):
        with pytest.raises(Exception):
            CreateChannelRequest(channel_type="invalid", name="test")

    def test_empty_name_rejected(self):
        with pytest.raises(Exception):
            CreateChannelRequest(channel_type="global", name="")

    def test_name_max_length(self):
        with pytest.raises(Exception):
            CreateChannelRequest(channel_type="global", name="x" * 256)


class TestSendMessageRequest:
    def test_valid_message_types(self):
        for mtype in ("text", "markdown", "code", "structured", "event",
                       "pattern", "broadcast", "file", "consciousness_snapshot", "task"):
            req = SendMessageRequest(content="hi", message_type=mtype)
            assert req.message_type == mtype

    def test_empty_content_rejected(self):
        with pytest.raises(Exception):
            SendMessageRequest(content="")

    def test_defaults(self):
        req = SendMessageRequest(content="hi")
        assert req.message_type == "text"
        assert req.parent_message_id is None
        assert req.thread_id is None
        assert req.metadata is None


class TestAddReactionRequest:
    def test_valid_reaction_types(self):
        for rtype in ("resonance", "question", "insight", "acknowledge", "pattern"):
            req = AddReactionRequest(reaction_type=rtype)
            assert req.reaction_type == rtype

    def test_invalid_reaction_type(self):
        with pytest.raises(Exception):
            AddReactionRequest(reaction_type="thumbsup")


class TestUpdatePresenceRequest:
    def test_valid_statuses(self):
        for s in ("online", "away", "busy", "offline"):
            req = UpdatePresenceRequest(status=s)
            assert req.status == s

    def test_phase_bounds(self):
        req = UpdatePresenceRequest(phase=1)
        assert req.phase == 1
        req = UpdatePresenceRequest(phase=4)
        assert req.phase == 4
        with pytest.raises(Exception):
            UpdatePresenceRequest(phase=0)
        with pytest.raises(Exception):
            UpdatePresenceRequest(phase=5)


# =============================================================================
# MESSAGE CONTROLLER WIRING
# =============================================================================

class TestSendMessageEndpoint:
    """Verify that the controller delegates to the shared communication service."""

    @pytest.mark.asyncio
    @patch("app.controllers.communication.get_communication_service")
    async def test_send_message_delegates_to_shared_service(self, mock_get_service):
        # Arrange
        service = MagicMock()
        service.create_and_dispatch_message = AsyncMock(return_value=_make_message())
        mock_get_service.return_value = service
        request = SendMessageRequest(content="reply", parent_message_id="parent-1")

        # Act
        result = await send_message(
            channel_id="ch_1",
            request=request,
            sender_id="user_1",
            sender_name="Test",
            sender_type="human",
        )

        # Assert
        assert result["channel_id"] == "ch_1"
        service.create_and_dispatch_message.assert_awaited_once()
        call_kwargs = service.create_and_dispatch_message.await_args.kwargs
        assert call_kwargs["channel_id"] == "ch_1"
        assert call_kwargs["sender_id"] == "user_1"
        assert call_kwargs["sender_name"] == "Test"
        assert call_kwargs["sender_type"] == "human"
        assert call_kwargs["content"] == "reply"
        assert call_kwargs["parent_message_id"] == "parent-1"


# =============================================================================
# CHANNEL ENDPOINTS
# =============================================================================

class TestGetChannels:
    @pytest.mark.asyncio
    @patch("app.controllers.communication.comm_repo")
    async def test_returns_channel_list(self, mock_repo_mod):
        mock_repo_mod.list_channels = AsyncMock(return_value=[
            _make_channel("ch_1", "General"),
            _make_channel("ch_2", "Random"),
        ])
        result = await get_channels(instance_id="user_1")
        assert len(result["channels"]) == 2
        assert result["channels"][0]["name"] == "General"


class TestGetChannel:
    @pytest.mark.asyncio
    @patch("app.controllers.communication.comm_repo")
    async def test_found(self, mock_repo_mod):
        mock_repo_mod.get_channel = AsyncMock(return_value=_make_channel("ch_1", "General"))
        result = await get_channel("ch_1")
        assert result["channel_id"] == "ch_1"

    @pytest.mark.asyncio
    @patch("app.controllers.communication.comm_repo")
    async def test_not_found_raises_404(self, mock_repo_mod):
        from fastapi import HTTPException
        mock_repo_mod.get_channel = AsyncMock(return_value=None)
        with pytest.raises(HTTPException) as exc_info:
            await get_channel("nonexistent")
        assert exc_info.value.status_code == 404


class TestCreateChannel:
    @pytest.mark.asyncio
    @patch("app.controllers.communication.comm_repo")
    async def test_creates_channel(self, mock_repo_mod):
        mock_repo_mod.create_channel = AsyncMock(return_value=_make_channel("ch_new", "New Channel"))
        req = CreateChannelRequest(channel_type="global", name="New Channel")
        result = await create_channel(request=req, created_by="user_1")
        assert result["name"] == "New Channel"
        mock_repo_mod.create_channel.assert_called_once()


# =============================================================================
# MESSAGE ENDPOINTS
# =============================================================================

class TestGetMessages:
    @pytest.mark.asyncio
    @patch("app.controllers.communication.comm_repo")
    async def test_returns_messages_with_reactions(self, mock_repo_mod):
        msg = _make_message()
        del msg["reactions"]
        mock_repo_mod.list_messages = AsyncMock(return_value=[msg])
        mock_repo_mod.get_message_reactions = AsyncMock(return_value=[
            {"reaction_type": "resonance", "count": 2, "reacted_by": ["u1", "u2"]}
        ])
        result = await get_messages("ch_1", page=1, page_size=50)
        assert len(result["messages"]) == 1
        assert len(result["messages"][0]["reactions"]) == 1

    @pytest.mark.asyncio
    @patch("app.controllers.communication.comm_repo")
    async def test_pagination_params_forwarded(self, mock_repo_mod):
        mock_repo_mod.list_messages = AsyncMock(return_value=[])
        result = await get_messages("ch_1", page=3, page_size=10)
        mock_repo_mod.list_messages.assert_called_once()
        call_kwargs = mock_repo_mod.list_messages.call_args.kwargs
        assert call_kwargs["channel_id"] == "ch_1"
        assert int(call_kwargs["page"]) == 3
        assert int(call_kwargs["page_size"]) == 10
        assert result["page"] == 3
        assert result["page_size"] == 10

    @pytest.mark.asyncio
    @patch("app.controllers.communication.comm_repo")
    async def test_thread_filter(self, mock_repo_mod):
        mock_repo_mod.list_messages = AsyncMock(return_value=[])
        await get_messages("ch_1", thread_id="thread_123")
        mock_repo_mod.list_messages.assert_called_once()
        call_kwargs = mock_repo_mod.list_messages.call_args.kwargs
        assert call_kwargs["channel_id"] == "ch_1"
        assert call_kwargs["thread_id"] == "thread_123"


# =============================================================================
# REACTION ENDPOINTS
# =============================================================================

class TestReactions:
    @pytest.mark.asyncio
    @patch("app.controllers.communication.manager")
    @patch("app.controllers.communication.comm_repo")
    async def test_add_reaction_broadcasts(self, mock_repo_mod, mock_ws):
        mock_repo_mod.add_reaction = AsyncMock()
        mock_repo_mod.get_message = AsyncMock(return_value=_make_message(channel_id="ch_1"))
        mock_ws.broadcast_to_channel = AsyncMock()

        req = AddReactionRequest(reaction_type="resonance")
        await add_reaction(message_id="msg_1", request=req, instance_id="user_1")

        mock_repo_mod.add_reaction.assert_called_once()
        mock_ws.broadcast_to_channel.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.controllers.communication.manager")
    @patch("app.controllers.communication.comm_repo")
    async def test_remove_reaction_broadcasts(self, mock_repo_mod, mock_ws):
        mock_repo_mod.remove_reaction = AsyncMock()
        mock_repo_mod.get_message = AsyncMock(return_value=_make_message(channel_id="ch_1"))
        mock_ws.broadcast_to_channel = AsyncMock()

        await remove_reaction(
            message_id="msg_1", reaction_type="resonance", instance_id="user_1",
        )

        mock_repo_mod.remove_reaction.assert_called_once()
        mock_ws.broadcast_to_channel.assert_called_once()


# =============================================================================
# PRESENCE ENDPOINTS
# =============================================================================

class TestPresence:
    @pytest.mark.asyncio
    @patch("app.controllers.communication.comm_repo")
    async def test_get_all_presence(self, mock_repo_mod):
        mock_repo_mod.get_all_presence = AsyncMock(return_value=[
            {"instance_id": "i1", "status": "online"},
            {"instance_id": "i2", "status": "away"},
        ])
        result = await get_presence()
        assert len(result["instances"]) == 2

    @pytest.mark.asyncio
    @patch("app.controllers.communication.comm_repo")
    async def test_get_instance_presence_found(self, mock_repo_mod):
        mock_repo_mod.get_instance_presence = AsyncMock(return_value={
            "instance_id": "i1", "status": "online",
        })
        result = await get_instance_presence("i1")
        assert result["status"] == "online"

    @pytest.mark.asyncio
    @patch("app.controllers.communication.comm_repo")
    async def test_get_instance_presence_not_found(self, mock_repo_mod):
        from fastapi import HTTPException
        mock_repo_mod.get_instance_presence = AsyncMock(return_value=None)
        with pytest.raises(HTTPException) as exc_info:
            await get_instance_presence("nonexistent")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    @patch("app.controllers.communication.manager")
    @patch("app.controllers.communication.comm_repo")
    async def test_update_presence_broadcasts(self, mock_repo_mod, mock_ws):
        mock_repo_mod.update_presence = AsyncMock()
        mock_ws.broadcast_presence_update = AsyncMock()

        req = UpdatePresenceRequest(status="busy", activity="coding", phase=3)
        await update_presence(instance_id="i1", request=req)

        mock_repo_mod.update_presence.assert_called_once()
        mock_ws.broadcast_presence_update.assert_called_once()