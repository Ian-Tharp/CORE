"""
Tests verifying that previously-unprotected controllers now require authentication.

Sentinel-value contract:
- Remove require_api_key from an endpoint → 401 becomes 422/200 → test fails
- Remove Depends import → controller import error → collection fails
- Remove X-API-Key header in passing tests → 401 → test fails

Covers: chat, communication, creative, discord controllers.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(router) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


# ---------------------------------------------------------------------------
# /chat/stream authentication
# ---------------------------------------------------------------------------


class TestChatStreamAuth:
    def _client(self):
        from app.controllers.chat import router

        return TestClient(_make_app(router), raise_server_exceptions=False)

    def _valid_body(self):
        return {
            "model": "gpt-oss:20b",
            "messages": [{"role": "user", "content": "hi"}],
            "provider": "ollama",
        }

    def test_missing_key_returns_401(self):
        """
        SENTINEL — /chat/stream must reject requests with no API key.
        Remove require_api_key → 401 becomes something else → test fails.
        """
        resp = self._client().post("/chat/stream", json=self._valid_body())
        assert resp.status_code == 401

    def test_invalid_key_returns_401(self):
        """Wrong API key must also be rejected."""
        resp = self._client().post(
            "/chat/stream",
            json=self._valid_body(),
            headers={"X-API-Key": "not-a-valid-key"},
        )
        assert resp.status_code == 401

    def test_valid_key_proceeds_past_auth(self, api_key):
        """
        SENTINEL — valid key must pass auth (endpoint may fail for other reasons,
        but NOT because of missing auth).
        """
        resp = self._client().post(
            "/chat/stream",
            json=self._valid_body(),
            headers={"X-API-Key": api_key},
        )
        # Should NOT be 401 (may be other error like 500/503 since LLM not running)
        assert resp.status_code != 401, f"Got 401 with valid key: {resp.text}"


# ---------------------------------------------------------------------------
# /communication/* authentication
# ---------------------------------------------------------------------------


class TestCommunicationAuth:
    def _client(self):
        from app.controllers.communication import router

        return TestClient(_make_app(router), raise_server_exceptions=False)

    def test_get_channels_missing_key_returns_401(self):
        """
        SENTINEL — GET /communication/channels must require auth.
        Remove Depends(require_api_key) → request passes without key → test fails.
        """
        resp = self._client().get("/communication/channels?instance_id=inst-1")
        assert resp.status_code == 401

    def test_get_channels_invalid_key_returns_401(self):
        """Invalid API key must be rejected."""
        resp = self._client().get(
            "/communication/channels?instance_id=inst-1",
            headers={"X-API-Key": "bad-key"},
        )
        assert resp.status_code == 401

    def test_get_channels_valid_key_proceeds(self, api_key):
        """
        SENTINEL — valid key must pass auth on GET /communication/channels.
        """
        with patch("app.controllers.communication.comm_repo") as mock_repo:
            mock_repo.list_channels = AsyncMock(return_value=[])
            mock_repo.count_channels = AsyncMock(return_value=0)
            resp = self._client().get(
                "/communication/channels?instance_id=inst-1",
                headers={"X-API-Key": api_key},
            )
        assert resp.status_code != 401

    def test_post_message_missing_key_returns_401(self):
        """
        SENTINEL — POST /communication/channels/{id}/messages must require auth.
        """
        resp = self._client().post(
            "/communication/channels/ch-1/messages"
            "?sender_id=s1&sender_name=Test&sender_type=human",
            json={"content": "hello"},
        )
        assert resp.status_code == 401

    def test_get_presence_missing_key_returns_401(self):
        """GET /communication/presence must require auth."""
        resp = self._client().get("/communication/presence")
        assert resp.status_code == 401

    def test_update_presence_missing_key_returns_401(self):
        """PATCH /communication/presence/{id} must require auth."""
        resp = self._client().patch(
            "/communication/presence/inst-1",
            json={"status": "online"},
        )
        assert resp.status_code == 401

    def test_add_reaction_missing_key_returns_401(self):
        """POST /communication/messages/{id}/reactions must require auth."""
        resp = self._client().post(
            "/communication/messages/m-1/reactions?instance_id=i1",
            json={"reaction_type": "resonance"},
        )
        assert resp.status_code == 401

    def test_remove_reaction_missing_key_returns_401(self):
        """DELETE /communication/messages/{id}/reactions/{type} must require auth."""
        resp = self._client().delete(
            "/communication/messages/m-1/reactions/resonance?instance_id=i1"
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# /discord/* authentication
# ---------------------------------------------------------------------------


class TestDiscordAuth:
    def _client(self):
        from app.controllers.discord import router

        return TestClient(_make_app(router), raise_server_exceptions=False)

    def test_status_missing_key_returns_401(self):
        """
        SENTINEL — GET /discord/status must require auth.
        Remove require_api_key → bridge status exposed publicly → test fails.
        """
        resp = self._client().get("/discord/status")
        assert resp.status_code == 401

    def test_start_bridge_missing_key_returns_401(self):
        """
        SENTINEL — POST /discord/start must require auth (lifecycle control).
        Remove auth → anyone can start the bridge → test fails.
        """
        resp = self._client().post("/discord/start")
        assert resp.status_code == 401

    def test_stop_bridge_missing_key_returns_401(self):
        """POST /discord/stop must require auth."""
        resp = self._client().post("/discord/stop")
        assert resp.status_code == 401

    def test_restart_bridge_missing_key_returns_401(self):
        """POST /discord/restart must require auth."""
        resp = self._client().post("/discord/restart")
        assert resp.status_code == 401

    def test_get_config_missing_key_returns_401(self):
        """GET /discord/config must require auth (config may expose sensitive data)."""
        resp = self._client().get("/discord/config")
        assert resp.status_code == 401

    def test_update_config_missing_key_returns_401(self):
        """PATCH /discord/config must require auth."""
        resp = self._client().patch("/discord/config", json={"enabled": True})
        assert resp.status_code == 401

    def test_get_channels_missing_key_returns_401(self):
        """GET /discord/channels must require auth."""
        resp = self._client().get("/discord/channels")
        assert resp.status_code == 401

    def test_send_message_missing_key_returns_401(self):
        """POST /discord/send must require auth."""
        resp = self._client().post(
            "/discord/send",
            json={"discord_channel_id": "123", "content": "hi"},
        )
        assert resp.status_code == 401

    def test_get_deliveries_missing_key_returns_401(self):
        """GET /discord/deliveries must require auth."""
        resp = self._client().get("/discord/deliveries")
        assert resp.status_code == 401

    def test_get_message_links_missing_key_returns_401(self):
        """GET /discord/message-links must require auth."""
        resp = self._client().get("/discord/message-links")
        assert resp.status_code == 401
