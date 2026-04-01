"""
Tests for AgentMCPService — cache validity, tool filtering, config building,
and graceful degradation when MCP adapters are absent.

Sentinel-value contract:
- Remove _is_cache_valid TTL check → stale tools served indefinitely → test fails
- Remove allowed_tools filter → agents receive all tools instead of their subset → test fails
- Remove unknown server fallback → agent_config ignored, service returns [] → test fails
- Remove empty env-var stripping → blank secrets sent to subprocess env → test fails
- Remove clear_cache(server_id) pop → specific cache eviction silently fails → test fails
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.agent_models import MCPServerConfig
from app.services.agent_mcp_service import AgentMCPService, get_agent_mcp_service


# ===========================================================================
# Helpers
# ===========================================================================

def _make_service() -> AgentMCPService:
    return AgentMCPService()


def _make_tool(name: str) -> MagicMock:
    t = MagicMock()
    t.name = name
    return t


def _make_mcp_config(
    server_id: str = "test-server",
    tools: List[str] | None = None,
    config: Dict[str, Any] | None = None,
) -> MCPServerConfig:
    return MCPServerConfig(
        server_id=server_id,
        tools=tools or [],
        config=config or {},
    )


# ===========================================================================
# _is_cache_valid — TTL logic
# ===========================================================================

class TestIsCacheValid:
    def test_missing_server_returns_false(self):
        """
        SENTINEL — server not in cache must return False.
        Return True always → every call uses stale (empty) cache → test fails.
        """
        svc = _make_service()
        assert svc._is_cache_valid("nonexistent") is False

    def test_missing_timestamp_returns_false(self):
        """Server in tool cache but without timestamp entry must return False."""
        svc = _make_service()
        svc._tool_cache["srv"] = {}
        # _last_cache_update not set
        assert svc._is_cache_valid("srv") is False

    def test_fresh_cache_returns_true(self):
        """Cache updated just now must be valid."""
        svc = _make_service()
        svc._tool_cache["srv"] = {"tool-1": _make_tool("tool-1")}
        svc._last_cache_update["srv"] = datetime.utcnow()
        assert svc._is_cache_valid("srv") is True

    def test_expired_cache_returns_false(self):
        """
        SENTINEL — cache older than TTL must return False.
        Remove age check → stale tools served forever → test fails.
        """
        svc = _make_service()
        svc._tool_cache["srv"] = {"tool-1": _make_tool("tool-1")}
        svc._last_cache_update["srv"] = datetime.utcnow() - timedelta(minutes=10)
        assert svc._is_cache_valid("srv") is False

    def test_cache_exactly_at_ttl_is_expired(self):
        """Cache age == TTL must be considered expired (not strictly less-than)."""
        svc = _make_service()
        svc._tool_cache["srv"] = {}
        svc._last_cache_update["srv"] = datetime.utcnow() - svc._cache_ttl - timedelta(seconds=1)
        assert svc._is_cache_valid("srv") is False


# ===========================================================================
# clear_cache — selective and full eviction
# ===========================================================================

class TestClearCache:
    def test_clear_all_removes_all_entries(self):
        """
        SENTINEL — clear_cache() with no args must empty both dicts.
        Only clear tool_cache → timestamps persist, next validity check broken → test fails.
        """
        svc = _make_service()
        svc._tool_cache["a"] = {}
        svc._tool_cache["b"] = {}
        svc._last_cache_update["a"] = datetime.utcnow()
        svc._last_cache_update["b"] = datetime.utcnow()
        svc.clear_cache()
        assert len(svc._tool_cache) == 0
        assert len(svc._last_cache_update) == 0

    def test_clear_specific_removes_only_that_server(self):
        """
        SENTINEL — clear_cache("a") must remove only server "a", leave "b" intact.
        Call .clear() always → all servers evicted → test fails.
        """
        svc = _make_service()
        svc._tool_cache["a"] = {}
        svc._tool_cache["b"] = {}
        svc._last_cache_update["a"] = datetime.utcnow()
        svc._last_cache_update["b"] = datetime.utcnow()
        svc.clear_cache("a")
        assert "a" not in svc._tool_cache
        assert "a" not in svc._last_cache_update
        assert "b" in svc._tool_cache
        assert "b" in svc._last_cache_update

    def test_clear_nonexistent_server_does_not_raise(self):
        """Clearing a server that doesn't exist must not raise."""
        svc = _make_service()
        svc.clear_cache("ghost")  # should not raise


# ===========================================================================
# _get_server_config — config building
# ===========================================================================

class TestGetServerConfig:
    def test_known_server_obsidian_has_stdio_transport(self):
        """'mcp-obsidian' must return a stdio transport config."""
        svc = _make_service()
        config = svc._get_server_config("mcp-obsidian", {})
        assert config["transport"] == "stdio"

    def test_known_server_memory_uses_npx(self):
        """'memory' must use npx as command."""
        svc = _make_service()
        config = svc._get_server_config("memory", {})
        assert config["command"] == "npx"

    def test_unknown_server_falls_back_to_agent_config(self):
        """
        SENTINEL — unrecognised server_id must use agent_config as fallback.
        Return {} always → agent's custom server never connected → test fails.
        """
        svc = _make_service()
        agent_cfg = {"url": "http://SENTINEL_HOST:9999", "transport": "http"}
        config = svc._get_server_config("custom-server", agent_cfg)
        assert config["url"] == "http://SENTINEL_HOST:9999"

    def test_empty_env_vars_stripped(self):
        """
        SENTINEL — blank env values must be removed from the config.
        Keep empty strings → subprocess receives blank env vars / blank secrets → test fails.
        """
        svc = _make_service()
        # Ensure OBSIDIAN_API_KEY is empty in env so _get_server_config produces blank
        import os
        with patch.dict(os.environ, {"OBSIDIAN_API_KEY": "", "OBSIDIAN_VAULT_PATH": ""}):
            config = svc._get_server_config("mcp-obsidian", {})
        env = config.get("env", {})
        for val in env.values():
            assert val != "", f"Empty env var found: {env}"

    def test_agent_config_overrides_env_for_obsidian(self):
        """agent_config values must override os.getenv for Obsidian config."""
        svc = _make_service()
        import os
        with patch.dict(os.environ, {"OBSIDIAN_API_KEY": "env_key"}):
            config = svc._get_server_config("mcp-obsidian", {"OBSIDIAN_API_KEY": "SENTINEL_OVERRIDE"})
        assert config["env"].get("OBSIDIAN_API_KEY") == "SENTINEL_OVERRIDE"


# ===========================================================================
# get_tools_for_agent — filtering and error isolation
# ===========================================================================

class TestGetToolsForAgent:
    @pytest.mark.asyncio
    async def test_no_mcp_adapters_returns_empty(self):
        """When langchain-mcp-adapters is not installed, must return []."""
        svc = _make_service()
        with patch("app.services.agent_mcp_service._HAS_MCP_ADAPTERS", False):
            result = await svc.get_tools_for_agent([_make_mcp_config()])
        assert result == []

    @pytest.mark.asyncio
    async def test_allowed_tools_filter_applied(self):
        """
        SENTINEL — only tools in allowed_tools list must be returned.
        Remove filter → agents receive every tool on server → test fails.
        """
        svc = _make_service()
        tool_a = _make_tool("SENTINEL_ALLOWED")
        tool_b = _make_tool("tool-b-not-allowed")
        mcp_cfg = _make_mcp_config(server_id="srv", tools=["SENTINEL_ALLOWED"])

        with patch("app.services.agent_mcp_service._HAS_MCP_ADAPTERS", True), \
             patch.object(svc, "_get_server_tools", new=AsyncMock(return_value=[tool_a, tool_b])):
            result = await svc.get_tools_for_agent([mcp_cfg])

        tool_names = [t.name for t in result]
        assert "SENTINEL_ALLOWED" in tool_names
        assert "tool-b-not-allowed" not in tool_names

    @pytest.mark.asyncio
    async def test_empty_allowed_tools_includes_all(self):
        """
        SENTINEL — allowed_tools=[] means include all tools from the server.
        Filter to empty list → agents always get zero tools → test fails.
        """
        svc = _make_service()
        tools = [_make_tool("tool-x"), _make_tool("tool-y")]
        mcp_cfg = _make_mcp_config(server_id="srv", tools=[])

        with patch("app.services.agent_mcp_service._HAS_MCP_ADAPTERS", True), \
             patch.object(svc, "_get_server_tools", new=AsyncMock(return_value=tools)):
            result = await svc.get_tools_for_agent([mcp_cfg])

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_server_exception_isolated_other_servers_continue(self):
        """
        SENTINEL — if one server raises, tools from other servers must still be returned.
        Re-raise exception → one bad MCP server kills all tool binding → test fails.
        """
        svc = _make_service()
        good_tool = _make_tool("SENTINEL_GOOD_TOOL")

        async def _fake_get(server_id, config):
            if server_id == "bad-server":
                raise RuntimeError("server unreachable")
            return [good_tool]

        with patch("app.services.agent_mcp_service._HAS_MCP_ADAPTERS", True), \
             patch.object(svc, "_get_server_tools", side_effect=_fake_get):
            result = await svc.get_tools_for_agent([
                _make_mcp_config(server_id="bad-server"),
                _make_mcp_config(server_id="good-server"),
            ])

        tool_names = [t.name for t in result]
        assert "SENTINEL_GOOD_TOOL" in tool_names

    @pytest.mark.asyncio
    async def test_multiple_servers_tools_combined(self):
        """Tools from multiple servers must all appear in the result."""
        svc = _make_service()
        tools_a = [_make_tool("tool-server-a")]
        tools_b = [_make_tool("tool-server-b")]

        async def _fake_get(server_id, config):
            return tools_a if server_id == "srv-a" else tools_b

        with patch("app.services.agent_mcp_service._HAS_MCP_ADAPTERS", True), \
             patch.object(svc, "_get_server_tools", side_effect=_fake_get):
            result = await svc.get_tools_for_agent([
                _make_mcp_config(server_id="srv-a"),
                _make_mcp_config(server_id="srv-b"),
            ])

        tool_names = [t.name for t in result]
        assert "tool-server-a" in tool_names
        assert "tool-server-b" in tool_names


# ===========================================================================
# list_available_servers — static list
# ===========================================================================

class TestListAvailableServers:
    @pytest.mark.asyncio
    async def test_returns_known_servers(self):
        """list_available_servers must include mcp-obsidian, memory, and filesystem."""
        svc = _make_service()
        servers = await svc.list_available_servers()
        server_ids = [s["server_id"] for s in servers]
        assert "mcp-obsidian" in server_ids
        assert "memory" in server_ids
        assert "filesystem" in server_ids

    @pytest.mark.asyncio
    async def test_each_server_has_required_fields(self):
        """Each server entry must have server_id, name, description, transport."""
        svc = _make_service()
        servers = await svc.list_available_servers()
        for s in servers:
            assert "server_id" in s
            assert "name" in s
            assert "description" in s
            assert "transport" in s


# ===========================================================================
# get_agent_mcp_service — singleton
# ===========================================================================

class TestGetAgentMcpService:
    def test_returns_same_instance_on_second_call(self):
        """
        SENTINEL — get_agent_mcp_service must be a singleton.
        Create new instance each call → tool cache never reused → test fails.
        """
        import app.services.agent_mcp_service as mod
        mod._agent_mcp_service = None

        s1 = get_agent_mcp_service()
        s2 = get_agent_mcp_service()
        assert s1 is s2
