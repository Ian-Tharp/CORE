"""
Tests for MCP server health checks in CORE.

Sentinel-value contract:
- Remove check_mcp_servers() → function missing → import error → test fails
- Remove command reachability check → missing binaries reported healthy → test fails
- Remove DEGRADED status for missing commands → broken status → test fails
- Remove check_mcp_servers from get_full_health() → MCP not checked → test fails
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.health import (
    HealthStatus,
    HealthCheck,
    check_mcp_servers,
    _load_mcp_config,
    get_full_health,
)


# ---------------------------------------------------------------------------
# _load_mcp_config
# ---------------------------------------------------------------------------


class TestLoadMcpConfig:
    def test_returns_empty_dict_when_no_file(self):
        """When no .mcp.json exists anywhere, returns empty mcpServers dict."""
        # Patch Path.is_file so no candidate is found
        with patch.dict(os.environ, {}, clear=False), patch(
            "app.core.health.Path"
        ) as mock_path_cls:
            # Make every Path(...).is_file() return False
            instance = MagicMock()
            instance.is_file.return_value = False
            mock_path_cls.return_value = instance
            mock_path_cls.cwd.return_value = instance
            # Also handle Path(__file__).resolve().parents[3] / ".mcp.json"
            instance.__truediv__ = lambda self, other: instance
            instance.resolve.return_value = instance
            instance.parents = [instance, instance, instance, instance]

            config = _load_mcp_config()
        assert config == {"mcpServers": {}}

    def test_loads_valid_json(self, tmp_path):
        """SENTINEL — valid .mcp.json must be parsed and returned."""
        cfg_data = {"mcpServers": {"test-server": {"command": "echo", "type": "stdio"}}}
        cfg_file = tmp_path / ".mcp.json"
        cfg_file.write_text(json.dumps(cfg_data))
        with patch.dict(os.environ, {"CORE_MCP_CONFIG": str(cfg_file)}):
            config = _load_mcp_config()
        assert "test-server" in config["mcpServers"]

    def test_invalid_json_falls_through_to_empty(self, tmp_path):
        """Corrupted JSON in CORE_MCP_CONFIG must not raise; falls through to empty when no other candidates."""
        cfg_file = tmp_path / ".mcp.json"
        cfg_file.write_text("{invalid json}")

        # Provide env var pointing to the bad file, but stub out all other Path candidates
        with patch.dict(os.environ, {"CORE_MCP_CONFIG": str(cfg_file)}):
            # Make Path.is_file() return False for all non-env-var paths to block fallback
            original_is_file = Path.is_file

            def _patched_is_file(self):
                # Allow CORE_MCP_CONFIG path to work normally (it exists — is_file True)
                # But block the two fallback candidates
                if str(self) == str(cfg_file):
                    return True  # Let it be found; parse will fail and we fall through
                return False

            with patch.object(Path, "is_file", _patched_is_file):
                config = _load_mcp_config()
        assert config == {"mcpServers": {}}


# ---------------------------------------------------------------------------
# check_mcp_servers
# ---------------------------------------------------------------------------


class TestCheckMcpServers:
    @pytest.mark.asyncio
    async def test_no_servers_returns_healthy(self, tmp_path):
        """
        SENTINEL — empty MCP config must return HEALTHY.
        Return DEGRADED for empty config → no-config environments break → test fails.
        """
        cfg_file = tmp_path / ".mcp.json"
        cfg_file.write_text(json.dumps({"mcpServers": {}}))
        with patch.dict(os.environ, {"CORE_MCP_CONFIG": str(cfg_file)}):
            result = await check_mcp_servers()
        assert result.status == HealthStatus.HEALTHY
        assert result.details["server_count"] == 0

    @pytest.mark.asyncio
    async def test_all_reachable_returns_healthy(self, tmp_path):
        """
        SENTINEL — servers with accessible commands must return HEALTHY.
        Flip to DEGRADED when all found → always-degraded → test fails.
        """
        cfg_data = {
            "mcpServers": {
                "echo-server": {"command": "python", "type": "stdio"},
            }
        }
        cfg_file = tmp_path / ".mcp.json"
        cfg_file.write_text(json.dumps(cfg_data))

        import shutil as _shutil

        with patch.dict(os.environ, {"CORE_MCP_CONFIG": str(cfg_file)}), patch(
            "app.core.health.shutil.which", return_value="/usr/bin/python"
        ):
            result = await check_mcp_servers()

        assert result.status == HealthStatus.HEALTHY
        assert result.details["servers"]["echo-server"] == "reachable"

    @pytest.mark.asyncio
    async def test_missing_command_returns_degraded(self, tmp_path):
        """
        SENTINEL — servers whose command is not on PATH must return DEGRADED.
        Return HEALTHY for missing commands → broken env unreported → test fails.
        """
        cfg_data = {
            "mcpServers": {
                "ghost-server": {"command": "nonexistent-binary-xyz", "type": "stdio"},
            }
        }
        cfg_file = tmp_path / ".mcp.json"
        cfg_file.write_text(json.dumps(cfg_data))

        with patch.dict(os.environ, {"CORE_MCP_CONFIG": str(cfg_file)}), patch(
            "app.core.health.shutil.which", return_value=None
        ):
            result = await check_mcp_servers()

        assert result.status == HealthStatus.DEGRADED
        assert "ghost-server" in result.details["servers"]
        assert result.details["servers"]["ghost-server"] == "command_not_found"
        assert result.details["missing_count"] == 1

    @pytest.mark.asyncio
    async def test_partial_reachability_returns_degraded(self, tmp_path):
        """
        SENTINEL — when some servers are reachable and some are not, status is DEGRADED.
        Always return HEALTHY → partial outage unreported → test fails.
        """
        cfg_data = {
            "mcpServers": {
                "ok-server": {"command": "python", "type": "stdio"},
                "bad-server": {"command": "missing-cmd", "type": "stdio"},
            }
        }
        cfg_file = tmp_path / ".mcp.json"
        cfg_file.write_text(json.dumps(cfg_data))

        def _which(cmd):
            return "/usr/bin/python" if cmd == "python" else None

        with patch.dict(os.environ, {"CORE_MCP_CONFIG": str(cfg_file)}), patch(
            "app.core.health.shutil.which", side_effect=_which
        ):
            result = await check_mcp_servers()

        assert result.status == HealthStatus.DEGRADED
        assert result.details["servers"]["ok-server"] == "reachable"
        assert result.details["servers"]["bad-server"] == "command_not_found"

    @pytest.mark.asyncio
    async def test_absolute_path_found_returns_reachable(self, tmp_path):
        """
        SENTINEL — absolute path that exists on disk must be reported as reachable.
        Ignore absolute paths → executables in known locations always missing → test fails.
        """
        # Create a dummy "binary"
        fake_binary = tmp_path / "my-server.exe"
        fake_binary.write_text("#!/bin/sh\necho ok")

        cfg_data = {
            "mcpServers": {
                "local-server": {"command": str(fake_binary), "type": "stdio"},
            }
        }
        cfg_file = tmp_path / ".mcp.json"
        cfg_file.write_text(json.dumps(cfg_data))

        with patch.dict(os.environ, {"CORE_MCP_CONFIG": str(cfg_file)}):
            result = await check_mcp_servers()

        assert result.status == HealthStatus.HEALTHY
        assert result.details["servers"]["local-server"] == "reachable"

    @pytest.mark.asyncio
    async def test_absolute_path_missing_returns_degraded(self, tmp_path):
        """Absolute path that does NOT exist must be reported as command_not_found."""
        cfg_data = {
            "mcpServers": {
                "ghost": {
                    "command": str(tmp_path / "ghost-binary.exe"),
                    "type": "stdio",
                },
            }
        }
        cfg_file = tmp_path / ".mcp.json"
        cfg_file.write_text(json.dumps(cfg_data))

        with patch.dict(os.environ, {"CORE_MCP_CONFIG": str(cfg_file)}):
            result = await check_mcp_servers()

        assert result.status == HealthStatus.DEGRADED
        assert result.details["servers"]["ghost"] == "command_not_found"

    @pytest.mark.asyncio
    async def test_result_is_health_check_instance(self, tmp_path):
        """check_mcp_servers must return a HealthCheck object."""
        cfg_file = tmp_path / ".mcp.json"
        cfg_file.write_text(json.dumps({"mcpServers": {}}))
        with patch.dict(os.environ, {"CORE_MCP_CONFIG": str(cfg_file)}):
            result = await check_mcp_servers()
        assert isinstance(result, HealthCheck)
        assert result.name == "mcp_servers"

    @pytest.mark.asyncio
    async def test_exception_returns_degraded_not_raises(self):
        """
        If an unexpected error occurs, check_mcp_servers must return DEGRADED, not raise.
        Let exceptions propagate → asyncio.gather captures as Exception → health breaks → test fails.
        """
        with patch(
            "app.core.health._load_mcp_config", side_effect=RuntimeError("disk error")
        ):
            result = await check_mcp_servers()
        assert result.status == HealthStatus.DEGRADED


# ---------------------------------------------------------------------------
# Integration: check_mcp_servers in get_full_health()
# ---------------------------------------------------------------------------


class TestMcpServersInFullHealth:
    def _ok_check(self, name: str):
        return AsyncMock(
            return_value=HealthCheck(name=name, status=HealthStatus.HEALTHY)
        )

    @pytest.mark.asyncio
    async def test_mcp_check_included_in_full_health(self):
        """
        SENTINEL — check_mcp_servers must be called by get_full_health().
        Remove it from gather() → spy not called → test fails.
        """
        from app.core import health as health_mod

        mcp_spy = AsyncMock(
            return_value=HealthCheck(name="mcp_servers", status=HealthStatus.HEALTHY)
        )

        with patch("app.core.health.check_database", self._ok_check("database")), patch(
            "app.core.health.check_ollama", self._ok_check("ollama")
        ), patch("app.core.health.check_redis", self._ok_check("redis")), patch(
            "app.core.health.check_websocket_manager", self._ok_check("ws")
        ), patch(
            "app.core.health.check_engine_state", self._ok_check("engine")
        ), patch(
            "app.core.health.check_mcp_servers", mcp_spy
        ), patch(
            "app.core.health.asyncio.create_task"
        ):

            await health_mod.get_full_health()

        assert mcp_spy.called, "check_mcp_servers was not called by get_full_health()"

    @pytest.mark.asyncio
    async def test_degraded_mcp_makes_full_health_degraded(self):
        """
        SENTINEL — DEGRADED MCP check must degrade the overall health status.
        Ignore MCP status → degraded MCP silently dropped → test fails.
        """
        from app.core import health as health_mod

        with patch("app.core.health.check_database", self._ok_check("database")), patch(
            "app.core.health.check_ollama", self._ok_check("ollama")
        ), patch("app.core.health.check_redis", self._ok_check("redis")), patch(
            "app.core.health.check_websocket_manager", self._ok_check("ws")
        ), patch(
            "app.core.health.check_engine_state", self._ok_check("engine")
        ), patch(
            "app.core.health.check_mcp_servers",
            AsyncMock(
                return_value=HealthCheck(
                    name="mcp_servers",
                    status=HealthStatus.DEGRADED,
                    message="1 server missing",
                )
            ),
        ), patch(
            "app.core.health.asyncio.create_task"
        ):

            result = await health_mod.get_full_health()

        assert result["status"] in (
            "degraded",
            "unhealthy",
        ), f"Expected degraded overall status, got {result['status']!r}"
