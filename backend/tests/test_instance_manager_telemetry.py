"""
Tests for agent instance lifecycle -> cognition telemetry emission.

Spawning, stopping, and restarting an agent instance publish WebSocket
agent_activity events so the command deck's activity stream and reactor reflect
real agent operations. These tests mock Docker + the repository and assert the
emits, isolated from real containers/DB.
"""

from unittest.mock import AsyncMock, MagicMock

import app.services.instance_manager as im


def _manager_with_container(container: MagicMock) -> im.InstanceManager:
    mgr = im.InstanceManager()
    mgr.docker_client = MagicMock()
    mgr.docker_client.containers.get.return_value = container
    mgr.docker_client.containers.run.return_value = container
    return mgr


class TestSafeEmit:
    async def test_swallows_errors(self):
        async def boom():
            raise RuntimeError("publish failed")

        await im._safe_emit(lambda: boom())  # must not raise

    async def test_awaits_factory(self):
        seen = {}

        async def coro():
            seen["ok"] = True

        await im._safe_emit(lambda: coro())
        assert seen.get("ok") is True


class TestSpawnTelemetry:
    async def test_spawn_emits_agent_online(self, monkeypatch):
        pub = AsyncMock()
        monkeypatch.setattr(im, "event_publisher", pub)
        monkeypatch.setattr(im, "create_instance", AsyncMock())
        monkeypatch.setattr(im, "update_instance_status", AsyncMock())

        container = MagicMock(
            id="container123abcdef", labels={"core.agent_id": "agent_x"}
        )
        mgr = _manager_with_container(container)
        monkeypatch.setattr(mgr, "_wait_for_healthy", AsyncMock())

        info = await mgr.spawn_instance(
            im.InstanceConfig(agent_id="agent_x", agent_role="researcher")
        )

        assert info.status == "ready"
        pub.agent_started.assert_called_once()
        kwargs = pub.agent_started.call_args.kwargs
        assert kwargs["agent_id"] == "agent_x"
        assert kwargs["action"] == "deployed"
        assert "researcher" in kwargs["message"]


class TestStopTelemetry:
    async def test_stop_emits_agent_complete(self, monkeypatch):
        pub = AsyncMock()
        monkeypatch.setattr(im, "event_publisher", pub)
        monkeypatch.setattr(im, "update_instance_status", AsyncMock())

        container = MagicMock(labels={"core.agent_id": "agent_y"})
        mgr = _manager_with_container(container)

        ok = await mgr.stop_instance("cid999")

        assert ok is True
        pub.agent_complete.assert_called_once()
        assert pub.agent_complete.call_args.kwargs["agent_id"] == "agent_y"
        assert pub.agent_complete.call_args.kwargs["action"] == "stopped"


class TestRestartTelemetry:
    async def test_restart_success_emits_agent_started(self, monkeypatch):
        pub = AsyncMock()
        monkeypatch.setattr(im, "event_publisher", pub)
        monkeypatch.setattr(im, "update_instance_status", AsyncMock())

        container = MagicMock(labels={"core.agent_id": "agent_z"})
        mgr = _manager_with_container(container)
        monkeypatch.setattr(mgr, "_wait_for_healthy", AsyncMock())

        ok = await mgr.restart_instance("cid-z")

        assert ok is True
        pub.agent_started.assert_called_once()
        assert pub.agent_started.call_args.kwargs["agent_id"] == "agent_z"
        assert pub.agent_started.call_args.kwargs["action"] == "restarted"

    async def test_restart_failure_emits_agent_error(self, monkeypatch):
        pub = AsyncMock()
        monkeypatch.setattr(im, "event_publisher", pub)
        monkeypatch.setattr(im, "update_instance_status", AsyncMock())

        container = MagicMock(labels={})
        container.restart.side_effect = RuntimeError("docker boom")
        mgr = _manager_with_container(container)

        ok = await mgr.restart_instance("cid-fail")

        assert ok is False
        pub.agent_error.assert_called_once()
        assert pub.agent_error.call_args.kwargs["action"] == "restart"
