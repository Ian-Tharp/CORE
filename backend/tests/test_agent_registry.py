"""
Tests for the Agent Registry Service.

Covers: registration, heartbeat, task lifecycle, stale detection,
deregistration, and helper queries.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.services.agent_registry import (
    AgentConfig,
    AgentDeregistration,
    AgentHeartbeatData,
    AgentRegistrationPayload,
    AgentRegistry,
    TaskAssignment,
    TaskCompletion,
    TaskRefusal,
)
from app.repository.instance_repository import AgentInstance, InstanceStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_instance(**overrides) -> AgentInstance:
    defaults = dict(
        id=uuid4(),
        container_id="abc123",
        agent_id="",
        agent_role="researcher",
        status=InstanceStatus.STARTING,
        capabilities=[],
        last_heartbeat=None,
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return AgentInstance(**defaults)


def _make_registration(**overrides) -> AgentRegistrationPayload:
    defaults = dict(
        container_id="abc123",
        role="researcher",
        capabilities=["web_search"],
        version="1.0.0",
    )
    defaults.update(overrides)
    return AgentRegistrationPayload(**defaults)


def _make_heartbeat(**overrides) -> AgentHeartbeatData:
    defaults = dict(status="ready", current_task=None, resource_usage={})
    defaults.update(overrides)
    return AgentHeartbeatData(**defaults)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    """Agent registration lifecycle."""

    @pytest.fixture
    def registry(self):
        return AgentRegistry()

    @pytest.fixture(autouse=True)
    def _patch_repo(self):
        with patch(
            "app.services.agent_registry.get_instance_by_container_id",
            new_callable=AsyncMock,
        ) as get, patch(
            "app.services.agent_registry.update_instance", new_callable=AsyncMock
        ) as upd:
            self.mock_get_instance = get
            self.mock_update_instance = upd
            yield

    @pytest.mark.asyncio
    async def test_register_returns_config(self, registry):
        inst = _make_instance()
        self.mock_get_instance.return_value = inst

        config = await registry.register_agent(_make_registration())

        assert isinstance(config, AgentConfig)
        assert config.model == "ollama/llama3.2"
        assert "web_search" in config.tools

    @pytest.mark.asyncio
    async def test_register_sets_agent_id(self, registry):
        inst = _make_instance(agent_id="")
        self.mock_get_instance.return_value = inst

        config = await registry.register_agent(_make_registration(role="researcher"))

        assert config.agent_id.startswith("researcher-")
        assert config.agent_id in registry.active_agents

    @pytest.mark.asyncio
    async def test_register_preserves_existing_agent_id(self, registry):
        inst = _make_instance(agent_id="custom-id-42")
        self.mock_get_instance.return_value = inst

        config = await registry.register_agent(_make_registration())

        assert config.agent_id == "custom-id-42"

    @pytest.mark.asyncio
    async def test_register_unknown_container_raises(self, registry):
        self.mock_get_instance.return_value = None

        with pytest.raises(ValueError, match="not found"):
            await registry.register_agent(
                _make_registration(container_id="nonexistent")
            )

    @pytest.mark.asyncio
    async def test_register_updates_database(self, registry):
        inst = _make_instance()
        self.mock_get_instance.return_value = inst

        await registry.register_agent(_make_registration())

        self.mock_update_instance.assert_awaited_once()
        call_args = self.mock_update_instance.call_args
        assert call_args[0][1]["status"] == InstanceStatus.READY.value


# ---------------------------------------------------------------------------
# Tools by role
# ---------------------------------------------------------------------------


class TestToolsForRole:
    def test_known_roles_return_specific_tools(self):
        reg = AgentRegistry()
        assert "web_search" in reg._get_tools_for_role("researcher")
        assert "file_write" in reg._get_tools_for_role("writer")
        assert "image_analysis" in reg._get_tools_for_role("analyst")
        assert "message" in reg._get_tools_for_role("coordinator")

    def test_unknown_role_returns_defaults(self):
        reg = AgentRegistry()
        tools = reg._get_tools_for_role("unknown_role")
        assert "web_search" in tools
        assert "file_read" in tools


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


class TestHeartbeat:
    @pytest.fixture
    def registry(self):
        reg = AgentRegistry()
        reg.active_agents["agent-1"] = {
            "container_id": "abc123",
            "instance_id": uuid4(),
            "role": "researcher",
            "capabilities": [],
            "version": "1.0.0",
            "registered_at": datetime.now(timezone.utc),
            "last_heartbeat": datetime.now(timezone.utc) - timedelta(seconds=30),
            "current_status": "ready",
            "current_task": None,
        }
        return reg

    @pytest.fixture(autouse=True)
    def _patch_repo(self):
        with patch(
            "app.services.agent_registry.update_heartbeat", new_callable=AsyncMock
        ) as hb:
            self.mock_update_heartbeat = hb
            yield

    @pytest.mark.asyncio
    async def test_heartbeat_ack(self, registry):
        resp = await registry.handle_heartbeat("agent-1", _make_heartbeat())
        assert resp["type"] == "heartbeat_ack"

    @pytest.mark.asyncio
    async def test_heartbeat_updates_timestamp(self, registry):
        before = registry.active_agents["agent-1"]["last_heartbeat"]
        await registry.handle_heartbeat("agent-1", _make_heartbeat())
        after = registry.active_agents["agent-1"]["last_heartbeat"]
        assert after > before

    @pytest.mark.asyncio
    async def test_heartbeat_unknown_agent_raises(self, registry):
        with pytest.raises(ValueError, match="not registered"):
            await registry.handle_heartbeat("ghost", _make_heartbeat())

    @pytest.mark.asyncio
    async def test_heartbeat_delivers_pending_task(self, registry):
        task = TaskAssignment(task_id="t1", task_type="research", payload={"q": "test"})
        registry.pending_tasks["agent-1"] = [task]

        resp = await registry.handle_heartbeat("agent-1", _make_heartbeat())

        assert resp["type"] == "task_assigned"
        assert resp["task_id"] == "t1"
        assert registry.pending_tasks["agent-1"] == []

    @pytest.mark.asyncio
    async def test_heartbeat_updates_status(self, registry):
        await registry.handle_heartbeat(
            "agent-1", _make_heartbeat(status="busy", current_task="t99")
        )
        assert registry.active_agents["agent-1"]["current_status"] == "busy"
        assert registry.active_agents["agent-1"]["current_task"] == "t99"


# ---------------------------------------------------------------------------
# Task completion & refusal
# ---------------------------------------------------------------------------


class TestTaskLifecycle:
    @pytest.fixture
    def registry(self):
        reg = AgentRegistry()
        reg.active_agents["agent-1"] = {
            "container_id": "abc123",
            "instance_id": uuid4(),
            "role": "researcher",
            "current_task": "t1",
        }
        return reg

    @pytest.fixture(autouse=True)
    def _patch_repo(self):
        with patch(
            "app.services.agent_registry.increment_task_completed",
            new_callable=AsyncMock,
        ) as comp, patch(
            "app.services.agent_registry.increment_task_refused", new_callable=AsyncMock
        ) as ref:
            self.mock_completed = comp
            self.mock_refused = ref
            yield

    @pytest.mark.asyncio
    async def test_task_completion_clears_current_task(self, registry):
        completion = TaskCompletion(
            task_id="t1", result={"answer": "42"}, duration_ms=500
        )
        await registry.handle_task_completion("agent-1", completion)
        assert registry.active_agents["agent-1"]["current_task"] is None

    @pytest.mark.asyncio
    async def test_task_completion_updates_metrics(self, registry):
        completion = TaskCompletion(task_id="t1", result={}, duration_ms=1234)
        await registry.handle_task_completion("agent-1", completion)
        self.mock_completed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_task_completion_unknown_agent(self, registry):
        with pytest.raises(ValueError):
            await registry.handle_task_completion(
                "ghost", TaskCompletion(task_id="t1", result={}, duration_ms=0)
            )

    @pytest.mark.asyncio
    async def test_task_refusal_records_metrics(self, registry):
        refusal = TaskRefusal(task_id="t1", reason="out of scope")
        await registry.handle_task_refusal("agent-1", refusal)
        self.mock_refused.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_task_refusal_unknown_agent(self, registry):
        with pytest.raises(ValueError):
            await registry.handle_task_refusal(
                "ghost", TaskRefusal(task_id="t1", reason="no")
            )


# ---------------------------------------------------------------------------
# Task assignment
# ---------------------------------------------------------------------------


class TestTaskAssignment:
    @pytest.fixture
    def registry(self):
        reg = AgentRegistry()
        reg.active_agents["agent-1"] = {"role": "researcher"}
        return reg

    @pytest.mark.asyncio
    async def test_assign_task_queues(self, registry):
        task = TaskAssignment(task_id="t1", task_type="research", payload={})
        result = await registry.assign_task("agent-1", task)
        assert result is True
        assert len(registry.pending_tasks["agent-1"]) == 1

    @pytest.mark.asyncio
    async def test_assign_task_unknown_agent(self, registry):
        task = TaskAssignment(task_id="t1", task_type="research", payload={})
        result = await registry.assign_task("ghost", task)
        assert result is False

    @pytest.mark.asyncio
    async def test_assign_multiple_tasks(self, registry):
        for i in range(3):
            await registry.assign_task(
                "agent-1", TaskAssignment(task_id=f"t{i}", task_type="x", payload={})
            )
        assert len(registry.pending_tasks["agent-1"]) == 3


# ---------------------------------------------------------------------------
# Deregistration
# ---------------------------------------------------------------------------


class TestDeregistration:
    @pytest.fixture
    def registry(self):
        reg = AgentRegistry()
        reg.active_agents["agent-1"] = {
            "container_id": "abc123",
            "instance_id": uuid4(),
        }
        reg.pending_tasks["agent-1"] = [
            TaskAssignment(task_id="t1", task_type="x", payload={})
        ]
        return reg

    @pytest.fixture(autouse=True)
    def _patch_repo(self):
        with patch(
            "app.services.agent_registry.update_instance_status", new_callable=AsyncMock
        ):
            yield

    @pytest.mark.asyncio
    async def test_deregister_removes_agent(self, registry):
        dereg = AgentDeregistration(reason="shutdown")
        await registry.deregister_agent("agent-1", dereg)
        assert "agent-1" not in registry.active_agents

    @pytest.mark.asyncio
    async def test_deregister_cleans_pending_tasks(self, registry):
        dereg = AgentDeregistration(reason="shutdown")
        await registry.deregister_agent("agent-1", dereg)
        assert "agent-1" not in registry.pending_tasks

    @pytest.mark.asyncio
    async def test_deregister_unknown_agent_is_noop(self, registry):
        dereg = AgentDeregistration(reason="shutdown")
        await registry.deregister_agent("ghost", dereg)  # should not raise


# ---------------------------------------------------------------------------
# Stale agent detection
# ---------------------------------------------------------------------------


class TestStaleAgentDetection:
    @pytest.fixture
    def registry(self):
        return AgentRegistry()

    @pytest.fixture(autouse=True)
    def _patch_repo(self):
        with patch(
            "app.services.agent_registry.update_instance_status", new_callable=AsyncMock
        ), patch(
            "app.services.agent_registry.get_instance_by_container_id",
            new_callable=AsyncMock,
        ) as get, patch(
            "app.services.agent_registry.instance_manager"
        ) as im:
            self.mock_get = get
            self.mock_instance_manager = im
            im.restart_instance = AsyncMock()
            yield

    @pytest.mark.asyncio
    async def test_healthy_agent_not_flagged(self, registry):
        registry.active_agents["a1"] = {
            "last_heartbeat": datetime.now(timezone.utc),
            "container_id": "c1",
            "current_status": "ready",
        }
        stale = await registry.check_stale_agents()
        assert stale == []
        assert "a1" in registry.active_agents

    @pytest.mark.asyncio
    async def test_stale_agent_marked_unhealthy(self, registry):
        self.mock_get.return_value = _make_instance(status=InstanceStatus.READY)
        registry.active_agents["a1"] = {
            "last_heartbeat": datetime.now(timezone.utc) - timedelta(seconds=100),
            "container_id": "c1",
            "current_status": "ready",
        }
        stale = await registry.check_stale_agents()
        assert "a1" in stale
        assert registry.active_agents["a1"]["current_status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_lost_agent_removed(self, registry):
        registry.active_agents["a1"] = {
            "last_heartbeat": datetime.now(timezone.utc) - timedelta(minutes=6),
            "container_id": "c1",
            "current_status": "ready",
        }
        stale = await registry.check_stale_agents()
        assert "a1" in stale
        assert "a1" not in registry.active_agents

    @pytest.mark.asyncio
    async def test_lost_agent_triggers_restart(self, registry):
        registry.active_agents["a1"] = {
            "last_heartbeat": datetime.now(timezone.utc) - timedelta(minutes=6),
            "container_id": "c1",
            "current_status": "ready",
        }
        await registry.check_stale_agents()
        self.mock_instance_manager.restart_instance.assert_awaited_once_with("c1")

    @pytest.mark.asyncio
    async def test_shutdown_flag_skips_check(self, registry):
        registry._shutdown = True
        registry.active_agents["a1"] = {
            "last_heartbeat": datetime.now(timezone.utc) - timedelta(minutes=10),
            "container_id": "c1",
        }
        stale = await registry.check_stale_agents()
        assert stale == []


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


class TestQueryHelpers:
    @pytest.fixture
    def registry(self):
        reg = AgentRegistry()
        reg.active_agents = {
            "r1": {
                "role": "researcher",
                "last_heartbeat": datetime.now(timezone.utc),
                "current_status": "ready",
            },
            "r2": {
                "role": "researcher",
                "last_heartbeat": datetime.now(timezone.utc) - timedelta(minutes=5),
                "current_status": "ready",
            },
            "w1": {
                "role": "writer",
                "last_heartbeat": datetime.now(timezone.utc),
                "current_status": "busy",
            },
        }
        return reg

    def test_list_active_agents(self, registry):
        assert set(registry.list_active_agents()) == {"r1", "r2", "w1"}

    def test_get_agents_by_role(self, registry):
        assert set(registry.get_agents_by_role("researcher")) == {"r1", "r2"}
        assert registry.get_agents_by_role("writer") == ["w1"]
        assert registry.get_agents_by_role("analyst") == []

    def test_get_healthy_agents(self, registry):
        healthy = registry.get_healthy_agents()
        assert "r1" in healthy
        assert "w1" in healthy
        assert "r2" not in healthy  # heartbeat too old

    def test_get_agent_info(self, registry):
        info = registry.get_agent_info("r1")
        assert info["role"] == "researcher"
        assert registry.get_agent_info("ghost") is None


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_sets_flag(self):
        reg = AgentRegistry()
        await reg.shutdown()
        assert reg._shutdown is True
