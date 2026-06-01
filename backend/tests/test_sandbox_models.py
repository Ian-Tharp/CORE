"""
Tests for sandbox pure models and helpers:
  - state_manager.py  (StateScope, AgentState, EventType, StateManager._make_state_key, get_status)
  - container_manager.py  (ContainerStatus, ContainerConfig, ContainerManager.get_status,
                            ContainerManager._build_docker_config)
  - security.py  (TrustLevel, AgentSecurityConfig defaults, TRUST_PRESETS, get_security_config)

Sentinel-value contract:
- Remove StateScope.GLOBAL branch in _make_state_key → global keys include agent_id → routing breaks → test fails
- Remove StateScope.SESSION branch → session keys include agent_id not task_id → session isolation broken → test fails
- Remove StateScope.TASK branch → task keys lose agent prefix → multi-agent collisions → test fails
- Remove get_status state_ttl_minutes key → monitoring dashboard missing TTL → test fails
- Remove ContainerManager.get_status pool_size_limit key → autoscaler has no reference size → test fails
- Remove TrustLevel.UNTRUSTED → security_opt never added → untrusted agents run without sandbox → test fails
"""

from __future__ import annotations

import pytest

from app.sandbox.state_manager import (
    AgentState,
    EventType,
    StateManager,
    StateScope,
)
from app.sandbox.container_manager import (
    ContainerConfig,
    ContainerManager,
    ContainerStatus,
)
from app.sandbox.security import (
    AgentSecurityConfig,
    TRUST_PRESETS,
    TrustLevel,
    get_security_config,
)


# ===========================================================================
# StateScope enum
# ===========================================================================


class TestStateScope:
    def test_task_value(self):
        """
        SENTINEL — TASK must equal 'task'.
        Change → _make_state_key uses wrong prefix → key collisions → test fails.
        """
        assert StateScope.TASK == "task"

    def test_session_value(self):
        """SESSION must equal 'session'."""
        assert StateScope.SESSION == "session"

    def test_agent_value(self):
        """AGENT must equal 'agent'."""
        assert StateScope.AGENT == "agent"

    def test_global_value(self):
        """
        SENTINEL — GLOBAL must equal 'global'.
        Change → global state keys mis-routed → all agents lose shared context → test fails.
        """
        assert StateScope.GLOBAL == "global"

    def test_exactly_four_scopes(self):
        """Exactly four StateScope values must be defined."""
        assert len(StateScope) == 4


# ===========================================================================
# AgentState defaults
# ===========================================================================


class TestAgentState:
    def test_default_scope_is_task(self):
        """
        SENTINEL — default scope must be TASK.
        Default GLOBAL → all unscoped state bleeds across agents → test fails.
        """
        state = AgentState(agent_id="SENTINEL_AGENT", task_id="SENTINEL_TASK")
        assert state.scope == StateScope.TASK.value  # use_enum_values=True

    def test_data_defaults_empty(self):
        """data defaults to empty dict."""
        state = AgentState(agent_id="a", task_id="t")
        assert state.data == {}

    def test_metadata_defaults_empty(self):
        """metadata defaults to empty dict."""
        state = AgentState(agent_id="a", task_id="t")
        assert state.metadata == {}

    def test_expires_at_defaults_none(self):
        """expires_at defaults to None (no expiry unless set)."""
        state = AgentState(agent_id="a", task_id="t")
        assert state.expires_at is None

    def test_id_auto_generated(self):
        """id must be auto-generated (unique per instance)."""
        s1 = AgentState(agent_id="a", task_id="t")
        s2 = AgentState(agent_id="a", task_id="t")
        assert s1.id != s2.id


# ===========================================================================
# EventType enum
# ===========================================================================


class TestEventType:
    def test_state_updated_value(self):
        """
        SENTINEL — STATE_UPDATED must equal 'state_updated'.
        Change → event consumers can't filter state changes → coordination lost → test fails.
        """
        assert EventType.STATE_UPDATED == "state_updated"

    def test_task_started_value(self):
        """TASK_STARTED must equal 'task_started'."""
        assert EventType.TASK_STARTED == "task_started"

    def test_task_completed_value(self):
        """TASK_COMPLETED must equal 'task_completed'."""
        assert EventType.TASK_COMPLETED == "task_completed"

    def test_task_failed_value(self):
        """TASK_FAILED must equal 'task_failed'."""
        assert EventType.TASK_FAILED == "task_failed"

    def test_agent_ready_value(self):
        """AGENT_READY must equal 'agent_ready'."""
        assert EventType.AGENT_READY == "agent_ready"

    def test_agent_busy_value(self):
        """AGENT_BUSY must equal 'agent_busy'."""
        assert EventType.AGENT_BUSY == "agent_busy"

    def test_artifact_created_value(self):
        """ARTIFACT_CREATED must equal 'artifact_created'."""
        assert EventType.ARTIFACT_CREATED == "artifact_created"

    def test_exactly_seven_event_types(self):
        """Exactly 7 EventType values must be defined."""
        assert len(EventType) == 7


# ===========================================================================
# StateManager._make_state_key — four scope branches
# ===========================================================================


class TestMakeStateKey:
    def setup_method(self):
        self.mgr = StateManager(data_dir="/tmp/sentinel_test_state")

    def test_global_scope_excludes_agent_and_task(self):
        """
        SENTINEL — GLOBAL key must NOT include agent_id or task_id.
        Remove GLOBAL branch → global keys gain agent_id prefix → different agents
        see different global state → cross-agent sharing breaks → test fails.
        """
        key = self.mgr._make_state_key(
            "SENTINEL_AGENT", "SENTINEL_TASK", StateScope.GLOBAL
        )
        assert "SENTINEL_AGENT" not in key
        assert "SENTINEL_TASK" not in key

    def test_global_scope_key_format(self):
        """
        SENTINEL — GLOBAL key must start with 'global:'.
        Change prefix → lookup by scope fails → test fails.
        """
        key = self.mgr._make_state_key("any", "any", StateScope.GLOBAL)
        assert key.startswith("global:")
        assert key.endswith(":global")

    def test_agent_scope_includes_agent_id(self):
        """
        SENTINEL — AGENT key must include agent_id.
        Remove agent_id → all agents share same agent-scope key → state trampled → test fails.
        """
        key = self.mgr._make_state_key(
            "SENTINEL_AGENT", "SENTINEL_TASK", StateScope.AGENT
        )
        assert "SENTINEL_AGENT" in key

    def test_agent_scope_excludes_task_id(self):
        """
        SENTINEL — AGENT key must NOT include task_id.
        Include task_id → each task gets different agent state → persistence broken → test fails.
        """
        key = self.mgr._make_state_key(
            "SENTINEL_AGENT", "SENTINEL_TASK", StateScope.AGENT
        )
        assert "SENTINEL_TASK" not in key

    def test_agent_scope_key_format(self):
        """AGENT key must start with 'agent:' and end with ':agent'."""
        key = self.mgr._make_state_key("SENTINEL_AGENT", "any", StateScope.AGENT)
        assert key.startswith("agent:")
        assert key.endswith(":agent")
        assert "SENTINEL_AGENT" in key

    def test_session_scope_includes_task_id(self):
        """
        SENTINEL — SESSION key must include task_id (session correlator).
        Use agent_id instead → cross-agent session isolation broken → test fails.
        """
        key = self.mgr._make_state_key(
            "SENTINEL_AGENT", "SENTINEL_TASK", StateScope.SESSION
        )
        assert "SENTINEL_TASK" in key

    def test_session_scope_excludes_agent_id(self):
        """
        SENTINEL — SESSION key must NOT include agent_id.
        Include agent_id → multiple agents in same session can't share → test fails.
        """
        key = self.mgr._make_state_key(
            "SENTINEL_AGENT", "SENTINEL_TASK", StateScope.SESSION
        )
        assert "SENTINEL_AGENT" not in key

    def test_session_scope_key_format(self):
        """SESSION key must start with 'session:' and end with ':session'."""
        key = self.mgr._make_state_key("any", "SENTINEL_TASK", StateScope.SESSION)
        assert key.startswith("session:")
        assert key.endswith(":session")

    def test_task_scope_includes_both_agent_and_task(self):
        """
        SENTINEL — TASK key must include BOTH agent_id and task_id.
        Drop agent_id → tasks from different agents collide → state corrupted → test fails.
        """
        key = self.mgr._make_state_key(
            "SENTINEL_AGENT", "SENTINEL_TASK", StateScope.TASK
        )
        assert "SENTINEL_AGENT" in key
        assert "SENTINEL_TASK" in key

    def test_task_scope_key_format(self):
        """TASK key must start with 'task:' and end with ':task'."""
        key = self.mgr._make_state_key(
            "SENTINEL_AGENT", "SENTINEL_TASK", StateScope.TASK
        )
        assert key.startswith("task:")
        assert key.endswith(":task")

    def test_all_four_keys_are_distinct(self):
        """All four scope keys for the same agent/task must be unique."""
        keys = [
            self.mgr._make_state_key("agent-1", "task-1", scope) for scope in StateScope
        ]
        assert len(set(keys)) == 4

    def test_different_agents_have_different_agent_scope_keys(self):
        """Two different agents must produce different AGENT-scope keys."""
        k1 = self.mgr._make_state_key("AGENT_A", "task", StateScope.AGENT)
        k2 = self.mgr._make_state_key("AGENT_B", "task", StateScope.AGENT)
        assert k1 != k2

    def test_different_tasks_have_different_session_scope_keys(self):
        """Two different task_ids must produce different SESSION-scope keys."""
        k1 = self.mgr._make_state_key("agent", "TASK_A", StateScope.SESSION)
        k2 = self.mgr._make_state_key("agent", "TASK_B", StateScope.SESSION)
        assert k1 != k2


# ===========================================================================
# StateManager.get_status
# ===========================================================================


class TestStateManagerGetStatus:
    def setup_method(self):
        self.mgr = StateManager(data_dir="/tmp/sentinel_state", state_ttl_minutes=45)

    def test_required_keys_present(self):
        """
        SENTINEL — get_status() must return all four required keys.
        Remove any key → monitoring dashboard breaks → test fails.
        """
        status = self.mgr.get_status()
        assert "state_entries" in status
        assert "subscribed_agents" in status
        assert "data_dir" in status
        assert "state_ttl_minutes" in status

    def test_state_entries_starts_zero(self):
        """Fresh manager has no state entries."""
        status = self.mgr.get_status()
        assert status["state_entries"] == 0

    def test_subscribed_agents_starts_zero(self):
        """Fresh manager has no event queue subscribers."""
        status = self.mgr.get_status()
        assert status["subscribed_agents"] == 0

    def test_state_ttl_minutes_matches_constructor(self):
        """
        SENTINEL — state_ttl_minutes must reflect the constructor argument.
        Hardcode a value → deployed TTL overrides are invisible to monitoring → test fails.
        """
        status = self.mgr.get_status()
        assert status["state_ttl_minutes"] == 45

    def test_data_dir_is_string(self):
        """data_dir must be a string (Path serialised)."""
        status = self.mgr.get_status()
        assert isinstance(status["data_dir"], str)


# ===========================================================================
# ContainerStatus enum
# ===========================================================================


class TestContainerStatus:
    def test_creating_value(self):
        """CREATING must equal 'creating'."""
        assert ContainerStatus.CREATING == "creating"

    def test_running_value(self):
        """RUNNING must equal 'running'."""
        assert ContainerStatus.RUNNING == "running"

    def test_idle_value(self):
        """IDLE must equal 'idle'."""
        assert ContainerStatus.IDLE == "idle"

    def test_stopped_value(self):
        """STOPPED must equal 'stopped'."""
        assert ContainerStatus.STOPPED == "stopped"

    def test_error_value(self):
        """
        SENTINEL — ERROR must equal 'error'.
        Change → error-status containers not cleaned up by cleanup loop → test fails.
        """
        assert ContainerStatus.ERROR == "error"

    def test_exactly_five_statuses(self):
        """Exactly 5 ContainerStatus values must be defined."""
        assert len(ContainerStatus) == 5


# ===========================================================================
# ContainerConfig defaults
# ===========================================================================


class TestContainerConfig:
    def test_default_image(self):
        """
        SENTINEL — default image must be 'python:3.12-slim'.
        Change to 3.11 → container agent environment diverges from tests → test fails.
        """
        cfg = ContainerConfig()
        assert cfg.image == "python:3.12-slim"

    def test_default_working_dir(self):
        """
        SENTINEL — working_dir must default to '/workspace'.
        Change → code executed in wrong directory → relative paths break → test fails.
        """
        cfg = ContainerConfig()
        assert cfg.working_dir == "/workspace"

    def test_environment_defaults_empty(self):
        """environment defaults to empty dict."""
        cfg = ContainerConfig()
        assert cfg.environment == {}

    def test_volumes_defaults_empty(self):
        """volumes defaults to empty dict."""
        cfg = ContainerConfig()
        assert cfg.volumes == {}

    def test_mcp_capabilities_defaults_empty(self):
        """mcp_capabilities defaults to empty list."""
        cfg = ContainerConfig()
        assert cfg.mcp_capabilities == []

    def test_name_auto_generated(self):
        """
        SENTINEL — name must be auto-generated starting with 'core-agent-'.
        Hardcode a name → multiple containers have same name → Docker conflict → test fails.
        """
        cfg = ContainerConfig()
        assert cfg.name.startswith("core-agent-")

    def test_each_config_has_unique_name(self):
        """Two ContainerConfigs must have different names."""
        c1 = ContainerConfig()
        c2 = ContainerConfig()
        assert c1.name != c2.name


# ===========================================================================
# ContainerManager.get_status
# ===========================================================================


class TestContainerManagerGetStatus:
    def setup_method(self):
        # Construct without calling initialize() to avoid Docker daemon
        self.mgr = ContainerManager(pool_size=5, container_ttl_minutes=30)

    def test_required_keys_present(self):
        """
        SENTINEL — get_status() must include 'pools', 'active_containers',
        'pool_size_limit', 'container_ttl_minutes'.
        Remove pool_size_limit → autoscaler has no reference ceiling → test fails.
        """
        status = self.mgr.get_status()
        assert "pools" in status
        assert "active_containers" in status
        assert "pool_size_limit" in status
        assert "container_ttl_minutes" in status

    def test_pool_size_limit_matches_constructor(self):
        """
        SENTINEL — pool_size_limit must reflect the constructor argument.
        Hardcode → deployed pool limit invisible to monitoring → test fails.
        """
        status = self.mgr.get_status()
        assert status["pool_size_limit"] == 5

    def test_container_ttl_minutes_matches_constructor(self):
        """container_ttl_minutes must reflect the constructor argument."""
        status = self.mgr.get_status()
        assert status["container_ttl_minutes"] == 30

    def test_active_containers_starts_zero(self):
        """Fresh manager has no active containers."""
        status = self.mgr.get_status()
        assert status["active_containers"] == 0

    def test_pools_keyed_by_trust_level_values(self):
        """
        SENTINEL — pools dict must contain keys for each TrustLevel value.
        Remove a TrustLevel from pools → agents at that level have no pool → crash → test fails.
        """
        status = self.mgr.get_status()
        pools = status["pools"]
        assert "trusted" in pools
        assert "sandboxed" in pools
        assert "untrusted" in pools

    def test_pool_counts_start_zero(self):
        """All pools must start at 0 containers."""
        status = self.mgr.get_status()
        for count in status["pools"].values():
            assert count == 0


# ===========================================================================
# ContainerManager._build_docker_config — pure config builder
# ===========================================================================


class TestBuildDockerConfig:
    def setup_method(self):
        self.mgr = ContainerManager()

    def test_name_passed_through(self):
        """
        SENTINEL — container name must appear in docker_config.
        Omit name → Docker assigns random name → container untrackable → test fails.
        """
        cfg = ContainerConfig(name="SENTINEL_CONTAINER")
        result = self.mgr._build_docker_config(cfg)
        assert result["name"] == "SENTINEL_CONTAINER"

    def test_image_passed_through(self):
        """image must appear in docker_config."""
        cfg = ContainerConfig(image="SENTINEL_IMAGE:latest")
        result = self.mgr._build_docker_config(cfg)
        assert result["image"] == "SENTINEL_IMAGE:latest"

    def test_working_dir_passed_through(self):
        """working_dir must appear in docker_config."""
        cfg = ContainerConfig(working_dir="/SENTINEL_DIR")
        result = self.mgr._build_docker_config(cfg)
        assert result["working_dir"] == "/SENTINEL_DIR"

    def test_detach_is_true(self):
        """
        SENTINEL — detach must be True.
        False → container runs in foreground → manager blocks forever → test fails.
        """
        cfg = ContainerConfig()
        result = self.mgr._build_docker_config(cfg)
        assert result["detach"] is True

    def test_memory_limit_applied(self):
        """
        SENTINEL — mem_limit must be set from security.memory_limit_mb.
        Omit → containers have no memory cap → OOM affects whole host → test fails.
        """
        cfg = ContainerConfig()
        result = self.mgr._build_docker_config(cfg)
        assert "mem_limit" in result
        # Must be formatted like '512m'
        assert result["mem_limit"].endswith("m")

    def test_cpu_quota_applied(self):
        """cpu_quota must be set (non-zero)."""
        cfg = ContainerConfig()
        result = self.mgr._build_docker_config(cfg)
        assert "cpu_quota" in result
        assert result["cpu_quota"] > 0

    def test_pids_limit_applied(self):
        """pids_limit must be set to cap process count."""
        cfg = ContainerConfig()
        result = self.mgr._build_docker_config(cfg)
        assert "pids_limit" in result

    def test_untrusted_gets_security_opt(self):
        """
        SENTINEL — UNTRUSTED containers must have security_opt with no-new-privileges.
        Remove UNTRUSTED branch → sandboxed agents can escalate privileges → test fails.
        """
        security = get_security_config(TrustLevel.UNTRUSTED)
        cfg = ContainerConfig(security=security)
        result = self.mgr._build_docker_config(cfg)
        assert "security_opt" in result
        assert "no-new-privileges" in result["security_opt"]

    def test_trusted_has_no_security_opt(self):
        """
        SENTINEL — TRUSTED containers must NOT have security_opt (no extra restrictions).
        Add restrictions to trusted → internal tooling breaks on privilege operations → test fails.
        """
        security = get_security_config(TrustLevel.TRUSTED)
        cfg = ContainerConfig(security=security)
        result = self.mgr._build_docker_config(cfg)
        assert "security_opt" not in result

    def test_no_network_when_disabled(self):
        """
        SENTINEL — network_mode must be 'none' when security.network_enabled is False.
        Skip this check → untrusted agents get network access → data exfiltration → test fails.
        """
        security = get_security_config(TrustLevel.UNTRUSTED)
        # Verify untrusted has network disabled
        assert security.network_enabled is False
        cfg = ContainerConfig(security=security)
        result = self.mgr._build_docker_config(cfg)
        assert result.get("network_mode") == "none"

    def test_read_only_root_when_security_requires(self):
        """read_only must be set when security.read_only_root is True."""
        security = get_security_config(TrustLevel.UNTRUSTED)
        if security.read_only_root:
            cfg = ContainerConfig(security=security)
            result = self.mgr._build_docker_config(cfg)
            assert result.get("read_only") is True


# ===========================================================================
# TrustLevel enum and TRUST_PRESETS
# ===========================================================================


class TestTrustLevel:
    def test_trusted_value(self):
        """TRUSTED must equal 'trusted'."""
        assert TrustLevel.TRUSTED == "trusted"

    def test_sandboxed_value(self):
        """SANDBOXED must equal 'sandboxed'."""
        assert TrustLevel.SANDBOXED == "sandboxed"

    def test_untrusted_value(self):
        """
        SENTINEL — UNTRUSTED must equal 'untrusted'.
        Change → get_status pools key mismatch → untrusted agents untracked → test fails.
        """
        assert TrustLevel.UNTRUSTED == "untrusted"

    def test_exactly_three_trust_levels(self):
        """Exactly 3 TrustLevel values must be defined."""
        assert len(TrustLevel) == 3

    def test_all_three_levels_have_presets(self):
        """
        SENTINEL — TRUST_PRESETS must contain all three TrustLevel keys.
        Missing UNTRUSTED → get_security_config falls back to wrong preset → test fails.
        """
        for level in TrustLevel:
            assert level in TRUST_PRESETS

    def test_untrusted_has_stricter_memory_than_trusted(self):
        """
        SENTINEL — UNTRUSTED memory limit must be <= TRUSTED limit.
        Set UNTRUSTED higher → untrusted agents get more resources → sandbox pointless → test fails.
        """
        trusted_mem = TRUST_PRESETS[TrustLevel.TRUSTED].memory_limit_mb
        untrusted_mem = TRUST_PRESETS[TrustLevel.UNTRUSTED].memory_limit_mb
        assert untrusted_mem <= trusted_mem

    def test_untrusted_has_network_disabled(self):
        """
        SENTINEL — UNTRUSTED preset must have network_enabled=False.
        Enable network → untrusted code can exfiltrate data → security breach → test fails.
        """
        assert TRUST_PRESETS[TrustLevel.UNTRUSTED].network_enabled is False

    def test_trusted_has_network_enabled(self):
        """TRUSTED preset must have network_enabled=True."""
        assert TRUST_PRESETS[TrustLevel.TRUSTED].network_enabled is True


# ===========================================================================
# get_security_config helper
# ===========================================================================


class TestGetSecurityConfig:
    def test_returns_config_for_each_level(self):
        """get_security_config must return AgentSecurityConfig for each trust level."""
        for level in TrustLevel:
            cfg = get_security_config(level)
            assert isinstance(cfg, AgentSecurityConfig)

    def test_trust_level_set_correctly(self):
        """
        SENTINEL — returned config must have the correct trust_level field.
        Copy wrong preset → untrusted agent gets trusted config → sandbox bypassed → test fails.
        """
        for level in TrustLevel:
            cfg = get_security_config(level)
            assert cfg.trust_level == level
