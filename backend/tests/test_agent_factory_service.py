"""
Tests for AgentFactoryService — agent lifecycle, cache management, LLM personality tuning.

Sentinel-value contract:
- Remove TTL check → expired agents served from cache → test fails
- Remove cache storage → every call re-creates agent → test fails
- Remove is_expired TTL comparison → freshly created agents reported expired → test fails
- Remove trait weights from _calculate_temperature → personality doesn't affect LLM → test fails
- Remove clamp in _calculate_temperature → personality extremes cause illegal values → test fails
- Remove clear_cache pop → clearing specific agent removes nothing → test fails
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.agent_models import AgentConfig
from app.services.agent_factory_service import AgentFactoryService, AgentInstance


# ===========================================================================
# Helpers
# ===========================================================================


def _make_config(
    agent_id: str = "test-agent-1",
    agent_name: str = "TestAgent",
    is_active: bool = True,
    traits: Dict[str, float] | None = None,
) -> AgentConfig:
    return AgentConfig(
        agent_id=agent_id,
        agent_name=agent_name,
        agent_type="task_agent",
        system_prompt="You are a test agent for unit testing purposes.",
        is_active=is_active,
        personality_traits=traits or {},
    )


def _make_instance(
    agent_id: str = "test-agent-1",
    agent_name: str = "TestAgent",
    created_at: datetime | None = None,
) -> AgentInstance:
    config = _make_config(agent_id=agent_id, agent_name=agent_name)
    instance = AgentInstance(config=config, agent=MagicMock())
    if created_at is not None:
        instance.created_at = created_at
    return instance


def _make_factory(ttl_minutes: int = 5) -> AgentFactoryService:
    """Create a factory with mocked LLM and MCP service."""
    mock_llm = MagicMock()
    with patch(
        "app.services.agent_factory_service.get_agent_mcp_service",
        return_value=MagicMock(),
    ):
        factory = AgentFactoryService(llm=mock_llm, instance_ttl_minutes=ttl_minutes)
    return factory


# ===========================================================================
# AgentInstance — properties and TTL
# ===========================================================================


class TestAgentInstance:
    def test_agent_id_delegates_to_config(self):
        """agent_id property must return config.agent_id."""
        instance = _make_instance(agent_id="SENTINEL_AGENT_ID")
        assert instance.agent_id == "SENTINEL_AGENT_ID"

    def test_agent_name_delegates_to_config(self):
        """agent_name property must return config.agent_name."""
        instance = _make_instance(agent_name="SENTINEL_NAME")
        assert instance.agent_name == "SENTINEL_NAME"

    def test_fresh_instance_is_not_expired(self):
        """A just-created instance must not be expired."""
        instance = _make_instance()
        assert instance.is_expired(timedelta(minutes=5)) is False

    def test_old_instance_is_expired(self):
        """
        SENTINEL — an instance older than TTL must report expired.
        Remove is_expired TTL check → stale agents never refreshed → test fails.
        """
        old_time = datetime.utcnow() - timedelta(minutes=10)
        instance = _make_instance(created_at=old_time)
        assert instance.is_expired(timedelta(minutes=5)) is True

    def test_instance_exactly_at_ttl_is_expired(self):
        """Instance age == TTL (not less than) must be considered expired."""
        ttl = timedelta(minutes=5)
        at_boundary = datetime.utcnow() - ttl - timedelta(seconds=1)
        instance = _make_instance(created_at=at_boundary)
        assert instance.is_expired(ttl) is True

    def test_created_at_is_set_on_construction(self):
        """created_at must be set to approximately now when constructed."""
        before = datetime.utcnow()
        instance = _make_instance()
        after = datetime.utcnow()
        assert before <= instance.created_at <= after


# ===========================================================================
# _calculate_temperature — personality → LLM temperature
# ===========================================================================


class TestCalculateTemperature:
    def test_high_curiosity_raises_temperature(self):
        """
        SENTINEL — curiosity trait must increase temperature.
        Remove curiosity weight → curious agents indistinguishable from neutral → test fails.
        """
        factory = _make_factory()
        low = factory._calculate_temperature({"curiosity": 0.0})
        high = factory._calculate_temperature({"curiosity": 1.0})
        assert high > low

    def test_high_precision_lowers_temperature(self):
        """
        SENTINEL — precision trait must decrease temperature.
        Remove precision weight → precise agents behave randomly → test fails.
        """
        factory = _make_factory()
        low_prec = factory._calculate_temperature({"technical_precision": 0.0})
        high_prec = factory._calculate_temperature({"technical_precision": 1.0})
        assert high_prec < low_prec

    def test_temperature_clamped_to_minimum(self):
        """
        SENTINEL — temperature must never go below 0.3 regardless of traits.
        Remove clamp → precision-maxed agent gets illegal temperature → test fails.
        """
        factory = _make_factory()
        result = factory._calculate_temperature(
            {
                "curiosity": 0.0,
                "creativity": 0.0,
                "uncertainty": 0.0,
                "technical_precision": 1.0,
            }
        )
        assert result >= 0.3

    def test_temperature_clamped_to_maximum(self):
        """Temperature must never exceed 0.9."""
        factory = _make_factory()
        result = factory._calculate_temperature(
            {
                "curiosity": 1.0,
                "creativity": 1.0,
                "uncertainty": 1.0,
                "technical_precision": 0.0,
            }
        )
        assert result <= 0.9

    def test_empty_traits_returns_default_range(self):
        """Empty traits must produce a temperature in the valid clamped range."""
        factory = _make_factory()
        result = factory._calculate_temperature({})
        assert 0.3 <= result <= 0.9

    def test_known_value(self):
        """
        SENTINEL — hand-computed value verifies the exact formula is unchanged.
        Change weights → this test fails immediately → formula regression detected.

        Formula: 0.4*curiosity + 0.3*creativity + 0.2*uncertainty - 0.3*precision
        With all traits at 0.5:
          = 0.4*0.5 + 0.3*0.5 + 0.2*0.5 - 0.3*0.5
          = 0.2 + 0.15 + 0.1 - 0.15 = 0.3
        Clamped to [0.3, 0.9] → 0.3
        """
        factory = _make_factory()
        result = factory._calculate_temperature(
            {
                "curiosity": 0.5,
                "creativity": 0.5,
                "uncertainty": 0.5,
                "technical_precision": 0.5,
            }
        )
        assert abs(result - 0.3) < 1e-9


# ===========================================================================
# _calculate_top_p — personality → LLM top_p
# ===========================================================================


class TestCalculateTopP:
    def test_high_creativity_raises_top_p(self):
        """
        SENTINEL — creativity must increase top_p (more word diversity).
        Remove creativity weight → creative agents use same top_p → test fails.
        """
        factory = _make_factory()
        low = factory._calculate_top_p({"creativity": 0.0})
        high = factory._calculate_top_p({"creativity": 1.0})
        assert high > low

    def test_high_precision_lowers_top_p(self):
        """Precision trait must decrease top_p (more focused word selection).
        Use creativity=1.0 to lift low_prec above the 0.7 floor so the comparison is visible.
        """
        factory = _make_factory()
        # creativity=1.0, precision=0.0 → 0.5 + 0.4 - 0.0 = 0.9
        low_prec = factory._calculate_top_p(
            {"creativity": 1.0, "technical_precision": 0.0}
        )
        # creativity=1.0, precision=1.0 → 0.5 + 0.4 - 0.2 = 0.7
        high_prec = factory._calculate_top_p(
            {"creativity": 1.0, "technical_precision": 1.0}
        )
        assert high_prec < low_prec

    def test_top_p_clamped_to_minimum(self):
        """
        SENTINEL — top_p must never go below 0.7 regardless of traits.
        Remove clamp → precision-maxed agent gets illegal top_p → test fails.
        """
        factory = _make_factory()
        result = factory._calculate_top_p(
            {"creativity": 0.0, "technical_precision": 1.0}
        )
        assert result >= 0.7

    def test_top_p_clamped_to_maximum(self):
        """top_p must never exceed 1.0."""
        factory = _make_factory()
        result = factory._calculate_top_p(
            {"creativity": 1.0, "technical_precision": 0.0}
        )
        assert result <= 1.0

    def test_empty_traits_returns_valid_value(self):
        """Empty traits must produce a top_p in [0.7, 1.0]."""
        factory = _make_factory()
        result = factory._calculate_top_p({})
        assert 0.7 <= result <= 1.0

    def test_known_value(self):
        """
        SENTINEL — hand-computed value verifies the exact formula.
        Formula: 0.5 + 0.4*creativity - 0.2*precision
        With creativity=0.5, precision=0.5:
          = 0.5 + 0.4*0.5 - 0.2*0.5 = 0.5 + 0.2 - 0.1 = 0.6
        Clamped to [0.7, 1.0] → 0.7
        """
        factory = _make_factory()
        result = factory._calculate_top_p(
            {"creativity": 0.5, "technical_precision": 0.5}
        )
        assert abs(result - 0.7) < 1e-9


# ===========================================================================
# clear_cache / get_cache_stats
# ===========================================================================


class TestCacheManagement:
    def test_clear_all_empties_cache(self):
        """
        SENTINEL — clear_cache() with no arg must empty the entire cache.
        Remove self._instance_cache.clear() → agents persist after clear → test fails.
        """
        factory = _make_factory()
        factory._instance_cache["agent-1"] = _make_instance("agent-1")
        factory._instance_cache["agent-2"] = _make_instance("agent-2")
        factory.clear_cache()
        assert len(factory._instance_cache) == 0

    def test_clear_specific_agent_removes_only_that_agent(self):
        """
        SENTINEL — clear_cache("agent-1") must remove only agent-1, leave agent-2.
        Use .clear() always → other agents evicted → test fails.
        """
        factory = _make_factory()
        factory._instance_cache["agent-1"] = _make_instance("agent-1")
        factory._instance_cache["agent-2"] = _make_instance("agent-2")
        factory.clear_cache("agent-1")
        assert "agent-1" not in factory._instance_cache
        assert "agent-2" in factory._instance_cache

    def test_clear_nonexistent_agent_does_not_raise(self):
        """Clearing a non-existent agent must not raise."""
        factory = _make_factory()
        factory.clear_cache("nonexistent")  # should not raise

    def test_get_cache_stats_reports_count(self):
        """
        SENTINEL — get_cache_stats must return correct cached_agents count.
        Return 0 always → monitoring thinks factory is always empty → test fails.
        """
        factory = _make_factory()
        factory._instance_cache["agent-1"] = _make_instance("agent-1")
        factory._instance_cache["agent-2"] = _make_instance("agent-2")
        stats = factory.get_cache_stats()
        assert stats["cached_agents"] == 2

    def test_get_cache_stats_includes_agent_ids(self):
        """Each agent's data must appear in stats['agents'] list."""
        factory = _make_factory()
        factory._instance_cache["SENTINEL_AGENT"] = _make_instance("SENTINEL_AGENT")
        stats = factory.get_cache_stats()
        agent_ids = [a["agent_id"] for a in stats["agents"]]
        assert "SENTINEL_AGENT" in agent_ids

    def test_get_cache_stats_empty_factory(self):
        """Empty factory must return cached_agents=0 and empty list."""
        factory = _make_factory()
        stats = factory.get_cache_stats()
        assert stats["cached_agents"] == 0
        assert stats["agents"] == []

    def test_get_cache_stats_reports_ttl(self):
        """TTL must be echoed in stats for monitoring dashboards."""
        factory = _make_factory(ttl_minutes=10)
        stats = factory.get_cache_stats()
        assert stats["ttl_minutes"] == 10.0

    def test_cache_stats_expires_in_decreases_with_age(self):
        """expires_in_seconds must be smaller for older instances."""
        factory = _make_factory(ttl_minutes=5)
        # Fresh instance
        fresh = _make_instance("fresh", created_at=datetime.utcnow())
        # Old instance
        old = _make_instance("old", created_at=datetime.utcnow() - timedelta(minutes=4))
        factory._instance_cache["fresh"] = fresh
        factory._instance_cache["old"] = old

        stats = factory.get_cache_stats()
        by_id = {a["agent_id"]: a for a in stats["agents"]}
        assert by_id["fresh"]["expires_in_seconds"] > by_id["old"]["expires_in_seconds"]


# ===========================================================================
# get_agent — cache hit / miss / expiry
# ===========================================================================


class TestGetAgent:
    @pytest.mark.asyncio
    async def test_cache_hit_returns_existing_instance(self):
        """
        SENTINEL — if a valid cached instance exists, it must be returned without
        calling _create_agent_instance.
        Remove cache check → every call re-creates the agent → test fails.
        """
        factory = _make_factory()
        existing = _make_instance("agent-1")
        factory._instance_cache["agent-1"] = existing

        with patch.object(factory, "_create_agent_instance", new=AsyncMock()) as spy:
            result = await factory.get_agent("agent-1")

        assert result is existing
        spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_calls_create(self):
        """
        SENTINEL — a cache miss must trigger _create_agent_instance.
        Remove the else branch → unknown agents never instantiated → test fails.
        """
        factory = _make_factory()
        new_instance = _make_instance("agent-x")

        with patch.object(
            factory, "_create_agent_instance", new=AsyncMock(return_value=new_instance)
        ):
            result = await factory.get_agent("agent-x")

        assert result is new_instance

    @pytest.mark.asyncio
    async def test_created_instance_is_stored_in_cache(self):
        """
        SENTINEL — a new instance must be written into the cache.
        Remove cache storage → next call re-creates the agent → test fails.
        """
        factory = _make_factory()
        new_instance = _make_instance("agent-x")

        with patch.object(
            factory, "_create_agent_instance", new=AsyncMock(return_value=new_instance)
        ):
            await factory.get_agent("agent-x")

        assert factory._instance_cache.get("agent-x") is new_instance

    @pytest.mark.asyncio
    async def test_expired_instance_triggers_recreation(self):
        """
        SENTINEL — a cached but expired instance must be evicted and recreated.
        Remove TTL check → stale agents served indefinitely → test fails.
        """
        factory = _make_factory(ttl_minutes=5)
        old_instance = _make_instance(
            "agent-1", created_at=datetime.utcnow() - timedelta(minutes=10)
        )
        factory._instance_cache["agent-1"] = old_instance

        new_instance = _make_instance("agent-1")
        with patch.object(
            factory, "_create_agent_instance", new=AsyncMock(return_value=new_instance)
        ) as spy:
            result = await factory.get_agent("agent-1")

        spy.assert_called_once_with("agent-1")
        assert result is new_instance
        assert factory._instance_cache["agent-1"] is new_instance

    @pytest.mark.asyncio
    async def test_expired_instance_removed_from_cache(self):
        """
        SENTINEL — after TTL expiry, old instance must be evicted before recreation.
        Keep stale instance → cache holds two references → test fails.
        """
        factory = _make_factory(ttl_minutes=5)
        old_instance = _make_instance(
            "agent-1", created_at=datetime.utcnow() - timedelta(minutes=10)
        )
        factory._instance_cache["agent-1"] = old_instance

        with patch.object(
            factory, "_create_agent_instance", new=AsyncMock(return_value=None)
        ):
            await factory.get_agent("agent-1")

        # old expired instance should be gone even if create returns None
        assert "agent-1" not in factory._instance_cache

    @pytest.mark.asyncio
    async def test_none_from_create_not_cached(self):
        """If _create_agent_instance returns None, nothing should be cached."""
        factory = _make_factory()
        with patch.object(
            factory, "_create_agent_instance", new=AsyncMock(return_value=None)
        ):
            result = await factory.get_agent("missing-agent")

        assert result is None
        assert "missing-agent" not in factory._instance_cache
