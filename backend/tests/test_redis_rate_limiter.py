"""
Tests for RedisRateLimiter — distributed rate limiting backed by Redis.

Sentinel-value contract:
- Removing Redis INCR call → counter never increments → test_redis_incr_called fails
- Removing TTL/EXPIRE → test_expire_set_on_first_request fails
- Removing fallback → Redis error propagates → test_redis_failure_falls_back fails
- Always returning allowed=True → test_over_limit_returns_denied fails
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# RedisRateLimiter — unit tests with mocked Redis
# ---------------------------------------------------------------------------

class TestRedisRateLimiter:
    """Tests using a fully mocked Redis pipeline."""

    def _make_limiter(self, rpm: int = 60, rph: int = 1000):
        from app.core.security import RedisRateLimiter
        return RedisRateLimiter(redis_url="redis://test:6379/0",
                                requests_per_minute=rpm,
                                requests_per_hour=rph)

    def _mock_redis(self, minute_count: int = 1, hour_count: int = 1):
        """Return a mock Redis client whose pipeline returns (minute, hour) counts.

        In redis.asyncio, pipeline() is synchronous (returns a Pipeline, not a
        coroutine), but pipeline.execute() IS async. Mock accordingly.
        """
        mock_redis = MagicMock()  # sync top-level client
        mock_pipe = MagicMock()   # sync pipeline (pipeline() is not awaited)
        mock_pipe.execute = AsyncMock(return_value=[minute_count, hour_count])
        mock_redis.pipeline.return_value = mock_pipe
        mock_redis.expire = AsyncMock()
        return mock_redis

    @pytest.mark.asyncio
    async def test_redis_incr_called(self):
        """
        SENTINEL TEST — check_rate_limit must call INCR via pipeline.
        Remove INCR calls → pipe.execute returns [0, 0] always → but more directly
        this test checks the pipeline is used.
        """
        limiter = self._make_limiter()
        mock_redis = self._mock_redis(minute_count=1, hour_count=1)

        with patch.object(limiter, "_get_client", new=AsyncMock(return_value=mock_redis)):
            result = await limiter.check_rate_limit("test-client")

        mock_redis.pipeline.assert_called_once()
        pipe = mock_redis.pipeline.return_value
        pipe.incr.assert_called()
        assert result["allowed"] is True

    @pytest.mark.asyncio
    async def test_expire_set_on_first_request(self):
        """
        SENTINEL TEST — on first request (count == 1) TTL must be set via EXPIRE.
        Remove expire call → keys persist forever → test fails.
        """
        limiter = self._make_limiter()
        mock_redis = self._mock_redis(minute_count=1, hour_count=1)

        with patch.object(limiter, "_get_client", new=AsyncMock(return_value=mock_redis)):
            await limiter.check_rate_limit("first-client")

        # expire should be called twice (minute key + hour key)
        assert mock_redis.expire.call_count == 2

    @pytest.mark.asyncio
    async def test_expire_not_set_on_subsequent_requests(self):
        """TTL is NOT reset when count > 1 (window has already started)."""
        limiter = self._make_limiter()
        mock_redis = self._mock_redis(minute_count=5, hour_count=50)

        with patch.object(limiter, "_get_client", new=AsyncMock(return_value=mock_redis)):
            await limiter.check_rate_limit("returning-client")

        mock_redis.expire.assert_not_called()

    @pytest.mark.asyncio
    async def test_over_limit_returns_denied(self):
        """
        SENTINEL TEST — when count > rpm, allowed must be False.
        Always returning True → test fails.
        """
        limiter = self._make_limiter(rpm=10, rph=1000)
        # Simulate minute counter at 11 (over the 10 rpm limit)
        mock_redis = self._mock_redis(minute_count=11, hour_count=50)

        with patch.object(limiter, "_get_client", new=AsyncMock(return_value=mock_redis)):
            result = await limiter.check_rate_limit("heavy-client")

        assert result["allowed"] is False
        assert result["retry_after"] is not None

    @pytest.mark.asyncio
    async def test_within_limit_returns_allowed(self):
        limiter = self._make_limiter(rpm=60, rph=1000)
        mock_redis = self._mock_redis(minute_count=1, hour_count=1)

        with patch.object(limiter, "_get_client", new=AsyncMock(return_value=mock_redis)):
            result = await limiter.check_rate_limit("normal-client")

        assert result["allowed"] is True
        assert result["backend"] == "redis"

    @pytest.mark.asyncio
    async def test_over_limit_decrements_counter(self):
        """
        SENTINEL TEST — when denied, DECR must be called to undo the INCR.
        Skip rollback → counter permanently inflated → test fails.
        """
        limiter = self._make_limiter(rpm=5, rph=1000)

        incr_pipe = MagicMock()
        incr_pipe.execute = AsyncMock(return_value=[6, 1])  # minute over limit

        rollback_pipe = MagicMock()
        rollback_pipe.execute = AsyncMock(return_value=[5, 0])

        mock_redis = MagicMock()
        mock_redis.pipeline.side_effect = [incr_pipe, rollback_pipe]
        mock_redis.expire = AsyncMock()

        with patch.object(limiter, "_get_client", new=AsyncMock(return_value=mock_redis)):
            result = await limiter.check_rate_limit("over-limit")

        assert result["allowed"] is False
        rollback_pipe.decr.assert_called()

    @pytest.mark.asyncio
    async def test_redis_failure_falls_back(self):
        """
        SENTINEL TEST — Redis errors must not propagate; fallback to in-memory.
        Remove try/except → Redis error crashes caller → test fails.
        """
        limiter = self._make_limiter()

        with patch.object(limiter, "_get_client",
                          new=AsyncMock(side_effect=ConnectionError("Redis down"))):
            result = await limiter.check_rate_limit("fallback-client")

        assert isinstance(result, dict)
        assert "allowed" in result
        assert result.get("backend") == "memory_fallback"

    @pytest.mark.asyncio
    async def test_result_contains_required_keys(self):
        """Result shape must match RateLimiter contract."""
        limiter = self._make_limiter()
        mock_redis = self._mock_redis(1, 1)

        with patch.object(limiter, "_get_client", new=AsyncMock(return_value=mock_redis)):
            result = await limiter.check_rate_limit("shape-client")

        for key in ("allowed", "minute_remaining", "hour_remaining"):
            assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Limiter class selection logic (tested without module reload)
# ---------------------------------------------------------------------------

class TestRateLimiterSelection:
    def test_redis_limiter_class_exists(self):
        """
        SENTINEL TEST — RedisRateLimiter class must be importable.
        Remove the class → ImportError → test fails.
        """
        from app.core.security import RedisRateLimiter
        assert RedisRateLimiter is not None

    def test_redis_limiter_stores_redis_url(self):
        """RedisRateLimiter must accept and store redis_url constructor arg."""
        from app.core.security import RedisRateLimiter
        limiter = RedisRateLimiter(redis_url="redis://custom:6379/1")
        assert limiter.redis_url == "redis://custom:6379/1"

    def test_memory_limiter_is_fallback_in_redis_limiter(self):
        """
        SENTINEL TEST — RedisRateLimiter must contain an in-memory fallback.
        Remove _fallback → Redis errors crash callers → test fails.
        """
        from app.core.security import RedisRateLimiter, RateLimiter
        limiter = RedisRateLimiter()
        assert isinstance(limiter._fallback, RateLimiter)

    def test_rpm_rph_forwarded_to_fallback(self):
        """Constructor rpm/rph must be forwarded to the in-memory fallback."""
        from app.core.security import RedisRateLimiter
        limiter = RedisRateLimiter(requests_per_minute=30, requests_per_hour=500)
        assert limiter._fallback.rpm == 30
        assert limiter._fallback.rph == 500
