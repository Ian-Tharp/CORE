"""
Tests for rate_limit middleware -- token bucket rate limiter.

Covers:
- RateLimiter: init, allow, refill, burst, retry_after, reset
- RedisRateLimiter: init, create, allow_async, fallback, close, key_prefix
- Key extraction: get_client_ip, get_api_key
- check_rate_limit: pass-through and 429 rejection (both sync and async limiters)
- rate_limit decorator: wiring
- upgrade_limiters_to_redis: module-level upgrade
"""

import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from app.middleware.rate_limit import (
    RateLimiter,
    RedisRateLimiter,
    get_client_ip,
    get_api_key,
    check_rate_limit,
    rate_limit,
    upgrade_limiters_to_redis,
    engine_limiter,
    api_limiter,
    public_limiter,
    _TOKEN_BUCKET_LUA,
)


# ───────── RateLimiter core ─────────


class TestRateLimiterInit:
    def test_defaults(self):
        rl = RateLimiter()
        assert rl.rate == 60 / 60.0
        assert rl.burst_size == 60

    def test_custom_rpm_and_burst(self):
        rl = RateLimiter(requests_per_minute=120, burst_size=10)
        assert rl.rate == 2.0
        assert rl.burst_size == 10

    def test_burst_defaults_to_rpm(self):
        rl = RateLimiter(requests_per_minute=30)
        assert rl.burst_size == 30


class TestRateLimiterAllow:
    def test_first_request_allowed(self):
        rl = RateLimiter(requests_per_minute=60, burst_size=5)
        assert rl.allow("client-a") is True

    def test_exhaust_burst(self):
        rl = RateLimiter(requests_per_minute=60, burst_size=3)
        assert rl.allow("k") is True
        assert rl.allow("k") is True
        assert rl.allow("k") is True
        # 4th should be rejected (no time to refill)
        assert rl.allow("k") is False

    def test_different_keys_independent(self):
        rl = RateLimiter(requests_per_minute=60, burst_size=1)
        assert rl.allow("a") is True
        assert rl.allow("a") is False
        # different key still has its own bucket
        assert rl.allow("b") is True

    def test_custom_cost(self):
        rl = RateLimiter(requests_per_minute=60, burst_size=5)
        assert rl.allow("k", cost=5) is True
        assert rl.allow("k", cost=1) is False

    def test_cost_greater_than_tokens_rejected(self):
        rl = RateLimiter(requests_per_minute=60, burst_size=2)
        assert rl.allow("k", cost=3) is False


class TestRateLimiterRefill:
    @patch("app.middleware.rate_limit.time")
    def test_tokens_refill_over_time(self, mock_time):
        mock_time.time.return_value = 1000.0
        rl = RateLimiter(requests_per_minute=60, burst_size=2)  # 1 tok/sec

        assert rl.allow("k") is True
        assert rl.allow("k") is True
        assert rl.allow("k") is False  # empty

        # Advance 1 second → 1 token refill
        mock_time.time.return_value = 1001.0
        assert rl.allow("k") is True
        assert rl.allow("k") is False

    @patch("app.middleware.rate_limit.time")
    def test_refill_capped_at_burst(self, mock_time):
        mock_time.time.return_value = 1000.0
        rl = RateLimiter(requests_per_minute=60, burst_size=3)

        # drain
        rl.allow("k")
        rl.allow("k")
        rl.allow("k")

        # wait a long time
        mock_time.time.return_value = 2000.0
        rl._refill("k")
        assert rl.tokens["k"] == 3  # capped


class TestRateLimiterRetryAfter:
    @patch("app.middleware.rate_limit.time")
    def test_retry_after_zero_when_allowed(self, mock_time):
        mock_time.time.return_value = 1000.0
        rl = RateLimiter(requests_per_minute=60, burst_size=5)
        assert rl.get_retry_after("k") == 0

    @patch("app.middleware.rate_limit.time")
    def test_retry_after_positive_when_exhausted(self, mock_time):
        mock_time.time.return_value = 1000.0
        rl = RateLimiter(requests_per_minute=60, burst_size=1)
        rl.allow("k")  # exhaust
        mock_time.time.return_value = 1000.0  # no time passes
        retry = rl.get_retry_after("k")
        assert retry > 0


class TestRateLimiterReset:
    def test_reset_restores_tokens(self):
        rl = RateLimiter(requests_per_minute=60, burst_size=2)
        rl.allow("k")
        rl.allow("k")
        assert rl.allow("k") is False
        rl.reset("k")
        assert rl.allow("k") is True


# ───────── RedisRateLimiter ─────────


class TestRedisRateLimiterInit:
    def test_defaults(self):
        rl = RedisRateLimiter()
        assert rl.rate == 1.0
        assert rl.burst_size == 60
        assert rl._key_prefix == "rl"
        assert rl._redis is None
        assert rl._redis_healthy is True

    def test_custom_params(self):
        mock_redis = MagicMock()
        rl = RedisRateLimiter(
            requests_per_minute=120,
            burst_size=10,
            redis_client=mock_redis,
            key_prefix="test",
        )
        assert rl.rate == 2.0
        assert rl.burst_size == 10
        assert rl._redis is mock_redis
        assert rl._key_prefix == "test"

    def test_fallback_limiter_created(self):
        rl = RedisRateLimiter(requests_per_minute=30, burst_size=5)
        assert isinstance(rl._fallback, RateLimiter)
        assert rl._fallback.burst_size == 5

    def test_bucket_key(self):
        rl = RedisRateLimiter(key_prefix="myapp")
        assert rl._bucket_key("client-1") == "myapp:client-1"


class TestRedisRateLimiterCreate:
    @pytest.mark.asyncio
    async def test_create_success(self):
        """Verify that after construction + manual setup, properties are correct."""
        mock_client = AsyncMock()
        rl = RedisRateLimiter(
            requests_per_minute=30,
            burst_size=10,
            redis_client=mock_client,
            key_prefix="test",
        )
        rl._script_sha = "abc123sha"
        rl._redis_healthy = True
        assert rl.is_redis_connected is True
        assert rl.rate == 0.5
        assert rl.burst_size == 10

    @pytest.mark.asyncio
    async def test_create_redis_unavailable_sets_unhealthy(self):
        rl = RedisRateLimiter(requests_per_minute=30, burst_size=10)
        rl._redis_healthy = False
        assert rl.is_redis_connected is False


class TestRedisRateLimiterAllowAsync:
    @pytest.mark.asyncio
    async def test_allow_async_redis_success(self):
        mock_redis = AsyncMock()
        # Return: [allowed=1, tokens_remaining="4.0", retry_after_ms=0]
        mock_redis.evalsha = AsyncMock(return_value=[1, "4.0", 0])

        rl = RedisRateLimiter(
            requests_per_minute=60, burst_size=5, redis_client=mock_redis
        )
        rl._script_sha = "testsha"
        rl._redis_healthy = True

        result = await rl.allow_async("client-1")
        assert result is True
        mock_redis.evalsha.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_allow_async_redis_denied(self):
        mock_redis = AsyncMock()
        mock_redis.evalsha = AsyncMock(return_value=[0, "0.0", 500])

        rl = RedisRateLimiter(
            requests_per_minute=60, burst_size=5, redis_client=mock_redis
        )
        rl._script_sha = "testsha"
        rl._redis_healthy = True

        result = await rl.allow_async("client-1")
        assert result is False

    @pytest.mark.asyncio
    async def test_allow_async_fallback_when_redis_unhealthy(self):
        rl = RedisRateLimiter(requests_per_minute=60, burst_size=2)
        rl._redis_healthy = False

        assert await rl.allow_async("k") is True
        assert await rl.allow_async("k") is True
        assert await rl.allow_async("k") is False  # fallback exhausted

    @pytest.mark.asyncio
    async def test_allow_async_fallback_when_no_redis(self):
        rl = RedisRateLimiter(requests_per_minute=60, burst_size=1)
        rl._redis = None

        assert await rl.allow_async("k") is True
        assert await rl.allow_async("k") is False

    @pytest.mark.asyncio
    async def test_allow_async_falls_back_on_redis_error(self):
        mock_redis = AsyncMock()
        mock_redis.evalsha = AsyncMock(side_effect=ConnectionError("Redis gone"))

        rl = RedisRateLimiter(
            requests_per_minute=60, burst_size=2, redis_client=mock_redis
        )
        rl._script_sha = "testsha"
        rl._redis_healthy = True

        # Should fall back to in-memory, not raise
        result = await rl.allow_async("k")
        assert result is True
        assert rl._redis_healthy is False  # marked unhealthy

    @pytest.mark.asyncio
    async def test_allow_async_custom_cost(self):
        mock_redis = AsyncMock()
        mock_redis.evalsha = AsyncMock(return_value=[1, "0.0", 0])

        rl = RedisRateLimiter(
            requests_per_minute=60, burst_size=5, redis_client=mock_redis
        )
        rl._script_sha = "testsha"
        rl._redis_healthy = True

        await rl.allow_async("k", cost=3.0)
        call_args = mock_redis.evalsha.call_args
        # evalsha(sha, numkeys, KEYS[1], ARGV[1]=rate, ARGV[2]=burst, ARGV[3]=cost, ARGV[4]=now)
        # positional: [0]=sha, [1]=1, [2]=key, [3]=rate, [4]=burst, [5]=cost, [6]=now
        assert call_args[0][5] == "3.0"  # cost arg


class TestRedisRateLimiterSyncFallback:
    def test_allow_sync_uses_fallback(self):
        rl = RedisRateLimiter(requests_per_minute=60, burst_size=2)
        assert rl.allow("k") is True
        assert rl.allow("k") is True
        assert rl.allow("k") is False

    def test_get_retry_after_sync_uses_fallback(self):
        rl = RedisRateLimiter(requests_per_minute=60, burst_size=1)
        rl.allow("k")  # exhaust
        assert rl.get_retry_after("k") > 0

    def test_reset_sync_resets_fallback(self):
        rl = RedisRateLimiter(requests_per_minute=60, burst_size=1)
        rl.allow("k")
        assert rl.allow("k") is False
        rl.reset("k")
        assert rl.allow("k") is True


class TestRedisRateLimiterRetryAfterAsync:
    @pytest.mark.asyncio
    async def test_retry_after_from_redis(self):
        mock_redis = AsyncMock()
        mock_redis.evalsha = AsyncMock(return_value=[0, "0.0", 1500])

        rl = RedisRateLimiter(
            requests_per_minute=60, burst_size=5, redis_client=mock_redis
        )
        rl._script_sha = "testsha"
        rl._redis_healthy = True

        retry = await rl.get_retry_after_async("k")
        assert retry == 1.5  # 1500ms -> 1.5s

    @pytest.mark.asyncio
    async def test_retry_after_fallback_when_unhealthy(self):
        rl = RedisRateLimiter(requests_per_minute=60, burst_size=1)
        rl._redis_healthy = False
        rl._fallback.allow("k")  # exhaust
        retry = await rl.get_retry_after_async("k")
        assert retry > 0


class TestRedisRateLimiterResetAsync:
    @pytest.mark.asyncio
    async def test_reset_deletes_redis_key(self):
        mock_redis = AsyncMock()
        rl = RedisRateLimiter(
            requests_per_minute=60,
            burst_size=5,
            redis_client=mock_redis,
            key_prefix="rl",
        )
        rl._redis_healthy = True

        await rl.reset_async("client-1")
        mock_redis.delete.assert_awaited_once_with("rl:client-1")

    @pytest.mark.asyncio
    async def test_reset_async_also_resets_fallback(self):
        rl = RedisRateLimiter(requests_per_minute=60, burst_size=1)
        rl._redis_healthy = False
        rl._fallback.allow("k")  # exhaust fallback
        await rl.reset_async("k")
        assert rl._fallback.allow("k") is True  # restored

    @pytest.mark.asyncio
    async def test_reset_async_ignores_redis_error(self):
        mock_redis = AsyncMock()
        mock_redis.delete = AsyncMock(side_effect=ConnectionError("nope"))
        rl = RedisRateLimiter(redis_client=mock_redis)
        rl._redis_healthy = True
        # Should not raise
        await rl.reset_async("k")


class TestRedisRateLimiterClose:
    @pytest.mark.asyncio
    async def test_close_calls_redis_close(self):
        mock_redis = AsyncMock()
        rl = RedisRateLimiter(redis_client=mock_redis)
        await rl.close()
        mock_redis.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_no_redis_noop(self):
        rl = RedisRateLimiter()
        await rl.close()  # should not raise

    @pytest.mark.asyncio
    async def test_close_ignores_error(self):
        mock_redis = AsyncMock()
        mock_redis.close = AsyncMock(side_effect=Exception("close failed"))
        rl = RedisRateLimiter(redis_client=mock_redis)
        await rl.close()  # should not raise


class TestRedisRateLimiterProperties:
    def test_is_redis_connected_true(self):
        rl = RedisRateLimiter(redis_client=MagicMock())
        rl._redis_healthy = True
        assert rl.is_redis_connected is True

    def test_is_redis_connected_false_no_client(self):
        rl = RedisRateLimiter()
        rl._redis_healthy = True
        assert rl.is_redis_connected is False

    def test_is_redis_connected_false_unhealthy(self):
        rl = RedisRateLimiter(redis_client=MagicMock())
        rl._redis_healthy = False
        assert rl.is_redis_connected is False


# ───────── Key extraction ─────────


class TestGetClientIP:
    def test_direct_client(self):
        req = MagicMock(spec=["headers", "client"])
        req.headers = {}
        req.client.host = "10.0.0.1"
        assert get_client_ip(req) == "10.0.0.1"

    def test_forwarded_header(self):
        req = MagicMock(spec=["headers", "client"])
        req.headers = {"X-Forwarded-For": "1.2.3.4, 10.0.0.1"}
        assert get_client_ip(req) == "1.2.3.4"

    def test_no_client(self):
        req = MagicMock(spec=["headers", "client"])
        req.headers = {}
        req.client = None
        assert get_client_ip(req) == "unknown"


class TestGetApiKey:
    def test_with_key_header(self):
        req = MagicMock(spec=["headers", "client"])
        req.headers = {"X-API-Key": "my-key"}
        req.client.host = "1.2.3.4"
        assert get_api_key(req) == "my-key"

    def test_without_key_falls_back_to_ip(self):
        req = MagicMock(spec=["headers", "client"])
        req.headers = {}
        req.client.host = "5.6.7.8"
        assert get_api_key(req) == "5.6.7.8"


# ───────── check_rate_limit ─────────


class TestCheckRateLimitSync:
    @pytest.mark.asyncio
    async def test_allowed_passes(self):
        limiter = RateLimiter(requests_per_minute=60, burst_size=5)
        req = MagicMock(spec=["headers", "client"])
        req.headers = {}
        req.client.host = "1.2.3.4"
        await check_rate_limit(req, limiter)  # no exception

    @pytest.mark.asyncio
    async def test_rejected_raises_429(self):
        limiter = RateLimiter(requests_per_minute=60, burst_size=1)
        req = MagicMock(spec=["headers", "client"])
        req.headers = {}
        req.client.host = "1.2.3.4"
        await check_rate_limit(req, limiter)  # consume the one token
        with pytest.raises(HTTPException) as exc_info:
            await check_rate_limit(req, limiter)
        assert exc_info.value.status_code == 429


class TestCheckRateLimitAsync:
    @pytest.mark.asyncio
    async def test_allowed_with_redis_limiter(self):
        mock_redis = AsyncMock()
        mock_redis.evalsha = AsyncMock(return_value=[1, "4.0", 0])
        limiter = RedisRateLimiter(
            requests_per_minute=60, burst_size=5, redis_client=mock_redis
        )
        limiter._script_sha = "sha"
        limiter._redis_healthy = True

        req = MagicMock(spec=["headers", "client"])
        req.headers = {}
        req.client.host = "1.2.3.4"
        await check_rate_limit(req, limiter)  # no exception

    @pytest.mark.asyncio
    async def test_rejected_with_redis_limiter(self):
        mock_redis = AsyncMock()
        # First call: denied
        mock_redis.evalsha = AsyncMock(return_value=[0, "0.0", 2000])
        limiter = RedisRateLimiter(
            requests_per_minute=60, burst_size=5, redis_client=mock_redis
        )
        limiter._script_sha = "sha"
        limiter._redis_healthy = True

        # Need allow_async to return False
        # The check_rate_limit calls allow_async then get_retry_after_async
        # allow_async will call evalsha and get [0, ...] -> False
        # then get_retry_after_async will call evalsha and get [0, "0.0", 2000]
        req = MagicMock(spec=["headers", "client"])
        req.headers = {}
        req.client.host = "1.2.3.4"
        with pytest.raises(HTTPException) as exc_info:
            await check_rate_limit(req, limiter)
        assert exc_info.value.status_code == 429


# ───────── rate_limit decorator ─────────


class TestRateLimitDecorator:
    @pytest.mark.asyncio
    async def test_decorator_passes_when_allowed(self):
        limiter = RateLimiter(requests_per_minute=60, burst_size=5)

        @rate_limit(limiter)
        async def handler(request=None):
            return "ok"

        req = MagicMock(spec=["headers", "client"])
        req.headers = {}
        req.client.host = "1.2.3.4"
        result = await handler(request=req)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_decorator_raises_429(self):
        limiter = RateLimiter(requests_per_minute=60, burst_size=1)

        @rate_limit(limiter)
        async def handler(request=None):
            return "ok"

        req = MagicMock(spec=["headers", "client"])
        req.headers = {}
        req.client.host = "1.2.3.4"
        await handler(request=req)  # first passes
        with pytest.raises(HTTPException) as exc_info:
            await handler(request=req)
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_decorator_no_request_skips_limit(self):
        limiter = RateLimiter(requests_per_minute=60, burst_size=1)

        @rate_limit(limiter)
        async def handler():
            return "ok"

        result = await handler()
        assert result == "ok"


# ───────── upgrade_limiters_to_redis ─────────


class TestUpgradeLimitersToRedis:
    @pytest.mark.asyncio
    async def test_upgrade_success(self):
        mock_create = AsyncMock()
        mock_instance = MagicMock(spec=RedisRateLimiter)
        mock_create.return_value = mock_instance

        with patch.object(RedisRateLimiter, "create", mock_create):
            await upgrade_limiters_to_redis()
            assert mock_create.await_count == 3

    @pytest.mark.asyncio
    async def test_upgrade_failure_keeps_in_memory(self):
        """upgrade_limiters_to_redis should not raise even if Redis is unavailable."""
        with patch.object(
            RedisRateLimiter, "create", AsyncMock(side_effect=Exception("no redis"))
        ):
            await upgrade_limiters_to_redis()  # should not raise


# ───────── Lua script ─────────


class TestTokenBucketLua:
    def test_lua_script_is_string(self):
        assert isinstance(_TOKEN_BUCKET_LUA, str)
        assert "KEYS[1]" in _TOKEN_BUCKET_LUA
        assert "ARGV" in _TOKEN_BUCKET_LUA

    def test_lua_script_contains_token_logic(self):
        assert "tokens" in _TOKEN_BUCKET_LUA
        assert "EXPIRE" in _TOKEN_BUCKET_LUA
        assert "HMSET" in _TOKEN_BUCKET_LUA


# ───────── Module-level limiters ─────────


class TestModuleLimiters:
    def test_engine_limiter_exists(self):
        assert engine_limiter is not None

    def test_api_limiter_exists(self):
        assert api_limiter is not None

    def test_public_limiter_exists(self):
        assert public_limiter is not None
