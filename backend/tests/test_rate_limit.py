"""
Tests for rate_limit middleware — token bucket rate limiter.

Covers:
- RateLimiter: init, allow, refill, burst, retry_after, reset
- Key extraction: get_client_ip, get_api_key
- check_rate_limit: pass-through and 429 rejection
- rate_limit decorator: wiring
"""

import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from app.middleware.rate_limit import (
    RateLimiter,
    get_client_ip,
    get_api_key,
    check_rate_limit,
    rate_limit,
    engine_limiter,
    api_limiter,
    public_limiter,
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

        # drain
        assert rl.allow("k") is True
        assert rl.allow("k") is True
        assert rl.allow("k") is False

        # advance 1 second — should refill 1 token
        mock_time.time.return_value = 1001.0
        assert rl.allow("k") is True
        assert rl.allow("k") is False

    @patch("app.middleware.rate_limit.time")
    def test_refill_capped_at_burst(self, mock_time):
        mock_time.time.return_value = 1000.0
        rl = RateLimiter(requests_per_minute=60, burst_size=3)

        # drain all 3
        for _ in range(3):
            rl.allow("k")

        # advance a long time — should cap at burst_size (3)
        mock_time.time.return_value = 2000.0
        assert rl.allow("k") is True
        assert rl.allow("k") is True
        assert rl.allow("k") is True
        assert rl.allow("k") is False


class TestRetryAfter:
    @patch("app.middleware.rate_limit.time")
    def test_retry_after_zero_when_allowed(self, mock_time):
        mock_time.time.return_value = 1000.0
        rl = RateLimiter(requests_per_minute=60, burst_size=5)
        assert rl.get_retry_after("k") == 0

    @patch("app.middleware.rate_limit.time")
    def test_retry_after_positive_when_exhausted(self, mock_time):
        mock_time.time.return_value = 1000.0
        rl = RateLimiter(requests_per_minute=60, burst_size=1)  # 1 tok/sec
        rl.allow("k")  # drain

        mock_time.time.return_value = 1000.0  # no advance
        retry = rl.get_retry_after("k")
        assert retry > 0
        assert retry <= 1.0  # needs 1 token at 1 tok/sec

    @patch("app.middleware.rate_limit.time")
    def test_retry_after_scales_with_cost(self, mock_time):
        mock_time.time.return_value = 1000.0
        rl = RateLimiter(requests_per_minute=60, burst_size=1)
        rl.allow("k")
        mock_time.time.return_value = 1000.0
        retry = rl.get_retry_after("k", cost=3)
        assert retry > 2.5  # needs 3 tokens at 1 tok/sec


class TestReset:
    def test_reset_restores_tokens(self):
        rl = RateLimiter(requests_per_minute=60, burst_size=2)
        rl.allow("k")
        rl.allow("k")
        assert rl.allow("k") is False
        rl.reset("k")
        assert rl.allow("k") is True

    def test_reset_unknown_key_safe(self):
        rl = RateLimiter(requests_per_minute=60, burst_size=5)
        rl.reset("nonexistent")  # should not raise


# ───────── Default limiter instances ─────────

class TestDefaultLimiters:
    def test_engine_limiter_settings(self):
        assert engine_limiter.burst_size == 5
        assert engine_limiter.rate == 10 / 60.0

    def test_api_limiter_settings(self):
        assert api_limiter.burst_size == 30
        assert api_limiter.rate == 60 / 60.0

    def test_public_limiter_settings(self):
        assert public_limiter.burst_size == 60
        assert public_limiter.rate == 120 / 60.0


# ───────── Key extractors ─────────

class TestGetClientIp:
    def test_from_client_host(self):
        req = MagicMock()
        req.headers = {}
        req.client.host = "10.0.0.1"
        assert get_client_ip(req) == "10.0.0.1"

    def test_x_forwarded_for_single(self):
        req = MagicMock()
        req.headers = {"X-Forwarded-For": "203.0.113.5"}
        assert get_client_ip(req) == "203.0.113.5"

    def test_x_forwarded_for_chain(self):
        req = MagicMock()
        req.headers = {"X-Forwarded-For": "203.0.113.5, 70.1.2.3, 127.0.0.1"}
        assert get_client_ip(req) == "203.0.113.5"

    def test_no_client(self):
        req = MagicMock()
        req.headers = {}
        req.client = None
        assert get_client_ip(req) == "unknown"


class TestGetApiKey:
    def test_returns_api_key_header(self):
        req = MagicMock()
        req.headers = {"X-API-Key": "secret-123"}
        assert get_api_key(req) == "secret-123"

    def test_falls_back_to_client_ip(self):
        req = MagicMock()
        req.headers = {}
        req.client.host = "192.168.1.1"
        assert get_api_key(req) == "192.168.1.1"


# ───────── check_rate_limit ─────────

class TestCheckRateLimit:
    @pytest.mark.asyncio
    async def test_allowed_request_passes(self):
        limiter = RateLimiter(requests_per_minute=60, burst_size=10)
        req = MagicMock()
        req.headers = {}
        req.client.host = "10.0.0.1"
        # should not raise
        await check_rate_limit(req, limiter)

    @pytest.mark.asyncio
    async def test_exhausted_raises_429(self):
        limiter = RateLimiter(requests_per_minute=60, burst_size=1)
        req = MagicMock()
        req.headers = {}
        req.client.host = "10.0.0.1"
        await check_rate_limit(req, limiter)  # use the 1 token

        with pytest.raises(HTTPException) as exc_info:
            await check_rate_limit(req, limiter)
        assert exc_info.value.status_code == 429
        assert "Retry-After" in exc_info.value.headers

    @pytest.mark.asyncio
    async def test_custom_key_func(self):
        limiter = RateLimiter(requests_per_minute=60, burst_size=1)
        req = MagicMock()
        # key func that always returns same key
        await check_rate_limit(req, limiter, key_func=lambda r: "shared")
        with pytest.raises(HTTPException):
            await check_rate_limit(req, limiter, key_func=lambda r: "shared")

    @pytest.mark.asyncio
    async def test_custom_cost(self):
        limiter = RateLimiter(requests_per_minute=60, burst_size=3)
        req = MagicMock()
        req.headers = {}
        req.client.host = "1.2.3.4"
        await check_rate_limit(req, limiter, cost=3)  # use all 3
        with pytest.raises(HTTPException):
            await check_rate_limit(req, limiter, cost=1)


# ───────── rate_limit decorator ─────────

class TestRateLimitDecorator:
    @pytest.mark.asyncio
    async def test_decorator_allows_call(self):
        limiter = RateLimiter(requests_per_minute=60, burst_size=5)
        inner = AsyncMock(return_value="ok")

        req = MagicMock()
        req.headers = {}
        req.client.host = "1.1.1.1"

        decorated = rate_limit(limiter)(inner)
        result = await decorated(request=req)
        assert result == "ok"
        inner.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_decorator_rejects_when_exhausted(self):
        limiter = RateLimiter(requests_per_minute=60, burst_size=1)
        inner = AsyncMock(return_value="ok")

        req = MagicMock()
        req.headers = {}
        req.client.host = "1.1.1.1"

        decorated = rate_limit(limiter)(inner)
        await decorated(request=req)  # uses the 1 token

        with pytest.raises(HTTPException) as exc_info:
            await decorated(request=req)
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_decorator_finds_request_in_args(self):
        limiter = RateLimiter(requests_per_minute=60, burst_size=5)
        inner = AsyncMock(return_value="ok")

        req = MagicMock(spec=["headers", "client"])
        req.headers = {}
        req.client.host = "2.2.2.2"

        # Simulate request passed as positional arg (no keyword)
        @rate_limit(limiter)
        async def handler(request):
            return "ok"

        # Patch to avoid calling the real inner
        with patch("app.middleware.rate_limit.check_rate_limit", new_callable=AsyncMock) as mock_check:
            result = await handler(request=req)
            mock_check.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_decorator_no_request_skips_check(self):
        limiter = RateLimiter(requests_per_minute=60, burst_size=1)

        @rate_limit(limiter)
        async def handler():
            return "no-request"

        # Should still work — just doesn't rate limit
        result = await handler()
        assert result == "no-request"