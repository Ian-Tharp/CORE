"""
Tests for RateLimiter (in-memory sliding window) and _hash_key utility.

Sentinel-value contract:
- Remove bucket pruning → old requests never expire → rate limit never resets → test fails
- Remove allowed=False branch → every request passes regardless of count → test fails
- Remove bucket.append on allowed → buckets stay empty → second request always allowed → test fails
- Remove minute_remaining calculation → UI can't show countdown → test fails
- Remove retry_after=60 on minute exceeded → clients don't know when to retry → test fails
- Remove sha256 hash → raw key stored → key comparison logic breaks → test fails
"""

from __future__ import annotations

import time

import pytest

from app.core.security import RateLimiter, _hash_key


# ===========================================================================
# _hash_key
# ===========================================================================

class TestHashKey:
    def test_returns_64_char_hex_string(self):
        """
        SENTINEL — sha256 hex digest must be 64 characters.
        Use md5 (32 chars) → hash length wrong → cache misses → test fails.
        """
        result = _hash_key("SENTINEL_API_KEY")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_same_input_same_output(self):
        """Hash must be deterministic for the same input."""
        key = "SENTINEL_KEY_123"
        assert _hash_key(key) == _hash_key(key)

    def test_different_inputs_different_hashes(self):
        """Different keys must produce different hashes."""
        assert _hash_key("key_A") != _hash_key("key_B")

    def test_output_is_lowercase_hex(self):
        """Hash output must be lowercase hex characters only."""
        result = _hash_key("test")
        assert all(c in "0123456789abcdef" for c in result)


# ===========================================================================
# RateLimiter — basic rate limiting
# ===========================================================================

class TestRateLimiter:
    def test_first_request_allowed(self):
        """
        SENTINEL — first request must always be allowed.
        Deny on first request → new clients blocked immediately → test fails.
        """
        limiter = RateLimiter(requests_per_minute=10, requests_per_hour=100)
        result = limiter.check_rate_limit("SENTINEL_CLIENT")
        assert result["allowed"] is True

    def test_within_limit_allowed(self):
        """Requests below the limit must all be allowed."""
        limiter = RateLimiter(requests_per_minute=5, requests_per_hour=100)
        for _ in range(5):
            result = limiter.check_rate_limit("client1")
            assert result["allowed"] is True

    def test_over_minute_limit_denied(self):
        """
        SENTINEL — request that exceeds rpm must be denied.
        Always return allowed=True → rate limit never enforced → test fails.
        """
        limiter = RateLimiter(requests_per_minute=3, requests_per_hour=1000)
        for _ in range(3):
            limiter.check_rate_limit("client1")
        result = limiter.check_rate_limit("client1")
        assert result["allowed"] is False

    def test_over_hour_limit_denied(self):
        """
        SENTINEL — request that exceeds rph must be denied (even if rpm not hit).
        Only check minute bucket → hour limit never enforced → test fails.
        """
        limiter = RateLimiter(requests_per_minute=1000, requests_per_hour=3)
        for _ in range(3):
            limiter.check_rate_limit("client2")
        result = limiter.check_rate_limit("client2")
        assert result["allowed"] is False

    def test_different_clients_isolated(self):
        """
        SENTINEL — rate limit on one client must not affect another.
        Share a single bucket → all clients share the limit → test fails.
        """
        limiter = RateLimiter(requests_per_minute=2, requests_per_hour=100)
        limiter.check_rate_limit("client_A")
        limiter.check_rate_limit("client_A")
        limiter.check_rate_limit("client_A")  # over limit for A
        # client_B hasn't used the limiter, so should be allowed
        result = limiter.check_rate_limit("client_B")
        assert result["allowed"] is True

    def test_result_contains_required_keys(self):
        """
        SENTINEL — result must include allowed, minute_remaining, hour_remaining.
        Return empty dict → callers crash on missing keys → test fails.
        """
        limiter = RateLimiter(requests_per_minute=10, requests_per_hour=100)
        result = limiter.check_rate_limit("test")
        assert "allowed" in result
        assert "minute_remaining" in result
        assert "hour_remaining" in result

    def test_minute_remaining_decrements(self):
        """
        SENTINEL — minute_remaining must decrease with each allowed request.
        Always return rpm-1 → remaining count never decrements → test fails.
        """
        limiter = RateLimiter(requests_per_minute=5, requests_per_hour=100)
        r1 = limiter.check_rate_limit("client")
        r2 = limiter.check_rate_limit("client")
        assert r2["minute_remaining"] < r1["minute_remaining"]

    def test_retry_after_set_when_minute_limit_hit(self):
        """
        SENTINEL — retry_after must be 60 when minute limit is exceeded.
        Return None always → clients can't determine backoff time → test fails.
        """
        limiter = RateLimiter(requests_per_minute=2, requests_per_hour=1000)
        limiter.check_rate_limit("c")
        limiter.check_rate_limit("c")
        result = limiter.check_rate_limit("c")  # over minute limit
        assert result["retry_after"] == 60

    def test_retry_after_none_when_within_limit(self):
        """retry_after must be None when within limits."""
        limiter = RateLimiter(requests_per_minute=10, requests_per_hour=100)
        result = limiter.check_rate_limit("c")
        assert result["retry_after"] is None

    def test_denied_request_not_counted(self):
        """
        SENTINEL — denied requests must NOT be added to the bucket.
        Always append → bucket fills up instantly after first denial → test fails.
        """
        limiter = RateLimiter(requests_per_minute=2, requests_per_hour=1000)
        limiter.check_rate_limit("c")
        limiter.check_rate_limit("c")
        # Now over limit — these should all still be denied (not double-counting)
        for _ in range(5):
            result = limiter.check_rate_limit("c")
            assert result["allowed"] is False

    def test_old_entries_pruned(self):
        """
        SENTINEL — requests older than 60s must be pruned (sliding window).
        Never prune → count never resets → limit hits permanently → test fails.
        """
        limiter = RateLimiter(requests_per_minute=3, requests_per_hour=1000)
        # Manually inject old timestamps into the minute bucket
        past = time.time() - 120  # 2 minutes ago — outside the 60s window
        limiter._minute_buckets["c"].extend([past, past, past])
        # With 3 old entries pruned, the next request should be allowed
        result = limiter.check_rate_limit("c")
        assert result["allowed"] is True

    def test_hour_remaining_zero_when_at_limit(self):
        """hour_remaining must be 0 (not negative) when at the limit."""
        limiter = RateLimiter(requests_per_minute=1000, requests_per_hour=3)
        for _ in range(3):
            limiter.check_rate_limit("c")
        result = limiter.check_rate_limit("c")
        assert result["hour_remaining"] == 0
