"""
Comprehensive unit tests for app.core.security module.

Tests cover:
- API key generation, validation, revocation, and listing
- Master key authentication
- Rate limiter (per-minute and per-hour windows)
- FastAPI dependencies (get_api_key, check_rate_limit)
- Permission decorator (require_permission)
- Security initialization (init_security)
"""

import os
import time
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from app.core.security import (
    generate_api_key,
    validate_api_key,
    revoke_api_key,
    list_api_keys,
    RateLimiter,
    get_api_key,
    check_rate_limit,
    require_permission,
    init_security,
    _API_KEYS,
    _hash_key,
    _get_master_api_key,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_api_keys():
    """Clear the in-memory key store before and after each test."""
    _API_KEYS.clear()
    yield
    _API_KEYS.clear()


@pytest.fixture
def rate_limiter():
    """Fresh rate limiter with small limits for testing."""
    return RateLimiter(requests_per_minute=3, requests_per_hour=10)


# ===================================================================
# API Key Generation
# ===================================================================

class TestGenerateApiKey:
    def test_returns_string_starting_with_core_prefix(self):
        key = generate_api_key("test-key")
        assert key.startswith("core_")

    def test_key_is_long_enough(self):
        key = generate_api_key("test-key")
        # "core_" + 32-byte urlsafe-b64 (~43 chars)
        assert len(key) > 40

    def test_stores_metadata_in_memory(self):
        generate_api_key("my-agent", description="Agent key", permissions=["read"])
        assert len(_API_KEYS) == 1
        stored = list(_API_KEYS.values())[0]
        assert stored["name"] == "my-agent"
        assert stored["description"] == "Agent key"
        assert stored["permissions"] == ["read"]
        assert stored["last_used"] is None
        assert stored["request_count"] == 0

    def test_default_permissions_is_wildcard(self):
        generate_api_key("wildcard-key")
        stored = list(_API_KEYS.values())[0]
        assert stored["permissions"] == ["*"]

    def test_two_keys_are_unique(self):
        k1 = generate_api_key("key-a")
        k2 = generate_api_key("key-b")
        assert k1 != k2

    def test_key_hash_stored_not_raw(self):
        key = generate_api_key("secure")
        assert key not in _API_KEYS, "Raw key must not be a dict key"
        assert _hash_key(key) in _API_KEYS


# ===================================================================
# API Key Validation
# ===================================================================

class TestValidateApiKey:
    def test_valid_key_returns_metadata(self):
        key = generate_api_key("valid")
        result = validate_api_key(key)
        assert result is not None
        assert result["name"] == "valid"

    def test_invalid_key_returns_none(self):
        assert validate_api_key("core_totallyinvalid") is None

    def test_validation_updates_last_used(self):
        key = generate_api_key("track")
        validate_api_key(key)
        stored = _API_KEYS[_hash_key(key)]
        assert stored["last_used"] is not None

    def test_validation_increments_request_count(self):
        key = generate_api_key("counter")
        for _ in range(5):
            validate_api_key(key)
        stored = _API_KEYS[_hash_key(key)]
        assert stored["request_count"] == 5

    @patch.dict(os.environ, {"CORE_API_KEY": "master-secret-123"})
    def test_master_key_valid(self):
        result = validate_api_key("master-secret-123")
        assert result is not None
        assert result["name"] == "master"
        assert result["is_master"] is True
        assert "*" in result["permissions"]

    @patch.dict(os.environ, {"CORE_API_KEY": "master-secret-123"})
    def test_master_key_wrong_value(self):
        assert validate_api_key("wrong-master") is None

    def test_no_master_key_env_still_checks_registered(self):
        key = generate_api_key("fallback")
        with patch.dict(os.environ, {}, clear=True):
            result = validate_api_key(key)
            assert result is not None


# ===================================================================
# API Key Revocation
# ===================================================================

class TestRevokeApiKey:
    def test_revoke_existing_key(self):
        generate_api_key("doomed")
        assert revoke_api_key("doomed") is True
        assert len(_API_KEYS) == 0

    def test_revoke_nonexistent_key(self):
        assert revoke_api_key("ghost") is False

    def test_revoked_key_no_longer_validates(self):
        key = generate_api_key("temp")
        revoke_api_key("temp")
        assert validate_api_key(key) is None

    def test_revoke_only_target_key(self):
        generate_api_key("keep")
        generate_api_key("remove")
        revoke_api_key("remove")
        assert len(_API_KEYS) == 1
        remaining = list(_API_KEYS.values())[0]
        assert remaining["name"] == "keep"


# ===================================================================
# List API Keys
# ===================================================================

class TestListApiKeys:
    def test_empty_list(self):
        assert list_api_keys() == []

    def test_lists_all_keys(self):
        generate_api_key("alpha")
        generate_api_key("beta")
        keys = list_api_keys()
        assert len(keys) == 2
        names = {k["name"] for k in keys}
        assert names == {"alpha", "beta"}

    def test_does_not_expose_raw_key(self):
        raw = generate_api_key("secret")
        keys = list_api_keys()
        # Ensure raw key string doesn't appear in listing
        listing_str = str(keys)
        assert raw not in listing_str

    def test_listing_contains_expected_fields(self):
        generate_api_key("fields-check", description="desc", permissions=["read"])
        entry = list_api_keys()[0]
        assert "name" in entry
        assert "description" in entry
        assert "permissions" in entry
        assert "created_at" in entry
        assert "last_used" in entry
        assert "request_count" in entry


# ===================================================================
# Hash Key
# ===================================================================

class TestHashKey:
    def test_deterministic(self):
        assert _hash_key("abc") == _hash_key("abc")

    def test_different_inputs_different_hashes(self):
        assert _hash_key("key1") != _hash_key("key2")

    def test_returns_hex_string(self):
        h = _hash_key("test")
        assert len(h) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in h)


# ===================================================================
# Rate Limiter
# ===================================================================

class TestRateLimiter:
    def test_allows_within_limit(self, rate_limiter):
        result = rate_limiter.check_rate_limit("client-a")
        assert result["allowed"] is True

    def test_blocks_after_exceeding_rpm(self, rate_limiter):
        for _ in range(3):
            rate_limiter.check_rate_limit("client-b")
        result = rate_limiter.check_rate_limit("client-b")
        assert result["allowed"] is False
        assert result["retry_after"] == 60

    def test_separate_clients_independent(self, rate_limiter):
        for _ in range(3):
            rate_limiter.check_rate_limit("client-c")
        # client-c blocked
        assert rate_limiter.check_rate_limit("client-c")["allowed"] is False
        # client-d fine
        assert rate_limiter.check_rate_limit("client-d")["allowed"] is True

    def test_remaining_decrements(self, rate_limiter):
        r1 = rate_limiter.check_rate_limit("client-e")
        r2 = rate_limiter.check_rate_limit("client-e")
        assert r1["minute_remaining"] > r2["minute_remaining"]

    def test_old_entries_cleaned(self, rate_limiter):
        # Manually inject old timestamps
        rate_limiter._minute_buckets["client-f"] = [time.time() - 120]
        result = rate_limiter.check_rate_limit("client-f")
        assert result["allowed"] is True

    def test_hour_limit_enforcement(self):
        limiter = RateLimiter(requests_per_minute=100, requests_per_hour=5)
        for _ in range(5):
            limiter.check_rate_limit("hourly")
        result = limiter.check_rate_limit("hourly")
        assert result["allowed"] is False

    def test_default_limits(self):
        limiter = RateLimiter()
        assert limiter.rpm == 60
        assert limiter.rph == 1000


# ===================================================================
# FastAPI Dependency: get_api_key
# ===================================================================

class TestGetApiKeyDependency:
    @pytest.mark.asyncio
    @patch.dict(os.environ, {"CORE_AUTH_DISABLED": "true"})
    async def test_auth_disabled_returns_dev_key(self):
        result = await get_api_key(api_key_header=None, api_key_query=None)
        assert result["name"] == "development"
        assert result["auth_disabled"] is True

    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=False)
    async def test_no_key_raises_401(self):
        # Ensure auth is NOT disabled
        os.environ.pop("CORE_AUTH_DISABLED", None)
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await get_api_key(api_key_header=None, api_key_query=None)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_key_raises_401(self):
        os.environ.pop("CORE_AUTH_DISABLED", None)
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await get_api_key(api_key_header="bad-key", api_key_query=None)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_header_key(self):
        os.environ.pop("CORE_AUTH_DISABLED", None)
        key = generate_api_key("header-test")
        result = await get_api_key(api_key_header=key, api_key_query=None)
        assert result["name"] == "header-test"

    @pytest.mark.asyncio
    async def test_valid_query_key(self):
        os.environ.pop("CORE_AUTH_DISABLED", None)
        key = generate_api_key("query-test")
        result = await get_api_key(api_key_header=None, api_key_query=key)
        assert result["name"] == "query-test"

    @pytest.mark.asyncio
    async def test_header_preferred_over_query(self):
        os.environ.pop("CORE_AUTH_DISABLED", None)
        key_h = generate_api_key("from-header")
        key_q = generate_api_key("from-query")
        result = await get_api_key(api_key_header=key_h, api_key_query=key_q)
        assert result["name"] == "from-header"


# ===================================================================
# FastAPI Dependency: check_rate_limit
# ===================================================================

class TestCheckRateLimitDependency:
    @pytest.mark.asyncio
    @patch.dict(os.environ, {"CORE_RATE_LIMIT_DISABLED": "true"})
    async def test_rate_limit_disabled(self):
        request = MagicMock()
        result = await check_rate_limit(request)
        assert result["allowed"] is True
        assert result["rate_limit_disabled"] is True

    @pytest.mark.asyncio
    async def test_rate_limit_allowed(self):
        os.environ.pop("CORE_RATE_LIMIT_DISABLED", None)
        request = MagicMock()
        request.client.host = "192.168.1.1"
        result = await check_rate_limit(request)
        assert result["allowed"] is True

    @pytest.mark.asyncio
    async def test_no_client_uses_unknown(self):
        os.environ.pop("CORE_RATE_LIMIT_DISABLED", None)
        request = MagicMock()
        request.client = None
        result = await check_rate_limit(request)
        assert result["allowed"] is True


# ===================================================================
# Permission Decorator
# ===================================================================

class TestRequirePermission:
    @pytest.mark.asyncio
    async def test_wildcard_permission_allows(self):
        @require_permission("admin:delete")
        async def protected(api_key=None):
            return "ok"

        result = await protected(api_key={"permissions": ["*"]})
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_specific_permission_allows(self):
        @require_permission("read:data")
        async def protected(api_key=None):
            return "ok"

        result = await protected(api_key={"permissions": ["read:data", "write:data"]})
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_missing_permission_raises_403(self):
        @require_permission("admin:nuke")
        async def protected(api_key=None):
            return "ok"

        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await protected(api_key={"permissions": ["read:only"]})
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_no_api_key_raises_401(self):
        @require_permission("any")
        async def protected(api_key=None):
            return "ok"

        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await protected(api_key=None)
        assert exc_info.value.status_code == 401


# ===================================================================
# Master Key Helper
# ===================================================================

class TestGetMasterApiKey:
    @patch.dict(os.environ, {"CORE_API_KEY": "my-master"})
    def test_returns_env_value(self):
        assert _get_master_api_key() == "my-master"

    def test_returns_none_when_unset(self):
        os.environ.pop("CORE_API_KEY", None)
        assert _get_master_api_key() is None


# ===================================================================
# Init Security
# ===================================================================

class TestInitSecurity:
    @patch.dict(os.environ, {"CORE_ENV": "development"})
    def test_dev_mode_generates_default_key(self):
        init_security()
        keys = list_api_keys()
        assert len(keys) == 1
        assert keys[0]["name"] == "core-ui-dev"

    @patch.dict(os.environ, {"CORE_ENV": "development"})
    def test_dev_mode_idempotent(self):
        init_security()
        init_security()
        keys = list_api_keys()
        # Second call sees keys exist, skips generation
        assert len(keys) == 1

    @patch.dict(os.environ, {"CORE_ENV": "production"})
    def test_production_no_default_key(self):
        init_security()
        assert list_api_keys() == []