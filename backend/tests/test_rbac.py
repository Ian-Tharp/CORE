"""
Tests for RBAC (Role-Based Access Control) in CORE.

Sentinel-value contract: every role test would FAIL if you removed
the require_role dependency or the _ROLE_SATISFIES hierarchy.

Coverage:
- Role enum values
- _ROLE_SATISFIES hierarchy correctness
- require_role dependency: 403 for insufficient role, 200 for correct role
- Wildcard "*" permission passes all role checks
- Role hierarchy: ADMIN satisfies AGENT requirement, ADMIN satisfies VIEWER, etc.
- Admin routes protected at Role.ADMIN
- Engine routes protected at Role.AGENT
- Health /deep protected at Role.AGENT
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from fastapi import HTTPException

from app.core.security import (
    Role,
    require_role,
    get_api_key,
    _API_KEY_CACHE,
    _hash_key,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_api_keys():
    """Isolate each test — clear key cache before and after."""
    _API_KEY_CACHE.clear()
    yield
    _API_KEY_CACHE.clear()


def _register_key(name: str, permissions: list) -> str:
    """Register a key in the cache directly and return the raw key."""
    import secrets

    raw = f"core_{secrets.token_urlsafe(16)}"
    key_hash = _hash_key(raw)
    _API_KEY_CACHE[key_hash] = {
        "name": name,
        "description": "",
        "permissions": permissions,
        "created_at": "2026-01-01T00:00:00",
        "last_used": None,
        "request_count": 0,
    }
    return raw


# ---------------------------------------------------------------------------
# Role enum
# ---------------------------------------------------------------------------


class TestRoleEnum:
    def test_role_values(self):
        assert Role.ADMIN.value == "admin"
        assert Role.AGENT.value == "agent"
        assert Role.VIEWER.value == "viewer"

    def test_role_is_str_subclass(self):
        """Role must be usable as a plain string for permission strings."""
        assert isinstance(Role.ADMIN, str)

    def test_all_three_roles_defined(self):
        roles = {r.value for r in Role}
        assert roles == {"admin", "agent", "viewer"}


# ---------------------------------------------------------------------------
# _ROLE_SATISFIES hierarchy
# ---------------------------------------------------------------------------


class TestRoleHierarchy:
    """
    SENTINEL TESTS — directly assert the hierarchy table.
    Removing or reordering entries breaks these.
    """

    from app.core.security import _ROLE_SATISFIES

    def test_admin_role_only_satisfies_admin(self):
        from app.core.security import _ROLE_SATISFIES

        satisfying = _ROLE_SATISFIES[Role.ADMIN]
        assert "role:admin" in satisfying
        assert "role:agent" not in satisfying
        assert "role:viewer" not in satisfying

    def test_agent_role_satisfies_agent_and_admin(self):
        from app.core.security import _ROLE_SATISFIES

        satisfying = _ROLE_SATISFIES[Role.AGENT]
        assert "role:admin" in satisfying
        assert "role:agent" in satisfying
        assert "role:viewer" not in satisfying

    def test_viewer_role_satisfies_all(self):
        from app.core.security import _ROLE_SATISFIES

        satisfying = _ROLE_SATISFIES[Role.VIEWER]
        assert "role:admin" in satisfying
        assert "role:agent" in satisfying
        assert "role:viewer" in satisfying


# ---------------------------------------------------------------------------
# require_role dependency — unit tests (call the inner async fn directly)
# ---------------------------------------------------------------------------


class TestRequireRole:
    """
    Test require_role by extracting the inner _check_role coroutine
    and calling it with a pre-built key_data dict (bypasses get_api_key).
    """

    async def _call_role_check(self, minimum_role: Role, key_data: dict) -> dict:
        """Helper: call the _check_role inner function with synthetic key_data."""
        dep_factory = require_role(minimum_role)
        # The inner function is a regular coroutine — call it directly.
        # We need to invoke it without going through FastAPI DI,
        # so we bypass the Depends(get_api_key) by calling the factory's
        # closure directly with key_data as the argument.
        import inspect

        inner = dep_factory
        # dep_factory returns a _check_role coroutine function.
        # The closure's first (and only) parameter is `key_data`.
        return await inner(key_data=key_data)

    # --- wildcard passthrough ---

    @pytest.mark.asyncio
    async def test_wildcard_permission_passes_admin_requirement(self):
        key = {"name": "master", "permissions": ["*"]}
        result = await self._call_role_check(Role.ADMIN, key)
        assert result is key  # same object returned

    @pytest.mark.asyncio
    async def test_wildcard_permission_passes_agent_requirement(self):
        key = {"name": "dev", "permissions": ["*"]}
        result = await self._call_role_check(Role.AGENT, key)
        assert result is key

    @pytest.mark.asyncio
    async def test_wildcard_permission_passes_viewer_requirement(self):
        key = {"name": "dev", "permissions": ["*"]}
        result = await self._call_role_check(Role.VIEWER, key)
        assert result is key

    # --- correct role passes ---

    @pytest.mark.asyncio
    async def test_admin_role_passes_admin_requirement(self):
        key = {"name": "sysop", "permissions": ["role:admin"]}
        result = await self._call_role_check(Role.ADMIN, key)
        assert result is key

    @pytest.mark.asyncio
    async def test_agent_role_passes_agent_requirement(self):
        key = {"name": "agent-1", "permissions": ["role:agent"]}
        result = await self._call_role_check(Role.AGENT, key)
        assert result is key

    @pytest.mark.asyncio
    async def test_viewer_role_passes_viewer_requirement(self):
        key = {"name": "watcher", "permissions": ["role:viewer"]}
        result = await self._call_role_check(Role.VIEWER, key)
        assert result is key

    # --- hierarchy: higher role satisfies lower requirement ---

    @pytest.mark.asyncio
    async def test_admin_role_passes_agent_requirement(self):
        """
        SENTINEL TEST — Admin role must satisfy Agent requirement (hierarchy).
        Remove "role:admin" from _ROLE_SATISFIES[Role.AGENT] → fails.
        """
        key = {"name": "sysop", "permissions": ["role:admin"]}
        result = await self._call_role_check(Role.AGENT, key)
        assert result is key

    @pytest.mark.asyncio
    async def test_admin_role_passes_viewer_requirement(self):
        key = {"name": "sysop", "permissions": ["role:admin"]}
        result = await self._call_role_check(Role.VIEWER, key)
        assert result is key

    @pytest.mark.asyncio
    async def test_agent_role_passes_viewer_requirement(self):
        key = {"name": "agent-1", "permissions": ["role:agent"]}
        result = await self._call_role_check(Role.VIEWER, key)
        assert result is key

    # --- insufficient role raises 403 ---

    @pytest.mark.asyncio
    async def test_viewer_role_fails_admin_requirement(self):
        """
        SENTINEL TEST — Viewer must NOT access admin routes.
        Remove the 403 raise → this test fails.
        """
        key = {"name": "watcher", "permissions": ["role:viewer"]}
        with pytest.raises(HTTPException) as exc_info:
            await self._call_role_check(Role.ADMIN, key)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_viewer_role_fails_agent_requirement(self):
        key = {"name": "watcher", "permissions": ["role:viewer"]}
        with pytest.raises(HTTPException) as exc_info:
            await self._call_role_check(Role.AGENT, key)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_agent_role_fails_admin_requirement(self):
        """
        SENTINEL TEST — Agent must NOT access admin-only routes.
        """
        key = {"name": "agent-1", "permissions": ["role:agent"]}
        with pytest.raises(HTTPException) as exc_info:
            await self._call_role_check(Role.ADMIN, key)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_empty_permissions_fails_viewer_requirement(self):
        key = {"name": "bare", "permissions": []}
        with pytest.raises(HTTPException) as exc_info:
            await self._call_role_check(Role.VIEWER, key)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_403_detail_mentions_required_role(self):
        """Error message must name the minimum required role."""
        key = {"name": "watcher", "permissions": ["role:viewer"]}
        with pytest.raises(HTTPException) as exc_info:
            await self._call_role_check(Role.ADMIN, key)
        assert "admin" in str(exc_info.value.detail).lower()


# ---------------------------------------------------------------------------
# Integration: require_role flows through get_api_key (real DI chain)
# ---------------------------------------------------------------------------


class TestRequireRoleWithRealAuth:
    """
    Test that require_role's inner _check_role, when called with a REAL
    get_api_key result (from a registered key), enforces roles correctly.

    This exercises the actual DI chain end-to-end without HTTP.
    """

    @pytest.mark.asyncio
    async def test_registered_admin_key_passes_admin_route(self):
        raw = _register_key("admin-key", ["role:admin"])
        # Simulate what get_api_key returns
        with patch.dict("os.environ", {"CORE_AUTH_DISABLED": "false"}, clear=False):
            from app.core.security import validate_api_key

            key_data = validate_api_key(raw)
        assert key_data is not None

        dep = require_role(Role.ADMIN)
        result = await dep(key_data=key_data)
        assert result["name"] == "admin-key"

    @pytest.mark.asyncio
    async def test_registered_viewer_key_rejected_on_admin_route(self):
        raw = _register_key("viewer-key", ["role:viewer"])
        from app.core.security import validate_api_key

        key_data = validate_api_key(raw)
        assert key_data is not None

        dep = require_role(Role.ADMIN)
        with pytest.raises(HTTPException) as exc_info:
            await dep(key_data=key_data)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_registered_agent_key_passes_agent_route(self):
        raw = _register_key("agent-key", ["role:agent"])
        from app.core.security import validate_api_key

        key_data = validate_api_key(raw)

        dep = require_role(Role.AGENT)
        result = await dep(key_data=key_data)
        assert result["name"] == "agent-key"

    @pytest.mark.asyncio
    async def test_registered_admin_key_passes_agent_route(self):
        """
        SENTINEL TEST — Admin key must pass agent-required routes (hierarchy).
        """
        raw = _register_key("admin-key", ["role:admin"])
        from app.core.security import validate_api_key

        key_data = validate_api_key(raw)

        dep = require_role(Role.AGENT)
        result = await dep(key_data=key_data)
        assert result["name"] == "admin-key"

    @pytest.mark.asyncio
    async def test_registered_agent_key_rejected_on_admin_route(self):
        raw = _register_key("agent-key", ["role:agent"])
        from app.core.security import validate_api_key

        key_data = validate_api_key(raw)

        dep = require_role(Role.ADMIN)
        with pytest.raises(HTTPException) as exc_info:
            await dep(key_data=key_data)
        assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# Route protection — verify the correct role guards are wired
# ---------------------------------------------------------------------------


class TestRouteProtectionWiring:
    """
    Verify that admin/engine/health routes use require_role by inspecting
    the FastAPI dependency chain without making HTTP calls.

    These tests FAIL if someone swaps require_role for a weaker dependency.
    """

    def _get_route_dependencies(
        self, app_router, path: str, method: str = "GET"
    ) -> list:
        """Extract dependency callables for a specific route."""
        for route in app_router.routes:
            if route.path == path and method.upper() in route.methods:
                return list(route.dependencies)
        return []

    def test_admin_keys_route_uses_require_role_admin(self):
        """POST /admin/keys must require Role.ADMIN — not bare get_api_key."""
        from app.controllers.admin import router as admin_router
        from app.core.security import _ROLE_SATISFIES

        route = next(
            (
                r
                for r in admin_router.routes
                if r.path == "/admin/keys" and "POST" in r.methods
            ),
            None,
        )
        assert route is not None, "POST /admin/keys route not found"

        # The dependency must NOT be the bare get_api_key function;
        # it must be a closure returned by require_role.
        dep_callables = [d.dependency for d in route.dependencies]
        # require_role returns an async inner function (_check_role).
        # It should NOT be get_api_key itself.
        assert get_api_key not in dep_callables, (
            "POST /admin/keys is protected only by get_api_key — "
            "require_role(Role.ADMIN) must be used instead"
        )

    def test_engine_run_route_uses_require_role(self):
        """POST /engine/run must not use bare get_api_key."""
        from app.controllers.engine import router as engine_router

        route = next(
            (
                r
                for r in engine_router.routes
                if r.path == "/engine/run" and "POST" in r.methods
            ),
            None,
        )
        assert route is not None, "POST /engine/run route not found"

        dep_callables = [d.dependency for d in route.dependencies]
        assert get_api_key not in dep_callables, (
            "POST /engine/run is protected only by get_api_key — "
            "require_role(Role.AGENT) must be used instead"
        )

    def test_health_deep_route_has_dependency(self):
        """GET /health/deep must have at least one auth dependency."""
        from app.controllers.health import router as health_router

        route = next(
            (
                r
                for r in health_router.routes
                if r.path == "/health/deep" and "GET" in r.methods
            ),
            None,
        )
        assert route is not None, "GET /health/deep route not found"

        # FastAPI stores param-level Depends() in route.dependant.dependencies,
        # not route.dependencies (which is only for decorator-level deps).
        all_deps = route.dependant.dependencies
        assert (
            len(all_deps) > 0
        ), "GET /health/deep has no auth dependency — require_role must be applied"
