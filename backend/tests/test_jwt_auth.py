"""
Tests for JWT session token support in CORE security module.

Sentinel-value contract:
- Remove create_jwt_token() → encode call missing → token absent → test fails
- Remove decode_jwt_token() → decode call missing → claims missing → test fails
- Remove expiry check → expired token accepted → test fails
- Remove issuer check → wrong-issuer token accepted → test fails
- Remove role mapping in get_jwt_bearer → require_jwt_role always 403 → test fails
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.security import (
    Role,
    JWTError,
    create_jwt_token,
    decode_jwt_token,
    get_jwt_bearer,
    require_jwt_role,
    _JWT_SECRET,
    _JWT_ALGORITHM,
    _JWT_ISSUER,
)


# ---------------------------------------------------------------------------
# create_jwt_token
# ---------------------------------------------------------------------------


class TestCreateJwtToken:
    def test_returns_string(self):
        """SENTINEL — create_jwt_token must return a non-empty string."""
        token = create_jwt_token("alice", Role.VIEWER)
        assert isinstance(token, str)
        assert len(token) > 0

    def test_token_has_three_parts(self):
        """JWT tokens must be header.payload.signature (3 dot-separated parts)."""
        token = create_jwt_token("alice", Role.VIEWER)
        assert token.count(".") == 2

    def test_subject_encoded(self):
        """
        SENTINEL — the subject must appear in the decoded payload.
        Remove sub from payload → missing → test fails.
        """
        token = create_jwt_token("sentinel-user", Role.ADMIN)
        payload = decode_jwt_token(token)
        assert payload["sub"] == "sentinel-user"

    def test_role_encoded(self):
        """
        SENTINEL — the role must appear in the decoded payload.
        Remove role from payload → missing → test fails.
        """
        token = create_jwt_token("alice", Role.AGENT)
        payload = decode_jwt_token(token)
        assert payload["role"] == Role.AGENT.value

    def test_issuer_encoded(self):
        """
        SENTINEL — the issuer must be 'core'.
        Change issuer → decode_jwt_token raises IssuerError → test fails.
        """
        token = create_jwt_token("alice", Role.VIEWER)
        payload = decode_jwt_token(token)
        assert payload["iss"] == _JWT_ISSUER

    def test_extra_claims_included(self):
        """Extra claims must be present in the payload."""
        token = create_jwt_token("alice", Role.VIEWER, extra_claims={"tenant": "acme"})
        payload = decode_jwt_token(token)
        assert payload["tenant"] == "acme"

    def test_expiry_is_in_future(self):
        """Token must expire in the future."""
        token = create_jwt_token("alice", Role.VIEWER, expires_minutes=30)
        payload = decode_jwt_token(token)
        exp = payload["exp"]
        assert exp > time.time()

    def test_custom_expiry_respected(self):
        """expires_minutes parameter must control expiry window."""
        token_short = create_jwt_token("alice", Role.VIEWER, expires_minutes=1)
        token_long = create_jwt_token("alice", Role.VIEWER, expires_minutes=120)
        exp_short = decode_jwt_token(token_short)["exp"]
        exp_long = decode_jwt_token(token_long)["exp"]
        # Long token expires at least 100 minutes later than short
        assert exp_long - exp_short > 100 * 60

    def test_different_roles_produce_different_payloads(self):
        token_admin = create_jwt_token("u", Role.ADMIN)
        token_viewer = create_jwt_token("u", Role.VIEWER)
        assert (
            decode_jwt_token(token_admin)["role"]
            != decode_jwt_token(token_viewer)["role"]
        )


# ---------------------------------------------------------------------------
# decode_jwt_token
# ---------------------------------------------------------------------------


class TestDecodeJwtToken:
    def test_valid_token_returns_payload(self):
        """SENTINEL — valid token must decode to payload with expected keys."""
        token = create_jwt_token("bob", Role.AGENT)
        payload = decode_jwt_token(token)
        assert "sub" in payload
        assert "role" in payload
        assert "exp" in payload
        assert "iss" in payload

    def test_tampered_token_raises_jwt_error(self):
        """
        SENTINEL — tokens with invalid signature must raise JWTError.
        Remove signature check → tampered tokens accepted → test fails.
        """
        token = create_jwt_token("bob", Role.ADMIN)
        # Flip a character in the signature (last segment)
        parts = token.split(".")
        parts[-1] = parts[-1][:-3] + "XXX"
        tampered = ".".join(parts)
        with pytest.raises(JWTError):
            decode_jwt_token(tampered)

    def test_expired_token_raises_jwt_error(self):
        """
        SENTINEL — tokens past their expiry must raise JWTError.
        Remove expiry claim → expired token accepted → test fails.
        """
        import jwt as _pyjwt

        now = datetime.now(tz=timezone.utc)
        payload = {
            "iss": _JWT_ISSUER,
            "sub": "bob",
            "role": Role.VIEWER.value,
            "iat": now,
            "exp": now,  # already expired (exp == iat == now)
        }
        token = _pyjwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)
        with pytest.raises(JWTError, match="expired"):
            decode_jwt_token(token)

    def test_wrong_issuer_raises_jwt_error(self):
        """
        SENTINEL — tokens from a different issuer must raise JWTError.
        Remove issuer check → any issuer accepted → test fails.
        """
        import jwt as _pyjwt
        from datetime import timedelta

        now = datetime.now(tz=timezone.utc)
        payload = {
            "iss": "evil-service",
            "sub": "bob",
            "role": Role.ADMIN.value,
            "iat": now,
            "exp": now + timedelta(minutes=60),
        }
        token = _pyjwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)
        with pytest.raises(JWTError, match="issuer"):
            decode_jwt_token(token)

    def test_garbage_string_raises_jwt_error(self):
        """Non-JWT strings must raise JWTError."""
        with pytest.raises(JWTError):
            decode_jwt_token("not.a.jwt")

    def test_wrong_key_raises_jwt_error(self):
        """Token signed with a different key must raise JWTError."""
        import jwt as _pyjwt
        from datetime import timedelta

        now = datetime.now(tz=timezone.utc)
        payload = {
            "iss": _JWT_ISSUER,
            "sub": "eve",
            "role": Role.VIEWER.value,
            "iat": now,
            "exp": now + timedelta(minutes=60),
        }
        wrong_key_token = _pyjwt.encode(
            payload, "wrong-secret", algorithm=_JWT_ALGORITHM
        )
        with pytest.raises(JWTError):
            decode_jwt_token(wrong_key_token)


# ---------------------------------------------------------------------------
# get_jwt_bearer FastAPI dependency
# ---------------------------------------------------------------------------


class TestGetJwtBearer:
    def _make_credentials(self, token: str):
        from fastapi.security import HTTPAuthorizationCredentials

        return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    @pytest.mark.asyncio
    async def test_valid_token_returns_key_data(self):
        """
        SENTINEL — valid Bearer token must produce key_data with name and permissions.
        Remove role mapping → permissions list empty → require_role always 403 → test fails.
        """
        token = create_jwt_token("charlie", Role.AGENT)
        creds = self._make_credentials(token)
        key_data = await get_jwt_bearer(creds)
        assert key_data["name"] == "charlie"
        assert "role:agent" in key_data["permissions"]
        assert key_data["role"] == Role.AGENT.value

    @pytest.mark.asyncio
    async def test_missing_credentials_raises_401(self):
        """
        SENTINEL — missing Bearer header must raise HTTP 401.
        Remove credentials check → None accepted → test fails.
        """
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await get_jwt_bearer(None)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_token_raises_401(self):
        """Invalid Bearer token must raise HTTP 401."""
        from fastapi import HTTPException

        creds = self._make_credentials("invalid.token.here")
        with pytest.raises(HTTPException) as exc_info:
            await get_jwt_bearer(creds)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_jwt_subject_in_key_data(self):
        """jwt_subject field must match the token's sub claim."""
        token = create_jwt_token("diana", Role.VIEWER)
        creds = self._make_credentials(token)
        key_data = await get_jwt_bearer(creds)
        assert key_data["jwt_subject"] == "diana"

    @pytest.mark.asyncio
    async def test_admin_role_in_permissions(self):
        """ADMIN token must have 'role:admin' in permissions list."""
        token = create_jwt_token("superuser", Role.ADMIN)
        creds = self._make_credentials(token)
        key_data = await get_jwt_bearer(creds)
        assert "role:admin" in key_data["permissions"]

    @pytest.mark.asyncio
    async def test_viewer_role_in_permissions(self):
        """VIEWER token must have 'role:viewer' in permissions list."""
        token = create_jwt_token("reader", Role.VIEWER)
        creds = self._make_credentials(token)
        key_data = await get_jwt_bearer(creds)
        assert "role:viewer" in key_data["permissions"]


# ---------------------------------------------------------------------------
# require_jwt_role dependency
# ---------------------------------------------------------------------------


class TestRequireJwtRole:
    def _make_credentials(self, token: str):
        from fastapi.security import HTTPAuthorizationCredentials

        return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    @pytest.mark.asyncio
    async def test_correct_role_allowed(self):
        """
        SENTINEL — token with matching role must pass require_jwt_role check.
        Remove role intersection check → always 403 → test fails.
        """
        from fastapi import HTTPException

        token = create_jwt_token("frank", Role.ADMIN)
        creds = self._make_credentials(token)
        dep = require_jwt_role(Role.ADMIN)
        # Should not raise
        key_data = await dep(await get_jwt_bearer(creds))
        assert key_data["role"] == Role.ADMIN.value

    @pytest.mark.asyncio
    async def test_higher_role_satisfies_lower_requirement(self):
        """
        SENTINEL — ADMIN token must satisfy AGENT-level require_jwt_role.
        Flatten hierarchy → ADMIN rejected for AGENT endpoints → test fails.
        """
        token = create_jwt_token("frank", Role.ADMIN)
        creds = self._make_credentials(token)
        dep = require_jwt_role(Role.AGENT)
        key_data = await dep(await get_jwt_bearer(creds))
        assert key_data is not None

    @pytest.mark.asyncio
    async def test_lower_role_rejected(self):
        """
        SENTINEL — VIEWER token must be rejected when ADMIN role is required.
        Allow all roles → security boundary broken → test fails.
        """
        from fastapi import HTTPException

        token = create_jwt_token("frank", Role.VIEWER)
        creds = self._make_credentials(token)
        dep = require_jwt_role(Role.ADMIN)
        with pytest.raises(HTTPException) as exc_info:
            await dep(await get_jwt_bearer(creds))
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_agent_rejected_for_admin_endpoint(self):
        """AGENT role must be rejected for ADMIN-only require_jwt_role."""
        from fastapi import HTTPException

        token = create_jwt_token("agent-user", Role.AGENT)
        creds = self._make_credentials(token)
        dep = require_jwt_role(Role.ADMIN)
        with pytest.raises(HTTPException) as exc_info:
            await dep(await get_jwt_bearer(creds))
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_viewer_passes_viewer_requirement(self):
        """VIEWER role must pass VIEWER-level requirement."""
        token = create_jwt_token("viewer-user", Role.VIEWER)
        creds = self._make_credentials(token)
        dep = require_jwt_role(Role.VIEWER)
        key_data = await dep(await get_jwt_bearer(creds))
        assert key_data is not None
