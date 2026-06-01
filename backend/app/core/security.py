"""
CORE Security Module - API Key Authentication, Rate Limiting, and RBAC

Provides:
- API key validation and management (database-backed with in-memory cache)
- Request rate limiting
- Role-Based Access Control (RBAC)
- JWT session tokens (create_jwt_token / decode_jwt_token / get_jwt_bearer)

DONE: JWT support for user sessions (HS256 signed, configurable expiry)
"""

from __future__ import annotations

import asyncio
import os
import secrets
import hashlib
import time
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional, Dict, Any, Callable
from collections import defaultdict
from functools import wraps

import jwt as _pyjwt

from fastapi import Depends, HTTPException, Security, Request, status
from fastapi.security import (
    APIKeyHeader,
    APIKeyQuery,
    HTTPBearer,
    HTTPAuthorizationCredentials,
)
import logging

logger = logging.getLogger(__name__)

# API key extraction strategies
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
API_KEY_QUERY = APIKeyQuery(name="api_key", auto_error=False)

# In-memory cache for API keys (loaded from database on startup + updated on mutations)
_API_KEY_CACHE: Dict[str, Dict[str, Any]] = {}

# Flag to track whether the DB-backed store has been loaded
_db_loaded: bool = False

# Rate limiting storage
_RATE_LIMIT_BUCKETS: Dict[str, Dict[str, Any]] = defaultdict(
    lambda: {"count": 0, "window_start": time.time()}
)


def _get_master_api_key() -> Optional[str]:
    """Get the master API key from environment."""
    return os.getenv("CORE_API_KEY")


def _hash_key(key: str) -> str:
    """Hash an API key for storage comparison."""
    return hashlib.sha256(key.encode()).hexdigest()


async def load_keys_from_db() -> None:
    """
    Load all active API keys from the database into the in-memory cache.
    Called at startup after ensure_api_key_tables().
    """
    global _db_loaded
    try:
        from app.repository import api_key_repository

        keys = await api_key_repository.list_all_active()
        _API_KEY_CACHE.clear()
        for key_data in keys:
            _API_KEY_CACHE[key_data["key_hash"]] = {
                "name": key_data["name"],
                "description": key_data["description"],
                "permissions": key_data["permissions"],
                "created_at": key_data["created_at"],
                "last_used": key_data["last_used"],
                "request_count": key_data["request_count"],
            }
        _db_loaded = True
        logger.info("Loaded %d API keys from database into cache", len(_API_KEY_CACHE))
    except Exception as exc:
        logger.warning(
            "Could not load API keys from database (falling back to memory-only): %s",
            exc,
        )


async def generate_api_key(
    name: str, description: str = "", permissions: list = None
) -> str:
    """
    Generate a new API key and persist it to the database.

    Args:
        name: Identifier for the key (e.g., "vigil-agent", "web-ui")
        description: Optional description
        permissions: List of permitted operations

    Returns:
        The raw API key (store this securely - it won't be shown again)
    """
    raw_key = f"core_{secrets.token_urlsafe(32)}"
    key_hash = _hash_key(raw_key)
    perms = permissions or ["*"]

    # Store in cache immediately
    _API_KEY_CACHE[key_hash] = {
        "name": name,
        "description": description,
        "permissions": perms,
        "created_at": datetime.utcnow().isoformat(),
        "last_used": None,
        "request_count": 0,
    }

    # Persist to database (best-effort; cache is authoritative for this request)
    try:
        from app.repository import api_key_repository

        await api_key_repository.store_key(
            key_hash=key_hash,
            name=name,
            description=description,
            permissions=perms,
        )
    except Exception as exc:
        logger.error("Failed to persist API key to database: %s", exc)

    logger.info("Generated API key for: %s", name)
    return raw_key


def generate_api_key_sync(
    name: str, description: str = "", permissions: list = None
) -> str:
    """
    Synchronous API key generation (cache-only, no DB persistence).

    Used during init_security() before the event loop is fully running.
    Keys generated here will be in-memory only until next startup
    when they can be regenerated.
    """
    raw_key = f"core_{secrets.token_urlsafe(32)}"
    key_hash = _hash_key(raw_key)
    perms = permissions or ["*"]

    _API_KEY_CACHE[key_hash] = {
        "name": name,
        "description": description,
        "permissions": perms,
        "created_at": datetime.utcnow().isoformat(),
        "last_used": None,
        "request_count": 0,
    }

    logger.info("Generated API key (cache-only) for: %s", name)
    return raw_key


def validate_api_key(key: str) -> Optional[Dict[str, Any]]:
    """
    Validate an API key and return its metadata.

    Args:
        key: The raw API key

    Returns:
        Key metadata if valid, None if invalid
    """
    # Check master key first
    master_key = _get_master_api_key()
    if master_key and key == master_key:
        return {
            "name": "master",
            "permissions": ["*"],
            "is_master": True,
        }

    # Check cache
    key_hash = _hash_key(key)
    if key_hash in _API_KEY_CACHE:
        key_data = _API_KEY_CACHE[key_hash]
        key_data["last_used"] = datetime.utcnow().isoformat()
        key_data["request_count"] += 1

        # Fire-and-forget DB update (non-blocking)
        try:
            loop = asyncio.get_running_loop()
            from app.repository import api_key_repository

            loop.create_task(api_key_repository.update_last_used(key_hash))
        except RuntimeError:
            pass  # No running loop (e.g. in tests)

        return key_data

    return None


async def revoke_api_key(name: str) -> bool:
    """
    Revoke an API key by name.

    Args:
        name: The key name to revoke

    Returns:
        True if key was found and revoked
    """
    found = False
    for key_hash, data in list(_API_KEY_CACHE.items()):
        if data["name"] == name:
            del _API_KEY_CACHE[key_hash]
            found = True
            break

    # Also deactivate in database
    try:
        from app.repository import api_key_repository

        db_revoked = await api_key_repository.deactivate_by_name(name)
        if db_revoked and not found:
            found = True
    except Exception as exc:
        logger.error("Failed to deactivate API key in database: %s", exc)

    if found:
        logger.info("Revoked API key: %s", name)
    return found


def list_api_keys() -> list:
    """List all registered API keys (without exposing the actual keys)."""
    return [
        {
            "name": data["name"],
            "description": data["description"],
            "permissions": data["permissions"],
            "created_at": data["created_at"],
            "last_used": data["last_used"],
            "request_count": data["request_count"],
        }
        for data in _API_KEY_CACHE.values()
    ]


# =============================================================================
# Rate Limiting
# =============================================================================


class RateLimiter:
    """
    In-process sliding window rate limiter.

    Suitable for single-instance deployments. For distributed (multi-process)
    rate limiting, see RedisRateLimiter below.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
    ):
        self.rpm = requests_per_minute
        self.rph = requests_per_hour
        self._minute_buckets: Dict[str, list] = defaultdict(list)
        self._hour_buckets: Dict[str, list] = defaultdict(list)

    def check_rate_limit(self, client_id: str) -> Dict[str, Any]:
        """
        Check if client is within rate limits.

        Args:
            client_id: Identifier for the client (IP, API key name, etc.)

        Returns:
            Dict with allowed status and limit info
        """
        now = time.time()
        minute_ago = now - 60
        hour_ago = now - 3600

        # Clean old entries
        self._minute_buckets[client_id] = [
            t for t in self._minute_buckets[client_id] if t > minute_ago
        ]
        self._hour_buckets[client_id] = [
            t for t in self._hour_buckets[client_id] if t > hour_ago
        ]

        minute_count = len(self._minute_buckets[client_id])
        hour_count = len(self._hour_buckets[client_id])

        allowed = minute_count < self.rpm and hour_count < self.rph

        if allowed:
            self._minute_buckets[client_id].append(now)
            self._hour_buckets[client_id].append(now)

        return {
            "allowed": allowed,
            "minute_remaining": max(0, self.rpm - minute_count - 1),
            "hour_remaining": max(0, self.rph - hour_count - 1),
            "retry_after": 60 if minute_count >= self.rpm else None,
        }


class RedisRateLimiter:
    """
    Redis-backed distributed rate limiter using INCR + EXPIRE (fixed window).

    Each client gets two Redis counters:
        core:ratelimit:{client_id}:minute  — expires in 60 s
        core:ratelimit:{client_id}:hour    — expires in 3600 s

    On first request the key is created and an expiry is set atomically.
    Falls back to the in-memory RateLimiter when Redis is unreachable.

    Usage:
        limiter = RedisRateLimiter(redis_url="redis://localhost:6379/0")
        result  = await limiter.check_rate_limit("192.168.1.1")
    """

    _KEY_PREFIX = "core:ratelimit"

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
    ):
        self.redis_url = redis_url
        self.rpm = requests_per_minute
        self.rph = requests_per_hour
        self._fallback = (
            RateLimiter(requests_per_minute=rpm, requests_per_hour=rph)
            if False
            else RateLimiter(
                requests_per_minute=requests_per_minute,
                requests_per_hour=requests_per_hour,
            )
        )
        self._client: Optional[Any] = None  # redis.asyncio.Redis

    async def _get_client(self) -> Any:
        """Return (or create) the Redis client."""
        if self._client is None:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(self.redis_url, decode_responses=True)
        return self._client

    async def check_rate_limit(self, client_id: str) -> Dict[str, Any]:
        """
        Increment Redis counters and check against configured limits.

        Returns the same shape as RateLimiter.check_rate_limit().
        Falls back to in-memory rate limiting on any Redis error.
        """
        try:
            redis = await self._get_client()
            minute_key = f"{self._KEY_PREFIX}:{client_id}:minute"
            hour_key = f"{self._KEY_PREFIX}:{client_id}:hour"

            # Increment both counters; INCR creates the key at 0 and returns 1 on first call
            pipe = redis.pipeline(transaction=False)
            pipe.incr(minute_key)
            pipe.incr(hour_key)
            minute_count, hour_count = await pipe.execute()

            # Set TTL only on first access (count == 1) — avoids resetting the window
            if minute_count == 1:
                await redis.expire(minute_key, 60)
            if hour_count == 1:
                await redis.expire(hour_key, 3600)

            allowed = minute_count <= self.rpm and hour_count <= self.rph

            if not allowed:
                # Roll back the increment so the counter stays accurate
                rollback_pipe = redis.pipeline(transaction=False)
                rollback_pipe.decr(minute_key)
                rollback_pipe.decr(hour_key)
                await rollback_pipe.execute()

            return {
                "allowed": allowed,
                "minute_remaining": max(0, self.rpm - minute_count),
                "hour_remaining": max(0, self.rph - hour_count),
                "retry_after": 60 if minute_count > self.rpm else None,
                "backend": "redis",
            }

        except Exception as exc:
            logger.warning(
                "Redis rate limiter unavailable, using in-memory fallback: %s", exc
            )
            result = self._fallback.check_rate_limit(client_id)
            result["backend"] = "memory_fallback"
            return result


# Global rate limiter — uses Redis if REDIS_URL env var is set, otherwise in-memory
_redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_rate_limiter: Any = (
    RedisRateLimiter(redis_url=_redis_url) if os.getenv("REDIS_URL") else RateLimiter()
)


# =============================================================================
# FastAPI Dependencies
# =============================================================================


async def get_api_key(
    api_key_header: str = Security(API_KEY_HEADER),
    api_key_query: str = Security(API_KEY_QUERY),
) -> Dict[str, Any]:
    """
    FastAPI dependency for API key authentication.

    Checks header first, then query parameter.
    Raises 401 if no valid key found.
    """
    # Check if auth is disabled (development mode)
    if os.getenv("CORE_AUTH_DISABLED", "").lower() == "true":
        return {"name": "development", "permissions": ["*"], "auth_disabled": True}

    # Try header first
    key = api_key_header or api_key_query

    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Provide via X-API-Key header or api_key query parameter.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    key_data = validate_api_key(key)

    if not key_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return key_data


async def check_rate_limit(request: Request) -> Dict[str, Any]:
    """
    FastAPI dependency for rate limiting.

    Uses client IP as identifier (or API key name if authenticated).
    """
    # Check if rate limiting is disabled
    if os.getenv("CORE_RATE_LIMIT_DISABLED", "").lower() == "true":
        return {"allowed": True, "rate_limit_disabled": True}

    client_id = request.client.host if request.client else "unknown"

    if asyncio.iscoroutinefunction(_rate_limiter.check_rate_limit):
        result = await _rate_limiter.check_rate_limit(client_id)
    else:
        result = _rate_limiter.check_rate_limit(client_id)

    if not result["allowed"]:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "Retry-After": str(result["retry_after"]),
                "X-RateLimit-Remaining": "0",
            },
        )

    return result


def require_permission(permission: str):
    """
    Decorator to require specific permission.

    Usage:
        @router.get("/admin/keys")
        @require_permission("admin:keys")
        async def list_keys(api_key: dict = Depends(get_api_key)):
            ...
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, api_key: dict = None, **kwargs):
            if not api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            permissions = api_key.get("permissions", [])

            # Wildcard permission grants all access
            if "*" in permissions:
                return await func(*args, api_key=api_key, **kwargs)

            # Check specific permission
            if permission not in permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission '{permission}' required",
                )

            return await func(*args, api_key=api_key, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Role-Based Access Control (RBAC)
# =============================================================================


class Role(str, Enum):
    """
    CORE role hierarchy.

    ADMIN  > AGENT > VIEWER

    Roles are stored as permission strings on API keys (e.g. "role:admin").
    Keys with the wildcard permission "*" pass all role checks.
    When checking a role, all higher roles also pass:
        require_role(AGENT) accepts keys with role:admin OR role:agent.
        require_role(ADMIN) accepts only role:admin (and "*").
    """

    ADMIN = "admin"
    AGENT = "agent"
    VIEWER = "viewer"


# Permission string format stored in API key permissions list
_ROLE_PERMISSION = {
    Role.ADMIN: "role:admin",
    Role.AGENT: "role:agent",
    Role.VIEWER: "role:viewer",
}

# Roles that satisfy a given minimum role requirement (inclusive hierarchy)
_ROLE_SATISFIES: Dict[Role, set] = {
    Role.VIEWER: {"role:admin", "role:agent", "role:viewer"},
    Role.AGENT: {"role:admin", "role:agent"},
    Role.ADMIN: {"role:admin"},
}


def require_role(minimum_role: Role) -> Callable:
    """
    FastAPI dependency factory for role-based access control.

    Returns a dependency that:
    1. Authenticates via get_api_key (inherits all existing auth logic)
    2. Checks that the key's permissions satisfy the minimum_role
    3. Raises HTTP 403 if the role requirement is not met
    4. Returns the key metadata dict on success (same as get_api_key)

    Usage:
        @router.post("/admin/keys")
        async def create_key(key_data: dict = Depends(require_role(Role.ADMIN))):
            ...

    Role hierarchy (each role also accepts higher roles):
        require_role(Role.VIEWER) — accepts VIEWER, AGENT, ADMIN, and "*"
        require_role(Role.AGENT)  — accepts AGENT, ADMIN, and "*"
        require_role(Role.ADMIN)  — accepts only ADMIN and "*"
    """
    satisfying_perms = _ROLE_SATISFIES[minimum_role]

    async def _check_role(
        key_data: Dict[str, Any] = Depends(get_api_key),
    ) -> Dict[str, Any]:
        permissions = key_data.get("permissions", [])

        # Wildcard grants all roles
        if "*" in permissions:
            return key_data

        # Check if any permission satisfies the required role
        if satisfying_perms.intersection(permissions):
            return key_data

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Insufficient permissions. Role '{minimum_role.value}' or higher required. "
                f"Current key permissions: {permissions}"
            ),
        )

    return _check_role


# =============================================================================
# JWT Session Tokens
# =============================================================================

# Secret key for signing JWTs. Must be set in production via CORE_JWT_SECRET.
# Falls back to a random value per process (tokens are session-scoped only).
_JWT_SECRET: str = os.getenv("CORE_JWT_SECRET", secrets.token_hex(32))
_JWT_ALGORITHM = "HS256"
_JWT_DEFAULT_EXPIRY_MINUTES = int(os.getenv("CORE_JWT_EXPIRY_MINUTES", "60"))
_JWT_ISSUER = "core"

# FastAPI bearer-token extractor (auto_error=False so we can produce a clean 401)
_bearer_scheme = HTTPBearer(auto_error=False)


class JWTError(Exception):
    """Raised when a JWT cannot be validated."""


def create_jwt_token(
    subject: str,
    role: "Role",
    expires_minutes: int = _JWT_DEFAULT_EXPIRY_MINUTES,
    extra_claims: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a signed JWT session token.

    Args:
        subject: Unique identifier for the session/user (e.g. username or API key name).
        role: The CORE Role granted to this token.
        expires_minutes: Token lifetime in minutes (default: CORE_JWT_EXPIRY_MINUTES env var or 60).
        extra_claims: Optional dict of additional claims to embed in the payload.

    Returns:
        Signed JWT string (HS256).
    """
    now = datetime.now(tz=timezone.utc)
    payload: Dict[str, Any] = {
        "iss": _JWT_ISSUER,
        "sub": subject,
        "role": role.value,
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
    }
    if extra_claims:
        payload.update(extra_claims)
    return _pyjwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)


def decode_jwt_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode a JWT session token.

    Args:
        token: A JWT string previously produced by create_jwt_token().

    Returns:
        The decoded payload dict (includes sub, role, exp, etc.).

    Raises:
        JWTError: If the token is invalid, expired, or has the wrong issuer.
    """
    try:
        payload = _pyjwt.decode(
            token,
            _JWT_SECRET,
            algorithms=[_JWT_ALGORITHM],
            issuer=_JWT_ISSUER,
        )
        return payload
    except _pyjwt.ExpiredSignatureError as exc:
        raise JWTError("Token has expired") from exc
    except _pyjwt.InvalidIssuerError as exc:
        raise JWTError("Token issuer invalid") from exc
    except _pyjwt.PyJWTError as exc:
        raise JWTError(f"Token validation failed: {exc}") from exc


async def get_jwt_bearer(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Dict[str, Any]:
    """
    FastAPI dependency: authenticate via JWT Bearer token.

    Extracts the ``Authorization: Bearer <token>`` header, verifies the JWT,
    and returns a key-data dict compatible with the rest of the auth pipeline
    (same shape as validate_api_key output: ``name``, ``permissions``, ``role``).

    Raises HTTP 401 if the header is missing or the token is invalid/expired.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = decode_jwt_token(credentials.credentials)
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )

    role_str = payload.get("role", Role.VIEWER.value)
    # Map role string to the matching role permission so require_role() works
    role_permission = f"role:{role_str}"
    return {
        "name": payload.get("sub", "jwt-session"),
        "permissions": [role_permission],
        "role": role_str,
        "jwt_subject": payload.get("sub"),
        "jwt_exp": payload.get("exp"),
    }


def require_jwt_role(minimum_role: "Role") -> Callable:
    """
    FastAPI dependency factory: authenticate via JWT and enforce a minimum role.

    Same semantics as require_role() but reads credentials from a Bearer token
    rather than the X-API-Key header.

    Usage::

        @router.get("/admin/stats")
        async def stats(key_data = Depends(require_jwt_role(Role.ADMIN))):
            ...
    """
    satisfying_perms = _ROLE_SATISFIES[minimum_role]

    async def _check(
        key_data: Dict[str, Any] = Depends(get_jwt_bearer),
    ) -> Dict[str, Any]:
        permissions = key_data.get("permissions", [])
        if satisfying_perms.intersection(permissions):
            return key_data
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Insufficient role. '{minimum_role.value}' or higher required. "
                f"Token role: {key_data.get('role')}"
            ),
        )

    return _check


# =============================================================================
# Initialization
# =============================================================================


def init_security():
    """
    Initialize security module.

    Called at startup to set up default keys.
    Note: DB-backed key loading happens separately via load_keys_from_db().
    """
    # Generate a default key for the UI if not in production
    if os.getenv("CORE_ENV", "development") != "production":
        if not _API_KEY_CACHE:
            key = generate_api_key_sync(
                name="core-ui-dev",
                description="Development key for CORE UI",
                permissions=["*"],
            )
            logger.info("Generated development API key: %s...", key[:20])

    logger.info("Security module initialized. Keys in cache: %d", len(_API_KEY_CACHE))
