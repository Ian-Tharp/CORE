"""
Rate Limiting Middleware

Token bucket rate limiting for CORE API with optional Redis backend.
Falls back to in-memory rate limiting when Redis is unavailable.

Usage:
    from app.middleware.rate_limit import RateLimiter, RedisRateLimiter, rate_limit

    # In-memory (single-process)
    limiter = RateLimiter(requests_per_minute=60)

    # Redis-backed (distributed, multi-process)
    limiter = await RedisRateLimiter.create(requests_per_minute=60)

    @router.post("/engine/run")
    @rate_limit(limiter, key_func=get_client_ip)
    async def run_core(...):
        ...
"""

import os
import time
import logging
from typing import Dict, Callable, Optional
from functools import wraps
from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Simple token bucket rate limiter (in-memory).

    Args:
        requests_per_minute: Maximum requests allowed per minute
        burst_size: Maximum burst size (defaults to requests_per_minute)
    """

    def __init__(self, requests_per_minute: int = 60, burst_size: Optional[int] = None):
        self.rate = requests_per_minute / 60.0  # tokens per second
        self.burst_size = burst_size or requests_per_minute
        self.tokens: Dict[str, float] = {}
        self.last_update: Dict[str, float] = {}

    def _refill(self, key: str) -> None:
        """Refill tokens based on time elapsed."""
        now = time.time()
        if key not in self.tokens:
            self.tokens[key] = self.burst_size
            self.last_update[key] = now
            return
        elapsed = now - self.last_update[key]
        self.tokens[key] = min(self.burst_size, self.tokens[key] + elapsed * self.rate)
        self.last_update[key] = now

    def allow(self, key: str, cost: float = 1.0) -> bool:
        """
        Check if request is allowed.

        Args:
            key: Identifier for the client (IP, API key, user ID)
            cost: Token cost for this request (default 1)

        Returns:
            True if allowed, False if rate limited
        """
        self._refill(key)

        if self.tokens[key] >= cost:
            self.tokens[key] -= cost
            return True

        return False

    def get_retry_after(self, key: str, cost: float = 1.0) -> float:
        """Get seconds until request would be allowed."""
        self._refill(key)

        if self.tokens[key] >= cost:
            return 0

        needed = cost - self.tokens[key]
        return needed / self.rate

    def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        self.tokens[key] = self.burst_size
        self.last_update[key] = time.time()


# ---------------------------------------------------------------------------
# Redis-backed rate limiter
# ---------------------------------------------------------------------------

# Lua script: atomic token bucket check in Redis.
# KEYS[1] = bucket key  ("rl:{prefix}:{client_id}")
# ARGV[1] = rate (tokens/sec), ARGV[2] = burst_size, ARGV[3] = cost, ARGV[4] = now (float)
# Returns: {allowed (0/1), tokens_remaining (float), retry_after_ms (int)}
_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

local data = redis.call('HMGET', key, 'tokens', 'last_update')
local tokens = tonumber(data[1])
local last_update = tonumber(data[2])

if tokens == nil then
    tokens = burst
    last_update = now
else
    local elapsed = now - last_update
    tokens = math.min(burst, tokens + elapsed * rate)
    last_update = now
end

local allowed = 0
if tokens >= cost then
    tokens = tokens - cost
    allowed = 1
end

redis.call('HMSET', key, 'tokens', tostring(tokens), 'last_update', tostring(last_update))
-- Expire key after 2x the time to fill the bucket (auto-cleanup idle clients)
local ttl = math.ceil(burst / rate * 2)
if ttl < 120 then ttl = 120 end
redis.call('EXPIRE', key, ttl)

local retry_after_ms = 0
if allowed == 0 then
    local needed = cost - tokens
    retry_after_ms = math.ceil(needed / rate * 1000)
end

return {allowed, tostring(tokens), retry_after_ms}
"""


class RedisRateLimiter:
    """
    Redis-backed token bucket rate limiter for distributed deployments.

    Uses a Lua script for atomic check-and-decrement, so it's safe across
    multiple workers / containers.  Falls back to an in-memory RateLimiter
    when Redis is unreachable.

    Args:
        requests_per_minute: Maximum requests allowed per minute
        burst_size: Maximum burst size (defaults to requests_per_minute)
        redis_client: An ``redis.asyncio.Redis`` instance (or None for auto-connect)
        key_prefix: Redis key namespace prefix
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: Optional[int] = None,
        redis_client=None,
        key_prefix: str = "rl",
    ):
        self.rate = requests_per_minute / 60.0
        self.burst_size = burst_size or requests_per_minute
        self._redis = redis_client
        self._script_sha: Optional[str] = None
        self._key_prefix = key_prefix
        # Fallback limiter used when Redis is down
        self._fallback = RateLimiter(
            requests_per_minute=requests_per_minute,
            burst_size=self.burst_size,
        )
        self._redis_healthy = True

    # -- Factory ----------------------------------------------------------

    @classmethod
    async def create(
        cls,
        requests_per_minute: int = 60,
        burst_size: Optional[int] = None,
        key_prefix: str = "rl",
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
    ) -> "RedisRateLimiter":
        """
        Create a RedisRateLimiter, connecting to Redis and pre-loading the
        Lua script.  Falls back gracefully if Redis is unreachable.
        """
        import redis.asyncio as aioredis

        host = redis_host or os.getenv("REDIS_HOST", "redis")
        port = redis_port or int(os.getenv("REDIS_PORT", "6379"))

        client = aioredis.Redis(host=host, port=port, decode_responses=True)
        instance = cls(
            requests_per_minute=requests_per_minute,
            burst_size=burst_size,
            redis_client=client,
            key_prefix=key_prefix,
        )
        try:
            await client.ping()
            instance._script_sha = await client.script_load(_TOKEN_BUCKET_LUA)
            logger.info(
                "RedisRateLimiter connected (host=%s:%s, prefix=%s, rpm=%d)",
                host,
                port,
                key_prefix,
                requests_per_minute,
            )
        except Exception as exc:
            logger.warning(
                "RedisRateLimiter could not connect to Redis (%s); using in-memory fallback",
                exc,
            )
            instance._redis_healthy = False

        return instance

    # -- Key helpers -------------------------------------------------------

    def _bucket_key(self, client_id: str) -> str:
        return f"{self._key_prefix}:{client_id}"

    # -- Core API (same interface as RateLimiter) --------------------------

    async def allow_async(self, key: str, cost: float = 1.0) -> bool:
        """Async check whether request is allowed."""
        if not self._redis_healthy or self._redis is None:
            return self._fallback.allow(key, cost)

        try:
            now = time.time()
            result = await self._redis.evalsha(
                self._script_sha,
                1,
                self._bucket_key(key),
                str(self.rate),
                str(self.burst_size),
                str(cost),
                str(now),
            )
            allowed = int(result[0])
            return allowed == 1
        except Exception as exc:
            logger.warning(
                "Redis rate-limit call failed (%s); falling back to memory", exc
            )
            self._redis_healthy = False
            return self._fallback.allow(key, cost)

    def allow(self, key: str, cost: float = 1.0) -> bool:
        """Sync fallback — always uses in-memory limiter."""
        return self._fallback.allow(key, cost)

    async def get_retry_after_async(self, key: str, cost: float = 1.0) -> float:
        """Async retry-after calculation."""
        if not self._redis_healthy or self._redis is None:
            return self._fallback.get_retry_after(key, cost)

        try:
            now = time.time()
            result = await self._redis.evalsha(
                self._script_sha,
                1,
                self._bucket_key(key),
                str(self.rate),
                str(self.burst_size),
                str(cost),
                str(now),
            )
            retry_ms = int(result[2])
            return retry_ms / 1000.0
        except Exception:
            return self._fallback.get_retry_after(key, cost)

    def get_retry_after(self, key: str, cost: float = 1.0) -> float:
        """Sync fallback."""
        return self._fallback.get_retry_after(key, cost)

    async def reset_async(self, key: str) -> None:
        """Reset rate limit in Redis."""
        if self._redis_healthy and self._redis is not None:
            try:
                await self._redis.delete(self._bucket_key(key))
            except Exception:
                pass
        self._fallback.reset(key)

    def reset(self, key: str) -> None:
        """Sync reset (in-memory only)."""
        self._fallback.reset(key)

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            try:
                await self._redis.close()
            except Exception:
                pass

    @property
    def is_redis_connected(self) -> bool:
        """Whether the limiter is actively using Redis."""
        return self._redis_healthy and self._redis is not None


# Default rate limiters for different endpoint types (in-memory; upgraded at startup)
engine_limiter = RateLimiter(
    requests_per_minute=10, burst_size=5
)  # Expensive operations
api_limiter = RateLimiter(requests_per_minute=60, burst_size=30)  # Standard API calls
public_limiter = RateLimiter(
    requests_per_minute=120, burst_size=60
)  # Health checks etc


async def upgrade_limiters_to_redis() -> None:
    """
    Replace module-level in-memory limiters with Redis-backed ones.
    Call once at application startup (after Redis is confirmed reachable).
    """
    global engine_limiter, api_limiter, public_limiter
    try:
        engine_limiter = await RedisRateLimiter.create(
            requests_per_minute=10,
            burst_size=5,
            key_prefix="rl:engine",
        )
        api_limiter = await RedisRateLimiter.create(
            requests_per_minute=60,
            burst_size=30,
            key_prefix="rl:api",
        )
        public_limiter = await RedisRateLimiter.create(
            requests_per_minute=120,
            burst_size=60,
            key_prefix="rl:public",
        )
        logger.info("Rate limiters upgraded to Redis backend")
    except Exception as exc:
        logger.warning("Could not upgrade rate limiters to Redis: %s", exc)


def get_client_ip(request: Request) -> str:
    """Extract client IP from request, handling proxies."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def get_api_key(request: Request) -> str:
    """Extract API key from request headers."""
    return request.headers.get("X-API-Key", get_client_ip(request))


async def check_rate_limit(
    request: Request,
    limiter,
    key_func: Callable[[Request], str] = get_client_ip,
    cost: float = 1.0,
) -> None:
    """
    Check rate limit and raise HTTPException if exceeded.

    Works with both RateLimiter and RedisRateLimiter.

    Args:
        request: FastAPI request object
        limiter: RateLimiter or RedisRateLimiter instance
        key_func: Function to extract rate limit key from request
        cost: Token cost for this request

    Raises:
        HTTPException 429 if rate limited
    """
    key = key_func(request)

    # Use async path if available (RedisRateLimiter)
    if hasattr(limiter, "allow_async"):
        allowed = await limiter.allow_async(key, cost)
        if not allowed:
            retry_after = await limiter.get_retry_after_async(key, cost)
            logger.warning(f"Rate limit exceeded for {key}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Retry after {retry_after:.1f} seconds.",
                headers={"Retry-After": str(int(retry_after) + 1)},
            )
    else:
        if not limiter.allow(key, cost):
            retry_after = limiter.get_retry_after(key, cost)
            logger.warning(f"Rate limit exceeded for {key}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Retry after {retry_after:.1f} seconds.",
                headers={"Retry-After": str(int(retry_after) + 1)},
            )


def rate_limit(
    limiter, key_func: Callable[[Request], str] = get_client_ip, cost: float = 1.0
):
    """
    Decorator for rate limiting endpoints.

    Works with both RateLimiter and RedisRateLimiter.

    Usage:
        @router.post("/expensive")
        @rate_limit(engine_limiter)
        async def expensive_operation(request: Request):
            ...
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, request: Request = None, **kwargs):
            # Try to get request from args or kwargs
            if request is None:
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

            if request:
                await check_rate_limit(request, limiter, key_func, cost)

            if request is not None:
                kwargs["request"] = request
            return await func(*args, **kwargs)

        return wrapper

    return decorator
