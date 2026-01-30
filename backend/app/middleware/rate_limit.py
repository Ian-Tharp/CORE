"""
Rate Limiting Middleware

Simple in-memory rate limiting for CORE API.
For production, use Redis-based rate limiting.

Usage:
    from app.middleware.rate_limit import RateLimiter, rate_limit
    
    limiter = RateLimiter(requests_per_minute=60)
    
    @router.post("/engine/run")
    @rate_limit(limiter, key_func=get_client_ip)
    async def run_core(...):
        ...
"""

import time
import logging
from typing import Dict, Callable, Optional
from collections import defaultdict
from functools import wraps
from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Simple token bucket rate limiter.
    
    Args:
        requests_per_minute: Maximum requests allowed per minute
        burst_size: Maximum burst size (defaults to requests_per_minute)
    """
    
    def __init__(self, requests_per_minute: int = 60, burst_size: Optional[int] = None):
        self.rate = requests_per_minute / 60.0  # tokens per second
        self.burst_size = burst_size or requests_per_minute
        self.tokens: Dict[str, float] = defaultdict(lambda: self.burst_size)
        self.last_update: Dict[str, float] = defaultdict(time.time)
    
    def _refill(self, key: str) -> None:
        """Refill tokens based on time elapsed."""
        now = time.time()
        elapsed = now - self.last_update[key]
        self.tokens[key] = min(
            self.burst_size,
            self.tokens[key] + elapsed * self.rate
        )
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


# Default rate limiters for different endpoint types
engine_limiter = RateLimiter(requests_per_minute=10, burst_size=5)  # Expensive operations
api_limiter = RateLimiter(requests_per_minute=60, burst_size=30)    # Standard API calls
public_limiter = RateLimiter(requests_per_minute=120, burst_size=60) # Health checks etc


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
    limiter: RateLimiter,
    key_func: Callable[[Request], str] = get_client_ip,
    cost: float = 1.0
) -> None:
    """
    Check rate limit and raise HTTPException if exceeded.
    
    Args:
        request: FastAPI request object
        limiter: RateLimiter instance to use
        key_func: Function to extract rate limit key from request
        cost: Token cost for this request
    
    Raises:
        HTTPException 429 if rate limited
    """
    key = key_func(request)
    
    if not limiter.allow(key, cost):
        retry_after = limiter.get_retry_after(key, cost)
        logger.warning(f"Rate limit exceeded for {key}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Retry after {retry_after:.1f} seconds.",
            headers={"Retry-After": str(int(retry_after) + 1)}
        )


def rate_limit(
    limiter: RateLimiter,
    key_func: Callable[[Request], str] = get_client_ip,
    cost: float = 1.0
):
    """
    Decorator for rate limiting endpoints.
    
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
            
            return await func(*args, request=request, **kwargs)
        return wrapper
    return decorator
