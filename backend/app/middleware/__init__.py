"""
CORE Middleware Package

Provides middleware for:
- Rate limiting
- Request logging
- Error handling
"""

from app.middleware.rate_limit import (
    RateLimiter,
    rate_limit,
    check_rate_limit,
    engine_limiter,
    api_limiter,
    public_limiter,
    get_client_ip,
    get_api_key,
)

from app.middleware.logging import (
    RequestLoggingMiddleware,
    get_correlation_id,
)

__all__ = [
    # Rate limiting
    "RateLimiter",
    "rate_limit",
    "check_rate_limit",
    "engine_limiter",
    "api_limiter",
    "public_limiter",
    "get_client_ip",
    "get_api_key",
    # Logging
    "RequestLoggingMiddleware",
    "get_correlation_id",
]
