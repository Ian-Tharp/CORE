"""
Request logging helpers.

The actual logging middleware lives in ``app.core.middleware.RequestContextMiddleware``.
This module provides the ``get_correlation_id`` helper for route handlers that
need the current request's tracing ID.
"""

from starlette.requests import Request
import uuid


def get_correlation_id(request: Request) -> str:
    """Get correlation ID from request state, or generate a fallback.

    Prefer ``correlation_id`` (set by ``CorrelationIDMiddleware``), then
    ``request_id`` (set by ``RequestContextMiddleware``).
    """
    return (
        getattr(request.state, "correlation_id", None)
        or getattr(request.state, "request_id", None)
        or str(uuid.uuid4())[:8]
    )
