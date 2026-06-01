"""Request correlation ID middleware for distributed tracing.

Provides a single source of truth for request tracing. Accepts an incoming
``X-Correlation-ID`` header (for distributed tracing across services) or
generates a fresh UUID segment. The value is stored on ``request.state`` as
both ``correlation_id`` and ``request_id`` so downstream middleware and
handlers can use either name consistently.
"""

import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger(__name__)

CORRELATION_HEADER = "X-Correlation-ID"
REQUEST_ID_HEADER = "X-Request-ID"


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Adds a correlation ID to each request for tracing.

    This middleware runs outermost so that all subsequent middleware and route
    handlers can access ``request.state.correlation_id``.

    Behaviour:
    - Accepts an incoming ``X-Correlation-ID`` header for distributed tracing.
    - Falls back to a short UUID when no header is present.
    - Sets both ``request.state.correlation_id`` and ``request.state.request_id``
      so consumers can use either attribute name.
    - Adds ``X-Correlation-ID`` and ``X-Request-ID`` response headers.
    """

    async def dispatch(self, request: Request, call_next):
        # Prefer incoming header for distributed tracing; generate if absent
        correlation_id = (
            request.headers.get(CORRELATION_HEADER) or str(uuid.uuid4())[:8]
        )

        # Store on request state — both names point to the same value
        request.state.correlation_id = correlation_id
        request.state.request_id = correlation_id

        response = await call_next(request)

        # Include in response headers for client-side tracing
        response.headers[CORRELATION_HEADER] = correlation_id
        response.headers[REQUEST_ID_HEADER] = correlation_id

        return response
