"""Request correlation ID middleware for tracing."""
import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger(__name__)

CORRELATION_HEADER = "X-Correlation-ID"


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Adds a correlation ID to each request for tracing."""

    async def dispatch(self, request: Request, call_next):
        # Use existing correlation ID from header, or generate new one
        correlation_id = request.headers.get(CORRELATION_HEADER, str(uuid.uuid4())[:8])

        # Store in request state for access in route handlers
        request.state.correlation_id = correlation_id

        response = await call_next(request)

        # Include correlation ID in response headers
        response.headers[CORRELATION_HEADER] = correlation_id

        return response
