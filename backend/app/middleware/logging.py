"""
Request Logging Middleware

Structured logging for all API requests.
Includes timing, status codes, and correlation IDs.

Usage in main.py:
    from app.middleware.logging import RequestLoggingMiddleware
    app.add_middleware(RequestLoggingMiddleware)
"""

import time
import uuid
import logging
import json
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("core.requests")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs all requests with timing and correlation IDs.
    
    Each request gets a unique correlation ID that can be used to trace
    the request through logs.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())[:8]
        
        # Add to request state for use in handlers
        request.state.correlation_id = correlation_id
        
        # Extract useful info
        client_ip = request.headers.get("X-Forwarded-For", 
                                        request.client.host if request.client else "unknown")
        api_key = request.headers.get("X-API-Key", "none")[:8] if request.headers.get("X-API-Key") else "none"
        
        # Start timer
        start_time = time.time()
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log the request
            log_data = {
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "client_ip": client_ip,
                "api_key": api_key,
            }
            
            # Choose log level based on status
            if response.status_code >= 500:
                logger.error(json.dumps(log_data))
            elif response.status_code >= 400:
                logger.warning(json.dumps(log_data))
            else:
                logger.info(json.dumps(log_data))
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            
            return response
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log_data = {
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "error": str(e),
                "duration_ms": round(duration_ms, 2),
                "client_ip": client_ip,
            }
            logger.error(json.dumps(log_data), exc_info=True)
            raise


def get_correlation_id(request: Request) -> str:
    """Get correlation ID from request, or generate new one."""
    return getattr(request.state, "correlation_id", str(uuid.uuid4())[:8])
