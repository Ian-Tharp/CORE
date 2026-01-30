"""
CORE Middleware - Request/Response logging and metrics.

Provides:
- Structured request logging with timing
- Request ID tracking
- Error handling middleware
- CORS with detailed configuration

RSI TODO: Add OpenTelemetry integration
RSI TODO: Add Prometheus metrics endpoint
RSI TODO: Add request body logging for debugging
"""

from __future__ import annotations

import time
import uuid
import logging
from typing import Callable
from datetime import datetime

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import json

logger = logging.getLogger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Adds request ID and timing to all requests.
    
    Logs:
    - Request method, path, client
    - Response status code
    - Request duration
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        
        # Record start time
        start_time = time.time()
        
        # Get client info
        client_host = request.client.host if request.client else "unknown"
        
        # Log request start
        logger.info(
            f"[{request_id}] → {request.method} {request.url.path} "
            f"from {client_host}"
        )
        
        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            # Log exception and re-raise
            duration = (time.time() - start_time) * 1000
            logger.error(
                f"[{request_id}] ✗ {request.method} {request.url.path} "
                f"failed after {duration:.1f}ms: {str(e)}"
            )
            raise
        
        # Calculate duration
        duration = (time.time() - start_time) * 1000
        
        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.1f}ms"
        
        # Log response
        status_emoji = "✓" if response.status_code < 400 else "✗"
        logger.info(
            f"[{request_id}] {status_emoji} {request.method} {request.url.path} "
            f"→ {response.status_code} in {duration:.1f}ms"
        )
        
        return response


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Global error handler that returns consistent JSON responses.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            # Get request ID if available
            request_id = getattr(request.state, "request_id", "unknown")
            
            # Log the error
            logger.exception(f"[{request_id}] Unhandled exception: {e}")
            
            # Return JSON error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": str(e),
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )


# =============================================================================
# Request Metrics
# =============================================================================

class RequestMetrics:
    """
    Collects request metrics for monitoring.
    
    RSI TODO: Export to Prometheus/StatsD
    """
    
    def __init__(self):
        self.total_requests = 0
        self.error_requests = 0
        self.total_duration_ms = 0.0
        self.requests_by_path: dict = {}
        self.requests_by_status: dict = {}
        self.start_time = datetime.utcnow()
    
    def record(self, path: str, method: str, status: int, duration_ms: float):
        """Record a request."""
        self.total_requests += 1
        self.total_duration_ms += duration_ms
        
        if status >= 400:
            self.error_requests += 1
        
        # Track by path
        path_key = f"{method} {path}"
        if path_key not in self.requests_by_path:
            self.requests_by_path[path_key] = {
                "count": 0,
                "total_ms": 0,
                "errors": 0
            }
        self.requests_by_path[path_key]["count"] += 1
        self.requests_by_path[path_key]["total_ms"] += duration_ms
        if status >= 400:
            self.requests_by_path[path_key]["errors"] += 1
        
        # Track by status
        status_key = str(status)
        self.requests_by_status[status_key] = self.requests_by_status.get(status_key, 0) + 1
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "total_requests": self.total_requests,
            "error_requests": self.error_requests,
            "error_rate": self.error_requests / self.total_requests if self.total_requests > 0 else 0,
            "avg_duration_ms": self.total_duration_ms / self.total_requests if self.total_requests > 0 else 0,
            "requests_per_second": self.total_requests / uptime if uptime > 0 else 0,
            "uptime_seconds": uptime,
            "by_status": self.requests_by_status,
            "top_endpoints": sorted(
                [
                    {
                        "path": path,
                        "count": data["count"],
                        "avg_ms": data["total_ms"] / data["count"] if data["count"] > 0 else 0,
                        "error_rate": data["errors"] / data["count"] if data["count"] > 0 else 0
                    }
                    for path, data in self.requests_by_path.items()
                ],
                key=lambda x: x["count"],
                reverse=True
            )[:10]
        }
    
    def reset(self):
        """Reset metrics."""
        self.__init__()


# Global metrics instance
_metrics = RequestMetrics()


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Records request metrics.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.time()
        response = await call_next(request)
        duration = (time.time() - start) * 1000
        
        # Record metrics (skip health endpoints)
        if not request.url.path.startswith("/health"):
            _metrics.record(
                path=request.url.path,
                method=request.method,
                status=response.status_code,
                duration_ms=duration
            )
        
        return response


def get_metrics() -> RequestMetrics:
    """Get the global metrics instance."""
    return _metrics


# =============================================================================
# Setup Function
# =============================================================================

def setup_middleware(app: FastAPI) -> None:
    """
    Configure all middleware for the FastAPI app.
    
    Order matters - middleware is executed in reverse order of addition.
    """
    # Error handler (outermost)
    app.add_middleware(ErrorHandlerMiddleware)
    
    # Request context (request ID, timing)
    app.add_middleware(RequestContextMiddleware)
    
    # Metrics collection
    app.add_middleware(MetricsMiddleware)
    
    logger.info("Middleware configured")
