"""
Error Handling Utilities

Standardized error types and handlers for CORE.
Provides consistent error responses across all endpoints.

Usage:
    from app.utils.errors import COREError, handle_core_errors
    
    @router.post("/endpoint")
    @handle_core_errors
    async def my_endpoint():
        if problem:
            raise COREError("Something went wrong", code="PROBLEM_CODE")
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict, Optional, Type
from functools import wraps

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str
    code: str
    details: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Agent not found",
                "code": "AGENT_NOT_FOUND",
                "details": {"agent_id": "unknown_agent"},
                "correlation_id": "abc123"
            }
        }


class COREError(Exception):
    """Base exception for CORE errors."""
    
    def __init__(
        self,
        message: str,
        code: str = "CORE_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)
    
    def to_response(self, correlation_id: Optional[str] = None) -> ErrorResponse:
        return ErrorResponse(
            error=self.message,
            code=self.code,
            details=self.details,
            correlation_id=correlation_id
        )


class NotFoundError(COREError):
    """Resource not found error."""
    
    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"{resource_type} '{resource_id}' not found",
            code=f"{resource_type.upper()}_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details=details or {"resource_type": resource_type, "resource_id": resource_id}
        )


class ValidationError(COREError):
    """Input validation error."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details or ({"field": field} if field else {})
        )


class AuthenticationError(COREError):
    """Authentication failed."""
    
    def __init__(
        self,
        message: str = "Authentication required",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code="AUTHENTICATION_ERROR",
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details
        )


class AuthorizationError(COREError):
    """Authorization denied."""
    
    def __init__(
        self,
        message: str = "Permission denied",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            code="AUTHORIZATION_ERROR",
            status_code=status.HTTP_403_FORBIDDEN,
            details=details
        )


class RateLimitError(COREError):
    """Rate limit exceeded."""
    
    def __init__(
        self,
        retry_after: int = 60,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            code="RATE_LIMIT_EXCEEDED",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details or {"retry_after": retry_after}
        )


class ServiceUnavailableError(COREError):
    """External service unavailable."""
    
    def __init__(
        self,
        service: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message or f"Service '{service}' is unavailable",
            code="SERVICE_UNAVAILABLE",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details or {"service": service}
        )


class ExecutionError(COREError):
    """CORE execution error."""
    
    def __init__(
        self,
        message: str,
        run_id: Optional[str] = None,
        node: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        err_details = details or {}
        if run_id:
            err_details["run_id"] = run_id
        if node:
            err_details["node"] = node
        
        super().__init__(
            message=message,
            code="EXECUTION_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=err_details
        )


def handle_core_errors(func):
    """
    Decorator to handle COREError exceptions.
    
    Converts exceptions to proper HTTP responses.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except COREError as e:
            # Get correlation ID if available
            correlation_id = None
            for arg in args:
                if isinstance(arg, Request):
                    correlation_id = getattr(arg.state, "correlation_id", None)
                    break
            
            logger.warning(
                f"CORE Error: {e.code} - {e.message}",
                extra={"details": e.details, "correlation_id": correlation_id}
            )
            
            raise HTTPException(
                status_code=e.status_code,
                detail=e.to_response(correlation_id).model_dump()
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Internal server error",
                    "code": "INTERNAL_ERROR",
                    "details": {"type": type(e).__name__}
                }
            )
    
    return wrapper


async def core_exception_handler(request: Request, exc: COREError) -> JSONResponse:
    """
    Global exception handler for COREError.
    
    Register with FastAPI:
        app.add_exception_handler(COREError, core_exception_handler)
    """
    correlation_id = getattr(request.state, "correlation_id", None)
    
    logger.warning(
        f"CORE Error: {exc.code} - {exc.message}",
        extra={"details": exc.details, "correlation_id": correlation_id}
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_response(correlation_id).model_dump()
    )
