"""
API Authentication Module

Provides simple API key authentication for CORE endpoints.
This is a lightweight auth layer - for production, consider OAuth2/JWT.

Usage in controllers:
    from app.auth import require_api_key
    
    @router.post("/engine/run")
    async def run_core(request: RunRequest, api_key: str = Depends(require_api_key)):
        ...
"""

import os
import logging
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

# API key header name
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Valid API keys (in production, store in database or secrets manager)
# For now, use environment variable or default dev key
VALID_API_KEYS = set(
    filter(None, [
        os.getenv("CORE_API_KEY"),
        os.getenv("CORE_API_KEY_2"),
        "dev-key-local-only",  # Default dev key (remove in production)
    ])
)

# Endpoints that don't require authentication
PUBLIC_ENDPOINTS = {
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
}


async def verify_api_key(api_key: Optional[str]) -> bool:
    """Verify if the provided API key is valid."""
    if not api_key:
        return False
    return api_key in VALID_API_KEYS


async def require_api_key(
    api_key: Optional[str] = Security(API_KEY_HEADER)
) -> str:
    """
    Dependency that requires a valid API key.
    
    Raises HTTPException 401 if key is missing or invalid.
    Returns the API key if valid.
    """
    if not api_key:
        logger.warning("API request without API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Pass X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    if api_key not in VALID_API_KEYS:
        logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return api_key


async def optional_api_key(
    api_key: Optional[str] = Security(API_KEY_HEADER)
) -> Optional[str]:
    """
    Dependency that accepts but doesn't require an API key.
    
    Useful for endpoints that work differently for authenticated vs anonymous users.
    Returns the API key if provided and valid, None otherwise.
    """
    if api_key and api_key in VALID_API_KEYS:
        return api_key
    return None


def add_api_key(key: str) -> None:
    """Add a new valid API key (runtime only, not persisted)."""
    VALID_API_KEYS.add(key)
    logger.info(f"Added API key: {key[:8]}...")


def remove_api_key(key: str) -> bool:
    """Remove an API key. Returns True if key existed."""
    if key in VALID_API_KEYS:
        VALID_API_KEYS.discard(key)
        logger.info(f"Removed API key: {key[:8]}...")
        return True
    return False


def list_api_key_count() -> int:
    """Return count of valid API keys (for monitoring)."""
    return len(VALID_API_KEYS)
