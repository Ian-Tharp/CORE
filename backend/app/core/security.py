"""
CORE Security Module - API Key Authentication and Rate Limiting

Provides:
- API key validation and management
- Request rate limiting
- Security middleware

RSI TODO: Add JWT support for user sessions
RSI TODO: Add role-based access control (RBAC)
RSI TODO: Add audit logging for sensitive operations
"""

from __future__ import annotations

import os
import secrets
import hashlib
import time
from datetime import datetime
from typing import Optional, Dict, Any
from collections import defaultdict
from functools import wraps

from fastapi import HTTPException, Security, Request, status
from fastapi.security import APIKeyHeader, APIKeyQuery
import logging

logger = logging.getLogger(__name__)

# API key extraction strategies
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
API_KEY_QUERY = APIKeyQuery(name="api_key", auto_error=False)

# In-memory storage for API keys (RSI TODO: Move to database)
_API_KEYS: Dict[str, Dict[str, Any]] = {}

# Rate limiting storage
_RATE_LIMIT_BUCKETS: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "window_start": time.time()})


def _get_master_api_key() -> Optional[str]:
    """Get the master API key from environment."""
    return os.getenv("CORE_API_KEY")


def _hash_key(key: str) -> str:
    """Hash an API key for storage comparison."""
    return hashlib.sha256(key.encode()).hexdigest()


def generate_api_key(name: str, description: str = "", permissions: list = None) -> str:
    """
    Generate a new API key.
    
    Args:
        name: Identifier for the key (e.g., "vigil-agent", "web-ui")
        description: Optional description
        permissions: List of permitted operations
    
    Returns:
        The raw API key (store this securely - it won't be shown again)
    """
    raw_key = f"core_{secrets.token_urlsafe(32)}"
    key_hash = _hash_key(raw_key)
    
    _API_KEYS[key_hash] = {
        "name": name,
        "description": description,
        "permissions": permissions or ["*"],
        "created_at": datetime.utcnow().isoformat(),
        "last_used": None,
        "request_count": 0
    }
    
    logger.info(f"Generated API key for: {name}")
    return raw_key


def validate_api_key(key: str) -> Optional[Dict[str, Any]]:
    """
    Validate an API key and return its metadata.
    
    Args:
        key: The raw API key
        
    Returns:
        Key metadata if valid, None if invalid
    """
    # Check master key first
    master_key = _get_master_api_key()
    if master_key and key == master_key:
        return {
            "name": "master",
            "permissions": ["*"],
            "is_master": True
        }
    
    # Check registered keys
    key_hash = _hash_key(key)
    if key_hash in _API_KEYS:
        key_data = _API_KEYS[key_hash]
        key_data["last_used"] = datetime.utcnow().isoformat()
        key_data["request_count"] += 1
        return key_data
    
    return None


def revoke_api_key(name: str) -> bool:
    """
    Revoke an API key by name.
    
    Args:
        name: The key name to revoke
        
    Returns:
        True if key was found and revoked
    """
    for key_hash, data in list(_API_KEYS.items()):
        if data["name"] == name:
            del _API_KEYS[key_hash]
            logger.info(f"Revoked API key: {name}")
            return True
    return False


def list_api_keys() -> list:
    """List all registered API keys (without exposing the actual keys)."""
    return [
        {
            "name": data["name"],
            "description": data["description"],
            "permissions": data["permissions"],
            "created_at": data["created_at"],
            "last_used": data["last_used"],
            "request_count": data["request_count"]
        }
        for data in _API_KEYS.values()
    ]


# =============================================================================
# Rate Limiting
# =============================================================================

class RateLimiter:
    """
    Simple sliding window rate limiter.
    
    RSI TODO: Use Redis for distributed rate limiting
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000
    ):
        self.rpm = requests_per_minute
        self.rph = requests_per_hour
        self._minute_buckets: Dict[str, list] = defaultdict(list)
        self._hour_buckets: Dict[str, list] = defaultdict(list)
    
    def check_rate_limit(self, client_id: str) -> Dict[str, Any]:
        """
        Check if client is within rate limits.
        
        Args:
            client_id: Identifier for the client (IP, API key name, etc.)
            
        Returns:
            Dict with allowed status and limit info
        """
        now = time.time()
        minute_ago = now - 60
        hour_ago = now - 3600
        
        # Clean old entries
        self._minute_buckets[client_id] = [
            t for t in self._minute_buckets[client_id] if t > minute_ago
        ]
        self._hour_buckets[client_id] = [
            t for t in self._hour_buckets[client_id] if t > hour_ago
        ]
        
        minute_count = len(self._minute_buckets[client_id])
        hour_count = len(self._hour_buckets[client_id])
        
        allowed = minute_count < self.rpm and hour_count < self.rph
        
        if allowed:
            self._minute_buckets[client_id].append(now)
            self._hour_buckets[client_id].append(now)
        
        return {
            "allowed": allowed,
            "minute_remaining": max(0, self.rpm - minute_count - 1),
            "hour_remaining": max(0, self.rph - hour_count - 1),
            "retry_after": 60 if minute_count >= self.rpm else None
        }


# Global rate limiter
_rate_limiter = RateLimiter()


# =============================================================================
# FastAPI Dependencies
# =============================================================================

async def get_api_key(
    api_key_header: str = Security(API_KEY_HEADER),
    api_key_query: str = Security(API_KEY_QUERY),
) -> Dict[str, Any]:
    """
    FastAPI dependency for API key authentication.
    
    Checks header first, then query parameter.
    Raises 401 if no valid key found.
    """
    # Check if auth is disabled (development mode)
    if os.getenv("CORE_AUTH_DISABLED", "").lower() == "true":
        return {"name": "development", "permissions": ["*"], "auth_disabled": True}
    
    # Try header first
    key = api_key_header or api_key_query
    
    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Provide via X-API-Key header or api_key query parameter.",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    key_data = validate_api_key(key)
    
    if not key_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    return key_data


async def check_rate_limit(request: Request) -> Dict[str, Any]:
    """
    FastAPI dependency for rate limiting.
    
    Uses client IP as identifier (or API key name if authenticated).
    """
    # Check if rate limiting is disabled
    if os.getenv("CORE_RATE_LIMIT_DISABLED", "").lower() == "true":
        return {"allowed": True, "rate_limit_disabled": True}
    
    client_id = request.client.host if request.client else "unknown"
    
    result = _rate_limiter.check_rate_limit(client_id)
    
    if not result["allowed"]:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "Retry-After": str(result["retry_after"]),
                "X-RateLimit-Remaining": "0"
            }
        )
    
    return result


def require_permission(permission: str):
    """
    Decorator to require specific permission.
    
    Usage:
        @router.get("/admin/keys")
        @require_permission("admin:keys")
        async def list_keys(api_key: dict = Depends(get_api_key)):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, api_key: dict = None, **kwargs):
            if not api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            permissions = api_key.get("permissions", [])
            
            # Wildcard permission grants all access
            if "*" in permissions:
                return await func(*args, api_key=api_key, **kwargs)
            
            # Check specific permission
            if permission not in permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission '{permission}' required"
                )
            
            return await func(*args, api_key=api_key, **kwargs)
        
        return wrapper
    return decorator


# =============================================================================
# Initialization
# =============================================================================

def init_security():
    """
    Initialize security module.
    
    Called at startup to set up default keys.
    """
    # Generate a default key for the UI if not in production
    if os.getenv("CORE_ENV", "development") != "production":
        if not _API_KEYS:
            key = generate_api_key(
                name="core-ui-dev",
                description="Development key for CORE UI",
                permissions=["*"]
            )
            logger.info(f"Generated development API key: {key[:20]}...")
    
    logger.info(f"Security module initialized. Keys registered: {len(_API_KEYS)}")
