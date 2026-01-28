"""
CORE Admin Controller - System administration endpoints.

Provides:
- Health check aggregation
- API key management
- Webhook management
- Metrics and statistics
- Run history and management

These endpoints require authentication (except health).

RSI TODO: Add audit logging for all admin actions
RSI TODO: Add role-based access control
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Depends, Query
from pydantic import BaseModel, Field

from app.core.security import get_api_key, generate_api_key, list_api_keys, revoke_api_key
from app.core.health import get_full_health, quick_health
from app.core.middleware import get_metrics
from app.services.webhook_service import get_webhook_service, WebhookEvent
from app.services.model_router import get_model_router
from app.repository import run_repository
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# =============================================================================
# Health Endpoints (No Auth Required)
# =============================================================================

@router.get("/health")
async def admin_health_check() -> Dict[str, str]:
    """
    Quick health check for load balancers.
    
    Returns simple status - use /admin/health/full for details.
    """
    return await quick_health()


@router.get("/health/full")
async def full_health_check() -> Dict[str, Any]:
    """
    Comprehensive health check for all services.
    
    Checks:
    - Database connectivity
    - Redis connectivity
    - Ollama availability
    - WebSocket manager status
    - Engine state
    """
    return await get_full_health()


# =============================================================================
# API Key Management (Auth Required)
# =============================================================================

class CreateKeyRequest(BaseModel):
    """Request to create a new API key."""
    name: str = Field(..., description="Identifier for the key")
    description: str = Field(default="", description="Optional description")
    permissions: List[str] = Field(default=["*"], description="Allowed operations")


class CreateKeyResponse(BaseModel):
    """Response with the new API key."""
    key: str = Field(..., description="The API key (save this - shown only once)")
    name: str
    message: str


@router.post("/keys", response_model=CreateKeyResponse)
async def create_api_key(
    request: CreateKeyRequest,
    api_key: dict = Depends(get_api_key)
) -> CreateKeyResponse:
    """
    Generate a new API key.
    
    Requires admin permissions.
    
    The returned key is shown only once - save it securely!
    """
    if "admin:keys" not in api_key.get("permissions", []) and "*" not in api_key.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission 'admin:keys' required"
        )
    
    key = generate_api_key(
        name=request.name,
        description=request.description,
        permissions=request.permissions
    )
    
    logger.info(f"API key created: {request.name} by {api_key.get('name')}")
    
    return CreateKeyResponse(
        key=key,
        name=request.name,
        message="Save this key securely - it won't be shown again"
    )


@router.get("/keys")
async def get_api_keys(api_key: dict = Depends(get_api_key)) -> Dict[str, Any]:
    """
    List all registered API keys.
    
    Does not expose the actual keys - only metadata.
    """
    keys = list_api_keys()
    return {
        "keys": keys,
        "total": len(keys)
    }


@router.delete("/keys/{key_name}")
async def delete_api_key(
    key_name: str,
    api_key: dict = Depends(get_api_key)
) -> Dict[str, str]:
    """
    Revoke an API key by name.
    """
    if not revoke_api_key(key_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Key '{key_name}' not found"
        )
    
    logger.info(f"API key revoked: {key_name} by {api_key.get('name')}")
    
    return {"message": f"Key '{key_name}' revoked"}


# =============================================================================
# Webhook Management
# =============================================================================

class RegisterWebhookRequest(BaseModel):
    """Request to register a webhook."""
    url: str = Field(..., description="Webhook endpoint URL")
    events: List[str] = Field(..., description="Events to subscribe to")
    secret: Optional[str] = Field(None, description="HMAC secret for signature verification")
    name: Optional[str] = Field(None, description="Human-readable name")
    headers: Optional[Dict[str, str]] = Field(None, description="Additional headers to send")


@router.post("/webhooks")
async def register_webhook(
    request: RegisterWebhookRequest,
    api_key: dict = Depends(get_api_key)
) -> Dict[str, Any]:
    """
    Register a new webhook endpoint.
    
    Available events:
    - run.started
    - run.completed
    - run.failed
    - node.started
    - node.completed
    - step.executed
    - agent.status_changed
    """
    service = get_webhook_service()
    
    webhook = service.register(
        url=request.url,
        events=request.events,
        secret=request.secret,
        headers=request.headers,
        name=request.name
    )
    
    logger.info(f"Webhook registered: {webhook.name} ({webhook.id})")
    
    return webhook.to_dict()


@router.get("/webhooks")
async def list_webhooks(api_key: dict = Depends(get_api_key)) -> Dict[str, Any]:
    """
    List all registered webhooks.
    """
    service = get_webhook_service()
    webhooks = service.list_webhooks()
    
    return {
        "webhooks": [w.to_dict() for w in webhooks],
        "total": len(webhooks)
    }


@router.delete("/webhooks/{webhook_id}")
async def unregister_webhook(
    webhook_id: str,
    api_key: dict = Depends(get_api_key)
) -> Dict[str, str]:
    """
    Unregister a webhook.
    """
    service = get_webhook_service()
    
    if not service.unregister(webhook_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook '{webhook_id}' not found"
        )
    
    return {"message": f"Webhook '{webhook_id}' unregistered"}


@router.get("/webhooks/deliveries")
async def get_webhook_deliveries(
    limit: int = Query(50, ge=1, le=200),
    api_key: dict = Depends(get_api_key)
) -> Dict[str, Any]:
    """
    Get recent webhook delivery attempts.
    """
    service = get_webhook_service()
    deliveries = service.get_recent_deliveries(limit)
    
    return {
        "deliveries": deliveries,
        "total": len(deliveries)
    }


@router.post("/webhooks/test/{webhook_id}")
async def test_webhook(
    webhook_id: str,
    api_key: dict = Depends(get_api_key)
) -> Dict[str, str]:
    """
    Send a test event to a webhook.
    """
    service = get_webhook_service()
    webhook = service.get_webhook(webhook_id)
    
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook '{webhook_id}' not found"
        )
    
    await service.fire(
        event=WebhookEvent.RUN_COMPLETED,
        payload={
            "test": True,
            "message": "This is a test webhook delivery"
        }
    )
    
    return {"message": "Test event queued for delivery"}


# =============================================================================
# Metrics and Statistics
# =============================================================================

@router.get("/metrics")
async def get_request_metrics(api_key: dict = Depends(get_api_key)) -> Dict[str, Any]:
    """
    Get request metrics and statistics.
    """
    metrics = get_metrics()
    return metrics.get_stats()


@router.post("/metrics/reset")
async def reset_metrics(api_key: dict = Depends(get_api_key)) -> Dict[str, str]:
    """
    Reset request metrics.
    """
    metrics = get_metrics()
    metrics.reset()
    
    logger.info(f"Metrics reset by {api_key.get('name')}")
    
    return {"message": "Metrics reset"}


@router.get("/stats")
async def get_system_stats(api_key: dict = Depends(get_api_key)) -> Dict[str, Any]:
    """
    Get comprehensive system statistics.
    """
    # Get various stats
    webhook_service = get_webhook_service()
    model_router = get_model_router()
    metrics = get_metrics()
    
    try:
        run_stats = await run_repository.get_run_stats()
    except Exception:
        run_stats = {"error": "Could not fetch run stats"}
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "requests": metrics.get_stats(),
        "runs": run_stats,
        "webhooks": webhook_service.get_stats(),
        "models": model_router.get_usage_stats()
    }


# =============================================================================
# Run Management
# =============================================================================

@router.get("/runs")
async def list_runs(
    user_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    api_key: dict = Depends(get_api_key)
) -> Dict[str, Any]:
    """
    List engine runs with filtering.
    """
    try:
        runs = await run_repository.list_runs(
            user_id=user_id,
            status=status,
            limit=limit,
            offset=offset
        )
        
        return {
            "runs": runs,
            "total": len(runs),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Failed to list runs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list runs"
        )


@router.get("/runs/stats")
async def get_run_stats(api_key: dict = Depends(get_api_key)) -> Dict[str, Any]:
    """
    Get run statistics.
    """
    try:
        return await run_repository.get_run_stats()
    except Exception as e:
        logger.error(f"Failed to get run stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get run stats"
        )


@router.delete("/runs/cleanup")
async def cleanup_old_runs(
    days: int = Query(30, ge=1, le=365),
    api_key: dict = Depends(get_api_key)
) -> Dict[str, Any]:
    """
    Delete runs older than specified days.
    """
    try:
        deleted = await run_repository.cleanup_old_runs(days)
        
        logger.info(f"Cleaned up {deleted} old runs (>{days} days) by {api_key.get('name')}")
        
        return {
            "message": f"Deleted {deleted} runs older than {days} days",
            "deleted_count": deleted
        }
    except Exception as e:
        logger.error(f"Failed to cleanup runs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cleanup runs"
        )


# =============================================================================
# Model Management
# =============================================================================

@router.get("/models")
async def list_models(
    provider: Optional[str] = None,
    tier: Optional[str] = None,
    api_key: dict = Depends(get_api_key)
) -> Dict[str, Any]:
    """
    List available models.
    """
    from app.services.model_router import ModelProvider, ModelTier
    
    router = get_model_router()
    
    provider_enum = ModelProvider(provider) if provider else None
    tier_enum = ModelTier(tier) if tier else None
    
    models = router.list_models(provider=provider_enum, tier=tier_enum)
    
    return {
        "models": [m.to_dict() for m in models],
        "total": len(models),
        "default_model": router.default_model
    }


@router.get("/models/usage")
async def get_model_usage(api_key: dict = Depends(get_api_key)) -> Dict[str, Any]:
    """
    Get model usage statistics.
    """
    router = get_model_router()
    return router.get_usage_stats()
