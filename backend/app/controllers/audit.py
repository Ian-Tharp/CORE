"""
Audit Log Controller

Provides read-only access to the audit trail for administrators.

Endpoints:
- GET  /audit           — Query audit events with filters
- GET  /audit/summary   — Aggregated audit summary
- DELETE /audit/prune   — Prune old audit events
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.auth import require_api_key
from app.core.security import get_api_key
from app.services.audit_service import audit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("", status_code=status.HTTP_200_OK)
async def list_audit_events(
    actor: Optional[str] = Query(None, description="Filter by actor"),
    action: Optional[str] = Query(None, description="Filter by action"),
    resource_type: Optional[str] = Query(None, description="Filter by resource type"),
    resource_id: Optional[str] = Query(None, description="Filter by resource id"),
    outcome: Optional[str] = Query(None, description="Filter by outcome"),
    since: Optional[str] = Query(None, description="ISO timestamp lower bound"),
    until: Optional[str] = Query(None, description="ISO timestamp upper bound"),
    limit: int = Query(50, ge=1, le=500, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    api_key: dict = Depends(get_api_key),
    _auth_key: str = Depends(require_api_key),
) -> Dict[str, Any]:
    """
    Query the audit log with optional filters.

    Requires authentication.  Returns events ordered by timestamp DESC.
    """
    since_dt = datetime.fromisoformat(since) if since else None
    until_dt = datetime.fromisoformat(until) if until else None

    events = await audit.get_events(
        actor=actor,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        outcome=outcome,
        since=since_dt,
        until=until_dt,
        limit=limit,
        offset=offset,
    )

    return {
        "events": events,
        "count": len(events),
        "limit": limit,
        "offset": offset,
    }


@router.get("/summary", status_code=status.HTTP_200_OK)
async def audit_summary(
    hours: int = Query(24, ge=1, le=720, description="Lookback window in hours"),
    api_key: dict = Depends(get_api_key),
    _auth_key: str = Depends(require_api_key),
) -> Dict[str, Any]:
    """
    Aggregated audit summary over a time window.

    Returns counts by action, actor, and outcome.
    """
    return await audit.get_summary(hours=hours)


@router.delete("/prune", status_code=status.HTTP_200_OK)
async def prune_audit_log(
    keep_days: int = Query(90, ge=1, le=730, description="Days of history to keep"),
    api_key: dict = Depends(get_api_key),
    _auth_key: str = Depends(require_api_key),
) -> Dict[str, Any]:
    """
    Prune old audit events.

    Deletes events older than keep_days.  Useful for storage management.
    """
    deleted = await audit.prune(keep_days=keep_days)

    # Self-audit the prune action
    await audit.log(
        actor=api_key.get("name", "unknown"),
        action="audit.prune",
        detail={"keep_days": keep_days, "deleted": deleted},
        outcome="success",
    )

    return {"deleted": deleted, "keep_days": keep_days}