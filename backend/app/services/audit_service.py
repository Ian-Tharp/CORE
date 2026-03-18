"""
Audit Service — Lightweight facade for audit logging.

Provides a fire-and-forget ``log()`` helper that controllers call after
security-sensitive operations.  Failures are logged but never propagated
to the caller, so audit recording cannot break business logic.

Usage::

    from app.services.audit_service import audit

    await audit.log(
        actor=api_key.get("name", "unknown"),
        action="api_key.create",
        resource_type="api_key",
        resource_id=key_name,
        detail={"permissions": perms},
        request=request,
    )
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import Request

from app.repository import audit_repository as repo

logger = logging.getLogger(__name__)


class AuditService:
    """Thin wrapper around the audit repository with convenience helpers."""

    async def log(
        self,
        *,
        actor: str,
        action: str,
        resource_type: str | None = None,
        resource_id: str | None = None,
        detail: Dict[str, Any] | None = None,
        request: Request | None = None,
        outcome: str = "success",
    ) -> Optional[str]:
        """
        Record an audit event.  Extracts IP and correlation_id from the
        *request* when provided.

        Returns the event UUID on success, None on failure.  Never raises.
        """
        ip_address: str | None = None
        correlation_id: str | None = None

        if request is not None:
            ip_address = request.client.host if request.client else None
            correlation_id = getattr(request.state, "correlation_id", None)

        try:
            return await repo.record(
                actor=actor,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                detail=detail,
                ip_address=ip_address,
                correlation_id=correlation_id,
                outcome=outcome,
            )
        except Exception as exc:
            logger.error("Audit log failed (non-fatal): %s", exc)
            return None

    # ------------------------------------------------------------------
    # Read helpers (delegate to repository)
    # ------------------------------------------------------------------

    async def get_events(self, **kwargs) -> list:
        return await repo.get_events(**kwargs)

    async def get_summary(self, hours: int = 24) -> dict:
        return await repo.get_summary(hours=hours)

    async def prune(self, keep_days: int = 90) -> int:
        return await repo.prune_old_events(keep_days=keep_days)


# Global singleton
audit = AuditService()