"""
REST endpoints for Agent Spawn Templates.

Provides CRUD for reusable agent templates and a spawn endpoint to
create new agents from a template with optional overrides.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status

from app.models.spawn_template_models import (
    SpawnFromTemplateRequest,
    SpawnResult,
    SpawnTemplate,
    SpawnTemplateCreate,
    SpawnTemplateUpdate,
)
from app.services.spawn_template_service import get_spawn_template_service

router = APIRouter(prefix="/spawn-templates", tags=["spawn-templates"])


# =============================================================================
# LIST / GET
# =============================================================================


@router.get("", status_code=status.HTTP_200_OK)
async def list_templates(
    role: Optional[str] = Query(None, description="Filter by role"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    builtin_only: bool = Query(False, description="Only return built-in templates"),
) -> Dict[str, Any]:
    """List all available spawn templates with optional filters."""
    svc = get_spawn_template_service()
    templates = svc.list_templates(role=role, tag=tag, builtin_only=builtin_only)
    return {
        "templates": [t.model_dump(mode="json") for t in templates],
        "count": len(templates),
    }


@router.get("/{template_id}", status_code=status.HTTP_200_OK, response_model=SpawnTemplate)
async def get_template(template_id: str) -> SpawnTemplate:
    """Get a single spawn template by ID."""
    svc = get_spawn_template_service()
    tmpl = svc.get_template(template_id)
    if tmpl is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{template_id}' not found",
        )
    return tmpl


# =============================================================================
# CREATE / UPDATE / DELETE
# =============================================================================


@router.post("", status_code=status.HTTP_201_CREATED, response_model=SpawnTemplate)
async def create_template(req: SpawnTemplateCreate) -> SpawnTemplate:
    """Create a new custom spawn template."""
    svc = get_spawn_template_service()
    try:
        return svc.create_template(req)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create template: {exc}",
        )


@router.patch("/{template_id}", status_code=status.HTTP_200_OK, response_model=SpawnTemplate)
async def update_template(template_id: str, patch: SpawnTemplateUpdate) -> SpawnTemplate:
    """Partially update an existing spawn template."""
    svc = get_spawn_template_service()
    updated = svc.update_template(template_id, patch)
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{template_id}' not found",
        )
    return updated


@router.delete("/{template_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def delete_template(template_id: str):
    """Delete a spawn template."""
    svc = get_spawn_template_service()
    deleted = svc.delete_template(template_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{template_id}' not found",
        )


# =============================================================================
# SPAWN
# =============================================================================


@router.post(
    "/{template_id}/spawn",
    status_code=status.HTTP_201_CREATED,
    response_model=SpawnResult,
)
async def spawn_from_template(
    template_id: str, req: SpawnFromTemplateRequest
) -> SpawnResult:
    """
    Spawn a new agent from a template.

    Provide a unique ``agent_id`` and optionally override personality
    traits, append to the system prompt, or attach extra MCP servers.
    """
    svc = get_spawn_template_service()
    try:
        return await svc.spawn_from_template(template_id, req)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to spawn agent: {exc}",
        )
