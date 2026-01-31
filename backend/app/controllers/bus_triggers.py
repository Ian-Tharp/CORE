"""
REST endpoints for Bus Triggers — reactive automation.

Register pattern-matching rules that automatically spawn Council sessions
or Catalyst runs when bus messages match.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, status

from app.models.bus_models import BusMessage
from app.services.bus_triggers import (
    BusTriggerService,
    TriggerAction,
    TriggerEvaluation,
    TriggerResult,
    TriggerRule,
    get_bus_trigger_service,
)

router = APIRouter(prefix="/bus/triggers", tags=["bus-triggers"])


# =============================================================================
# TRIGGER CRUD
# =============================================================================


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=Dict[str, Any],
    summary="Register a new bus trigger",
)
async def register_trigger(rule: TriggerRule) -> Dict[str, Any]:
    """
    Register a trigger rule.

    The rule's ``pattern`` is a regex that will be tested (case-insensitive)
    against every bus message's topic + payload text.  When matched, the
    configured ``action`` is executed automatically.
    """
    svc = get_bus_trigger_service()
    try:
        trigger_id = svc.register_trigger(rule)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    return {
        "trigger_id": trigger_id,
        "name": rule.name,
        "action": rule.action.value,
        "pattern": rule.pattern,
        "enabled": rule.enabled,
    }


@router.get(
    "",
    status_code=status.HTTP_200_OK,
    response_model=Dict[str, Any],
    summary="List all registered triggers",
)
async def list_triggers() -> Dict[str, Any]:
    """Return every registered trigger rule."""
    svc = get_bus_trigger_service()
    triggers = svc.list_triggers()
    return {
        "triggers": [t.model_dump(mode="json") for t in triggers],
        "count": len(triggers),
    }


@router.get(
    "/{trigger_id}",
    status_code=status.HTTP_200_OK,
    response_model=Dict[str, Any],
    summary="Get a single trigger by id",
)
async def get_trigger(trigger_id: str) -> Dict[str, Any]:
    """Return a single trigger rule or 404."""
    svc = get_bus_trigger_service()
    rule = svc.get_trigger(trigger_id)
    if rule is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trigger '{trigger_id}' not found",
        )
    return rule.model_dump(mode="json")


@router.delete(
    "/{trigger_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a trigger",
)
async def remove_trigger(trigger_id: str) -> None:
    """Delete a trigger by id.  Returns 404 if it doesn't exist."""
    svc = get_bus_trigger_service()
    removed = svc.remove_trigger(trigger_id)
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trigger '{trigger_id}' not found",
        )


# =============================================================================
# DRY-RUN EVALUATION
# =============================================================================


@router.post(
    "/evaluate",
    status_code=status.HTTP_200_OK,
    response_model=TriggerEvaluation,
    summary="Evaluate a message against all triggers (dry run)",
)
async def evaluate_message(message: BusMessage) -> TriggerEvaluation:
    """
    Test which triggers would fire for a given message **without**
    actually executing any actions.  Useful for debugging rules.
    """
    svc = get_bus_trigger_service()
    return await svc.evaluate_message(message)
