from __future__ import annotations

"""REST endpoints for managing chat conversations."""

import asyncio
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status, Query

from app.repository.conversation_repository import (
    list_conversations,
    count_conversations,
    create_conversation,
    get_conversation,
    update_title,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])
logger = logging.getLogger(__name__)


@router.get("/", status_code=status.HTTP_200_OK)
async def get_conversations(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    search: Optional[str] = Query(None, max_length=200),
) -> dict:
    """Return paginated conversations with total count.

    Response shape: {conversations, total, page, page_size}
    """
    try:
        convs, total = await asyncio.gather(
            list_conversations(page=page, page_size=page_size, search=search),
            count_conversations(search=search),
        )
        result: List[dict] = []
        for c in convs:
            if "messages" in c and isinstance(c["messages"], int):
                result.append({"id": c["id"], "title": c.get("title", ""), "messages": c["messages"]})
            else:
                result.append({
                    "id": c["id"],
                    "title": c.get("title", ""),
                    "messages": len(c.get("messages", [])),
                })
        return {"conversations": result, "total": total, "page": page, "page_size": page_size}
    except Exception:
        logger.exception("Failed to list conversations")
        return {"conversations": [], "total": 0, "page": page, "page_size": page_size}


@router.post("/", status_code=status.HTTP_201_CREATED)
async def post_conversation():
    """Create a new empty conversation and return its id."""
    try:
        conv_id = await create_conversation()
        return {"id": conv_id}
    except Exception:
        # If schema is missing during first boot, try to create it lazily
        from app.dependencies import setup_db_schema

        await setup_db_schema()
        conv_id = await create_conversation()
        return {"id": conv_id}


@router.get("/{conv_id}", status_code=status.HTTP_200_OK)
async def get_single_conversation(conv_id: str):
    conv = await get_conversation(conv_id)
    if conv is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
        )
    return conv


from pydantic import BaseModel, Field


class PatchConversation(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)


@router.patch("/{conv_id}", status_code=status.HTTP_204_NO_CONTENT)
async def patch_conversation(conv_id: str, payload: PatchConversation):
    """Update mutable fields of a conversation (currently only title)."""
    await update_title(conv_id, payload.title)
    return None
