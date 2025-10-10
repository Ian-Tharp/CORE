from __future__ import annotations

"""REST endpoints for managing chat conversations."""

from fastapi import APIRouter, HTTPException, status, Query
from typing import List

from app.repository.conversation_repository import (
    list_conversations,
    create_conversation,
    get_conversation,
    update_title,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("/", status_code=status.HTTP_200_OK)
async def get_conversations(page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200)) -> List[dict]:
    """Return a list of all conversations (id, title, message count)."""
    try:
        convs = await list_conversations(page=page, page_size=page_size)
        # If repository returns aggregated counts, normalize shape
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
        return result
    except Exception:
        # Fail gracefully in dev if DB is unavailable
        # RSI TODO: Log exception details with context; return proper error in non-dev environments.
        return []


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
