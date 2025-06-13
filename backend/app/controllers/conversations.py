from __future__ import annotations

"""REST endpoints for managing chat conversations."""

from fastapi import APIRouter, HTTPException, status
from typing import List

from repository.conversation_repository import (
    list_conversations,
    create_conversation,
    get_conversation,
    update_title,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("/", status_code=status.HTTP_200_OK)
async def get_conversations() -> List[dict]:
    """Return a list of all conversations (id, title, message count)."""
    convs = await list_conversations()
    return [
        {
            "id": c["id"],
            "title": c.get("title", ""),
            "messages": len(c.get("messages", [])),
        }
        for c in convs
    ]


@router.post("/", status_code=status.HTTP_201_CREATED)
async def post_conversation():
    """Create a new empty conversation and return its id."""
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


@router.patch("/{conv_id}", status_code=status.HTTP_204_NO_CONTENT)
async def patch_conversation(conv_id: str, payload: dict):
    """Update mutable fields of a conversation (currently only title)."""
    title = payload.get("title")
    if not title:
        raise HTTPException(status_code=400, detail="'title' is required")

    await update_title(conv_id, title)
    return None
