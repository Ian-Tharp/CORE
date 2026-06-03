from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.auth import require_api_key
from app.dependencies import _get_openai_client
from app.repository import creative_repository as repo


router = APIRouter(prefix="/creative", tags=["creative"])


class WikiUpsertRequest(BaseModel):
    world_id: Optional[str] = None
    title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WikiPageModel(WikiUpsertRequest):
    id: str
    created_at: str
    updated_at: str


@router.post("/wiki", response_model=Dict[str, str])
async def create_wiki(
    payload: WikiUpsertRequest, _auth: str = Depends(require_api_key)
) -> Dict[str, str]:
    page_id = await repo.create_wiki_page(
        payload.world_id, payload.title, payload.content, payload.metadata
    )
    return {"id": page_id}


@router.put("/wiki/{page_id}")
async def update_wiki(
    page_id: str, payload: WikiUpsertRequest, _auth: str = Depends(require_api_key)
) -> Dict[str, str]:
    try:
        await repo.update_wiki_page(
            page_id,
            title=payload.title,
            content=payload.content,
            metadata=payload.metadata or {},
        )
        return {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to update wiki: {exc}")


@router.get("/wiki", response_model=List[WikiPageModel])
async def list_wiki(
    world_id: Optional[str] = None, _auth: str = Depends(require_api_key)
) -> List[WikiPageModel]:
    pages = await repo.list_wiki_pages(world_id)
    return [WikiPageModel(**p) for p in pages]


class CharacterCreateRequest(BaseModel):
    world_id: Optional[str] = None
    name: str
    traits: Dict[str, Any] = Field(default_factory=dict)


class CharacterModel(BaseModel):
    id: str
    world_id: Optional[str] = None
    name: str
    traits: Dict[str, Any] = Field(default_factory=dict)
    image_b64: Optional[str] = None
    created_at: str
    updated_at: str


@router.post("/characters", response_model=Dict[str, str])
async def create_character(
    payload: CharacterCreateRequest, _auth: str = Depends(require_api_key)
) -> Dict[str, str]:
    character_id = await repo.create_character(
        payload.world_id, payload.name, payload.traits
    )
    return {"id": character_id}


@router.get("/characters", response_model=List[CharacterModel])
async def list_characters(
    world_id: Optional[str] = None, _auth: str = Depends(require_api_key)
) -> List[CharacterModel]:
    """List a world's inhabitants (with portraits) so they reload from the DB."""
    chars = await repo.list_characters(world_id)
    return [CharacterModel(**c) for c in chars]


class ImageGenRequest(BaseModel):
    prompt: str
    size: str = "1024x1024"


async def _generate_image_b64(prompt: str, size: str) -> str:
    """Generate image base64 while handling model-specific Images API options."""
    client = _get_openai_client()
    model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
    request: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "size": size,
    }
    # `gpt-image-1` returns base64 image data by default and rejects
    # `response_format`; older image models need it to avoid URL-only output.
    if not model.startswith("gpt-image"):
        request["response_format"] = "b64_json"

    result = await client.images.generate(**request)
    b64 = result.data[0].b64_json  # type: ignore[attr-defined]
    if not b64:
        raise RuntimeError("Image generation returned empty base64 image data")
    return b64


@router.post("/image", response_model=Dict[str, str])
async def generate_image(
    payload: ImageGenRequest, _auth: str = Depends(require_api_key)
) -> Dict[str, str]:
    """Generate an image from a prompt and return it as base64 (for world art, etc.)."""
    try:
        return {"b64": await _generate_image_b64(payload.prompt, payload.size)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to generate image: {exc}")


class CharacterImageRequest(BaseModel):
    prompt: str
    size: str = "512x512"


@router.post("/characters/{character_id}/image", response_model=Dict[str, str])
async def generate_character_image(
    character_id: str,
    payload: CharacterImageRequest,
    _auth: str = Depends(require_api_key),
) -> Dict[str, str]:
    try:
        b64 = await _generate_image_b64(payload.prompt, payload.size)
        await repo.update_character_image(character_id, b64)
        return {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to generate image: {exc}")
