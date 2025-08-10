from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

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
async def create_wiki(payload: WikiUpsertRequest) -> Dict[str, str]:
    page_id = await repo.create_wiki_page(payload.world_id, payload.title, payload.content, payload.metadata)
    return {"id": page_id}


@router.put("/wiki/{page_id}")
async def update_wiki(page_id: str, payload: WikiUpsertRequest) -> Dict[str, str]:
    try:
        await repo.update_wiki_page(page_id, title=payload.title, content=payload.content, metadata=payload.metadata or {})
        return {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to update wiki: {exc}")


@router.get("/wiki", response_model=List[WikiPageModel])
async def list_wiki(world_id: Optional[str] = None) -> List[WikiPageModel]:
    pages = await repo.list_wiki_pages(world_id)
    return [WikiPageModel(**p) for p in pages]


class CharacterCreateRequest(BaseModel):
    world_id: Optional[str] = None
    name: str
    traits: Dict[str, Any] = Field(default_factory=dict)


@router.post("/characters", response_model=Dict[str, str])
async def create_character(payload: CharacterCreateRequest) -> Dict[str, str]:
    character_id = await repo.create_character(payload.world_id, payload.name, payload.traits)
    return {"id": character_id}


class CharacterImageRequest(BaseModel):
    prompt: str
    size: str = "512x512"


@router.post("/characters/{character_id}/image", response_model=Dict[str, str])
async def generate_character_image(character_id: str, payload: CharacterImageRequest) -> Dict[str, str]:
    try:
        client = _get_openai_client()
        # Using Images API for base64 response
        result = await client.images.generate(model="gpt-image-1", prompt=payload.prompt, size=payload.size, response_format="b64_json")
        b64 = result.data[0].b64_json  # type: ignore[attr-defined]
        if not b64:
            raise RuntimeError("Image generation returned empty response")
        await repo.update_character_image(character_id, b64)
        return {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to generate image: {exc}")


