from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.repository import world_repository as repo


router = APIRouter(prefix="/worlds", tags=["worlds"])


class HexWorldConfigModel(BaseModel):
    radius: float
    gridWidth: int
    gridHeight: int
    elevation: float


class LayersModel(BaseModel):
    terrain: List[Dict[str, Any]] = Field(default_factory=list)
    biome: List[Dict[str, Any]] = Field(default_factory=list)
    resources: List[Dict[str, Any]] = Field(default_factory=list)


class CreateWorldRequest(BaseModel):
    name: str
    config: HexWorldConfigModel
    origin: Optional[str] = "human"
    tags: Optional[List[str]] = None


class CreateWorldResponse(BaseModel):
    id: str
    name: str


class SnapshotRequest(BaseModel):
    config: HexWorldConfigModel
    layers: Optional[LayersModel] = None
    tiles: Optional[List[Dict[str, Any]]] = None
    preview: Optional[str] = None


class SnapshotResponse(BaseModel):
    id: str
    world_id: str


class SnapshotPayload(BaseModel):
    id: str
    created_at: str
    config: Dict[str, Any]
    layers: Optional[Dict[str, Any]] = None
    tiles: Optional[List[Dict[str, Any]]] = None
    preview: Optional[str] = None


class RenameWorldRequest(BaseModel):
    name: str


class WorldMetadataRequest(BaseModel):
    # The full TileMetadataSnapshot (per-tile metadata map + connection graph).
    snapshot: Dict[str, Any] = Field(default_factory=dict)


@router.post("", response_model=CreateWorldResponse)
async def create_world(payload: CreateWorldRequest) -> CreateWorldResponse:
    try:
        world_id = await repo.create_world(
            payload.name, origin=(payload.origin or "human"), tags=payload.tags
        )
        return CreateWorldResponse(id=world_id, name=payload.name)
    except Exception as exc:  # noqa: BLE001
        # If schema is out-of-date, the repo will already try a fallback. Bubble a clear message if still failing.
        raise HTTPException(status_code=400, detail=f"Failed to create world: {exc}")


@router.get("", response_model=List[Dict[str, str]])
async def list_worlds(
    limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)
) -> List[Dict[str, str]]:
    return await repo.list_worlds(limit=limit, offset=offset)


@router.post("/{world_id}/snapshots", response_model=SnapshotResponse)
async def create_snapshot(world_id: str, payload: SnapshotRequest) -> SnapshotResponse:
    try:
        snapshot_id = await repo.create_snapshot(
            world_id,
            config=payload.config.model_dump(),
            layers=payload.layers.model_dump() if payload.layers else None,
            tiles=payload.tiles,
            preview=payload.preview,
        )
        return SnapshotResponse(id=snapshot_id, world_id=world_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to create snapshot: {exc}")


@router.get("/{world_id}/snapshots/latest", response_model=SnapshotPayload)
async def get_latest_snapshot(world_id: str) -> SnapshotPayload:
    latest = await repo.get_latest_snapshot(world_id)
    if latest is None:
        raise HTTPException(status_code=404, detail="No snapshot found")
    return SnapshotPayload(**latest)


@router.get("/{world_id}/snapshots", response_model=List[Dict[str, str]])
async def list_world_snapshots(
    world_id: str, limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)
) -> List[Dict[str, str]]:
    return await repo.list_snapshots(world_id, limit=limit, offset=offset)


@router.delete("/{world_id}")
async def delete_world(world_id: str) -> Dict[str, str]:
    await repo.delete_world(world_id)
    return {"status": "deleted"}


@router.delete("/{world_id}/snapshots/{snapshot_id}")
async def delete_world_snapshot(world_id: str, snapshot_id: str) -> Dict[str, str]:
    await repo.delete_snapshot(world_id, snapshot_id)
    return {"status": "deleted"}


@router.patch("/{world_id}")
async def rename_world(world_id: str, payload: RenameWorldRequest) -> Dict[str, str]:
    await repo.update_world_name(world_id, payload.name)
    return {"status": "ok"}


@router.put("/{world_id}/metadata")
async def save_world_metadata(
    world_id: str, payload: WorldMetadataRequest
) -> Dict[str, str]:
    """Persist a world's tile-metadata snapshot (names, tags, notes, links, connections)."""
    await repo.save_world_metadata(world_id, payload.snapshot)
    return {"status": "ok"}


@router.get("/{world_id}/metadata", response_model=Dict[str, Any])
async def get_world_metadata(world_id: str) -> Dict[str, Any]:
    """Return a world's tile-metadata snapshot (empty object if none saved)."""
    snapshot = await repo.get_world_metadata(world_id)
    return snapshot or {}


@router.post("/{world_id}/knowledge/ingest-wiki")
async def ingest_world_wiki(world_id: str) -> Dict[str, int]:
    """Ingest this world's wiki pages into the knowledgebase (world-scoped, RAG)."""
    from app.services import knowledgebase_service as kb

    return await kb.ingest_world_wiki(world_id)


@router.get("/{world_id}/knowledge", response_model=List[Dict[str, Any]])
async def list_world_knowledge(world_id: str) -> List[Dict[str, Any]]:
    """List the knowledge documents linked to this world."""
    from app.repository import knowledgebase_repository as kb_repo

    return await kb_repo.list_documents_by_world(world_id)


class KnowledgeSearchRequest(BaseModel):
    query: str
    max_chunks: int = 8


@router.post("/{world_id}/knowledge/search")
async def search_world_knowledge(
    world_id: str, payload: KnowledgeSearchRequest
) -> Dict[str, Any]:
    """Semantic search scoped to this world's knowledge (world-filtered RAG)."""
    from app.services import knowledgebase_service as kb

    return await kb.retrieve_context_by_world(
        query=payload.query, world_id=world_id, max_chunks=payload.max_chunks
    )


class WorldAssetRequest(BaseModel):
    image_b64: str
    kind: str = "art"
    title: Optional[str] = None
    tile_index: Optional[int] = None


@router.post("/{world_id}/assets", response_model=Dict[str, str])
async def create_world_asset(
    world_id: str, payload: WorldAssetRequest
) -> Dict[str, str]:
    """Persist a generated image (world art, etc.) immediately to the DB."""
    asset_id = await repo.create_world_asset(
        world_id,
        image_b64=payload.image_b64,
        kind=payload.kind,
        title=payload.title,
        tile_index=payload.tile_index,
    )
    return {"id": asset_id}


@router.get("/{world_id}/assets", response_model=List[Dict[str, Any]])
async def list_world_assets(
    world_id: str, tile_index: Optional[int] = Query(None)
) -> List[Dict[str, Any]]:
    """List a world's assets (optionally a single tile's) so they reload from the DB."""
    return await repo.list_world_assets(world_id, tile_index)


@router.delete("/{world_id}/assets/{asset_id}")
async def delete_world_asset(world_id: str, asset_id: str) -> Dict[str, str]:
    await repo.delete_world_asset(asset_id)
    return {"status": "deleted"}


@router.get("/by-name/{name}")
async def get_world_by_name(name: str) -> Optional[Dict[str, str]]:
    """Find a world by its name. Returns the world record or null if not found."""
    world = await repo.get_world_by_name(name)
    return world
