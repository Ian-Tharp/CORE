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


@router.post("", response_model=CreateWorldResponse)
async def create_world(payload: CreateWorldRequest) -> CreateWorldResponse:
    try:
        world_id = await repo.create_world(payload.name, origin=(payload.origin or "human"), tags=payload.tags)
        return CreateWorldResponse(id=world_id, name=payload.name)
    except Exception as exc:  # noqa: BLE001
        # If schema is out-of-date, the repo will already try a fallback. Bubble a clear message if still failing.
        raise HTTPException(status_code=400, detail=f"Failed to create world: {exc}")


@router.get("", response_model=List[Dict[str, str]])
async def list_worlds(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)) -> List[Dict[str, str]]:
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
async def list_world_snapshots(world_id: str, limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0)) -> List[Dict[str, str]]:
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


