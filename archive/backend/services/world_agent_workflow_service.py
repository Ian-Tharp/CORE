from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.repository import creative_repository, world_repository
from app.services import lore_service
from app.controllers.core_entry import _llm_or_stub


class WorldAgentLoreRequest(BaseModel):
    tile_index: int = Field(..., ge=0)
    kind: str = "Overview"
    focus: str = "an evocative encyclopedic overview of this world"
    world_name: Optional[str] = None
    user_context: Optional[str] = None
    agent_id: Optional[str] = None
    model: Optional[str] = None


class WorldAgentAuditResult(BaseModel):
    approved: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    contradictions: List[str] = Field(default_factory=list)
    missing_details: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class WorldAgentLoreResponse(BaseModel):
    title: str
    content: str
    generated_by: str
    audit: WorldAgentAuditResult


class WorldAgentLoreSaveRequest(BaseModel):
    tile_index: int = Field(..., ge=0)
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    generated_by: Optional[str] = None
    audit: Optional[WorldAgentAuditResult] = None


class WorldAgentLoreSaveResponse(BaseModel):
    id: str
    title: str


class WorldAgentAuditRequest(BaseModel):
    tile_index: int = Field(..., ge=0)
    content: str = ""
    world_name: Optional[str] = None
    user_context: Optional[str] = None


class WorldAgentAuditResponse(BaseModel):
    generated_by: str = "canon_continuity_auditor"
    audit: WorldAgentAuditResult


class WorldConnectionSuggestion(BaseModel):
    from_tile_index: int
    to_tile_index: int
    type: str
    label: str
    rationale: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class WorldAgentConnectionsRequest(BaseModel):
    tile_index: int = Field(..., ge=0)
    max_suggestions: int = Field(3, ge=1, le=8)
    world_name: Optional[str] = None
    user_context: Optional[str] = None


class WorldAgentConnectionsResponse(BaseModel):
    generated_by: str = "world_connection_cartographer"
    suggestions: List[WorldConnectionSuggestion]


class WorldAgentImagePromptRequest(BaseModel):
    tile_index: int = Field(..., ge=0)
    world_name: Optional[str] = None
    user_context: Optional[str] = None
    style: Optional[str] = None


class WorldAgentImagePromptResponse(BaseModel):
    generated_by: str = "visual_prompt_director"
    prompt: str
    palette: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)


class WorldAgentWorkflowService:
    """Runs modular world-agent workflows over selected Command Center worlds."""

    async def generate_lore(
        self, world_id: str, request: WorldAgentLoreRequest
    ) -> WorldAgentLoreResponse:
        context = await self.build_context(world_id, request)
        content = await lore_service.generate_lore_draft(
            kind=request.kind,
            focus=request.focus,
            world_name=request.world_name or context.get("world_name"),
            context=context["summary"],
            model=request.model,
            llm_runner=_llm_or_stub,
        )
        title = self.extract_title(content, request.kind)
        audit = self.audit_lore(content, context)
        return WorldAgentLoreResponse(
            title=title,
            content=content,
            generated_by=request.agent_id or "world_lore_architect",
            audit=audit,
        )

    async def save_lore(
        self, world_id: str, request: WorldAgentLoreSaveRequest
    ) -> WorldAgentLoreSaveResponse:
        page_id = await creative_repository.create_wiki_page(
            world_id,
            request.title,
            request.content,
            {
                "template": request.title,
                "source": "world_agent",
                "generated_by": request.generated_by,
                "audit": request.audit.model_dump() if request.audit else None,
                "tile_index": request.tile_index,
            },
        )
        return WorldAgentLoreSaveResponse(id=page_id, title=request.title)

    async def audit_world(
        self, world_id: str, request: WorldAgentAuditRequest
    ) -> WorldAgentAuditResponse:
        context = await self.build_context(
            world_id,
            WorldAgentLoreRequest(
                tile_index=request.tile_index,
                world_name=request.world_name,
                user_context=request.user_context,
            ),
        )
        content = request.content or context["summary"]
        return WorldAgentAuditResponse(audit=self.audit_lore(content, context))

    async def suggest_connections(
        self, world_id: str, request: WorldAgentConnectionsRequest
    ) -> WorldAgentConnectionsResponse:
        latest_snapshot = await world_repository.get_latest_snapshot(world_id) or {}
        layers = latest_snapshot.get("layers") or {}
        tile_states = self._all_tile_states(layers)
        source = tile_states.get(request.tile_index, self._tile_state(layers, request.tile_index))
        candidates: List[WorldConnectionSuggestion] = []

        for idx, state in sorted(tile_states.items()):
            if idx == request.tile_index:
                continue
            conn_type = self._connection_type(source, state)
            confidence = self._connection_confidence(source, state, request.tile_index, idx)
            candidates.append(
                WorldConnectionSuggestion(
                    from_tile_index=request.tile_index,
                    to_tile_index=idx,
                    type=conn_type,
                    label=conn_type.replace("_", " ").title(),
                    rationale=self._connection_rationale(source, state, idx, conn_type),
                    confidence=confidence,
                )
            )

        candidates.sort(key=lambda item: item.confidence, reverse=True)
        return WorldAgentConnectionsResponse(suggestions=candidates[: request.max_suggestions])

    async def generate_image_prompt(
        self, world_id: str, request: WorldAgentImagePromptRequest
    ) -> WorldAgentImagePromptResponse:
        context = await self.build_context(
            world_id,
            WorldAgentLoreRequest(
                tile_index=request.tile_index,
                world_name=request.world_name,
                user_context=request.user_context,
            ),
        )
        tile_state = context["tile_state"]
        world_name = request.world_name or context.get("world_name") or f"World {request.tile_index}"
        terrain = tile_state.get("terrain", "plain")
        biome = tile_state.get("biome", "none")
        resource = tile_state.get("resource", "none")
        style = request.style or "luminous solarpunk science-fantasy concept art"
        prompt = (
            f'Wide cinematic portrait of "{world_name}", a {biome} {terrain} world '
            f"with resource state {resource}. {style}. "
            "Show the world's defining geography, atmosphere, inhabitants or signs of culture, "
            "and a grounded futuristic interface-friendly composition. "
            "Ultra-detailed digital matte painting, volumetric light, sharp readable focal point."
        )
        return WorldAgentImagePromptResponse(
            prompt=prompt,
            palette=self._image_palette(terrain, biome, resource),
            constraints=[
                "Keep the world terrain and biome visible.",
                "Avoid unreadable clutter; preserve a clear hero silhouette.",
                "Maintain CORE's solarpunk futuristic tone.",
            ],
        )

    async def build_context(
        self, world_id: str, request: WorldAgentLoreRequest
    ) -> Dict[str, Any]:
        metadata_snapshot = await world_repository.get_world_metadata(world_id) or {}
        latest_snapshot = await world_repository.get_latest_snapshot(world_id) or {}
        wiki_pages = await creative_repository.list_wiki_pages(world_id)

        tile_metadata = self._tile_metadata(
            metadata_snapshot.get("metadata", {}),
            request.tile_index,
        )
        tile_state = self._tile_state(latest_snapshot.get("layers"), request.tile_index)

        context_parts = [
            f"world_id: {world_id}",
            f"tile_index: {request.tile_index}",
            f"world_name: {request.world_name or tile_metadata.get('name') or 'Unnamed World'}",
            f"terrain: {tile_state.get('terrain', 'plain')}",
            f"biome: {tile_state.get('biome', 'none')}",
            f"resource: {tile_state.get('resource', 'none')}",
        ]
        if tile_metadata.get("description"):
            context_parts.append(f"description: {tile_metadata['description']}")
        if tile_metadata.get("tags"):
            context_parts.append(f"tags: {', '.join(tile_metadata['tags'])}")
        if tile_metadata.get("quickNotes"):
            notes = [
                n.get("content", "")
                for n in tile_metadata["quickNotes"]
                if isinstance(n, dict) and n.get("content")
            ]
            if notes:
                context_parts.append(f"notes: {' | '.join(notes[:5])}")
        if wiki_pages:
            wiki_summary = [
                f"{p.get('title', 'Untitled')}: {str(p.get('content', ''))[:420]}"
                for p in wiki_pages[:5]
            ]
            context_parts.append(f"linked_world_wiki: {' || '.join(wiki_summary)}")
        if request.user_context:
            context_parts.append(f"user_context: {request.user_context}")

        return {
            "world_id": world_id,
            "tile_index": request.tile_index,
            "world_name": request.world_name or tile_metadata.get("name"),
            "tile_metadata": tile_metadata,
            "tile_state": tile_state,
            "wiki_pages": wiki_pages,
            "summary": "\n".join(context_parts),
        }

    def extract_title(self, content: str, fallback: str) -> str:
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip() or fallback
        return fallback

    def audit_lore(self, content: str, context: Dict[str, Any]) -> WorldAgentAuditResult:
        tile_state = context.get("tile_state", {})
        terrain = str(tile_state.get("terrain", "plain"))
        biome = str(tile_state.get("biome", "none"))
        lower = content.lower()

        missing: List[str] = []
        contradictions: List[str] = []
        suggestions: List[str] = []

        if terrain and terrain != "plain" and terrain.lower() not in lower:
            missing.append(f"Generated lore does not explicitly mention terrain '{terrain}'.")
        if biome and biome != "none" and biome.lower() not in lower:
            missing.append(f"Generated lore does not explicitly mention biome '{biome}'.")
        if len(content.strip()) < 240:
            missing.append("Generated lore is short for a durable wiki page.")
        if "# " not in content:
            missing.append("Generated lore should include a Markdown H1 title.")
        if "unknown" in lower or "not provided" in lower:
            suggestions.append("Replace placeholder uncertainty with concrete world details.")

        confidence = max(0.2, 0.94 - len(missing) * 0.1 - len(contradictions) * 0.2)
        return WorldAgentAuditResult(
            approved=confidence >= 0.7 and not contradictions,
            confidence=round(confidence, 2),
            contradictions=contradictions,
            missing_details=missing,
            suggestions=suggestions,
        )

    def _tile_metadata(self, raw_metadata: Any, tile_index: int) -> Dict[str, Any]:
        if isinstance(raw_metadata, dict):
            value = raw_metadata.get(str(tile_index)) or raw_metadata.get(tile_index)
            return value if isinstance(value, dict) else {}
        return {}

    def _tile_state(self, layers: Any, tile_index: int) -> Dict[str, str]:
        state = {"terrain": "plain", "biome": "none", "resource": "none"}
        if not isinstance(layers, dict):
            return state

        for entry in layers.get("terrain", []) or []:
            if entry.get("index") == tile_index:
                state["terrain"] = str(entry.get("state", "plain"))
        for entry in layers.get("biome", []) or []:
            if entry.get("index") == tile_index:
                state["biome"] = str(entry.get("state", "none"))
        for entry in layers.get("resources", []) or []:
            if entry.get("index") == tile_index:
                state["resource"] = str(entry.get("state", "none"))
        return state

    def _all_tile_states(self, layers: Any) -> Dict[int, Dict[str, str]]:
        states: Dict[int, Dict[str, str]] = {}
        if not isinstance(layers, dict):
            return states

        for layer_name, key in (("terrain", "terrain"), ("biome", "biome"), ("resources", "resource")):
            for entry in layers.get(layer_name, []) or []:
                index = entry.get("index")
                if not isinstance(index, int):
                    continue
                states.setdefault(index, {"terrain": "plain", "biome": "none", "resource": "none"})
                states[index][key] = str(entry.get("state", states[index][key]))
        return states

    def _connection_type(self, source: Dict[str, str], target: Dict[str, str]) -> str:
        if source.get("resource") == "node" or target.get("resource") == "node":
            return "trade"
        if source.get("biome") == target.get("biome") and source.get("biome") != "none":
            return "alliance"
        if source.get("terrain") == "water" and target.get("terrain") == "mountain":
            return "mystery"
        if source.get("terrain") != target.get("terrain"):
            return "influence"
        return "portal"

    def _connection_confidence(
        self, source: Dict[str, str], target: Dict[str, str], source_index: int, target_index: int
    ) -> float:
        score = 0.55
        if source.get("biome") == target.get("biome") and source.get("biome") != "none":
            score += 0.18
        if source.get("resource") == "node" or target.get("resource") == "node":
            score += 0.14
        if abs(source_index - target_index) <= 6:
            score += 0.08
        return round(min(score, 0.94), 2)

    def _connection_rationale(
        self, source: Dict[str, str], target: Dict[str, str], target_index: int, conn_type: str
    ) -> str:
        return (
            f"Tile {target_index} has terrain={target.get('terrain', 'plain')}, "
            f"biome={target.get('biome', 'none')}, resource={target.get('resource', 'none')}; "
            f"recommended as {conn_type} based on contrast/similarity with "
            f"source terrain={source.get('terrain', 'plain')} and biome={source.get('biome', 'none')}."
        )

    def _image_palette(self, terrain: str, biome: str, resource: str) -> List[str]:
        palette = ["cyan energy", "deep navy atmosphere", "warm solar amber"]
        if terrain == "water":
            palette.extend(["aquatic blue", "bioluminescent teal"])
        if terrain == "mountain":
            palette.extend(["basalt brown", "golden ridge light"])
        if biome == "forest":
            palette.extend(["verdant green", "moss glow"])
        if biome == "desert":
            palette.extend(["sand gold", "rust orange"])
        if biome == "tundra":
            palette.extend(["ice blue", "aurora violet"])
        if resource == "node":
            palette.append("magenta resource glow")
        return palette


def get_world_agent_workflow_service() -> WorldAgentWorkflowService:
    return WorldAgentWorkflowService()
