from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.world_agent_workflow_service import (
    WorldAgentAuditRequest,
    WorldAgentConnectionsRequest,
    WorldAgentImagePromptRequest,
    WorldAgentLoreRequest,
    WorldAgentWorkflowService,
)


@pytest.mark.asyncio
async def test_build_context_includes_selected_tile_state_and_metadata():
    service = WorldAgentWorkflowService()
    service._tile_metadata = MagicMock(  # type: ignore[method-assign]
        return_value={
            "name": "Verdant Gate",
            "description": "A living threshold world.",
            "tags": ["portal", "forest"],
            "quickNotes": [{"content": "Ancient gates bloom at dawn."}],
        }
    )
    service._tile_state = MagicMock(  # type: ignore[method-assign]
        return_value={"terrain": "mountain", "biome": "forest", "resource": "node"}
    )

    from app.services import world_agent_workflow_service as module

    original_metadata = module.world_repository.get_world_metadata
    original_snapshot = module.world_repository.get_latest_snapshot
    original_wiki = module.creative_repository.list_wiki_pages
    module.world_repository.get_world_metadata = AsyncMock(return_value={"metadata": {}})
    module.world_repository.get_latest_snapshot = AsyncMock(return_value={"layers": {}})
    module.creative_repository.list_wiki_pages = AsyncMock(
        return_value=[{"title": "Old Canon", "content": "Gatekeepers maintain the roots."}]
    )
    try:
        context = await service.build_context(
            "world-1", WorldAgentLoreRequest(tile_index=3, world_name="Verdant Gate")
        )
    finally:
        module.world_repository.get_world_metadata = original_metadata
        module.world_repository.get_latest_snapshot = original_snapshot
        module.creative_repository.list_wiki_pages = original_wiki

    assert "terrain: mountain" in context["summary"]
    assert "biome: forest" in context["summary"]
    assert "resource: node" in context["summary"]
    assert "Ancient gates bloom at dawn" in context["summary"]
    assert "Old Canon" in context["summary"]


def test_audit_flags_missing_grounding_for_structured_lore():
    service = WorldAgentWorkflowService()
    result = service.audit_lore(
        "# A Short Page\n\nThis world is mysterious.",
        {"tile_state": {"terrain": "water", "biome": "tundra"}},
    )

    assert result.approved is False
    assert any("terrain 'water'" in item for item in result.missing_details)
    assert any("biome 'tundra'" in item for item in result.missing_details)


@pytest.mark.asyncio
async def test_audit_world_returns_auditor_response():
    service = WorldAgentWorkflowService()
    service.build_context = AsyncMock(  # type: ignore[method-assign]
        return_value={"tile_state": {"terrain": "water", "biome": "none"}, "summary": "terrain: water"}
    )

    result = await service.audit_world(
        "world-1",
        WorldAgentAuditRequest(tile_index=2, content="# Water World\n\nA water world with tides."),
    )

    assert result.generated_by == "canon_continuity_auditor"
    assert result.audit.confidence >= 0.7


@pytest.mark.asyncio
async def test_suggest_connections_returns_ranked_candidates():
    service = WorldAgentWorkflowService()
    from app.services import world_agent_workflow_service as module

    original_snapshot = module.world_repository.get_latest_snapshot
    module.world_repository.get_latest_snapshot = AsyncMock(
        return_value={
            "layers": {
                "terrain": [{"index": 0, "state": "water"}, {"index": 1, "state": "mountain"}],
                "biome": [{"index": 0, "state": "forest"}, {"index": 2, "state": "forest"}],
                "resources": [{"index": 2, "state": "node"}],
            }
        }
    )
    try:
        result = await service.suggest_connections(
            "world-1",
            WorldAgentConnectionsRequest(tile_index=0, max_suggestions=2),
        )
    finally:
        module.world_repository.get_latest_snapshot = original_snapshot

    assert result.generated_by == "world_connection_cartographer"
    assert len(result.suggestions) == 2
    assert result.suggestions[0].from_tile_index == 0
    assert result.suggestions[0].type in {"trade", "alliance", "mystery", "influence", "portal"}


@pytest.mark.asyncio
async def test_generate_image_prompt_uses_world_context():
    service = WorldAgentWorkflowService()
    service.build_context = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "world_name": "Azure Canopy",
            "tile_state": {"terrain": "water", "biome": "forest", "resource": "node"},
            "summary": "terrain: water\nbiome: forest\nresource: node",
        }
    )

    result = await service.generate_image_prompt(
        "world-1", WorldAgentImagePromptRequest(tile_index=2, world_name="Azure Canopy")
    )

    assert result.generated_by == "visual_prompt_director"
    assert "Azure Canopy" in result.prompt
    assert "water" in result.prompt
    assert "bioluminescent teal" in result.palette
