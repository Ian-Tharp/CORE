"""
Comprehensive tests for SpawnTemplateService.

Tests cover:
- Built-in template loading
- CRUD operations (create, read, update, delete)
- Template filtering (by role, tag, builtin_only)
- Spawn from template with overrides
- Edge cases and error handling
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime, timezone

from app.services.spawn_template_service import SpawnTemplateService
from app.models.spawn_template_models import (
    SpawnFromTemplateRequest,
    SpawnTemplate,
    SpawnTemplateCreate,
    SpawnTemplateUpdate,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def service():
    """Fresh SpawnTemplateService instance with built-in templates."""
    return SpawnTemplateService()


@pytest.fixture
def create_request():
    """Valid SpawnTemplateCreate request."""
    return SpawnTemplateCreate(
        name="Test Agent",
        description="A test agent template",
        role="tester",
        agent_type="task_agent",
        system_prompt="You are a test agent. Follow instructions carefully.",
        personality_traits={"precision": 0.9, "creativity": 0.3},
        capabilities=[],
        interests=["testing", "qa"],
        tags=["test", "custom"],
        version="1.0.0",
        author="test-suite",
    )


# =============================================================================
# BUILT-IN TEMPLATES
# =============================================================================


class TestBuiltinTemplates:
    def test_builtins_loaded_on_init(self, service):
        templates = service.list_templates()
        assert len(templates) >= 4  # researcher, analyst, coordinator, writer

    def test_builtin_ids_are_prefixed(self, service):
        for tmpl in service.list_templates(builtin_only=True):
            assert tmpl.id.startswith("tmpl-")

    def test_builtin_templates_are_marked(self, service):
        for tmpl in service.list_templates(builtin_only=True):
            assert tmpl.is_builtin is True

    def test_researcher_template_exists(self, service):
        tmpl = service.get_template("tmpl-researcher")
        assert tmpl is not None
        assert tmpl.role == "researcher"
        assert tmpl.name == "Researcher"

    def test_analyst_template_exists(self, service):
        tmpl = service.get_template("tmpl-analyst")
        assert tmpl is not None
        assert tmpl.role == "analyst"

    def test_coordinator_template_exists(self, service):
        tmpl = service.get_template("tmpl-coordinator")
        assert tmpl is not None
        assert tmpl.role == "coordinator"

    def test_writer_template_exists(self, service):
        tmpl = service.get_template("tmpl-writer")
        assert tmpl is not None
        assert tmpl.role == "writer"

    def test_world_agent_templates_exist(self, service):
        expected = {
            "world_lore_architect",
            "canon_continuity_auditor",
            "world_connection_cartographer",
            "visual_prompt_director",
            "inhabitant_culture_designer",
        }
        roles = {tmpl.role for tmpl in service.list_templates(tag="procedural-worlds")}
        assert expected.issubset(roles)

    def test_builtins_have_system_prompts(self, service):
        for tmpl in service.list_templates(builtin_only=True):
            assert tmpl.system_prompt
            assert len(tmpl.system_prompt) > 10

    def test_builtins_have_capabilities(self, service):
        for tmpl in service.list_templates(builtin_only=True):
            assert len(tmpl.capabilities) > 0

    def test_builtins_have_personality_traits(self, service):
        for tmpl in service.list_templates(builtin_only=True):
            assert len(tmpl.personality_traits) > 0
            for val in tmpl.personality_traits.values():
                assert 0.0 <= val <= 1.0


# =============================================================================
# CREATE
# =============================================================================


class TestCreateTemplate:
    def test_create_returns_template(self, service, create_request):
        result = service.create_template(create_request)
        assert isinstance(result, SpawnTemplate)
        assert result.name == "Test Agent"
        assert result.role == "tester"

    def test_create_generates_unique_id(self, service, create_request):
        t1 = service.create_template(create_request)
        t2 = service.create_template(create_request)
        assert t1.id != t2.id

    def test_created_template_is_not_builtin(self, service, create_request):
        result = service.create_template(create_request)
        assert result.is_builtin is False

    def test_created_template_is_retrievable(self, service, create_request):
        created = service.create_template(create_request)
        retrieved = service.get_template(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.name == created.name

    def test_create_preserves_all_fields(self, service, create_request):
        result = service.create_template(create_request)
        assert result.description == "A test agent template"
        assert result.agent_type == "task_agent"
        assert result.personality_traits == {"precision": 0.9, "creativity": 0.3}
        assert result.interests == ["testing", "qa"]
        assert result.tags == ["test", "custom"]
        assert result.version == "1.0.0"
        assert result.author == "test-suite"

    def test_create_sets_created_at(self, service, create_request):
        result = service.create_template(create_request)
        assert result.created_at is not None
        assert isinstance(result.created_at, datetime)

    def test_create_increases_template_count(self, service, create_request):
        before = len(service.list_templates())
        service.create_template(create_request)
        after = len(service.list_templates())
        assert after == before + 1


# =============================================================================
# READ / GET
# =============================================================================


class TestGetTemplate:
    def test_get_existing_template(self, service):
        tmpl = service.get_template("tmpl-researcher")
        assert tmpl is not None

    def test_get_nonexistent_returns_none(self, service):
        result = service.get_template("tmpl-nonexistent-xyz")
        assert result is None

    def test_get_empty_id_returns_none(self, service):
        result = service.get_template("")
        assert result is None


# =============================================================================
# LIST / FILTER
# =============================================================================


class TestListTemplates:
    def test_list_all_returns_builtins(self, service):
        templates = service.list_templates()
        assert len(templates) >= 4

    def test_filter_by_role(self, service):
        results = service.list_templates(role="researcher")
        assert len(results) == 1
        assert results[0].role == "researcher"

    def test_filter_by_nonexistent_role(self, service):
        results = service.list_templates(role="nonexistent-role")
        assert len(results) == 0

    def test_filter_by_tag(self, service):
        results = service.list_templates(tag="core-builtin")
        assert len(results) >= 4  # all builtins have this tag

    def test_filter_by_nonexistent_tag(self, service):
        results = service.list_templates(tag="nonexistent-tag")
        assert len(results) == 0

    def test_builtin_only_filter(self, service, create_request):
        service.create_template(create_request)
        all_templates = service.list_templates()
        builtin_only = service.list_templates(builtin_only=True)
        assert len(builtin_only) < len(all_templates)
        for tmpl in builtin_only:
            assert tmpl.is_builtin is True

    def test_combined_filters(self, service):
        results = service.list_templates(role="researcher", tag="core-builtin")
        assert len(results) == 1
        assert results[0].role == "researcher"

    def test_filter_role_and_tag_mismatch(self, service):
        results = service.list_templates(role="researcher", tag="writing")
        assert len(results) == 0


# =============================================================================
# UPDATE
# =============================================================================


class TestUpdateTemplate:
    def test_update_name(self, service):
        patch = SpawnTemplateUpdate(name="Updated Researcher")
        result = service.update_template("tmpl-researcher", patch)
        assert result is not None
        assert result.name == "Updated Researcher"

    def test_update_sets_updated_at(self, service):
        patch = SpawnTemplateUpdate(name="Updated")
        result = service.update_template("tmpl-researcher", patch)
        assert result.updated_at is not None

    def test_update_nonexistent_returns_none(self, service):
        patch = SpawnTemplateUpdate(name="Ghost")
        result = service.update_template("tmpl-nonexistent", patch)
        assert result is None

    def test_update_empty_patch_returns_unchanged(self, service):
        original = service.get_template("tmpl-researcher")
        patch = SpawnTemplateUpdate()
        result = service.update_template("tmpl-researcher", patch)
        assert result.name == original.name
        assert result.role == original.role

    def test_update_multiple_fields(self, service):
        patch = SpawnTemplateUpdate(
            name="Super Researcher",
            description="Enhanced research capabilities",
            tags=["research", "enhanced", "v2"],
        )
        result = service.update_template("tmpl-researcher", patch)
        assert result.name == "Super Researcher"
        assert result.description == "Enhanced research capabilities"
        assert "enhanced" in result.tags

    def test_update_persists_in_store(self, service):
        patch = SpawnTemplateUpdate(name="Persisted Name")
        service.update_template("tmpl-researcher", patch)
        retrieved = service.get_template("tmpl-researcher")
        assert retrieved.name == "Persisted Name"

    def test_update_custom_template(self, service, create_request):
        created = service.create_template(create_request)
        patch = SpawnTemplateUpdate(name="Updated Custom")
        result = service.update_template(created.id, patch)
        assert result.name == "Updated Custom"


# =============================================================================
# DELETE
# =============================================================================


class TestDeleteTemplate:
    def test_delete_existing_returns_true(self, service, create_request):
        created = service.create_template(create_request)
        assert service.delete_template(created.id) is True

    def test_delete_removes_from_store(self, service, create_request):
        created = service.create_template(create_request)
        service.delete_template(created.id)
        assert service.get_template(created.id) is None

    def test_delete_nonexistent_returns_false(self, service):
        assert service.delete_template("tmpl-nonexistent") is False

    def test_delete_builtin_returns_true(self, service):
        assert service.delete_template("tmpl-researcher") is True
        assert service.get_template("tmpl-researcher") is None

    def test_delete_decreases_count(self, service, create_request):
        created = service.create_template(create_request)
        before = len(service.list_templates())
        service.delete_template(created.id)
        after = len(service.list_templates())
        assert after == before - 1


# =============================================================================
# SPAWN FROM TEMPLATE
# =============================================================================


class TestSpawnFromTemplate:
    @pytest.fixture
    def spawn_request(self):
        return SpawnFromTemplateRequest(
            agent_id="agent-test-001",
        )

    @pytest.mark.asyncio
    async def test_spawn_creates_agent(self, service, spawn_request):
        with patch(
            "app.services.spawn_template_service.create_agent", new_callable=AsyncMock
        ) as mock_create:
            result = await service.spawn_from_template("tmpl-researcher", spawn_request)
            assert result.agent_id == "agent-test-001"
            assert result.template_id == "tmpl-researcher"
            assert result.template_name == "Researcher"
            assert result.role == "researcher"
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_spawn_nonexistent_template_raises(self, service, spawn_request):
        with pytest.raises(ValueError, match="not found"):
            await service.spawn_from_template("tmpl-nonexistent", spawn_request)

    @pytest.mark.asyncio
    async def test_spawn_with_name_override(self, service):
        req = SpawnFromTemplateRequest(
            agent_id="agent-custom-name",
            agent_name="My Custom Agent",
        )
        with patch(
            "app.services.spawn_template_service.create_agent", new_callable=AsyncMock
        ):
            result = await service.spawn_from_template("tmpl-researcher", req)
            assert result.agent_name == "My Custom Agent"

    @pytest.mark.asyncio
    async def test_spawn_default_name_includes_template(self, service, spawn_request):
        with patch(
            "app.services.spawn_template_service.create_agent", new_callable=AsyncMock
        ):
            result = await service.spawn_from_template("tmpl-researcher", spawn_request)
            assert "Researcher" in result.agent_name

    @pytest.mark.asyncio
    async def test_spawn_with_system_prompt_append(self, service):
        req = SpawnFromTemplateRequest(
            agent_id="agent-prompt-test",
            system_prompt_append="Also focus on machine learning topics.",
        )
        with patch(
            "app.services.spawn_template_service.create_agent", new_callable=AsyncMock
        ) as mock_create:
            await service.spawn_from_template("tmpl-researcher", req)
            call_args = mock_create.call_args[0][0]
            assert "machine learning" in call_args.system_prompt

    @pytest.mark.asyncio
    async def test_spawn_with_personality_overrides(self, service):
        req = SpawnFromTemplateRequest(
            agent_id="agent-personality-test",
            personality_overrides={"curiosity": 0.5, "new_trait": 0.8},
        )
        with patch(
            "app.services.spawn_template_service.create_agent", new_callable=AsyncMock
        ) as mock_create:
            await service.spawn_from_template("tmpl-researcher", req)
            call_args = mock_create.call_args[0][0]
            assert call_args.personality_traits["curiosity"] == 0.5
            assert call_args.personality_traits["new_trait"] == 0.8

    @pytest.mark.asyncio
    async def test_spawn_with_extra_interests(self, service):
        req = SpawnFromTemplateRequest(
            agent_id="agent-interests-test",
            extra_interests=["quantum-computing", "biotech"],
        )
        with patch(
            "app.services.spawn_template_service.create_agent", new_callable=AsyncMock
        ) as mock_create:
            await service.spawn_from_template("tmpl-researcher", req)
            call_args = mock_create.call_args[0][0]
            assert "quantum-computing" in call_args.interests
            assert "biotech" in call_args.interests

    @pytest.mark.asyncio
    async def test_spawn_inactive_agent(self, service):
        req = SpawnFromTemplateRequest(
            agent_id="agent-inactive-test",
            is_active=False,
        )
        with patch(
            "app.services.spawn_template_service.create_agent", new_callable=AsyncMock
        ) as mock_create:
            await service.spawn_from_template("tmpl-researcher", req)
            call_args = mock_create.call_args[0][0]
            assert call_args.is_active is False

    @pytest.mark.asyncio
    async def test_spawn_result_has_spawned_at(self, service, spawn_request):
        with patch(
            "app.services.spawn_template_service.create_agent", new_callable=AsyncMock
        ):
            result = await service.spawn_from_template("tmpl-researcher", spawn_request)
            assert result.spawned_at is not None

    @pytest.mark.asyncio
    async def test_spawn_from_custom_template(self, service, create_request):
        custom = service.create_template(create_request)
        req = SpawnFromTemplateRequest(agent_id="agent-from-custom")
        with patch(
            "app.services.spawn_template_service.create_agent", new_callable=AsyncMock
        ):
            result = await service.spawn_from_template(custom.id, req)
            assert result.template_id == custom.id
            assert result.role == "tester"


# =============================================================================
# SINGLETON
# =============================================================================


class TestSingleton:
    def test_get_service_returns_instance(self):
        from app.services.spawn_template_service import (
            get_spawn_template_service,
            _service,
        )

        svc = get_spawn_template_service()
        assert isinstance(svc, SpawnTemplateService)

    def test_get_service_returns_same_instance(self):
        from app.services.spawn_template_service import get_spawn_template_service

        s1 = get_spawn_template_service()
        s2 = get_spawn_template_service()
        assert s1 is s2
