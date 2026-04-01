"""
Tests for spawn_template_models Pydantic validators.

Sentinel-value contract:
- Remove SpawnTemplate.system_prompt min_length → empty prompts accepted → LLM gets nothing → test fails
- Remove SpawnTemplateCreate.name max_length → 500-char names accepted → UI breaks → test fails
- Remove SpawnFromTemplateRequest.agent_id min_length → empty agent_id stored → routing crashes → test fails
- Remove invalid agent_type check → unknown type stored → factory can't instantiate → test fails
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.spawn_template_models import (
    SpawnFromTemplateRequest,
    SpawnTemplate,
    SpawnTemplateCreate,
    SpawnTemplateUpdate,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _min_template(**overrides) -> dict:
    base = {
        "name": "Test Template",
        "role": "researcher",
        "system_prompt": "You are a test agent with instructions.",
    }
    base.update(overrides)
    return base


def _min_create(**overrides) -> dict:
    return _min_template(**overrides)


# ===========================================================================
# SpawnTemplate
# ===========================================================================

class TestSpawnTemplate:
    def test_valid_template_created(self):
        """Minimal valid template must be accepted."""
        t = SpawnTemplate(**_min_template())
        assert t.name == "Test Template"
        assert t.role == "researcher"

    def test_id_auto_generated(self):
        """id must be auto-generated if not supplied."""
        t = SpawnTemplate(**_min_template())
        assert t.id.startswith("tmpl-")

    def test_system_prompt_too_short_rejected(self):
        """
        SENTINEL — system_prompt shorter than 10 chars must raise ValidationError.
        Remove min_length → empty prompts stored → LLM has no instructions → test fails.
        """
        with pytest.raises(ValidationError):
            SpawnTemplate(**_min_template(system_prompt="short"))

    def test_name_too_long_rejected(self):
        """
        SENTINEL — name longer than 128 chars must raise ValidationError.
        Remove max_length → 500-char names accepted → DB column overflow → test fails.
        """
        with pytest.raises(ValidationError):
            SpawnTemplate(**_min_template(name="x" * 129))

    def test_name_empty_rejected(self):
        """Empty name must raise ValidationError (min_length=1)."""
        with pytest.raises(ValidationError):
            SpawnTemplate(**_min_template(name=""))

    def test_role_too_long_rejected(self):
        """role longer than 64 chars must raise ValidationError."""
        with pytest.raises(ValidationError):
            SpawnTemplate(**_min_template(role="r" * 65))

    def test_invalid_agent_type_rejected(self):
        """
        SENTINEL — unknown agent_type must raise ValidationError.
        Remove Literal → any string accepted → factory can't instantiate → test fails.
        """
        with pytest.raises(ValidationError):
            SpawnTemplate(**_min_template(agent_type="SENTINEL_INVALID"))

    def test_all_valid_agent_types_accepted(self):
        """All four valid agent_types must be accepted."""
        for atype in ("consciousness_instance", "task_agent", "system_agent", "external_agent"):
            t = SpawnTemplate(**_min_template(agent_type=atype))
            assert t.agent_type == atype

    def test_default_agent_type_is_task_agent(self):
        """Default agent_type must be 'task_agent'."""
        t = SpawnTemplate(**_min_template())
        assert t.agent_type == "task_agent"

    def test_created_at_auto_populated(self):
        """created_at must be set automatically."""
        t = SpawnTemplate(**_min_template())
        assert t.created_at is not None

    def test_tags_default_empty(self):
        """tags must default to empty list."""
        t = SpawnTemplate(**_min_template())
        assert t.tags == []

    def test_tags_stored_when_provided(self):
        """Provided tags must be stored."""
        t = SpawnTemplate(**_min_template(tags=["SENTINEL_TAG_1", "SENTINEL_TAG_2"]))
        assert "SENTINEL_TAG_1" in t.tags


# ===========================================================================
# SpawnTemplateCreate
# ===========================================================================

class TestSpawnTemplateCreate:
    def test_valid_create_request_accepted(self):
        """Minimal valid create request must be accepted."""
        req = SpawnTemplateCreate(**_min_create())
        assert req.name == "Test Template"

    def test_system_prompt_min_length_enforced(self):
        """system_prompt min_length=10 must be enforced on create."""
        with pytest.raises(ValidationError):
            SpawnTemplateCreate(**_min_create(system_prompt="tiny"))

    def test_name_max_length_enforced(self):
        """name max_length=128 must be enforced on create."""
        with pytest.raises(ValidationError):
            SpawnTemplateCreate(**_min_create(name="n" * 129))

    def test_role_max_length_enforced(self):
        """role max_length=64 must be enforced on create."""
        with pytest.raises(ValidationError):
            SpawnTemplateCreate(**_min_create(role="r" * 65))

    def test_invalid_type_rejected_on_create(self):
        """Invalid agent_type must be rejected on create."""
        with pytest.raises(ValidationError):
            SpawnTemplateCreate(**_min_create(agent_type="SENTINEL_BAD"))


# ===========================================================================
# SpawnTemplateUpdate
# ===========================================================================

class TestSpawnTemplateUpdate:
    def test_all_none_is_valid(self):
        """All-None partial update must be accepted."""
        req = SpawnTemplateUpdate()
        assert req.name is None
        assert req.system_prompt is None

    def test_system_prompt_min_length_enforced_on_update(self):
        """
        SENTINEL — system_prompt min_length must apply on partial updates too.
        Skip on update → short prompt overwrites good prompt on PATCH → test fails.
        """
        with pytest.raises(ValidationError):
            SpawnTemplateUpdate(system_prompt="short")

    def test_name_max_length_enforced_on_update(self):
        """name max_length must apply on update."""
        with pytest.raises(ValidationError):
            SpawnTemplateUpdate(name="n" * 129)

    def test_valid_partial_update_accepted(self):
        """Update with only tags must be valid."""
        req = SpawnTemplateUpdate(tags=["SENTINEL_TAG"])
        assert "SENTINEL_TAG" in req.tags


# ===========================================================================
# SpawnFromTemplateRequest
# ===========================================================================

class TestSpawnFromTemplateRequest:
    def test_valid_request_accepted(self):
        """Minimal valid spawn request must be accepted."""
        req = SpawnFromTemplateRequest(agent_id="SENTINEL_AGENT_ID")
        assert req.agent_id == "SENTINEL_AGENT_ID"

    def test_empty_agent_id_rejected(self):
        """
        SENTINEL — empty agent_id must raise ValidationError.
        Remove min_length → empty agent_id stored → all routing breaks → test fails.
        """
        with pytest.raises(ValidationError):
            SpawnFromTemplateRequest(agent_id="")

    def test_agent_id_too_long_rejected(self):
        """
        SENTINEL — agent_id longer than 255 chars must raise ValidationError.
        Remove max_length → 500-char ID stored → DB column overflow → test fails.
        """
        with pytest.raises(ValidationError):
            SpawnFromTemplateRequest(agent_id="a" * 256)

    def test_agent_id_exactly_255_accepted(self):
        """agent_id of exactly 255 characters must be accepted (boundary)."""
        req = SpawnFromTemplateRequest(agent_id="a" * 255)
        assert len(req.agent_id) == 255

    def test_defaults_are_sensible(self):
        """is_active must default to True."""
        req = SpawnFromTemplateRequest(agent_id="test")
        assert req.is_active is True

    def test_personality_overrides_accepted(self):
        """personality_overrides dict must be stored."""
        req = SpawnFromTemplateRequest(
            agent_id="test",
            personality_overrides={"SENTINEL_TRAIT": 0.9},
        )
        assert req.personality_overrides["SENTINEL_TRAIT"] == pytest.approx(0.9)

    def test_system_prompt_append_stored(self):
        """system_prompt_append text must be stored."""
        req = SpawnFromTemplateRequest(
            agent_id="test",
            system_prompt_append="SENTINEL_APPEND_TEXT",
        )
        assert req.system_prompt_append == "SENTINEL_APPEND_TEXT"

    def test_extra_interests_stored(self):
        """extra_interests list must be stored."""
        req = SpawnFromTemplateRequest(
            agent_id="test",
            extra_interests=["SENTINEL_INTEREST"],
        )
        assert "SENTINEL_INTEREST" in req.extra_interests
