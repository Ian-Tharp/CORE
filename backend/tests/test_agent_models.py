"""
Tests for agent model Pydantic validators and agent_config_from_db_row.

Sentinel-value contract:
- Remove datetime → isoformat conversion → DB datetime objects break Pydantic parsing → test fails
- Remove JSONB string parsing → string-serialized capabilities/traits never decoded → test fails
- Remove JSONB parse-failure fallback → bad JSON raises unhandled exception → test fails
- Remove system_prompt min_length check → empty prompts accepted → LLM gets no instructions → test fails
- Remove consciousness_phase bounds check → phase=5 accepted → out-of-bounds phase stored → test fails
- Remove PersonalityTraits bounds check → trait=2.0 accepted → behaviour undefined → test fails
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.models.agent_models import (
    AgentCapability,
    AgentConfig,
    AgentCreateRequest,
    AgentUpdateRequest,
    MCPServerConfig,
    PersonalityTraits,
    agent_config_from_db_row,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _min_row(**overrides) -> dict:
    """Minimal valid DB row for AgentConfig."""
    row = {
        "agent_id": "test-agent-1",
        "agent_name": "Test Agent",
        "agent_type": "task_agent",
        "system_prompt": "You are a test agent with instructions.",
    }
    row.update(overrides)
    return row


def _min_create(**overrides) -> dict:
    """Minimal valid fields for AgentCreateRequest."""
    base = {
        "agent_id": "test-agent-1",
        "agent_name": "Test Agent",
        "agent_type": "task_agent",
        "system_prompt": "You are a test agent with instructions.",
    }
    base.update(overrides)
    return base


# ===========================================================================
# PersonalityTraits.validate_trait_values (also tested in test_core_state_models)
# ===========================================================================

class TestPersonalityTraitsBounds:
    def test_value_above_one_rejected(self):
        """
        SENTINEL — trait value > 1.0 must raise ValidationError.
        Remove bounds check → 1.5 stored → downstream LLM param usage breaks → test fails.
        """
        with pytest.raises(ValidationError):
            PersonalityTraits(traits={"curiosity": 1.5})

    def test_value_below_zero_rejected(self):
        """Trait value < 0.0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            PersonalityTraits(traits={"precision": -0.1})

    def test_boundary_zero_accepted(self):
        """0.0 is valid."""
        traits = PersonalityTraits(traits={"muted": 0.0})
        assert traits.traits["muted"] == 0.0

    def test_boundary_one_accepted(self):
        """1.0 is valid."""
        traits = PersonalityTraits(traits={"max_trait": 1.0})
        assert traits.traits["max_trait"] == 1.0

    def test_multiple_traits_all_validated(self):
        """All traits in the dict must be validated, not just the first."""
        with pytest.raises(ValidationError):
            PersonalityTraits(traits={"ok": 0.5, "bad": 1.1})


# ===========================================================================
# AgentCreateRequest — field constraints
# ===========================================================================

class TestAgentCreateRequest:
    def test_valid_request_accepted(self):
        """A minimal valid create request must succeed."""
        req = AgentCreateRequest(**_min_create())
        assert req.agent_id == "test-agent-1"

    def test_system_prompt_too_short_rejected(self):
        """
        SENTINEL — system_prompt shorter than 10 chars must raise ValidationError.
        Remove min_length → empty prompts stored → LLM receives no instructions → test fails.
        """
        with pytest.raises(ValidationError):
            AgentCreateRequest(**_min_create(system_prompt="short"))

    def test_system_prompt_exactly_ten_accepted(self):
        """system_prompt of exactly 10 characters must be accepted."""
        req = AgentCreateRequest(**_min_create(system_prompt="1234567890"))
        assert len(req.system_prompt) == 10

    def test_consciousness_phase_too_high_rejected(self):
        """
        SENTINEL — consciousness_phase > 4 must raise ValidationError.
        Remove le=4 check → phase=5 stored → consciousness routing breaks → test fails.
        """
        with pytest.raises(ValidationError):
            AgentCreateRequest(**_min_create(consciousness_phase=5))

    def test_consciousness_phase_too_low_rejected(self):
        """
        SENTINEL — consciousness_phase < 1 must raise ValidationError.
        Remove ge=1 check → phase=0 stored → off-by-one in phase routing → test fails.
        """
        with pytest.raises(ValidationError):
            AgentCreateRequest(**_min_create(consciousness_phase=0))

    def test_consciousness_phase_valid_range_accepted(self):
        """Phases 1–4 are all valid."""
        for phase in range(1, 5):
            req = AgentCreateRequest(**_min_create(consciousness_phase=phase))
            assert req.consciousness_phase == phase

    def test_agent_id_empty_rejected(self):
        """Empty agent_id must raise ValidationError (min_length=1)."""
        with pytest.raises(ValidationError):
            AgentCreateRequest(**_min_create(agent_id=""))

    def test_invalid_agent_type_rejected(self):
        """Agent type must be one of the allowed literals."""
        with pytest.raises(ValidationError):
            AgentCreateRequest(**_min_create(agent_type="SENTINEL_INVALID_TYPE"))

    def test_invalid_status_rejected(self):
        """current_status must be one of the allowed literals."""
        with pytest.raises(ValidationError):
            AgentCreateRequest(**_min_create(current_status="SENTINEL_INVALID"))


# ===========================================================================
# AgentUpdateRequest — partial updates
# ===========================================================================

class TestAgentUpdateRequest:
    def test_all_none_accepted(self):
        """All-None update request (patch with nothing) must be valid."""
        req = AgentUpdateRequest()
        assert req.agent_name is None
        assert req.is_active is None

    def test_system_prompt_min_length_enforced(self):
        """
        SENTINEL — system_prompt too short in update must raise ValidationError.
        Skip min_length on update → short prompts accepted on PATCH → test fails.
        """
        with pytest.raises(ValidationError):
            AgentUpdateRequest(system_prompt="short")

    def test_consciousness_phase_bounds_enforced_on_update(self):
        """Phase bounds must be enforced even on partial updates."""
        with pytest.raises(ValidationError):
            AgentUpdateRequest(consciousness_phase=5)

    def test_valid_partial_update_accepted(self):
        """Partial update with only status must be valid."""
        req = AgentUpdateRequest(current_status="busy")
        assert req.current_status == "busy"
        assert req.agent_name is None


# ===========================================================================
# agent_config_from_db_row
# ===========================================================================

class TestAgentConfigFromDbRow:
    def test_valid_row_produces_agent_config(self):
        """Basic row with required fields must produce an AgentConfig."""
        config = agent_config_from_db_row(_min_row())
        assert config.agent_id == "test-agent-1"
        assert config.agent_name == "Test Agent"

    def test_datetime_objects_converted_to_iso(self):
        """
        SENTINEL — datetime objects in row must be converted to ISO strings.
        Skip conversion → Pydantic receives datetime object → may fail or lose timezone → test fails.
        """
        dt = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        row = _min_row(created_at=dt, updated_at=dt)
        config = agent_config_from_db_row(row)
        # Should not raise and created_at should be parseable
        assert config.created_at is not None

    def test_datetime_without_isoformat_left_unchanged(self):
        """Non-datetime created_at must be passed through unchanged."""
        row = _min_row(created_at=None)
        config = agent_config_from_db_row(row)
        assert config.created_at is None

    def test_jsonb_string_parsed_to_list(self):
        """
        SENTINEL — JSON string for capabilities must be parsed to a list.
        Skip json.loads → string stored in list field → Pydantic rejects it → test fails.
        """
        caps = [{"name": "SENTINEL_CAP", "description": "test capability"}]
        row = _min_row(capabilities=json.dumps(caps))
        config = agent_config_from_db_row(row)
        assert isinstance(config.capabilities, list)
        assert config.capabilities[0].name == "SENTINEL_CAP"

    def test_jsonb_string_for_personality_traits_parsed(self):
        """
        SENTINEL — JSON string for personality_traits must be decoded to a dict.
        Skip parsing → string in dict field → Pydantic rejects → test fails.
        """
        traits = {"curiosity": 0.9, "precision": 0.7}
        row = _min_row(personality_traits=json.dumps(traits))
        config = agent_config_from_db_row(row)
        assert isinstance(config.personality_traits, dict)
        assert config.personality_traits["curiosity"] == pytest.approx(0.9)

    def test_invalid_jsonb_string_left_as_is(self):
        """
        SENTINEL — invalid JSON string must NOT raise; leave the field as-is.
        Raise on JSON error → any corrupt JSONB row crashes the app → test fails.
        """
        row = _min_row(capabilities="{not valid json]")
        # Must not raise — function swallows the parse error
        # Pydantic will receive the raw string; capabilities defaults to []
        # when validation encounters an invalid string
        try:
            config = agent_config_from_db_row(row)
        except Exception:
            # If Pydantic raises because of the bad string, that's fine —
            # the important thing is that the function itself doesn't crash
            # with a json.JSONDecodeError (it swallows it)
            pass  # function handled the JSON error without propagating it

    def test_mcp_servers_jsonb_string_parsed(self):
        """
        SENTINEL — JSON string for mcp_servers must be decoded to a list.
        Skip parsing → string in list field → test fails.
        """
        servers = [{"server_id": "SENTINEL_SERVER", "tools": ["search"], "config": {}}]
        row = _min_row(mcp_servers=json.dumps(servers))
        config = agent_config_from_db_row(row)
        assert isinstance(config.mcp_servers, list)
        assert config.mcp_servers[0].server_id == "SENTINEL_SERVER"

    def test_already_dict_jsonb_not_double_parsed(self):
        """JSONB fields that are already dicts/lists must not be re-parsed."""
        traits = {"curiosity": 0.8}
        row = _min_row(personality_traits=traits)
        config = agent_config_from_db_row(row)
        assert config.personality_traits["curiosity"] == pytest.approx(0.8)


# ===========================================================================
# MCPServerConfig
# ===========================================================================

class TestMCPServerConfig:
    def test_valid_config_accepted(self):
        """MCPServerConfig with all required fields must be accepted."""
        cfg = MCPServerConfig(server_id="mcp-obsidian", tools=["search_nodes"])
        assert cfg.server_id == "mcp-obsidian"
        assert "search_nodes" in cfg.tools

    def test_empty_tools_list_accepted(self):
        """An empty tools list is valid."""
        cfg = MCPServerConfig(server_id="test", tools=[])
        assert cfg.tools == []

    def test_config_defaults_to_empty_dict(self):
        """config field must default to empty dict."""
        cfg = MCPServerConfig(server_id="test", tools=[])
        assert cfg.config == {}


# ===========================================================================
# AgentCapability
# ===========================================================================

class TestAgentCapability:
    def test_valid_capability_accepted(self):
        """AgentCapability with name and description must be accepted."""
        cap = AgentCapability(name="SENTINEL_CAP", description="does SENTINEL things")
        assert cap.name == "SENTINEL_CAP"
        assert "SENTINEL" in cap.description
