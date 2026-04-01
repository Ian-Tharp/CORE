"""
Tests for app/models/comprehension_state.py — ComprehensionContext Literals,
KnowledgeBaseResult fields, CapabilitiesResult fields.

Sentinel-value contract:
- Remove ComprehensionContext.routed_to Literal → 'SENTINEL_ROUTE' accepted → bad routing stored → test fails
- Remove ComprehensionContext.reason Literal → unknown reason stored → comprehension routing broken → test fails
- Remove ComprehensionContext.input_classified default False → field required at construction → callers break → test fails
- Remove KnowledgeBaseResult.found_answer field → callers can't check if answer was found → test fails
- Remove CapabilitiesResult.can_handle field → comprehension can't determine scope → routing breaks → test fails
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.comprehension_state import (
    CapabilitiesResult,
    ComprehensionContext,
    KnowledgeBaseResult,
)


# ===========================================================================
# ComprehensionContext — Literals and defaults
# ===========================================================================

class TestComprehensionContext:
    def test_conversation_route_accepted(self):
        """
        SENTINEL — 'conversation' must be a valid routed_to value.
        Remove Literal → any string accepted → bad routing stored in state → test fails.
        """
        ctx = ComprehensionContext(routed_to="conversation", reason="WITHIN_EXPERTISE")
        assert ctx.routed_to == "conversation"

    def test_orchestration_route_accepted(self):
        """'orchestration' must be a valid routed_to value."""
        ctx = ComprehensionContext(routed_to="orchestration", reason="WITHIN_EXPERTISE")
        assert ctx.routed_to == "orchestration"

    def test_invalid_routed_to_rejected(self):
        """
        SENTINEL — unknown routed_to must raise ValidationError.
        Remove Literal → 'SENTINEL_ROUTE' accepted → LangGraph routing lookup fails → test fails.
        """
        with pytest.raises(ValidationError):
            ComprehensionContext(routed_to="SENTINEL_ROUTE", reason="WITHIN_EXPERTISE")

    def test_within_expertise_reason_accepted(self):
        """
        SENTINEL — 'WITHIN_EXPERTISE' must be a valid reason.
        Change string → reason check in comprehension agent fails → in-scope tasks rejected → test fails.
        """
        ctx = ComprehensionContext(routed_to="orchestration", reason="WITHIN_EXPERTISE")
        assert ctx.reason == "WITHIN_EXPERTISE"

    def test_outside_expertise_reason_accepted(self):
        """'OUTSIDE_EXPERTISE' must be a valid reason."""
        ctx = ComprehensionContext(routed_to="conversation", reason="OUTSIDE_EXPERTISE")
        assert ctx.reason == "OUTSIDE_EXPERTISE"

    def test_invalid_reason_rejected(self):
        """
        SENTINEL — unknown reason must raise ValidationError.
        Remove Literal → 'SENTINEL_REASON' accepted → comprehension routing uses unknown reason → test fails.
        """
        with pytest.raises(ValidationError):
            ComprehensionContext(routed_to="conversation", reason="SENTINEL_REASON")

    def test_input_classified_defaults_false(self):
        """
        SENTINEL — input_classified must default to False.
        Make required → callers without that flag crash → comprehension graph can't start → test fails.
        """
        ctx = ComprehensionContext(routed_to="conversation", reason="WITHIN_EXPERTISE")
        assert ctx.input_classified is False

    def test_knowledge_base_checked_defaults_false(self):
        """knowledge_base_checked must default to False."""
        ctx = ComprehensionContext(routed_to="conversation", reason="WITHIN_EXPERTISE")
        assert ctx.knowledge_base_checked is False

    def test_capabilities_checked_defaults_false(self):
        """capabilities_checked must default to False."""
        ctx = ComprehensionContext(routed_to="conversation", reason="WITHIN_EXPERTISE")
        assert ctx.capabilities_checked is False

    def test_all_three_flags_can_be_set_true(self):
        """All three boolean flags must be settable to True."""
        ctx = ComprehensionContext(
            routed_to="orchestration",
            reason="WITHIN_EXPERTISE",
            input_classified=True,
            knowledge_base_checked=True,
            capabilities_checked=True,
        )
        assert ctx.input_classified is True
        assert ctx.knowledge_base_checked is True
        assert ctx.capabilities_checked is True


# ===========================================================================
# KnowledgeBaseResult
# ===========================================================================

class TestKnowledgeBaseResult:
    def test_valid_result_with_answer(self):
        """
        SENTINEL — found_answer=True with answer string must be accepted.
        Remove found_answer field → callers can't determine if KB had a match → test fails.
        """
        result = KnowledgeBaseResult(
            found_answer=True,
            confidence=0.9,
            sources=["SENTINEL_SOURCE"],
            answer="SENTINEL_ANSWER",
        )
        assert result.found_answer is True
        assert result.answer == "SENTINEL_ANSWER"

    def test_not_found_with_none_answer(self):
        """found_answer=False with answer=None must be accepted."""
        result = KnowledgeBaseResult(
            found_answer=False,
            confidence=0.1,
            sources=[],
            answer=None,
        )
        assert result.found_answer is False
        assert result.answer is None

    def test_sources_stored(self):
        """sources list must be stored correctly."""
        result = KnowledgeBaseResult(
            found_answer=True,
            confidence=0.8,
            sources=["SENTINEL_SRC_1", "SENTINEL_SRC_2"],
            answer="ans",
        )
        assert "SENTINEL_SRC_1" in result.sources
        assert "SENTINEL_SRC_2" in result.sources

    def test_confidence_stored(self):
        """confidence value must be stored correctly."""
        result = KnowledgeBaseResult(
            found_answer=True,
            confidence=0.75,
            sources=[],
            answer="ans",
        )
        assert result.confidence == 0.75


# ===========================================================================
# CapabilitiesResult
# ===========================================================================

class TestCapabilitiesResult:
    def test_can_handle_true_stored(self):
        """
        SENTINEL — can_handle=True must be stored and retrievable.
        Remove can_handle → comprehension can't determine scope → all tasks routed as out-of-scope → test fails.
        """
        result = CapabilitiesResult(
            can_handle=True,
            matching_capabilities=["SENTINEL_CAP"],
            required_tools=["SENTINEL_TOOL"],
            mcp_servers=["SENTINEL_MCP"],
            confidence=0.95,
        )
        assert result.can_handle is True

    def test_can_handle_false_stored(self):
        """can_handle=False must be stored."""
        result = CapabilitiesResult(
            can_handle=False,
            matching_capabilities=[],
            required_tools=[],
            mcp_servers=[],
            confidence=0.1,
        )
        assert result.can_handle is False

    def test_matching_capabilities_stored(self):
        """
        SENTINEL — matching_capabilities must be stored.
        Return always empty → agent factory never binds right tools → test fails.
        """
        result = CapabilitiesResult(
            can_handle=True,
            matching_capabilities=["SENTINEL_CAP_A", "SENTINEL_CAP_B"],
            required_tools=[],
            mcp_servers=[],
            confidence=0.8,
        )
        assert "SENTINEL_CAP_A" in result.matching_capabilities
        assert "SENTINEL_CAP_B" in result.matching_capabilities

    def test_required_tools_stored(self):
        """required_tools must be stored."""
        result = CapabilitiesResult(
            can_handle=True,
            matching_capabilities=[],
            required_tools=["SENTINEL_TOOL"],
            mcp_servers=[],
            confidence=0.8,
        )
        assert "SENTINEL_TOOL" in result.required_tools

    def test_mcp_servers_stored(self):
        """
        SENTINEL — mcp_servers must be stored.
        Return empty always → no MCP tools bound → all tool-requiring tasks fail → test fails.
        """
        result = CapabilitiesResult(
            can_handle=True,
            matching_capabilities=[],
            required_tools=[],
            mcp_servers=["SENTINEL_MCP_SERVER"],
            confidence=0.9,
        )
        assert "SENTINEL_MCP_SERVER" in result.mcp_servers

    def test_confidence_stored(self):
        """confidence value must be stored."""
        result = CapabilitiesResult(
            can_handle=True,
            matching_capabilities=[],
            required_tools=[],
            mcp_servers=[],
            confidence=0.65,
        )
        assert result.confidence == 0.65
