"""
Tests for comprehension_models.py — pure properties and to_task_kwargs.

Sentinel-value contract:
- Remove total_matches sum → count always wrong → routing decisions misled → test fails
- Remove all_matches concatenation → only one tier returned → context lost → test fails
- Remove to_task_kwargs original_input inclusion → orchestration gets no content → test fails
- Remove to_task_kwargs complexity_score → task router can't prioritise → test fails
- Remove suggested_priority bounds check → priority=0 accepted → DB constraint violated → test fails
- Remove content_must_not_be_empty → whitespace input processed → LLM gets blank prompt → test fails
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.models.comprehension_models import (
    ActionType,
    ComprehensionInput,
    ComprehensionResult,
    ComplexityScore,
    ContextMatch,
    ExtractedEntity,
    IntentAnalysis,
    MatchedCapability,
    MemoryMatch,
    SourceType,
    UrgencyLevel,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_memory(tier: str = "semantic", similarity: float = 0.8) -> MemoryMatch:
    return MemoryMatch(
        memory_id=uuid4(),
        content="test content",
        similarity=similarity,
        memory_tier=tier,
    )


def _make_intent(summary: str = "do something", action: ActionType = ActionType.COMMAND) -> IntentAnalysis:
    return IntentAnalysis(action_type=action, summary=summary, confidence=0.9)


def _make_complexity(overall: float = 0.5) -> ComplexityScore:
    return ComplexityScore(
        overall=overall, reasoning_depth=0.5, breadth=0.5, novelty=0.5
    )


def _make_comprehension_result(**kwargs) -> ComprehensionResult:
    defaults = dict(
        input=ComprehensionInput(content="SENTINEL_CONTENT", source_type=SourceType.USER),
        intent=_make_intent(),
        confidence=0.9,
    )
    defaults.update(kwargs)
    return ComprehensionResult(**defaults)


# ===========================================================================
# ComprehensionInput validator
# ===========================================================================

class TestComprehensionInputValidator:
    def test_valid_content_accepted_and_stripped(self):
        """Leading/trailing whitespace must be stripped on accept."""
        inp = ComprehensionInput(content="  SENTINEL  ", source_type=SourceType.USER)
        assert inp.content == "SENTINEL"

    def test_empty_string_rejected(self):
        """
        SENTINEL — empty string must raise ValidationError.
        Remove empty check → LLM receives empty prompt → test fails.
        """
        with pytest.raises(ValidationError, match="empty"):
            ComprehensionInput(content="", source_type=SourceType.USER)

    def test_whitespace_only_rejected(self):
        """
        SENTINEL — whitespace-only string must raise ValidationError.
        Skip strip check → ' ' accepted as input → test fails.
        """
        with pytest.raises(ValidationError, match="empty"):
            ComprehensionInput(content="   \t\n", source_type=SourceType.USER)


# ===========================================================================
# ContextMatch.total_matches / all_matches
# ===========================================================================

class TestContextMatchProperties:
    def test_total_matches_sums_all_tiers(self):
        """
        SENTINEL — total_matches must equal semantic + episodic + procedural counts.
        Return len(semantic_matches) only → other tiers invisible → routing starved → test fails.
        """
        ctx = ContextMatch(
            semantic_matches=[_make_memory("semantic")] * 3,
            episodic_matches=[_make_memory("episodic")] * 2,
            procedural_matches=[_make_memory("procedural")] * 1,
        )
        assert ctx.total_matches == 6

    def test_total_matches_zero_when_empty(self):
        """Empty context must have total_matches=0."""
        ctx = ContextMatch()
        assert ctx.total_matches == 0

    def test_all_matches_concatenates_all_tiers(self):
        """
        SENTINEL — all_matches must include matches from all three tiers.
        Return only semantic_matches → 2 tiers invisible → context incomplete → test fails.
        """
        sem = _make_memory("semantic")
        ep = _make_memory("episodic")
        proc = _make_memory("procedural")
        ctx = ContextMatch(
            semantic_matches=[sem],
            episodic_matches=[ep],
            procedural_matches=[proc],
        )
        all_m = ctx.all_matches
        assert len(all_m) == 3
        assert sem in all_m
        assert ep in all_m
        assert proc in all_m

    def test_all_matches_empty_when_no_matches(self):
        """all_matches on empty context must return empty list."""
        ctx = ContextMatch()
        assert ctx.all_matches == []

    def test_total_matches_reflects_addition(self):
        """total_matches must count all items individually."""
        ctx = ContextMatch(
            semantic_matches=[_make_memory()] * 5,
        )
        assert ctx.total_matches == 5


# ===========================================================================
# ComprehensionResult.to_task_kwargs
# ===========================================================================

class TestToTaskKwargs:
    def test_original_input_included(self):
        """
        SENTINEL — original_input in payload must equal the ComprehensionInput.content.
        Omit original_input → orchestration has no text to work with → test fails.
        """
        result = _make_comprehension_result()
        kwargs = result.to_task_kwargs()
        assert kwargs["payload"]["original_input"] == "SENTINEL_CONTENT"

    def test_task_type_is_suggested_task_type(self):
        """
        SENTINEL — task_type must match suggested_task_type.
        Hardcode 'general' → routing ignores comprehension's classification → test fails.
        """
        result = _make_comprehension_result(suggested_task_type="SENTINEL_TASK_TYPE")
        kwargs = result.to_task_kwargs()
        assert kwargs["task_type"] == "SENTINEL_TASK_TYPE"

    def test_priority_is_suggested_priority(self):
        """
        SENTINEL — priority must match suggested_priority.
        Always return 5 → comprehension's urgency assessment ignored → test fails.
        """
        result = _make_comprehension_result(suggested_priority=8)
        kwargs = result.to_task_kwargs()
        assert kwargs["priority"] == 8

    def test_complexity_score_included_in_payload(self):
        """
        SENTINEL — complexity_score must appear in payload.
        Omit it → task router can't gauge resource allocation → test fails.
        """
        result = _make_comprehension_result(
            complexity=_make_complexity(overall=0.77)
        )
        kwargs = result.to_task_kwargs()
        assert kwargs["payload"]["complexity_score"] == pytest.approx(0.77)

    def test_intent_summary_in_payload(self):
        """
        SENTINEL — intent summary must appear in payload.
        Omit it → orchestration doesn't know what was requested → test fails.
        """
        result = _make_comprehension_result(
            intent=_make_intent(summary="SENTINEL_SUMMARY")
        )
        kwargs = result.to_task_kwargs()
        assert kwargs["payload"]["intent_summary"] == "SENTINEL_SUMMARY"

    def test_action_type_value_in_payload(self):
        """
        SENTINEL — action_type value must be in payload.
        Store ActionType enum object → serialisation breaks downstream → test fails.
        """
        result = _make_comprehension_result(
            intent=_make_intent(action=ActionType.QUERY)
        )
        kwargs = result.to_task_kwargs()
        # Must be string value, not enum object
        assert kwargs["payload"]["action_type"] == ActionType.QUERY.value

    def test_comprehension_id_in_payload(self):
        """
        SENTINEL — comprehension_id must be in payload for traceability.
        Omit it → orchestration can't link back to comprehension → test fails.
        """
        result = _make_comprehension_result()
        kwargs = result.to_task_kwargs()
        assert "comprehension_id" in kwargs["payload"]
        assert isinstance(kwargs["payload"]["comprehension_id"], str)

    def test_context_matches_count_in_payload(self):
        """context_matches must reflect the number of memory matches found."""
        ctx = ContextMatch(semantic_matches=[_make_memory()] * 3)
        result = _make_comprehension_result(context=ctx)
        kwargs = result.to_task_kwargs()
        assert kwargs["payload"]["context_matches"] == 3

    def test_required_capabilities_from_suggested(self):
        """required_capabilities in kwargs must match suggested_capabilities."""
        result = _make_comprehension_result(
            suggested_capabilities=["SENTINEL_CAP_1", "SENTINEL_CAP_2"]
        )
        kwargs = result.to_task_kwargs()
        assert "SENTINEL_CAP_1" in kwargs["required_capabilities"]
        assert "SENTINEL_CAP_2" in kwargs["required_capabilities"]

    def test_source_type_value_in_payload(self):
        """source_type must be the enum value string, not the enum object."""
        result = _make_comprehension_result()
        kwargs = result.to_task_kwargs()
        assert kwargs["payload"]["source_type"] == SourceType.USER.value


# ===========================================================================
# ComprehensionResult bounds
# ===========================================================================

class TestComprehensionResultBounds:
    def test_priority_below_1_rejected(self):
        """
        SENTINEL — priority < 1 must raise ValidationError.
        Remove ge=1 → priority=0 stored → DB constraint violated → test fails.
        """
        with pytest.raises(ValidationError):
            _make_comprehension_result(suggested_priority=0)

    def test_priority_above_10_rejected(self):
        """
        SENTINEL — priority > 10 must raise ValidationError.
        Remove le=10 → priority=11 stored → out-of-range value → test fails.
        """
        with pytest.raises(ValidationError):
            _make_comprehension_result(suggested_priority=11)

    def test_priority_boundaries_accepted(self):
        """Boundary values 1 and 10 must be accepted."""
        low = _make_comprehension_result(suggested_priority=1)
        high = _make_comprehension_result(suggested_priority=10)
        assert low.suggested_priority == 1
        assert high.suggested_priority == 10

    def test_confidence_above_1_rejected(self):
        """confidence > 1.0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            _make_comprehension_result(confidence=1.1)

    def test_confidence_below_0_rejected(self):
        """confidence < 0.0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            _make_comprehension_result(confidence=-0.1)
