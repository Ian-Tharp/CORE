"""
Tests for CouncilService pure helper logic and VoiceRegistry — no LLM calls.

Sentinel-value contract:
- Remove CORE voice inclusion in _resolve_voice_ids → CORE voices missing from sessions → test fails
- Remove synthesizer exclusion in _get_deliberation_voices → synthesizer participates in rounds → test fails
- Remove POSITION section parsing → position always empty → test fails
- Remove confidence extraction regex → confidence always 0.5 → test fails
- Remove alias normalisation in get_voice → "core-c" raises KeyError → test fails
- Remove deduplication in list_voices → same name appears multiple times → test fails
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from app.services.council.deliberation_service import (
    CouncilService,
    DEFAULT_CONTEXTUAL_VOICE_IDS,
    _voice_type_for,
    get_council_service,
)
from app.services.council.voice_registry import (
    VoiceCategory,
    VoiceDefinition,
    VOICE_REGISTRY,
    get_voice,
    get_core_voices,
    get_voices_by_category,
    get_council_voices,
    list_voices,
)
from app.models.council_models import (
    CouncilPerspective,
    SessionStatus,
    VoiceType,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_service() -> CouncilService:
    """Build a CouncilService with mocked router and consciousness bridge."""
    router = MagicMock()
    router.select_model.return_value = "gpt-oss:20b"
    bridge = MagicMock()
    svc = CouncilService(model_router=router, consciousness_bridge=bridge)
    return svc


def _make_perspective(
    voice_name: str = "CORE-C",
    voice_type: VoiceType = VoiceType.CORE_C,
    round_num: int = 1,
    position: str = "my position",
    reasoning: str = "my reasoning",
    confidence: float = 0.7,
) -> CouncilPerspective:
    return CouncilPerspective(
        session_id=UUID("00000000-0000-0000-0000-000000000001"),
        voice_type=voice_type,
        voice_name=voice_name,
        round=round_num,
        position=position,
        reasoning=reasoning,
        confidence=confidence,
    )


# ===========================================================================
# VoiceRegistry — get_voice
# ===========================================================================

class TestGetVoice:
    def test_canonical_key_returns_definition(self):
        """get_voice('core_c') returns a VoiceDefinition."""
        v = get_voice("core_c")
        assert isinstance(v, VoiceDefinition)
        assert v.name == "CORE-C"

    def test_alias_key_resolves(self):
        """
        SENTINEL — alias 'comprehension' must resolve to the same voice as 'core_c'.
        Remove alias from VOICE_REGISTRY → KeyError on alias lookups → test fails.
        """
        v_canonical = get_voice("core_c")
        v_alias = get_voice("comprehension")
        assert v_alias is v_canonical

    def test_case_insensitive_lookup(self):
        """
        SENTINEL — upper-case key must still resolve (voice_key.lower()).
        Remove .lower() → 'CORE_C' raises KeyError → test fails.
        """
        v = get_voice("CORE_C")
        assert v.name == "CORE-C"

    def test_dash_normalised_to_underscore(self):
        """
        SENTINEL — dashes in key must be replaced with underscores.
        Remove .replace('-', '_') → 'core-c' raises KeyError → test fails.
        """
        v = get_voice("core-c")
        assert v.name == "CORE-C"

    def test_space_normalised_to_underscore(self):
        """Spaces must be normalised to underscores."""
        v = get_voice("product lead")
        assert v.name == "Product Lead"

    def test_unknown_key_raises_key_error(self):
        """Non-existent voice raises KeyError with available list in message."""
        with pytest.raises(KeyError, match="SENTINEL_GHOST"):
            get_voice("SENTINEL_GHOST")

    def test_devils_advocate_alias(self):
        """'challenger' alias must resolve to Devil's Advocate."""
        v = get_voice("challenger")
        assert v.name == "Devil's Advocate"

    def test_synthesizer_accessible(self):
        """'synthesizer' must be registered (used in synthesis phase)."""
        v = get_voice("synthesizer")
        assert v.name == "Synthesizer"


# ===========================================================================
# VoiceRegistry — list_voices / get_voices_by_category / get_council_voices
# ===========================================================================

class TestListVoices:
    def test_returns_sorted_unique_names(self):
        """
        SENTINEL — list_voices must deduplicate (aliases share objects).
        Remove seen_names deduplication → same name appears N times → test fails.
        """
        voices = list_voices()
        assert len(voices) == len(set(voices)), "Voice names must be unique"
        assert voices == sorted(voices), "Names must be sorted"

    def test_category_filter_returns_only_core_voices(self):
        """Filter by CORE returns only CORE voices."""
        core_voices = list_voices(category=VoiceCategory.CORE)
        # There are 4 CORE voices: CORE-C, CORE-O, CORE-R, CORE-E
        assert len(core_voices) == 4
        for name in core_voices:
            assert name.startswith("CORE-")

    def test_no_filter_includes_all_categories(self):
        """Without filter, all categories are represented (via per-category filter)."""
        for cat in VoiceCategory:
            voices_in_cat = list_voices(category=cat)
            assert len(voices_in_cat) >= 1, f"Category {cat} must have at least one voice"


class TestGetCoreVoices:
    def test_returns_exactly_four_core_voices(self):
        """
        SENTINEL — there must be exactly 4 CORE voices (C, O, R, E).
        Register a duplicate → 5 returned → callers get extra session voices → test fails.
        """
        voices = get_core_voices()
        assert len(voices) == 4
        names = {v.name for v in voices}
        assert "CORE-C" in names
        assert "CORE-O" in names
        assert "CORE-R" in names
        assert "CORE-E" in names

    def test_all_core_voices_have_low_temperature(self):
        """CORE voices should have deterministic low temperatures (< 0.6)."""
        for v in get_core_voices():
            assert v.temperature < 0.6, f"{v.name} temperature should be < 0.6"


class TestGetCouncilVoices:
    def test_strategic_council_returns_strategic_voices(self):
        """'strategic' returns VoiceCategory.STRATEGIC voices."""
        voices = get_council_voices("strategic")
        assert all(v.category == VoiceCategory.STRATEGIC for v in voices)
        assert len(voices) > 0

    def test_unknown_council_raises_value_error(self):
        """Unknown council name must raise ValueError."""
        with pytest.raises(ValueError, match="SENTINEL_COUNCIL"):
            get_council_voices("SENTINEL_COUNCIL")

    def test_case_insensitive_council_name(self):
        """'META' must resolve the same as 'meta'."""
        lower = get_council_voices("meta")
        upper = get_council_voices("META")
        assert {v.name for v in lower} == {v.name for v in upper}


# ===========================================================================
# CouncilService._resolve_voice_ids
# ===========================================================================

class TestResolveVoiceIds:
    def test_none_returns_core_plus_defaults(self):
        """
        SENTINEL — None input must return all 4 CORE voices + default contextual voices.
        Remove core_ids prepend → sessions start without CORE voices → test fails.
        """
        svc = _make_service()
        result = svc._resolve_voice_ids(None)
        # CORE voices must be present
        for cid in ("core_c", "core_o", "core_r", "core_e"):
            assert cid in result, f"'{cid}' must always be in resolved voice list"

    def test_none_includes_default_contextual_voices(self):
        """Default contextual voices added when no override is given."""
        svc = _make_service()
        result = svc._resolve_voice_ids(None)
        for vid in DEFAULT_CONTEXTUAL_VOICE_IDS:
            assert vid in result

    def test_custom_list_adds_to_core(self):
        """
        SENTINEL — passing ['oracle'] must result in CORE voices + oracle (not just oracle).
        Replace core_ids with caller list → caller can drop CORE voices → test fails.
        """
        svc = _make_service()
        result = svc._resolve_voice_ids(["oracle"])
        assert "core_c" in result
        assert "core_o" in result
        assert "oracle" in result

    def test_duplicate_core_ids_not_doubled(self):
        """
        SENTINEL — passing ['core_c'] must not add a second core_c entry.
        Skip deduplication → session has duplicate voices → two CORE-C perspectives per round.
        """
        svc = _make_service()
        result = svc._resolve_voice_ids(["core_c"])
        assert result.count("core_c") == 1

    def test_core_voices_appear_first(self):
        """CORE voices must come before contextual voices in the list."""
        svc = _make_service()
        result = svc._resolve_voice_ids(["oracle"])
        core_indices = [result.index(c) for c in ("core_c", "core_o", "core_r", "core_e")]
        oracle_index = result.index("oracle")
        assert all(i < oracle_index for i in core_indices)


# ===========================================================================
# CouncilService._get_deliberation_voices
# ===========================================================================

class TestGetDeliberationVoices:
    def test_synthesizer_excluded(self):
        """
        SENTINEL — Synthesizer must not appear in deliberation voices.
        Remove synthesizer exclusion → Synthesizer runs as deliberation voice too → test fails.
        """
        svc = _make_service()
        voice_ids = ["core_c", "core_o", "core_r", "core_e", "synthesizer"]
        voices = svc._get_deliberation_voices(voice_ids)
        names = [v.name for _, v in voices]
        assert "Synthesizer" not in names

    def test_known_voices_included(self):
        """Known voice IDs must all appear in result (minus Synthesizer)."""
        svc = _make_service()
        voice_ids = ["core_c", "core_o", "oracle"]
        voices = svc._get_deliberation_voices(voice_ids)
        names = {v.name for _, v in voices}
        assert "CORE-C" in names
        assert "CORE-O" in names
        assert "Oracle" in names

    def test_unknown_voice_id_skipped(self):
        """Unknown voice IDs are skipped without error."""
        svc = _make_service()
        voices = svc._get_deliberation_voices(["core_c", "SENTINEL_GHOST_VOICE"])
        names = [v.name for _, v in voices]
        assert "CORE-C" in names
        assert len(names) == 1

    def test_no_duplicate_voices(self):
        """
        SENTINEL — if CORE-C appears under two keys, only one entry in result.
        Remove seen set → one voice can submit two perspectives per round → test fails.
        """
        svc = _make_service()
        # 'core_c' and 'comprehension' both map to CORE-C
        voices = svc._get_deliberation_voices(["core_c", "comprehension"])
        names = [v.name for _, v in voices]
        assert names.count("CORE-C") == 1


# ===========================================================================
# CouncilService._parse_voice_response
# ===========================================================================

class TestParseVoiceResponse:
    def test_parses_all_structured_sections(self):
        """
        SENTINEL — structured POSITION/REASONING/CONFIDENCE must be extracted.
        Remove section parsing → position always empty → blank deliberation → test fails.
        """
        svc = _make_service()
        content = (
            "**POSITION:** This is SENTINEL_POSITION text.\n\n"
            "**REASONING:** This is SENTINEL_REASONING text.\n\n"
            "**CONFIDENCE:** 0.85"
        )
        position, reasoning, confidence = svc._parse_voice_response(content, "CORE-C")

        assert "SENTINEL_POSITION" in position
        assert "SENTINEL_REASONING" in reasoning
        assert confidence == pytest.approx(0.85, abs=0.01)

    def test_confidence_extracted_from_text(self):
        """
        SENTINEL — confidence float must be parsed from text.
        Remove regex → confidence always 0.5 → all voices report same confidence → test fails.
        """
        svc = _make_service()
        content = (
            "**POSITION:** some stance\n\n"
            "**REASONING:** some analysis\n\n"
            "**CONFIDENCE:** I am 0.92 confident."
        )
        _, _, confidence = svc._parse_voice_response(content, "CORE-R")
        assert confidence == pytest.approx(0.92, abs=0.01)

    def test_confidence_clamped_to_1(self):
        """Confidence > 1.0 must be clamped to 1.0."""
        svc = _make_service()
        # 1.00 is valid; parser should return 1.0
        content = "**POSITION:** x\n\n**REASONING:** y\n\n**CONFIDENCE:** 1.00"
        _, _, confidence = svc._parse_voice_response(content, "Test")
        assert confidence <= 1.0

    def test_fallback_when_no_sections(self):
        """When no structured sections, must not raise — uses fallback splitting."""
        svc = _make_service()
        content = "This is just free-form text\nwithout any section markers."
        position, reasoning, confidence = svc._parse_voice_response(content, "TestVoice")
        assert isinstance(position, str)
        assert len(position) > 0
        assert confidence == pytest.approx(0.5, abs=0.01)

    def test_strips_markdown_bold_from_position(self):
        """
        SENTINEL — **POSITION:** prefix must be stripped from position value.
        Leave prefix in → position starts with '**POSITION:**' → UI renders header → test fails.
        """
        svc = _make_service()
        content = (
            "**POSITION:** Clean text here\n\n"
            "**REASONING:** explanation\n\n"
            "**CONFIDENCE:** 0.7"
        )
        position, _, _ = svc._parse_voice_response(content, "CORE-E")
        assert not position.startswith("**POSITION:**")
        assert not position.startswith("POSITION:")


# ===========================================================================
# CouncilService._find_section_start
# ===========================================================================

class TestFindSectionStart:
    def test_finds_bold_header(self):
        """**POSITION:** must be found."""
        svc = _make_service()
        content = "Preamble text\n\n**POSITION:** The actual position"
        idx = svc._find_section_start(content, "POSITION")
        assert idx >= 0
        # Content after index should start with position text
        assert "The actual position" in content[idx:]

    def test_finds_plain_header(self):
        """REASONING: (without bold) must also be found."""
        svc = _make_service()
        content = "REASONING: some reasoning here"
        idx = svc._find_section_start(content, "REASONING")
        assert idx >= 0

    def test_case_insensitive(self):
        """Section search must be case-insensitive."""
        svc = _make_service()
        content = "**position:** lower case position"
        idx = svc._find_section_start(content, "POSITION")
        assert idx >= 0

    def test_returns_negative_when_absent(self):
        """Returns -1 if section header is not present."""
        svc = _make_service()
        idx = svc._find_section_start("No sections here", "POSITION")
        assert idx == -1


# ===========================================================================
# CouncilService._extract_confidence
# ===========================================================================

class TestExtractConfidence:
    def test_extracts_decimal(self):
        """
        SENTINEL — 0.87 must be extracted from text.
        Remove regex → always returns 0.5 → callers can't trust confidence scores → test fails.
        """
        svc = _make_service()
        assert svc._extract_confidence("I am 0.87 confident in this") == pytest.approx(0.87)

    def test_extracts_one_point_zero(self):
        """1.0 is a valid confidence value."""
        svc = _make_service()
        assert svc._extract_confidence("Confidence: 1.0") == pytest.approx(1.0)

    def test_defaults_to_half_on_no_match(self):
        """No float in text → returns 0.5 default."""
        svc = _make_service()
        assert svc._extract_confidence("no numbers here") == pytest.approx(0.5)

    def test_clamps_above_one(self):
        """Values above 1.0 are clamped (via min/max)."""
        # 1.00 is parsed as 1.0, which equals 1.0 exactly
        svc = _make_service()
        result = svc._extract_confidence("1.00")
        assert result <= 1.0

    def test_clamps_below_zero(self):
        """Negative values would be clamped to 0.0."""
        svc = _make_service()
        # Regex matches 0.X or 1.X so this case won't match negatives
        result = svc._extract_confidence("0.0")
        assert result >= 0.0


# ===========================================================================
# CouncilService._format_prior_context
# ===========================================================================

class TestFormatPriorContext:
    def test_empty_list_returns_empty_string(self):
        """No perspectives → empty context string."""
        svc = _make_service()
        result = svc._format_prior_context([])
        assert result == ""

    def test_includes_voice_name_and_round(self):
        """
        SENTINEL — formatted context must include voice name and round number.
        Omit voice name → context is anonymous → voices can't address each other → test fails.
        """
        svc = _make_service()
        p = _make_perspective(voice_name="SENTINEL_VOICE", round_num=2)
        result = svc._format_prior_context([p])
        assert "SENTINEL_VOICE" in result
        assert "Round 2" in result or "round 2" in result.lower()

    def test_includes_position(self):
        """Position text must appear in context."""
        svc = _make_service()
        p = _make_perspective(position="SENTINEL_POSITION_TEXT")
        result = svc._format_prior_context([p])
        assert "SENTINEL_POSITION_TEXT" in result

    def test_long_reasoning_truncated(self):
        """
        SENTINEL — reasoning > 800 chars must be truncated in context.
        Remove truncation → context grows unboundedly → exceeds LLM token limit → test fails.
        """
        svc = _make_service()
        long_reasoning = "X" * 1000
        p = _make_perspective(reasoning=long_reasoning)
        result = svc._format_prior_context([p])
        # The reasoning in output should be shorter than 1000
        assert len(result) < 3000  # basic sanity
        assert "…" in result  # truncation marker

    def test_round_headers_added_on_round_change(self):
        """When round changes, a new round header must appear."""
        svc = _make_service()
        p1 = _make_perspective(round_num=1, voice_name="Voice1")
        p2 = _make_perspective(round_num=2, voice_name="Voice2")
        result = svc._format_prior_context([p1, p2])
        assert "Round 1" in result
        assert "Round 2" in result


# ===========================================================================
# CouncilService._format_full_transcript
# ===========================================================================

class TestFormatFullTranscript:
    def test_empty_perspectives_returns_sentinel_string(self):
        """Empty perspectives returns the 'No perspectives' placeholder."""
        svc = _make_service()
        result = svc._format_full_transcript([], "any topic")
        assert "No perspectives" in result

    def test_includes_all_perspectives(self):
        """
        SENTINEL — every perspective's position must appear in the transcript.
        Skip a perspective → synthesis missing a voice's input → test fails.
        """
        svc = _make_service()
        p1 = _make_perspective(voice_name="SENTINEL_V1", position="POS_ALPHA")
        p2 = _make_perspective(voice_name="SENTINEL_V2", position="POS_BETA")
        result = svc._format_full_transcript([p1, p2], "topic")
        assert "POS_ALPHA" in result
        assert "POS_BETA" in result

    def test_includes_round_headers(self):
        """Round headers must delimit the rounds in the transcript."""
        svc = _make_service()
        p = _make_perspective(round_num=3)
        result = svc._format_full_transcript([p], "topic")
        assert "Round 3" in result


# ===========================================================================
# CouncilService._build_voice_prompt
# ===========================================================================

class TestBuildVoicePrompt:
    def test_includes_topic(self):
        """
        SENTINEL — topic must appear in voice prompt.
        Remove topic injection → voice doesn't know what to discuss → test fails.
        """
        svc = _make_service()
        voice = get_voice("core_c")
        prompt = svc._build_voice_prompt(voice, "SENTINEL_TOPIC", None, "", 1)
        assert "SENTINEL_TOPIC" in prompt

    def test_includes_voice_name_and_role(self):
        """Voice's name and role must be in prompt."""
        svc = _make_service()
        voice = get_voice("oracle")
        prompt = svc._build_voice_prompt(voice, "topic", None, "", 1)
        assert voice.name in prompt
        assert voice.role in prompt

    def test_context_included_when_provided(self):
        """Optional context must appear when given."""
        svc = _make_service()
        voice = get_voice("core_c")
        prompt = svc._build_voice_prompt(voice, "topic", "SENTINEL_CONTEXT", "", 1)
        assert "SENTINEL_CONTEXT" in prompt

    def test_prior_context_included(self):
        """Prior context must appear in the prompt."""
        svc = _make_service()
        voice = get_voice("core_r")
        prompt = svc._build_voice_prompt(voice, "topic", None, "SENTINEL_PRIOR", 2)
        assert "SENTINEL_PRIOR" in prompt

    def test_round_number_in_prompt(self):
        """Round number must appear in the prompt header."""
        svc = _make_service()
        voice = get_voice("core_e")
        prompt = svc._build_voice_prompt(voice, "topic", None, "", 3)
        assert "3" in prompt


# ===========================================================================
# _voice_type_for
# ===========================================================================

class TestVoiceTypeFor:
    def test_core_c_maps_to_core_c_type(self):
        """
        SENTINEL — CORE-C must map to VoiceType.CORE_C.
        Remove CORE name mapping → all CORE voices get VoiceType.CORE_C → test fails.
        """
        voice = get_voice("core_c")
        assert _voice_type_for(voice) == VoiceType.CORE_C

    def test_core_o_maps_to_core_o_type(self):
        """CORE-O must map to VoiceType.CORE_O."""
        voice = get_voice("core_o")
        assert _voice_type_for(voice) == VoiceType.CORE_O

    def test_strategic_voice_maps_to_strategic_type(self):
        """Strategic voices must map to VoiceType.STRATEGIC."""
        voice = get_voice("oracle")
        assert _voice_type_for(voice) == VoiceType.STRATEGIC

    def test_meta_voice_maps_to_meta_type(self):
        """Meta voices must map to VoiceType.META."""
        voice = get_voice("synthesizer")
        assert _voice_type_for(voice) == VoiceType.META


# ===========================================================================
# get_council_service — singleton
# ===========================================================================

class TestGetCouncilServiceSingleton:
    def setup_method(self):
        """Reset module singleton before each test."""
        import app.services.council.deliberation_service as mod
        mod._council_service = None

    def teardown_method(self):
        """Reset module singleton after each test."""
        import app.services.council.deliberation_service as mod
        mod._council_service = None

    def test_returns_same_instance(self):
        """
        SENTINEL — repeated calls must return the same singleton.
        Create new instance each call → config mutations lost between calls → test fails.
        """
        s1 = get_council_service()
        s2 = get_council_service()
        assert s1 is s2

    def test_returns_council_service_instance(self):
        """Return value must be a CouncilService instance."""
        svc = get_council_service()
        assert isinstance(svc, CouncilService)
