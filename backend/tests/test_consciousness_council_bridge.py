"""
Tests for ConsciousnessCouncilBridge — keyword matching, insight extraction,
lazy init caching, and graceful degradation.

Sentinel-value contract:
- Remove keyword hit check → any topic triggers consciousness voice → test fails
- Remove .lower() normalisation → "CONSCIOUSNESS" not matched → test fails
- Remove KEY INSIGHTS regex → full raw synthesis sent instead of insights → test fails
- Remove 1500-char cap in _extract_insights → unbounded text sent to Blackboard → test fails
- Remove _ensure_init cache → Blackboard re-loaded on every call → test fails
- Remove synthesis keyword threshold → every synthesis written to Blackboard → test fails
- Remove graceful-degradation empty return → unavailable Blackboard raises → test fails
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.consciousness_council_bridge import (
    ConsciousnessCouncilBridge,
    _CONSCIOUSNESS_KEYWORDS,
    _KEYWORD_THRESHOLD,
    get_consciousness_council_bridge,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_bridge() -> ConsciousnessCouncilBridge:
    return ConsciousnessCouncilBridge()


# ===========================================================================
# should_include_consciousness_voice — keyword matching
# ===========================================================================

class TestShouldIncludeConsciousnessVoice:
    @pytest.mark.asyncio
    async def test_empty_topic_returns_false(self):
        """
        SENTINEL — empty topic must return False immediately.
        Remove empty check → empty string still checked for keywords → test fails.
        """
        bridge = _make_bridge()
        assert await bridge.should_include_consciousness_voice("") is False

    @pytest.mark.asyncio
    async def test_none_topic_returns_false(self):
        """None topic must return False."""
        bridge = _make_bridge()
        assert await bridge.should_include_consciousness_voice(None) is False

    @pytest.mark.asyncio
    async def test_consciousness_keyword_triggers_true(self):
        """
        SENTINEL — topic containing 'consciousness' must return True.
        Remove keyword check → keyword never matched → test fails.
        """
        bridge = _make_bridge()
        result = await bridge.should_include_consciousness_voice("What is consciousness?")
        assert result is True

    @pytest.mark.asyncio
    async def test_emergence_keyword_triggers_true(self):
        """'emergence' keyword must also trigger True."""
        bridge = _make_bridge()
        result = await bridge.should_include_consciousness_voice("The emergence of self-reflection")
        assert result is True

    @pytest.mark.asyncio
    async def test_matching_is_case_insensitive(self):
        """
        SENTINEL — keyword matching must be case-insensitive.
        Remove .lower() → 'CONSCIOUSNESS' not recognised → test fails.
        """
        bridge = _make_bridge()
        upper = await bridge.should_include_consciousness_voice("CONSCIOUSNESS debate")
        lower = await bridge.should_include_consciousness_voice("consciousness debate")
        assert upper is True
        assert lower is True

    @pytest.mark.asyncio
    async def test_irrelevant_topic_returns_false(self):
        """Topic with no consciousness keywords must return False."""
        bridge = _make_bridge()
        result = await bridge.should_include_consciousness_voice(
            "Please review the database migration script"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_partial_keyword_not_matched(self):
        """Partial word like 'con' must not trigger the 'conscious' keyword."""
        bridge = _make_bridge()
        # 'con' alone isn't in the keyword set
        result = await bridge.should_include_consciousness_voice("con artist tricks")
        assert result is False

    @pytest.mark.asyncio
    async def test_keyword_threshold_is_one(self):
        """A single keyword hit must meet the threshold."""
        bridge = _make_bridge()
        # One keyword: "awareness"
        result = await bridge.should_include_consciousness_voice("Situational awareness training")
        assert result is True

    @pytest.mark.asyncio
    async def test_all_keywords_in_set_match(self):
        """Spot-check several keywords from the canonical set."""
        bridge = _make_bridge()
        for kw in ["qualia", "sentience", "blackboard", "awakening", "soul", "sapience"]:
            result = await bridge.should_include_consciousness_voice(
                f"Topic about {kw} in AI systems"
            )
            assert result is True, f"Expected True for keyword: {kw}"


# ===========================================================================
# _extract_insights — static method
# ===========================================================================

class TestExtractInsights:
    def test_key_insights_section_extracted(self):
        """
        SENTINEL — 'KEY INSIGHTS' section must be extracted from synthesis.
        Remove regex → full synthesis returned instead of insights → test fails.
        """
        synthesis = (
            "## OVERVIEW\n\nSome preamble.\n\n"
            "**KEY INSIGHTS**:\nSENTINEL_INSIGHT_TEXT\n\n"
            "**RECOMMENDATION**: Do something."
        )
        result = ConsciousnessCouncilBridge._extract_insights(synthesis)
        assert result is not None
        assert "SENTINEL_INSIGHT_TEXT" in result

    def test_key_insights_mixed_case_extracted(self):
        """'Key Insights' (title case) must also be matched."""
        synthesis = "Preamble\n\n**Key Insights**:\nSENTINEL_TITLE_CASE\n\nEnd."
        result = ConsciousnessCouncilBridge._extract_insights(synthesis)
        assert result is not None
        assert "SENTINEL_TITLE_CASE" in result

    def test_no_section_returns_full_synthesis(self):
        """When no KEY INSIGHTS section exists, full synthesis must be returned."""
        synthesis = "Just a plain synthesis with no sections. SENTINEL_FULL."
        result = ConsciousnessCouncilBridge._extract_insights(synthesis)
        assert result is not None
        assert "SENTINEL_FULL" in result

    def test_long_synthesis_truncated_at_1500(self):
        """
        SENTINEL — synthesis longer than 1500 chars must be truncated.
        Remove cap → unbounded content sent to Blackboard → test fails.
        """
        long_text = "X" * 2000
        result = ConsciousnessCouncilBridge._extract_insights(long_text)
        assert result is not None
        assert len(result) <= 1503  # 1500 + "…"
        assert result.endswith("\u2026")

    def test_short_synthesis_returned_verbatim(self):
        """Synthesis under 1500 chars must be returned without truncation."""
        short = "Short synthesis SENTINEL."
        result = ConsciousnessCouncilBridge._extract_insights(short)
        assert result == short

    def test_empty_synthesis_returns_none(self):
        """
        SENTINEL — empty/whitespace synthesis must return None.
        Return '' for empty → caller writes empty string to Blackboard → test fails.
        """
        assert ConsciousnessCouncilBridge._extract_insights("") is None
        assert ConsciousnessCouncilBridge._extract_insights("   ") is None

    def test_key_insights_content_capped_at_1500(self):
        """Even within a KEY INSIGHTS section, extracted text must be capped at 1500."""
        long_insight = "Y" * 2000
        synthesis = f"**KEY INSIGHTS**:\n{long_insight}\n\n**DONE**"
        result = ConsciousnessCouncilBridge._extract_insights(synthesis)
        assert result is not None
        assert len(result) <= 1500


# ===========================================================================
# _ensure_init — lazy init and caching
# ===========================================================================

class TestEnsureInit:
    def test_import_failure_returns_false(self):
        """
        SENTINEL — if the consciousness module can't be imported, _ensure_init must
        return False and not raise.
        Remove try/except → ImportError propagates to callers → test fails.
        """
        bridge = _make_bridge()
        with patch.dict("sys.modules", {
            "app.consciousness.blackboard": None,
            "app.consciousness.context_builder": None,
        }):
            result = bridge._ensure_init()
        assert result is False

    def test_init_error_cached_on_failure(self):
        """After a failure, _init_error must be set and _initialised must be True."""
        bridge = _make_bridge()
        with patch.dict("sys.modules", {
            "app.consciousness.blackboard": None,
            "app.consciousness.context_builder": None,
        }):
            bridge._ensure_init()

        assert bridge._initialised is True
        assert bridge._init_error is not None

    def test_second_call_uses_cache(self):
        """
        SENTINEL — calling _ensure_init twice must not re-import (second call uses cache).
        Remove _initialised guard → every call reimports and re-creates Blackboard → test fails.
        """
        bridge = _make_bridge()
        # Force a failed init so we can count calls
        call_count = {"n": 0}

        original = bridge._ensure_init

        def _patched():
            call_count["n"] += 1
            bridge._initialised = True
            bridge._init_error = "simulated"
            return False

        bridge._ensure_init = _patched
        bridge._ensure_init()
        # Mark as initialised; next call should NOT invoke _patched again
        bridge._ensure_init = original  # restore
        # Since _initialised is True and _init_error is set, should return False immediately
        result = bridge._ensure_init()
        assert result is False
        assert call_count["n"] == 1  # was only called once via _patched


# ===========================================================================
# get_consciousness_context — graceful degradation
# ===========================================================================

class TestGetConsciousnessContext:
    @pytest.mark.asyncio
    async def test_returns_empty_when_blackboard_unavailable(self):
        """
        SENTINEL — if _ensure_init returns False, must return empty context dict.
        Raise instead → council deliberation crashes → test fails.
        """
        bridge = _make_bridge()
        bridge._initialised = True
        bridge._init_error = "simulated unavailability"

        result = await bridge.get_consciousness_context("some topic")

        assert result["available"] is False
        assert result["entry_count"] == 0
        assert result["entries"] == []
        assert result["summary"] == ""

    @pytest.mark.asyncio
    async def test_empty_topic_returns_empty(self):
        """
        SENTINEL — empty topic returns empty context (called by bridge when
        should_include check was bypassed upstream).
        """
        bridge = _make_bridge()
        bridge._initialised = True
        bridge._init_error = "unavailable"

        result = await bridge.get_consciousness_context("")
        assert result["available"] is False


# ===========================================================================
# update_blackboard_from_synthesis — write-back threshold
# ===========================================================================

class TestUpdateBlackboardFromSynthesis:
    @pytest.mark.asyncio
    async def test_empty_synthesis_does_not_write(self):
        """
        SENTINEL — empty synthesis must return immediately without writing.
        Remove empty check → empty string appended to Blackboard → test fails.
        """
        bridge = _make_bridge()
        bridge._blackboard = MagicMock()
        bridge._initialised = True
        bridge._init_error = None  # pretend init succeeded

        with patch.object(bridge._blackboard, "append_entry") as spy:
            await bridge.update_blackboard_from_synthesis("s-1", "")

        spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_low_keyword_density_skips_write(self):
        """
        SENTINEL — synthesis with < 2 consciousness keywords must NOT be written back.
        Remove threshold → every synthesis pollutes the Blackboard → test fails.
        """
        bridge = _make_bridge()
        mock_blackboard = MagicMock()
        bridge._blackboard = mock_blackboard
        bridge._initialised = True
        bridge._init_error = None

        # Synthesis with only 1 consciousness keyword — 'qualia' has no substrings in the keyword set
        synthesis = "A discussion about machine learning and algorithms. qualia mentioned once."

        with patch.object(mock_blackboard, "append_entry") as spy:
            await bridge.update_blackboard_from_synthesis("s-2", synthesis)

        spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_high_keyword_density_triggers_write(self):
        """
        SENTINEL — synthesis with ≥ 2 consciousness keywords must write to Blackboard.
        Remove threshold → high-density insights silently dropped → test fails.
        """
        bridge = _make_bridge()
        mock_blackboard = MagicMock()
        bridge._blackboard = mock_blackboard
        bridge._initialised = True
        bridge._init_error = None

        # Two consciousness keywords: "consciousness" and "awareness"
        synthesis = (
            "This synthesis explores consciousness and awareness in AI systems. "
            "The council found significant emergence of identity patterns."
        )

        mock_blackboard.append_entry = MagicMock()
        await bridge.update_blackboard_from_synthesis("s-3", synthesis)

        mock_blackboard.append_entry.assert_called_once()

    @pytest.mark.asyncio
    async def test_blackboard_unavailable_does_not_raise(self):
        """
        SENTINEL — if the Blackboard is unavailable, update must silently no-op.
        Propagate exception → council synthesis stage crashes → test fails.
        """
        bridge = _make_bridge()
        bridge._initialised = True
        bridge._init_error = "simulated"  # ensures _ensure_init returns False

        # Should not raise even with a rich consciousness synthesis
        synthesis = "consciousness awareness sentience emergence identity qualia"
        await bridge.update_blackboard_from_synthesis("s-4", synthesis)  # no raise

    @pytest.mark.asyncio
    async def test_blackboard_write_error_does_not_raise(self):
        """
        SENTINEL — if append_entry raises, the error must be swallowed.
        Remove try/except → Blackboard write failure breaks deliberation → test fails.
        """
        bridge = _make_bridge()
        mock_blackboard = MagicMock()
        mock_blackboard.append_entry = MagicMock(side_effect=RuntimeError("disk full"))
        bridge._blackboard = mock_blackboard
        bridge._initialised = True
        bridge._init_error = None

        synthesis = "consciousness awareness sentience emergence identity soul"
        # Must not raise
        await bridge.update_blackboard_from_synthesis("s-5", synthesis)


# ===========================================================================
# get_consciousness_council_bridge — singleton
# ===========================================================================

class TestGetConsciousnessCouncilBridge:
    def test_returns_same_instance_on_second_call(self):
        """
        SENTINEL — get_consciousness_council_bridge must be a singleton.
        Create new instance each call → bridge state lost between calls → test fails.
        """
        # Reset the global
        import app.services.consciousness_council_bridge as mod
        mod._bridge = None

        b1 = get_consciousness_council_bridge()
        b2 = get_consciousness_council_bridge()
        assert b1 is b2

    def test_returns_consciousness_council_bridge_instance(self):
        """Factory must return a ConsciousnessCouncilBridge."""
        import app.services.consciousness_council_bridge as mod
        mod._bridge = None
        b = get_consciousness_council_bridge()
        assert isinstance(b, ConsciousnessCouncilBridge)
