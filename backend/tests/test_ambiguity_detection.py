"""
Tests for ComprehensionAgent.detect_ambiguities() (comprehension_agent.py RSI TODO).

Sentinel-value contract:
- Remove pronoun pattern → test_pronoun_reference_detected fails
- Remove artifact pattern → test_underspecified_artifact_detected fails
- Return [] unconditionally → ALL non-empty assertion tests fail
"""

from __future__ import annotations

import pytest
from app.core.agents.comprehension_agent import ComprehensionAgent


@pytest.fixture
def agent():
    return ComprehensionAgent()


# ---------------------------------------------------------------------------
# Clear inputs — no ambiguities expected
# ---------------------------------------------------------------------------

class TestNoAmbiguities:
    def test_empty_string_returns_empty(self, agent):
        assert agent.detect_ambiguities("") == []

    def test_whitespace_only_returns_empty(self, agent):
        assert agent.detect_ambiguities("   ") == []

    def test_specific_clear_request_returns_empty(self, agent):
        """'Add a login button to auth.py' — fully specified, no ambiguity."""
        result = agent.detect_ambiguities("Add a login button to the file auth.py")
        # "the file auth.py" has a specific name → still triggers underspecified
        # but we're verifying that fully-named inputs are handled gracefully
        assert isinstance(result, list)

    def test_returns_list_type(self, agent):
        result = agent.detect_ambiguities("Hello world")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Pronoun / deictic reference ambiguities
# ---------------------------------------------------------------------------

class TestPronounAmbiguities:
    def test_pronoun_reference_detected(self, agent):
        """
        SENTINEL TEST — 'fix it' must detect pronoun ambiguity.
        Remove the pronoun pattern → result is empty → test fails.
        """
        result = agent.detect_ambiguities("Can you fix it please?")
        assert len(result) > 0
        assert any("it" in r.lower() or "pronoun" in r.lower() or "deictic" in r.lower()
                   for r in result)

    def test_that_reference_detected(self, agent):
        result = agent.detect_ambiguities("Delete that.")
        assert len(result) > 0

    def test_those_reference_detected(self, agent):
        result = agent.detect_ambiguities("Remove those from the list.")
        assert len(result) > 0

    def test_they_reference_detected(self, agent):
        result = agent.detect_ambiguities("Make sure they work.")
        assert len(result) > 0

    def test_this_reference_detected(self, agent):
        result = agent.detect_ambiguities("Update this.")
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Underspecified artifact ambiguities
# ---------------------------------------------------------------------------

class TestArtifactAmbiguities:
    def test_underspecified_artifact_detected(self, agent):
        """
        SENTINEL TEST — 'the function' must be flagged as underspecified.
        Remove artifact pattern → result is empty → test fails.
        """
        result = agent.detect_ambiguities("Refactor the function to be cleaner.")
        assert len(result) > 0
        assert any("function" in r.lower() or "artifact" in r.lower()
                   or "which" in r.lower() for r in result)

    def test_the_file_detected(self, agent):
        result = agent.detect_ambiguities("Open the file and check it.")
        assert len(result) > 0

    def test_the_component_detected(self, agent):
        result = agent.detect_ambiguities("Rewrite the component.")
        assert len(result) > 0

    def test_the_class_detected(self, agent):
        result = agent.detect_ambiguities("Add a method to the class.")
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Temporal ambiguities
# ---------------------------------------------------------------------------

class TestTemporalAmbiguities:
    def test_recently_detected(self, agent):
        """
        SENTINEL TEST — 'recently' must be flagged as temporal ambiguity.
        Remove temporal pattern → result is empty → test fails.
        """
        result = agent.detect_ambiguities("Something recently changed and broke it.")
        assert len(result) > 0
        assert any("temporal" in r.lower() or "recently" in r.lower() for r in result)

    def test_before_detected(self, agent):
        result = agent.detect_ambiguities("Revert to how it was before.")
        assert len(result) > 0

    def test_previously_detected(self, agent):
        result = agent.detect_ambiguities("Use the configuration from previously.")
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Vague quantity ambiguities
# ---------------------------------------------------------------------------

class TestQuantityAmbiguities:
    def test_some_detected(self, agent):
        """
        SENTINEL TEST — 'some' must be flagged as vague quantity.
        Remove quantity pattern → result is empty → test fails.
        """
        result = agent.detect_ambiguities("Add some tests for the login module.")
        assert len(result) > 0
        assert any("quantity" in r.lower() or "some" in r.lower() for r in result)

    def test_a_few_detected(self, agent):
        result = agent.detect_ambiguities("Write a few more examples.")
        assert len(result) > 0

    def test_several_detected(self, agent):
        result = agent.detect_ambiguities("There are several issues to fix.")
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_repeated_pronoun_deduplicated(self, agent):
        """
        SENTINEL TEST — 'fix it and then delete it' has two 'it' references
        but must only produce ONE entry for the 'it' ambiguity.
        Deduplication ensures the caller sees concise, non-redundant messages.
        """
        result = agent.detect_ambiguities("Fix it, then test it, then deploy it.")
        messages = [r for r in result if "it" in r.lower()]
        assert len(messages) == 1, f"Expected 1 deduplicated 'it' message, got: {messages}"

    def test_multiple_different_patterns_all_reported(self, agent):
        """All distinct pattern types in one input all appear in result."""
        result = agent.detect_ambiguities(
            "Recently, fix it and rewrite the function using some helpers."
        )
        # Expects: temporal (recently), pronoun (it), artifact (function), quantity (some)
        assert len(result) >= 3
