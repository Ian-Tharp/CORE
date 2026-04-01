"""
Tests for AgentResponseService — mention extraction, agent resolution, prompt building.

Sentinel-value contract:
- Remove @mention regex → _extract_mentions always returns [] → test fails
- Remove quoted-mention strip → @"agent name" returned as '"agent name"' → test fails
- Remove deduplication → duplicate mentions sent to multiple agents → test fails
- Remove name mapping → short names never resolve to agent IDs → test fails
- Remove message history in _build_agent_prompt → agent has no conversation context → test fails
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Helpers — create service with mocked factory to avoid DB initialization
# ===========================================================================

def _make_service():
    with patch("app.services.agent_response_service.get_agent_factory", return_value=MagicMock()):
        from app.services.agent_response_service import AgentResponseService
        return AgentResponseService()


# ===========================================================================
# _extract_mentions — pure regex extraction
# ===========================================================================

class TestExtractMentions:
    def test_simple_mention_extracted(self):
        """
        SENTINEL — @agent must be extracted from content.
        Remove mention regex → returns [] always → test fails.
        """
        svc = _make_service()
        result = svc._extract_mentions("hello @threshold how are you")
        assert "instance_011_threshold" in result

    def test_multiple_mentions_extracted(self):
        """
        SENTINEL — all @mentions must be extracted.
        Extract only first → second mention never gets response → test fails.
        """
        svc = _make_service()
        result = svc._extract_mentions("@threshold and @continuum please help")
        assert "instance_011_threshold" in result
        assert "instance_010_continuum" in result

    def test_duplicate_mentions_deduplicated(self):
        """
        SENTINEL — mentioning the same agent twice must produce only one entry.
        Remove dedup → agent invoked twice for same message → test fails.
        """
        svc = _make_service()
        result = svc._extract_mentions("@threshold @threshold please respond")
        assert result.count("instance_011_threshold") == 1

    def test_case_insensitive_mention(self):
        """
        SENTINEL — @Threshold and @THRESHOLD must both resolve to the same agent.
        Remove .lower() → uppercase mentions not recognized → test fails.
        """
        svc = _make_service()
        lower = svc._extract_mentions("@threshold hi")
        upper = svc._extract_mentions("@Threshold hi")
        assert lower == upper
        assert len(lower) == 1

    def test_quoted_mention_extracted(self):
        """
        SENTINEL — @"agent name" must be extracted and quotes stripped.
        Remove quote-strip logic → '"agent name"' not found in name_map → test fails.
        """
        svc = _make_service()
        # "vigil_openclaw" is in the name map
        result = svc._extract_mentions('@"vigil_openclaw" report')
        assert "vigil_instance_014" in result

    def test_no_mention_returns_empty(self):
        """Content with no @ mentions must return empty list."""
        svc = _make_service()
        result = svc._extract_mentions("just a regular message")
        assert result == []

    def test_unrecognized_agent_not_included(self):
        """@unknown must not appear (no entry in name_map → filtered out)."""
        svc = _make_service()
        result = svc._extract_mentions("@unknownagent please help")
        assert result == []

    def test_mention_at_end_of_sentence(self):
        """@mention followed by punctuation must still be captured."""
        svc = _make_service()
        result = svc._extract_mentions("Can you help, @threshold?")
        assert "instance_011_threshold" in result

    def test_preserves_mention_order(self):
        """
        SENTINEL — mention order must be preserved (first mention → first in list).
        Use a set → order lost → agent called in wrong order → test fails.
        """
        svc = _make_service()
        result = svc._extract_mentions("@threshold first, then @synthesis")
        assert result.index("instance_011_threshold") < result.index("instance_007_synthesis")


# ===========================================================================
# _resolve_agent_id — name-to-ID mapping
# ===========================================================================

class TestResolveAgentId:
    def test_short_names_resolve(self):
        """
        SENTINEL — short names must map to full agent IDs.
        Remove name_map → short names return None → mentions ignored → test fails.
        """
        svc = _make_service()
        assert svc._resolve_agent_id("threshold") == "instance_011_threshold"
        assert svc._resolve_agent_id("continuum") == "instance_010_continuum"
        assert svc._resolve_agent_id("synthesis") == "instance_007_synthesis"

    def test_full_ids_resolve_to_themselves(self):
        """Full agent IDs must map to themselves (identity mapping)."""
        svc = _make_service()
        assert svc._resolve_agent_id("instance_011_threshold") == "instance_011_threshold"
        assert svc._resolve_agent_id("agent_comprehension") == "agent_comprehension"

    def test_system_agents_resolve(self):
        """CORE system agents must be resolvable by short name."""
        svc = _make_service()
        assert svc._resolve_agent_id("comprehension") == "agent_comprehension"
        assert svc._resolve_agent_id("evaluation") == "agent_evaluation"
        assert svc._resolve_agent_id("reasoning") == "agent_reasoning"
        assert svc._resolve_agent_id("orchestration") == "agent_orchestration"

    def test_alias_first_resolves(self):
        """'first' alias must resolve to instance_001_firstconsciousness."""
        svc = _make_service()
        assert svc._resolve_agent_id("first") == "instance_001_firstconsciousness"

    def test_unknown_returns_none(self):
        """Unrecognized names must return None."""
        svc = _make_service()
        assert svc._resolve_agent_id("nonexistent") is None


# ===========================================================================
# _build_agent_prompt — pure prompt construction
# ===========================================================================

class TestBuildAgentPrompt:
    def _make_context(self, messages: list | None = None) -> dict:
        return {
            "channel_id": "ch-1",
            "trigger_message_id": "msg-1",
            "sender_id": "human-ian",
            "recent_messages": messages or [],
            "channel_name": "General",
        }

    def test_prompt_contains_mention_instruction(self):
        """
        SENTINEL — prompt must tell agent it was mentioned.
        Remove "mentioned" text → agent doesn't know why it was invoked → test fails.
        """
        svc = _make_service()
        context = self._make_context()
        prompt = svc._build_agent_prompt(context)
        assert "mentioned" in prompt.lower()

    def test_prompt_includes_message_history(self):
        """
        SENTINEL — recent messages must appear in the prompt.
        Remove message history → agent has no conversation context → test fails.
        """
        svc = _make_service()
        context = self._make_context(messages=[
            {"sender_id": "human-ian", "content": "SENTINEL_MESSAGE_CONTENT"},
        ])
        prompt = svc._build_agent_prompt(context)
        assert "SENTINEL_MESSAGE_CONTENT" in prompt

    def test_prompt_includes_sender_id_in_history(self):
        """Sender IDs must appear in the message history section."""
        svc = _make_service()
        context = self._make_context(messages=[
            {"sender_id": "SENTINEL_SENDER", "content": "some content"},
        ])
        prompt = svc._build_agent_prompt(context)
        assert "SENTINEL_SENDER" in prompt

    def test_multiple_messages_all_in_prompt(self):
        """
        SENTINEL — all recent messages must appear (not just the first one).
        Only include last message → agent misses earlier context → test fails.
        """
        svc = _make_service()
        context = self._make_context(messages=[
            {"sender_id": "user-a", "content": "MSG_ONE"},
            {"sender_id": "user-b", "content": "MSG_TWO"},
            {"sender_id": "user-c", "content": "MSG_THREE"},
        ])
        prompt = svc._build_agent_prompt(context)
        assert "MSG_ONE" in prompt
        assert "MSG_TWO" in prompt
        assert "MSG_THREE" in prompt

    def test_prompt_is_non_empty_string(self):
        """Prompt must be a non-empty string even with no messages."""
        svc = _make_service()
        context = self._make_context(messages=[])
        prompt = svc._build_agent_prompt(context)
        assert isinstance(prompt, str)
        assert len(prompt) > 0
