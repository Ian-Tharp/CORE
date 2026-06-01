"""
Tests for title_generation_service — pure string helpers and LLM call.

Sentinel-value contract:
- Remove quote stripping in _clean_title → titles with quotes passed through raw → test fails
- Remove prefix stripping in _clean_title → "Title: Foo" never cleaned → test fails
- Remove truncation in _clean_title → very long titles returned uncut → test fails
- Remove fallback in generate_conversation_title → HTTP errors raise instead of returning title → test fails
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.title_generation_service import (
    _clean_title,
    _fallback_title,
    generate_conversation_title,
)


# ===========================================================================
# _clean_title — pure string transformation
# ===========================================================================


class TestCleanTitle:
    def test_double_quotes_stripped(self):
        """
        SENTINEL — leading and trailing double quotes must be removed.
        Remove strip('"') → titles arrive as '"Foo Bar"' not 'Foo Bar' → test fails.
        """
        assert _clean_title('"Foo Bar"') == "Foo Bar"

    def test_single_quotes_stripped(self):
        """Single-quote wrapping must also be removed."""
        assert _clean_title("'Foo Bar'") == "Foo Bar"

    def test_title_prefix_removed(self):
        """
        SENTINEL — 'Title: ' prefix must be stripped.
        Remove prefix logic → LLM verbosity leaks into displayed title → test fails.
        """
        result = _clean_title("Title: Quantum Computing Basics")
        assert result == "Quantum Computing Basics"

    def test_title_lowercase_prefix_removed(self):
        """'title: ' (lowercase) must also be stripped."""
        result = _clean_title("title: Python Async Help")
        assert result == "Python Async Help"

    def test_heres_a_title_prefix_removed(self):
        """'Here's a title: ' prefix must be stripped."""
        result = _clean_title("Here's a title: React State Debugging")
        assert result == "React State Debugging"

    def test_truncation_at_60_chars(self):
        """
        SENTINEL — titles longer than 60 chars must be truncated to 57 + '...'.
        Remove truncation → 200-char titles stored unclipped → test fails.
        """
        long_title = "A" * 80
        result = _clean_title(long_title)
        assert len(result) <= 60
        assert result.endswith("...")

    def test_short_title_not_truncated(self):
        """Titles under 60 chars must pass through unchanged (modulo other transforms)."""
        short = "Short Title Here"
        assert _clean_title(short) == "Short Title Here"

    def test_first_letter_capitalized(self):
        """
        SENTINEL — first character must be uppercased.
        Remove capitalize → 'python async help' displayed lowercase → test fails.
        """
        result = _clean_title("python async help")
        assert result[0] == "P"

    def test_empty_string_stays_empty(self):
        """Empty string must be returned as-is (no index error)."""
        assert _clean_title("") == ""

    def test_strip_strips_quotes_not_whitespace(self):
        """strip() removes quote chars; internal whitespace is preserved as-is."""
        # The function strips '"' and "'" chars, not spaces
        result = _clean_title('"Foo Bar"')
        assert result == "Foo Bar"
        # Whitespace inside is preserved (no generic .strip())
        result2 = _clean_title("A B C")
        assert result2 == "A B C"

    def test_exactly_60_chars_not_truncated(self):
        """A 60-character title must not be truncated."""
        exact = "A" * 60
        result = _clean_title(exact)
        assert not result.endswith("...")
        assert len(result) == 60


# ===========================================================================
# _fallback_title — pure fallback logic
# ===========================================================================


class TestFallbackTitle:
    def test_empty_message_returns_new_conversation(self):
        """
        SENTINEL — empty user_message must return 'New Conversation'.
        Return '' for empty input → empty title shown to user → test fails.
        """
        assert _fallback_title("") == "New Conversation"

    def test_none_returns_new_conversation(self):
        """None user_message must return 'New Conversation'."""
        assert _fallback_title(None) == "New Conversation"

    def test_short_message_returned_verbatim(self):
        """Message under 50 chars must be returned as-is."""
        msg = "How do I debug React state?"
        assert _fallback_title(msg) == msg

    def test_long_message_truncated_to_50(self):
        """
        SENTINEL — message longer than 50 chars must be truncated to 50 + '...'.
        Remove truncation → 500-char conversation opener used as title → test fails.
        """
        long_msg = "A" * 100
        result = _fallback_title(long_msg)
        assert result.endswith("...")
        assert len(result) <= 53  # 50 + 3 for "..."

    def test_exactly_50_chars_not_truncated(self):
        """50-character message must NOT get '...' appended."""
        msg = "B" * 50
        result = _fallback_title(msg)
        assert not result.endswith("...")


# ===========================================================================
# generate_conversation_title — LLM integration with httpx mock
# ===========================================================================


class TestGenerateConversationTitle:
    @pytest.mark.asyncio
    async def test_returns_cleaned_title_from_llm(self):
        """
        SENTINEL — title from LLM must be cleaned and returned.
        Remove _clean_title call → raw LLM verbosity shown to user → test fails.
        """
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"response": '"SENTINEL_TITLE"'}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch(
            "app.services.title_generation_service.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await generate_conversation_title("user msg", "assistant msg")

        # Quotes stripped by _clean_title
        assert result == "SENTINEL_TITLE"

    @pytest.mark.asyncio
    async def test_http_error_returns_fallback(self):
        """
        SENTINEL — on HTTP failure, fallback title must be returned (no raise).
        Remove try/except → any Ollama outage crashes the calling endpoint → test fails.
        """
        import httpx

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("down"))

        with patch(
            "app.services.title_generation_service.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await generate_conversation_title(
                "SENTINEL_USER_MESSAGE_HERE", "assistant msg"
            )

        # Fallback must not raise; must return something based on user_message
        assert isinstance(result, str)
        assert len(result) > 0
        assert (
            "SENTINEL_USER_MESSAGE" in result or result == "SENTINEL_USER_MESSAGE_HERE"
        )

    @pytest.mark.asyncio
    async def test_empty_llm_response_uses_fallback(self):
        """
        SENTINEL — if LLM returns empty string, fallback title must kick in.
        Remove 'if not title: title = _fallback_title(...)' → empty title returned → test fails.
        """
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"response": "   "}  # whitespace only

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch(
            "app.services.title_generation_service.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await generate_conversation_title(
                "SENTINEL_FALLBACK_MSG", "response"
            )

        assert result  # must not be empty
        assert "SENTINEL_FALLBACK" in result or len(result) > 0

    @pytest.mark.asyncio
    async def test_user_message_truncated_in_prompt(self):
        """
        SENTINEL — user_message[:200] limits prompt size sent to LLM.
        The prompt must contain at most 200 chars of user_message — verified by
        checking the POST body sent to Ollama contains only the truncated prefix.
        """
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"response": "Short Title"}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        long_msg = "X" * 500  # 500-char message
        with patch(
            "app.services.title_generation_service.httpx.AsyncClient",
            return_value=mock_client,
        ):
            await generate_conversation_title(long_msg, "response")

        call_kwargs = mock_client.post.call_args
        prompt_body = (
            call_kwargs[1]["json"]["prompt"]
            if call_kwargs[1]
            else call_kwargs[0][1]["prompt"]
        )
        # The prompt must not contain the full 500 Xs
        assert "X" * 201 not in prompt_body
