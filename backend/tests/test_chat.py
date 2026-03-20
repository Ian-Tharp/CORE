"""Tests for the chat controller — input validation, provider/model checks, and streaming."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from app.controllers.chat import (
    ChatRequest,
    Message,
    _validate_provider_model,
    _validate_total_size,
    chat_stream,
    MAX_MESSAGES,
    MAX_MESSAGE_CHARS,
    MAX_TOTAL_CHARS,
)


# ────────────────────────────────────────────────────────────────────
# Message model validation
# ────────────────────────────────────────────────────────────────────


class TestMessageValidation:
    """Tests for the Message pydantic model."""

    def test_valid_user_message(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_valid_system_message(self):
        msg = Message(role="system", content="You are helpful.")
        assert msg.role == "system"

    def test_valid_assistant_message(self):
        msg = Message(role="assistant", content="Hi there!")
        assert msg.role == "assistant"

    def test_invalid_role_rejected(self):
        with pytest.raises(Exception):
            Message(role="hacker", content="inject")

    def test_content_at_limit_accepted(self):
        msg = Message(role="user", content="a" * MAX_MESSAGE_CHARS)
        assert len(msg.content) == MAX_MESSAGE_CHARS

    def test_content_over_limit_rejected(self):
        with pytest.raises(Exception):
            Message(role="user", content="a" * (MAX_MESSAGE_CHARS + 1))

    def test_empty_content_accepted(self):
        msg = Message(role="user", content="")
        assert msg.content == ""


# ────────────────────────────────────────────────────────────────────
# ChatRequest model validation
# ────────────────────────────────────────────────────────────────────


class TestChatRequestValidation:
    """Tests for the ChatRequest pydantic model."""

    def test_minimal_valid_request(self):
        req = ChatRequest(
            model="gpt-4o",
            messages=[Message(role="user", content="Hi")],
        )
        assert req.model == "gpt-4o"
        assert req.stream is True
        assert req.provider == "openai"

    def test_empty_messages_rejected(self):
        with pytest.raises(Exception):
            ChatRequest(model="gpt-4o", messages=[])

    def test_max_messages_accepted(self):
        msgs = [Message(role="user", content="x") for _ in range(MAX_MESSAGES)]
        req = ChatRequest(model="gpt-4o", messages=msgs)
        assert len(req.messages) == MAX_MESSAGES

    def test_over_max_messages_rejected(self):
        msgs = [Message(role="user", content="x") for _ in range(MAX_MESSAGES + 1)]
        with pytest.raises(Exception):
            ChatRequest(model="gpt-4o", messages=msgs)

    def test_optional_fields_defaults(self):
        req = ChatRequest(
            model="test",
            messages=[Message(role="user", content="Hi")],
        )
        assert req.conversation_id is None
        assert req.kb_mode is None
        assert req.kb_file_id is None

    def test_all_optional_fields_set(self):
        req = ChatRequest(
            model="gpt-4o",
            messages=[Message(role="user", content="Hi")],
            conversation_id="conv-123",
            stream=False,
            provider="anthropic",
            kb_mode="all",
            kb_file_id="file-456",
            kb_embedding_provider="openai",
            kb_local_model="nomic-embed-text",
        )
        assert req.conversation_id == "conv-123"
        assert req.stream is False
        assert req.provider == "anthropic"


# ────────────────────────────────────────────────────────────────────
# Provider / model validation
# ────────────────────────────────────────────────────────────────────


class TestProviderModelValidation:
    """Tests for _validate_provider_model helper."""

    def test_valid_openai_provider_registered_model(self):
        # gpt-4o is in MODELS — should not raise
        _validate_provider_model("openai", "gpt-4o")

    def test_valid_anthropic_provider_registered_model(self):
        _validate_provider_model("anthropic", "claude-3-5-sonnet")

    def test_ollama_accepts_any_model(self):
        _validate_provider_model("ollama", "my-custom-gguf:latest")

    def test_local_accepts_any_model(self):
        _validate_provider_model("local", "whatever-i-want")

    def test_unknown_provider_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_provider_model("azure", "gpt-4o")
        assert exc_info.value.status_code == 400
        assert "Unsupported provider" in exc_info.value.detail

    def test_unregistered_cloud_model_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_provider_model("openai", "gpt-99-turbo")
        assert exc_info.value.status_code == 400
        assert "Unsupported model" in exc_info.value.detail

    def test_none_provider_skips_provider_check(self):
        # provider=None should only validate model against registry
        _validate_provider_model(None, "gpt-4o")

    def test_model_name_matches_config_model_name(self):
        # Match on ModelConfig.model_name (e.g. "claude-3-5-sonnet-20241022")
        _validate_provider_model("anthropic", "claude-3-5-sonnet-20241022")

    def test_model_key_match(self):
        # Match on the MODELS dict key directly
        _validate_provider_model("openai", "gpt-4o-mini")


# ────────────────────────────────────────────────────────────────────
# Total size validation
# ────────────────────────────────────────────────────────────────────


class TestTotalSizeValidation:
    """Tests for _validate_total_size helper."""

    def test_small_messages_accepted(self):
        msgs = [Message(role="user", content="Hello, world!")]
        _validate_total_size(msgs)  # should not raise

    def test_at_limit_accepted(self):
        # Split across multiple messages to stay within per-message limit
        chunk_size = MAX_MESSAGE_CHARS - 1
        num_chunks = MAX_TOTAL_CHARS // chunk_size
        remainder = MAX_TOTAL_CHARS % chunk_size
        msgs = [Message(role="user", content="a" * chunk_size) for _ in range(num_chunks)]
        if remainder:
            msgs.append(Message(role="user", content="a" * remainder))
        _validate_total_size(msgs)

    def test_over_limit_rejected(self):
        # Many messages within per-message limit but exceeding total limit
        chunk_size = MAX_MESSAGE_CHARS - 1
        num_chunks = (MAX_TOTAL_CHARS // chunk_size) + 2  # guaranteed over total
        msgs = [Message(role="user", content="a" * chunk_size) for _ in range(num_chunks)]
        with pytest.raises(HTTPException) as exc_info:
            _validate_total_size(msgs)
        assert exc_info.value.status_code == 413

    def test_many_small_messages_within_limit(self):
        msgs = [Message(role="user", content="hi") for _ in range(50)]
        _validate_total_size(msgs)


# ────────────────────────────────────────────────────────────────────
# chat_stream endpoint (integration-level with mocks)
# ────────────────────────────────────────────────────────────────────


class TestChatStreamEndpoint:
    """Tests for the chat_stream endpoint logic."""

    def _make_raw_request(self, correlation_id=None):
        """Build a mock FastAPI Request with optional correlation_id on state."""
        raw = MagicMock()
        raw.state = MagicMock()
        if correlation_id:
            raw.state.correlation_id = correlation_id
        else:
            raw.state.correlation_id = None
        return raw

    @pytest.mark.asyncio
    async def test_bad_provider_returns_400(self):
        req = ChatRequest(
            model="gpt-4o",
            messages=[Message(role="user", content="Hi")],
            provider="azure",
        )
        with pytest.raises(HTTPException) as exc_info:
            await chat_stream(req, self._make_raw_request())
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_bad_model_returns_400(self):
        req = ChatRequest(
            model="nonexistent-model",
            messages=[Message(role="user", content="Hi")],
            provider="openai",
        )
        with pytest.raises(HTTPException) as exc_info:
            await chat_stream(req, self._make_raw_request())
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_oversized_payload_returns_413(self):
        # Build a request just over the total size limit
        big_content = "x" * (MAX_TOTAL_CHARS + 1)
        # We need to bypass Message validation for content length,
        # so test via the total-size path with many messages
        msgs = [Message(role="user", content="a" * (MAX_MESSAGE_CHARS - 1)) for _ in range(12)]
        req = ChatRequest(
            model="gpt-4o",
            messages=msgs,
            provider="ollama",  # skip model validation
        )
        with pytest.raises(HTTPException) as exc_info:
            await chat_stream(req, self._make_raw_request())
        assert exc_info.value.status_code == 413

    @pytest.mark.asyncio
    @patch("app.controllers.chat.create_conversation", new_callable=AsyncMock)
    @patch("app.controllers.chat.chat_service")
    async def test_successful_stream_returns_response(self, mock_chat_svc, mock_create_conv):
        mock_create_conv.return_value = "conv-001"

        async def fake_stream(**kwargs):
            yield 'data: {"delta": "Hello"}\n\n'
            yield 'data: {"delta": " world"}\n\n'

        mock_chat_svc.return_value = fake_stream()

        req = ChatRequest(
            model="phi3:mini",
            messages=[Message(role="user", content="Hi")],
            provider="ollama",
        )
        with patch("app.controllers.chat.append_message", new_callable=AsyncMock):
            with patch("app.controllers.chat.get_conversation", new_callable=AsyncMock, return_value=None):
                response = await chat_stream(req, self._make_raw_request("test-corr-id"))

        assert response.status_code == 200
        assert response.headers["X-Correlation-Id"] == "test-corr-id"
        assert response.headers["X-Conversation-Id"] == "conv-001"

    @pytest.mark.asyncio
    @patch("app.controllers.chat.create_conversation", new_callable=AsyncMock)
    @patch("app.controllers.chat.chat_service")
    async def test_existing_conversation_appends_message(self, mock_chat_svc, mock_create_conv):
        async def fake_stream(**kwargs):
            yield 'data: {"delta": "ok"}\n\n'

        mock_chat_svc.return_value = fake_stream()

        req = ChatRequest(
            model="phi3:mini",
            messages=[Message(role="user", content="continue")],
            provider="ollama",
            conversation_id="existing-conv",
        )
        with patch("app.controllers.chat.append_message", new_callable=AsyncMock) as mock_append:
            with patch("app.controllers.chat.get_conversation", new_callable=AsyncMock, return_value=None):
                response = await chat_stream(req, self._make_raw_request())

        assert response.headers["X-Conversation-Id"] == "existing-conv"
        mock_create_conv.assert_not_called()
        mock_append.assert_called_once()

    @pytest.mark.asyncio
    async def test_correlation_id_fallback_when_middleware_missing(self):
        """When middleware doesn't set correlation_id, endpoint generates its own."""
        req = ChatRequest(
            model="gpt-4o",
            messages=[Message(role="user", content="Hi")],
            provider="ollama",
        )
        raw = MagicMock()
        raw.state = MagicMock(spec=[])  # no correlation_id attribute

        with patch("app.controllers.chat.create_conversation", new_callable=AsyncMock, return_value="c1"):
            with patch("app.controllers.chat.chat_service") as mock_svc:
                async def empty_stream(**kwargs):
                    return
                    yield  # make it a generator

                mock_svc.return_value = empty_stream()
                with patch("app.controllers.chat.append_message", new_callable=AsyncMock):
                    with patch("app.controllers.chat.get_conversation", new_callable=AsyncMock, return_value=None):
                        response = await chat_stream(req, raw)

        # Should have a valid UUID correlation id
        corr = response.headers.get("X-Correlation-Id")
        assert corr is not None
        assert len(corr) == 36  # UUID format


# ────────────────────────────────────────────────────────────────────
# Edge cases
# ────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_single_message_request(self):
        req = ChatRequest(
            model="gpt-4o",
            messages=[Message(role="user", content="solo")],
        )
        assert len(req.messages) == 1

    def test_unicode_content_accepted(self):
        msg = Message(role="user", content="こんにちは世界 🌍 مرحبا")
        assert "🌍" in msg.content

    def test_multiline_content_accepted(self):
        msg = Message(role="user", content="line1\nline2\nline3")
        assert "\n" in msg.content

    def test_system_user_assistant_sequence(self):
        req = ChatRequest(
            model="test",
            messages=[
                Message(role="system", content="You are helpful."),
                Message(role="user", content="Hi"),
                Message(role="assistant", content="Hello!"),
                Message(role="user", content="Follow-up"),
            ],
            provider="ollama",
        )
        assert len(req.messages) == 4

    def test_provider_none_allowed(self):
        """provider=None should be accepted by the model."""
        req = ChatRequest(
            model="gpt-4o",
            messages=[Message(role="user", content="Hi")],
            provider=None,
        )
        assert req.provider is None