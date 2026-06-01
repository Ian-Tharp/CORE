"""
Tests for chat_service — LLM dispatch, circuit breaking, retry, and SSE formatting.

Sentinel-value contract:
- Remove _normalise_provider → "local" goes to openai path → test fails
- Remove circuit breaker check → OPEN breaker doesn't block request → test fails
- Remove message truncation → oversized context sent to provider → test fails
- Remove breaker.record_success → breaker never recovers → test fails
- Remove breaker.record_failure → breaker never opens on repeated failures → test fails
- Remove retry loop → first failure is final → test fails
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.chat_service import (
    MAX_MESSAGES,
    _normalise_provider,
    chat_service,
)


# ===========================================================================
# _normalise_provider — pure function
# ===========================================================================


class TestNormaliseProvider:
    def test_ollama_stays_ollama(self):
        """SENTINEL — "ollama" must stay "ollama"."""
        assert _normalise_provider("ollama") == "ollama"

    def test_local_normalises_to_ollama(self):
        """
        SENTINEL — "local" and "local-ollama" must map to "ollama".
        Remove the "local" alias → local LLM calls go to OpenAI path → test fails.
        """
        assert _normalise_provider("local") == "ollama"
        assert _normalise_provider("local-ollama") == "ollama"

    def test_case_insensitive(self):
        """Provider normalisation must be case-insensitive."""
        assert _normalise_provider("Ollama") == "ollama"
        assert _normalise_provider("LOCAL") == "ollama"

    def test_openai_stays_openai(self):
        """Unknown providers default to "openai"."""
        assert _normalise_provider("openai") == "openai"
        assert _normalise_provider("gpt-4") == "openai"
        assert _normalise_provider("anthropic") == "openai"


# ===========================================================================
# chat_service — circuit breaker gate
# ===========================================================================


class TestCircuitBreakerGate:
    def _make_open_breaker(self):
        b = MagicMock()
        b.allow_request.return_value = False
        return b

    def _make_closed_breaker(self):
        b = MagicMock()
        b.allow_request.return_value = True
        b.record_success = MagicMock()
        b.record_failure = MagicMock()
        return b

    @pytest.mark.asyncio
    async def test_open_circuit_yields_error_sse(self):
        """
        SENTINEL — OPEN circuit breaker must yield an error SSE and return early.
        Remove breaker check → request goes to provider even when circuit is OPEN → test fails.
        """
        open_breaker = self._make_open_breaker()
        with patch(
            "app.services.chat_service.get_circuit_breaker", return_value=open_breaker
        ):
            chunks = []
            async for chunk in chat_service(
                model="gpt-4",
                messages=[{"role": "user", "content": "hi"}],
                provider="openai",
            ):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert "circuit_open" in chunks[0]
        assert "event: error" in chunks[0]

    @pytest.mark.asyncio
    async def test_open_circuit_does_not_call_provider(self):
        """
        SENTINEL — when circuit is OPEN, no provider call must be made.
        Remove early return → provider called despite open circuit → test fails.
        """
        open_breaker = self._make_open_breaker()
        with patch(
            "app.services.chat_service.get_circuit_breaker", return_value=open_breaker
        ), patch("app.services.chat_service._get_openai_client") as mock_client:
            async for _ in chat_service(
                model="gpt-4",
                messages=[{"role": "user", "content": "hi"}],
                provider="openai",
            ):
                pass

        mock_client.assert_not_called()


# ===========================================================================
# chat_service — message truncation
# ===========================================================================


class TestMessageTruncation:
    @pytest.mark.asyncio
    async def test_messages_over_limit_are_truncated(self):
        """
        SENTINEL — when more than MAX_MESSAGES messages, oversized context must be trimmed.
        Remove truncation → provider receives full history → context window exceeded → test fails.
        """
        # Build a large history (10 extra messages)
        messages = [
            {"role": "user", "content": f"msg {i}"} for i in range(MAX_MESSAGES + 10)
        ]

        captured_messages = []

        async def _fake_ollama(**kwargs):
            captured_messages.extend(kwargs.get("messages", []))
            return
            yield  # make it an async generator

        breaker = MagicMock()
        breaker.allow_request.return_value = True
        breaker.record_success = MagicMock()
        breaker.record_failure = MagicMock()

        with patch(
            "app.services.chat_service.get_circuit_breaker", return_value=breaker
        ), patch("app.services.chat_service._stream_from_ollama", new=_fake_ollama):
            async for _ in chat_service(
                model="m", messages=messages, provider="ollama"
            ):
                pass

        assert len(captured_messages) <= MAX_MESSAGES

    @pytest.mark.asyncio
    async def test_system_messages_preserved_during_truncation(self):
        """
        SENTINEL — system messages must survive truncation (they carry instructions).
        Drop system messages during truncation → LLM loses its persona → test fails.
        """
        system_msg = {"role": "system", "content": "SENTINEL_SYSTEM"}
        user_msgs = [
            {"role": "user", "content": f"msg {i}"} for i in range(MAX_MESSAGES)
        ]
        messages = [system_msg] + user_msgs  # One over limit

        captured_messages = []

        async def _fake_ollama(**kwargs):
            captured_messages.extend(kwargs.get("messages", []))
            return
            yield

        breaker = MagicMock()
        breaker.allow_request.return_value = True
        breaker.record_success = MagicMock()
        breaker.record_failure = MagicMock()

        with patch(
            "app.services.chat_service.get_circuit_breaker", return_value=breaker
        ), patch("app.services.chat_service._stream_from_ollama", new=_fake_ollama):
            async for _ in chat_service(
                model="m", messages=messages, provider="ollama"
            ):
                pass

        roles = [m["role"] for m in captured_messages]
        assert "system" in roles, "System message lost during truncation"
        contents = [m["content"] for m in captured_messages]
        assert "SENTINEL_SYSTEM" in contents


# ===========================================================================
# chat_service — circuit breaker telemetry (record_success / record_failure)
# ===========================================================================


class TestCircuitBreakerTelemetry:
    @pytest.mark.asyncio
    async def test_record_success_called_after_successful_stream(self):
        """
        SENTINEL — on a clean stream, breaker.record_success() must be called.
        Remove record_success → breaker never resets from HALF_OPEN → test fails.
        """

        async def _fake_ollama(**kwargs):
            yield f"data: {json.dumps({'delta': 'hello'})}\n\n"

        breaker = MagicMock()
        breaker.allow_request.return_value = True
        breaker.record_success = MagicMock()
        breaker.record_failure = MagicMock()

        with patch(
            "app.services.chat_service.get_circuit_breaker", return_value=breaker
        ), patch("app.services.chat_service._stream_from_ollama", new=_fake_ollama):
            async for _ in chat_service(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                provider="ollama",
            ):
                pass

        breaker.record_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_failure_called_after_all_retries_exhausted(self):
        """
        SENTINEL — when every attempt fails, breaker.record_failure() must be called.
        Remove record_failure → circuit never opens on repeated failures → test fails.
        """

        async def _failing_ollama(**kwargs):
            raise ConnectionError("no ollama")
            yield  # make generator

        breaker = MagicMock()
        breaker.allow_request.return_value = True
        breaker.record_success = MagicMock()
        breaker.record_failure = MagicMock()

        with patch(
            "app.services.chat_service.get_circuit_breaker", return_value=breaker
        ), patch(
            "app.services.chat_service._stream_from_ollama", new=_failing_ollama
        ), patch(
            "app.services.chat_service.asyncio.sleep", new=AsyncMock()
        ):
            async for _ in chat_service(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                provider="ollama",
            ):
                pass

        breaker.record_failure.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_sse_yielded_after_all_retries_exhausted(self):
        """
        SENTINEL — exhausted retries must yield a provider_error SSE.
        Remove the final error yield → caller silently gets no content → test fails.
        """

        async def _failing_ollama(**kwargs):
            raise ConnectionError("SENTINEL_ERROR_MSG")
            yield

        breaker = MagicMock()
        breaker.allow_request.return_value = True
        breaker.record_success = MagicMock()
        breaker.record_failure = MagicMock()

        chunks = []
        with patch(
            "app.services.chat_service.get_circuit_breaker", return_value=breaker
        ), patch(
            "app.services.chat_service._stream_from_ollama", new=_failing_ollama
        ), patch(
            "app.services.chat_service.asyncio.sleep", new=AsyncMock()
        ):
            async for chunk in chat_service(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                provider="ollama",
            ):
                chunks.append(chunk)

        error_chunks = [c for c in chunks if "provider_error" in c]
        assert len(error_chunks) == 1, "Expected exactly one provider_error SSE"


# ===========================================================================
# chat_service — retry behavior
# ===========================================================================


class TestRetryBehavior:
    @pytest.mark.asyncio
    async def test_retries_on_transient_failure(self):
        """
        SENTINEL — transient failures must trigger retries up to MAX_RETRIES times.
        Remove the while loop → first failure is final, no retry → test fails.
        """
        from app.services.chat_service import MAX_RETRIES

        call_count = 0

        async def _flaky_ollama(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < MAX_RETRIES + 1:
                raise ConnectionError("transient")
            yield f"data: {json.dumps({'delta': 'ok'})}\n\n"

        breaker = MagicMock()
        breaker.allow_request.return_value = True
        breaker.record_success = MagicMock()
        breaker.record_failure = MagicMock()

        with patch(
            "app.services.chat_service.get_circuit_breaker", return_value=breaker
        ), patch(
            "app.services.chat_service._stream_from_ollama", new=_flaky_ollama
        ), patch(
            "app.services.chat_service.asyncio.sleep", new=AsyncMock()
        ):
            chunks = []
            async for chunk in chat_service(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                provider="ollama",
            ):
                chunks.append(chunk)

        # Should have succeeded on final retry — no error chunk
        error_chunks = [c for c in chunks if "event: error" in c]
        assert len(error_chunks) == 0, "Should succeed after retries"
        assert (
            call_count == MAX_RETRIES + 1
        )  # Failed MAX_RETRIES times, succeeded on last
