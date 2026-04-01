"""
Tests for CircuitBreaker and chat_service circuit breaker + retry integration.

Sentinel-value contract:
- Remove record_failure() → circuit never opens → test_circuit_opens_after_threshold fails
- Remove OPEN state check in allow_request() → requests not rejected → test_open_circuit_rejects fails
- Remove retry loop → no second attempt → test_retry_on_failure fails
- Remove message truncation → all messages sent → test_message_limit_truncates fails
"""

from __future__ import annotations

import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    ProviderUnavailableError,
    get_circuit_breaker,
    reset_all_breakers,
)


# ---------------------------------------------------------------------------
# CircuitBreaker state machine
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_breakers():
    """Ensure every test starts with a clean circuit breaker registry."""
    reset_all_breakers()
    yield
    reset_all_breakers()


class TestCircuitBreakerStateMachine:
    def test_initial_state_is_closed(self):
        breaker = CircuitBreaker("test", failure_threshold=3)
        assert breaker.state == CircuitState.CLOSED

    def test_allows_requests_when_closed(self):
        breaker = CircuitBreaker("test")
        assert breaker.allow_request() is True

    def test_circuit_opens_after_threshold(self):
        """
        SENTINEL TEST — after failure_threshold failures, state must be OPEN.
        Remove record_failure() logic → state stays CLOSED → test fails.
        """
        breaker = CircuitBreaker("test", failure_threshold=3)
        for _ in range(3):
            breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

    def test_open_circuit_rejects_requests(self):
        """
        SENTINEL TEST — OPEN circuit must return False from allow_request().
        Remove OPEN check → returns True → test fails.
        """
        breaker = CircuitBreaker("test", failure_threshold=1)
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        assert breaker.allow_request() is False

    def test_circuit_transitions_to_half_open_after_timeout(self):
        """
        SENTINEL TEST — after recovery_timeout elapses, allow_request() must
        return True and move state to HALF_OPEN.
        Remove timeout check → stays OPEN → test fails.
        """
        breaker = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        time.sleep(0.02)  # wait for recovery timeout
        result = breaker.allow_request()

        assert result is True
        assert breaker.state == CircuitState.HALF_OPEN

    def test_half_open_closes_on_success(self):
        """Successful probe in HALF_OPEN closes the circuit."""
        breaker = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)
        breaker.record_failure()
        time.sleep(0.02)
        breaker.allow_request()  # transition to HALF_OPEN

        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self):
        """
        SENTINEL TEST — a failure in HALF_OPEN must reopen the circuit.
        Remove HALF_OPEN → OPEN transition → stays HALF_OPEN → test fails.
        """
        breaker = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)
        breaker.record_failure()
        time.sleep(0.02)
        breaker.allow_request()  # → HALF_OPEN

        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        """Two failures then success: consecutive_failures resets to 0."""
        breaker = CircuitBreaker("test", failure_threshold=5)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()
        assert breaker.stats()["consecutive_failures"] == 0

    def test_reset_returns_to_closed(self):
        breaker = CircuitBreaker("test", failure_threshold=1)
        breaker.record_failure()
        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.allow_request() is True

    def test_stats_returns_required_keys(self):
        breaker = CircuitBreaker("test")
        stats = breaker.stats()
        for key in ("name", "state", "consecutive_failures"):
            assert key in stats

    def test_get_circuit_breaker_returns_same_instance(self):
        """Registry must return the same instance for the same provider."""
        a = get_circuit_breaker("my-provider")
        b = get_circuit_breaker("my-provider")
        assert a is b

    def test_failure_below_threshold_keeps_closed(self):
        breaker = CircuitBreaker("test", failure_threshold=5)
        for _ in range(4):
            breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.allow_request() is True


# ---------------------------------------------------------------------------
# chat_service integration
# ---------------------------------------------------------------------------

async def _collect(gen) -> list:
    """Drain an async generator and return all yielded strings."""
    return [chunk async for chunk in gen]


class TestChatServiceCircuitBreaker:
    @pytest.mark.asyncio
    async def test_open_circuit_yields_error_sse(self):
        """
        SENTINEL TEST — when circuit is OPEN, chat_service must yield an
        error SSE without calling the provider.
        Remove circuit check → provider called despite OPEN → test fails.
        """
        from app.services import chat_service as svc_mod
        from app.core.circuit_breaker import get_circuit_breaker

        # Force the breaker open
        breaker = get_circuit_breaker("openai")
        breaker._state = CircuitState.OPEN
        breaker._opened_at = time.monotonic()

        chunks = await _collect(svc_mod.chat_service(model="gpt-x", messages=[], provider="openai"))

        assert len(chunks) == 1
        assert "circuit_open" in chunks[0]
        assert "event: error" in chunks[0]

    @pytest.mark.asyncio
    async def test_success_records_to_breaker(self):
        """
        SENTINEL TEST — a successful stream must call breaker.record_success().
        Remove record_success() → breaker never closes after recovery → test fails.
        """
        from app.services import chat_service as svc_mod
        from app.core.circuit_breaker import get_circuit_breaker

        breaker = get_circuit_breaker("openai")
        breaker.reset()

        # Mock the OpenAI streaming path
        mock_chunk = MagicMock()
        mock_chunk.model_dump.return_value = {"delta": "hello"}

        async def _fake_stream():
            yield mock_chunk

        mock_response = AsyncMock()
        mock_response.__aiter__ = MagicMock(return_value=_fake_stream())

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)

        with patch("app.services.chat_service._get_openai_client", return_value=mock_client):
            with patch.object(breaker, "record_success", wraps=breaker.record_success) as spy:
                await _collect(svc_mod.chat_service(model="gpt-x", messages=[], provider="openai"))

        spy.assert_called_once()

    @pytest.mark.asyncio
    async def test_failure_records_to_breaker(self):
        """
        SENTINEL TEST — provider error must call breaker.record_failure().
        Remove record_failure() → breaker never opens → test fails.
        """
        from app.services import chat_service as svc_mod
        from app.core.circuit_breaker import get_circuit_breaker

        breaker = get_circuit_breaker("openai")
        breaker.reset()

        with patch(
            "app.services.chat_service._get_openai_client",
            side_effect=RuntimeError("provider down"),
        ):
            with patch("app.services.chat_service.asyncio.sleep", new=AsyncMock()):
                with patch.object(breaker, "record_failure", wraps=breaker.record_failure) as spy:
                    chunks = await _collect(
                        svc_mod.chat_service(model="gpt-x", messages=[], provider="openai")
                    )

        spy.assert_called_once()
        error_chunks = [c for c in chunks if "event: error" in c]
        assert len(error_chunks) == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """
        SENTINEL TEST — a transient failure must be retried MAX_RETRIES times.
        Remove retry loop → only 1 attempt → call_count == 1 → test fails.
        """
        import app.services.chat_service as svc_mod
        original_max = svc_mod.MAX_RETRIES
        try:
            svc_mod.MAX_RETRIES = 2
            call_count = 0

            def _failing_client():
                nonlocal call_count
                call_count += 1
                raise RuntimeError("transient error")

            with patch("app.services.chat_service._get_openai_client", side_effect=_failing_client):
                with patch("app.services.chat_service.asyncio.sleep", new=AsyncMock()):
                    await _collect(svc_mod.chat_service(model="g", messages=[], provider="openai"))

            # Should have been called MAX_RETRIES + 1 times (initial + retries)
            assert call_count == svc_mod.MAX_RETRIES + 1
        finally:
            svc_mod.MAX_RETRIES = original_max

    @pytest.mark.asyncio
    async def test_message_limit_truncates(self):
        """
        SENTINEL TEST — when messages > MAX_MESSAGES, only MAX_MESSAGES are sent.
        Remove truncation → all 100 messages sent → call receives 100 → test fails.
        """
        import app.services.chat_service as svc_mod
        original_max = svc_mod.MAX_MESSAGES
        try:
            svc_mod.MAX_MESSAGES = 5
            messages = [{"role": "user", "content": f"msg {i}"} for i in range(20)]

            captured_messages = []

            async def _fake_create(model, input, stream):
                captured_messages.extend(input)
                return  # return None so the async for loop immediately ends

            mock_client = AsyncMock()
            mock_client.responses.create = _fake_create

            # Need an async generator for the streaming path
            async def _empty_gen():
                return
                yield  # noqa: unreachable

            async def _fake_create2(model, input, stream):
                captured_messages.clear()
                captured_messages.extend(input)
                return _empty_gen()

            mock_client.responses.create = _fake_create2

            with patch("app.services.chat_service._get_openai_client", return_value=mock_client):
                with patch("app.services.chat_service.asyncio.sleep", new=AsyncMock()):
                    await _collect(svc_mod.chat_service(model="g", messages=messages, provider="openai"))

            assert len(captured_messages) <= svc_mod.MAX_MESSAGES
        finally:
            svc_mod.MAX_MESSAGES = original_max

    @pytest.mark.asyncio
    async def test_ollama_provider_normalised(self):
        """'local' and 'local-ollama' both resolve to the Ollama path."""
        from app.services import chat_service as svc_mod

        for alias in ("local", "local-ollama", "ollama"):
            assert svc_mod._normalise_provider(alias) == "ollama"

    def test_normalise_openai(self):
        from app.services import chat_service as svc_mod
        assert svc_mod._normalise_provider("openai") == "openai"
        assert svc_mod._normalise_provider("OPENAI") == "openai"
