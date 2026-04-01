from __future__ import annotations

"""Service layer for chat-related operations.

This module abstracts away direct OpenAI calls so that controller
end-points can remain thin, declarative, and easily testable.

Provider abstraction:
  Providers are normalised to "openai" or "ollama".  Each provider has an
  independent circuit breaker (3 consecutive failures → OPEN; 60 s recovery).
  Failed requests are retried up to MAX_RETRIES times with exponential backoff
  before the failure is recorded against the circuit breaker.
"""

from typing import AsyncGenerator, Dict, List
import asyncio
import json
import logging
import time
import httpx
from app.dependencies import _get_openai_client, _get_ollama_base_url
from app.core.circuit_breaker import get_circuit_breaker, ProviderUnavailableError


logger = logging.getLogger(__name__)

# Public symbol exports
__all__ = ["chat_service"]

# Retry configuration — applies per streaming request before the failure
# is recorded against the circuit breaker.
MAX_RETRIES = 2
RETRY_BASE_DELAY = 0.5  # seconds; doubles on each attempt

# Maximum number of messages passed to a provider (truncate oldest first).
MAX_MESSAGES = 50


def _normalise_provider(provider: str) -> str:
    if provider.lower() in {"ollama", "local", "local-ollama"}:
        return "ollama"
    return "openai"


# ---------------------------------------------------------------------------
# Public service-layer API
# ---------------------------------------------------------------------------


async def chat_service(
    *,
    model: str,
    messages: List[Dict[str, str]],
    provider: str = "openai",
) -> AsyncGenerator[str, None]:
    """Yield Server-Sent Event (SSE) formatted chunks from an AI provider.

    Wraps the provider call with:
    - Circuit breaker: rejects calls when provider is in OPEN state.
    - Retry with exponential backoff: up to MAX_RETRIES before propagating.
    - Message limit: truncates to MAX_MESSAGES (keeps system msg + most recent).

    Parameters
    ----------
    model: str
        The chat model name for the chosen provider.
    messages: list[dict[str, str]]
        Chat history in OpenAI message format.
    provider: str
        "openai" | "ollama" | "local" | "local-ollama"

    Yields
    ------
    str
        Pre-formatted SSE ``data: ...`` strings.
    """
    canonical = _normalise_provider(provider)
    breaker = get_circuit_breaker(canonical)

    if not breaker.allow_request():
        logger.warning("Circuit open for provider '%s' — rejecting request", canonical)
        yield f"event: error\ndata: {json.dumps({'error': f'Provider {canonical!r} is temporarily unavailable. Please try again later.', 'code': 'circuit_open'})}\n\n"
        return

    # Enforce message limit: keep system messages + most recent user/assistant
    if len(messages) > MAX_MESSAGES:
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]
        messages = system_msgs + other_msgs[-(MAX_MESSAGES - len(system_msgs)):]
        logger.debug("Truncated messages to %d (limit=%d)", len(messages), MAX_MESSAGES)

    attempt = 0
    last_exc: Exception | None = None

    while attempt <= MAX_RETRIES:
        if attempt > 0:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.info("Retrying provider '%s' (attempt %d/%d) after %.1fs", canonical, attempt, MAX_RETRIES, delay)
            await asyncio.sleep(delay)

        try:
            start_time = time.monotonic()
            chunks_yielded = 0

            if canonical == "ollama":
                async for sse in _stream_from_ollama(model=model, messages=messages):
                    yield sse
                    chunks_yielded += 1
            else:
                # OpenAI path
                client = _get_openai_client()
                response = await client.responses.create(
                    model=model,
                    input=messages,
                    stream=True,
                )
                async for chunk in response:
                    logger.debug("Service received chunk: %s", chunk)
                    data: Dict[str, object] = chunk.model_dump(exclude_none=True)
                    await asyncio.sleep(0)
                    yield f"data: {json.dumps(data)}\n\n"
                    chunks_yielded += 1

            # Success: record to circuit breaker and log telemetry
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.info(
                "Provider '%s' stream complete: model=%s chunks=%d latency_ms=%.1f",
                canonical, model, chunks_yielded, elapsed_ms,
            )
            breaker.record_success()
            return

        except ProviderUnavailableError:
            # Circuit already open — bail immediately (no retry)
            raise
        except Exception as exc:
            last_exc = exc
            attempt += 1
            logger.warning(
                "Provider '%s' error (attempt %d/%d): %s",
                canonical, attempt, MAX_RETRIES + 1, exc,
            )

    # All retries exhausted — record failure and yield error SSE
    if last_exc is not None:
        breaker.record_failure()
        logger.error("Provider '%s' failed after %d attempts: %s", canonical, MAX_RETRIES + 1, last_exc)
        yield f"event: error\ndata: {json.dumps({'error': str(last_exc), 'code': 'provider_error'})}\n\n"


async def _stream_from_ollama(*, model: str, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """Stream chat completions from an Ollama server and emit SSE-formatted chunks.

    This uses Ollama's native REST API `/api/chat` with streaming enabled and
    rewraps incremental message content as `{ "delta": "..." }` SSE events.

    Sends periodic heartbeat/status events during model loading to keep connection alive
    and provide user feedback.
    """
    base_url = _get_ollama_base_url()
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    # Timeout for the entire request (2 minutes should be enough even for cold model loads)
    timeout_config = httpx.Timeout(120.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout_config) as client:
        try:
            # Send initial status event
            yield f"event: status\ndata: {json.dumps({'message': 'Connecting to model...'})}\n\n"

            async with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()

                # Send status after connection established
                yield f"event: status\ndata: {json.dumps({'message': 'Waiting for model response...'})}\n\n"

                # Simple approach: use asyncio.wait_for with periodic timeouts to send heartbeats
                line_iterator = resp.aiter_lines()
                heartbeat_seconds = 0

                while True:
                    try:
                        # Wait up to 5 seconds for next line
                        line = await asyncio.wait_for(anext(line_iterator), timeout=5.0)

                        if not line:
                            continue

                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # Separate thinking from content for progressive disclosure in UI
                        message = obj.get("message", {})

                        # Send thinking as separate event type
                        thinking = message.get("thinking", "")
                        if thinking:
                            yield f"event: thinking\ndata: {json.dumps({'delta': thinking})}\n\n"

                        # Send content as regular message
                        content = message.get("content", "")
                        if content:
                            yield f"data: {json.dumps({'delta': content})}\n\n"

                        # Stop when the stream signals completion
                        if obj.get("done") is True:
                            break

                    except asyncio.TimeoutError:
                        # No response in 5 seconds - send heartbeat
                        heartbeat_seconds += 5
                        logger.debug("Sending heartbeat after %ds of silence", heartbeat_seconds)
                        yield f"event: heartbeat\ndata: {json.dumps({'elapsed': heartbeat_seconds, 'message': f'Generating response... ({heartbeat_seconds}s)'})}\n\n"
                    except StopAsyncIteration:
                        # Stream ended
                        break

        except httpx.TimeoutException as timeout_err:
            logger.error("Ollama request timeout: %s", timeout_err)
            yield f"event: error\ndata: {json.dumps({'error': 'Request timeout - model may still be loading. Please try again.', 'code': 'timeout'})}\n\n"
        except httpx.HTTPError as http_err:
            logger.error("Ollama HTTP error: %s", http_err)
            yield f"event: error\ndata: {json.dumps({'error': str(http_err), 'code': 'http_error'})}\n\n"
