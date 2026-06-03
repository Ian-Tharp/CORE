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
from app.dependencies import (
    _get_openai_client,
    _get_ollama_base_url,
    get_local_provider,
    get_ollama_client,
)
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
    p = provider.lower()
    if p in {"ollama", "local", "local-ollama"}:
        return "ollama"
    if p == "anthropic":
        return "anthropic"
    return "openai"


# Anthropic requires an explicit output-token cap (this bounds the response, it
# is not the context window). Kept modest for interactive chat turns.
ANTHROPIC_MAX_TOKENS = 2048


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
        messages = system_msgs + other_msgs[-(MAX_MESSAGES - len(system_msgs)) :]
        logger.debug("Truncated messages to %d (limit=%d)", len(messages), MAX_MESSAGES)

    attempt = 0
    last_exc: Exception | None = None

    while attempt <= MAX_RETRIES:
        if attempt > 0:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.info(
                "Retrying provider '%s' (attempt %d/%d) after %.1fs",
                canonical,
                attempt,
                MAX_RETRIES,
                delay,
            )
            await asyncio.sleep(delay)

        try:
            start_time = time.monotonic()
            chunks_yielded = 0

            if canonical == "ollama":
                # The local provider is agnostic: Ollama (native API) or LM Studio
                # (OpenAI-compatible), selected by CORE_LOCAL_PROVIDER. Route to the
                # active one so a LM-Studio-only machine never reaches for Ollama.
                if get_local_provider() == "lmstudio":
                    streamer = _stream_from_lmstudio(model=model, messages=messages)
                else:
                    streamer = _stream_from_ollama(model=model, messages=messages)
                async for sse in streamer:
                    yield sse
                    chunks_yielded += 1
            elif canonical == "anthropic":
                async for sse in _stream_from_anthropic(model=model, messages=messages):
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
                canonical,
                model,
                chunks_yielded,
                elapsed_ms,
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
                canonical,
                attempt,
                MAX_RETRIES + 1,
                exc,
            )

    # All retries exhausted — record failure and yield error SSE
    if last_exc is not None:
        breaker.record_failure()
        logger.error(
            "Provider '%s' failed after %d attempts: %s",
            canonical,
            MAX_RETRIES + 1,
            last_exc,
        )
        yield f"event: error\ndata: {json.dumps({'error': str(last_exc), 'code': 'provider_error'})}\n\n"


async def _stream_from_ollama(
    *, model: str, messages: List[Dict[str, str]]
) -> AsyncGenerator[str, None]:
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
                        logger.debug(
                            "Sending heartbeat after %ds of silence", heartbeat_seconds
                        )
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


async def _stream_from_lmstudio(
    *, model: str, messages: List[Dict[str, str]]
) -> AsyncGenerator[str, None]:
    """Stream chat completions from LM Studio and emit SSE-formatted chunks.

    LM Studio exposes an OpenAI-compatible server, so this uses the shared
    provider-aware client (`get_ollama_client`, which points at LM Studio when
    ``CORE_LOCAL_PROVIDER=lmstudio``) and its ``/v1/chat/completions`` streaming
    endpoint. Content deltas are rewrapped as ``{ "delta": "..." }`` events, and
    reasoning deltas (emitted by reasoning models as ``reasoning_content``) are
    surfaced as ``thinking`` events for progressive disclosure — mirroring the
    Ollama path. Periodic heartbeats keep the connection alive during cold loads.
    """
    client = get_ollama_client()

    yield f"event: status\ndata: {json.dumps({'message': 'Connecting to model...'})}\n\n"

    # Let connection/setup failures raise so the caller's retry + circuit breaker
    # can act on them (consistent with the OpenAI path).
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    yield f"event: status\ndata: {json.dumps({'message': 'Waiting for model response...'})}\n\n"

    iterator = stream.__aiter__()
    heartbeat_seconds = 0

    while True:
        try:
            chunk = await asyncio.wait_for(iterator.__anext__(), timeout=5.0)
        except asyncio.TimeoutError:
            heartbeat_seconds += 5
            yield f"event: heartbeat\ndata: {json.dumps({'elapsed': heartbeat_seconds, 'message': f'Generating response... ({heartbeat_seconds}s)'})}\n\n"
            continue
        except StopAsyncIteration:
            break

        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        delta = choices[0].delta

        # Reasoning models (also exposed by LM Studio) stream their chain of
        # thought separately; surface it as a thinking event when present.
        thinking = getattr(delta, "reasoning_content", None) or getattr(
            delta, "reasoning", None
        )
        if thinking:
            yield f"event: thinking\ndata: {json.dumps({'delta': thinking})}\n\n"

        content = getattr(delta, "content", None)
        if content:
            yield f"data: {json.dumps({'delta': content})}\n\n"


async def _stream_from_anthropic(
    *, model: str, messages: List[Dict[str, str]]
) -> AsyncGenerator[str, None]:
    """Stream chat completions from the Anthropic Messages API as SSE chunks.

    Differences from the OpenAI path that this normalises:
    - Anthropic takes the system prompt as a top-level ``system`` argument, not
      as a message, so system turns are merged out of the conversation list.
    - A registry key (e.g. ``claude-3-haiku``) is resolved to its real API model
      id (``claude-3-haiku-20240307``) via the model config.
    Text deltas are rewrapped as ``{ "delta": "..." }`` events so the controller
    and UI parse them identically to the Ollama/OpenAI streams.
    """
    from app.dependencies import _get_async_anthropic_client
    from app.config.models import get_model_config

    client = _get_async_anthropic_client()

    cfg = get_model_config(model)
    api_model = cfg.model_name if cfg else model

    system_parts = [
        m.get("content", "") for m in messages if m.get("role") == "system"
    ]
    system_prompt = "\n\n".join(p for p in system_parts if p)
    turns = [
        {"role": m["role"], "content": m.get("content", "")}
        for m in messages
        if m.get("role") in {"user", "assistant"}
    ]

    yield f"event: status\ndata: {json.dumps({'message': 'Connecting to model...'})}\n\n"

    create_kwargs: Dict[str, object] = {
        "model": api_model,
        "max_tokens": ANTHROPIC_MAX_TOKENS,
        "messages": turns,
    }
    if system_prompt:
        create_kwargs["system"] = system_prompt

    # Let setup/auth failures raise so the caller's retry + circuit breaker act
    # (consistent with the OpenAI/LM Studio paths).
    async with client.messages.stream(**create_kwargs) as stream:
        yield f"event: status\ndata: {json.dumps({'message': 'Waiting for model response...'})}\n\n"
        async for text in stream.text_stream:
            if text:
                yield f"data: {json.dumps({'delta': text})}\n\n"
