from __future__ import annotations

"""Service layer for chat-related operations.

This module abstracts away direct OpenAI calls so that controller
end-points can remain thin, declarative, and easily testable.
"""

from typing import AsyncGenerator, Dict, List
import asyncio
import json
import logging
import httpx
from app.dependencies import _get_openai_client, _get_ollama_base_url


logger = logging.getLogger(__name__)

# Public symbol exports
__all__ = ["chat_service"]


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

    Parameters
    ----------
    model: str
        The name of the OpenAI chat model to use (e.g. ``gpt-4o``).
    messages: list[dict[str, str]]
        The chat history in the shape expected by the OpenAI API.

    Yields
    ------
    str
        Pre-formatted SSE ``data: ...`` strings ready to be returned by
        ``fastapi.responses.StreamingResponse``.
    """

    try:
        # RSI TODO: Add per-provider telemetry (latency, ttfb, tokens/sec) and structured logs.
        # RSI TODO: Implement provider abstraction layer and circuit breakers/retries.
        # RSI TODO: Enforce max tokens/messages and redact PII before sending to providers - ollama is fine since it's local.
        if provider.lower() in {"ollama", "local", "local-ollama"}:
            async for sse in _stream_from_ollama(model=model, messages=messages):
                yield sse
            return

        # Default: OpenAI
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

    except Exception as exc:  # noqa: BLE001
        logger.error("Streaming error: %s", exc)
        yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"


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
