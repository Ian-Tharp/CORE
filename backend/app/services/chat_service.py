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
    """
    base_url = _get_ollama_base_url()
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        # Some Ollama versions may send partial lines; skip until valid
                        continue

                    # Incremental assistant content
                    delta = obj.get("message", {}).get("content") or ""
                    if delta:
                        yield f"data: {json.dumps({'delta': delta})}\n\n"

                    # Stop when the stream signals completion
                    if obj.get("done") is True:
                        break
        except httpx.HTTPError as http_err:
            yield f"event: error\ndata: {json.dumps({'error': str(http_err)})}\n\n"
