from __future__ import annotations

"""Service layer for chat-related operations.

This module abstracts away direct OpenAI calls so that controller
end-points can remain thin, declarative, and easily testable.
"""

from typing import AsyncGenerator, Dict, List
import asyncio
import json
import logging
from dependencies import _get_openai_client


logger = logging.getLogger(__name__)

__all__ = ["generate_chat_stream"]


# ---------------------------------------------------------------------------
# Public service-layer API
# ---------------------------------------------------------------------------
async def generate_chat_stream(
    *,
    model: str,
    messages: List[Dict[str, str]],
) -> AsyncGenerator[str, None]:
    """Yield Server-Sent Event (SSE) formatted chunks from an OpenAI chat stream.

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

    client = _get_openai_client()

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )

        async for chunk in response:
            logger.debug("Service received chunk: %s", chunk)

            # Build a minimal dict matching the chunk structure expected by the
            # front-end while stripping out ``None`` values for compactness.
            data: Dict[str, object] = {
                "id": getattr(chunk, "id", None),
                "object": getattr(chunk, "object", None),
                "created": getattr(chunk, "created", None),
                "model": getattr(chunk, "model", None),
                "system_fingerprint": getattr(chunk, "system_fingerprint", None),
                "choices": [
                    {
                        "index": choice.index,
                        "delta": (
                            getattr(choice.delta, "content", None)
                            if hasattr(choice, "delta")
                            else None
                        ),
                        "finish_reason": getattr(choice, "finish_reason", None),
                    }
                    for choice in getattr(chunk, "choices", [])
                ],
            }

            # Remove keys whose value is ``None`` for cleaner JSON output.
            data = {k: v for k, v in data.items() if v is not None}

            # Allow the event loop to process other tasks to avoid starvation.
            await asyncio.sleep(0)

            yield f"data: {json.dumps(data)}\n\n"

    except Exception as exc:  # pylint: disable=broad-except
        # Log the error internally and propagate it to the client in SSE format.
        logger.error("Streaming error: %s", exc)
        yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
