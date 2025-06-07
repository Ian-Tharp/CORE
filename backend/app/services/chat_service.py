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

# Public symbol exports
__all__ = ["chat_service"]


# ---------------------------------------------------------------------------
# Public service-layer API
# ---------------------------------------------------------------------------

# NOTE: This function was formerly named ``generate_chat_stream``. It has been
# renamed to ``chat_service`` to better reflect its role as the primary entry
# point for streaming chat completions to the controller layer.

async def chat_service(
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
        # Using the newer "Responses" API (see: https://github.com/openai/openai-python#responses-api)
        response = await client.responses.create(
            model=model,
            input=messages,  # Our ``messages`` already match the simple text schema
            stream=True,
        )

        async for chunk in response:
            logger.debug("Service received chunk: %s", chunk)

            # The OpenAI Python SDK v1+ returns Pydantic models. We can leverage
            # their built-in ``model_dump`` helper to serialise the chunk to a
            # plain-old Python ``dict`` while automatically filtering ``None``
            # values. This keeps the service logic compact and forwards the
            # exact response schema expected by downstream consumers.
            data: Dict[str, object] = chunk.model_dump(exclude_none=True)

            # Allow the event loop to process other tasks to avoid starvation.
            await asyncio.sleep(0)

            yield f"data: {json.dumps(data)}\n\n"

    except Exception as exc:  # pylint: disable=broad-except
        # Log the error internally and propagate it to the client in SSE format.
        logger.error("Streaming error: %s", exc)
        yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
