from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, AsyncGenerator
import logging

from services.chat_service import chat_service
from repository.conversation_repository import (
    create_conversation,
    append_message,
)

router = APIRouter()
logger = logging.getLogger(__name__)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    conversation_id: str | None = None
    stream: bool = True  # Default to streaming


@router.post(
    "/chat/stream", response_class=StreamingResponse, status_code=status.HTTP_200_OK
)
async def chat_stream(request: ChatRequest):
    """
    Stream chat responses from OpenAI's /v1/chat/completions endpoint.
    """

    # -------------------------------------------------------------------
    # 1. Ensure we have a conversation id and persist the incoming user msg
    # -------------------------------------------------------------------
    user_message_dict = request.messages[-1].dict()

    conv_id: str
    if request.conversation_id is None:
        # Create new conversation with the *full* message history the client
        # sent (so that system prompts etc. are not lost).
        conv_id = await create_conversation([m.dict() for m in request.messages])
    else:
        conv_id = request.conversation_id
        # Only append the most recent user message – older ones are already
        # persisted server-side.
        await append_message(conv_id, user_message_dict)

    # -------------------------------------------------------------------
    # 2. Stream assistant response while buffering it so we can persist it
    # -------------------------------------------------------------------

    async def event_stream() -> AsyncGenerator[str, None]:
        assistant_accum = ""

        async for chunk in chat_service(
            model=request.model,
            messages=[msg.dict() for msg in request.messages],
        ):
            logger.debug("Received chunk: %s", chunk)

            # Try to accumulate the assistant message content from the SSE
            # payload.  We purposefully do *not* parse the JSON with Pydantic
            # here for performance. The payload looks like::
            #   data: {"delta": "text"}
            try:
                import json as _json

                if chunk.startswith("data:"):
                    data_str = chunk.partition(":")[2].strip()
                    data_json = _json.loads(data_str)
                    assistant_accum += data_json.get("delta", "")
            except Exception:  # noqa: BLE001
                pass

            yield chunk

        # Persist the assistant reply once the OpenAI stream ends.
        if assistant_accum:
            await append_message(
                conv_id,
                {
                    "role": "assistant",
                    "content": assistant_accum,
                },
            )

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Encoding": "none",
        "Content-Type": "text/event-stream",
        "Transfer-Encoding": "chunked",
        "X-Vercel-AI-Data-Stream": "v1",
        "X-Conversation-Id": conv_id,
    }
    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=headers
    )
