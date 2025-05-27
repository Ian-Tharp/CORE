from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, AsyncGenerator, Dict, Any
import logging

from services.chat_service import generate_chat_stream

router = APIRouter()
logger = logging.getLogger(__name__)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = True  # Default to streaming


@router.post(
    "/chat/stream", response_class=StreamingResponse, status_code=status.HTTP_200_OK
)
async def chat_stream(request: ChatRequest):
    """
    Stream chat responses from OpenAI's /v1/chat/completions endpoint.
    """

    async def event_stream() -> AsyncGenerator[str, None]:
        async for chunk in generate_chat_stream(
            model=request.model,
            messages=[msg.dict() for msg in request.messages],
        ):
            logger.debug(f"Received chunk: {chunk}")
            yield chunk

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Encoding": "none",
        "Content-Type": "text/event-stream",
        "Transfer-Encoding": "chunked",
        "X-Vercel-AI-Data-Stream": "v1",
    }
    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=headers
    )
