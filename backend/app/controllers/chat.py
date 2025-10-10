from fastapi import APIRouter, status, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, AsyncGenerator, Optional
import logging
import uuid

from app.services.chat_service import chat_service
from app.repository.conversation_repository import (
    create_conversation,
    append_message,
)
from app.services import knowledgebase_service as kb_svc

router = APIRouter()
logger = logging.getLogger(__name__)

ALLOWED_PROVIDERS = {"openai", "anthropic", "ollama", "local"}
ALLOWED_MODELS = {"gpt-5", "gpt-4.1", "gpt-4o", "gpt-4o-mini", "o3-mini", "claude-3-5"}

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    conversation_id: str | None = None
    stream: bool = True  # Default to streaming
    provider: Optional[str] = "openai"  # "openai" | "anthropic" | "ollama" (local)
    # Optional knowledgebase RAG configuration
    kb_mode: Optional[str] = None  # 'all' | 'file'
    kb_file_id: Optional[str] = None
    kb_embedding_provider: Optional[str] = None  # 'openai' | 'local'
    kb_local_model: Optional[str] = None


@router.post(
    "/chat/stream", response_class=StreamingResponse, status_code=status.HTTP_200_OK
)
async def chat_stream(request: ChatRequest):
    """
    Stream chat responses from OpenAI's /v1/chat/completions endpoint.
    """
    # RSI TODO: Add request-level correlation id and include in logs + response headers for traceability.
    # RSI TODO: Enforce provider/model allowlist via config; validate inputs and size limits.
    # RSI TODO: Surface structured error events with codes; add rate limiting/backpressure.

    # Correlation id for tracing
    correlation_id = str(uuid.uuid4())
    logger.info("chat_stream start correlation_id=%s provider=%s model=%s", correlation_id, request.provider, request.model)

    # Provider/model validation
    if request.provider and request.provider not in ALLOWED_PROVIDERS:
        raise HTTPException(status_code=400, detail="Unsupported provider")
    if request.model and request.model not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported model")

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

        # Optionally augment messages with RAG context from knowledgebase
        final_messages = [msg.dict() for msg in request.messages]
        if request.kb_mode in {"all", "file"}:
            user_msg = request.messages[-1].content if request.messages else ""
            ctx = await kb_svc.retrieve_context(
                query=user_msg,
                mode=request.kb_mode,
                file_id=request.kb_file_id,
                provider=(request.kb_embedding_provider or "openai"),
                local_model=request.kb_local_model,
            )
            final_messages = kb_svc.build_rag_messages(final_messages, context_chunks=ctx.get("chunks", []))

        try:
            async for chunk in chat_service(
                model=request.model,
                messages=final_messages,
                provider=request.provider or "openai",
            ):
                logger.debug("Received chunk: %s", chunk)

                # Try to accumulate the assistant message content from the SSE
                try:
                    import json as _json

                    if chunk.startswith("data:"):
                        data_str = chunk.partition(":")[2].strip()
                        data_json = _json.loads(data_str)
                        assistant_accum += data_json.get("delta", "")
                except Exception:  # noqa: BLE001
                    pass

                yield chunk
        except Exception as exc:  # noqa: BLE001
            import json as _json
            logger.error("chat_stream error correlation_id=%s error=%s", correlation_id, str(exc))
            err = {"code": "stream_error", "message": str(exc), "correlation_id": correlation_id}
            yield f"event: error\ndata: {_json.dumps(err)}\n\n"

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
        "X-Correlation-Id": correlation_id,
        # Explicit CORS headers to avoid browser blocking even on SSE streams
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "*",
        "Access-Control-Allow-Headers": "*",
    }
    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=headers
    )
