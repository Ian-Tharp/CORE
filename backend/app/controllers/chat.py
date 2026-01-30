from fastapi import APIRouter, status, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, AsyncGenerator, Optional
import logging
import uuid
import asyncio

from app.services.chat_service import chat_service
from app.repository.conversation_repository import (
    create_conversation,
    append_message,
    get_conversation,
    update_title,
)
from app.services import knowledgebase_service as kb_svc
from app.services.title_generation_service import generate_conversation_title

router = APIRouter()
logger = logging.getLogger(__name__)

ALLOWED_PROVIDERS = {"openai", "anthropic", "ollama", "local"}
ALLOWED_MODELS = {"gpt-5", "gpt-4.1", "gpt-4o", "gpt-4o-mini", "o3-mini", "claude-3-5"}


async def _generate_and_update_title(conv_id: str, user_msg: str, assistant_msg: str) -> None:
    """Background task to generate and update conversation title.

    This runs asynchronously after the first assistant response completes.
    """
    try:
        logger.info(f"Generating title for conversation {conv_id}")
        title = await generate_conversation_title(user_msg, assistant_msg)
        await update_title(conv_id, title)
        logger.info(f"Updated conversation {conv_id} with title: {title}")
    except Exception as e:
        logger.error(f"Failed to generate/update title for {conv_id}: {e}")

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
    # Only validate models for OpenAI/Anthropic; Ollama has its own model registry
    if request.provider not in {"ollama", "local"}:
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
        thinking_accum = ""

        # Optionally augment messages with RAG context from knowledgebase
        final_messages = [msg.dict() for msg in request.messages]
        if request.kb_mode in {"all", "file"}:
            user_msg = request.messages[-1].content if request.messages else ""
            try:
                # Auto-detect embedding provider: use local embeddings if using local LLM provider
                embedding_provider = request.kb_embedding_provider
                if not embedding_provider:
                    embedding_provider = "local" if request.provider == "ollama" else "openai"

                ctx = await kb_svc.retrieve_context(
                    query=user_msg,
                    mode=request.kb_mode,
                    file_id=request.kb_file_id,
                    provider=embedding_provider,
                    local_model=request.kb_local_model,
                )
                final_messages = kb_svc.build_rag_messages(final_messages, context_chunks=ctx.get("chunks", []))
            except Exception as kb_exc:
                logger.error("KB retrieval error correlation_id=%s error=%s", correlation_id, str(kb_exc))
                # Continue without KB augmentation if retrieval fails
                import json as _json
                err = {"code": "kb_error", "message": f"Knowledge base retrieval failed: {str(kb_exc)}", "correlation_id": correlation_id}
                yield f"event: warning\ndata: {_json.dumps(err)}\n\n"

        try:
            async for chunk in chat_service(
                model=request.model,
                messages=final_messages,
                provider=request.provider or "openai",
            ):
                logger.debug("Received chunk: %s", chunk)

                # Try to accumulate the assistant message content and thinking from the SSE
                try:
                    import json as _json

                    # Parse event type (thinking vs message/content)
                    event_type = "message"  # default
                    if chunk.startswith("event:"):
                        event_line, _, data_line = chunk.partition("\n")
                        event_type = event_line.partition(":")[2].strip()
                        chunk_to_parse = data_line
                    else:
                        chunk_to_parse = chunk

                    if chunk_to_parse.startswith("data:"):
                        data_str = chunk_to_parse.partition(":")[2].strip()
                        data_json = _json.loads(data_str)
                        delta = data_json.get("delta", "")

                        if event_type == "thinking":
                            thinking_accum += delta
                        elif event_type == "message":
                            assistant_accum += delta
                except Exception:  # noqa: BLE001
                    pass

                yield chunk
        except Exception as exc:  # noqa: BLE001
            import json as _json
            logger.error("chat_stream error correlation_id=%s error=%s", correlation_id, str(exc))
            err = {"code": "stream_error", "message": str(exc), "correlation_id": correlation_id}
            yield f"event: error\ndata: {_json.dumps(err)}\n\n"

        # Persist the assistant reply (and thinking if present) once the stream ends.
        if assistant_accum or thinking_accum:
            await append_message(
                conv_id,
                {
                    "role": "assistant",
                    "content": assistant_accum,
                    "thinking": thinking_accum if thinking_accum else None,
                },
            )

            # Generate title after first assistant response (async, non-blocking)
            # Check if this is the first assistant message (should have exactly 2 messages: user + assistant)
            conversation = await get_conversation(conv_id)
            if conversation and len(conversation.get("messages", [])) == 2:
                # This is the first response - generate title in background
                user_msg = conversation["messages"][0].get("content", "")
                assistant_msg = assistant_accum

                # Fire and forget - don't block the stream response
                asyncio.create_task(
                    _generate_and_update_title(conv_id, user_msg, assistant_msg)
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
