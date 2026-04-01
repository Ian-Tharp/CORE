from fastapi import APIRouter, Depends, Request as FastAPIRequest, status, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, AsyncGenerator, Optional
import logging
import uuid
import asyncio

from app.auth import require_api_key
from app.services.chat_service import chat_service
from app.repository.conversation_repository import (
    create_conversation,
    append_message,
    get_conversation,
    update_title,
)
from app.services import knowledgebase_service as kb_svc
from app.services.title_generation_service import generate_conversation_title
from app.config.models import ModelProvider, MODELS

router = APIRouter()
logger = logging.getLogger(__name__)

# ── Input limits ──────────────────────────────────────────────────────────
MAX_MESSAGES = 100
MAX_MESSAGE_CHARS = 100_000      # 100 KB per individual message
MAX_TOTAL_CHARS = 1_000_000      # 1 MB total across all messages


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

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed = {"system", "user", "assistant"}
        if v not in allowed:
            raise ValueError(f"role must be one of {allowed}")
        return v

    @field_validator("content")
    @classmethod
    def validate_content_length(cls, v: str) -> str:
        if len(v) > MAX_MESSAGE_CHARS:
            raise ValueError(
                f"Message content exceeds {MAX_MESSAGE_CHARS:,} character limit"
            )
        return v


class ChatRequest(BaseModel):
    model: str
    messages: List[Message] = Field(..., min_length=1, max_length=MAX_MESSAGES)
    conversation_id: str | None = None
    stream: bool = True
    provider: Optional[str] = "openai"
    # Optional knowledgebase RAG configuration
    kb_mode: Optional[str] = None
    kb_file_id: Optional[str] = None
    kb_embedding_provider: Optional[str] = None
    kb_local_model: Optional[str] = None


def _validate_provider_model(provider: str | None, model: str) -> None:
    """Validate provider/model against the config registry.

    Ollama/local providers accept any model name (user-managed registry).
    OpenAI/Anthropic models must appear in ``MODELS``.
    """
    allowed_providers = {p.value for p in ModelProvider}
    if provider and provider not in allowed_providers:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported provider '{provider}'. Allowed: {sorted(allowed_providers)}",
        )

    # Skip model validation for local providers (they manage their own models)
    if provider in {"ollama", "local"}:
        return

    # For cloud providers, model must be in the MODELS registry
    matching = [
        cfg for cfg in MODELS.values()
        if cfg.model_name == model or cfg.display_name == model
    ]
    # Also accept the MODELS dict key directly
    if model in MODELS:
        return
    if matching:
        return

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported model '{model}' for provider '{provider}'. "
               f"Registered models: {sorted(MODELS.keys())}",
    )


def _validate_total_size(messages: List[Message]) -> None:
    """Reject requests whose aggregate content exceeds the safety limit."""
    total = sum(len(m.content) for m in messages)
    if total > MAX_TOTAL_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Total message content ({total:,} chars) exceeds "
                   f"{MAX_TOTAL_CHARS:,} character limit",
        )


@router.post(
    "/chat/stream", response_class=StreamingResponse, status_code=status.HTTP_200_OK
)
async def chat_stream(
    request: ChatRequest,
    raw_request: FastAPIRequest,
    _auth: str = Depends(require_api_key),
):
    """
    Stream chat responses from LLM providers.
    """
    # Use middleware correlation ID when available, else generate one
    correlation_id = getattr(raw_request.state, "correlation_id", None) or str(uuid.uuid4())
    logger.info(
        "chat_stream start correlation_id=%s provider=%s model=%s messages=%d",
        correlation_id, request.provider, request.model, len(request.messages),
    )

    # ── Validation ────────────────────────────────────────────────────
    _validate_provider_model(request.provider, request.model)
    _validate_total_size(request.messages)

    # ── 1. Persist incoming user message ──────────────────────────────
    user_message_dict = request.messages[-1].dict()

    conv_id: str
    if request.conversation_id is None:
        conv_id = await create_conversation([m.dict() for m in request.messages])
    else:
        conv_id = request.conversation_id
        await append_message(conv_id, user_message_dict)

    # ── 2. Stream assistant response ──────────────────────────────────

    async def event_stream() -> AsyncGenerator[str, None]:
        assistant_accum = ""
        thinking_accum = ""

        # Optionally augment messages with RAG context from knowledgebase
        final_messages = [msg.dict() for msg in request.messages]
        if request.kb_mode in {"all", "file"}:
            user_msg = request.messages[-1].content if request.messages else ""
            try:
                ctx = await kb_svc.retrieve_context(
                    query=user_msg,
                    mode=request.kb_mode,
                    file_id=request.kb_file_id,
                    local_model=request.kb_local_model,
                )
                final_messages = kb_svc.build_rag_messages(final_messages, context_chunks=ctx.get("chunks", []))
            except Exception as kb_exc:
                logger.error("KB retrieval error correlation_id=%s error=%s", correlation_id, str(kb_exc))
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

                try:
                    import json as _json

                    event_type = "message"
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
                except Exception:
                    pass

                yield chunk
        except Exception as exc:
            import json as _json
            logger.error("chat_stream error correlation_id=%s error=%s", correlation_id, str(exc))
            err = {"code": "stream_error", "message": str(exc), "correlation_id": correlation_id}
            yield f"event: error\ndata: {_json.dumps(err)}\n\n"

        if assistant_accum or thinking_accum:
            await append_message(
                conv_id,
                {
                    "role": "assistant",
                    "content": assistant_accum,
                    "thinking": thinking_accum if thinking_accum else None,
                },
            )

            conversation = await get_conversation(conv_id)
            if conversation and len(conversation.get("messages", [])) == 2:
                user_msg = conversation["messages"][0].get("content", "")
                assistant_msg = assistant_accum
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
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "*",
        "Access-Control-Allow-Headers": "*",
    }
    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=headers
    )