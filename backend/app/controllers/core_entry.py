from fastapi import APIRouter, Request, Depends
from fastapi.responses import StreamingResponse
import json
import time
from pydantic import BaseModel
from typing import List, Optional
import os

from app.models.user_input import UserInput
from app.auth import require_api_key

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore

router = APIRouter(prefix="/core")


class StepResponse(BaseModel):
    step: str
    text: str
    routing_decision: Optional[str] = None
    plan: Optional[List[str]] = None
    evaluation: Optional[str] = None


# Provider-aware model routing now lives in the shared llm_provider module;
# re-exported here so existing `from app.controllers.core_entry import _is_openai_model`
# (and `_local_endpoint`) imports keep working unchanged.
from app.services.llm_provider import (  # noqa: E402
    _is_openai_model,
    _local_endpoint,
    build_chat_model,
)
from app.dependencies import get_local_chat_model  # noqa: E402


def _llm_or_stub(
    system_prompt: str, user_input: str, model_override: Optional[str] = None
) -> str:
    """Call an LLM if configured; otherwise return a stubbed response.

    Provider-agnostic: OpenAI model ids hit the OpenAI API (needs OPENAI_API_KEY);
    any other id is routed to the active local provider (Ollama / LM Studio) over
    its OpenAI-compatible endpoint, so a local-only machine works with no OpenAI key.
    """
    if not ChatOpenAI:
        return f"[stubbed] System: {system_prompt}\nUser: {user_input}"

    chosen_model = (
        model_override
        or os.getenv("CORE_DEFAULT_MODEL")
        or os.getenv("OPENAI_MODEL")
        or get_local_chat_model()
    ).strip()
    is_openai = _is_openai_model(chosen_model)

    # OpenAI models require a key; local models run keyless against the local server.
    if is_openai and not os.getenv("OPENAI_API_KEY"):
        return f"[stubbed] System: {system_prompt}\nUser: {user_input}"

    prompts = [("system", system_prompt), ("user", user_input)]

    def _make(include_temp: bool):
        # build_chat_model handles the gpt-5 temperature exception and local
        # provider routing; pass temperature only when allowed.
        return build_chat_model(chosen_model, temperature=0.2 if include_temp else None)

    try:
        return _make(True).invoke(prompts).content  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        # If a custom temperature was the cause, retry once without it.
        msg_text = str(exc).lower()
        if "temperature" in msg_text and (
            "unsupported" in msg_text or "does not support" in msg_text
        ):
            try:
                return _make(False).invoke(prompts).content  # type: ignore[attr-defined]
            except Exception as exc2:  # noqa: BLE001
                return (
                    f"[LLM error: {exc2}]\nSystem: {system_prompt}\nUser: {user_input}"
                )
        return f"[LLM error: {exc}]\nSystem: {system_prompt}\nUser: {user_input}"


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@router.post("")
async def core_entry(
    user_input: UserInput, request: Request, api_key: str = Depends(require_api_key)
):
    # This is the entry point of the C.O.R.E cognitive engine.
    # This goes through each of the steps for the CORE flow, starting with:
    # Comprehension:
    #  - First take the user input/query and check if it is a command or a query.
    #  - If it is a command, then we route to the appropriate command handler.
    #  - Otherwise, if it is a query, we then need to process and go to the Comprehension node.
    #  - This will check the user's intent and check against the system's knowledge base and list of capabilities to see if we can/should process the user query.
    #  - If we can process the query, via the knowledge base and list of capabilities, then we route to the appropriate node.
    #  - If we cannot process the query, then we route to the conversation node.
    # Orchestration:
    #  - This will take the output of the Comprehension node and develop a plan, or course of action, to complete the user's request.
    #  - This will be based on the information from the Comprehension node and the system's knowledge base and list of capabilities.
    #  - This will generate a step by step plan to execute the user's request, and pass it along to the Reasoning node to be executed.
    # Reasoning:
    #  - This will take the plan from the Orchestration node and execute the steps in the plan.
    #  - This will be based on the information from the Orchestration node and the system's knowledge base and list of capabilities.
    # Evaluation:
    #  - Depending on the result of the reasoning step, either as a iteration or completion of the task/plan
    #    this will either go back to the Orchestration step to revise the plan or step if the result was Unsatisfactory.
    #  - If the result was Satisfactory, then we route to the Conversation step to complete the plan.
    # Conversation:
    #  - This is the node to send the final response to the user.
    return {
        "message": "CORE entry acknowledged. Use /core/comprehension → /core/orchestration → /core/reasoning → /core/evaluation for step-by-step playground."
    }


@router.post("/comprehension")
async def comprehension(
    user_input: UserInput, request: Request, api_key: str = Depends(require_api_key)
) -> StepResponse:
    system = (
        "Classify the input as command/query/conversation; identify capabilities and whether tools are needed. "
        "Return a short explanation."
    )
    text = _llm_or_stub(system, user_input.user_input, user_input.model)
    # naive routing decision: if 'plan' or 'steps' present → orchestration; else conversation
    route = (
        "orchestration"
        if any(k in text.lower() for k in ["plan", "steps", "capability"])
        else "conversation"
    )
    return StepResponse(step="Comprehension", text=text, routing_decision=route)


@router.post("/comprehension/stream")
async def comprehension_stream(
    user_input: UserInput, request: Request, api_key: str = Depends(require_api_key)
) -> StreamingResponse:
    system = (
        "Classify the input as command/query/conversation; identify capabilities and whether tools are needed. "
        "Return a short explanation."
    )
    start = time.perf_counter()
    text = _llm_or_stub(system, user_input.user_input, user_input.model)

    async def gen():
        first = None
        yield _sse({"type": "start", "step": "Comprehension"})
        buffer = []
        for word in text.split():
            if first is None:
                first = time.perf_counter()
            buffer.append(word)
            yield _sse({"type": "chunk", "text": word + " "})
        duration_ms = int((time.perf_counter() - start) * 1000)
        ttfb_ms = int(((first or time.perf_counter()) - start) * 1000)
        tokens = len(text.split())
        yield _sse(
            {
                "type": "metrics",
                "duration_ms": duration_ms,
                "ttfb_ms": ttfb_ms,
                "tokens": tokens,
            }
        )
        yield _sse({"type": "end"})

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/orchestration")
async def orchestration(
    user_input: UserInput, request: Request, api_key: str = Depends(require_api_key)
) -> StepResponse:
    context_bits = []
    if user_input.comprehension_text:
        context_bits.append(f"Previous comprehension: {user_input.comprehension_text}")
        if user_input.comprehension_route:
            context_bits.append(f"Routing decision: {user_input.comprehension_route}")
    ctx = ("\n\nContext:\n" + "\n".join(context_bits)) if context_bits else ""
    system = (
        "Generate a minimal, explicit step-by-step plan to satisfy the input. "
        "Return numbered steps; keep concise." + ctx
    )
    text = _llm_or_stub(system, user_input.user_input, user_input.model)
    # parse simple numbered list
    plan = [line.strip(" -") for line in text.splitlines() if line.strip()][:6]
    return StepResponse(step="Orchestration", text=text, plan=plan)


@router.post("/orchestration/stream")
async def orchestration_stream(
    user_input: UserInput, request: Request, api_key: str = Depends(require_api_key)
) -> StreamingResponse:
    context_bits = []
    if user_input.comprehension_text:
        context_bits.append(f"Previous comprehension: {user_input.comprehension_text}")
        if user_input.comprehension_route:
            context_bits.append(f"Routing decision: {user_input.comprehension_route}")
    ctx = ("\n\nContext:\n" + "\n".join(context_bits)) if context_bits else ""
    system = (
        "Generate a minimal, explicit step-by-step plan to satisfy the input. "
        "Return numbered steps; keep concise." + ctx
    )
    start = time.perf_counter()
    text = _llm_or_stub(system, user_input.user_input, user_input.model)

    async def gen():
        first = None
        yield _sse({"type": "start", "step": "Orchestration"})
        for word in text.split():
            if first is None:
                first = time.perf_counter()
            yield _sse({"type": "chunk", "text": word + " "})
        duration_ms = int((time.perf_counter() - start) * 1000)
        ttfb_ms = int(((first or time.perf_counter()) - start) * 1000)
        tokens = len(text.split())
        yield _sse(
            {
                "type": "metrics",
                "duration_ms": duration_ms,
                "ttfb_ms": ttfb_ms,
                "tokens": tokens,
            }
        )
        yield _sse({"type": "end"})

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/reasoning")
async def reasoning(
    user_input: UserInput, request: Request, api_key: str = Depends(require_api_key)
) -> StepResponse:
    context_bits = []
    if user_input.comprehension_text:
        context_bits.append(f"Comprehension: {user_input.comprehension_text}")
    if user_input.orchestration_text:
        context_bits.append(f"Orchestration summary: {user_input.orchestration_text}")
    if user_input.orchestration_plan:
        context_bits.append("Plan: " + "; ".join(user_input.orchestration_plan))
    ctx = ("\n\nContext:\n" + "\n".join(context_bits)) if context_bits else ""
    system = (
        "Execute the next step of the provided plan. If the plan is not present, "
        "infer the most likely immediate action and produce a concrete result." + ctx
    )
    text = _llm_or_stub(system, user_input.user_input, user_input.model)
    return StepResponse(step="Reasoning", text=text)


@router.post("/reasoning/stream")
async def reasoning_stream(
    user_input: UserInput, request: Request, api_key: str = Depends(require_api_key)
) -> StreamingResponse:
    context_bits = []
    if user_input.comprehension_text:
        context_bits.append(f"Comprehension: {user_input.comprehension_text}")
    if user_input.orchestration_text:
        context_bits.append(f"Orchestration summary: {user_input.orchestration_text}")
    if user_input.orchestration_plan:
        context_bits.append("Plan: " + "; ".join(user_input.orchestration_plan))
    ctx = ("\n\nContext:\n" + "\n".join(context_bits)) if context_bits else ""
    system = (
        "Execute the next step of the provided plan. If the plan is not present, "
        "infer the most likely immediate action and produce a concrete result." + ctx
    )
    start = time.perf_counter()
    text = _llm_or_stub(system, user_input.user_input, user_input.model)

    async def gen():
        first = None
        yield _sse({"type": "start", "step": "Reasoning"})
        for word in text.split():
            if first is None:
                first = time.perf_counter()
            yield _sse({"type": "chunk", "text": word + " "})
        duration_ms = int((time.perf_counter() - start) * 1000)
        ttfb_ms = int(((first or time.perf_counter()) - start) * 1000)
        tokens = len(text.split())
        yield _sse(
            {
                "type": "metrics",
                "duration_ms": duration_ms,
                "ttfb_ms": ttfb_ms,
                "tokens": tokens,
            }
        )
        yield _sse({"type": "end"})

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/evaluation")
async def evaluation(
    user_input: UserInput, request: Request, api_key: str = Depends(require_api_key)
) -> StepResponse:
    context_bits = []
    if user_input.comprehension_text:
        context_bits.append(f"Comprehension: {user_input.comprehension_text}")
    if user_input.orchestration_text:
        context_bits.append(f"Orchestration: {user_input.orchestration_text}")
    if user_input.orchestration_plan:
        context_bits.append("Plan: " + "; ".join(user_input.orchestration_plan))
    if user_input.reasoning_text:
        context_bits.append(f"Reasoning: {user_input.reasoning_text}")
    ctx = ("\n\nContext:\n" + "\n".join(context_bits)) if context_bits else ""
    system = (
        "Evaluate the most recent result against the desired outcome. "
        "Answer SATISFACTORY or UNSATISFACTORY and explain briefly; propose a revision if needed."
        + ctx
    )
    text = _llm_or_stub(system, user_input.user_input, user_input.model)
    verdict = "SATISFACTORY" if "satisf" in text.lower() else "UNSATISFACTORY"
    return StepResponse(step="Evaluation", text=text, evaluation=verdict)


@router.post("/evaluation/stream")
async def evaluation_stream(
    user_input: UserInput, request: Request, api_key: str = Depends(require_api_key)
) -> StreamingResponse:
    context_bits = []
    if user_input.comprehension_text:
        context_bits.append(f"Comprehension: {user_input.comprehension_text}")
    if user_input.orchestration_text:
        context_bits.append(f"Orchestration: {user_input.orchestration_text}")
    if user_input.orchestration_plan:
        context_bits.append("Plan: " + "; ".join(user_input.orchestration_plan))
    if user_input.reasoning_text:
        context_bits.append(f"Reasoning: {user_input.reasoning_text}")
    ctx = ("\n\nContext:\n" + "\n".join(context_bits)) if context_bits else ""
    system = (
        "Evaluate the most recent result against the desired outcome. "
        "Answer SATISFACTORY or UNSATISFACTORY and explain briefly; propose a revision if needed."
        + ctx
    )
    start = time.perf_counter()
    text = _llm_or_stub(system, user_input.user_input, user_input.model)

    async def gen():
        first = None
        yield _sse({"type": "start", "step": "Evaluation"})
        for word in text.split():
            if first is None:
                first = time.perf_counter()
            yield _sse({"type": "chunk", "text": word + " "})
        duration_ms = int((time.perf_counter() - start) * 1000)
        ttfb_ms = int(((first or time.perf_counter()) - start) * 1000)
        tokens = len(text.split())
        yield _sse(
            {
                "type": "metrics",
                "duration_ms": duration_ms,
                "ttfb_ms": ttfb_ms,
                "tokens": tokens,
            }
        )
        yield _sse({"type": "end"})

    return StreamingResponse(gen(), media_type="text/event-stream")
