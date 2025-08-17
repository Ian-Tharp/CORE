from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import json
import time
from pydantic import BaseModel
from typing import List, Optional
import os

from app.models.user_input import UserInput

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


def _llm_or_stub(system_prompt: str, user_input: str, model_override: Optional[str] = None) -> str:
    """Call an LLM if configured; otherwise return a stubbed response.

    This keeps the playground usable without credentials while enabling
    real model calls in properly configured environments.
    """
    if ChatOpenAI and os.getenv("OPENAI_API_KEY"):
        chosen_model = (model_override or os.getenv("OPENAI_MODEL", "gpt-4o-mini")).strip()
        prompts = [("system", system_prompt), ("user", user_input)]

        def _supports_temperature(model_name: str) -> bool:
            # Extensible: add models known to reject custom temperatures
            no_temp = {"gpt-5"}
            return model_name not in no_temp

        try:
            kwargs = {"model": chosen_model}
            if _supports_temperature(chosen_model):
                kwargs["temperature"] = 0.2  # default conservative
            llm = ChatOpenAI(**kwargs)
            msg = llm.invoke(prompts)
            return msg.content  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            # If temperature was the cause and the model was misclassified, fallback once
            msg_text = str(exc).lower()
            if "temperature" in msg_text and ("unsupported" in msg_text or "does not support" in msg_text):
                try:
                    llm = ChatOpenAI(model=chosen_model)
                    msg = llm.invoke(prompts)
                    return msg.content  # type: ignore[attr-defined]
                except Exception as exc2:  # noqa: BLE001
                    return f"[LLM error: {exc2}]\nSystem: {system_prompt}\nUser: {user_input}"
            return f"[LLM error: {exc}]\nSystem: {system_prompt}\nUser: {user_input}"
    # Fallback stub
    return f"[stubbed] System: {system_prompt}\nUser: {user_input}"


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@router.post("")
async def core_entry(user_input: UserInput, request: Request):
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
    return {"message": "CORE entry acknowledged. Use /core/comprehension → /core/orchestration → /core/reasoning → /core/evaluation for step-by-step playground."}


@router.post("/comprehension")
async def comprehension(user_input: UserInput, request: Request) -> StepResponse:
    system = (
        "Classify the input as command/query/conversation; identify capabilities and whether tools are needed. "
        "Return a short explanation."
    )
    text = _llm_or_stub(system, user_input.user_input, user_input.model)
    # naive routing decision: if 'plan' or 'steps' present → orchestration; else conversation
    route = "orchestration" if any(k in text.lower() for k in ["plan", "steps", "capability"]) else "conversation"
    return StepResponse(step="Comprehension", text=text, routing_decision=route)


@router.post("/comprehension/stream")
async def comprehension_stream(user_input: UserInput, request: Request) -> StreamingResponse:
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
        yield _sse({"type": "metrics", "duration_ms": duration_ms, "ttfb_ms": ttfb_ms, "tokens": tokens})
        yield _sse({"type": "end"})

    return StreamingResponse(gen(), media_type="text/event-stream")

@router.post("/orchestration")
async def orchestration(user_input: UserInput, request: Request) -> StepResponse:
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
async def orchestration_stream(user_input: UserInput, request: Request) -> StreamingResponse:
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
        yield _sse({"type": "metrics", "duration_ms": duration_ms, "ttfb_ms": ttfb_ms, "tokens": tokens})
        yield _sse({"type": "end"})

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/reasoning")
async def reasoning(user_input: UserInput, request: Request) -> StepResponse:
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
async def reasoning_stream(user_input: UserInput, request: Request) -> StreamingResponse:
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
        yield _sse({"type": "metrics", "duration_ms": duration_ms, "ttfb_ms": ttfb_ms, "tokens": tokens})
        yield _sse({"type": "end"})

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/evaluation")
async def evaluation(user_input: UserInput, request: Request) -> StepResponse:
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
        "Answer SATISFACTORY or UNSATISFACTORY and explain briefly; propose a revision if needed." + ctx
    )
    text = _llm_or_stub(system, user_input.user_input, user_input.model)
    verdict = "SATISFACTORY" if "satisf" in text.lower() else "UNSATISFACTORY"
    return StepResponse(step="Evaluation", text=text, evaluation=verdict)


@router.post("/evaluation/stream")
async def evaluation_stream(user_input: UserInput, request: Request) -> StreamingResponse:
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
        "Answer SATISFACTORY or UNSATISFACTORY and explain briefly; propose a revision if needed." + ctx
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
        yield _sse({"type": "metrics", "duration_ms": duration_ms, "ttfb_ms": ttfb_ms, "tokens": tokens})
        yield _sse({"type": "end"})

    return StreamingResponse(gen(), media_type="text/event-stream")
