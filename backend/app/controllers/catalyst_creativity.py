from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
import asyncio
import json
from app.dependencies import _get_openai_client
from app.models.user_input import UserInput
from typing import List

router = APIRouter(prefix="/catalyst-creativity", tags=["catalyst-creativity"])

@router.post("")
async def catalyst_creativity():
    pass

@router.post("/divergence", response_class=StreamingResponse)
async def catalyst_creativity_divergence(
    user_input: UserInput,
    divergence_number: int = Query(..., ge=1, le=10, description="Number of divergent ideas to generate, max 10"),
):
    """Stream divergent ideas token-by-token using SSE."""
    client = _get_openai_client()
    queue: asyncio.Queue[None | str] = asyncio.Queue()

    async def stream_idea(idx: int):
        messages = [
            {"role": "system", "content": "You are a creativity assistant that generates divergent ideas."},
            {"role": "user", "content": f"Given the prompt: \"{user_input.user_input}\", generate a novel divergent idea #{idx+1}."}
        ]
        try:
            response = await client.responses.create(
                model="gpt-4.5-preview",
                input=messages,
                stream=True,
            )
            async for chunk in response:
                data = chunk.model_dump(exclude_none=True)
                delta = data.get("delta", "")
                event = {"index": idx, "delta": delta}
                sse = f"id: {idx}\nevent: divergence\ndata: {json.dumps(event)}\n\n"
                await queue.put(sse)
        except Exception as exc:
            error_event = {"index": idx, "error": str(exc)}
            await queue.put(f"event: error\ndata: {json.dumps(error_event)}\n\n")
        finally:
            # signal completion for this idea stream
            await queue.put(None)

    # launch all idea streams in parallel
    for i in range(divergence_number):
        asyncio.create_task(stream_idea(i))

    async def event_stream():
        completed = 0
        while completed < divergence_number:
            item = await queue.get()
            if item is None:
                completed += 1
            else:
                yield item

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.post("/convergence")
async def catalyst_creativity_convergence(user_input: UserInput, divergence_output: List[str], convergence_number: int):
    # This is the convergence step of the catalyst creativity process.
    # This will take the user input and generate a list of very convergent ideas.
    # The convergence input number is the number of different ideas to converge on.
    return []

@router.post("/synthesis")
async def catalyst_creativity_synthesis(user_input: UserInput, divergence_output: List[str], convergence_output: List[str]):
    # This is the synthesis step of the catalyst creativity process.
    # This will take the full divergence and convergence outputs and synthesize them into a single idea.
    return ""

