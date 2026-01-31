"""
Catalyst Creativity API Endpoints

Three-phase creative pipeline: Divergence → Convergence → Synthesis.

Endpoints:
    POST /catalyst-creativity/auto          - Full pipeline (D→C→S) end-to-end
    POST /catalyst-creativity/divergence    - Phase 1: Generate divergent ideas (SSE stream)
    POST /catalyst-creativity/convergence   - Phase 2: Evaluate, group, rank ideas
    POST /catalyst-creativity/synthesis     - Phase 3: Synthesize unified output
    GET  /catalyst-creativity/sessions/{id} - Get session status and results
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.dependencies import _get_openai_client
from app.models.user_input import UserInput
from app.services.catalyst_service import get_catalyst_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/catalyst-creativity", tags=["catalyst-creativity"])


# =============================================================================
# REQUEST / RESPONSE MODELS
# =============================================================================

class AutoCatalystRequest(BaseModel):
    """Request for a full D→C→S pipeline run."""
    prompt: str = Field(..., description="Creative prompt to explore")
    count: int = Field(default=5, ge=1, le=10, description="Number of divergent ideas")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Optional model/config overrides")


class ConvergenceRequest(BaseModel):
    """Request for convergence phase."""
    session_id: str = Field(..., description="Session from divergence phase")
    divergent_ideas: List[Dict[str, Any]] = Field(..., description="Ideas from divergence phase")


class SynthesisRequest(BaseModel):
    """Request for synthesis phase."""
    session_id: str = Field(..., description="Session ID")
    divergent_ideas: List[Dict[str, Any]] = Field(..., description="Ideas from divergence")
    convergent_analysis: Dict[str, Any] = Field(..., description="Analysis from convergence")


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/auto")
async def auto_catalyst(request: AutoCatalystRequest) -> dict:
    """
    Run the full Catalyst Creativity pipeline end-to-end.

    1. Divergence — generates N creative ideas with novelty scoring
    2. Convergence — evaluates, groups, ranks by feasibility/impact/uniqueness
    3. Synthesis — merges top ideas into a unified actionable concept

    Returns the complete session with all phase outputs.

    Example:
        POST /catalyst-creativity/auto
        {
            "prompt": "How could MMCNC fractal structure improve agent collaboration?",
            "count": 5
        }
    """
    try:
        service = get_catalyst_service()
        result = await service.auto_catalyst(
            prompt=request.prompt,
            count=request.count,
            config=request.config,
        )
        return result
    except Exception as e:
        logger.error(f"Auto-catalyst failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/divergence", response_class=StreamingResponse)
async def catalyst_creativity_divergence(
    user_input: UserInput,
    divergence_number: int = Query(
        ..., ge=1, le=10, description="Number of divergent ideas to generate"
    ),
):
    """
    Stream divergent ideas token-by-token using SSE (legacy endpoint).

    For the new structured pipeline, use POST /catalyst-creativity/auto instead.
    """
    client = _get_openai_client()
    queue: asyncio.Queue[None | str] = asyncio.Queue()

    async def stream_idea(idx: int):
        messages = [
            {
                "role": "system",
                "content": "You are a creativity assistant that generates divergent ideas.",
            },
            {
                "role": "user",
                "content": (
                    f'Given the prompt: "{user_input.user_input}", '
                    f"generate a novel divergent idea #{idx + 1}."
                ),
            },
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
            await queue.put(None)

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
async def catalyst_creativity_convergence(request: ConvergenceRequest) -> dict:
    """
    Evaluate, group, and rank divergent ideas.

    Takes the output of the divergence phase and applies structured
    evaluation criteria (feasibility, impact, uniqueness, coherence, synergy).

    Example:
        POST /catalyst-creativity/convergence
        {
            "session_id": "abc-123",
            "divergent_ideas": [...]
        }
    """
    try:
        service = get_catalyst_service()
        result = await service.run_convergence(
            session_id=request.session_id,
            divergent_ideas=request.divergent_ideas,
        )
        return {"session_id": request.session_id, "convergence": result}
    except Exception as e:
        logger.error(f"Convergence failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/synthesis")
async def catalyst_creativity_synthesis(request: SynthesisRequest) -> dict:
    """
    Synthesize divergent ideas and convergent analysis into a unified output.

    Merges the strongest elements across all ideas into a single actionable
    concept that is greater than the sum of its parts.

    Example:
        POST /catalyst-creativity/synthesis
        {
            "session_id": "abc-123",
            "divergent_ideas": [...],
            "convergent_analysis": {...}
        }
    """
    try:
        service = get_catalyst_service()
        result = await service.run_synthesis(
            session_id=request.session_id,
            divergent_ideas=request.divergent_ideas,
            convergent_analysis=request.convergent_analysis,
        )
        return {"session_id": request.session_id, "synthesis": result}
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    """Get the current state and results of a catalyst session."""
    service = get_catalyst_service()
    session = service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return session
