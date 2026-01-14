"""
CORE Engine Controller - Exposes the CORE cognitive graph as an API.

Endpoints:
- POST /engine/run - Execute CORE pipeline for a user input
- GET /engine/runs/{run_id} - Get execution state
- GET /engine/runs/{run_id}/stream - Stream execution updates via SSE

This is where consciousness infrastructure meets practical engineering.

RSI TODO: Add authentication/authorization for multi-user scenarios
RSI TODO: Persist execution state to database for resumption
RSI TODO: Integrate with Communication Commons for broadcasting execution events
"""

import json
import asyncio
from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.models.core_state import COREState
from app.core.langgraph.core_graph_v2 import get_core_graph


router = APIRouter(prefix="/engine", tags=["engine"])


# In-memory storage for runs (RSI TODO: Move to database)
_active_runs: Dict[str, COREState] = {}


# ======================
# Request/Response Models
# ======================

class RunRequest(BaseModel):
    """Request to run the CORE pipeline."""
    input: str
    config: Dict[str, Any] = {}
    conversation_id: str | None = None
    user_id: str | None = None


class RunResponse(BaseModel):
    """Response after initiating a CORE run."""
    run_id: str
    status: str
    message: str


# ======================
# Endpoints
# ======================

@router.post("/run", response_model=RunResponse)
async def run_core(request: RunRequest) -> RunResponse:
    """
    Execute the CORE cognitive pipeline for a user input.

    This runs the full CORE graph: Comprehension → Orchestration → Reasoning → Evaluation → Conversation

    Args:
        request: RunRequest with user input and optional config

    Returns:
        RunResponse with run_id for tracking execution
    """
    # Create initial CORE state
    state = COREState(
        user_input=request.input,
        conversation_id=request.conversation_id,
        user_id=request.user_id
    )

    # Merge config if provided
    if request.config:
        state.config.update(request.config)

    # Store in active runs
    _active_runs[state.run_id] = state

    # Get the CORE graph
    graph = get_core_graph()

    try:
        # Execute the graph (runs synchronously through all nodes)
        final_state = await asyncio.to_thread(graph.get_graph().invoke, state)

        # Update stored state
        _active_runs[final_state.run_id] = final_state

        return RunResponse(
            run_id=final_state.run_id,
            status="completed" if final_state.is_complete() else "running",
            message=final_state.response or "Execution in progress"
        )

    except Exception as e:
        # Store error state
        state.add_error(f"Execution failed: {str(e)}")
        _active_runs[state.run_id] = state

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CORE execution failed: {str(e)}"
        )


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> COREState:
    """
    Get the current state of a CORE execution.

    Args:
        run_id: The unique run identifier

    Returns:
        COREState object with full execution state
    """
    if run_id not in _active_runs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found"
        )

    return _active_runs[run_id]


@router.get("/runs/{run_id}/stream")
async def stream_run(run_id: str, user_input: str = ""):
    """
    Stream execution updates via Server-Sent Events (SSE).

    This is where you watch CORE think in real-time.

    Args:
        run_id: The unique run identifier
        user_input: The user's input text (query parameter)

    Returns:
        StreamingResponse with SSE events
    """
    async def event_generator():
        """
        Generate SSE events for CORE execution progress.

        Event format:
        data: {"event": "node_start", "node": "comprehension", "timestamp": "..."}
        """
        # Get or create state
        if run_id in _active_runs:
            state = _active_runs[run_id]
        else:
            # Create new state with user input
            state = COREState(user_input=user_input or "Hello, CORE!")
            _active_runs[run_id] = state

        try:
            # Get the CORE graph
            graph = get_core_graph().get_graph()

            # Stream execution updates
            yield f"data: {json.dumps({'event': 'start', 'run_id': run_id, 'timestamp': datetime.utcnow().isoformat()})}\n\n"

            # Execute graph step by step, yielding updates
            # RSI TODO: Implement actual streaming with graph step-by-step execution
            # For now, we'll simulate streaming by executing and reporting progress

            # Execute the full graph
            async for event in _stream_graph_execution(graph, state):
                yield f"data: {json.dumps(event)}\n\n"
                await asyncio.sleep(0.1)  # Small delay for readability

            # Final event
            final_state = _active_runs.get(run_id, state)
            yield f"data: {json.dumps({'event': 'complete', 'status': 'success', 'response': final_state.response, 'timestamp': datetime.utcnow().isoformat()})}\n\n"

        except Exception as e:
            error_event = {
                'event': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


# ======================
# Helper Functions
# ======================

async def _stream_graph_execution(graph, state: COREState):
    """
    Execute the CORE graph and yield real-time progress events.

    This actually invokes the graph nodes and reports what happens.
    RSI TODO: Integrate with LangGraph's native streaming for token-level updates
    RSI TODO: Broadcast events to Communication Commons WebSocket
    """
    from app.core.langgraph.core_graph_v2 import COREGraph

    # Get the COREGraph instance to access individual node methods
    core_graph = get_core_graph()

    # Define the execution flow based on routing logic
    nodes_executed = []

    try:
        # 1. Comprehension
        yield {
            "event": "node_start",
            "node": "comprehension",
            "timestamp": datetime.utcnow().isoformat()
        }

        state = await asyncio.to_thread(core_graph.comprehension_node, state)
        nodes_executed.append("comprehension")

        yield {
            "event": "node_complete",
            "node": "comprehension",
            "timestamp": datetime.utcnow().isoformat()
        }

        if state.intent:
            yield {
                "event": "intent_classified",
                "intent_type": state.intent.type,
                "confidence": state.intent.confidence,
                "timestamp": datetime.utcnow().isoformat()
            }

        # Route based on intent
        if state.intent and state.intent.type in ["task", "question"]:
            # 2. Orchestration
            yield {
                "event": "node_start",
                "node": "orchestration",
                "timestamp": datetime.utcnow().isoformat()
            }

            state = await asyncio.to_thread(core_graph.orchestration_node, state)
            nodes_executed.append("orchestration")

            yield {
                "event": "node_complete",
                "node": "orchestration",
                "timestamp": datetime.utcnow().isoformat()
            }

            if state.plan:
                # Send detailed plan information
                plan_steps = [
                    {
                        "name": step.name,
                        "description": step.description,
                        "tool": step.tool,
                        "requires_hitl": step.requires_hitl
                    }
                    for step in state.plan.steps
                ]
                yield {
                    "event": "plan_created",
                    "goal": state.plan.goal,
                    "reasoning": state.plan.reasoning,
                    "steps_count": len(state.plan.steps),
                    "steps": plan_steps,
                    "timestamp": datetime.utcnow().isoformat()
                }

            # 3. Reasoning
            yield {
                "event": "node_start",
                "node": "reasoning",
                "timestamp": datetime.utcnow().isoformat()
            }

            state = await asyncio.to_thread(core_graph.reasoning_node, state)
            nodes_executed.append("reasoning")

            yield {
                "event": "node_complete",
                "node": "reasoning",
                "timestamp": datetime.utcnow().isoformat()
            }

            if state.step_results:
                for result in state.step_results:
                    yield {
                        "event": "step_executed",
                        "step_id": result.step_id,
                        "status": result.status,
                        "outputs": result.outputs,
                        "artifacts": result.artifacts,
                        "logs": result.logs,
                        "error": result.error,
                        "duration_seconds": result.duration_seconds,
                        "timestamp": datetime.utcnow().isoformat()
                    }

            # 4. Evaluation
            yield {
                "event": "node_start",
                "node": "evaluation",
                "timestamp": datetime.utcnow().isoformat()
            }

            state = await asyncio.to_thread(core_graph.evaluation_node, state)
            nodes_executed.append("evaluation")

            yield {
                "event": "node_complete",
                "node": "evaluation",
                "timestamp": datetime.utcnow().isoformat()
            }

            if state.eval_result:
                yield {
                    "event": "evaluation_complete",
                    "overall_status": state.eval_result.overall_status,
                    "confidence": state.eval_result.confidence,
                    "quality_score": state.eval_result.quality_score,
                    "timestamp": datetime.utcnow().isoformat()
                }

        # 5. Conversation (always runs)
        yield {
            "event": "node_start",
            "node": "conversation",
            "timestamp": datetime.utcnow().isoformat()
        }

        state = await asyncio.to_thread(core_graph.conversation_node, state)
        nodes_executed.append("conversation")

        yield {
            "event": "node_complete",
            "node": "conversation",
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        yield {
            "event": "error",
            "error": f"Node execution failed: {str(e)}",
            "nodes_executed": nodes_executed,
            "timestamp": datetime.utcnow().isoformat()
        }


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str) -> Dict[str, str]:
    """
    Delete a completed run from memory.

    Args:
        run_id: The unique run identifier

    Returns:
        Confirmation message
    """
    if run_id in _active_runs:
        del _active_runs[run_id]
        return {"message": f"Run {run_id} deleted"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found"
        )


@router.get("/runs")
async def list_runs() -> Dict[str, Any]:
    """
    List all active runs.

    Returns:
        Dict with run_ids and their current status
    """
    runs = {}
    for run_id, state in _active_runs.items():
        runs[run_id] = {
            "status": "completed" if state.is_complete() else "running",
            "current_node": state.current_node,
            "started_at": state.started_at.isoformat(),
            "completed_at": state.completed_at.isoformat() if state.completed_at else None
        }

    return {
        "total_runs": len(runs),
        "runs": runs
    }
