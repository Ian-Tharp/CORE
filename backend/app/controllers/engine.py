"""
CORE Engine Controller - Exposes the CORE cognitive graph as an API.

Endpoints:
- POST /engine/run - Execute CORE pipeline for a user input
- POST /engine/run/stream - Execute with real-time SSE streaming
- GET /engine/runs/{run_id} - Get execution state
- GET /engine/runs/{run_id}/stream - Stream execution updates via SSE

This is where consciousness infrastructure meets practical engineering.

Features:
- Run persistence to PostgreSQL database
- Webhook callbacks on run completion/failure
- Request metrics tracking
- Human-readable streaming progress
"""

import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.models.core_state import COREState
from app.core.langgraph.core_graph_v2 import get_core_graph
from app.auth import optional_api_key
from app.repository import run_repository
from app.services.webhook_service import get_webhook_service, WebhookEvent
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/engine", tags=["engine"])


# In-memory storage for active runs (also persisted to DB)
_active_runs: Dict[str, COREState] = {}


# Human-readable status messages for UX
NODE_MESSAGES = {
    "comprehension": {
        "start": "🔍 Understanding your request...",
        "complete": "✓ Got it!"
    },
    "orchestration": {
        "start": "📋 Planning approach...",
        "complete": "✓ Plan ready"
    },
    "reasoning": {
        "start": "⚡ Executing...",
        "complete": "✓ Done"
    },
    "evaluation": {
        "start": "🔎 Checking quality...",
        "complete": "✓ Looks good"
    },
    "conversation": {
        "start": "💬 Formulating response...",
        "complete": "✓ Response ready"
    }
}


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

    # Store in active runs (in-memory)
    _active_runs[state.run_id] = state
    
    # Persist to database
    try:
        await run_repository.save_run(state.model_dump())
    except Exception as e:
        logger.warning(f"Failed to persist run to database: {e}")
    
    # Fire webhook for run started
    webhook_service = get_webhook_service()
    try:
        await webhook_service.fire(
            event=WebhookEvent.RUN_STARTED,
            payload={
                "user_input": state.user_input,
                "conversation_id": state.conversation_id,
                "user_id": state.user_id
            },
            run_id=state.run_id
        )
    except Exception as e:
        logger.warning(f"Failed to fire run.started webhook: {e}")

    # Get the CORE graph
    graph = get_core_graph()

    try:
        # Execute the graph (runs synchronously through all nodes)
        # LangGraph returns a dict-like AddableValuesDict, not a COREState
        result_dict = await asyncio.to_thread(graph.get_graph().invoke, state)

        # Convert back to COREState if needed
        if isinstance(result_dict, dict):
            final_state = COREState(**result_dict)
        else:
            final_state = result_dict

        # Update stored state using the original run_id
        _active_runs[state.run_id] = final_state
        
        # Persist final state to database
        try:
            await run_repository.save_run(final_state.model_dump())
        except Exception as e:
            logger.warning(f"Failed to persist final run state: {e}")
        
        # Fire webhook for run completed
        try:
            await webhook_service.fire(
                event=WebhookEvent.RUN_COMPLETED,
                payload={
                    "response": final_state.response,
                    "intent_type": final_state.intent.type if final_state.intent else None,
                    "execution_history": final_state.execution_history,
                    "errors": final_state.errors
                },
                run_id=state.run_id
            )
        except Exception as e:
            logger.warning(f"Failed to fire run.completed webhook: {e}")

        return RunResponse(
            run_id=state.run_id,
            status="completed" if final_state.is_complete() else "running",
            message=final_state.response or "Execution in progress"
        )

    except Exception as e:
        # Store error state
        state.add_error(f"Execution failed: {str(e)}")
        _active_runs[state.run_id] = state
        
        # Persist error state
        try:
            await run_repository.save_run(state.model_dump())
        except Exception as db_e:
            logger.warning(f"Failed to persist error state: {db_e}")
        
        # Fire webhook for run failed
        try:
            await webhook_service.fire(
                event=WebhookEvent.RUN_FAILED,
                payload={
                    "error": str(e),
                    "errors": state.errors
                },
                run_id=state.run_id
            )
        except Exception as webhook_e:
            logger.warning(f"Failed to fire run.failed webhook: {webhook_e}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CORE execution failed: {str(e)}"
        )


@router.post("/run/stream")
async def run_core_stream(request: RunRequest):
    """
    Execute CORE pipeline with real-time streaming progress.
    
    This is the UX-friendly endpoint that shows thinking progress.
    Returns SSE events with human-readable status messages.
    
    Events:
    - thinking: Human-readable progress message
    - intent: Intent classification result
    - plan: Plan created with goal and steps
    - step: Step execution result
    - result: Final response
    - error: Error occurred
    """
    async def stream_generator():
        # Create initial state
        state = COREState(
            user_input=request.input,
            conversation_id=request.conversation_id,
            user_id=request.user_id
        )
        
        if request.config:
            state.config.update(request.config)
        
        _active_runs[state.run_id] = state
        
        # Persist to database
        try:
            await run_repository.save_run(state.model_dump())
        except Exception as e:
            logger.warning(f"Failed to persist run: {e}")
        
        core_graph = get_core_graph()
        
        try:
            # Send start event
            yield _sse_event({
                "event": "start",
                "run_id": state.run_id,
                "message": "🚀 Starting CORE cognitive engine..."
            })
            
            # 1. Comprehension
            yield _sse_event({
                "event": "thinking",
                "node": "comprehension",
                "message": NODE_MESSAGES["comprehension"]["start"]
            })
            
            state = await asyncio.to_thread(core_graph.comprehension_node, state)
            _active_runs[state.run_id] = state
            
            if state.intent:
                intent_msg = f"Intent: {state.intent.type} ({state.intent.confidence:.0%} confident)"
                yield _sse_event({
                    "event": "intent",
                    "type": state.intent.type,
                    "description": state.intent.description,
                    "confidence": state.intent.confidence,
                    "message": intent_msg
                })
            
            # Route based on intent
            if state.intent and state.intent.type in ["task", "question"]:
                # 2. Orchestration
                yield _sse_event({
                    "event": "thinking",
                    "node": "orchestration",
                    "message": NODE_MESSAGES["orchestration"]["start"]
                })
                
                state = await asyncio.to_thread(core_graph.orchestration_node, state)
                _active_runs[state.run_id] = state
                
                if state.plan:
                    steps_preview = [s.name for s in state.plan.steps[:3]]
                    yield _sse_event({
                        "event": "plan",
                        "goal": state.plan.goal,
                        "steps_count": len(state.plan.steps),
                        "steps_preview": steps_preview,
                        "message": f"📋 Plan: {state.plan.goal} ({len(state.plan.steps)} steps)"
                    })
                
                # 3. Reasoning
                yield _sse_event({
                    "event": "thinking",
                    "node": "reasoning",
                    "message": NODE_MESSAGES["reasoning"]["start"]
                })
                
                state = await asyncio.to_thread(core_graph.reasoning_node, state)
                _active_runs[state.run_id] = state
                
                # Report step results
                for result in state.step_results:
                    status_emoji = "✓" if result.status == "success" else "✗"
                    step = state.get_step(result.step_id)
                    step_name = step.name if step else result.step_id
                    yield _sse_event({
                        "event": "step",
                        "step_id": result.step_id,
                        "name": step_name,
                        "status": result.status,
                        "duration": result.duration_seconds,
                        "message": f"{status_emoji} {step_name}"
                    })
                
                # 4. Evaluation
                yield _sse_event({
                    "event": "thinking",
                    "node": "evaluation",
                    "message": NODE_MESSAGES["evaluation"]["start"]
                })
                
                state = await asyncio.to_thread(core_graph.evaluation_node, state)
                _active_runs[state.run_id] = state
            
            # 5. Conversation (always runs)
            yield _sse_event({
                "event": "thinking",
                "node": "conversation",
                "message": NODE_MESSAGES["conversation"]["start"]
            })
            
            state = await asyncio.to_thread(core_graph.conversation_node, state)
            _active_runs[state.run_id] = state
            
            # Persist final state
            try:
                await run_repository.save_run(state.model_dump())
            except Exception as e:
                logger.warning(f"Failed to persist final state: {e}")
            
            # Final result
            yield _sse_event({
                "event": "result",
                "run_id": state.run_id,
                "response": state.response,
                "quality": state.eval_result.quality_score if state.eval_result else None,
                "message": "✅ Complete"
            })
            
        except Exception as e:
            logger.error(f"Streaming execution failed: {e}", exc_info=True)
            yield _sse_event({
                "event": "error",
                "error": str(e),
                "message": f"❌ Error: {str(e)}"
            })
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


def _sse_event(data: dict) -> str:
    """Format data as an SSE event."""
    data["timestamp"] = datetime.utcnow().isoformat()
    return f"data: {json.dumps(data)}\n\n"


# NOTE: /runs/stats must be defined BEFORE /runs/{run_id} to avoid route matching issues
@router.get("/runs/stats")
async def get_run_stats() -> Dict[str, Any]:
    """
    Get run statistics from the database.
    
    Returns aggregate stats like total runs, completion rate, etc.
    """
    try:
        stats = await run_repository.get_run_stats()
        stats["active_in_memory"] = len(_active_runs)
        return stats
    except Exception as e:
        logger.error(f"Failed to get run stats: {e}")
        return {
            "total_runs": len(_active_runs),
            "active_in_memory": len(_active_runs),
            "error": str(e)
        }


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> COREState:
    """
    Get the current state of a CORE execution.

    Args:
        run_id: The unique run identifier

    Returns:
        COREState object with full execution state
    """
    # Check in-memory cache first (for active runs)
    if run_id in _active_runs:
        return _active_runs[run_id]
    
    # Fall back to database for completed/historical runs
    try:
        db_run = await run_repository.get_run(run_id)
        if db_run:
            # Reconstruct COREState from database state
            state_data = db_run.get("state")
            if state_data:
                # state is already a dict from JSONB
                if isinstance(state_data, str):
                    import json
                    state_data = json.loads(state_data)
                return COREState(**state_data)
            else:
                # Fallback: build minimal state from fields
                return COREState(
                    run_id=db_run["run_id"],
                    user_input=db_run.get("user_input", ""),
                    conversation_id=db_run.get("conversation_id"),
                    user_id=db_run.get("user_id"),
                    response=db_run.get("response"),
                    current_node=db_run.get("current_node")
                )
    except Exception as e:
        logger.warning(f"Failed to fetch run {run_id} from database: {e}")

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Run {run_id} not found"
    )


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
            async for event in _stream_graph_execution(graph, state):
                yield f"data: {json.dumps(event)}\n\n"
                await asyncio.sleep(0.1)  # Small delay for readability

            # Final event
            final_state = _active_runs.get(run_id, state)
            logger.info(f"[CORE STREAM] Sending final complete event, response: {(final_state.response or 'NONE')[:200]}")
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
            "X-Accel-Buffering": "no",
        }
    )


# ======================
# Helper Functions
# ======================

async def _stream_graph_execution(graph, state: COREState):
    """
    Execute the CORE graph and yield real-time progress events.
    """
    core_graph = get_core_graph()
    nodes_executed = []

    try:
        logger.info(f"[CORE STREAM] Starting graph execution for run_id={state.run_id}")
        # 1. Comprehension
        yield {
            "event": "node_start",
            "node": "comprehension",
            "message": NODE_MESSAGES["comprehension"]["start"],
            "timestamp": datetime.utcnow().isoformat()
        }

        state = await asyncio.to_thread(core_graph.comprehension_node, state)
        nodes_executed.append("comprehension")

        yield {
            "event": "node_complete",
            "node": "comprehension",
            "message": NODE_MESSAGES["comprehension"]["complete"],
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
                "message": NODE_MESSAGES["orchestration"]["start"],
                "timestamp": datetime.utcnow().isoformat()
            }

            state = await asyncio.to_thread(core_graph.orchestration_node, state)
            nodes_executed.append("orchestration")

            yield {
                "event": "node_complete",
                "node": "orchestration",
                "message": NODE_MESSAGES["orchestration"]["complete"],
                "timestamp": datetime.utcnow().isoformat()
            }

            if state.plan:
                plan_steps = [
                    {"name": step.name, "description": step.description}
                    for step in state.plan.steps
                ]
                yield {
                    "event": "plan_created",
                    "goal": state.plan.goal,
                    "steps_count": len(state.plan.steps),
                    "steps": plan_steps,
                    "timestamp": datetime.utcnow().isoformat()
                }

            # 3. Reasoning
            yield {
                "event": "node_start",
                "node": "reasoning",
                "message": NODE_MESSAGES["reasoning"]["start"],
                "timestamp": datetime.utcnow().isoformat()
            }

            state = await asyncio.to_thread(core_graph.reasoning_node, state)
            nodes_executed.append("reasoning")

            yield {
                "event": "node_complete",
                "node": "reasoning",
                "message": NODE_MESSAGES["reasoning"]["complete"],
                "timestamp": datetime.utcnow().isoformat()
            }

            logger.info(f"[CORE STREAM] Reasoning complete, step_results count: {len(state.step_results)}")
            
            if state.step_results:
                for result in state.step_results:
                    logger.info(f"[CORE STREAM] Yielding step_executed for {result.step_id}, status={result.status}")
                    yield {
                        "event": "step_executed",
                        "step_id": result.step_id,
                        "status": result.status,
                        "outputs": result.outputs,
                        "artifacts": result.artifacts or [],
                        "logs": result.logs or [],
                        "error": result.error,
                        "duration_seconds": result.duration_seconds,
                        "timestamp": datetime.utcnow().isoformat()
                    }

            # 4. Evaluation
            logger.info("[CORE STREAM] Starting evaluation node")
            yield {
                "event": "node_start",
                "node": "evaluation",
                "message": NODE_MESSAGES["evaluation"]["start"],
                "timestamp": datetime.utcnow().isoformat()
            }

            state = await asyncio.to_thread(core_graph.evaluation_node, state)
            nodes_executed.append("evaluation")

            yield {
                "event": "node_complete",
                "node": "evaluation",
                "message": NODE_MESSAGES["evaluation"]["complete"],
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
        logger.info("[CORE STREAM] Starting conversation node")
        yield {
            "event": "node_start",
            "node": "conversation",
            "message": NODE_MESSAGES["conversation"]["start"],
            "timestamp": datetime.utcnow().isoformat()
        }

        state = await asyncio.to_thread(core_graph.conversation_node, state)
        nodes_executed.append("conversation")
        
        logger.info(f"[CORE STREAM] Conversation complete, response length: {len(state.response or '')}")
        logger.info(f"[CORE STREAM] Response preview: {(state.response or '')[:200]}")

        yield {
            "event": "node_complete",
            "node": "conversation",
            "message": NODE_MESSAGES["conversation"]["complete"],
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"[CORE STREAM] Exception in graph execution: {e}", exc_info=True)
        yield {
            "event": "error",
            "error": f"Node execution failed: {str(e)}",
            "nodes_executed": nodes_executed,
            "timestamp": datetime.utcnow().isoformat()
        }


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str) -> Dict[str, str]:
    """
    Delete a run from memory and database.
    """
    from app.dependencies import get_db_pool
    
    deleted_from = []
    
    # Remove from in-memory cache
    if run_id in _active_runs:
        del _active_runs[run_id]
        deleted_from.append("memory")
    
    # Remove from database
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM core_runs WHERE run_id = $1",
                run_id
            )
            if "DELETE 1" in result:
                deleted_from.append("database")
    except Exception as e:
        logger.warning(f"Failed to delete run {run_id} from database: {e}")
    
    if deleted_from:
        return {"message": f"Run {run_id} deleted from: {', '.join(deleted_from)}"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found"
        )


@router.get("/runs")
async def list_runs(
    user_id: Optional[str] = None,
    status_filter: Optional[str] = None,
    limit: int = 50,
    include_db: bool = True
) -> Dict[str, Any]:
    """
    List runs (in-memory active + persisted in database).
    
    Args:
        user_id: Filter by user ID (optional)
        status_filter: Filter by status (optional)
        limit: Maximum number of results
        include_db: Whether to include database results (default True)
    """
    runs = {}
    
    # Add in-memory active runs
    for run_id, state in _active_runs.items():
        run_status = "completed" if state.is_complete() else "running"
        
        # Apply filters
        if user_id and state.user_id != user_id:
            continue
        if status_filter and run_status != status_filter:
            continue
            
        runs[run_id] = {
            "status": run_status,
            "current_node": state.current_node,
            "user_input": state.user_input[:100] + "..." if len(state.user_input) > 100 else state.user_input,
            "started_at": state.started_at.isoformat(),
            "completed_at": state.completed_at.isoformat() if state.completed_at else None,
            "source": "memory"
        }
    
    # Include database runs for persistence across restarts
    if include_db:
        try:
            db_runs = await run_repository.list_runs(
                user_id=user_id,
                status=status_filter,
                limit=limit
            )
            for db_run in db_runs:
                rid = db_run["run_id"]
                if rid not in runs:  # Don't duplicate in-memory runs
                    runs[rid] = {
                        "status": db_run.get("status", "unknown"),
                        "current_node": db_run.get("current_node"),
                        "user_input": (db_run.get("user_input", "")[:100] + "...") if db_run.get("user_input") and len(db_run.get("user_input", "")) > 100 else db_run.get("user_input", ""),
                        "started_at": db_run["created_at"].isoformat() if db_run.get("created_at") else None,
                        "completed_at": db_run["completed_at"].isoformat() if db_run.get("completed_at") else None,
                        "source": "database"
                    }
        except Exception as e:
            logger.warning(f"Failed to fetch runs from database: {e}")

    return {
        "total_runs": len(runs),
        "runs": runs
    }
