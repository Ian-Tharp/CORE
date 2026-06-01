"""
Intelligence Layer Logging for CORE Pipeline.

Records structured pipeline run summaries to a dedicated Python logger.
Each log entry captures enough context for retrospective analysis, model
performance comparison, and quality trend monitoring.

Log entries are emitted as structured JSON on the "core.intelligence" logger.
A structured log handler (e.g. JSON formatter → file / ELK stack) can pick
them up without any code changes.

Schema per entry:
    {
        "event":            "pipeline.run_complete",
        "session_id":       str | None,
        "user_input_len":   int,
        "intent_type":      str | None,
        "task_category":    str | None,
        "intent_confidence": float | None,
        "plan_goal":        str | None,
        "plan_step_count":  int,
        "plan_revisions":   int,
        "steps_succeeded":  int,
        "steps_failed":     int,
        "eval_status":      str | None,
        "eval_quality":     float | None,
        "eval_confidence":  float | None,
        "eval_next_action": str | None,
        "response_len":     int,
        "nodes_visited":    list[str],
        "errors_count":     int,
        "timestamp":        ISO-8601 str,
    }
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.core_state import COREState

_log = logging.getLogger("core.intelligence")


def record_pipeline_run(state: "COREState") -> dict:
    """
    Build and emit a structured log entry for a completed pipeline run.

    Safe to call at any pipeline stage — missing fields are handled gracefully.
    Emits the entry via ``_log.info()`` as JSON and returns the dict so callers
    (and tests) can inspect what was recorded.

    Args:
        state: The final COREState after the pipeline completes.

    Returns:
        The structured log entry dict.
    """
    # Step result tallies
    steps_succeeded = sum(1 for r in state.step_results if r.status == "success")
    steps_failed = sum(1 for r in state.step_results if r.status == "failure")

    entry = {
        "event": "pipeline.run_complete",
        "session_id": getattr(state, "session_id", None),
        "user_input_len": len(state.user_input or ""),
        # Intent
        "intent_type": state.intent.type if state.intent else None,
        "task_category": state.intent.task_category if state.intent else None,
        "intent_confidence": (float(state.intent.confidence) if state.intent else None),
        # Plan
        "plan_goal": state.plan.goal if state.plan else None,
        "plan_step_count": len(state.plan.steps) if state.plan else 0,
        "plan_revisions": state.plan_revisions,
        # Step execution
        "steps_succeeded": steps_succeeded,
        "steps_failed": steps_failed,
        # Evaluation
        "eval_status": (
            state.eval_result.overall_status if state.eval_result else None
        ),
        "eval_quality": (
            float(state.eval_result.quality_score) if state.eval_result else None
        ),
        "eval_confidence": (
            float(state.eval_result.confidence) if state.eval_result else None
        ),
        "eval_next_action": (
            state.eval_result.next_action if state.eval_result else None
        ),
        # Response
        "response_len": len(state.response or ""),
        # Pipeline path
        "nodes_visited": list(state.execution_history or []),
        "errors_count": len(state.errors or []),
        # Timing
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }

    _log.info(json.dumps(entry))
    return entry


def get_intelligence_logger() -> logging.Logger:
    """Return the intelligence logger (useful for test-time handler injection)."""
    return _log
