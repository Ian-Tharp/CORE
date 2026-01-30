"""
Catalyst Creativity Service — Divergence → Convergence → Synthesis pipeline.

Orchestrates a three-phase creative process using LLM calls via the
ModelRouter.  Sessions are stored in-memory for now (swap to DB later).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from app.prompts.catalyst_prompts import (
    CONVERGENCE_SYSTEM_PROMPT,
    DIVERGENCE_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
)
from app.services.model_router import ModelRouter, get_model_router

logger = logging.getLogger(__name__)

__all__ = ["CatalystService", "get_catalyst_service"]


# ---------------------------------------------------------------------------
# Session bookkeeping
# ---------------------------------------------------------------------------

class SessionPhase(str, Enum):
    CREATED = "created"
    DIVERGENCE = "divergence"
    CONVERGENCE = "convergence"
    SYNTHESIS = "synthesis"
    COMPLETE = "complete"
    ERROR = "error"


class CatalystSession:
    """Lightweight in-memory session for one creative pipeline run."""

    def __init__(self, session_id: str, prompt: str, config: Dict[str, Any]):
        self.session_id = session_id
        self.prompt = prompt
        self.config = config
        self.phase = SessionPhase.CREATED
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at
        self.divergence_result: Optional[List[Dict[str, Any]]] = None
        self.convergence_result: Optional[Dict[str, Any]] = None
        self.synthesis_result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None

    def _touch(self):
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "prompt": self.prompt,
            "config": self.config,
            "phase": self.phase.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "divergence_result": self.divergence_result,
            "convergence_result": self.convergence_result,
            "synthesis_result": self.synthesis_result,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------

class CatalystService:
    """Orchestrates the Divergence → Convergence → Synthesis pipeline."""

    def __init__(self, router: Optional[ModelRouter] = None):
        self._router = router or get_model_router()
        self._sessions: Dict[str, CatalystSession] = {}

    # -- helpers --------------------------------------------------------------

    def _select_model(self, config: Dict[str, Any]) -> str:
        """Pick an LLM model id from explicit config or auto-select."""
        if model := config.get("model"):
            return model
        return self._router.select_model(task_type="creative", prefer_local=False)

    async def _llm_json(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: str,
        temperature: float = 0.9,
    ) -> Any:
        """Call the LLM and parse the response as JSON."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = await self._router.complete(
            model_id=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
        )
        raw = response.get("content", "")
        # Strip markdown fences if the model wraps them anyway
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (```json or ```)
            first_newline = cleaned.index("\n")
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("LLM returned non-JSON. Raw output:\n%s", raw)
            raise ValueError(f"LLM response was not valid JSON: {exc}") from exc

    # -- public API -----------------------------------------------------------

    def start_session(
        self, prompt: str, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new catalyst session and return its metadata."""
        session_id = str(uuid.uuid4())
        session = CatalystSession(
            session_id=session_id,
            prompt=prompt,
            config=config or {},
        )
        self._sessions[session_id] = session
        logger.info("Catalyst session created: %s", session_id)
        return session.to_dict()

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return session state or None if not found."""
        session = self._sessions.get(session_id)
        return session.to_dict() if session else None

    # -- Phase 1: Divergence --------------------------------------------------

    async def run_divergence(
        self,
        session_id: str,
        prompt: str,
        count: int = 5,
    ) -> List[Dict[str, Any]]:
        """Generate *count* divergent ideas for *prompt*."""
        session = self._sessions.get(session_id)
        if session:
            session.phase = SessionPhase.DIVERGENCE
            session._touch()

        model_id = self._select_model(session.config if session else {})

        user_msg = (
            f"Generate exactly {count} divergent creative ideas for the following prompt.\n\n"
            f"PROMPT: {prompt}"
        )

        try:
            ideas: List[Dict[str, Any]] = await self._llm_json(
                system_prompt=DIVERGENCE_SYSTEM_PROMPT,
                user_prompt=user_msg,
                model_id=model_id,
                temperature=0.95,
            )
            # Ensure list
            if not isinstance(ideas, list):
                ideas = [ideas]

            if session:
                session.divergence_result = ideas
                session._touch()

            logger.info(
                "Divergence complete for session %s — %d ideas generated",
                session_id,
                len(ideas),
            )
            return ideas

        except Exception as exc:
            if session:
                session.phase = SessionPhase.ERROR
                session.error = str(exc)
                session._touch()
            raise

    # -- Phase 2: Convergence -------------------------------------------------

    async def run_convergence(
        self,
        session_id: str,
        divergent_ideas: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate, group, and rank *divergent_ideas*."""
        session = self._sessions.get(session_id)
        if session:
            session.phase = SessionPhase.CONVERGENCE
            session._touch()

        model_id = self._select_model(session.config if session else {})

        user_msg = (
            "Evaluate, group, and rank the following divergent ideas.\n\n"
            f"IDEAS:\n{json.dumps(divergent_ideas, indent=2)}"
        )

        try:
            result: Dict[str, Any] = await self._llm_json(
                system_prompt=CONVERGENCE_SYSTEM_PROMPT,
                user_prompt=user_msg,
                model_id=model_id,
                temperature=0.4,  # more analytical, less random
            )

            if session:
                session.convergence_result = result
                session._touch()

            logger.info("Convergence complete for session %s", session_id)
            return result

        except Exception as exc:
            if session:
                session.phase = SessionPhase.ERROR
                session.error = str(exc)
                session._touch()
            raise

    # -- Phase 3: Synthesis ---------------------------------------------------

    async def run_synthesis(
        self,
        session_id: str,
        divergent_ideas: List[Dict[str, Any]],
        convergent_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Synthesize divergent ideas + convergent analysis into a unified output."""
        session = self._sessions.get(session_id)
        if session:
            session.phase = SessionPhase.SYNTHESIS
            session._touch()

        model_id = self._select_model(session.config if session else {})
        original_prompt = session.prompt if session else "(unknown)"

        user_msg = (
            f"ORIGINAL PROMPT: {original_prompt}\n\n"
            f"DIVERGENT IDEAS:\n{json.dumps(divergent_ideas, indent=2)}\n\n"
            f"CONVERGENT ANALYSIS:\n{json.dumps(convergent_analysis, indent=2)}"
        )

        try:
            result: Dict[str, Any] = await self._llm_json(
                system_prompt=SYNTHESIS_SYSTEM_PROMPT,
                user_prompt=user_msg,
                model_id=model_id,
                temperature=0.7,
            )

            if session:
                session.synthesis_result = result
                session.phase = SessionPhase.COMPLETE
                session._touch()

            logger.info("Synthesis complete for session %s", session_id)
            return result

        except Exception as exc:
            if session:
                session.phase = SessionPhase.ERROR
                session.error = str(exc)
                session._touch()
            raise

    # -- Full pipeline --------------------------------------------------------

    async def auto_catalyst(
        self,
        prompt: str,
        count: int = 5,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the full Divergence → Convergence → Synthesis pipeline end-to-end.

        Returns the complete session dict including all phase outputs.
        """
        session_meta = self.start_session(prompt, config)
        session_id = session_meta["session_id"]

        try:
            # Phase 1
            ideas = await self.run_divergence(session_id, prompt, count=count)

            # Phase 2
            convergence = await self.run_convergence(session_id, ideas)

            # Phase 3
            synthesis = await self.run_synthesis(session_id, ideas, convergence)

            return self.get_session(session_id)  # type: ignore[return-value]

        except Exception as exc:
            logger.error("auto_catalyst failed for session %s: %s", session_id, exc)
            # Session already marked ERROR inside the phase methods
            result = self.get_session(session_id)
            if result:
                return result
            raise


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_catalyst_service: Optional[CatalystService] = None


def get_catalyst_service() -> CatalystService:
    """Return the global CatalystService singleton."""
    global _catalyst_service
    if _catalyst_service is None:
        _catalyst_service = CatalystService()
    return _catalyst_service
