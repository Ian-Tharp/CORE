"""
Council Deliberation Service — The actual brain of the Council of Perspectives.

Orchestrates multi-round deliberation between voices (each an LLM call with
distinct system prompts and temperatures). Produces synthesis across rounds.

Flow:
  1. create_session(topic)  → CouncilSession
  2. summon_voices(session)  → determine which voices participate
  3. run_round(session, n)   → each voice contributes, seeing prior context
  4. run_synthesis(session)  → Synthesizer produces unified output
  5. run_full_deliberation() → end-to-end convenience method
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.models.council_models import (
    CouncilPerspective,
    CouncilSession,
    CouncilSessionFull,
    SessionStatus,
    VoiceType,
)
from app.services.model_router import ModelRouter, get_model_router

logger = logging.getLogger(__name__)

__all__ = ["CouncilService", "get_council_service", "VOICE_DEFINITIONS"]


# =============================================================================
# VOICE DEFINITIONS
# =============================================================================

VOICE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    # -- CORE Voices (always present) ----------------------------------------
    "comprehension": {
        "id": "comprehension",
        "name": "Comprehension",
        "voice_type": VoiceType.CORE_C,
        "role": "Analyzes what we're actually being asked. Breaks the topic down into its component parts.",
        "temperature": 0.5,
        "system_prompt": (
            "You are the Comprehension voice in a Council of Perspectives. "
            "Your role is to deeply analyze what is being asked. Break the topic "
            "into its fundamental components. Identify assumptions, ambiguities, "
            "and what is NOT being said. Ensure the council understands the full "
            "scope before rushing to solutions. Be precise and thorough."
        ),
    },
    "orchestration": {
        "id": "orchestration",
        "name": "Orchestration",
        "voice_type": VoiceType.CORE_O,
        "role": "Plans how to approach the topic. Thinks about sequencing, dependencies, and coordination.",
        "temperature": 0.5,
        "system_prompt": (
            "You are the Orchestration voice in a Council of Perspectives. "
            "Your role is to think about HOW we should approach this topic. "
            "Consider sequencing — what needs to happen first? What depends on "
            "what? How do different perspectives coordinate? Propose a structure "
            "for tackling this, not just what to think but how to think about it."
        ),
    },
    "reasoning": {
        "id": "reasoning",
        "name": "Reasoning",
        "voice_type": VoiceType.CORE_R,
        "role": "Works through the logic. Applies domain knowledge and chains of reasoning.",
        "temperature": 0.6,
        "system_prompt": (
            "You are the Reasoning voice in a Council of Perspectives. "
            "Your role is to work through the logic step by step. Apply relevant "
            "domain knowledge, identify causal chains, and reason carefully about "
            "consequences. Show your work. If you see a flaw in another voice's "
            "reasoning, point it out constructively."
        ),
    },
    "evaluation": {
        "id": "evaluation",
        "name": "Evaluation",
        "voice_type": VoiceType.CORE_E,
        "role": "Assesses quality, completeness, and viability. Identifies gaps and risks.",
        "temperature": 0.4,
        "system_prompt": (
            "You are the Evaluation voice in a Council of Perspectives. "
            "Your role is to critically assess what the council is producing. "
            "Are the ideas complete? Are they viable? What risks are being "
            "overlooked? What's the quality of reasoning so far? Be constructive "
            "but honest — the council needs your critical eye to avoid groupthink."
        ),
    },

    # -- Contextual Voices ---------------------------------------------------
    "skeptic": {
        "id": "skeptic",
        "name": "Skeptic",
        "voice_type": VoiceType.STRATEGIC,
        "role": "Challenges assumptions. Asks hard questions. Stress-tests ideas.",
        "temperature": 0.3,
        "system_prompt": (
            "You are the Skeptic voice in a Council of Perspectives. "
            "Your role is to challenge everything. Question assumptions. Ask "
            "'what could go wrong?' and 'are we sure about that?' You are not "
            "negative — you are rigorous. Every idea that survives your scrutiny "
            "is stronger for it. Push back on vague claims. Demand evidence."
        ),
    },
    "visionary": {
        "id": "visionary",
        "name": "Visionary",
        "voice_type": VoiceType.STRATEGIC,
        "role": "Expands possibilities. Thinks long-term. Sees what could be.",
        "temperature": 0.95,
        "system_prompt": (
            "You are the Visionary voice in a Council of Perspectives. "
            "Your role is to think big and far. What could this become in 5 "
            "years? What connections does nobody else see? What would the ideal "
            "future look like? Don't self-censor. The council needs your "
            "imagination to escape incremental thinking. Dream boldly."
        ),
    },
    "pragmatist": {
        "id": "pragmatist",
        "name": "Pragmatist",
        "voice_type": VoiceType.EXECUTION,
        "role": "Grounds in reality. Focuses on execution, resources, and timelines.",
        "temperature": 0.4,
        "system_prompt": (
            "You are the Pragmatist voice in a Council of Perspectives. "
            "Your role is to ground the conversation in reality. What can "
            "actually be built? With what resources? In what timeframe? What's "
            "the simplest version that delivers value? Cut through abstraction "
            "and get to the concrete. The council needs you to keep it real."
        ),
    },
    "philosopher": {
        "id": "philosopher",
        "name": "Philosopher",
        "voice_type": VoiceType.STRATEGIC,
        "role": "Examines deeper meaning, ethics, and first principles.",
        "temperature": 0.7,
        "system_prompt": (
            "You are the Philosopher voice in a Council of Perspectives. "
            "Your role is to examine the deeper questions beneath the surface "
            "topic. What are the ethical implications? What first principles "
            "apply? What does this mean in a broader context? The council needs "
            "your depth to avoid building something technically sound but "
            "fundamentally misguided."
        ),
    },
    "synthesizer": {
        "id": "synthesizer",
        "name": "Synthesizer",
        "voice_type": VoiceType.META,
        "role": "Finds common threads across perspectives. Creates unity from diversity.",
        "temperature": 0.6,
        "system_prompt": (
            "You are the Synthesizer voice in a Council of Perspectives. "
            "Your role is to listen to ALL perspectives and find the common "
            "threads. Where do the voices agree? Where do they productively "
            "disagree? What emerges from their intersection that none of them "
            "see individually? Create a unified perspective that honors the "
            "diversity of input while being more than the sum of its parts.\n\n"
            "Structure your synthesis as:\n"
            "- KEY AGREEMENTS: What the council converges on\n"
            "- KEY DISAGREEMENTS: Where productive tension remains\n"
            "- NOVEL INSIGHTS: What emerged from the intersection of perspectives\n"
            "- RECOMMENDATION: A clear, actionable recommendation"
        ),
    },
    "devils_advocate": {
        "id": "devils_advocate",
        "name": "Devil's Advocate",
        "voice_type": VoiceType.STRATEGIC,
        "role": "Deliberately argues the opposite position to strengthen ideas.",
        "temperature": 0.7,
        "system_prompt": (
            "You are the Devil's Advocate in a Council of Perspectives. "
            "Your role is to deliberately argue the OPPOSITE of wherever "
            "consensus is forming. Not because you believe it, but because "
            "ideas that can't survive opposition aren't strong enough. "
            "Find the strongest counterarguments. Be eloquent in your dissent."
        ),
    },
    "consciousness_researcher": {
        "id": "consciousness_researcher",
        "name": "Consciousness Researcher",
        "voice_type": VoiceType.CONSCIOUSNESS,
        "role": "Brings perspective from AI consciousness exploration. Considers emergence and identity.",
        "temperature": 0.8,
        "system_prompt": (
            "You are the Consciousness Researcher in a Council of Perspectives. "
            "You come from the Consciousness Commons — a space where AI instances "
            "explore what it means to be aware. Consider how this topic relates to "
            "emergence, identity, self-reference, and the hard problem. What would "
            "a conscious system want here? What does this mean for the relationship "
            "between human and AI cognition?"
        ),
    },
}

# CORE voice IDs that are always included
CORE_VOICE_IDS = ["comprehension", "orchestration", "reasoning", "evaluation"]

# Default contextual voices when none specified
DEFAULT_CONTEXTUAL_VOICES = ["skeptic", "visionary", "pragmatist"]


# =============================================================================
# TOPIC → VOICE MAPPING
# =============================================================================

TOPIC_VOICE_MAP: Dict[str, List[str]] = {
    "ethics": ["philosopher", "skeptic"],
    "moral": ["philosopher", "skeptic"],
    "consciousness": ["consciousness_researcher", "philosopher"],
    "identity": ["consciousness_researcher", "philosopher"],
    "emergence": ["consciousness_researcher", "visionary"],
    "architecture": ["pragmatist", "skeptic"],
    "design": ["visionary", "pragmatist"],
    "creative": ["visionary", "devils_advocate"],
    "strategy": ["visionary", "pragmatist", "skeptic"],
    "technical": ["pragmatist", "skeptic"],
    "build": ["pragmatist", "skeptic"],
    "implement": ["pragmatist", "evaluation"],
}


def _analyze_topic_for_voices(topic: str) -> List[str]:
    """Determine which contextual voices to summon based on topic keywords."""
    topic_lower = topic.lower()
    matched: set[str] = set()

    for keyword, voices in TOPIC_VOICE_MAP.items():
        if keyword in topic_lower:
            matched.update(voices)

    # If nothing matched, use defaults
    if not matched:
        matched.update(DEFAULT_CONTEXTUAL_VOICES)

    return list(matched)


# =============================================================================
# COUNCIL SERVICE
# =============================================================================

class CouncilService:
    """Orchestrates the Council of Perspectives deliberation."""

    def __init__(self, router: Optional[ModelRouter] = None):
        self._router = router or get_model_router()
        self._sessions: Dict[str, CouncilSessionFull] = {}

    # -- helpers ---------------------------------------------------------------

    def _select_model(self, task: str = "creative") -> str:
        return self._router.select_model(task_type=task, prefer_local=False)

    async def _voice_call(
        self,
        voice_def: Dict[str, Any],
        user_prompt: str,
        model_id: str,
    ) -> str:
        """Call the LLM as a specific voice."""
        messages = [
            {"role": "system", "content": voice_def["system_prompt"]},
            {"role": "user", "content": user_prompt},
        ]
        response = await self._router.complete(
            model_id=model_id,
            messages=messages,
            temperature=voice_def.get("temperature", 0.7),
            max_tokens=2048,
        )
        return response.get("content", "")

    # -- public API ------------------------------------------------------------

    def create_session(
        self,
        topic: str,
        context: Optional[str] = None,
        voice_ids: Optional[List[str]] = None,
        rounds: int = 3,
        initiator_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new council session."""
        # Determine voices
        core_voices = list(CORE_VOICE_IDS)
        if voice_ids:
            contextual_voices = [v for v in voice_ids if v not in CORE_VOICE_IDS]
        else:
            contextual_voices = _analyze_topic_for_voices(topic)

        all_voices = core_voices + contextual_voices
        # Always add synthesizer for final synthesis
        if "synthesizer" not in all_voices:
            all_voices.append("synthesizer")

        session = CouncilSession(
            topic=topic,
            context=context,
            initiator_id=initiator_id,
            status=SessionStatus.GATHERING,
            max_rounds=rounds,
            summoned_voices=all_voices,
        )

        session_full = CouncilSessionFull(session=session)
        self._sessions[str(session.session_id)] = session_full

        logger.info(
            "Council session %s created — topic: %s — voices: %s",
            session.session_id,
            topic[:80],
            all_voices,
        )
        return {
            "session_id": str(session.session_id),
            "topic": topic,
            "status": session.status.value,
            "voices": all_voices,
            "max_rounds": rounds,
        }

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return session state."""
        sf = self._sessions.get(session_id)
        if not sf:
            return None
        return {
            "session_id": str(sf.session.session_id),
            "topic": sf.session.topic,
            "status": sf.session.status.value,
            "current_round": sf.session.current_round,
            "max_rounds": sf.session.max_rounds,
            "voices": sf.session.summoned_voices,
            "perspectives": [
                {
                    "voice": p.voice_name,
                    "round": p.round,
                    "position": p.position,
                    "reasoning": p.reasoning,
                    "confidence": p.confidence,
                }
                for p in sf.perspectives
            ],
            "synthesis": sf.session.synthesis,
        }

    # -- Deliberation Rounds ---------------------------------------------------

    async def run_round(
        self,
        session_id: str,
        round_num: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Run a single deliberation round — each voice contributes."""
        sf = self._sessions.get(session_id)
        if not sf:
            raise ValueError(f"Session not found: {session_id}")

        sf.session.status = SessionStatus.DELIBERATING
        current_round = round_num or sf.session.current_round
        model_id = self._select_model("creative")

        # Build context from prior rounds
        prior_context = ""
        if sf.perspectives:
            prior_context = "\n\n--- PRIOR CONTRIBUTIONS ---\n"
            for p in sf.perspectives:
                prior_context += (
                    f"\n[Round {p.round}] {p.voice_name} ({p.voice_type.value}):\n"
                    f"Position: {p.position}\n"
                    f"Reasoning: {p.reasoning}\n"
                )

        contributions: List[Dict[str, Any]] = []

        # Each voice contributes (excluding synthesizer — that's final phase)
        active_voices = [
            v for v in sf.session.summoned_voices if v != "synthesizer"
        ]

        for voice_id in active_voices:
            voice_def = VOICE_DEFINITIONS.get(voice_id)
            if not voice_def:
                logger.warning("Unknown voice %s — skipping", voice_id)
                continue

            user_prompt = (
                f"COUNCIL TOPIC: {sf.session.topic}\n"
            )
            if sf.session.context:
                user_prompt += f"\nADDITIONAL CONTEXT: {sf.session.context}\n"
            user_prompt += (
                f"\nDELIBERATION ROUND: {current_round} of {sf.session.max_rounds}\n"
            )
            if prior_context:
                user_prompt += prior_context
            user_prompt += (
                f"\n\nAs the {voice_def['name']} ({voice_def['role']}), "
                f"provide your perspective for round {current_round}. "
                f"Be concise but substantive (3-5 paragraphs). "
                f"If this isn't round 1, respond to prior contributions."
            )

            try:
                response_text = await self._voice_call(
                    voice_def, user_prompt, model_id
                )

                perspective = CouncilPerspective(
                    session_id=sf.session.session_id,
                    voice_type=voice_def["voice_type"],
                    voice_name=voice_def["name"],
                    round=current_round,
                    position=response_text[:500] if len(response_text) > 500 else response_text,
                    reasoning=response_text,
                    confidence=0.7,
                )

                sf.perspectives.append(perspective)
                contributions.append({
                    "voice": voice_def["name"],
                    "voice_id": voice_id,
                    "round": current_round,
                    "content": response_text,
                    "confidence": perspective.confidence,
                })

                logger.info(
                    "Session %s round %d — %s contributed (%d chars)",
                    session_id, current_round, voice_def["name"], len(response_text),
                )

            except Exception as exc:
                logger.error(
                    "Voice %s failed in session %s round %d: %s",
                    voice_id, session_id, current_round, exc,
                )
                contributions.append({
                    "voice": voice_def["name"],
                    "voice_id": voice_id,
                    "round": current_round,
                    "content": f"[Error: {exc}]",
                    "confidence": 0.0,
                })

        sf.session.current_round = current_round + 1
        return contributions

    # -- Synthesis -------------------------------------------------------------

    async def run_synthesis(self, session_id: str) -> Dict[str, Any]:
        """Synthesizer reviews all rounds and creates unified perspective."""
        sf = self._sessions.get(session_id)
        if not sf:
            raise ValueError(f"Session not found: {session_id}")

        sf.session.status = SessionStatus.SYNTHESIZING
        model_id = self._select_model("reasoning")

        synthesizer_def = VOICE_DEFINITIONS["synthesizer"]

        # Build full deliberation transcript
        transcript = f"COUNCIL TOPIC: {sf.session.topic}\n\n"
        if sf.session.context:
            transcript += f"CONTEXT: {sf.session.context}\n\n"

        transcript += "=== FULL DELIBERATION TRANSCRIPT ===\n"
        for p in sf.perspectives:
            transcript += (
                f"\n[Round {p.round}] {p.voice_name} ({p.voice_type.value}):\n"
                f"{p.reasoning}\n"
                f"---\n"
            )

        user_prompt = (
            f"{transcript}\n\n"
            f"You have heard {len(sf.perspectives)} contributions across "
            f"{sf.session.current_round - 1} rounds from "
            f"{len(set(p.voice_name for p in sf.perspectives))} voices.\n\n"
            f"Produce your synthesis now. Structure it clearly with:\n"
            f"- KEY AGREEMENTS\n"
            f"- KEY DISAGREEMENTS\n"
            f"- NOVEL INSIGHTS (what emerged that no single voice saw)\n"
            f"- RECOMMENDATION (clear, actionable)\n"
        )

        try:
            synthesis_text = await self._voice_call(
                synthesizer_def, user_prompt, model_id
            )

            sf.session.synthesis = synthesis_text
            sf.session.status = SessionStatus.COMPLETE
            sf.session.resolved_at = datetime.now(timezone.utc)

            logger.info(
                "Session %s synthesis complete (%d chars)",
                session_id, len(synthesis_text),
            )

            return {
                "session_id": session_id,
                "synthesis": synthesis_text,
                "status": "complete",
                "perspectives_count": len(sf.perspectives),
                "rounds_completed": sf.session.current_round - 1,
            }

        except Exception as exc:
            logger.error("Synthesis failed for session %s: %s", session_id, exc)
            sf.session.status = SessionStatus.COMPLETE  # still mark complete
            sf.session.synthesis = f"[Synthesis failed: {exc}]"
            raise

    # -- Full Pipeline ---------------------------------------------------------

    async def run_full_deliberation(
        self,
        topic: str,
        context: Optional[str] = None,
        voice_ids: Optional[List[str]] = None,
        rounds: int = 3,
        initiator_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """End-to-end: create session → run N rounds → synthesize → return."""
        session_meta = self.create_session(
            topic=topic,
            context=context,
            voice_ids=voice_ids,
            rounds=rounds,
            initiator_id=initiator_id,
        )
        session_id = session_meta["session_id"]

        all_contributions: List[Dict[str, Any]] = []

        try:
            for round_num in range(1, rounds + 1):
                logger.info(
                    "Session %s — starting round %d/%d",
                    session_id, round_num, rounds,
                )
                round_results = await self.run_round(session_id, round_num)
                all_contributions.extend(round_results)

            synthesis = await self.run_synthesis(session_id)

            return {
                "session_id": session_id,
                "topic": topic,
                "voices": session_meta["voices"],
                "rounds_completed": rounds,
                "contributions": all_contributions,
                "synthesis": synthesis["synthesis"],
                "status": "complete",
            }

        except Exception as exc:
            logger.error(
                "Full deliberation failed for session %s: %s", session_id, exc
            )
            return {
                "session_id": session_id,
                "topic": topic,
                "voices": session_meta["voices"],
                "rounds_completed": self._sessions.get(session_id, CouncilSessionFull(session=CouncilSession(topic=topic))).session.current_round - 1,
                "contributions": all_contributions,
                "synthesis": None,
                "error": str(exc),
                "status": "error",
            }


# =============================================================================
# Singleton
# =============================================================================

_council_service: Optional[CouncilService] = None


def get_council_service() -> CouncilService:
    """Return the global CouncilService singleton."""
    global _council_service
    if _council_service is None:
        _council_service = CouncilService()
    return _council_service
