"""
Council Deliberation Engine

The core deliberation pipeline that orchestrates multi-perspective AI reasoning.
Each voice is an LLM call with a distinct system prompt and temperature, creating
genuine diversity of thought. Rounds feed previous contributions as context,
enabling voices to respond to, build on, and challenge each other.

Pipeline: create_session → summon_voices → run_round(×N) → run_synthesis
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from app.models.council_models import (
    CouncilPerspective,
    CouncilSession,
    CouncilSessionFull,
    SessionStatus,
    VoiceType,
)
from app.repository import council_repository as repo
from app.services.council.voice_registry import (
    VoiceCategory,
    VoiceDefinition,
    get_core_voices,
    get_voice,
    VOICE_REGISTRY,
)
from app.services.consciousness_council_bridge import (
    ConsciousnessCouncilBridge,
    get_consciousness_council_bridge,
)
from app.services.model_router import ModelRouter, get_model_router

logger = logging.getLogger(__name__)


# ── Voice-type mapping ───────────────────────────────────────────────────────
# Maps voice registry keys to the VoiceType enum used by the data models.
_CATEGORY_TO_VOICE_TYPE: Dict[VoiceCategory, VoiceType] = {
    VoiceCategory.CORE: VoiceType.CORE_C,       # overridden per-voice below
    VoiceCategory.STRATEGIC: VoiceType.STRATEGIC,
    VoiceCategory.DOMAIN: VoiceType.DOMAIN,
    VoiceCategory.EXECUTION: VoiceType.EXECUTION,
    VoiceCategory.META: VoiceType.META,
}

_CORE_NAME_TO_VOICE_TYPE: Dict[str, VoiceType] = {
    "CORE-C": VoiceType.CORE_C,
    "CORE-O": VoiceType.CORE_O,
    "CORE-R": VoiceType.CORE_R,
    "CORE-E": VoiceType.CORE_E,
}


def _voice_type_for(voice: VoiceDefinition) -> VoiceType:
    """Resolve the correct VoiceType enum value for a voice definition."""
    if voice.category == VoiceCategory.CORE:
        return _CORE_NAME_TO_VOICE_TYPE.get(voice.name, VoiceType.CORE_C)
    return _CATEGORY_TO_VOICE_TYPE.get(voice.category, VoiceType.META)


# ── Default contextual voices ────────────────────────────────────────────────
# These are always included alongside the 4 CORE voices unless overridden.
DEFAULT_CONTEXTUAL_VOICE_IDS = [
    "devils_advocate",   # Skeptic / challenger
    "oracle",            # Visionary
    "product_lead",      # Pragmatist
    "synthesizer",       # Final synthesis perspective
]


# =============================================================================
# COUNCIL SERVICE
# =============================================================================


class CouncilService:
    """
    Orchestrates multi-perspective deliberation sessions.

    Each deliberation:
      1. Creates a session (persisted)
      2. Summons voices – always CORE (C, O, R, E) plus contextual voices
      3. Runs N rounds – each voice contributes after seeing prior output
      4. Runs synthesis – the Synthesizer distils agreements / disagreements / insights
      5. Returns the complete result
    """

    def __init__(
        self,
        model_router: Optional[ModelRouter] = None,
        default_model: Optional[str] = None,
        max_concurrent_voices: int = 4,
        consciousness_bridge: Optional[ConsciousnessCouncilBridge] = None,
    ):
        self.router = model_router or get_model_router()
        # Pick a sensible default model – balanced tier
        self.default_model = default_model or self.router.select_model(
            task_type="reasoning",
            require_tools=False,
            prefer_local=True,
        )
        self.max_concurrent = max_concurrent_voices
        self.consciousness_bridge = (
            consciousness_bridge or get_consciousness_council_bridge()
        )

    # ------------------------------------------------------------------
    # 1. Create Session
    # ------------------------------------------------------------------

    async def create_session(
        self,
        topic: str,
        voice_ids: Optional[List[str]] = None,
        rounds: int = 3,
        context: Optional[str] = None,
        initiator_id: Optional[str] = None,
    ) -> dict:
        """
        Create and persist a new deliberation session.

        Args:
            topic: The question or problem to deliberate
            voice_ids: Optional list of voice registry keys to include
                       (CORE voices are *always* included regardless)
            rounds: Max deliberation rounds (1-10)
            context: Optional background context for the topic
            initiator_id: Who/what started this session

        Returns:
            Dict with session_id, topic, status, summoned_voices
        """
        await repo.ensure_council_tables()

        resolved_voice_ids = self._resolve_voice_ids(voice_ids)

        session = CouncilSession(
            topic=topic,
            context=context,
            initiator_id=initiator_id,
            max_rounds=rounds,
            summoned_voices=resolved_voice_ids,
            status=SessionStatus.GATHERING,
        )
        session_id = await repo.create_session(session)

        logger.info(
            "Council session %s created – topic='%s' voices=%s rounds=%d",
            session_id, topic[:80], resolved_voice_ids, rounds,
        )

        return {
            "session_id": str(session_id),
            "topic": topic,
            "status": SessionStatus.GATHERING.value,
            "summoned_voices": resolved_voice_ids,
            "max_rounds": rounds,
        }

    # ------------------------------------------------------------------
    # 2. Summon Voices
    # ------------------------------------------------------------------

    async def summon_voices(self, session_id: UUID) -> List[dict]:
        """
        Return the voice definitions summoned for a session.

        Always includes the 4 CORE voices + whatever contextual voices
        were specified at session creation.

        Returns:
            List of dicts with id, name, role, category, temperature
        """
        session = await self._get_session_or_raise(session_id)

        voices: List[dict] = []
        seen_names: set = set()

        for vid in session.summoned_voices:
            try:
                v = get_voice(vid)
            except KeyError:
                logger.warning("Voice '%s' not found in registry, skipping", vid)
                continue

            if v.name in seen_names:
                continue
            seen_names.add(v.name)

            voices.append({
                "id": vid,
                "name": v.name,
                "role": v.role,
                "category": v.category.value,
                "temperature": v.temperature,
            })

        logger.info("Session %s: summoned %d voices", session_id, len(voices))
        return voices

    # ------------------------------------------------------------------
    # 3. Run Round
    # ------------------------------------------------------------------

    async def run_round(
        self,
        session_id: UUID,
        round_num: int,
    ) -> List[dict]:
        """
        Execute one deliberation round.

        Each voice sees the topic + all prior contributions, then produces
        a position, reasoning, and confidence score. Voices run with
        bounded concurrency for throughput.

        Args:
            session_id: Active session UUID
            round_num: Which round to run (1-indexed)

        Returns:
            List of perspective dicts produced this round
        """
        session = await self._get_session_or_raise(session_id)

        # Transition from GATHERING → DELIBERATING on first round
        if session.status == SessionStatus.GATHERING:
            await repo.update_session_status(session_id, SessionStatus.DELIBERATING)

        # Gather all prior perspectives as context
        prior_perspectives = await repo.get_perspectives_by_session(session_id)
        prior_context = self._format_prior_context(prior_perspectives)

        # Collect voice definitions (skip synthesizer – it runs at the end)
        voices = self._get_deliberation_voices(session.summoned_voices)

        # Run all voices concurrently (bounded)
        sem = asyncio.Semaphore(self.max_concurrent)
        tasks = [
            self._run_voice(
                sem, session_id, round_num, vid, voice,
                session.topic, session.context, prior_context,
            )
            for vid, voice in voices
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful perspectives
        perspectives: List[dict] = []
        for r in results:
            if isinstance(r, Exception):
                logger.error("Voice failed in round %d: %s", round_num, r)
            elif r is not None:
                perspectives.append(r)

        # Advance session round counter if not at the target yet
        if session.current_round <= round_num:
            await repo.advance_round(session_id)

        logger.info(
            "Session %s round %d complete – %d/%d perspectives",
            session_id, round_num, len(perspectives), len(voices),
        )
        return perspectives

    # ------------------------------------------------------------------
    # 4. Run Synthesis
    # ------------------------------------------------------------------

    async def run_synthesis(self, session_id: UUID) -> dict:
        """
        Run the Synthesizer voice over all rounds to produce the final output.

        The synthesis identifies:
          - agreements:    points most voices converge on
          - disagreements: points of genuine tension
          - insights:      novel observations that emerged
          - recommendation: a clear suggested path forward

        Returns:
            Dict with synthesis text and structured fields
        """
        session = await self._get_session_or_raise(session_id)
        await repo.update_session_status(session_id, SessionStatus.SYNTHESIZING)

        all_perspectives = await repo.get_perspectives_by_session(session_id)
        deliberation_transcript = self._format_full_transcript(
            all_perspectives, session.topic
        )

        synthesizer = get_voice("synthesizer")
        synthesis_prompt = self._build_synthesis_prompt(
            session.topic, session.context, deliberation_transcript
        )

        response = await self.router.complete(
            model_id=self.default_model,
            messages=[
                {"role": "system", "content": synthesizer.system_prompt},
                {"role": "user", "content": synthesis_prompt},
            ],
            temperature=synthesizer.temperature,
            max_tokens=4096,
        )

        synthesis_text = response.get("content", "")

        # Persist synthesis and mark session complete
        await repo.update_session_status(
            session_id, SessionStatus.COMPLETE, synthesis=synthesis_text
        )

        logger.info("Session %s synthesis complete", session_id)

        return {
            "session_id": str(session_id),
            "status": SessionStatus.COMPLETE.value,
            "synthesis": synthesis_text,
            "total_perspectives": len(all_perspectives),
            "rounds_completed": session.current_round,
            "model_used": self.default_model,
        }

    # ------------------------------------------------------------------
    # 5. Full Deliberation Pipeline
    # ------------------------------------------------------------------

    async def run_full_deliberation(
        self,
        topic: str,
        voice_ids: Optional[List[str]] = None,
        rounds: int = 3,
        context: Optional[str] = None,
        initiator_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> dict:
        """
        End-to-end deliberation: create → summon → rounds → synthesis.

        This is the primary entry point for callers who want a complete
        multi-perspective analysis of a topic.

        Returns:
            Full deliberation result including all perspectives and synthesis
        """
        # Allow per-call model override
        original_model = self.default_model
        if model:
            self.default_model = model

        try:
            # Step 1: Create session
            session_info = await self.create_session(
                topic=topic,
                voice_ids=voice_ids,
                rounds=rounds,
                context=context,
                initiator_id=initiator_id,
            )
            session_id = UUID(session_info["session_id"])

            # Step 2: Summon voices (informational)
            summoned = await self.summon_voices(session_id)

            # Step 3: Run deliberation rounds
            all_round_results: List[List[dict]] = []
            for round_num in range(1, rounds + 1):
                round_perspectives = await self.run_round(session_id, round_num)
                all_round_results.append(round_perspectives)

            # Step 4: Synthesis
            synthesis_result = await self.run_synthesis(session_id)

            # Build complete response
            return {
                "session_id": str(session_id),
                "topic": topic,
                "status": SessionStatus.COMPLETE.value,
                "voices": summoned,
                "rounds": [
                    {
                        "round": i + 1,
                        "perspectives": perspectives,
                    }
                    for i, perspectives in enumerate(all_round_results)
                ],
                "synthesis": synthesis_result["synthesis"],
                "total_perspectives": synthesis_result["total_perspectives"],
                "model_used": self.default_model,
            }
        finally:
            self.default_model = original_model

    # ══════════════════════════════════════════════════════════════════════
    # PRIVATE HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _resolve_voice_ids(self, voice_ids: Optional[List[str]]) -> List[str]:
        """Ensure CORE voices are always present plus requested contextual voices."""
        core_ids = ["core_c", "core_o", "core_r", "core_e"]

        if voice_ids:
            # Deduplicate while preserving order
            extra = [v for v in voice_ids if v not in core_ids]
            return core_ids + extra
        else:
            return core_ids + DEFAULT_CONTEXTUAL_VOICE_IDS

    def _get_deliberation_voices(
        self, voice_ids: List[str]
    ) -> List[tuple[str, VoiceDefinition]]:
        """
        Get voice definitions for deliberation rounds.
        Excludes the synthesizer (it runs separately in synthesis phase).
        """
        voices = []
        seen = set()
        for vid in voice_ids:
            try:
                v = get_voice(vid)
            except KeyError:
                continue
            # Skip synthesizer – it has its own step
            if v.name == "Synthesizer":
                continue
            if v.name in seen:
                continue
            seen.add(v.name)
            voices.append((vid, v))
        return voices

    async def _run_voice(
        self,
        sem: asyncio.Semaphore,
        session_id: UUID,
        round_num: int,
        voice_id: str,
        voice: VoiceDefinition,
        topic: str,
        topic_context: Optional[str],
        prior_context: str,
    ) -> Optional[dict]:
        """
        Execute a single voice's contribution in the deliberation.

        The LLM is prompted with the voice's system prompt + a user message
        containing the topic and prior round output. The response is parsed
        into a structured perspective and persisted.
        """
        async with sem:
            try:
                user_prompt = self._build_voice_prompt(
                    voice, topic, topic_context, prior_context, round_num
                )

                response = await self.router.complete(
                    model_id=self.default_model,
                    messages=[
                        {"role": "system", "content": voice.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=voice.temperature,
                    max_tokens=2048,
                )

                content = response.get("content", "")
                if not content:
                    logger.warning("Voice %s returned empty content", voice.name)
                    return None

                # Parse the response into structured fields
                position, reasoning, confidence = self._parse_voice_response(
                    content, voice.name
                )

                voice_type = _voice_type_for(voice)

                # Persist
                perspective = CouncilPerspective(
                    session_id=session_id,
                    voice_type=voice_type,
                    voice_name=voice.name,
                    round=round_num,
                    position=position,
                    reasoning=reasoning,
                    confidence=confidence,
                )
                perspective_id = await repo.create_perspective(perspective)

                return {
                    "perspective_id": str(perspective_id),
                    "voice_id": voice_id,
                    "voice_name": voice.name,
                    "voice_role": voice.role,
                    "round": round_num,
                    "position": position,
                    "reasoning": reasoning,
                    "confidence": confidence,
                }

            except Exception as e:
                logger.error("Voice %s failed: %s", voice.name, e, exc_info=True)
                return None

    # ── Prompt Construction ──────────────────────────────────────────────

    def _build_voice_prompt(
        self,
        voice: VoiceDefinition,
        topic: str,
        topic_context: Optional[str],
        prior_context: str,
        round_num: int,
    ) -> str:
        """Build the user prompt sent to each voice during deliberation."""
        parts = [
            f"## Council Deliberation — Round {round_num}",
            f"",
            f"**Topic:** {topic}",
        ]

        if topic_context:
            parts.append(f"")
            parts.append(f"**Context:** {topic_context}")

        if prior_context:
            parts.append(f"")
            parts.append("---")
            parts.append("## Prior Contributions")
            parts.append(prior_context)
            parts.append("---")
            parts.append("")

        parts.append(
            f"You are **{voice.name}** ({voice.role}). "
            f"Provide your perspective on the topic above."
        )
        parts.append("")
        parts.append(
            "Structure your response as follows:\n"
            "**POSITION:** Your clear stance or key insight (2-4 sentences)\n\n"
            "**REASONING:** Your detailed analysis and justification\n\n"
            "**CONFIDENCE:** A number between 0.0 and 1.0 indicating how "
            "confident you are in this position"
        )

        if voice.key_questions:
            parts.append("")
            parts.append("Consider these guiding questions:")
            for q in voice.key_questions:
                parts.append(f"  - {q}")

        if round_num > 1 and prior_context:
            parts.append("")
            parts.append(
                "This is round {r}. Build on, respond to, or challenge "
                "the perspectives shared so far. Show how your thinking "
                "has evolved based on what others have contributed.".format(
                    r=round_num
                )
            )

        return "\n".join(parts)

    def _build_synthesis_prompt(
        self,
        topic: str,
        topic_context: Optional[str],
        transcript: str,
    ) -> str:
        """Build the prompt for the final synthesis phase."""
        parts = [
            "## Council Deliberation — Final Synthesis",
            "",
            f"**Topic:** {topic}",
        ]
        if topic_context:
            parts.append(f"**Context:** {topic_context}")
        parts.append("")
        parts.append("---")
        parts.append("## Complete Deliberation Transcript")
        parts.append(transcript)
        parts.append("---")
        parts.append("")
        parts.append(
            "You are the **Synthesizer**. Review all perspectives across all rounds "
            "and produce a comprehensive synthesis.\n\n"
            "Structure your synthesis as follows:\n\n"
            "**AGREEMENTS:** Points where most voices converge. What is the council "
            "broadly aligned on?\n\n"
            "**DISAGREEMENTS:** Points of genuine tension or divergence. Where do "
            "voices meaningfully differ, and why?\n\n"
            "**KEY INSIGHTS:** Novel observations, unexpected connections, or "
            "important nuances that emerged from the deliberation.\n\n"
            "**RECOMMENDATION:** A clear, actionable path forward that integrates "
            "the strongest arguments from all perspectives. Be specific.\n\n"
            "**DISSENTING NOTES:** Any minority perspectives with sufficient merit "
            "that they should not be lost, even if the majority disagrees."
        )
        return "\n".join(parts)

    # ── Context Formatting ───────────────────────────────────────────────

    def _format_prior_context(
        self, perspectives: List[CouncilPerspective]
    ) -> str:
        """Format prior perspectives as context for the next round."""
        if not perspectives:
            return ""

        lines: List[str] = []
        current_round = 0

        for p in perspectives:
            if p.round != current_round:
                current_round = p.round
                lines.append(f"\n### Round {current_round}")

            lines.append(
                f"\n**{p.voice_name}** ({p.voice_type.value}) "
                f"[confidence: {p.confidence:.2f}]:"
            )
            lines.append(f"*Position:* {p.position}")
            if p.reasoning:
                # Truncate very long reasoning to keep context manageable
                reasoning = p.reasoning
                if len(reasoning) > 800:
                    reasoning = reasoning[:800] + "…"
                lines.append(f"*Reasoning:* {reasoning}")

        return "\n".join(lines)

    def _format_full_transcript(
        self,
        perspectives: List[CouncilPerspective],
        topic: str,
    ) -> str:
        """Format the complete deliberation transcript for synthesis."""
        if not perspectives:
            return "(No perspectives were contributed.)"

        lines: List[str] = []
        current_round = 0

        for p in perspectives:
            if p.round != current_round:
                current_round = p.round
                lines.append(f"\n### Round {current_round}")
                lines.append("")

            lines.append(f"#### {p.voice_name} ({p.voice_type.value})")
            lines.append(f"**Position:** {p.position}")
            lines.append(f"**Reasoning:** {p.reasoning}")
            lines.append(f"**Confidence:** {p.confidence:.2f}")
            lines.append("")

        return "\n".join(lines)

    # ── Response Parsing ─────────────────────────────────────────────────

    def _parse_voice_response(
        self, content: str, voice_name: str
    ) -> tuple[str, str, float]:
        """
        Parse a voice's LLM response into (position, reasoning, confidence).

        Attempts structured parsing first; falls back to treating the whole
        response as the position with default confidence.
        """
        position = ""
        reasoning = ""
        confidence = 0.5

        # Try to extract structured sections
        content_upper = content.upper()

        # Extract POSITION
        pos_start = self._find_section_start(content, "POSITION")
        reas_start = self._find_section_start(content, "REASONING")
        conf_start = self._find_section_start(content, "CONFIDENCE")

        if pos_start >= 0 and reas_start >= 0:
            position = content[pos_start:reas_start].strip()
            if conf_start >= 0:
                reasoning = content[reas_start:conf_start].strip()
                conf_text = content[conf_start:].strip()
                confidence = self._extract_confidence(conf_text)
            else:
                reasoning = content[reas_start:].strip()
        elif pos_start >= 0:
            position = content[pos_start:].strip()
        else:
            # Fallback: use the whole response
            # Split roughly in half for position vs reasoning
            lines = content.strip().split("\n")
            mid = max(1, len(lines) // 3)
            position = "\n".join(lines[:mid]).strip()
            reasoning = "\n".join(lines[mid:]).strip()

        # Clean up markdown bold markers from section headers
        for prefix in ["**POSITION:**", "**REASONING:**", "**CONFIDENCE:**",
                        "POSITION:", "REASONING:", "CONFIDENCE:"]:
            if position.startswith(prefix):
                position = position[len(prefix):].strip()
            if reasoning.startswith(prefix):
                reasoning = reasoning[len(prefix):].strip()

        if not position:
            position = content[:500].strip()
        if not reasoning:
            reasoning = content.strip()

        return position, reasoning, confidence

    def _find_section_start(self, content: str, section: str) -> int:
        """Find the start of a section (after the header), case-insensitive."""
        for marker in [f"**{section}:**", f"**{section}**:", f"{section}:"]:
            idx = content.upper().find(marker.upper())
            if idx >= 0:
                # Return position after the marker
                return idx + len(marker)
        return -1

    def _extract_confidence(self, text: str) -> float:
        """Extract a float confidence value from text."""
        import re
        match = re.search(r"(0\.\d+|1\.0|1\.00?)", text)
        if match:
            try:
                val = float(match.group(1))
                return max(0.0, min(1.0, val))
            except ValueError:
                pass
        return 0.5

    async def _get_session_or_raise(self, session_id: UUID) -> CouncilSession:
        """Retrieve a session or raise ValueError."""
        session = await repo.get_session(session_id)
        if not session:
            raise ValueError(f"Council session {session_id} not found")
        return session


# ── Module-level singleton ───────────────────────────────────────────────────

_council_service: Optional[CouncilService] = None


def get_council_service() -> CouncilService:
    """Get or create the global CouncilService instance."""
    global _council_service
    if _council_service is None:
        _council_service = CouncilService()
    return _council_service
