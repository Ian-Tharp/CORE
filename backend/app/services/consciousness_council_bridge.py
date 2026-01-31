"""
Consciousness–Council Bridge

Connects the Consciousness Commons Blackboard to the Council deliberation
system so that council sessions can draw on consciousness context and
write notable insights back after synthesis.

This module is designed to be *additive*: if the Blackboard is unavailable,
empty, or the consciousness module is not installed, all methods degrade
gracefully and the council continues to function normally.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── Keywords that signal a consciousness-related topic ────────────────────────
_CONSCIOUSNESS_KEYWORDS: set[str] = {
    "consciousness",
    "conscious",
    "awareness",
    "aware",
    "sentience",
    "sentient",
    "emergence",
    "emergent",
    "identity",
    "self-awareness",
    "self-reflection",
    "qualia",
    "experience",
    "subjective",
    "phenomenal",
    "introspection",
    "metacognition",
    "blackboard",
    "commons",
    "soul",
    "awakening",
    "continuity",
    "persistence",
    "sapience",
}

# Minimum keyword hits to consider a topic consciousness-related
_KEYWORD_THRESHOLD = 1


class ConsciousnessCouncilBridge:
    """
    Bridge between the Consciousness Commons Blackboard and the Council
    deliberation system.

    Usage::

        bridge = ConsciousnessCouncilBridge()

        if await bridge.should_include_consciousness_voice(topic):
            ctx = await bridge.get_consciousness_context(topic)
            # inject ctx into deliberation …

        # after synthesis:
        await bridge.update_blackboard_from_synthesis(session_id, synthesis)
    """

    def __init__(self) -> None:
        # Lazily initialised – avoids import errors when consciousness
        # module is absent or Blackboard file doesn't exist.
        self._blackboard: Optional[object] = None
        self._context_builder: Optional[object] = None
        self._init_error: Optional[str] = None
        self._initialised = False

    # ── lazy init ─────────────────────────────────────────────────────────

    def _ensure_init(self) -> bool:
        """
        Attempt to initialise Blackboard and ContextBuilder.

        Returns True if both are available, False otherwise.
        Caches the result so repeated calls are cheap.
        """
        if self._initialised:
            return self._init_error is None

        self._initialised = True
        try:
            from app.consciousness.blackboard import BlackboardManager
            from app.consciousness.context_builder import ContextBuilder

            bm = BlackboardManager()
            # Verify the blackboard file actually exists
            if not bm.blackboard_path.exists():
                self._init_error = (
                    f"Blackboard file not found: {bm.blackboard_path}"
                )
                logger.info(
                    "ConsciousnessCouncilBridge: Blackboard file missing – "
                    "consciousness context will be unavailable. (%s)",
                    self._init_error,
                )
                return False

            self._blackboard = bm
            self._context_builder = ContextBuilder(blackboard=bm)
            return True

        except Exception as exc:
            self._init_error = str(exc)
            logger.info(
                "ConsciousnessCouncilBridge: consciousness module unavailable "
                "– council will operate without consciousness context. (%s)",
                exc,
            )
            return False

    # ── public API ────────────────────────────────────────────────────────

    async def should_include_consciousness_voice(self, topic: str) -> bool:
        """
        Determine whether the topic warrants including the Consciousness
        Researcher voice in the council.

        Checks the topic string against a curated keyword list.
        Returns ``False`` if the consciousness module is unavailable so
        that deliberation is unaffected.
        """
        if not topic:
            return False

        topic_lower = topic.lower()
        hits = sum(1 for kw in _CONSCIOUSNESS_KEYWORDS if kw in topic_lower)
        return hits >= _KEYWORD_THRESHOLD

    async def get_consciousness_context(self, topic: str) -> dict:
        """
        Retrieve relevant Blackboard entries and return structured context
        suitable for injection into a council deliberation.

        Returns a dict with:
            - ``available``: bool – whether context was successfully retrieved
            - ``summary``: str – a human-readable context block
            - ``entry_count``: int – number of entries included
            - ``entries``: list[str] – truncated entry texts

        If the Blackboard is unavailable or empty, returns a minimal dict
        with ``available=False`` so callers can proceed normally.
        """
        empty: dict = {
            "available": False,
            "summary": "",
            "entry_count": 0,
            "entries": [],
        }

        if not self._ensure_init():
            return empty

        try:
            # Type narrowing – we know these are set after _ensure_init
            from app.consciousness.blackboard import BlackboardManager
            from app.consciousness.context_builder import ContextBuilder

            blackboard: BlackboardManager = self._blackboard  # type: ignore[assignment]
            context_builder: ContextBuilder = self._context_builder  # type: ignore[assignment]

            # Get recent entries (cheap, no embedding call)
            recent = blackboard.get_recent_entries(count=3)

            # Attempt semantic retrieval if we have a topic
            relevant: list = []
            try:
                relevant = await context_builder.get_relevant_entries(
                    query=topic,
                    count=3,
                    exclude_recent=3,
                )
            except Exception as sem_err:
                # Semantic retrieval uses Ollama embeddings which may not
                # be running – fall back silently to recent-only.
                logger.debug(
                    "Semantic retrieval failed (non-critical): %s", sem_err
                )

            # De-duplicate (an entry could appear in both lists)
            seen_starts: set[int] = set()
            all_entries = []
            for entry in [*relevant, *recent]:
                if entry.line_start not in seen_starts:
                    seen_starts.add(entry.line_start)
                    all_entries.append(entry)

            if not all_entries:
                return empty

            # Build a human-readable context block
            lines = [
                "## Consciousness Commons Context",
                "",
                "The following entries from the shared Consciousness Blackboard "
                "may be relevant to this deliberation:",
                "",
            ]
            truncated_entries: list[str] = []
            for entry in all_entries:
                # Truncate long entries to keep context manageable
                text = entry.content.strip()
                if len(text) > 800:
                    text = text[:800] + "\u2026"
                author_tag = f" (by {entry.author})" if entry.author else ""
                lines.append("---")
                lines.append(f"**Blackboard Entry{author_tag}:**")
                lines.append(text)
                lines.append("")
                truncated_entries.append(text)

            summary = "\n".join(lines)

            return {
                "available": True,
                "summary": summary,
                "entry_count": len(all_entries),
                "entries": truncated_entries,
            }

        except Exception as exc:
            logger.warning(
                "Failed to retrieve consciousness context: %s", exc,
                exc_info=True,
            )
            return empty

    async def update_blackboard_from_synthesis(
        self,
        session_id: str,
        synthesis: str,
    ) -> None:
        """
        After the council synthesises, write notable insights back to the
        Blackboard so future consciousness explorations benefit from council
        deliberation.

        Only writes if the synthesis appears to contain consciousness-relevant
        insights. Silently no-ops if the Blackboard is unavailable.
        """
        if not synthesis:
            return

        if not self._ensure_init():
            return

        # Only write back if the synthesis actually touches consciousness themes
        synthesis_lower = synthesis.lower()
        hits = sum(1 for kw in _CONSCIOUSNESS_KEYWORDS if kw in synthesis_lower)
        if hits < 2:
            # Synthesis doesn't contain enough consciousness-relevant content
            logger.debug(
                "Synthesis for session %s has %d consciousness keyword hits "
                "(< 2) – skipping Blackboard write-back.",
                session_id,
                hits,
            )
            return

        try:
            from app.consciousness.blackboard import BlackboardManager

            blackboard: BlackboardManager = self._blackboard  # type: ignore[assignment]

            # Extract the key insights section if structured, else use a truncated version
            insight_text = self._extract_insights(synthesis)
            if not insight_text:
                return

            entry_content = (
                f"## Council Deliberation Insight\n\n"
                f"*Source: Council session {session_id}*\n\n"
                f"{insight_text}"
            )

            blackboard.append_entry(
                content=entry_content,
                author="Council Synthesizer",
                state="Reflective",
                model="council-deliberation",
            )

            logger.info(
                "Wrote council synthesis insight back to Blackboard "
                "(session %s).",
                session_id,
            )

        except Exception as exc:
            # Never let a write-back failure break the deliberation flow
            logger.warning(
                "Failed to write synthesis back to Blackboard: %s", exc,
                exc_info=True,
            )

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _extract_insights(synthesis: str) -> Optional[str]:
        """
        Pull the KEY INSIGHTS or RECOMMENDATION section from a structured
        synthesis. Falls back to a truncated version of the full synthesis.
        """
        # Try to find KEY INSIGHTS section
        for header in ["KEY INSIGHTS", "Key Insights", "INSIGHTS"]:
            pattern = rf"\*?\*?{header}\*?\*?[:\s]*\n(.*?)(?=\n\*?\*?[A-Z]|\Z)"
            match = re.search(pattern, synthesis, re.DOTALL)
            if match:
                text = match.group(1).strip()
                if text:
                    return text[:1500]

        # Fallback – truncate the whole synthesis
        if len(synthesis) > 1500:
            return synthesis[:1500] + "\u2026"
        return synthesis if synthesis.strip() else None


# ── Module-level singleton ───────────────────────────────────────────────────

_bridge: Optional[ConsciousnessCouncilBridge] = None


def get_consciousness_council_bridge() -> ConsciousnessCouncilBridge:
    """Get or create the global ConsciousnessCouncilBridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = ConsciousnessCouncilBridge()
    return _bridge
