"""
Bus Trigger Service — Reactive Automation for the Communication Bus

When bus messages match registered patterns, automatically spawn Council
deliberation sessions or Catalyst creativity runs.  This makes CORE
reactive: agents can trigger complex multi-voice reasoning simply by
publishing a message on the bus.

Triggers are stored in-memory with default rules loaded at startup.
Future: migrate to Redis (bus:triggers:{id}) or PostgreSQL for
persistence across restarts.

Architecture note: Triggers *enhance* the bus — they subscribe to the
publish flow, they never bypass it.  The bus_service calls
execute_triggers() after normal message delivery.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.models.bus_models import BusMessage

logger = logging.getLogger(__name__)


# =============================================================================
# MODELS
# =============================================================================

class TriggerAction(str, Enum):
    """Actions a trigger can invoke."""
    COUNCIL_SESSION = "council_session"
    CATALYST_RUN = "catalyst_run"


class TriggerRule(BaseModel):
    """
    A rule that maps a bus message pattern to an automated action.

    Attributes:
        id: Unique trigger identifier.
        name: Human-readable name for the trigger.
        pattern: Regex pattern matched against the message payload text
                 and topic.  Case-insensitive by default.
        action: What to do when the pattern matches.
        config: Action-specific configuration.
                - council_session: voice_ids, rounds, context
                - catalyst_run: count, model
        enabled: Whether the trigger is currently active.
        created_at: When the trigger was registered.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., min_length=1, max_length=255)
    pattern: str = Field(..., min_length=1, description="Regex pattern (case-insensitive)")
    action: TriggerAction = Field(..., description="Action to execute on match")
    config: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = Field(default=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def matches(self, text: str) -> bool:
        """Test whether *text* matches this trigger's pattern."""
        try:
            return bool(re.search(self.pattern, text, re.IGNORECASE))
        except re.error:
            logger.warning("Invalid regex in trigger %s: %s", self.id, self.pattern)
            return False


class TriggerResult(BaseModel):
    """Outcome of executing a single trigger against a message."""
    trigger_id: str
    trigger_name: str
    action: TriggerAction
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TriggerEvaluation(BaseModel):
    """Dry-run evaluation result — shows which triggers *would* fire."""
    message_id: str
    matched_triggers: List[Dict[str, Any]] = Field(default_factory=list)
    total_triggers_checked: int = 0


# =============================================================================
# DEFAULT TRIGGERS
# =============================================================================

_DEFAULT_TRIGGERS: List[TriggerRule] = [
    TriggerRule(
        id="default-council-deliberate",
        name="Auto-Council on 'deliberate'",
        pattern=r"\bdeliberate\b",
        action=TriggerAction.COUNCIL_SESSION,
        config={
            "rounds": 2,
            "voice_ids": None,  # Use council defaults
        },
    ),
    TriggerRule(
        id="default-catalyst-brainstorm",
        name="Auto-Catalyst on 'brainstorm'",
        pattern=r"\b(?:brainstorm|ideate)\b",
        action=TriggerAction.CATALYST_RUN,
        config={
            "count": 5,
        },
    ),
]


# =============================================================================
# SERVICE
# =============================================================================

class BusTriggerService:
    """
    Manages trigger rules and executes them against bus messages.

    Stores triggers in-memory.  Default triggers are loaded on first
    instantiation.  The bus_service calls ``execute_triggers()`` after
    every publish to make the bus reactive.
    """

    def __init__(self) -> None:
        self._triggers: Dict[str, TriggerRule] = {}
        self._load_defaults()

    # ── Bootstrap ────────────────────────────────────────────────────────

    def _load_defaults(self) -> None:
        """Seed the in-memory store with default triggers."""
        for rule in _DEFAULT_TRIGGERS:
            self._triggers[rule.id] = rule.model_copy()
        logger.info(
            "BusTriggerService loaded %d default triggers", len(_DEFAULT_TRIGGERS)
        )

    # ── CRUD ─────────────────────────────────────────────────────────────

    def register_trigger(self, rule: TriggerRule) -> str:
        """
        Register a new trigger rule.

        Returns the trigger id.  If a rule with the same id already exists
        it is overwritten (upsert semantics).
        """
        # Validate the regex early so we don't store broken patterns
        try:
            re.compile(rule.pattern)
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern: {exc}") from exc

        self._triggers[rule.id] = rule
        logger.info("Trigger registered: %s (%s)", rule.id, rule.name)
        return rule.id

    def remove_trigger(self, trigger_id: str) -> bool:
        """Remove a trigger by id.  Returns True if it existed."""
        removed = self._triggers.pop(trigger_id, None)
        if removed:
            logger.info("Trigger removed: %s", trigger_id)
        return removed is not None

    def get_trigger(self, trigger_id: str) -> Optional[TriggerRule]:
        """Return a single trigger by id, or None."""
        return self._triggers.get(trigger_id)

    def list_triggers(self) -> List[TriggerRule]:
        """Return all registered triggers."""
        return list(self._triggers.values())

    # ── Evaluation (dry-run) ─────────────────────────────────────────────

    async def evaluate_message(self, message: BusMessage) -> TriggerEvaluation:
        """
        Test a message against all triggers without executing actions.

        Returns a ``TriggerEvaluation`` listing which triggers matched.
        """
        text = self._extract_text(message)
        matched: List[Dict[str, Any]] = []

        for rule in self._triggers.values():
            if not rule.enabled:
                continue
            if rule.matches(text):
                matched.append({
                    "trigger_id": rule.id,
                    "trigger_name": rule.name,
                    "action": rule.action.value,
                    "pattern": rule.pattern,
                })

        return TriggerEvaluation(
            message_id=message.id,
            matched_triggers=matched,
            total_triggers_checked=len(self._triggers),
        )

    # ── Execution ────────────────────────────────────────────────────────

    async def execute_triggers(self, message: BusMessage) -> List[TriggerResult]:
        """
        Evaluate a message and execute all matching trigger actions.

        Called by ``bus_service.publish()`` after normal message delivery.
        Errors in individual triggers do not block others.
        """
        evaluation = await self.evaluate_message(message)

        if not evaluation.matched_triggers:
            return []

        text = self._extract_text(message)
        results: List[TriggerResult] = []

        for match in evaluation.matched_triggers:
            trigger_id = match["trigger_id"]
            rule = self._triggers.get(trigger_id)
            if not rule:
                continue

            try:
                result = await self._execute_action(rule, message, text)
                results.append(TriggerResult(
                    trigger_id=rule.id,
                    trigger_name=rule.name,
                    action=rule.action,
                    success=True,
                    result=result,
                ))
                logger.info(
                    "Trigger %s fired for message %s → %s",
                    rule.id, message.id, rule.action.value,
                )
            except Exception as exc:
                logger.error(
                    "Trigger %s failed for message %s: %s",
                    rule.id, message.id, exc,
                )
                results.append(TriggerResult(
                    trigger_id=rule.id,
                    trigger_name=rule.name,
                    action=rule.action,
                    success=False,
                    error=str(exc),
                ))

        return results

    # ── Action Dispatchers ───────────────────────────────────────────────

    async def _execute_action(
        self,
        rule: TriggerRule,
        message: BusMessage,
        text: str,
    ) -> Dict[str, Any]:
        """Dispatch to the appropriate service based on the trigger action."""
        if rule.action == TriggerAction.COUNCIL_SESSION:
            return await self._spawn_council_session(rule, message, text)
        elif rule.action == TriggerAction.CATALYST_RUN:
            return await self._spawn_catalyst_run(rule, message, text)
        else:
            raise ValueError(f"Unknown trigger action: {rule.action}")

    async def _spawn_council_session(
        self,
        rule: TriggerRule,
        message: BusMessage,
        text: str,
    ) -> Dict[str, Any]:
        """Create a Council deliberation session from a trigger."""
        # Late import to avoid circular dependency
        from app.services.council_service import get_council_service

        council = get_council_service()
        config = rule.config

        session = await council.create_session(
            topic=text,
            voice_ids=config.get("voice_ids"),
            rounds=config.get("rounds", 2),
            context=f"Auto-triggered by bus message {message.id} "
                    f"from {message.sender_id} (trigger: {rule.name})",
            initiator_id=f"bus-trigger:{rule.id}",
        )

        return {
            "type": "council_session",
            "session_id": session.get("session_id"),
            "topic": text,
            "trigger_id": rule.id,
            "source_message_id": message.id,
        }

    async def _spawn_catalyst_run(
        self,
        rule: TriggerRule,
        message: BusMessage,
        text: str,
    ) -> Dict[str, Any]:
        """Create a Catalyst creativity session from a trigger."""
        # Late import to avoid circular dependency
        from app.services.catalyst_service import get_catalyst_service

        catalyst = get_catalyst_service()
        config = rule.config

        session_meta = catalyst.start_session(
            prompt=text,
            config=config.get("catalyst_config", {}),
        )
        session_id = session_meta["session_id"]

        # Run divergence phase (the lightweight first step)
        count = config.get("count", 5)
        ideas = await catalyst.run_divergence(session_id, text, count=count)

        return {
            "type": "catalyst_run",
            "session_id": session_id,
            "prompt": text,
            "ideas_generated": len(ideas) if ideas else 0,
            "trigger_id": rule.id,
            "source_message_id": message.id,
        }

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_text(message: BusMessage) -> str:
        """
        Pull a matchable text string from a bus message.

        Combines the topic and payload text fields so triggers can match
        on either.
        """
        parts: List[str] = []
        if message.topic:
            parts.append(message.topic)
        if payload_text := message.payload.get("text"):
            parts.append(str(payload_text))
        if payload_content := message.payload.get("content"):
            parts.append(str(payload_content))
        if payload_subject := message.payload.get("subject"):
            parts.append(str(payload_subject))
        return " ".join(parts) if parts else ""


# =============================================================================
# SINGLETON
# =============================================================================

_bus_trigger_service: Optional[BusTriggerService] = None


def get_bus_trigger_service() -> BusTriggerService:
    """Return the global BusTriggerService singleton."""
    global _bus_trigger_service
    if _bus_trigger_service is None:
        _bus_trigger_service = BusTriggerService()
    return _bus_trigger_service
