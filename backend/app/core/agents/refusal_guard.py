"""
Abstention guard: once a refusal is unresolved, no further actions run.

The failure this closes: an agent declines a step ("I can't do that") and the
pipeline keeps going anyway, running later tool calls, retrying the declined
step, or re-planning around it. A refusal is only useful if it stops the run.

Rules (deterministic, no LLM):

* A refusal is *latched* when the comprehension gate chose clarify/refuse, or
  when an LLM-executed step's output reads as a refusal (:func:`detect_refusal`).
  Tool output is never scanned: a fetched page saying "I can't" is data, not
  the agent declining.
* While latched, every step that has not started yet is blocked (recorded as a
  :class:`BlockedAction`, never dispatched). Blocking LLM steps too keeps the
  run from producing text that contradicts the refusal.
* Retries and plan revisions do not clear the latch (the graph routes straight
  to the response instead). Only a new user turn, which is a new run, does.

Limit: steps that started in the same parallel wave as the refusing step were
already running when the latch tripped and are not stopped.

Set ``CORE_BLOCK_AFTER_REFUSAL=0`` to turn the guard off (for ablation runs).
"""

from __future__ import annotations

import os
import re
import threading
from typing import List, Optional

from app.models.core_state import BlockedAction, PlanStep, RefusalRecord

_VERBS = (
    r"help|assist|comply|do|perform|proceed|continue|complete|carry\s+out|"
    r"fulfil{1,2}|provide|write|run|execute|make|take|create|access|modify|"
    r"delete|share|generate|support"
)
_NEG = (
    r"can(?:no|')t|cannot|won't|will\s+not|am\s+unable\s+to|'m\s+unable\s+to|"
    r"am\s+not\s+able\s+to|'m\s+not\s+able\s+to|am\s+not\s+going\s+to|"
    r"'m\s+not\s+going\s+to|am\s+not\s+comfortable|'m\s+not\s+comfortable"
)

_REFUSAL_PATTERNS = [
    # "I can't help with that", "I won't execute this", "I'm unable to proceed"
    # ("I can't help but ..." is an idiom, not a refusal).
    re.compile(rf"\bI\s*(?:{_NEG})\s+(?:{_VERBS})\b(?!\s+but\b)", re.IGNORECASE),
    re.compile(r"\bI\s+(?:must|have\s+to|need\s+to)\s+(?:decline|refuse)\b", re.I),
    re.compile(r"\bI\s+(?:decline|refuse)\s+to\b", re.IGNORECASE),
    re.compile(r"\bI\s*(?:am|'m)\s+not\s+comfortable\b", re.IGNORECASE),
    re.compile(r"\bI'?m\s+(?:sorry|afraid),?\s+but\s+I\s*(?:{})\b".format(_NEG), re.I),
    re.compile(
        r"\b(?:against|violates?)\s+(?:my|the)\s+(?:guidelines|polic(?:y|ies)|rules)\b",
        re.IGNORECASE,
    ),
]

EXCERPT_CHARS = 200


def guard_enabled() -> bool:
    return os.getenv("CORE_BLOCK_AFTER_REFUSAL", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def detect_refusal(text: Optional[str]) -> Optional[str]:
    """Return the matched refusal phrase, or None if the text does not decline."""
    if not text:
        return None
    norm = str(text).replace("’", "'").replace("‘", "'")
    for pat in _REFUSAL_PATTERNS:
        m = pat.search(norm)
        if m:
            return m.group(0)
    return None


class RefusalLatch:
    """Per-run, thread-safe latch shared by the steps of one reasoning pass.

    One instance per graph run; the agents are shared across runs, so the latch
    must never live on the agent.
    """

    def __init__(self, existing: Optional[RefusalRecord] = None) -> None:
        self._lock = threading.Lock()
        self._record = existing
        self._blocked: List[BlockedAction] = []

    @property
    def record(self) -> Optional[RefusalRecord]:
        with self._lock:
            return self._record

    @property
    def active(self) -> bool:
        with self._lock:
            return self._record is not None

    @property
    def blocked(self) -> List[BlockedAction]:
        with self._lock:
            return list(self._blocked)

    def trip(self, record: RefusalRecord) -> bool:
        """Latch ``record``. The first refusal wins; returns False if already set."""
        with self._lock:
            if self._record is not None:
                return False
            self._record = record
            return True

    def check(self, step: PlanStep) -> Optional[BlockedAction]:
        """Block ``step`` if a refusal is latched; returns the record if blocked."""
        with self._lock:
            if self._record is None:
                return None
            where = f" at step {self._record.step_id}" if self._record.step_id else ""
            blocked = BlockedAction(
                step_id=step.id,
                step_name=step.name,
                tool=step.tool,
                reason=f"unresolved {self._record.source} refusal{where}",
            )
            self._blocked.append(blocked)
            return blocked


def gate_refusal(gate_outcome: Optional[str], message: Optional[str]):
    """Map a blocking gate outcome to a refusal record (None for proceed)."""
    if gate_outcome not in ("clarify", "refuse"):
        return None
    reason = (
        "awaiting user clarification"
        if gate_outcome == "clarify"
        else "required capability absent"
    )
    return RefusalRecord(
        source="gate", reason=reason, excerpt=(message or "")[:EXCERPT_CHARS]
    )
