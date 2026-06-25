"""
Pure routing decision after comprehension.

Kept as a standalone, model-free function so the graph's conditional edge can be
unit-tested without constructing the whole pipeline. The tri-state gate has
already run; this only maps its outcome (+ intent type) to the next node:

- gate `clarify` / `refuse` → `conversation` (the gate BLOCKS; surface its message)
- otherwise (gate `proceed`, or the gate didn't run) → route by intent type:
  actionable (task/question) → `orchestration`; everything else → `conversation`

The gate only *blocks*; it never forces the loop. If the gate hasn't run
(`gate_outcome is None`, e.g. a comprehension error or a direct unit call), we fall
back to intent-type routing rather than silently sending everything to chat.
"""

from __future__ import annotations

from typing import Literal, Optional

from app.models.core_state import UserIntent

NextNode = Literal["orchestration", "conversation"]


def route_after_comprehension(
    intent: Optional[UserIntent], gate_outcome: Optional[str]
) -> NextNode:
    if intent is None:
        return "conversation"
    # The gate blocks only on an explicit clarify/refuse.
    if gate_outcome in ("clarify", "refuse"):
        return "conversation"
    # proceed, or gate didn't run → route by intent type.
    if intent.type in ("task", "question"):
        return "orchestration"
    return "conversation"
