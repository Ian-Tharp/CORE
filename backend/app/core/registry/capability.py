"""
The Capability Registry and its entries.

A `CapabilityEntry` describes one thing the system can do, in an MCP-compatible
shape (id, description, JSON-Schema params, side-effect class). The
`CapabilityRegistry` holds them and offers a **lightweight lexical retrieval**
(`search`) for capability-presence checks — deliberately model-free and
deterministic so comprehension's front gate is cheap and offline-testable. An
embedding-backed retrieval is a drop-in upgrade behind the same `search` method;
lexical overlap is the Phase-1 default (cheap beats clever where it works).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

SideEffect = Literal["none", "read", "write", "network"]

# Minimal stopword set so a query of only function words matches nothing.
_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "to",
    "and",
    "or",
    "in",
    "on",
    "for",
    "with",
    "is",
    "it",
    "this",
    "that",
    "do",
    "i",
    "you",
    "me",
    "my",
    "your",
    "from",
    "at",
    "by",
    "be",
    "as",
    "are",
    "was",
    "can",
    "could",
    "would",
    "should",
    "please",
    "into",
    "out",
    "up",
    "down",
    "so",
    "if",
    "then",
    "than",
    "we",
    "us",
    "our",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    """Content tokens: lowercase alphanumerics, stopwords and 1-char dropped."""
    return {
        t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1
    }


@dataclass(frozen=True)
class CapabilityEntry:
    """One capability the system exposes (MCP-tool-compatible)."""

    id: str
    name: str
    description: str
    params_schema: Dict = field(default_factory=dict)
    side_effects: SideEffect = "none"
    source: str = "builtin"  # "builtin" | "mcp:<server>"
    examples: List[str] = field(default_factory=list)

    def _index_text(self) -> str:
        # id split on . and _ contributes tokens too (file_operations.read → file operations read)
        parts = [
            self.id.replace(".", " ").replace("_", " "),
            self.name,
            self.description,
        ]
        parts.extend(self.examples)
        return " ".join(parts)


class CapabilityRegistry:
    """Holds capability entries and answers presence/retrieval queries."""

    def __init__(self) -> None:
        self._entries: Dict[str, CapabilityEntry] = {}

    def register(self, entry: CapabilityEntry) -> None:
        if entry.id in self._entries:
            raise ValueError(f"duplicate capability id: {entry.id}")
        self._entries[entry.id] = entry

    def get(self, capability_id: str) -> Optional[CapabilityEntry]:
        return self._entries.get(capability_id)

    def all(self) -> List[CapabilityEntry]:
        return list(self._entries.values())

    def search(self, query: str, k: int = 5) -> List[CapabilityEntry]:
        """Return up to `k` entries by content-word overlap, best first.

        Entries with zero overlap are excluded — an empty result is the
        capability-*absence* signal the front gate uses to refuse honestly.
        """
        q = _tokens(query)
        if not q:
            return []
        scored = []
        for entry in self._entries.values():
            overlap = len(q & _tokens(entry._index_text()))
            if overlap > 0:
                scored.append((overlap, entry.id, entry))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [entry for _, _, entry in scored[:k]]

    def has_capability(self, query: str) -> bool:
        """True if at least one capability plausibly covers the query."""
        return bool(self.search(query, k=1))
