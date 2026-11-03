"""
Memory grounding for comprehension (spec Phase 3, §4.2(b) and §4.5).

Given a request, return the known facts/episodes relevant to it, with scores,
and flag **knowledge gaps**: the request leans on something CORE should already
know ("like last time", "my preferred stack", "as we discussed") but memory has
nothing that covers it. Gaps become ambiguities on the `UserIntent`, so the
existing tri-state gate turns them into a clarify, with no new gate logic.

Design constraints (same as comprehension_gate.py and registry/grounding.py):
- Small interface: `MemoryGrounder.ground(query, user_id=..., k=...)`.
- Synchronous and model-free by default. The comprehension node is sync, and
  the front gate must stay cheap and deterministic.
- Never raises into the caller's flow: the graph wraps calls, and the store
  grounder swallows backend errors and returns no hits.
- Off unless `CORE_COMPREHENSION_MEMORY_GROUNDING` is set, so the default
  pipeline is unchanged.

Default backend is LangGraph's `BaseStore` (`InMemoryStore` in tests,
`PostgresStore` when `CORE_MEMORY_STORE=postgres` and the package that provides
it is installed). Items live under `("core_memory", <user_id|"global">, <kind>)`
with a `{"content": str, ...}` value. When the store has a semantic index the
item score is used; otherwise (the default: no embedder call) a lexical overlap
score is computed here, so behaviour is identical offline and in tests.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Protocol, Tuple
from uuid import uuid4

from app.core.registry.capability import _tokens

logger = logging.getLogger(__name__)

MemoryKind = Literal["fact", "episode"]

NAMESPACE_ROOT = "core_memory"
GLOBAL_SCOPE = "global"

_FLAG_ENV = "CORE_COMPREHENSION_MEMORY_GROUNDING"
_TRUTHY = {"1", "true", "yes", "on"}


def memory_grounding_enabled() -> bool:
    """True when comprehension should consult memory. Read at call time so the
    flag can be flipped without rebuilding the graph (and patched in tests)."""
    return os.getenv(_FLAG_ENV, "").strip().lower() in _TRUTHY


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

# Words that signal "you should already know this" but carry no topic. They are
# removed from the scoring query so "like last time" doesn't match every episode
# that happens to contain "last".
_CUE_WORDS = {
    "last",
    "time",
    "before",
    "usual",
    "usually",
    "again",
    "previous",
    "previously",
    "earlier",
    "remember",
    "recall",
    "discussed",
    "talked",
    "about",
    "agreed",
    "decided",
    "said",
    "told",
    "mentioned",
    "suggested",
    "recommended",
    "like",
    "same",
    "way",
    "favorite",
    "favourite",
    "preferred",
    "regular",
    "default",
}


def _stem(token: str) -> str:
    """Crude, deterministic suffix strip so 'deployed'/'deploys' meet 'deploy'."""
    for suffix, min_len in (("ing", 6), ("ed", 5), ("es", 5), ("s", 4)):
        if token.endswith(suffix) and len(token) >= min_len:
            return token[: -len(suffix)]
    return token


def _content_tokens(text: str) -> set:
    return {_stem(t) for t in _tokens(text or "")}


def _query_tokens(text: str) -> set:
    return {t for t in _content_tokens(text) if t not in _CUE_WORDS}


# ---------------------------------------------------------------------------
# Knowledge-dependency cues
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnowledgeCue:
    """A phrase showing the request relies on prior knowledge.

    `topic` holds the content tokens the knowledge must cover (e.g. "stack" in
    "my preferred stack"). An empty topic means any relevant memory covers it.
    """

    phrase: str
    topic: frozenset = frozenset()


_TOPIC = r"((?:[a-z0-9_+.#-]+\s+){0,1}[a-z0-9_+.#-]+)"

# Strong cues only. A bare "my"/"our" is NOT a cue: "fix my code in main.py" is
# self-contained, and treating it as a gap would make the gate over-conservative.
_CUE_PATTERNS: List[Tuple[str, bool]] = [
    # (pattern, has_topic_group)
    (r"\b(?:like|as|same as)\s+(?:last\s+time|before|usual|previously)\b", False),
    (r"\bthe\s+usual\b", False),
    (r"\b(?:the\s+)?same\s+(?:way|as\s+last\s+time)\b", False),
    (r"\blast\s+time\b", False),
    (
        r"\b(?:we|you|i)\s+(?:already\s+)?(?:discussed|talked\s+about|agreed(?:\s+on)?"
        r"|decided(?:\s+on)?|mentioned|told\s+you(?:\s+about)?)\s+" + _TOPIC,
        True,
    ),
    (r"\bas\s+(?:we|you|i)\s+(?:discussed|agreed|decided|said|mentioned)\b", False),
    (r"\byou\s+(?:told|said|mentioned|suggested|recommended)\b", False),
    (r"\b(?:remember|recall)\s+(?:my|our|the|what|when|that|how)\b", False),
    (
        r"\b(?:my|our)\s+(?:usual|favou?rite|preferred|regular|default)\s+" + _TOPIC,
        True,
    ),
    (
        r"\b(?:my|our)\s+(name|birthday|address|email|timezone|time\s+zone"
        r"|preferences?|team|manager|boss|company|employer|stack|setup|project)\b",
        True,
    ),
]
_COMPILED_CUES = [(re.compile(p, re.IGNORECASE), has) for p, has in _CUE_PATTERNS]


def detect_knowledge_dependencies(text: str) -> List[KnowledgeCue]:
    """Return the cues in `text` that point at knowledge CORE should hold.

    Deterministic and order-preserving; duplicate phrases are dropped.
    """
    if not text or not text.strip():
        return []
    lowered = text.lower()
    cues: List[KnowledgeCue] = []
    seen: set = set()
    for pattern, has_topic in _COMPILED_CUES:
        for match in pattern.finditer(lowered):
            phrase = match.group(0).strip()
            if phrase in seen or any(phrase in s for s in seen):
                continue
            topic: frozenset = frozenset()
            if has_topic and match.lastindex:
                topic = frozenset(_query_tokens(match.group(match.lastindex)))
            seen.add(phrase)
            cues.append(KnowledgeCue(phrase=phrase, topic=topic))
    return cues


# ---------------------------------------------------------------------------
# The interface and its implementations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryHit:
    """One known fact or episode relevant to a request."""

    key: str
    kind: str
    content: str
    score: float
    namespace: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "content": self.content,
            "score": round(self.score, 4),
            "namespace": list(self.namespace),
        }


class MemoryGrounder(Protocol):
    """Given a request, return relevant known facts/episodes, best first."""

    def ground(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        k: int = 5,
    ) -> List[MemoryHit]: ...


class NullMemoryGrounder:
    """Knows nothing. Used when no store is configured."""

    def ground(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        k: int = 5,
    ) -> List[MemoryHit]:
        return []


class StoreMemoryGrounder:
    """`MemoryGrounder` over a LangGraph `BaseStore`.

    Reads the user's namespace and the global one. Scores are the store's own
    (semantic index) when present, otherwise lexical overlap: the share of the
    request's content words (cue words excluded) found in the item.
    """

    def __init__(
        self,
        store: Any,
        *,
        namespace_root: str = NAMESPACE_ROOT,
        min_score: float = 0.2,
        scan_limit: int = 200,
    ) -> None:
        self.store = store
        self.namespace_root = namespace_root
        self.min_score = min_score
        self.scan_limit = scan_limit

    # -- writes (used by tests today; by a consolidation node later) ---------

    def remember(
        self,
        content: str,
        *,
        kind: MemoryKind = "fact",
        user_id: Optional[str] = None,
        key: Optional[str] = None,
        **metadata: Any,
    ) -> str:
        """Store a fact or episode and return its key."""
        key = key or str(uuid4())
        value = {"content": content, "kind": kind, **metadata}
        self.store.put(self._namespace(user_id, kind), key, value)
        return key

    # -- reads ---------------------------------------------------------------

    def ground(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        k: int = 5,
    ) -> List[MemoryHit]:
        q = _query_tokens(query)
        if not q or k <= 0:
            return []

        best: Dict[Tuple[Tuple[str, ...], str], MemoryHit] = {}
        for prefix in self._prefixes(user_id):
            for item in self._search(prefix, query):
                content = str((item.value or {}).get("content", ""))
                if not content:
                    continue
                lexical = len(q & _content_tokens(content)) / len(q)
                store_score = getattr(item, "score", None)
                score = max(lexical, float(store_score or 0.0))
                if score < self.min_score:
                    continue
                namespace = tuple(item.namespace)
                kind = str((item.value or {}).get("kind") or namespace[-1])
                best[(namespace, item.key)] = MemoryHit(
                    key=item.key,
                    kind=kind,
                    content=content,
                    score=score,
                    namespace=namespace,
                )

        hits = sorted(best.values(), key=lambda h: (-h.score, h.key))
        return hits[:k]

    def _namespace(self, user_id: Optional[str], kind: str) -> Tuple[str, ...]:
        return (self.namespace_root, user_id or GLOBAL_SCOPE, kind)

    def _prefixes(self, user_id: Optional[str]) -> List[Tuple[str, ...]]:
        prefixes = [(self.namespace_root, GLOBAL_SCOPE)]
        if user_id and user_id != GLOBAL_SCOPE:
            prefixes.insert(0, (self.namespace_root, user_id))
        return prefixes

    def _search(self, prefix: Tuple[str, ...], query: str) -> Iterable[Any]:
        try:
            return self.store.search(prefix, query=query, limit=self.scan_limit)
        except Exception as exc:  # a broken store means "knows nothing"
            logger.debug("memory store search failed (non-critical): %s", exc)
            return []


# ---------------------------------------------------------------------------
# Assessment: hits + cues → gaps
# ---------------------------------------------------------------------------


@dataclass
class KnowledgeAssessment:
    """What memory says about one request."""

    hits: List[MemoryHit] = field(default_factory=list)
    cues: List[KnowledgeCue] = field(default_factory=list)
    gaps: List[KnowledgeCue] = field(default_factory=list)

    @property
    def has_gaps(self) -> bool:
        return bool(self.gaps)

    def gap_questions(self) -> List[str]:
        """One clarify question per gap, phrased for the user."""
        return [
            f"You referred to '{gap.phrase}', but I don't have that in memory. "
            "Could you fill me in?"
            for gap in self.gaps
        ]

    def context_block(self, max_chars: int = 600) -> str:
        """Short summary of the relevant memory, for the comprehension context."""
        if not self.hits:
            return ""
        lines = ["Known from memory:"]
        used = len(lines[0])
        for hit in self.hits:
            line = f"- [{hit.kind}] {hit.content}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines) if len(lines) > 1 else ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": [h.to_dict() for h in self.hits],
            "cues": [c.phrase for c in self.cues],
            "gaps": [g.phrase for g in self.gaps],
        }


def _covers(cue: KnowledgeCue, hits: List[MemoryHit]) -> bool:
    if not hits:
        return False
    if not cue.topic:
        return True
    return any(cue.topic & _content_tokens(h.content) for h in hits)


def assess_knowledge(
    query: str,
    grounder: MemoryGrounder,
    *,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    k: int = 5,
) -> KnowledgeAssessment:
    """Ground `query` in memory and list the cues memory does not cover."""
    cues = detect_knowledge_dependencies(query)
    hits = grounder.ground(query, user_id=user_id, conversation_id=conversation_id, k=k)
    gaps = [cue for cue in cues if not _covers(cue, hits)]
    return KnowledgeAssessment(hits=hits, cues=cues, gaps=gaps)


def apply_to_intent(intent: Any, assessment: KnowledgeAssessment) -> None:
    """Feed an assessment into a `UserIntent` in place.

    Gaps become ambiguities (actionable intents only: the gate ignores the
    rest), so the existing gate clarifies. Relevant memory is appended to
    `context_retrieved` for the downstream nodes.
    """
    if intent is None:
        return
    if intent.type in ("task", "question"):
        for question in assessment.gap_questions():
            if question not in intent.ambiguities:
                intent.ambiguities.append(question)
    block = assessment.context_block()
    if block:
        intent.context_retrieved = (
            f"{intent.context_retrieved}\n\n{block}"
            if intent.context_retrieved
            else block
        )


# ---------------------------------------------------------------------------
# Store construction
# ---------------------------------------------------------------------------


def build_memory_store() -> Any:
    """Build the LangGraph store selected by `CORE_MEMORY_STORE`.

    - `memory` (default): `InMemoryStore`, empty at start.
    - `postgres`: `PostgresStore` on `CORE_MEMORY_STORE_DSN` (or `DATABASE_URL`).
      It ships in `langgraph-checkpoint-postgres`; if that is not installed or
      the database is unreachable, this logs a warning and falls back to an
      in-memory store rather than failing the request.
    """
    from langgraph.store.memory import InMemoryStore

    backend = os.getenv("CORE_MEMORY_STORE", "memory").strip().lower()
    if backend != "postgres":
        return InMemoryStore()

    dsn = os.getenv("CORE_MEMORY_STORE_DSN") or os.getenv("DATABASE_URL")
    if not dsn:
        logger.warning("CORE_MEMORY_STORE=postgres but no DSN set; using in-memory")
        return InMemoryStore()
    try:
        from langgraph.store.postgres import PostgresStore  # optional package
        from psycopg import Connection
        from psycopg.rows import dict_row

        conn = Connection.connect(
            dsn, autocommit=True, prepare_threshold=0, row_factory=dict_row
        )
        store = PostgresStore(conn)
        store.setup()
        return store
    except Exception as exc:
        logger.warning("PostgresStore unavailable (%s); using in-memory store", exc)
        return InMemoryStore()
