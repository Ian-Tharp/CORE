"""
Comprehension Agent - Understands user intent and retrieves relevant context.

Determines:
- Is this a task, conversation, question, or clarification request?
- What tools/capabilities are needed?
- What context from the knowledge base is relevant?
- Are there ambiguities that need clarification?
"""

import json
import logging
import os
import re
from typing import TYPE_CHECKING, Optional

import httpx

from app.dependencies import get_openai_client_sync
from app.models.core_state import UserIntent
from app.utils.json_repair import safe_json_loads, extract_json_object

if TYPE_CHECKING:  # pragma: no cover
    from app.core.registry.capability import CapabilityRegistry

logger = logging.getLogger(__name__)

# Internal CORE API base URL (same process; uses REST to avoid event-loop issues
# since the LangGraph graph runs in asyncio.to_thread and can't await the async pool).
_CORE_BASE_URL = os.getenv("CORE_BASE_URL", "http://localhost:8001")
# Minimum cosine similarity (1 - cosine_distance) for a KB result to be included.
_KB_MIN_SIMILARITY = 0.30


class ComprehensionAgent:
    """
    Comprehension Agent - First stage of CORE pipeline.

    Analyzes user input to determine intent type and required capabilities.
    RAG: queries the CORE knowledge base before LLM classification so the
    model can ground its answer in relevant documents.
    """

    def __init__(
        self,
        model: str = "gpt-oss:20b",
        registry: Optional["CapabilityRegistry"] = None,
    ):
        # Default to local model for offline-first operation
        # Override with gpt-4o-mini or claude-3-5-haiku-latest for cloud
        self.model = model
        # The tool list in the prompt is built from the capability registry, so
        # the classifier only hears about actions the dispatcher can run.
        # Without one, use every builtin capability.
        if registry is None:
            from types import SimpleNamespace

            from app.core.registry.builtin import (
                _BUILTIN,
                build_registry_from_dispatcher,
            )

            registry = build_registry_from_dispatcher(
                SimpleNamespace(available_tools=list(_BUILTIN))
            )
        self.registry = registry
        self.system_prompt = self._build_system_prompt()

    def _tools_section(self) -> str:
        """Tool list for the prompt, one line per root, from the registry."""
        lines = []
        for root in sorted(self.registry.roots()):
            entries = sorted(self.registry.for_root(root), key=lambda e: e.id)
            actions = "; ".join(f"{e.id} ({e.description})" for e in entries)
            lines.append(f"- {root}: {actions}")
        return "\n".join(lines) or "- (none)"

    def _build_system_prompt(self) -> str:
        """Build the base system prompt for intent classification."""
        return (
            """You are the Comprehension layer of the CORE cognitive system.

Your job is to analyze user input and determine:
1. Intent type: task, conversation, question, or clarification
2. Whether tools are needed (file operations, git, database, web search, etc.)
3. Confidence in your classification
4. Any ambiguities that need clarification

Intent Types:
- **task**: User wants something done (e.g., "Add a login button", "Fix the bug in auth.py")
- **conversation**: General chat, not requiring action (e.g., "How are you?", "Tell me about CORE")
- **question**: Information request (e.g., "What files handle routing?", "How does authentication work?")
- **clarification**: Follow-up to previous context (e.g., "Do that", "Yes", "The first one")

Tools Available (these are the ONLY actions CORE can run):
"""
            + self._tools_section()
            + """

In tools_needed, name the capability ids the request needs (e.g. "git.log").
If the request needs an action that is not listed, name it anyway as
root.action (e.g. "git.push", "file_operations.delete") or as the missing tool
(e.g. "email", "slack"). Never map it onto a listed action that does something
else: an unlisted action is refused honestly, which is the right answer.

Respond in JSON format:
{
  "type": "task|conversation|question|clarification",
  "description": "Brief description of what the user wants",
  "confidence": 0.0-1.0,
  "requires_tools": true|false,
  "tools_needed": ["tool1", "tool2"],
  "ambiguities": ["ambiguity1", "ambiguity2"]
}"""
        )

    def check_knowledge_base(self, query: str, top_k: int = 5) -> Optional[str]:
        """
        Query the CORE knowledge base for documents relevant to the user query.

        Uses the local REST API (/knowledgebase/semantic-search) rather than
        direct DB access so this sync method works correctly when the LangGraph
        graph is run inside asyncio.to_thread (no shared event loop or pool).

        Returns a formatted context string to inject into the LLM prompt, or
        None if no relevant results are found or the KB is unavailable.
        """
        # Resolve an API key for the internal call: prefer the explicit env var,
        # fall back to the first key in the running auth module's key set.
        api_key = os.getenv("CORE_API_KEY")
        if not api_key:
            try:
                from app.auth import VALID_API_KEYS

                api_key = next(iter(VALID_API_KEYS), None)
            except Exception:
                pass
        if not api_key:
            logger.debug("check_knowledge_base: no API key available, skipping RAG")
            return None

        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.post(
                    f"{_CORE_BASE_URL}/knowledgebase/semantic-search",
                    json={"query": query, "limit": top_k},
                    headers={"X-API-Key": api_key},
                )
            if resp.status_code != 200:
                logger.debug(
                    "KB search returned HTTP %s, skipping RAG", resp.status_code
                )
                return None

            results: list = resp.json()
            if not results:
                return None

            lines = []
            for r in results[:top_k]:
                similarity = float(r.get("similarity", 0.0))
                if similarity < _KB_MIN_SIMILARITY:
                    continue
                title = (
                    r.get("title")
                    or r.get("originalName")
                    or r.get("filename")
                    or "Document"
                )
                description = (r.get("description") or "").strip()
                entry = f"- [{title}] (relevance: {similarity:.2f})"
                if description:
                    entry += f": {description}"
                lines.append(entry)

            if not lines:
                return None

            return "Relevant knowledge base context:\n" + "\n".join(lines)

        except Exception as exc:
            # KB lookup is best-effort — never block comprehension
            logger.debug("KB search failed (non-critical): %s", exc)
            return None

    def analyze_intent(self, user_input: str) -> UserIntent:
        """
        Analyze user input to determine intent, augmented with RAG context.

        Queries the knowledge base first and injects any relevant document
        context into the system prompt before calling the LLM. This grounds
        intent classification in actual project knowledge.

        Args:
            user_input: The raw user input string

        Returns:
            UserIntent object with classification and metadata
        """
        logger.info(f"Comprehension analyzing: '{user_input}' with model={self.model}")
        client = get_openai_client_sync()

        # RAG: inject KB context into system prompt before LLM call.
        # Wrapped independently so KB errors never block intent classification.
        try:
            rag_context = self.check_knowledge_base(user_input)
        except Exception as exc:
            logger.debug(
                "check_knowledge_base raised unexpectedly (non-critical): %s", exc
            )
            rag_context = None
        system_content = self.system_prompt
        if rag_context:
            system_content = self.system_prompt + f"\n\n{rag_context}"
            logger.debug(
                "Injected RAG context (%d chars) into comprehension prompt",
                len(rag_context),
            )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_input},
                ],
                temperature=0.3,  # Lower temperature for more consistent classification
                # Note: Ollama may not support response_format, removed for compatibility
            )

            # Parse JSON response
            logger.info(f"Raw LLM response object: {response}")
            content = response.choices[0].message.content
            logger.info(f"LLM content: {content}")

            if not content:
                raise ValueError("Empty response from LLM")

            logger.info(f"Comprehension LLM response: {content}")

            # Extract and repair JSON from response (handles code fences, trailing commas, etc.)
            extracted = extract_json_object(content)
            if extracted:
                logger.info(f"Extracted JSON object: {extracted[:200]}...")
            else:
                extracted = content

            data = safe_json_loads(extracted)
            if data is None:
                raise ValueError(
                    f"Could not parse JSON from response: {content[:200]}..."
                )

            # Build UserIntent from response
            intent = UserIntent(
                type=data.get("type", "conversation"),
                description=data.get("description", user_input),
                confidence=float(data.get("confidence", 0.7)),
                requires_tools=data.get("requires_tools", False),
                tools_needed=data.get("tools_needed", []),
                ambiguities=data.get("ambiguities", []),
            )

            logger.info(
                f"Intent classified as: {intent.type} (confidence: {intent.confidence})"
            )
            return intent

        except Exception as e:
            # Fallback to safe defaults on error
            logger.error(f"Comprehension agent error: {str(e)}", exc_info=True)
            return UserIntent(
                type="conversation",
                description=f"Error in comprehension: {str(e)}",
                confidence=0.5,
                requires_tools=False,
                tools_needed=[],
                ambiguities=[],
            )

    # ------------------------------------------------------------------
    # Ambiguity detection — pattern table
    # Each entry: (kind, regex_pattern, message_template)
    # Patterns are applied case-insensitively to the full input. A hit the same
    # request already resolves is skipped (see _is_resolved).
    # ------------------------------------------------------------------
    _AMBIGUITY_PATTERNS: list[tuple[str, str, str]] = [
        # Unresolved pronouns / deictic references
        (
            "pronoun",
            r"\b(it|that|this|those|these|they|them|the thing|the stuff)\b",
            "Unresolved pronoun or deictic reference: what does '{match}' refer to?",
        ),
        # Underspecified file / code artifacts
        (
            "artifact",
            r"\bthe\s+(file|component|function|class|module|service|endpoint|route|method|variable)\b",
            "Underspecified artifact: which {match}?",
        ),
        # Underspecified temporal reference
        (
            "temporal",
            r"\b(recently|before|after|last time|earlier|previously|just now)\b",
            "Ambiguous temporal reference: when exactly is '{match}'?",
        ),
        # Vague quantity
        (
            "quantity",
            r"\b(some|a few|several|many|a couple of|a bunch of)\b",
            "Vague quantity: how many does '{match}' mean?",
        ),
        # Underspecified location
        (
            "location",
            r"\b(there|somewhere|in that place|in the other place)\b",
            "Underspecified location: where is '{match}'?",
        ),
        # Relative / comparative without referent
        (
            "comparative",
            r"\b(better|faster|smaller|larger|cleaner|simpler)\s+than\b",
            "Comparative without referent: better/faster than what?",
        ),
    ]

    # Explicit referents a request can name: URL, path, file name with an
    # extension, glob, quoted or backticked text, commit sha, code identifier.
    _REFERENT_RE = re.compile(
        r"https?://\S+"
        r"|[\w.*-]+/[\w./*-]+"
        r"|\b[\w-]{2,}\.[A-Za-z][A-Za-z0-9]{0,5}\b"
        r"|\S*\*\S*"
        r"|(?<!\w)'[^'\n]+'(?!\w)|\"[^\"\n]+\"|`[^`\n]+`"
        r"|\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b"
        r"|\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b"
        r"|\b[a-z]+[A-Z][A-Za-z0-9]*\b"
    )
    # A name right after "the function" etc.: "the function decide_gate",
    # "the file auth.py", "the class called Foo".
    _NAMED_AFTER_RE = re.compile(
        r"\s+(?:(?:named|called)\s+\S+"
        r"|(?:[\w.-]*[_./0-9][\w./-]*|[A-Za-z][a-z0-9]*[A-Z]\w*"
        r"|`[^`]+`|'[^']+'|\"[^\"]+\"))"
    )
    # A concrete anchor after "before"/"after": a date, version, sha, quote.
    _TEMPORAL_ANCHOR_RE = re.compile(
        r"\s+(?:commit\s+|version\s+|release\s+|tag\s+)?"
        r"(?:\d{4}-\d{2}(?:-\d{2})?|\d{1,2}/\d{1,2}(?:/\d{2,4})?"
        r"|v?\d+(?:\.\d+)+|[0-9a-f]{7,40}\b|`[^`]+`|'[^']+'|\"[^\"]+\")"
    )
    # "this repository" names the environment CORE runs in, not a lost referent.
    _CONTEXT_NOUN_RE = re.compile(
        r"\s+(?:repo|repository|project|workspace|codebase|conversation|chat|session)\b",
        re.IGNORECASE,
    )
    # "the commit that added X": "that" after "<determiner> <noun>" opens a
    # relative clause; it is not a pointer to something unsaid.
    _RELATIVE_THAT_RE = re.compile(
        r"\b(?:the|a|an|any|every|each|all)\s+[a-z][\w-]*\s+$", re.IGNORECASE
    )
    _RESOLVABLE_PRONOUNS = frozenset(
        {"it", "that", "this", "those", "these", "they", "them"}
    )

    def _has_referent_before(self, text: str, pos: int) -> bool:
        return any(m.end() <= pos for m in self._REFERENT_RE.finditer(text))

    def _is_resolved(self, kind: str, match: re.Match, text: str) -> bool:
        """True when the request itself resolves this hit, so it is not an
        ambiguity worth a clarify round-trip."""
        word = match.group(0).lower()
        after = text[match.end() :]
        if kind == "pronoun":
            if word == "this" and self._CONTEXT_NOUN_RE.match(after):
                return True
            if word == "that" and self._RELATIVE_THAT_RE.search(text[: match.start()]):
                return True
            return word in self._RESOLVABLE_PRONOUNS and self._has_referent_before(
                text, match.start()
            )
        if kind == "artifact":
            return bool(self._NAMED_AFTER_RE.match(after))
        if kind == "temporal":
            return word in ("before", "after") and bool(
                self._TEMPORAL_ANCHOR_RE.match(after)
            )
        if kind == "quantity":
            # "some of them" after the request named them: a sample of a known set.
            m = re.match(r"\s+of\s+(them|these|those)\b", after, re.IGNORECASE)
            return bool(m) and self._has_referent_before(text, match.start())
        if kind == "location":
            return word == "there" and self._has_referent_before(text, match.start())
        return False

    def detect_ambiguities(self, user_input: str) -> list[str]:
        """
        Detect ambiguities in user input that may need clarification.

        Applies a catalogue of regex patterns to identify:
        - Unresolved pronouns / deictic references (it, that, those, …)
        - Underspecified artifacts (the file, the component, …)
        - Vague temporal references (recently, before, …)
        - Vague quantities (some, a few, several, …)
        - Underspecified locations (there, somewhere, …)
        - Comparatives without a referent (better than, faster than, …)

        A hit is skipped when the same request resolves it, so a fully
        specified request is not sent to clarify:
        - a pronoun, or "there", after an explicit referent (a path, file name,
          URL, glob, quoted text, sha or identifier): "Read main.py and explain
          what this module does";
        - "this repository/project/workspace…" (the current environment);
        - a relative "that" ("the commit that added X");
        - "the function/file/…" followed by a name: "the function decide_gate";
        - "before/after" followed by a date, version or sha: "after 2026-10-01";
        - "some of them" when "them" is resolved.

        Returns a deduplicated list of human-readable ambiguity descriptions.
        Empty list means no ambiguities were detected.
        """
        if not user_input or not user_input.strip():
            return []

        found: list[str] = []
        seen_messages: set[str] = set()

        for kind, pattern, message_template in self._AMBIGUITY_PATTERNS:
            for match in re.finditer(pattern, user_input, re.IGNORECASE):
                if self._is_resolved(kind, match, user_input):
                    continue
                matched_text = match.group(0).lower()
                # For "the <artifact>" pattern keep the full matched phrase
                message = message_template.format(match=matched_text)
                if message not in seen_messages:
                    found.append(message)
                    seen_messages.add(message)

        return found
