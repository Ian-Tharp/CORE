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
from typing import Optional

import httpx

from app.dependencies import get_openai_client_sync
from app.models.core_state import UserIntent
from app.utils.json_repair import safe_json_loads, extract_json_object

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

    def __init__(self, model: str = "gpt-oss:20b"):
        # Default to local model for offline-first operation
        # Override with gpt-4o-mini or claude-3-5-haiku-latest for cloud
        self.model = model
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the base system prompt for intent classification."""
        return """You are the Comprehension layer of the CORE cognitive system.

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

Tools Available:
- file_operations: Read, write, edit, search files
- git: Branch, commit, diff, create PR
- database: Query, schema, explain
- web_research: Search docs, Stack Overflow, GitHub

Respond in JSON format:
{
  "type": "task|conversation|question|clarification",
  "description": "Brief description of what the user wants",
  "confidence": 0.0-1.0,
  "requires_tools": true|false,
  "tools_needed": ["tool1", "tool2"],
  "ambiguities": ["ambiguity1", "ambiguity2"]
}"""

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
                logger.debug("KB search returned HTTP %s, skipping RAG", resp.status_code)
                return None

            results: list = resp.json()
            if not results:
                return None

            lines = []
            for r in results[:top_k]:
                similarity = float(r.get("similarity", 0.0))
                if similarity < _KB_MIN_SIMILARITY:
                    continue
                title = r.get("title") or r.get("originalName") or r.get("filename") or "Document"
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
            logger.debug("check_knowledge_base raised unexpectedly (non-critical): %s", exc)
            rag_context = None
        system_content = self.system_prompt
        if rag_context:
            system_content = self.system_prompt + f"\n\n{rag_context}"
            logger.debug("Injected RAG context (%d chars) into comprehension prompt", len(rag_context))

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
                raise ValueError(f"Could not parse JSON from response: {content[:200]}...")

            # Build UserIntent from response
            intent = UserIntent(
                type=data.get("type", "conversation"),
                description=data.get("description", user_input),
                confidence=float(data.get("confidence", 0.7)),
                requires_tools=data.get("requires_tools", False),
                tools_needed=data.get("tools_needed", []),
                ambiguities=data.get("ambiguities", [])
            )

            logger.info(f"Intent classified as: {intent.type} (confidence: {intent.confidence})")
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
                ambiguities=[]
            )

    def detect_ambiguities(self, user_input: str) -> list[str]:
        """
        Detect ambiguities in user input that need clarification.

        RSI TODO: Implement ambiguity detection (e.g., "the component" - which one?)
        RSI TODO: Use entity recognition to identify underspecified references
        """
        # Placeholder for future ambiguity detection
        return []
