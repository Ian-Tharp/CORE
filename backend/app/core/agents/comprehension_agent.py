"""
Comprehension Agent - Understands user intent and retrieves relevant context.

Determines:
- Is this a task, conversation, question, or clarification request?
- What tools/capabilities are needed?
- What context from the knowledge base is relevant?
- Are there ambiguities that need clarification?

RSI TODO: Integrate with knowledge base for RAG-enhanced comprehension
RSI TODO: Add multi-turn clarification dialog support
"""

import json
import logging
from typing import Optional
from app.dependencies import get_openai_client_sync
from app.models.core_state import UserIntent
from app.utils.json_repair import safe_json_loads, extract_json_object

logger = logging.getLogger(__name__)


class ComprehensionAgent:
    """
    Comprehension Agent - First stage of CORE pipeline.

    Analyzes user input to determine intent type and required capabilities.
    """

    def __init__(self, model: str = "gpt-oss:20b"):
        # Default to local model for offline-first operation
        # Override with gpt-4o-mini or claude-3-5-haiku-latest for cloud
        self.model = model
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt for intent classification."""
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

    def analyze_intent(self, user_input: str) -> UserIntent:
        """
        Analyze user input to determine intent.

        Args:
            user_input: The raw user input string

        Returns:
            UserIntent object with classification and metadata
        """
        logger.info(f"Comprehension analyzing: '{user_input}' with model={self.model}")
        client = get_openai_client_sync()

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
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

    def check_knowledge_base(self, query: str) -> Optional[str]:
        """
        Query the knowledge base for relevant context.

        RSI TODO: Implement RAG-based retrieval from pgvector
        RSI TODO: Rank results by relevance and recency
        """
        # Placeholder for future KB integration
        return None

    def detect_ambiguities(self, user_input: str) -> list[str]:
        """
        Detect ambiguities in user input that need clarification.

        RSI TODO: Implement ambiguity detection (e.g., "the component" - which one?)
        RSI TODO: Use entity recognition to identify underspecified references
        """
        # Placeholder for future ambiguity detection
        return []
