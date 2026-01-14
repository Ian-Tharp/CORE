"""Service for generating conversation titles based on chat context.

This module uses an LLM to generate concise, descriptive titles for conversations
based on the first user message and assistant response.
"""

import os
import logging
from typing import List, Dict
import httpx

logger = logging.getLogger(__name__)

# Configuration
TITLE_MODEL = os.getenv("TITLE_GENERATION_MODEL", "gpt-oss:20b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")


async def generate_conversation_title(user_message: str, assistant_message: str) -> str:
    """Generate a concise title for a conversation.

    Args:
        user_message: The first user message in the conversation
        assistant_message: The assistant's response to the first message

    Returns:
        A 3-5 word title, or a fallback if generation fails
    """
    try:
        # Smart prompt engineering for concise titles
        system_prompt = """Generate a concise 3-5 word title for this conversation.
The title should capture the main topic or question.
Do not use quotes. Be specific and descriptive.

Examples:
- User: "How do I debug React state?" → React State Debugging
- User: "Help me with Python async" → Python Async Help
- User: "What's the weather?" → Weather Inquiry
- User: "Explain quantum computing" → Quantum Computing Explanation"""

        prompt = f"""User asked: "{user_message[:200]}"
Assistant responded: "{assistant_message[:200]}"

Title:"""

        # Call Ollama API for title generation
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": TITLE_MODEL,
                    "prompt": f"{system_prompt}\n\n{prompt}",
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Low temperature for consistency
                        "num_predict": 20,   # Limit tokens for brevity
                    },
                },
            )
            response.raise_for_status()
            result = response.json()
            title = result.get("response", "").strip()

            # Clean up the title
            title = _clean_title(title)

            # Fallback if empty
            if not title:
                title = _fallback_title(user_message)

            logger.info(f"Generated title: {title}")
            return title

    except Exception as e:
        logger.error(f"Title generation failed: {e}")
        return _fallback_title(user_message)


def _clean_title(title: str) -> str:
    """Clean and validate the generated title."""
    # Remove quotes if present
    title = title.strip('"\'')

    # Remove common prefixes
    prefixes = ["Title:", "title:", "The title is:", "A title:", "Here's a title:"]
    for prefix in prefixes:
        if title.lower().startswith(prefix.lower()):
            title = title[len(prefix):].strip()

    # Truncate if too long
    if len(title) > 60:
        title = title[:57] + "..."

    # Capitalize first letter
    if title:
        title = title[0].upper() + title[1:]

    return title


def _fallback_title(user_message: str) -> str:
    """Generate a fallback title from the user message."""
    # Use first 50 chars of user message
    if user_message:
        title = user_message[:50].strip()
        if len(user_message) > 50:
            title += "..."
        return title
    return "New Conversation"
