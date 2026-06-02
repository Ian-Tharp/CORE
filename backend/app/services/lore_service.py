from __future__ import annotations

"""Schema-aware world lore generation.

Generates a structured wiki page of a given *kind* (Overview, History, Peoples…)
for a world, grounded in whatever is known about it, and persists it as a real
wiki page tagged with that kind in its metadata — so generated lore populates the
same wiki schema the rest of the app reads (and can be ingested into knowledge).
"""

import asyncio
import logging
import os
from typing import Any, Dict, Optional

# Reuse the provider-agnostic LLM call (OpenAI ids -> OpenAI, else local provider).
from app.controllers.core_entry import _llm_or_stub
from app.repository import creative_repository

logger = logging.getLogger(__name__)

LORE_SYSTEM = (
    "You are a master world-builder and loremaster. Write a single rich, vivid, "
    "internally consistent wiki page in Markdown for a fictional world. Invent "
    "evocative proper nouns and concrete specifics that fit the world's known "
    "details. Begin with exactly one '# ' title line, then well-structured "
    "sections using '## ' headings. Write only the page — no preamble or "
    "commentary outside it."
)


async def generate_lore_page(
    world_id: str,
    *,
    kind: str,
    focus: str,
    world_name: Optional[str] = None,
    context: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a `kind` wiki page for a world and persist it; returns the page."""
    user = (
        f"World name: {world_name or 'an unnamed world'}.\n"
        f"Known details: {context or 'none provided yet'}.\n"
        f'Write the "{kind}" page for this world. Focus: {focus}\n'
        "Return only the Markdown page, starting with its '# ' title."
    )
    # Prefer a fast cloud model for prose when a key is present (local models are
    # too slow for a full page); fall back to the configured default otherwise.
    chosen = model or (
        "gpt-4o-mini"
        if os.getenv("OPENAI_API_KEY")
        else os.getenv("CORE_DEFAULT_MODEL")
    )
    # _llm_or_stub is synchronous (LangChain invoke) — run off the event loop.
    text = (await asyncio.to_thread(_llm_or_stub, LORE_SYSTEM, user, chosen)) or ""
    body = text.strip()

    # Derive the title from the first H1, else fall back to the kind.
    title = kind
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip() or kind
            break

    page_id = await creative_repository.create_wiki_page(
        world_id,
        title,
        body,
        {"template": kind, "source": "ai"},
    )
    return {"id": page_id, "title": title, "content": body}
