"""Shared, provider-aware LLM construction.

A single place that turns a model id into a configured ``ChatOpenAI``:

- OpenAI ids (``gpt-*``, ``o1``…) hit the OpenAI API (needs ``OPENAI_API_KEY``).
- Any other id (``google/gemma-4-e4b``, ``llama3.2``…) is routed to the active
  local provider (Ollama / LM Studio) over its OpenAI-compatible endpoint, so a
  local-only machine works with no OpenAI key.

Both the per-step engine (``core_entry``) and the Agent Factory build their LLMs
through this module so provider routing lives in exactly one place.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore

# Models that reject a custom ``temperature`` (e.g. some reasoning models).
_NO_TEMPERATURE_MODELS = {"gpt-5"}


def _is_openai_model(model: str) -> bool:
    """Heuristic: is this an OpenAI chat/reasoning model id?

    Everything else is treated as a local model and routed to the active local
    provider, so behaviour matches whatever the machine is configured to run.
    """
    m = (model or "").lower()
    return m.startswith(("gpt", "o1", "o3", "o4", "chatgpt", "text-", "davinci"))


def _local_endpoint() -> Tuple[str, str]:
    """``(base_url, api_key)`` for the active local provider's OpenAI-compatible API."""
    from app.dependencies import get_local_provider, _get_ollama_base_url

    if get_local_provider() == "lmstudio":
        return (
            os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
            os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
        )
    return f"{_get_ollama_base_url()}/v1", "ollama"


def build_chat_model(
    model: str,
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
):
    """Build a provider-aware ``ChatOpenAI`` for ``model``.

    ``temperature`` is silently omitted for models that reject it
    (:data:`_NO_TEMPERATURE_MODELS`). Local model ids get the local provider's
    ``base_url``/``api_key``; OpenAI ids are left to the default OpenAI client.
    """
    if ChatOpenAI is None:  # pragma: no cover
        raise RuntimeError("langchain_openai is not installed")

    kwargs = {"model": model}
    if temperature is not None and model not in _NO_TEMPERATURE_MODELS:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if not _is_openai_model(model):
        base_url, api_key = _local_endpoint()
        kwargs["base_url"] = base_url
        kwargs["api_key"] = api_key
    return ChatOpenAI(**kwargs)
