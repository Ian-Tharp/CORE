# ADR-007: Anthropic as a first-class chat provider

**Status:** ✅ Implemented
**Date:** 2026-06-03
**Author:** Ian + Claude Code

## Context

The `/chat/stream` route supported only two streaming paths: OpenAI and the local provider
(Ollama / LM Studio). `_normalise_provider()` in `chat_service.py` collapsed anything that
wasn't a local alias to `"openai"`, so a request with `provider: "anthropic"` was silently
sent to the OpenAI client with a `claude-*` model and failed. Anthropic models were listed in
the config registry but had no execution path, and the UI had no way to select Anthropic.

## Decision

Make **Anthropic** a first-class chat provider end to end.

- **Backend** (`backend/app/services/chat_service.py`):
  - `_normalise_provider()` now keeps `"anthropic"` as its own canonical provider (own
    circuit breaker), instead of coercing it to `"openai"`.
  - New `_stream_from_anthropic()` SSE path: hoists `system` messages to the top-level
    Anthropic `system` argument, resolves a registry key (e.g. `claude-3-haiku`) to its real
    API model id, and rewraps text deltas as `{ "delta": ... }` so the controller/UI parse
    them identically to the OpenAI/local streams.
  - Async client `_get_async_anthropic_client()` in `dependencies.py`.
- **Config** (`backend/app/config/models.py`): registered `claude-haiku-4-5`
  (`claude-haiku-4-5-20251001`) alongside the existing `claude-3-haiku` / `claude-3-5-sonnet`.
- **Frontend** (`chat-window`): added the **Anthropic** provider option and its model list;
  widened `ChatService.sendMessage`'s `provider` type to include `'anthropic'`.

See [`deployment/chat-providers.md`](../deployment/chat-providers.md) for usage.

## Consequences

**Positive:** users can run cloud Claude (incl. Haiku 4.5) from the chat route/UI; the path is
provider-symmetric with OpenAI/local. Verified end-to-end against the running stack.

**Negative / notes:** Anthropic output is capped by `ANTHROPIC_MAX_TOKENS` (2048) for
interactive turns; the API key lives in `backend/.env` (`ANTHROPIC_API_KEY`).
