# Chat Providers (OpenAI · Anthropic · Local)

_Last updated: 2026-06-03_

The `/chat/stream` route streams from three provider families, selected per request via the
`provider` field. For **local** setup (Ollama / LM Studio) see
[`local-llm-providers.md`](./local-llm-providers.md); this doc covers provider/model selection
at the chat layer.

## Selecting a provider

`POST /chat/stream` (requires `X-API-Key`):

```jsonc
{
  "provider": "anthropic",       // "openai" | "anthropic" | "ollama" (local)
  "model": "claude-haiku-4-5",   // registry key or provider model id
  "stream": true,
  "messages": [{ "role": "user", "content": "Hello" }]
}
```

`chat_service._normalise_provider()` maps the value to a canonical provider, each with its own
circuit breaker and streaming path:

| `provider` value | Canonical | Streaming path |
|---|---|---|
| `openai` (default), `gpt-*`, unknown | `openai` | OpenAI Responses API |
| `anthropic` | `anthropic` | Anthropic Messages API (`_stream_from_anthropic`) |
| `ollama` / `local` / `local-ollama` | `ollama` | active local provider (Ollama or LM Studio per `CORE_LOCAL_PROVIDER`) |

In the UI, the chat window's provider dropdown offers **OpenAI · Anthropic · Local**, and the
model list updates per provider.

## Models

Cloud models are defined in [`backend/app/config/models.py`](../../backend/app/config/models.py).
A registry **key** (e.g. `claude-3-haiku`) is resolved to its real API id before the call.

| Provider | Keys (examples) | API id |
|---|---|---|
| Anthropic | `claude-haiku-4-5` | `claude-haiku-4-5-20251001` |
| | `claude-3-5-sonnet` | `claude-3-5-sonnet-20241022` |
| | `claude-3-haiku` | `claude-3-haiku-20240307` |
| OpenAI | `gpt-4o`, `gpt-4o-mini` | same |
| Local | any id loaded in Ollama / LM Studio | passthrough |

## Keys & limits

- `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` live in `backend/.env` (gitignored).
- Anthropic responses are capped at `ANTHROPIC_MAX_TOKENS` (2048) per turn — an output cap,
  not the context window.

## Notes

- Anthropic `system` messages are hoisted to the top-level `system` argument automatically.
- Text deltas from every provider are normalized to `data: {"delta": "..."}` SSE events, so
  the controller and UI handle all three identically.

See also: [ADR-007 — Anthropic streaming provider](../adr/007-anthropic-streaming-provider.md).
