# Local LLM Providers (Ollama & LM Studio)

_Last updated: 2026-06-01_

CORE routes completions through `ModelRouter`
([`backend/app/services/model_router.py`](../../backend/app/services/model_router.py)),
which talks to each provider over an **OpenAI-compatible API**. Two local providers
are supported (both free, no network egress) alongside the cloud providers:

| Provider | Default endpoint | Notes |
|----------|------------------|-------|
| **Ollama** | `http://ollama:11434/v1` (`OLLAMA_BASE_URL`) | runs as a Docker service |
| **LM Studio** | `http://localhost:1234/v1` (`LMSTUDIO_BASE_URL`) | runs on the **host** |
| OpenAI | api.openai.com | needs `OPENAI_API_KEY` |
| Anthropic | api.anthropic.com | needs `ANTHROPIC_API_KEY` |

## Using LM Studio

[LM Studio](https://lmstudio.ai/) exposes a local OpenAI-compatible server, so it
slots in as a drop-in local provider.

### 1. Start the LM Studio server
- Load a model in the LM Studio app.
- Open the **Developer** (or **Local Server**) tab and click **Start Server**. It
  listens on `http://localhost:1234/v1` by default (port configurable in the tab).
- The API key is ignored on localhost — any non-empty string works (`lm-studio`).
- If the **backend runs in Docker** (it does in this repo), the container reaches the
  host via `host.docker.internal`. The compose files already set
  `LMSTUDIO_BASE_URL=http://host.docker.internal:1234/v1` and an
  `extra_hosts: host.docker.internal:host-gateway` entry. In LM Studio's server
  settings, bind to `0.0.0.0` (not just `127.0.0.1`) so the container can reach it.

### 2. Configure CORE (local settings)
Set these env vars (e.g. in `backend/.env`, or your shell, or override in compose):

| Variable | Purpose | Example |
|----------|---------|---------|
| `LMSTUDIO_MODELS` | **Required** to activate — comma-separated ids of the model(s) you've loaded in LM Studio. Registers them in the model router. | `qwen2.5-7b-instruct,llama-3.2-3b-instruct` |
| `LMSTUDIO_BASE_URL` | Server URL. Defaults to `http://localhost:1234/v1`; the dockerized backend defaults to `http://host.docker.internal:1234/v1`. | |
| `LMSTUDIO_API_KEY` | Any non-empty string. | `lm-studio` |
| `LMSTUDIO_CONTEXT_WINDOW` | Context window registered for the models. | `8192` |
| `CORE_LOCAL_PROVIDER` | `ollama` (default) or `lmstudio` — which local provider auto-selection prefers. | `lmstudio` |
| `CORE_DEFAULT_MODEL` | Make a specific model the default. Set to an LM Studio model id to use it by default. | `qwen2.5-7b-instruct` |

> LM Studio stays **dormant** until you set `LMSTUDIO_MODELS` — by default nothing is
> registered and Ollama remains the local provider, so this change is zero-impact
> until you opt in.

### 3. Apply and verify
```bash
# After editing env / compose:
docker compose up -d core-backend

# Confirm the model is registered (lists provider=lmstudio):
curl -s http://localhost:8001/admin/models -H "X-API-Key: $CORE_API_KEY" | jq '.models[] | select(.provider=="lmstudio")'
```

The model router will now prefer your LM Studio model for local inference (per
`CORE_LOCAL_PROVIDER` / `CORE_DEFAULT_MODEL`) and fall back through the chain on error.

## How it works
- `ModelProvider.LMSTUDIO` is a registered provider; `ModelRouter.get_client()` builds
  an `AsyncOpenAI` client at `LMSTUDIO_BASE_URL`.
- `LMSTUDIO_MODELS` is parsed at startup by `_register_env_lmstudio_models()` and each
  id is added to the `MODELS` registry (zero cost, `BALANCED` tier).
- `select_model(..., prefer_local=True)` treats Ollama **and** LM Studio as local and
  biases toward `CORE_LOCAL_PROVIDER`.

## Sources
- [LM Studio — OpenAI compatibility endpoints](https://lmstudio.ai/docs/developer/openai-compat)
- [LM Studio — Local LLM API server](https://lmstudio.ai/docs/developer/core/server)
