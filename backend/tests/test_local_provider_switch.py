"""
Tests for the agnostic local-provider switch (Ollama <-> LM Studio).

CORE_LOCAL_PROVIDER selects which local provider the shared local client, the
default local chat model, and (elsewhere) the router/embeddings use. It defaults
to ``ollama`` so machines without the env var are unaffected.
"""

import app.dependencies as deps


class TestLocalProvider:
    def test_default_is_ollama(self, monkeypatch):
        monkeypatch.delenv("CORE_LOCAL_PROVIDER", raising=False)
        assert deps.get_local_provider() == "ollama"

    def test_lmstudio_when_set_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("CORE_LOCAL_PROVIDER", "  LMStudio ")
        assert deps.get_local_provider() == "lmstudio"


class TestLocalClient:
    def test_ollama_client_by_default(self, monkeypatch):
        monkeypatch.delenv("CORE_LOCAL_PROVIDER", raising=False)
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
        deps._get_ollama_base_url.cache_clear()
        deps.get_ollama_client.cache_clear()
        try:
            client = deps.get_ollama_client()
            assert "ollama:11434" in str(client.base_url)
        finally:
            deps.get_ollama_client.cache_clear()
            deps._get_ollama_base_url.cache_clear()

    def test_lmstudio_client_when_configured(self, monkeypatch):
        monkeypatch.setenv("CORE_LOCAL_PROVIDER", "lmstudio")
        monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://host.docker.internal:1234/v1")
        deps.get_ollama_client.cache_clear()
        try:
            client = deps.get_ollama_client()
            assert "host.docker.internal:1234" in str(client.base_url)
        finally:
            deps.get_ollama_client.cache_clear()


class TestLocalChatModel:
    def test_ollama_default(self, monkeypatch):
        monkeypatch.delenv("CORE_LOCAL_PROVIDER", raising=False)
        monkeypatch.delenv("CORE_LOCAL_CHAT_MODEL", raising=False)
        assert deps.get_local_chat_model() == "llama3.2"

    def test_lmstudio_uses_default_model(self, monkeypatch):
        monkeypatch.setenv("CORE_LOCAL_PROVIDER", "lmstudio")
        monkeypatch.setenv("CORE_DEFAULT_MODEL", "google/gemma-4-e4b")
        assert deps.get_local_chat_model() == "google/gemma-4-e4b"

    def test_lmstudio_falls_back_to_first_registered(self, monkeypatch):
        monkeypatch.setenv("CORE_LOCAL_PROVIDER", "lmstudio")
        monkeypatch.delenv("CORE_DEFAULT_MODEL", raising=False)
        monkeypatch.setenv("LMSTUDIO_MODELS", "qwen-a, qwen-b")
        assert deps.get_local_chat_model() == "qwen-a"
