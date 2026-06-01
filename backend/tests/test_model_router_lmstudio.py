"""
Tests for the LM Studio local provider in the model router.

LM Studio exposes an OpenAI-compatible server; the router treats it as a local
provider configured entirely via environment variables.
"""

import app.services.model_router as mr


class TestLmStudioProvider:
    def test_lmstudio_is_a_local_provider(self):
        assert mr.ModelProvider.LMSTUDIO in mr.LOCAL_PROVIDERS

    def test_client_uses_configured_base_url(self, monkeypatch):
        monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://host.docker.internal:1234/v1")
        monkeypatch.setenv("LMSTUDIO_API_KEY", "lm-studio")

        router = mr.ModelRouter()
        client = router.get_client(mr.ModelProvider.LMSTUDIO)

        assert "host.docker.internal:1234" in str(client.base_url)

    def test_client_defaults_to_localhost(self, monkeypatch):
        monkeypatch.delenv("LMSTUDIO_BASE_URL", raising=False)
        router = mr.ModelRouter()
        client = router.get_client(mr.ModelProvider.LMSTUDIO)
        assert "localhost:1234" in str(client.base_url)


class TestLmStudioModelRegistration:
    def test_registers_models_from_env(self, monkeypatch):
        monkeypatch.setenv("LMSTUDIO_MODELS", "qwen-test-7b, llama-test-3b")
        try:
            mr._register_env_lmstudio_models()
            assert "qwen-test-7b" in mr.MODELS
            assert "llama-test-3b" in mr.MODELS
            cfg = mr.MODELS["qwen-test-7b"]
            assert cfg.provider == mr.ModelProvider.LMSTUDIO
            assert cfg.cost_per_1k_output == 0.0
        finally:
            mr.MODELS.pop("qwen-test-7b", None)
            mr.MODELS.pop("llama-test-3b", None)

    def test_empty_env_registers_nothing(self, monkeypatch):
        monkeypatch.setenv("LMSTUDIO_MODELS", "")
        before = set(mr.MODELS)
        mr._register_env_lmstudio_models()
        assert set(mr.MODELS) == before


class TestLocalProviderPreference:
    def test_select_model_prefers_configured_local_provider(self, monkeypatch):
        monkeypatch.setenv("LMSTUDIO_MODELS", "lmstudio-balanced-test")
        monkeypatch.setenv("CORE_LOCAL_PROVIDER", "lmstudio")
        try:
            mr._register_env_lmstudio_models()
            router = mr.ModelRouter()
            chosen = router.select_model(task_type="creative", prefer_local=True)
            assert mr.MODELS[chosen].provider == mr.ModelProvider.LMSTUDIO
        finally:
            mr.MODELS.pop("lmstudio-balanced-test", None)

    def test_default_local_provider_is_ollama(self, monkeypatch):
        monkeypatch.delenv("CORE_LOCAL_PROVIDER", raising=False)
        router = mr.ModelRouter()
        chosen = router.select_model(task_type="creative", prefer_local=True)
        assert mr.MODELS[chosen].provider == mr.ModelProvider.OLLAMA
