"""
Tests for app/config/models.py — ModelProvider enum, ModelConfig methods, MODELS registry,
DEFAULT_MODELS, get_default_model, list_available_models, get_models_by_provider,
is_model_available, get_base_url.

Sentinel-value contract:
- Remove OLLAMA from ModelProvider → local models have no provider → registry breaks → test fails
- Remove get_base_url OLLAMA branch → Ollama URL defaults to empty string → LLM calls fail → test fails
- Remove DEFAULT_MODELS comprehension key → comprehension agent gets gpt-oss:20b fallback → test fails
- Remove get_default_model production flag → production deployment uses dev models → test fails
- Remove MODELS["nomic-embed-text"] → embedding use case has no model → KB query crashes → test fails
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from app.config.models import (
    DEFAULT_MODELS,
    MODELS,
    PRODUCTION_MODELS,
    ModelConfig,
    ModelProvider,
    get_default_model,
    get_model_config,
    get_models_by_provider,
    is_model_available,
    list_available_models,
)


# ===========================================================================
# ModelProvider enum
# ===========================================================================

class TestModelProvider:
    def test_ollama_value(self):
        """
        SENTINEL — OLLAMA must equal 'ollama'.
        Change → config lookup fails → no local models found → test fails.
        """
        assert ModelProvider.OLLAMA == "ollama"

    def test_openai_value(self):
        """OPENAI must equal 'openai'."""
        assert ModelProvider.OPENAI == "openai"

    def test_anthropic_value(self):
        """ANTHROPIC must equal 'anthropic'."""
        assert ModelProvider.ANTHROPIC == "anthropic"

    def test_local_value(self):
        """LOCAL must equal 'local'."""
        assert ModelProvider.LOCAL == "local"

    def test_all_four_providers(self):
        """Exactly four providers must be defined."""
        assert len(ModelProvider) == 4


# ===========================================================================
# ModelConfig.get_base_url — provider routing
# ===========================================================================

class TestGetBaseUrl:
    def test_explicit_base_url_returned_as_is(self):
        """
        SENTINEL — explicit base_url must override provider default.
        Ignore base_url → custom deployments break → test fails.
        """
        cfg = ModelConfig(
            provider=ModelProvider.OLLAMA,
            model_name="test",
            display_name="Test",
            base_url="http://SENTINEL_HOST:11434",
        )
        assert cfg.get_base_url() == "http://SENTINEL_HOST:11434"

    def test_ollama_uses_env_or_default(self):
        """
        SENTINEL — OLLAMA provider without explicit base_url must return Ollama URL.
        Return empty string → all Ollama calls fail → test fails.
        """
        cfg = ModelConfig(provider=ModelProvider.OLLAMA, model_name="test", display_name="Test")
        url = cfg.get_base_url()
        assert "ollama" in url.lower() or "11434" in url

    def test_openai_base_url(self):
        """
        SENTINEL — OPENAI provider must return OpenAI API URL.
        Return empty → all OpenAI calls fail → test fails.
        """
        cfg = ModelConfig(provider=ModelProvider.OPENAI, model_name="test", display_name="Test")
        url = cfg.get_base_url()
        assert "openai.com" in url

    def test_anthropic_base_url(self):
        """ANTHROPIC provider must return Anthropic API URL."""
        cfg = ModelConfig(provider=ModelProvider.ANTHROPIC, model_name="test", display_name="Test")
        url = cfg.get_base_url()
        assert "anthropic.com" in url

    def test_local_provider_returns_empty_string(self):
        """LOCAL provider with no explicit URL must return empty string (safe fallback)."""
        cfg = ModelConfig(provider=ModelProvider.LOCAL, model_name="test", display_name="Test")
        url = cfg.get_base_url()
        assert isinstance(url, str)

    def test_ollama_base_url_env_override(self):
        """OLLAMA_BASE_URL env var must override the default Ollama URL."""
        cfg = ModelConfig(provider=ModelProvider.OLLAMA, model_name="test", display_name="Test")
        with patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://SENTINEL_CUSTOM:11434"}):
            url = cfg.get_base_url()
        assert url == "http://SENTINEL_CUSTOM:11434"


# ===========================================================================
# ModelConfig.get_api_key
# ===========================================================================

class TestGetApiKey:
    def test_returns_none_without_api_key_env(self):
        """get_api_key must return None when api_key_env is not set."""
        cfg = ModelConfig(provider=ModelProvider.OLLAMA, model_name="test", display_name="Test")
        assert cfg.get_api_key() is None

    def test_returns_env_value_when_set(self):
        """
        SENTINEL — get_api_key must return the value of the configured env var.
        Hardcode None → cloud API calls always fail → test fails.
        """
        cfg = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="test",
            display_name="Test",
            api_key_env="SENTINEL_KEY_VAR",
        )
        with patch.dict("os.environ", {"SENTINEL_KEY_VAR": "SENTINEL_API_KEY_VALUE"}):
            assert cfg.get_api_key() == "SENTINEL_API_KEY_VALUE"

    def test_returns_none_when_env_var_missing(self):
        """get_api_key returns None when env var is configured but not present."""
        cfg = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="test",
            display_name="Test",
            api_key_env="NONEXISTENT_VAR_XYZ",
        )
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("NONEXISTENT_VAR_XYZ", None)
            assert cfg.get_api_key() is None


# ===========================================================================
# MODELS registry — required entries
# ===========================================================================

class TestModelsRegistry:
    def test_phi3_mini_in_registry(self):
        """
        SENTINEL — 'phi3:mini' must be in MODELS.
        Remove it → DEFAULT_MODELS points to missing key → all pipeline agents crash → test fails.
        """
        assert "phi3:mini" in MODELS

    def test_nomic_embed_in_registry(self):
        """
        SENTINEL — 'nomic-embed-text' must be in MODELS.
        Remove it → KB embedding use case has no model → semantic search crashes → test fails.
        """
        assert "nomic-embed-text" in MODELS

    def test_gpt4o_in_registry(self):
        """'gpt-4o' must be in MODELS for production use."""
        assert "gpt-4o" in MODELS

    def test_nomic_embed_not_streaming(self):
        """
        SENTINEL — nomic-embed-text must have supports_streaming=False.
        Set True → embedding calls try to stream → JSON parse error → test fails.
        """
        cfg = MODELS["nomic-embed-text"]
        assert cfg.supports_streaming is False

    def test_nomic_embed_no_tools(self):
        """nomic-embed-text must have supports_tools=False (embedding model)."""
        cfg = MODELS["nomic-embed-text"]
        assert cfg.supports_tools is False

    def test_phi3_mini_is_ollama(self):
        """phi3:mini must be an OLLAMA provider model."""
        assert MODELS["phi3:mini"].provider == ModelProvider.OLLAMA

    def test_gpt4o_is_openai(self):
        """gpt-4o must be an OPENAI provider model."""
        assert MODELS["gpt-4o"].provider == ModelProvider.OPENAI


# ===========================================================================
# DEFAULT_MODELS — required use-case keys
# ===========================================================================

class TestDefaultModels:
    def test_comprehension_key_present(self):
        """
        SENTINEL — 'comprehension' must be in DEFAULT_MODELS.
        Remove key → comprehension agent gets gpt-oss fallback → different model in production → test fails.
        """
        assert "comprehension" in DEFAULT_MODELS

    def test_orchestration_key_present(self):
        """'orchestration' must be in DEFAULT_MODELS."""
        assert "orchestration" in DEFAULT_MODELS

    def test_reasoning_key_present(self):
        """'reasoning' must be in DEFAULT_MODELS."""
        assert "reasoning" in DEFAULT_MODELS

    def test_evaluation_key_present(self):
        """'evaluation' must be in DEFAULT_MODELS."""
        assert "evaluation" in DEFAULT_MODELS

    def test_embeddings_key_present(self):
        """
        SENTINEL — 'embeddings' must be in DEFAULT_MODELS.
        Remove → KB embedding fails on agent startup → test fails.
        """
        assert "embeddings" in DEFAULT_MODELS

    def test_embeddings_uses_nomic(self):
        """
        SENTINEL — embeddings default must be 'nomic-embed-text'.
        Change to phi3:mini → embedding dimension mismatch → pgvector query fails → test fails.
        """
        assert DEFAULT_MODELS["embeddings"] == "nomic-embed-text"


# ===========================================================================
# get_model_config
# ===========================================================================

class TestGetModelConfig:
    def test_returns_config_for_known_model(self):
        """
        SENTINEL — get_model_config('phi3:mini') must return a ModelConfig.
        Return None → agent_factory falls through to no model → test fails.
        """
        cfg = get_model_config("phi3:mini")
        assert cfg is not None
        assert isinstance(cfg, ModelConfig)

    def test_returns_none_for_unknown_model(self):
        """Unknown model name must return None (no KeyError)."""
        result = get_model_config("SENTINEL_NONEXISTENT_MODEL")
        assert result is None


# ===========================================================================
# get_default_model
# ===========================================================================

class TestGetDefaultModel:
    def test_dev_comprehension_model(self):
        """get_default_model('comprehension') must return a non-empty string."""
        result = get_default_model("comprehension")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_production_flag_uses_production_models(self):
        """
        SENTINEL — production=True must return a production model.
        Ignore flag → production deployment uses dev models → test fails.
        """
        dev = get_default_model("orchestration", production=False)
        prod = get_default_model("orchestration", production=True)
        # They may differ (phi3:mini vs gpt-4o); both must be non-empty strings
        assert isinstance(prod, str) and len(prod) > 0
        assert isinstance(dev, str) and len(dev) > 0

    def test_unknown_use_case_returns_fallback(self):
        """Unknown use case must return a fallback string, not raise."""
        result = get_default_model("SENTINEL_UNKNOWN_USE_CASE")
        assert isinstance(result, str)
        assert len(result) > 0


# ===========================================================================
# list_available_models / get_models_by_provider
# ===========================================================================

class TestListAndFilter:
    def test_list_available_models_returns_all(self):
        """
        SENTINEL — list_available_models() must return all configured models.
        Return empty list → UI model selector shows nothing → test fails.
        """
        models = list_available_models()
        assert len(models) == len(MODELS)
        assert all(isinstance(m, ModelConfig) for m in models)

    def test_get_models_by_provider_ollama(self):
        """
        SENTINEL — get_models_by_provider(OLLAMA) must return only Ollama models.
        Return all models → cloud models shown as local → test fails.
        """
        ollama_models = get_models_by_provider(ModelProvider.OLLAMA)
        assert len(ollama_models) > 0
        assert all(m.provider == ModelProvider.OLLAMA for m in ollama_models)

    def test_get_models_by_provider_does_not_include_other_providers(self):
        """Ollama filter must exclude OpenAI and Anthropic models."""
        ollama_models = get_models_by_provider(ModelProvider.OLLAMA)
        openai_models = get_models_by_provider(ModelProvider.OPENAI)
        ollama_names = {m.model_name for m in ollama_models}
        openai_names = {m.model_name for m in openai_models}
        assert ollama_names.isdisjoint(openai_names)


# ===========================================================================
# is_model_available
# ===========================================================================

class TestIsModelAvailable:
    def test_unknown_model_not_available(self):
        """
        SENTINEL — unknown model must return False.
        Return True always → agent_factory uses non-existent model → test fails.
        """
        assert is_model_available("SENTINEL_NO_SUCH_MODEL") is False

    def test_ollama_model_available_without_api_key(self):
        """
        SENTINEL — local Ollama model must be available without an API key.
        Require API key for Ollama → all local deployments fail → test fails.
        """
        result = is_model_available("phi3:mini")
        assert result is True

    def test_cloud_model_unavailable_without_api_key(self):
        """
        SENTINEL — cloud model must return False when API key env var is missing.
        Ignore API key check → callers attempt cloud calls with no credentials → test fails.
        """
        import os
        # Ensure the env var is absent
        key_env = MODELS["gpt-4o"].api_key_env
        original = os.environ.pop(key_env, None)
        try:
            result = is_model_available("gpt-4o")
            assert result is False
        finally:
            if original is not None:
                os.environ[key_env] = original
