"""
Behavior tests for app.services.llm_provider — provider-aware ChatOpenAI construction.

ChatOpenAI is patched with a recording fake so no client is constructed and no
network is touched; tests assert on the kwargs the real client would receive.
"""

from __future__ import annotations

import pytest

import app.dependencies as deps
import app.services.llm_provider as lp


class _FakeChat:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def fake_chat(monkeypatch):
    monkeypatch.setattr(lp, "ChatOpenAI", _FakeChat)
    return _FakeChat


@pytest.fixture
def ollama_env(monkeypatch):
    monkeypatch.delenv("CORE_LOCAL_PROVIDER", raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama-test:11434")
    cache_clear = getattr(deps._get_ollama_base_url, "cache_clear", None)
    if cache_clear:
        cache_clear()
    yield
    if cache_clear:
        cache_clear()


@pytest.fixture
def lmstudio_env(monkeypatch):
    monkeypatch.setenv("CORE_LOCAL_PROVIDER", "lmstudio")
    monkeypatch.delenv("LMSTUDIO_BASE_URL", raising=False)
    monkeypatch.delenv("LMSTUDIO_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# _is_openai_model
# ---------------------------------------------------------------------------


class TestIsOpenAIModel:
    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4o",
            "GPT-4o-mini",
            "gpt-5",
            "o1",
            "o3-mini",
            "o4-mini",
            "chatgpt-4o-latest",
            "text-davinci-003",
            "davinci-002",
        ],
    )
    def test_openai_ids_recognised(self, model):
        assert lp._is_openai_model(model) is True

    @pytest.mark.parametrize(
        "model",
        ["llama3.2", "google/gemma-4-e4b", "qwen2.5:7b", "mistral", "local-model"],
    )
    def test_local_ids_not_openai(self, model):
        assert lp._is_openai_model(model) is False

    @pytest.mark.parametrize("model", ["", None])
    def test_empty_or_none_is_not_openai(self, model):
        assert lp._is_openai_model(model) is False


# ---------------------------------------------------------------------------
# _local_endpoint
# ---------------------------------------------------------------------------


class TestLocalEndpoint:
    def test_ollama_default_endpoint(self, ollama_env):
        assert lp._local_endpoint() == ("http://ollama-test:11434/v1", "ollama")

    def test_lmstudio_defaults(self, lmstudio_env):
        assert lp._local_endpoint() == ("http://localhost:1234/v1", "lm-studio")

    def test_lmstudio_env_overrides(self, lmstudio_env, monkeypatch):
        monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://box:9999/v1")
        monkeypatch.setenv("LMSTUDIO_API_KEY", "secret")
        assert lp._local_endpoint() == ("http://box:9999/v1", "secret")

    def test_unknown_provider_falls_back_to_ollama(self, ollama_env, monkeypatch):
        monkeypatch.setenv("CORE_LOCAL_PROVIDER", "vllm")
        base_url, key = lp._local_endpoint()
        assert base_url == "http://ollama-test:11434/v1"
        assert key == "ollama"


# ---------------------------------------------------------------------------
# build_chat_model
# ---------------------------------------------------------------------------


class TestBuildChatModel:
    def test_openai_model_uses_default_client(self, fake_chat):
        llm = lp.build_chat_model("gpt-4o", temperature=0.3, top_p=0.9)
        assert isinstance(llm, _FakeChat)
        assert llm.kwargs == {"model": "gpt-4o", "temperature": 0.3, "top_p": 0.9}
        assert "base_url" not in llm.kwargs and "api_key" not in llm.kwargs

    def test_minimal_kwargs_when_no_sampling_params(self, fake_chat):
        llm = lp.build_chat_model("gpt-4o")
        assert llm.kwargs == {"model": "gpt-4o"}

    def test_temperature_zero_is_passed_through(self, fake_chat):
        # 0.0 is a valid temperature and must not be treated as "unset".
        llm = lp.build_chat_model("gpt-4o", temperature=0.0)
        assert llm.kwargs["temperature"] == 0.0

    def test_no_temperature_model_drops_temperature(self, fake_chat):
        llm = lp.build_chat_model("gpt-5", temperature=0.7, top_p=0.5)
        assert "temperature" not in llm.kwargs
        assert llm.kwargs["top_p"] == 0.5

    def test_local_model_routed_to_ollama(self, fake_chat, ollama_env):
        llm = lp.build_chat_model("llama3.2", temperature=0.2)
        assert llm.kwargs == {
            "model": "llama3.2",
            "temperature": 0.2,
            "base_url": "http://ollama-test:11434/v1",
            "api_key": "ollama",
        }

    def test_local_model_routed_to_lmstudio(self, fake_chat, lmstudio_env):
        llm = lp.build_chat_model("google/gemma-4-e4b")
        assert llm.kwargs["base_url"] == "http://localhost:1234/v1"
        assert llm.kwargs["api_key"] == "lm-studio"

    def test_missing_langchain_raises(self, monkeypatch):
        monkeypatch.setattr(lp, "ChatOpenAI", None)
        with pytest.raises(RuntimeError, match="langchain_openai"):
            lp.build_chat_model("gpt-4o")

    def test_constructor_error_propagates(self, monkeypatch):
        def boom(**kwargs):
            raise ValueError("bad config")

        monkeypatch.setattr(lp, "ChatOpenAI", boom)
        with pytest.raises(ValueError, match="bad config"):
            lp.build_chat_model("gpt-4o")
