"""
Behavior tests for the /core/* step-by-step playground API (app.controllers.core_entry).

The LLM boundary is mocked by patching ``build_chat_model`` in the controller
module; nothing reaches the network. Most tests override ``require_api_key``
on a minimal app; a dedicated class exercises the real auth dependency.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.auth as auth
import app.controllers.core_entry as ce
from app.auth import require_api_key


class FakeLLM:
    """Records the prompts it is invoked with and returns canned content."""

    def __init__(self, content="ok", error=None):
        self.content = content
        self.error = error
        self.calls = []

    def invoke(self, prompts):
        self.calls.append(prompts)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content=self.content)


class FakeFactory:
    """Stand-in for build_chat_model; records (model, temperature) per build."""

    def __init__(self, *llms):
        self.llms = list(llms)
        self.builds = []

    def __call__(self, model, temperature=None, top_p=None):
        self.builds.append((model, temperature))
        return self.llms.pop(0) if len(self.llms) > 1 else self.llms[0]


@pytest.fixture
def clean_env(monkeypatch):
    for var in (
        "CORE_DEFAULT_MODEL",
        "OPENAI_MODEL",
        "OPENAI_API_KEY",
        "CORE_LOCAL_PROVIDER",
        "CORE_LOCAL_CHAT_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(ce.router)
    app.dependency_overrides[require_api_key] = lambda: "test"
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def llm(monkeypatch, clean_env):
    """Install a single FakeLLM; tests set .content / .error as needed."""
    fake = FakeLLM()
    factory = FakeFactory(fake)
    monkeypatch.setattr(ce, "build_chat_model", factory)
    fake.factory = factory
    return fake


def _body(**extra):
    return {"message_id": "m1", "user_input": "hello", "model": "llama3.2", **extra}


def _events(resp):
    return [
        json.loads(line[len("data: ") :])
        for line in resp.text.split("\n\n")
        if line.startswith("data: ")
    ]


# ---------------------------------------------------------------------------
# Auth (real dependency)
# ---------------------------------------------------------------------------


class TestAuth:
    @pytest.fixture
    def real_client(self, monkeypatch):
        monkeypatch.setattr(auth, "VALID_API_KEYS", {"good-key"})
        app = FastAPI()
        app.include_router(ce.router)
        return TestClient(app)

    @pytest.mark.parametrize(
        "path",
        [
            "/core",
            "/core/comprehension",
            "/core/orchestration/stream",
            "/core/evaluation",
        ],
    )
    def test_missing_key_401(self, real_client, path):
        assert real_client.post(path, json=_body()).status_code == 401

    def test_wrong_key_401(self, real_client):
        resp = real_client.post(
            "/core/reasoning", json=_body(), headers={"X-API-Key": "nope"}
        )
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid API key"

    def test_valid_key_accepted(self, real_client, llm):
        resp = real_client.post(
            "/core", json=_body(), headers={"X-API-Key": "good-key"}
        )
        assert resp.status_code == 200
        assert "CORE entry acknowledged" in resp.json()["message"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize(
        "body",
        [
            {"user_input": "hi"},  # missing message_id
            {"message_id": "m1"},  # missing user_input
            {"message_id": "m1", "user_input": "hi", "orchestration_plan": "x"},
        ],
    )
    def test_invalid_body_422(self, client, llm, body):
        resp = client.post("/core/comprehension", json=body)
        assert resp.status_code == 422
        assert llm.calls == []


# ---------------------------------------------------------------------------
# Non-streaming steps
# ---------------------------------------------------------------------------


class TestSteps:
    def test_entry_does_not_call_llm(self, client, llm):
        resp = client.post("/core", json=_body())
        assert resp.status_code == 200
        assert "/core/comprehension" in resp.json()["message"]
        assert llm.calls == []

    def test_comprehension_routes_to_orchestration(self, client, llm):
        llm.content = "This is a query needing a multi-step PLAN."
        resp = client.post("/core/comprehension", json=_body(user_input="do X"))
        assert resp.status_code == 200
        data = resp.json()
        assert data["step"] == "Comprehension"
        assert data["text"] == llm.content
        assert data["routing_decision"] == "orchestration"
        (system, user) = llm.calls[0]
        assert system[0] == "system" and "Classify" in system[1]
        assert user == ("user", "do X")
        assert llm.factory.builds == [("llama3.2", 0.2)]

    def test_comprehension_routes_to_conversation(self, client, llm):
        llm.content = "Just small talk."
        data = client.post("/core/comprehension", json=_body()).json()
        assert data["routing_decision"] == "conversation"

    def test_orchestration_parses_plan_and_caps_at_six(self, client, llm):
        llm.content = "\n".join(f"- step {i}" for i in range(1, 9)) + "\n\n"
        data = client.post("/core/orchestration", json=_body()).json()
        assert data["step"] == "Orchestration"
        assert data["plan"] == [f"step {i}" for i in range(1, 7)]

    def test_orchestration_includes_prior_context(self, client, llm):
        client.post(
            "/core/orchestration",
            json=_body(comprehension_text="needs tools", comprehension_route="orch"),
        )
        system = llm.calls[0][0][1]
        assert "Previous comprehension: needs tools" in system
        assert "Routing decision: orch" in system

    def test_orchestration_route_ignored_without_comprehension_text(self, client, llm):
        client.post("/core/orchestration", json=_body(comprehension_route="orch"))
        assert "Context:" not in llm.calls[0][0][1]

    def test_orchestration_empty_llm_output_gives_empty_plan(self, client, llm):
        llm.content = ""
        data = client.post("/core/orchestration", json=_body()).json()
        assert data["plan"] == []
        assert data["text"] == ""

    def test_reasoning_includes_context(self, client, llm):
        llm.content = "result"
        data = client.post(
            "/core/reasoning",
            json=_body(
                comprehension_text="C",
                orchestration_text="O",
                orchestration_plan=["a", "b"],
            ),
        ).json()
        assert data == {
            "step": "Reasoning",
            "text": "result",
            "routing_decision": None,
            "plan": None,
            "evaluation": None,
        }
        system = llm.calls[0][0][1]
        assert "Comprehension: C" in system
        assert "Orchestration summary: O" in system
        assert "Plan: a; b" in system

    def test_reasoning_without_context(self, client, llm):
        client.post("/core/reasoning", json=_body())
        assert "Context:" not in llm.calls[0][0][1]

    def test_evaluation_satisfactory(self, client, llm):
        llm.content = "SATISFACTORY - the result meets the goal."
        data = client.post(
            "/core/evaluation",
            json=_body(
                comprehension_text="C",
                orchestration_text="O",
                orchestration_plan=["a"],
                reasoning_text="R",
            ),
        ).json()
        assert data["step"] == "Evaluation"
        assert data["evaluation"] == "SATISFACTORY"
        system = llm.calls[0][0][1]
        for frag in ("Comprehension: C", "Orchestration: O", "Plan: a", "Reasoning: R"):
            assert frag in system

    def test_evaluation_unsatisfactory_verdict(self, client, llm):
        """Regression: 'UNSATISFACTORY' contains 'satisf' and was misread as a pass."""
        llm.content = "UNSATISFACTORY - the plan missed step 2. Revise."
        data = client.post("/core/evaluation", json=_body()).json()
        assert data["evaluation"] == "UNSATISFACTORY"

    def test_evaluation_no_verdict_defaults_unsatisfactory(self, client, llm):
        llm.content = "I cannot tell."
        data = client.post("/core/evaluation", json=_body()).json()
        assert data["evaluation"] == "UNSATISFACTORY"


# ---------------------------------------------------------------------------
# Streaming steps
# ---------------------------------------------------------------------------


class TestStreaming:
    @pytest.mark.parametrize(
        "path,step",
        [
            ("/core/comprehension/stream", "Comprehension"),
            ("/core/orchestration/stream", "Orchestration"),
            ("/core/reasoning/stream", "Reasoning"),
            ("/core/evaluation/stream", "Evaluation"),
        ],
    )
    def test_stream_emits_start_chunks_metrics_end(self, client, llm, path, step):
        llm.content = "alpha beta  gamma"
        resp = client.post(
            path,
            json=_body(
                comprehension_text="C",
                comprehension_route="r",
                orchestration_text="O",
                orchestration_plan=["p"],
                reasoning_text="R",
            ),
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = _events(resp)
        assert events[0] == {"type": "start", "step": step}
        chunks = [e["text"] for e in events if e["type"] == "chunk"]
        assert chunks == ["alpha ", "beta ", "gamma "]
        metrics = events[-2]
        assert metrics["type"] == "metrics"
        assert metrics["tokens"] == 3
        assert metrics["duration_ms"] >= 0 and metrics["ttfb_ms"] >= 0
        assert events[-1] == {"type": "end"}
        # Comprehension is the first step and takes no prior context.
        assert ("Context:" in llm.calls[0][0][1]) == (step != "Comprehension")

    @pytest.mark.parametrize(
        "path",
        [
            "/core/comprehension/stream",
            "/core/orchestration/stream",
            "/core/reasoning/stream",
            "/core/evaluation/stream",
        ],
    )
    def test_stream_empty_output(self, client, llm, path):
        llm.content = ""
        events = _events(client.post(path, json=_body()))
        assert [e["type"] for e in events] == ["start", "metrics", "end"]
        assert events[1]["tokens"] == 0
        assert "Context:" not in llm.calls[0][0][1]


# ---------------------------------------------------------------------------
# _llm_or_stub: model selection, stubbing, error handling
# ---------------------------------------------------------------------------


class TestLlmOrStub:
    def test_stub_when_langchain_missing(self, monkeypatch, llm):
        monkeypatch.setattr(ce, "ChatOpenAI", None)
        out = ce._llm_or_stub("SYS", "hi", "gpt-4o")
        assert out == "[stubbed] System: SYS\nUser: hi"
        assert llm.factory.builds == []

    def test_openai_model_without_key_is_stubbed(self, llm):
        out = ce._llm_or_stub("SYS", "hi", "gpt-4o")
        assert out.startswith("[stubbed]")
        assert llm.factory.builds == []

    def test_openai_model_with_key_calls_llm(self, monkeypatch, llm):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
        llm.content = "answer"
        assert ce._llm_or_stub("SYS", "hi", "gpt-4o") == "answer"
        assert llm.factory.builds == [("gpt-4o", 0.2)]

    def test_local_model_runs_without_openai_key(self, llm):
        llm.content = "local answer"
        assert ce._llm_or_stub("SYS", "hi", "qwen2.5:7b") == "local answer"
        assert llm.factory.builds == [("qwen2.5:7b", 0.2)]

    def test_model_precedence_env_default(self, monkeypatch, llm):
        monkeypatch.setenv("CORE_DEFAULT_MODEL", "  mistral  ")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
        ce._llm_or_stub("SYS", "hi")
        assert llm.factory.builds[0][0] == "mistral"

    def test_model_precedence_openai_model_env(self, monkeypatch, llm):
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
        ce._llm_or_stub("SYS", "hi")
        assert llm.factory.builds[0][0] == "gpt-4o-mini"

    def test_model_falls_back_to_local_chat_model(self, monkeypatch, llm):
        monkeypatch.setenv("CORE_LOCAL_CHAT_MODEL", "phi3")
        ce._llm_or_stub("SYS", "hi")
        assert llm.factory.builds[0][0] == "phi3"

    def test_generic_llm_error_returned_as_text(self, llm):
        llm.error = ConnectionError("connection refused")
        out = ce._llm_or_stub("SYS", "hi", "llama3.2")
        assert out == "[LLM error: connection refused]\nSystem: SYS\nUser: hi"
        # No temperature retry for unrelated errors.
        assert llm.factory.builds == [("llama3.2", 0.2)]

    def test_unsupported_temperature_retries_without_it(self, monkeypatch, clean_env):
        bad = FakeLLM(error=ValueError("temperature is Unsupported for this model"))
        good = FakeLLM(content="retried ok")
        factory = FakeFactory(bad, good)
        monkeypatch.setattr(ce, "build_chat_model", factory)
        assert ce._llm_or_stub("SYS", "hi", "llama3.2") == "retried ok"
        assert factory.builds == [("llama3.2", 0.2), ("llama3.2", None)]

    def test_temperature_retry_failure_reports_second_error(
        self, monkeypatch, clean_env
    ):
        bad = FakeLLM(error=ValueError("model does not support temperature"))
        worse = FakeLLM(error=RuntimeError("server down"))
        factory = FakeFactory(bad, worse)
        monkeypatch.setattr(ce, "build_chat_model", factory)
        out = ce._llm_or_stub("SYS", "hi", "llama3.2")
        assert out.startswith("[LLM error: server down]")
        assert len(factory.builds) == 2

    def test_llm_error_surfaces_via_endpoint_as_200(self, client, llm):
        llm.error = TimeoutError("timed out")
        resp = client.post("/core/reasoning", json=_body())
        assert resp.status_code == 200
        assert resp.json()["text"].startswith("[LLM error: timed out]")

    def test_build_failure_is_caught(self, monkeypatch, clean_env):
        def boom(model, temperature=None, top_p=None):
            raise RuntimeError("langchain_openai is not installed")

        monkeypatch.setattr(ce, "build_chat_model", boom)
        out = ce._llm_or_stub("SYS", "hi", "llama3.2")
        assert out.startswith("[LLM error: langchain_openai is not installed]")

    def test_reexports_for_backcompat(self):
        from app.services import llm_provider

        assert ce._is_openai_model is llm_provider._is_openai_model
        assert ce._local_endpoint is llm_provider._local_endpoint
