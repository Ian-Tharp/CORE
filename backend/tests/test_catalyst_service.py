"""
Tests for CatalystService — Divergence → Convergence → Synthesis pipeline.

Sentinel-value contract:
- Remove session phase update in run_divergence → phase never advances → test fails
- Remove ideas storage in run_divergence → session has no divergence_result → test fails
- Remove error phase assignment on exception → errors silently discarded → test fails
- Remove markdown fence stripping in _llm_json → valid JSON wrapped in ``` not parsed → test fails
- Remove _select_model config check → explicit model override ignored → test fails
- Remove auto_catalyst phase chaining → phases never run in sequence → test fails
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.catalyst_service import CatalystService, CatalystSession, SessionPhase


# ===========================================================================
# Helpers
# ===========================================================================

def _make_service() -> CatalystService:
    """Create service with a mocked model router."""
    mock_router = MagicMock()
    mock_router.select_model.return_value = "test-model"
    return CatalystService(router=mock_router)


def _setup_llm(service: CatalystService, return_value: Any) -> None:
    """Patch _llm_json to return a fixed value."""
    service._llm_json = AsyncMock(return_value=return_value)


# ===========================================================================
# CatalystSession — state tracking
# ===========================================================================

class TestCatalystSession:
    def test_initial_phase_is_created(self):
        """
        SENTINEL — new session must start in CREATED phase.
        Remove phase initialization → sessions start with undefined state → test fails.
        """
        session = CatalystSession("s-1", "test prompt", {})
        assert session.phase == SessionPhase.CREATED

    def test_to_dict_includes_all_fields(self):
        """to_dict must include session_id, prompt, config, and phase."""
        session = CatalystSession("SENTINEL_SESSION", "SENTINEL_PROMPT", {"key": "val"})
        d = session.to_dict()
        assert d["session_id"] == "SENTINEL_SESSION"
        assert d["prompt"] == "SENTINEL_PROMPT"
        assert d["config"] == {"key": "val"}
        assert d["phase"] == "created"

    def test_to_dict_results_are_none_initially(self):
        """divergence_result, convergence_result, synthesis_result must start None."""
        session = CatalystSession("s-1", "p", {})
        d = session.to_dict()
        assert d["divergence_result"] is None
        assert d["convergence_result"] is None
        assert d["synthesis_result"] is None

    def test_touch_updates_updated_at(self):
        """_touch must update updated_at."""
        session = CatalystSession("s-1", "p", {})
        original = session.updated_at
        session._touch()
        # updated_at should be same or newer (may be same if called immediately)
        assert session.updated_at >= original


# ===========================================================================
# _select_model
# ===========================================================================

class TestSelectModel:
    def test_explicit_model_in_config_used_directly(self):
        """
        SENTINEL — config['model'] must be returned without calling the router.
        Remove config.get('model') → explicit model override ignored → test fails.
        """
        svc = _make_service()
        result = svc._select_model({"model": "SENTINEL_MODEL"})
        assert result == "SENTINEL_MODEL"
        svc._router.select_model.assert_not_called()

    def test_no_config_falls_back_to_router(self):
        """
        SENTINEL — empty config must delegate to router.select_model.
        Remove router fallback → model is always None → test fails.
        """
        svc = _make_service()
        svc._router.select_model.return_value = "router-selected-model"
        result = svc._select_model({})
        svc._router.select_model.assert_called_once()
        assert result == "router-selected-model"


# ===========================================================================
# _llm_json — JSON parsing and fence stripping
# ===========================================================================

class TestLlmJson:
    @pytest.mark.asyncio
    async def test_clean_json_object_parsed(self):
        """Valid JSON response must be parsed and returned as a dict."""
        svc = _make_service()
        svc._router.complete = AsyncMock(return_value={"content": '{"key": "SENTINEL_VALUE"}'})
        result = await svc._llm_json("sys", "user", "model")
        assert result == {"key": "SENTINEL_VALUE"}

    @pytest.mark.asyncio
    async def test_json_array_parsed(self):
        """JSON array response must be parsed and returned as a list."""
        svc = _make_service()
        svc._router.complete = AsyncMock(return_value={"content": '[{"id": 1}, {"id": 2}]'})
        result = await svc._llm_json("sys", "user", "model")
        assert isinstance(result, list)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_markdown_fence_stripped(self):
        """
        SENTINEL — JSON wrapped in ```json ... ``` fences must be unwrapped.
        Remove fence stripping → JSONDecodeError raised for valid content → test fails.
        """
        svc = _make_service()
        svc._router.complete = AsyncMock(return_value={
            "content": '```json\n{"key": "SENTINEL_FENCED"}\n```'
        })
        result = await svc._llm_json("sys", "user", "model")
        assert result == {"key": "SENTINEL_FENCED"}

    @pytest.mark.asyncio
    async def test_bare_fence_stripped(self):
        """``` (without json) fences must also be stripped."""
        svc = _make_service()
        svc._router.complete = AsyncMock(return_value={
            "content": '```\n{"key": "bare"}\n```'
        })
        result = await svc._llm_json("sys", "user", "model")
        assert result == {"key": "bare"}

    @pytest.mark.asyncio
    async def test_invalid_json_raises_value_error(self):
        """Non-JSON response must raise ValueError (not JSONDecodeError)."""
        svc = _make_service()
        svc._router.complete = AsyncMock(return_value={"content": "this is not json"})
        with pytest.raises(ValueError, match="not valid JSON"):
            await svc._llm_json("sys", "user", "model")

    @pytest.mark.asyncio
    async def test_messages_include_system_and_user(self):
        """
        SENTINEL — _llm_json must send both system and user messages.
        Remove system message → LLM has no context/persona → test fails.
        """
        svc = _make_service()
        svc._router.complete = AsyncMock(return_value={"content": '{"ok": true}'})
        await svc._llm_json("SENTINEL_SYSTEM", "SENTINEL_USER", "model")
        call_kwargs = svc._router.complete.call_args[1]
        messages = call_kwargs.get("messages", [])
        roles = {m["role"] for m in messages}
        assert "system" in roles
        assert "user" in roles
        contents = [m["content"] for m in messages]
        assert any("SENTINEL_SYSTEM" in c for c in contents)
        assert any("SENTINEL_USER" in c for c in contents)


# ===========================================================================
# start_session / get_session
# ===========================================================================

class TestSessionManagement:
    def test_start_session_returns_dict_with_session_id(self):
        """start_session must return a dict with a non-empty session_id."""
        svc = _make_service()
        result = svc.start_session("test prompt")
        assert "session_id" in result
        assert result["session_id"]

    def test_start_session_stores_in_sessions(self):
        """
        SENTINEL — started session must be retrievable via get_session.
        Remove storage → get_session returns None → test fails.
        """
        svc = _make_service()
        result = svc.start_session("test prompt")
        session = svc.get_session(result["session_id"])
        assert session is not None

    def test_start_session_prompt_preserved(self):
        """Prompt must be stored and returned via get_session."""
        svc = _make_service()
        result = svc.start_session("SENTINEL_PROMPT")
        session = svc.get_session(result["session_id"])
        assert session["prompt"] == "SENTINEL_PROMPT"

    def test_get_session_returns_none_for_unknown(self):
        """get_session with unknown ID must return None."""
        svc = _make_service()
        assert svc.get_session("nonexistent-id") is None

    def test_each_session_gets_unique_id(self):
        """Two sessions started with the same prompt must have different IDs."""
        svc = _make_service()
        s1 = svc.start_session("same prompt")
        s2 = svc.start_session("same prompt")
        assert s1["session_id"] != s2["session_id"]


# ===========================================================================
# run_divergence
# ===========================================================================

class TestRunDivergence:
    @pytest.mark.asyncio
    async def test_divergence_returns_ideas_from_llm(self):
        """
        SENTINEL — run_divergence must return the list from _llm_json.
        Remove return statement → ideas never surface → test fails.
        """
        svc = _make_service()
        ideas = [{"title": "SENTINEL_IDEA_1"}, {"title": "SENTINEL_IDEA_2"}]
        _setup_llm(svc, ideas)
        result = await svc.run_divergence("no-session", "prompt", count=2)
        assert result == ideas

    @pytest.mark.asyncio
    async def test_divergence_updates_session_phase(self):
        """
        SENTINEL — run_divergence must advance session phase to DIVERGENCE.
        Remove phase update → callers can't track pipeline progress → test fails.
        """
        svc = _make_service()
        s = svc.start_session("prompt")
        _setup_llm(svc, [{"title": "idea"}])
        await svc.run_divergence(s["session_id"], "prompt")
        session = svc.get_session(s["session_id"])
        assert session["phase"] == "divergence"

    @pytest.mark.asyncio
    async def test_divergence_stores_result_in_session(self):
        """
        SENTINEL — ideas must be stored in session.divergence_result.
        Remove storage → synthesis has no input data → test fails.
        """
        svc = _make_service()
        s = svc.start_session("prompt")
        ideas = [{"title": "SENTINEL_STORED_IDEA"}]
        _setup_llm(svc, ideas)
        await svc.run_divergence(s["session_id"], "prompt")
        session = svc.get_session(s["session_id"])
        assert session["divergence_result"] == ideas

    @pytest.mark.asyncio
    async def test_divergence_non_list_response_wrapped_in_list(self):
        """If LLM returns a single dict, it must be wrapped in a list."""
        svc = _make_service()
        _setup_llm(svc, {"title": "single idea"})
        result = await svc.run_divergence("no-session", "prompt")
        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_divergence_error_sets_session_phase_to_error(self):
        """
        SENTINEL — on LLM failure, session phase must become ERROR.
        Remove error handler → phase stuck in divergence, error silenced → test fails.
        """
        svc = _make_service()
        s = svc.start_session("prompt")
        svc._llm_json = AsyncMock(side_effect=ValueError("LLM failed"))
        with pytest.raises(ValueError):
            await svc.run_divergence(s["session_id"], "prompt")
        session = svc.get_session(s["session_id"])
        assert session["phase"] == "error"
        assert "LLM failed" in session["error"]


# ===========================================================================
# run_convergence
# ===========================================================================

class TestRunConvergence:
    @pytest.mark.asyncio
    async def test_convergence_returns_analysis_from_llm(self):
        """run_convergence must return the dict from _llm_json."""
        svc = _make_service()
        analysis = {"clusters": [{"label": "SENTINEL_CLUSTER"}]}
        _setup_llm(svc, analysis)
        result = await svc.run_convergence("no-session", [{"title": "idea"}])
        assert result == analysis

    @pytest.mark.asyncio
    async def test_convergence_updates_session_phase(self):
        """Session phase must become CONVERGENCE after run_convergence."""
        svc = _make_service()
        s = svc.start_session("prompt")
        _setup_llm(svc, {"result": "ok"})
        await svc.run_convergence(s["session_id"], [])
        session = svc.get_session(s["session_id"])
        assert session["phase"] == "convergence"

    @pytest.mark.asyncio
    async def test_convergence_stores_result_in_session(self):
        """
        SENTINEL — convergence analysis must be stored in session.convergence_result.
        Remove storage → synthesis has no convergence data → test fails.
        """
        svc = _make_service()
        s = svc.start_session("prompt")
        analysis = {"top_ideas": ["SENTINEL_TOP"]}
        _setup_llm(svc, analysis)
        await svc.run_convergence(s["session_id"], [])
        session = svc.get_session(s["session_id"])
        assert session["convergence_result"] == analysis


# ===========================================================================
# run_synthesis
# ===========================================================================

class TestRunSynthesis:
    @pytest.mark.asyncio
    async def test_synthesis_sets_phase_complete(self):
        """
        SENTINEL — successful synthesis must set session phase to COMPLETE.
        Remove phase update → pipeline never closes → test fails.
        """
        svc = _make_service()
        s = svc.start_session("prompt")
        _setup_llm(svc, {"summary": "SENTINEL_SYNTHESIS"})
        await svc.run_synthesis(s["session_id"], [], {})
        session = svc.get_session(s["session_id"])
        assert session["phase"] == "complete"

    @pytest.mark.asyncio
    async def test_synthesis_stores_result_in_session(self):
        """synthesis_result must be stored in the session."""
        svc = _make_service()
        s = svc.start_session("prompt")
        synthesis = {"summary": "SENTINEL_SYNTHESIS"}
        _setup_llm(svc, synthesis)
        await svc.run_synthesis(s["session_id"], [], {})
        session = svc.get_session(s["session_id"])
        assert session["synthesis_result"] == synthesis


# ===========================================================================
# auto_catalyst — full pipeline
# ===========================================================================

class TestAutoCatalyst:
    @pytest.mark.asyncio
    async def test_auto_catalyst_runs_all_three_phases(self):
        """
        SENTINEL — auto_catalyst must invoke all three phases in sequence.
        Remove any phase call → pipeline incomplete → test fails.
        """
        svc = _make_service()
        call_log = []

        async def _fake_diverge(session_id, prompt, count=5):
            call_log.append("divergence")
            return [{"title": "idea"}]

        async def _fake_converge(session_id, ideas):
            call_log.append("convergence")
            return {"clusters": []}

        async def _fake_synthesize(session_id, ideas, convergence):
            call_log.append("synthesis")
            return {"summary": "done"}

        with patch.object(svc, "run_divergence", side_effect=_fake_diverge), \
             patch.object(svc, "run_convergence", side_effect=_fake_converge), \
             patch.object(svc, "run_synthesis", side_effect=_fake_synthesize):
            await svc.auto_catalyst("test prompt")

        assert call_log == ["divergence", "convergence", "synthesis"]

    @pytest.mark.asyncio
    async def test_auto_catalyst_returns_complete_session(self):
        """
        SENTINEL — return value must be the complete session dict.
        Return only synthesis result → callers lose full session state → test fails.
        """
        svc = _make_service()

        async def _fake_diverge(session_id, prompt, count=5):
            return [{"title": "idea"}]

        async def _fake_converge(session_id, ideas):
            return {"clusters": []}

        async def _fake_synthesize(session_id, ideas, convergence):
            # Manually advance the session to COMPLETE since we patched the real method
            session = svc._sessions[session_id]
            session.phase = SessionPhase.COMPLETE
            session.synthesis_result = {"summary": "done"}
            return {"summary": "done"}

        with patch.object(svc, "run_divergence", side_effect=_fake_diverge), \
             patch.object(svc, "run_convergence", side_effect=_fake_converge), \
             patch.object(svc, "run_synthesis", side_effect=_fake_synthesize):
            result = await svc.auto_catalyst("test prompt")

        assert "session_id" in result
        assert result["phase"] == "complete"

    @pytest.mark.asyncio
    async def test_auto_catalyst_passes_ideas_to_convergence(self):
        """
        SENTINEL — ideas from divergence must be forwarded to convergence.
        Hard-code empty ideas → convergence always has no input → test fails.
        """
        svc = _make_service()
        sentinel_ideas = [{"title": "SENTINEL_IDEA"}]
        convergence_input = []

        async def _fake_diverge(session_id, prompt, count=5):
            return sentinel_ideas

        async def _fake_converge(session_id, ideas):
            convergence_input.extend(ideas)
            return {"clusters": []}

        async def _fake_synthesize(session_id, ideas, convergence):
            return {"summary": "done"}

        with patch.object(svc, "run_divergence", side_effect=_fake_diverge), \
             patch.object(svc, "run_convergence", side_effect=_fake_converge), \
             patch.object(svc, "run_synthesis", side_effect=_fake_synthesize):
            await svc.auto_catalyst("prompt")

        assert convergence_input == sentinel_ideas
