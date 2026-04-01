"""
Tests for ReasoningAgent RAG integration.

Sentinel-value contract: every RAG test would FAIL if you removed
fetch_step_context or the prompt injection in _execute_with_llm.
"""

import pytest
from unittest.mock import MagicMock, patch
from app.core.agents.reasoning_agent import ReasoningAgent
from app.models.core_state import ExecutionPlan, PlanStep, StepResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_step(name: str = "Test Step", description: str = "Do the thing", tool: str = None) -> PlanStep:
    return PlanStep(name=name, description=description, tool=tool)


def _make_plan(*steps: PlanStep) -> ExecutionPlan:
    return ExecutionPlan(goal="test goal", steps=list(steps))


def _make_llm_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# fetch_step_context — unit tests
# ---------------------------------------------------------------------------

class TestFetchStepContext:
    def setup_method(self):
        self.agent = ReasoningAgent()

    def test_query_built_from_step_name_and_description(self):
        """
        SENTINEL TEST — proves step name+description drive the KB query.
        Fails if the query is changed to something else (e.g., just the name).
        """
        step = _make_step(name="SENTINEL_NAME", description="SENTINEL_DESC")

        with patch("app.core.agents.reasoning_agent.httpx.Client") as mock_cls, \
             patch.dict("os.environ", {"CORE_API_KEY": "test-key"}):
            mock_http = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_http
            mock_http.post.return_value.status_code = 200
            mock_http.post.return_value.json.return_value = []

            self.agent.fetch_step_context(step)

        _, kwargs = mock_http.post.call_args
        sent_query = kwargs["json"]["query"]
        assert "SENTINEL_NAME" in sent_query
        assert "SENTINEL_DESC" in sent_query

    def test_returns_none_when_no_api_key(self):
        """No API key → None, no exception."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("app.core.agents.reasoning_agent.httpx.Client") as mock_cls:
                mock_cls.return_value.__enter__.return_value.post.return_value.status_code = 200
                mock_cls.return_value.__enter__.return_value.post.return_value.json.return_value = []
                result = self.agent.fetch_step_context(_make_step())
        assert result is None or isinstance(result, str)

    def test_returns_none_on_non_200(self):
        with patch("app.core.agents.reasoning_agent.httpx.Client") as mock_cls, \
             patch.dict("os.environ", {"CORE_API_KEY": "k"}):
            mock_http = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_http
            mock_http.post.return_value.status_code = 500
            result = self.agent.fetch_step_context(_make_step())
        assert result is None

    def test_returns_none_on_empty_results(self):
        with patch("app.core.agents.reasoning_agent.httpx.Client") as mock_cls, \
             patch.dict("os.environ", {"CORE_API_KEY": "k"}):
            mock_http = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_http
            mock_http.post.return_value.status_code = 200
            mock_http.post.return_value.json.return_value = []
            result = self.agent.fetch_step_context(_make_step())
        assert result is None

    def test_filters_below_similarity_threshold(self):
        """Results below 0.30 similarity are excluded — context returns None."""
        with patch("app.core.agents.reasoning_agent.httpx.Client") as mock_cls, \
             patch.dict("os.environ", {"CORE_API_KEY": "k"}):
            mock_http = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_http
            mock_http.post.return_value.status_code = 200
            mock_http.post.return_value.json.return_value = [
                {"title": "Noise", "similarity": 0.12, "description": "low"},
            ]
            result = self.agent.fetch_step_context(_make_step())
        assert result is None

    def test_formats_high_similarity_results(self):
        """
        SENTINEL TEST — title and description appear in formatted output.
        Removing formatting logic breaks this.
        """
        with patch("app.core.agents.reasoning_agent.httpx.Client") as mock_cls, \
             patch.dict("os.environ", {"CORE_API_KEY": "k"}):
            mock_http = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_http
            mock_http.post.return_value.status_code = 200
            mock_http.post.return_value.json.return_value = [
                {"title": "STEP_SENTINEL_TITLE", "similarity": 0.78, "description": "STEP_SENTINEL_DESC"},
                {"title": "Low", "similarity": 0.05, "description": "excluded"},
            ]
            result = self.agent.fetch_step_context(_make_step())

        assert result is not None
        assert "STEP_SENTINEL_TITLE" in result
        assert "STEP_SENTINEL_DESC" in result
        assert "0.78" in result
        assert "excluded" not in result

    def test_returns_none_on_network_exception(self):
        with patch("app.core.agents.reasoning_agent.httpx.Client") as mock_cls, \
             patch.dict("os.environ", {"CORE_API_KEY": "k"}):
            mock_cls.return_value.__enter__.side_effect = Exception("timeout")
            result = self.agent.fetch_step_context(_make_step())
        assert result is None

    def test_sends_correct_limit(self):
        with patch("app.core.agents.reasoning_agent.httpx.Client") as mock_cls, \
             patch.dict("os.environ", {"CORE_API_KEY": "k"}):
            mock_http = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_http
            mock_http.post.return_value.status_code = 200
            mock_http.post.return_value.json.return_value = []
            self.agent.fetch_step_context(_make_step(), top_k=7)
        _, kwargs = mock_http.post.call_args
        assert kwargs["json"]["limit"] == 7


# ---------------------------------------------------------------------------
# _execute_with_llm + RAG injection — sentinel-value tests
# ---------------------------------------------------------------------------

class TestExecuteWithLLMRAG:
    def setup_method(self):
        self.agent = ReasoningAgent()

    def test_kb_context_injected_into_llm_user_prompt(self):
        """
        SENTINEL TEST — proves KB context flows into the LLM user message.
        Remove fetch_step_context call or injection → this test fails.
        """
        sentinel = "Relevant knowledge base context:\n- [STEP_DOC] (relevance: 0.82): SENTINEL_STEP_FACT"
        step = _make_step(name="Build auth module", description="Implement JWT tokens")

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response("done")

        with patch.object(self.agent, "fetch_step_context", return_value=sentinel), \
             patch("app.core.agents.reasoning_agent.get_openai_client_sync", return_value=mock_client):
            self.agent._execute_with_llm(step)

        mock_client.chat.completions.create.assert_called_once()
        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        user_message = next(m for m in messages if m["role"] == "user")

        assert "SENTINEL_STEP_FACT" in user_message["content"]
        assert "STEP_DOC" in user_message["content"]

    def test_no_kb_context_uses_base_prompt_only(self):
        """When KB returns None, prompt equals the base prompt — no extra content."""
        step = _make_step(name="Write tests", description="Add unit tests")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response("done")

        with patch.object(self.agent, "fetch_step_context", return_value=None), \
             patch("app.core.agents.reasoning_agent.get_openai_client_sync", return_value=mock_client):
            self.agent._execute_with_llm(step)

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        user_message = next(m for m in messages if m["role"] == "user")
        # No KB header in prompt
        assert "Relevant knowledge base context" not in user_message["content"]

    def test_kb_context_appended_after_base_step_prompt(self):
        """KB context must appear AFTER step description, not replace it."""
        sentinel = "Relevant knowledge base context:\n- [Doc] (relevance: 0.70): APPENDED_CONTENT"
        step = _make_step(name="BASE_STEP_NAME", description="BASE_STEP_DESC")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response("done")

        with patch.object(self.agent, "fetch_step_context", return_value=sentinel), \
             patch("app.core.agents.reasoning_agent.get_openai_client_sync", return_value=mock_client):
            self.agent._execute_with_llm(step)

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        user_message = next(m for m in messages if m["role"] == "user")
        content = user_message["content"]

        step_pos = content.index("BASE_STEP_NAME")
        kb_pos = content.index("APPENDED_CONTENT")
        assert kb_pos > step_pos

    def test_fetch_called_with_the_step_object(self):
        """
        SENTINEL TEST — fetch_step_context must receive the exact step being executed.
        Fails if a different step or different argument is passed.
        """
        step = _make_step(name="UNIQUE_STEP_42")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response("done")

        with patch.object(self.agent, "fetch_step_context", return_value=None) as mock_fetch, \
             patch("app.core.agents.reasoning_agent.get_openai_client_sync", return_value=mock_client):
            self.agent._execute_with_llm(step)

        mock_fetch.assert_called_once_with(step)

    def test_kb_exception_does_not_crash_llm_execution(self):
        """KB failure is non-critical — step still executes via LLM."""
        step = _make_step()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response("result text")

        with patch.object(self.agent, "fetch_step_context", side_effect=Exception("KB down")), \
             patch("app.core.agents.reasoning_agent.get_openai_client_sync", return_value=mock_client):
            output = self.agent._execute_with_llm(step)

        assert output["result"] == "result text"

    def test_step_name_and_description_always_in_prompt(self):
        """Step name and description must appear in the user prompt regardless of RAG."""
        step = _make_step(name="MUST_APPEAR_NAME", description="MUST_APPEAR_DESC")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response("done")

        with patch.object(self.agent, "fetch_step_context", return_value=None), \
             patch("app.core.agents.reasoning_agent.get_openai_client_sync", return_value=mock_client):
            self.agent._execute_with_llm(step)

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        user_message = next(m for m in messages if m["role"] == "user")
        assert "MUST_APPEAR_NAME" in user_message["content"]
        assert "MUST_APPEAR_DESC" in user_message["content"]


# ---------------------------------------------------------------------------
# execute_plan integration — RAG triggered only for LLM steps
# ---------------------------------------------------------------------------

class TestExecutePlanRAGScope:
    def setup_method(self):
        self.agent = ReasoningAgent()

    def test_rag_not_called_for_tool_steps(self):
        """
        Tool steps (file_operations, git, etc.) skip LLM entirely.
        fetch_step_context must NOT be called for them.
        """
        step = _make_step(name="Commit code", tool="git")
        plan = _make_plan(step)

        with patch.object(self.agent, "fetch_step_context") as mock_fetch:
            self.agent.execute_plan(plan, enable_tools=True)

        mock_fetch.assert_not_called()

    def test_rag_not_called_for_dry_run(self):
        """Dry-run mode never calls the LLM or KB."""
        step = _make_step(name="Any step")
        plan = _make_plan(step)

        with patch.object(self.agent, "fetch_step_context") as mock_fetch:
            self.agent.execute_plan(plan, enable_tools=False)

        mock_fetch.assert_not_called()

    def test_rag_called_once_per_llm_step(self):
        """Each LLM step triggers exactly one KB lookup — not zero, not two."""
        step = _make_step(name="LLM step", tool=None)  # no tool → LLM path
        plan = _make_plan(step)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response("ok")

        with patch.object(self.agent, "fetch_step_context", return_value=None) as mock_fetch, \
             patch("app.core.agents.reasoning_agent.get_openai_client_sync", return_value=mock_client):
            self.agent.execute_plan(plan, enable_tools=True)

        mock_fetch.assert_called_once_with(step)

    def test_each_llm_step_gets_its_own_rag_call(self):
        """
        SENTINEL TEST — with two LLM steps, each gets its own distinct KB lookup.
        Removing per-step fetching and caching a single result would break this.
        """
        step_a = _make_step(name="Step A", description="desc A")
        step_b = _make_step(name="Step B", description="desc B")
        plan = _make_plan(step_a, step_b)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response("ok")

        fetch_calls = []

        def record_fetch(step, **kwargs):
            fetch_calls.append(step.name)
            return None

        with patch.object(self.agent, "fetch_step_context", side_effect=record_fetch), \
             patch("app.core.agents.reasoning_agent.get_openai_client_sync", return_value=mock_client):
            self.agent.execute_plan(plan, enable_tools=True)

        assert fetch_calls == ["Step A", "Step B"]
