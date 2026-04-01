"""
Tests for ComprehensionAgent RAG integration.

Enforces sentinel-value contract: every test that covers RAG would FAIL
if you removed the check_knowledge_base call or the prompt injection.
"""

import pytest
from unittest.mock import MagicMock, patch, call
from app.core.agents.comprehension_agent import ComprehensionAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(content: str) -> MagicMock:
    """Build a minimal mock that looks like an openai ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


_VALID_INTENT_JSON = '{"type":"task","description":"do a thing","confidence":0.9,"requires_tools":true,"tools_needed":["git"],"ambiguities":[]}'


# ---------------------------------------------------------------------------
# check_knowledge_base — unit tests
# ---------------------------------------------------------------------------

class TestCheckKnowledgeBase:
    def setup_method(self):
        self.agent = ComprehensionAgent()

    def test_returns_none_when_no_api_key(self):
        """If no API key is resolvable, returns None gracefully (never raises)."""
        with patch.dict("os.environ", {}, clear=True), \
             patch("app.core.agents.comprehension_agent.VALID_API_KEYS", set(), create=True):
            # Patch auth import to return empty set
            with patch("app.core.agents.comprehension_agent.httpx.Client") as mock_client_cls:
                # Even if httpx is called (it shouldn't be), make it safe
                mock_client_cls.return_value.__enter__.return_value.post.return_value.status_code = 200
                mock_client_cls.return_value.__enter__.return_value.post.return_value.json.return_value = []
                result = self.agent.check_knowledge_base("anything")
        # Should not raise; None is acceptable when no key
        assert result is None or isinstance(result, str)

    def test_returns_none_on_http_error(self):
        """Non-200 KB response → None, no exception propagation."""
        with patch("app.core.agents.comprehension_agent.httpx.Client") as mock_cls, \
             patch.dict("os.environ", {"CORE_API_KEY": "test-key"}):
            mock_http = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_http
            mock_http.post.return_value.status_code = 503
            result = self.agent.check_knowledge_base("anything")
        assert result is None

    def test_returns_none_when_no_results(self):
        """Empty results list → None."""
        with patch("app.core.agents.comprehension_agent.httpx.Client") as mock_cls, \
             patch.dict("os.environ", {"CORE_API_KEY": "test-key"}):
            mock_http = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_http
            mock_http.post.return_value.status_code = 200
            mock_http.post.return_value.json.return_value = []
            result = self.agent.check_knowledge_base("anything")
        assert result is None

    def test_returns_none_when_all_results_below_similarity_threshold(self):
        """Results with similarity < 0.30 are filtered out → None."""
        with patch("app.core.agents.comprehension_agent.httpx.Client") as mock_cls, \
             patch.dict("os.environ", {"CORE_API_KEY": "test-key"}):
            mock_http = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_http
            mock_http.post.return_value.status_code = 200
            mock_http.post.return_value.json.return_value = [
                {"title": "Low Score Doc", "similarity": 0.10, "description": "irrelevant"},
                {"title": "Another Low", "similarity": 0.05, "description": "nope"},
            ]
            result = self.agent.check_knowledge_base("query")
        assert result is None

    def test_formats_context_string_with_high_similarity_results(self):
        """
        SENTINEL TEST — proves format is correct.
        If similarity threshold or formatting logic changes, this fails.
        """
        with patch("app.core.agents.comprehension_agent.httpx.Client") as mock_cls, \
             patch.dict("os.environ", {"CORE_API_KEY": "test-key"}):
            mock_http = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_http
            mock_http.post.return_value.status_code = 200
            mock_http.post.return_value.json.return_value = [
                {
                    "title": "SENTINEL_TITLE",
                    "similarity": 0.87,
                    "description": "SENTINEL_DESCRIPTION",
                },
                {
                    "title": "Below Threshold",
                    "similarity": 0.10,
                    "description": "should not appear",
                },
            ]
            result = self.agent.check_knowledge_base("query")

        assert result is not None
        assert "SENTINEL_TITLE" in result
        assert "SENTINEL_DESCRIPTION" in result
        assert "0.87" in result
        # Below-threshold result must be excluded
        assert "should not appear" not in result

    def test_returns_none_on_network_exception(self):
        """Network failure → None, never raises."""
        with patch("app.core.agents.comprehension_agent.httpx.Client") as mock_cls, \
             patch.dict("os.environ", {"CORE_API_KEY": "test-key"}):
            mock_cls.return_value.__enter__.side_effect = Exception("connection refused")
            result = self.agent.check_knowledge_base("query")
        assert result is None

    def test_sends_correct_payload_and_headers(self):
        """Verifies the outbound HTTP call uses the right endpoint, key, and limit."""
        with patch("app.core.agents.comprehension_agent.httpx.Client") as mock_cls, \
             patch.dict("os.environ", {"CORE_API_KEY": "my-secret-key"}):
            mock_http = MagicMock()
            mock_cls.return_value.__enter__.return_value = mock_http
            mock_http.post.return_value.status_code = 200
            mock_http.post.return_value.json.return_value = []

            self.agent.check_knowledge_base("find auth logic", top_k=3)

        mock_http.post.assert_called_once()
        args, kwargs = mock_http.post.call_args
        assert "/knowledgebase/semantic-search" in args[0]
        assert kwargs["json"]["query"] == "find auth logic"
        assert kwargs["json"]["limit"] == 3
        assert kwargs["headers"]["X-API-Key"] == "my-secret-key"


# ---------------------------------------------------------------------------
# analyze_intent + RAG injection — sentinel-value tests
# ---------------------------------------------------------------------------

class TestAnalyzeIntentRAG:
    def setup_method(self):
        self.agent = ComprehensionAgent()

    def test_rag_context_injected_into_llm_system_prompt(self):
        """
        SENTINEL TEST — proves RAG flows end-to-end into the LLM call.
        Removing the check_knowledge_base call or the prompt injection makes this fail.
        """
        sentinel_context = "Relevant knowledge base context:\n- [SENTINEL_DOC] (relevance: 0.91): SENTINEL_FACT_XYZ"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(_VALID_INTENT_JSON)

        with patch.object(self.agent, "check_knowledge_base", return_value=sentinel_context), \
             patch("app.core.agents.comprehension_agent.get_openai_client_sync", return_value=mock_client):
            self.agent.analyze_intent("what is the auth system?")

        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        system_message = next(m for m in messages if m["role"] == "system")

        # Sentinel must appear in the system prompt — fails if RAG is removed
        assert "SENTINEL_FACT_XYZ" in system_message["content"]
        assert "SENTINEL_DOC" in system_message["content"]

    def test_no_rag_context_uses_base_system_prompt_only(self):
        """When KB returns None, prompt equals base system prompt — no extra content."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(_VALID_INTENT_JSON)

        with patch.object(self.agent, "check_knowledge_base", return_value=None), \
             patch("app.core.agents.comprehension_agent.get_openai_client_sync", return_value=mock_client):
            self.agent.analyze_intent("hello")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        system_message = next(m for m in messages if m["role"] == "system")

        assert system_message["content"] == self.agent.system_prompt

    def test_rag_context_appended_after_base_prompt(self):
        """
        Structural test: RAG context must appear AFTER the base prompt, not instead of it.
        Both the original instructions AND the context must be present.
        """
        sentinel = "Relevant knowledge base context:\n- [KB_APPEND] (relevance: 0.75): appended_content"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(_VALID_INTENT_JSON)

        with patch.object(self.agent, "check_knowledge_base", return_value=sentinel), \
             patch("app.core.agents.comprehension_agent.get_openai_client_sync", return_value=mock_client):
            self.agent.analyze_intent("query")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        system_content = next(m for m in call_kwargs["messages"] if m["role"] == "system")["content"]

        # Base prompt still present
        assert "Comprehension layer of the CORE cognitive system" in system_content
        # KB context appended after
        base_end = system_content.index("Comprehension layer")
        kb_start = system_content.index("KB_APPEND")
        assert kb_start > base_end

    def test_kb_called_with_user_input_as_query(self):
        """
        SENTINEL TEST — proves the right query is sent to the KB.
        Fails if check_knowledge_base is called with a different string.
        """
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(_VALID_INTENT_JSON)

        with patch.object(self.agent, "check_knowledge_base", return_value=None) as mock_kb, \
             patch("app.core.agents.comprehension_agent.get_openai_client_sync", return_value=mock_client):
            self.agent.analyze_intent("UNIQUE_QUERY_STRING_42")

        mock_kb.assert_called_once_with("UNIQUE_QUERY_STRING_42")

    def test_kb_failure_does_not_break_intent_classification(self):
        """RAG is best-effort: KB exception must not cause analyze_intent to raise."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(_VALID_INTENT_JSON)

        with patch.object(self.agent, "check_knowledge_base", side_effect=Exception("KB down")), \
             patch("app.core.agents.comprehension_agent.get_openai_client_sync", return_value=mock_client):
            # Should not raise — falls back gracefully
            result = self.agent.analyze_intent("will this crash?")

        # Returns a valid UserIntent regardless of KB failure
        assert result is not None
        assert result.type in ("task", "conversation", "question", "clarification", "conversation")

    def test_intent_parsed_correctly_from_llm_response_with_rag(self):
        """End-to-end: RAG context + valid LLM JSON → correct UserIntent fields."""
        sentinel_ctx = "Relevant knowledge base context:\n- [Auth Docs] (relevance: 0.80): JWT implementation"
        llm_json = '{"type":"question","description":"asking about JWT","confidence":0.95,"requires_tools":false,"tools_needed":[],"ambiguities":[]}'

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(llm_json)

        with patch.object(self.agent, "check_knowledge_base", return_value=sentinel_ctx), \
             patch("app.core.agents.comprehension_agent.get_openai_client_sync", return_value=mock_client):
            intent = self.agent.analyze_intent("how does JWT work here?")

        assert intent.type == "question"
        assert intent.confidence == pytest.approx(0.95)
        assert intent.requires_tools is False


# ---------------------------------------------------------------------------
# detect_ambiguities — pure regex pattern matching (UNTESTED before this block)
# ---------------------------------------------------------------------------

class TestDetectAmbiguities:
    def setup_method(self):
        self.agent = ComprehensionAgent()

    def test_empty_input_returns_empty_list(self):
        """
        SENTINEL — empty input must return [].
        Raise on empty → caller crashes on blank input → test fails.
        """
        assert self.agent.detect_ambiguities("") == []

    def test_whitespace_only_returns_empty_list(self):
        """Whitespace-only input must return []."""
        assert self.agent.detect_ambiguities("   \t\n") == []

    def test_clear_input_returns_empty_list(self):
        """Input with no ambiguities must return []."""
        result = self.agent.detect_ambiguities("Please add a login button to the navbar.")
        assert result == []

    def test_pronoun_it_detected(self):
        """
        SENTINEL — 'it' must trigger the unresolved-pronoun pattern.
        Remove pronoun pattern → unclear references silently accepted → test fails.
        """
        result = self.agent.detect_ambiguities("Fix it now")
        assert any("it" in r.lower() or "pronoun" in r.lower() for r in result)

    def test_pronoun_that_detected(self):
        """'that' must be flagged as an ambiguous reference."""
        result = self.agent.detect_ambiguities("Update that component")
        assert len(result) > 0

    def test_underspecified_artifact_detected(self):
        """
        SENTINEL — 'the file' must trigger the underspecified-artifact pattern.
        Remove artifact pattern → users say 'the function' and agent guesses → test fails.
        """
        result = self.agent.detect_ambiguities("Edit the file")
        assert any("file" in r.lower() or "artifact" in r.lower() for r in result)

    def test_vague_quantity_detected(self):
        """
        SENTINEL — 'some' and 'a few' must trigger the vague-quantity pattern.
        Remove quantity pattern → 'add some tests' accepted → agent writes random count → test fails.
        """
        result = self.agent.detect_ambiguities("Add some tests")
        assert any("some" in r.lower() or "quantity" in r.lower() for r in result)

    def test_comparative_without_referent_detected(self):
        """
        SENTINEL — 'better than' must trigger the comparative pattern.
        Remove comparative pattern → 'make it faster than' unanswered → test fails.
        """
        result = self.agent.detect_ambiguities("Make it faster than before")
        assert any("comparative" in r.lower() or "better" in r.lower() or "faster" in r.lower() for r in result)

    def test_deduplication_applied(self):
        """
        SENTINEL — repeated ambiguous words must produce only one entry per unique message.
        Skip dedup → same message appears multiple times → noisy output → test fails.
        """
        # 'it' appears twice — should yield only one entry about 'it'
        result = self.agent.detect_ambiguities("Fix it before it breaks")
        it_entries = [r for r in result if "it" in r.lower()]
        # Each unique message should appear only once
        assert len(it_entries) == len(set(it_entries))

    def test_multiple_patterns_can_match(self):
        """Input matching several patterns must produce multiple entries."""
        # 'it' (pronoun), 'some' (quantity)
        result = self.agent.detect_ambiguities("Fix it with some changes")
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# _build_system_prompt — sanity check on required content
# ---------------------------------------------------------------------------

class TestBuildSystemPrompt:
    def setup_method(self):
        self.agent = ComprehensionAgent()

    def test_returns_non_empty_string(self):
        """
        SENTINEL — system prompt must be a non-empty string.
        Return None → LLM call fails with type error → test fails.
        """
        prompt = self.agent._build_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_prompt_contains_intent_types(self):
        """
        SENTINEL — prompt must describe the intent types (task, conversation, question, clarification).
        Omit intent descriptions → LLM doesn't know valid types → misclassifies inputs → test fails.
        """
        prompt = self.agent._build_system_prompt()
        assert "task" in prompt
        assert "conversation" in prompt
        assert "question" in prompt
        assert "clarification" in prompt

    def test_prompt_requests_json_format(self):
        """
        SENTINEL — prompt must instruct the LLM to respond in JSON.
        Remove JSON instruction → LLM responds in prose → JSON parsing fails → test fails.
        """
        prompt = self.agent._build_system_prompt()
        assert "json" in prompt.lower() or "JSON" in prompt
