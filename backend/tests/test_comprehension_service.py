"""
Tests for ComprehensionService — the CORE intent-parsing and routing engine.

Sentinel-value contract:
- Remove keyword branch for "create/build/generate" → CREATION action misclassified → test fails
- Remove urgency boost in _adjust_priority → CRITICAL input gets wrong priority → test fails
- Remove multi-agent path in _determine_handling_mode → high-complexity routed wrong → test fails
- Remove intent_weight from _score_confidence → weighted average wrong → test fails
- Remove _parse_intent call in comprehend → no intent in result → test fails
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.comprehension_models import (
    ActionType,
    CapabilityMatch,
    ComprehensionInput,
    ComplexityScore,
    ContextMatch,
    HandlingMode,
    IntentAnalysis,
    SourceType,
    UrgencyLevel,
)
from app.services.comprehension_service import ComprehensionService


# ===========================================================================
# Helpers
# ===========================================================================


def _make_service() -> ComprehensionService:
    svc = ComprehensionService()
    svc._initialized = True
    return svc


def _make_input(content: str) -> ComprehensionInput:
    return ComprehensionInput(content=content, source_type=SourceType.USER)


def _make_intent(
    action: ActionType = ActionType.QUERY,
    confidence: float = 0.8,
    urgency: UrgencyLevel = UrgencyLevel.MEDIUM,
) -> IntentAnalysis:
    return IntentAnalysis(
        action_type=action,
        summary="test intent",
        confidence=confidence,
        urgency=urgency,
    )


def _make_complexity(overall: float = 0.3, breadth: float = 0.3) -> ComplexityScore:
    return ComplexityScore(
        overall=overall,
        reasoning_depth=overall * 0.8,
        breadth=breadth,
        novelty=0.5,
    )


def _make_capabilities(
    has_agents: bool = True,
    agent_ids: list | None = None,
    best_score: float = 0.7,
) -> CapabilityMatch:
    return CapabilityMatch(
        has_capable_agents=has_agents,
        matched_agent_ids=agent_ids or (["agent-1"] if has_agents else []),
        best_agent_score=best_score,
    )


# ===========================================================================
# _heuristic_intent_parse — keyword routing
# ===========================================================================


class TestHeuristicIntentParse:
    def test_creation_keywords_map_to_creation(self):
        """
        SENTINEL — "create", "build", "generate", "make", "write" must map to CREATION.
        Remove the creation keyword branch → CREATION inputs misclassified → test fails.
        """
        svc = _make_service()
        for word in ["create", "build", "generate", "make", "write"]:
            intent, _, _, _, _ = svc._heuristic_intent_parse(
                _make_input(f"Please {word} a report")
            )
            assert (
                intent.action_type == ActionType.CREATION
            ), f"'{word}' should map to CREATION, got {intent.action_type}"

    def test_analysis_keywords_map_to_analysis(self):
        """
        SENTINEL — "analyze", "evaluate", "assess", "compare" must map to ANALYSIS.
        Remove branch → test fails.
        """
        svc = _make_service()
        for word in ["analyze", "evaluate", "assess", "compare"]:
            intent, _, _, _, _ = svc._heuristic_intent_parse(
                _make_input(f"Please {word} the situation")
            )
            assert (
                intent.action_type == ActionType.ANALYSIS
            ), f"'{word}' should map to ANALYSIS, got {intent.action_type}"

    def test_question_words_map_to_query(self):
        """
        SENTINEL — "what", "who", "how", "why", "?" must map to QUERY.
        Remove branch → test fails.
        """
        svc = _make_service()
        for phrase in [
            "what is this?",
            "who did it",
            "how does it work",
            "why did this happen",
        ]:
            intent, _, _, _, _ = svc._heuristic_intent_parse(_make_input(phrase))
            assert (
                intent.action_type == ActionType.QUERY
            ), f"'{phrase}' should map to QUERY, got {intent.action_type}"

    def test_command_keywords_map_to_command(self):
        """SENTINEL — "run", "execute", "deploy" must map to COMMAND."""
        svc = _make_service()
        for word in ["run", "execute", "deploy"]:
            intent, _, _, _, _ = svc._heuristic_intent_parse(
                _make_input(f"Please {word} the service")
            )
            assert intent.action_type == ActionType.COMMAND

    def test_planning_keywords_map_to_planning(self):
        """SENTINEL — "plan", "schedule", "roadmap", "strategy" must map to PLANNING."""
        svc = _make_service()
        for word in ["plan", "schedule", "roadmap", "strategy"]:
            intent, _, _, _, _ = svc._heuristic_intent_parse(
                _make_input(f"Help me {word} the project")
            )
            assert intent.action_type == ActionType.PLANNING

    def test_no_keywords_falls_back_to_conversation(self):
        """Unrecognized input must fall through to CONVERSATION."""
        svc = _make_service()
        intent, _, _, _, _ = svc._heuristic_intent_parse(_make_input("hello there"))
        assert intent.action_type == ActionType.CONVERSATION

    def test_urgent_keywords_set_critical_urgency(self):
        """
        SENTINEL — "urgent", "asap", "critical", "emergency" must set CRITICAL urgency.
        Remove urgency branch → wrong priority returned → test fails.
        """
        svc = _make_service()
        for word in ["urgent", "asap", "critical", "emergency"]:
            intent, _, _, priority, _ = svc._heuristic_intent_parse(
                _make_input(f"This is {word}: fix the bug")
            )
            assert (
                intent.urgency == UrgencyLevel.CRITICAL
            ), f"'{word}' should set CRITICAL urgency"
            assert priority == 9, f"CRITICAL should yield priority 9, got {priority}"

    def test_low_priority_keywords_set_low_urgency(self):
        """ "no rush", "whenever" must set LOW urgency."""
        svc = _make_service()
        intent, _, _, priority, _ = svc._heuristic_intent_parse(
            _make_input("no rush, finish this whenever you can")
        )
        assert intent.urgency == UrgencyLevel.LOW
        assert priority == 3

    def test_heuristic_confidence_is_lower_than_llm(self):
        """Heuristic parse must return confidence ≤ 0.5 (signals LLM preferred)."""
        svc = _make_service()
        intent, _, _, _, _ = svc._heuristic_intent_parse(_make_input("do something"))
        assert intent.confidence <= 0.5

    def test_complexity_scales_with_word_count(self):
        """
        SENTINEL — longer text must produce higher complexity.overall.
        Remove word_count scaling → all inputs same complexity → test fails.
        """
        svc = _make_service()
        short_input = _make_input("fix bug")
        long_input = _make_input(" ".join(["analyze"] * 150))

        _, short_complexity, _, _, _ = svc._heuristic_intent_parse(short_input)
        _, long_complexity, _, _, _ = svc._heuristic_intent_parse(long_input)

        assert long_complexity.overall > short_complexity.overall

    def test_keywords_extracted_from_content(self):
        """Keywords list must be non-empty for substantive input."""
        svc = _make_service()
        intent, _, _, _, keywords = svc._heuristic_intent_parse(
            _make_input("analyze the database performance metrics")
        )
        assert len(keywords) > 0


# ===========================================================================
# _determine_handling_mode — routing decision
# ===========================================================================


class TestDetermineHandlingMode:
    def test_no_agents_conversation_still_single(self):
        """CONVERSATION with no capable agents must still try SINGLE_AGENT (any agent can converse)."""
        svc = _make_service()
        intent = _make_intent(action=ActionType.CONVERSATION)
        result = svc._determine_handling_mode(
            intent,
            _make_complexity(0.3),
            _make_capabilities(has_agents=False),
        )
        assert result == HandlingMode.SINGLE_AGENT

    def test_no_agents_non_conversation_requires_human(self):
        """
        SENTINEL — non-conversational input with no agents must return HUMAN_REQUIRED.
        Remove human_required path → unsatisfiable requests dispatched to nobody → test fails.
        """
        svc = _make_service()
        intent = _make_intent(action=ActionType.COMMAND)
        result = svc._determine_handling_mode(
            intent,
            _make_complexity(0.3),
            _make_capabilities(has_agents=False),
        )
        assert result == HandlingMode.HUMAN_REQUIRED

    def test_simple_query_with_high_confidence_returns_no_agent(self):
        """
        SENTINEL — high-confidence QUERY with low complexity can be answered from context.
        Remove NO_AGENT path → every simple query wastes an agent call → test fails.
        """
        svc = _make_service()
        intent = _make_intent(action=ActionType.QUERY, confidence=0.9)
        result = svc._determine_handling_mode(
            intent,
            _make_complexity(overall=0.1, breadth=0.1),
            _make_capabilities(has_agents=True),
        )
        assert result == HandlingMode.NO_AGENT

    def test_high_complexity_broad_task_returns_multi_agent(self):
        """
        SENTINEL — overall>0.7 AND breadth>0.6 must route to MULTI_AGENT.
        Remove multi-agent path → complex tasks routed to a single agent → test fails.
        """
        svc = _make_service()
        intent = _make_intent(action=ActionType.ANALYSIS, confidence=0.7)
        result = svc._determine_handling_mode(
            intent,
            _make_complexity(overall=0.8, breadth=0.7),
            _make_capabilities(has_agents=True, agent_ids=["a1", "a2", "a3"]),
        )
        assert result == HandlingMode.MULTI_AGENT

    def test_many_agent_matches_with_broad_breadth_returns_multi_agent(self):
        """
        SENTINEL — >2 matched agents AND breadth>0.5 must be MULTI_AGENT.
        Remove this condition → broad multi-skill tasks routed to single agent → test fails.
        """
        svc = _make_service()
        intent = _make_intent(action=ActionType.CREATION, confidence=0.6)
        result = svc._determine_handling_mode(
            intent,
            _make_complexity(overall=0.5, breadth=0.6),
            _make_capabilities(
                has_agents=True,
                agent_ids=["a1", "a2", "a3"],
                best_score=0.6,
            ),
        )
        assert result == HandlingMode.MULTI_AGENT

    def test_normal_task_returns_single_agent(self):
        """Normal task with moderate complexity returns SINGLE_AGENT."""
        svc = _make_service()
        intent = _make_intent(action=ActionType.CREATION, confidence=0.7)
        result = svc._determine_handling_mode(
            intent,
            _make_complexity(overall=0.4, breadth=0.3),
            _make_capabilities(has_agents=True, agent_ids=["a1"]),
        )
        assert result == HandlingMode.SINGLE_AGENT


# ===========================================================================
# _adjust_priority — urgency-based priority
# ===========================================================================


class TestAdjustPriority:
    def test_critical_urgency_adds_3(self):
        """
        SENTINEL — CRITICAL urgency must boost base priority by +3.
        Remove urgency_boost map → CRITICAL treated same as MEDIUM → test fails.
        """
        svc = _make_service()
        intent = _make_intent(urgency=UrgencyLevel.CRITICAL)
        result = svc._adjust_priority(5, intent, ContextMatch())
        assert result == 8

    def test_high_urgency_adds_1(self):
        """HIGH urgency must boost by +1."""
        svc = _make_service()
        intent = _make_intent(urgency=UrgencyLevel.HIGH)
        result = svc._adjust_priority(5, intent, ContextMatch())
        assert result == 6

    def test_medium_urgency_no_change(self):
        """MEDIUM urgency must leave priority unchanged."""
        svc = _make_service()
        intent = _make_intent(urgency=UrgencyLevel.MEDIUM)
        result = svc._adjust_priority(5, intent, ContextMatch())
        assert result == 5

    def test_low_urgency_subtracts_1(self):
        """LOW urgency must decrease priority by 1."""
        svc = _make_service()
        intent = _make_intent(urgency=UrgencyLevel.LOW)
        result = svc._adjust_priority(5, intent, ContextMatch())
        assert result == 4

    def test_none_urgency_subtracts_2(self):
        """NONE urgency must decrease priority by 2."""
        svc = _make_service()
        intent = _make_intent(urgency=UrgencyLevel.NONE)
        result = svc._adjust_priority(5, intent, ContextMatch())
        assert result == 3

    def test_priority_clamped_to_1_at_minimum(self):
        """Priority must never go below 1."""
        svc = _make_service()
        intent = _make_intent(urgency=UrgencyLevel.NONE)
        result = svc._adjust_priority(1, intent, ContextMatch())
        assert result == 1

    def test_priority_clamped_to_10_at_maximum(self):
        """Priority must never exceed 10."""
        svc = _make_service()
        intent = _make_intent(urgency=UrgencyLevel.CRITICAL)
        result = svc._adjust_priority(9, intent, ContextMatch())
        assert result == 10

    def test_strong_context_lowers_priority_by_1(self):
        """
        SENTINEL — if best_match_score > 0.8, priority decreases by 1 (less uncertain).
        Remove context adjustment → high-context queries over-prioritized → test fails.
        """
        svc = _make_service()
        intent = _make_intent(urgency=UrgencyLevel.MEDIUM)
        context = ContextMatch(has_relevant_context=True, best_match_score=0.9)
        result = svc._adjust_priority(5, intent, context)
        assert result == 4


# ===========================================================================
# _score_confidence — weighted average
# ===========================================================================


class TestScoreConfidence:
    def test_high_intent_confidence_produces_high_score(self):
        """
        SENTINEL — intent confidence (weight=0.5) dominates the score.
        Remove intent_weight → high-confidence intents don't raise overall → test fails.
        """
        svc = _make_service()
        intent = _make_intent(confidence=1.0)
        result = svc._score_confidence(
            intent,
            ContextMatch(),  # no context → 0.3 base
            _make_capabilities(has_agents=False),  # no agents → 0.3 base
        )
        # intent=1.0*0.5 + context=0.3*0.25 + cap=0.3*0.25 = 0.5 + 0.075 + 0.075 = 0.65
        assert abs(result - 0.65) < 0.01

    def test_zero_intent_confidence_gives_low_score(self):
        """Zero intent confidence must drive the overall score down."""
        svc = _make_service()
        intent = _make_intent(confidence=0.0)
        result = svc._score_confidence(
            intent,
            ContextMatch(),
            _make_capabilities(has_agents=False),
        )
        # intent=0*0.5 + context=0.3*0.25 + cap=0.3*0.25 = 0.15
        assert abs(result - 0.15) < 0.01

    def test_capable_agents_raise_score(self):
        """
        SENTINEL — capable agents must boost capability_score (base 0.3 → score+0.3).
        Remove capable_agents check → agency bonus never applied → test fails.
        """
        svc = _make_service()
        intent = _make_intent(confidence=0.5)
        no_agents = _make_capabilities(has_agents=False)
        with_agents = _make_capabilities(has_agents=True, best_score=0.7)

        score_no_agents = svc._score_confidence(intent, ContextMatch(), no_agents)
        score_with_agents = svc._score_confidence(intent, ContextMatch(), with_agents)

        assert score_with_agents > score_no_agents

    def test_strong_context_raises_score(self):
        """
        SENTINEL — relevant context with best_match_score=0.9 must lift context_score.
        Remove context score branch → context presence doesn't help confidence → test fails.
        """
        svc = _make_service()
        intent = _make_intent(confidence=0.5)
        no_context = ContextMatch()
        with_context = ContextMatch(has_relevant_context=True, best_match_score=0.9)

        score_no = svc._score_confidence(
            intent, no_context, _make_capabilities(has_agents=False)
        )
        score_yes = svc._score_confidence(
            intent, with_context, _make_capabilities(has_agents=False)
        )

        assert score_yes > score_no

    def test_score_always_between_0_and_1(self):
        """Score must be clamped to [0, 1]."""
        svc = _make_service()
        intent = _make_intent(confidence=1.0)
        context = ContextMatch(has_relevant_context=True, best_match_score=1.0)
        caps = _make_capabilities(has_agents=True, best_score=1.0)
        result = svc._score_confidence(intent, context, caps)
        assert 0.0 <= result <= 1.0


# ===========================================================================
# comprehend() — end-to-end with mocked I/O
# ===========================================================================


class TestComprehendEndToEnd:
    @pytest.mark.asyncio
    async def test_comprehend_returns_completed_status(self):
        """
        SENTINEL — comprehend() must return COMPLETED status for a valid input.
        Remove status assignment → callers can't distinguish success from failure → test fails.
        """
        svc = _make_service()

        with patch.object(
            svc, "_call_llm_for_intent", new=AsyncMock(side_effect=Exception("no LLM"))
        ), patch.object(
            svc, "_search_memory_context", new=AsyncMock(return_value=ContextMatch())
        ), patch.object(
            svc, "_match_capabilities", new=AsyncMock(return_value=_make_capabilities())
        ), patch(
            "app.services.comprehension_service.comprehension_repository"
        ) as mock_repo:
            mock_repo.store_comprehension_result = AsyncMock(return_value=None)

            result = await svc.comprehend(_make_input("create a report"))

        from app.models.comprehension_models import ComprehensionStatus

        assert result.status == ComprehensionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_comprehend_includes_intent_in_result(self):
        """
        SENTINEL — comprehend() must populate result.intent from _parse_intent.
        Remove _parse_intent call → no intent in result → test fails.
        """
        svc = _make_service()

        with patch.object(
            svc, "_call_llm_for_intent", new=AsyncMock(side_effect=Exception("no LLM"))
        ), patch.object(
            svc, "_search_memory_context", new=AsyncMock(return_value=ContextMatch())
        ), patch.object(
            svc, "_match_capabilities", new=AsyncMock(return_value=_make_capabilities())
        ), patch(
            "app.services.comprehension_service.comprehension_repository"
        ) as mock_repo:
            mock_repo.store_comprehension_result = AsyncMock(return_value=None)

            result = await svc.comprehend(_make_input("analyze the database"))

        assert result.intent is not None
        assert result.intent.action_type == ActionType.ANALYSIS

    @pytest.mark.asyncio
    async def test_comprehend_stores_result(self):
        """
        SENTINEL — comprehend() must call store_comprehension_result to persist analytics.
        Remove the store call → no history, analytics broken → test fails.
        """
        svc = _make_service()

        with patch.object(
            svc, "_call_llm_for_intent", new=AsyncMock(side_effect=Exception("no LLM"))
        ), patch.object(
            svc, "_search_memory_context", new=AsyncMock(return_value=ContextMatch())
        ), patch.object(
            svc, "_match_capabilities", new=AsyncMock(return_value=_make_capabilities())
        ), patch(
            "app.services.comprehension_service.comprehension_repository"
        ) as mock_repo:
            mock_repo.store_comprehension_result = AsyncMock(return_value=None)

            await svc.comprehend(_make_input("generate a plan"))

        mock_repo.store_comprehension_result.assert_called_once()

    @pytest.mark.asyncio
    async def test_comprehend_error_returns_failed_status(self):
        """
        SENTINEL — if all internal steps fail, comprehend() must return FAILED status
        (not raise) so the caller can handle gracefully.
        Remove exception handler → caller crashes → test fails.
        """
        svc = _make_service()

        with patch.object(
            svc, "_parse_intent", new=AsyncMock(side_effect=RuntimeError("boom"))
        ):
            result = await svc.comprehend(_make_input("something"))

        from app.models.comprehension_models import ComprehensionStatus

        assert result.status == ComprehensionStatus.FAILED
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_comprehend_records_processing_time(self):
        """processing_time_ms must be set and non-negative."""
        svc = _make_service()

        with patch.object(
            svc, "_call_llm_for_intent", new=AsyncMock(side_effect=Exception("no LLM"))
        ), patch.object(
            svc, "_search_memory_context", new=AsyncMock(return_value=ContextMatch())
        ), patch.object(
            svc, "_match_capabilities", new=AsyncMock(return_value=_make_capabilities())
        ), patch(
            "app.services.comprehension_service.comprehension_repository"
        ) as mock_repo:
            mock_repo.store_comprehension_result = AsyncMock(return_value=None)

            result = await svc.comprehend(_make_input("run the tests"))

        assert result.processing_time_ms >= 0
