"""
Comprehension Engine Tests

Comprehensive tests for the Comprehension Engine following AAA format.
Tests cover intent parsing, context matching, capability matching,
complexity analysis, end-to-end flow, feedback, and edge cases.

20+ test scenarios covering the full Comprehension Engine.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Any
from uuid import uuid4, UUID
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from app.models.comprehension_models import (
    ComprehensionInput,
    ComprehensionResult,
    IntentAnalysis,
    ContextMatch,
    CapabilityMatch,
    ComplexityScore,
    ComprehensionFeedback,
    SystemCapability,
    SourceType,
    ActionType,
    UrgencyLevel,
    HandlingMode,
    ComprehensionStatus,
    ExtractedEntity,
    MemoryMatch,
    MatchedCapability,
)
from app.models.task_models import TaskType
from app.services.comprehension_service import ComprehensionService


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def comprehension_service():
    """Create a ComprehensionService instance for testing."""
    service = ComprehensionService()
    service._initialized = True
    return service


@pytest.fixture
def sample_user_input():
    """Create a sample user input."""
    return ComprehensionInput(
        content="Research the latest developments in quantum computing and summarize the key findings",
        source_type=SourceType.USER,
        source_id="user-123",
        metadata={"session": "test"},
        conversation_id="conv-456",
    )


@pytest.fixture
def sample_agent_input():
    """Create a sample agent input."""
    return ComprehensionInput(
        content="I need help analyzing the performance metrics from the last sprint",
        source_type=SourceType.AGENT,
        source_id="agent-researcher-001",
        metadata={"agent_role": "researcher"},
    )


@pytest.fixture
def sample_system_input():
    """Create a sample system input."""
    return ComprehensionInput(
        content="Alert: Memory usage has exceeded 85% threshold on node-3",
        source_type=SourceType.SYSTEM,
        source_id="system-monitor",
        metadata={"alert_level": "warning"},
    )


@pytest.fixture
def sample_tool_output():
    """Create a sample tool output input."""
    return ComprehensionInput(
        content="Search results returned 15 documents matching 'consciousness emergence protocol'",
        source_type=SourceType.TOOL_OUTPUT,
        source_id="obsidian-search",
        metadata={"tool": "mcp-obsidian", "result_count": 15},
    )


@pytest.fixture
def sample_llm_response():
    """Create a sample LLM intent parsing response."""
    return {
        "action_type": "query",
        "summary": "Research quantum computing developments and summarize findings",
        "entities": [
            {"name": "quantum computing", "entity_type": "topic", "confidence": 0.95},
            {"name": "latest developments", "entity_type": "temporal", "confidence": 0.8},
        ],
        "urgency": "medium",
        "confidence": 0.85,
        "keywords": ["research", "quantum", "computing", "developments", "summarize"],
        "requires_clarification": False,
        "clarification_questions": [],
        "complexity": {
            "overall": 0.6,
            "reasoning_depth": 0.5,
            "breadth": 0.4,
            "novelty": 0.7,
            "estimated_duration_seconds": 120,
        },
        "suggested_task_type": "research",
        "suggested_priority": 5,
        "suggested_capabilities": ["web_search", "analysis", "writing"],
    }


@pytest.fixture
def mock_agent_rows():
    """Create mock database rows for agent instances."""
    return [
        {
            "id": uuid4(),
            "agent_id": "researcher-001",
            "agent_role": "researcher",
            "capabilities": '["web_search", "analysis", "writing"]',
            "status": "ready",
        },
        {
            "id": uuid4(),
            "agent_id": "analyst-001",
            "agent_role": "analyst",
            "capabilities": '["analysis", "data_processing", "monitoring"]',
            "status": "ready",
        },
        {
            "id": uuid4(),
            "agent_id": "coder-001",
            "agent_role": "coder",
            "capabilities": '["code", "debugging", "testing"]',
            "status": "ready",
        },
    ]


# =============================================================================
# MODEL TESTS
# =============================================================================

class TestComprehensionModels:
    """Tests for Comprehension Engine Pydantic models."""

    def test_comprehension_input_creation(self):
        """Test creating a valid ComprehensionInput."""
        # Arrange
        content = "Analyze the system performance"
        source_type = SourceType.USER

        # Act
        input_data = ComprehensionInput(
            content=content,
            source_type=source_type,
            source_id="user-1",
        )

        # Assert
        assert input_data.content == content
        assert input_data.source_type == SourceType.USER
        assert input_data.source_id == "user-1"
        assert input_data.id is not None

    def test_comprehension_input_rejects_empty_content(self):
        """Test that empty content is rejected."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Content cannot be empty"):
            ComprehensionInput(
                content="",
                source_type=SourceType.USER,
            )

    def test_comprehension_input_rejects_whitespace_content(self):
        """Test that whitespace-only content is rejected."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Content cannot be empty"):
            ComprehensionInput(
                content="   \n\t  ",
                source_type=SourceType.USER,
            )

    def test_intent_analysis_creation(self):
        """Test creating a valid IntentAnalysis."""
        # Arrange & Act
        intent = IntentAnalysis(
            action_type=ActionType.QUERY,
            summary="Find information about AI",
            confidence=0.85,
            entities=[
                ExtractedEntity(name="AI", entity_type="topic", confidence=0.9)
            ],
            urgency=UrgencyLevel.MEDIUM,
            keywords=["AI", "information"],
        )

        # Assert
        assert intent.action_type == ActionType.QUERY
        assert intent.confidence == 0.85
        assert len(intent.entities) == 1
        assert intent.entities[0].name == "AI"

    def test_context_match_total_matches(self):
        """Test ContextMatch total_matches property."""
        # Arrange
        context = ContextMatch(
            semantic_matches=[
                MemoryMatch(memory_id=uuid4(), content="a", similarity=0.8, memory_tier="semantic"),
                MemoryMatch(memory_id=uuid4(), content="b", similarity=0.7, memory_tier="semantic"),
            ],
            episodic_matches=[
                MemoryMatch(memory_id=uuid4(), content="c", similarity=0.6, memory_tier="episodic"),
            ],
            procedural_matches=[],
            has_relevant_context=True,
            best_match_score=0.8,
        )

        # Act
        total = context.total_matches

        # Assert
        assert total == 3

    def test_comprehension_result_to_task_kwargs(self):
        """Test converting ComprehensionResult to Task kwargs."""
        # Arrange
        result = ComprehensionResult(
            input=ComprehensionInput(
                content="Build a REST API",
                source_type=SourceType.USER,
            ),
            intent=IntentAnalysis(
                action_type=ActionType.CREATION,
                summary="Build a REST API",
                confidence=0.9,
            ),
            suggested_task_type="code",
            suggested_priority=7,
            suggested_capabilities=["code", "api_design"],
            confidence=0.85,
        )

        # Act
        task_kwargs = result.to_task_kwargs()

        # Assert
        assert task_kwargs["task_type"] == "code"
        assert task_kwargs["priority"] == 7
        assert task_kwargs["required_capabilities"] == ["code", "api_design"]
        assert "original_input" in task_kwargs["payload"]
        assert task_kwargs["payload"]["action_type"] == "creation"

    def test_complexity_score_validation(self):
        """Test ComplexityScore validation bounds."""
        # Arrange & Act
        score = ComplexityScore(
            overall=0.7,
            reasoning_depth=0.8,
            breadth=0.5,
            novelty=0.6,
        )

        # Assert
        assert score.overall == 0.7
        assert 0.0 <= score.overall <= 1.0

    def test_all_source_types(self):
        """Test all source types can be used."""
        # Arrange & Act & Assert
        for st in SourceType:
            input_data = ComprehensionInput(
                content=f"Test {st.value}",
                source_type=st,
            )
            assert input_data.source_type == st


# =============================================================================
# INTENT PARSING TESTS
# =============================================================================

class TestIntentParsing:
    """Tests for intent parsing logic."""

    def test_heuristic_parse_query(self, comprehension_service):
        """Test heuristic parsing of a query."""
        # Arrange
        input_data = ComprehensionInput(
            content="What is the current status of the deployment?",
            source_type=SourceType.USER,
        )

        # Act
        intent, complexity, task_type, priority, caps = (
            comprehension_service._heuristic_intent_parse(input_data)
        )

        # Assert
        assert intent.action_type == ActionType.QUERY
        assert intent.confidence == 0.4  # Heuristic confidence is lower
        assert task_type == TaskType.RESEARCH

    def test_heuristic_parse_creation(self, comprehension_service):
        """Test heuristic parsing of a creation request."""
        # Arrange
        input_data = ComprehensionInput(
            content="Create a new database migration script for the users table",
            source_type=SourceType.USER,
        )

        # Act
        intent, complexity, task_type, priority, caps = (
            comprehension_service._heuristic_intent_parse(input_data)
        )

        # Assert
        assert intent.action_type == ActionType.CREATION
        assert task_type == TaskType.WRITING

    def test_heuristic_parse_command(self, comprehension_service):
        """Test heuristic parsing of a command."""
        # Arrange
        input_data = ComprehensionInput(
            content="Execute the deployment pipeline and restart the services",
            source_type=SourceType.USER,
        )

        # Act
        intent, complexity, task_type, priority, caps = (
            comprehension_service._heuristic_intent_parse(input_data)
        )

        # Assert
        assert intent.action_type == ActionType.COMMAND
        assert task_type == TaskType.CODE

    def test_heuristic_parse_analysis(self, comprehension_service):
        """Test heuristic parsing of an analysis request."""
        # Arrange
        input_data = ComprehensionInput(
            content="Analyze the error logs from last week and compare with previous month",
            source_type=SourceType.USER,
        )

        # Act
        intent, complexity, task_type, priority, caps = (
            comprehension_service._heuristic_intent_parse(input_data)
        )

        # Assert
        assert intent.action_type == ActionType.ANALYSIS
        assert task_type == TaskType.ANALYSIS

    def test_heuristic_parse_monitoring(self, comprehension_service):
        """Test heuristic parsing of a monitoring request."""
        # Arrange
        input_data = ComprehensionInput(
            content="Monitor the CPU usage on the production servers",
            source_type=SourceType.USER,
        )

        # Act
        intent, complexity, task_type, priority, caps = (
            comprehension_service._heuristic_intent_parse(input_data)
        )

        # Assert
        assert intent.action_type == ActionType.MONITORING
        assert task_type == TaskType.MONITORING

    def test_heuristic_parse_planning(self, comprehension_service):
        """Test heuristic parsing of a planning request."""
        # Arrange
        input_data = ComprehensionInput(
            content="Plan the roadmap for Q3 and schedule the milestones",
            source_type=SourceType.USER,
        )

        # Act
        intent, complexity, task_type, priority, caps = (
            comprehension_service._heuristic_intent_parse(input_data)
        )

        # Assert
        assert intent.action_type == ActionType.PLANNING
        assert task_type == TaskType.PLANNING

    def test_heuristic_parse_urgent_input(self, comprehension_service):
        """Test heuristic parsing detects urgency."""
        # Arrange
        input_data = ComprehensionInput(
            content="URGENT: The production database is down, need immediate fix",
            source_type=SourceType.USER,
        )

        # Act
        intent, complexity, task_type, priority, caps = (
            comprehension_service._heuristic_intent_parse(input_data)
        )

        # Assert
        assert intent.urgency == UrgencyLevel.CRITICAL
        assert priority == 9

    def test_heuristic_parse_low_priority(self, comprehension_service):
        """Test heuristic parsing detects low priority."""
        # Arrange
        input_data = ComprehensionInput(
            content="No rush, but when you can, please look into the documentation updates",
            source_type=SourceType.USER,
        )

        # Act
        intent, complexity, task_type, priority, caps = (
            comprehension_service._heuristic_intent_parse(input_data)
        )

        # Assert
        assert intent.urgency == UrgencyLevel.LOW
        assert priority == 3

    def test_build_intent_from_llm_response(self, comprehension_service, sample_llm_response):
        """Test building intent from a valid LLM response."""
        # Arrange
        llm_result = sample_llm_response

        # Act
        intent, complexity, task_type, priority, caps = (
            comprehension_service._build_intent_from_llm(llm_result)
        )

        # Assert
        assert intent.action_type == ActionType.QUERY
        assert intent.confidence == 0.85
        assert len(intent.entities) == 2
        assert intent.entities[0].name == "quantum computing"
        assert complexity.overall == 0.6
        assert task_type == "research"
        assert priority == 5
        assert "web_search" in caps

    def test_build_intent_handles_invalid_action_type(self, comprehension_service):
        """Test graceful handling of invalid action type from LLM."""
        # Arrange
        llm_result = {
            "action_type": "invalid_type",
            "summary": "test",
            "confidence": 0.5,
        }

        # Act
        intent, _, _, _, _ = comprehension_service._build_intent_from_llm(llm_result)

        # Assert
        assert intent.action_type == ActionType.UNKNOWN

    def test_keyword_extraction(self, comprehension_service):
        """Test that keywords are properly extracted in heuristic mode."""
        # Arrange
        input_data = ComprehensionInput(
            content="Implement a distributed caching system using Redis for session management",
            source_type=SourceType.USER,
        )

        # Act
        intent, _, _, _, _ = comprehension_service._heuristic_intent_parse(input_data)

        # Assert
        assert len(intent.keywords) > 0
        assert any("redis" in kw.lower() for kw in intent.keywords)


# =============================================================================
# CONTEXT MATCHING TESTS
# =============================================================================

class TestContextMatching:
    """Tests for memory context matching."""

    @pytest.mark.asyncio
    async def test_context_search_returns_empty_on_embedding_failure(self, comprehension_service, sample_user_input):
        """Test context search returns empty ContextMatch when embedding fails."""
        # Arrange
        intent = IntentAnalysis(
            action_type=ActionType.QUERY,
            summary="test",
            confidence=0.8,
            keywords=["test"],
        )

        with patch("app.services.comprehension_service.embedding_service") as mock_embed:
            mock_embed.generate_embedding = AsyncMock(return_value=[0.0] * 768)

            # Act
            context = await comprehension_service._search_memory_context(sample_user_input, intent)

            # Assert
            assert isinstance(context, ContextMatch)
            assert not context.has_relevant_context

    @pytest.mark.asyncio
    async def test_context_search_handles_exception(self, comprehension_service, sample_user_input):
        """Test context search handles exceptions gracefully."""
        # Arrange
        intent = IntentAnalysis(
            action_type=ActionType.QUERY,
            summary="test",
            confidence=0.8,
            keywords=["test"],
        )

        with patch("app.services.comprehension_service.embedding_service") as mock_embed:
            mock_embed.generate_embedding = AsyncMock(side_effect=Exception("Connection error"))

            # Act
            context = await comprehension_service._search_memory_context(sample_user_input, intent)

            # Assert
            assert isinstance(context, ContextMatch)
            assert not context.has_relevant_context
            assert context.total_matches == 0


# =============================================================================
# CAPABILITY MATCHING TESTS
# =============================================================================

class TestCapabilityMatching:
    """Tests for capability matching against agents."""

    @pytest.mark.asyncio
    async def test_capability_match_with_agents(self, comprehension_service, mock_agent_rows):
        """Test capability matching finds relevant agents."""
        # Arrange
        intent = IntentAnalysis(
            action_type=ActionType.QUERY,
            summary="Research quantum computing",
            confidence=0.85,
            keywords=["research", "analysis"],
        )
        suggested_caps = ["web_search", "analysis"]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_agent_rows)
        mock_acquire_ctx = AsyncMock()
        mock_acquire_ctx.__aenter__.return_value = mock_conn
        mock_acquire_ctx.__aexit__.return_value = None
        mock_pool = MagicMock()
        mock_pool.acquire.return_value = mock_acquire_ctx

        with patch("app.dependencies.get_db_pool", new_callable=AsyncMock, return_value=mock_pool):
            # Act
            result = await comprehension_service._match_capabilities(intent, suggested_caps)

            # Assert
            assert isinstance(result, CapabilityMatch)
            assert result.has_capable_agents
            assert len(result.matched_agent_ids) > 0

    @pytest.mark.asyncio
    async def test_capability_match_no_agents_available(self, comprehension_service):
        """Test capability matching when no agents are available."""
        # Arrange
        intent = IntentAnalysis(
            action_type=ActionType.QUERY,
            summary="test",
            confidence=0.8,
            keywords=["obscure_capability"],
        )

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_acquire_ctx = AsyncMock()
        mock_acquire_ctx.__aenter__.return_value = mock_conn
        mock_acquire_ctx.__aexit__.return_value = None
        mock_pool = MagicMock()
        mock_pool.acquire.return_value = mock_acquire_ctx

        with patch("app.dependencies.get_db_pool", new_callable=AsyncMock, return_value=mock_pool):
            # Act
            result = await comprehension_service._match_capabilities(intent, [])

            # Assert
            assert not result.has_capable_agents
            assert len(result.matched_agent_ids) == 0

    @pytest.mark.asyncio
    async def test_capability_match_handles_exception(self, comprehension_service):
        """Test capability matching handles DB exceptions gracefully."""
        # Arrange
        intent = IntentAnalysis(
            action_type=ActionType.QUERY,
            summary="test",
            confidence=0.8,
            keywords=["test"],
        )

        with patch("app.dependencies.get_db_pool", new_callable=AsyncMock, side_effect=Exception("DB error")):
            # Act
            result = await comprehension_service._match_capabilities(intent, [])

            # Assert
            assert isinstance(result, CapabilityMatch)
            assert not result.has_capable_agents


# =============================================================================
# COMPLEXITY ANALYSIS TESTS
# =============================================================================

class TestComplexityAnalysis:
    """Tests for complexity analysis."""

    @pytest.mark.asyncio
    async def test_simple_input_low_complexity(self, comprehension_service):
        """Test that simple input gets low complexity score."""
        # Arrange
        content = "What time is it?"

        # Act
        score = await comprehension_service.analyze_complexity(content)

        # Assert
        assert score.overall < 0.3
        assert score.reasoning_depth < 0.5

    @pytest.mark.asyncio
    async def test_complex_input_higher_complexity(self, comprehension_service):
        """Test that complex input gets higher complexity score."""
        # Arrange
        content = (
            "Analyze the system architecture then evaluate each microservice for performance. "
            "After that, compare the results with industry benchmarks and additionally "
            "create a comprehensive report explaining why certain services underperform. "
            "Furthermore, propose a step-by-step migration plan for the next quarter."
        )

        # Act
        score = await comprehension_service.analyze_complexity(content)

        # Assert
        assert score.overall > 0.3
        assert score.breadth > 0.2
        assert score.reasoning_depth > 0.2

    @pytest.mark.asyncio
    async def test_complexity_duration_estimate(self, comprehension_service):
        """Test that duration estimate is proportional to content length."""
        # Arrange
        short = "Hello"
        long = " ".join(["word"] * 100)

        # Act
        short_score = await comprehension_service.analyze_complexity(short)
        long_score = await comprehension_service.analyze_complexity(long)

        # Assert
        assert short_score.estimated_duration_seconds is not None
        assert long_score.estimated_duration_seconds is not None
        assert long_score.estimated_duration_seconds > short_score.estimated_duration_seconds


# =============================================================================
# HANDLING MODE TESTS
# =============================================================================

class TestHandlingMode:
    """Tests for handling mode determination."""

    def test_high_complexity_multi_agent(self, comprehension_service):
        """Test high complexity + breadth triggers multi-agent mode."""
        # Arrange
        intent = IntentAnalysis(
            action_type=ActionType.ANALYSIS,
            summary="Complex multi-domain analysis",
            confidence=0.8,
        )
        complexity = ComplexityScore(
            overall=0.9,
            reasoning_depth=0.8,
            breadth=0.8,
            novelty=0.5,
        )
        capabilities = CapabilityMatch(
            has_capable_agents=True,
            matched_agent_ids=["a", "b", "c"],
        )

        # Act
        mode = comprehension_service._determine_handling_mode(intent, complexity, capabilities)

        # Assert
        assert mode == HandlingMode.MULTI_AGENT

    def test_simple_query_single_agent(self, comprehension_service):
        """Test simple query uses single agent."""
        # Arrange
        intent = IntentAnalysis(
            action_type=ActionType.QUERY,
            summary="Simple question",
            confidence=0.9,
        )
        complexity = ComplexityScore(
            overall=0.3,
            reasoning_depth=0.2,
            breadth=0.2,
            novelty=0.1,
        )
        capabilities = CapabilityMatch(
            has_capable_agents=True,
            matched_agent_ids=["a"],
        )

        # Act
        mode = comprehension_service._determine_handling_mode(intent, complexity, capabilities)

        # Assert
        assert mode == HandlingMode.SINGLE_AGENT

    def test_no_capable_agents_human_required(self, comprehension_service):
        """Test that no capable agents triggers human_required."""
        # Arrange
        intent = IntentAnalysis(
            action_type=ActionType.COMMAND,
            summary="Do something exotic",
            confidence=0.5,
        )
        complexity = ComplexityScore(
            overall=0.5,
            reasoning_depth=0.5,
            breadth=0.5,
            novelty=0.5,
        )
        capabilities = CapabilityMatch(has_capable_agents=False)

        # Act
        mode = comprehension_service._determine_handling_mode(intent, complexity, capabilities)

        # Assert
        assert mode == HandlingMode.HUMAN_REQUIRED

    def test_conversation_always_single_agent(self, comprehension_service):
        """Test conversation type defaults to single agent even without capabilities."""
        # Arrange
        intent = IntentAnalysis(
            action_type=ActionType.CONVERSATION,
            summary="Just chatting",
            confidence=0.9,
        )
        complexity = ComplexityScore(
            overall=0.1,
            reasoning_depth=0.1,
            breadth=0.1,
            novelty=0.1,
        )
        capabilities = CapabilityMatch(has_capable_agents=False)

        # Act
        mode = comprehension_service._determine_handling_mode(intent, complexity, capabilities)

        # Assert
        assert mode == HandlingMode.SINGLE_AGENT

    def test_trivial_query_no_agent(self, comprehension_service):
        """Test trivial high-confidence query returns NO_AGENT."""
        # Arrange
        intent = IntentAnalysis(
            action_type=ActionType.QUERY,
            summary="Simple factoid",
            confidence=0.95,
        )
        complexity = ComplexityScore(
            overall=0.1,
            reasoning_depth=0.1,
            breadth=0.1,
            novelty=0.1,
        )
        capabilities = CapabilityMatch(has_capable_agents=True, matched_agent_ids=["a"])

        # Act
        mode = comprehension_service._determine_handling_mode(intent, complexity, capabilities)

        # Assert
        assert mode == HandlingMode.NO_AGENT


# =============================================================================
# PRIORITY ADJUSTMENT TESTS
# =============================================================================

class TestPriorityAdjustment:
    """Tests for priority adjustment logic."""

    def test_critical_urgency_boosts_priority(self, comprehension_service):
        """Test critical urgency increases priority."""
        # Arrange
        intent = IntentAnalysis(
            action_type=ActionType.COMMAND,
            summary="Critical",
            confidence=0.8,
            urgency=UrgencyLevel.CRITICAL,
        )
        context = ContextMatch()

        # Act
        adjusted = comprehension_service._adjust_priority(5, intent, context)

        # Assert
        assert adjusted == 8  # 5 + 3 for critical

    def test_low_urgency_lowers_priority(self, comprehension_service):
        """Test low urgency decreases priority."""
        # Arrange
        intent = IntentAnalysis(
            action_type=ActionType.QUERY,
            summary="No rush",
            confidence=0.8,
            urgency=UrgencyLevel.LOW,
        )
        context = ContextMatch()

        # Act
        adjusted = comprehension_service._adjust_priority(5, intent, context)

        # Assert
        assert adjusted == 4  # 5 - 1 for low

    def test_strong_context_lowers_priority(self, comprehension_service):
        """Test strong context match lowers priority slightly."""
        # Arrange
        intent = IntentAnalysis(
            action_type=ActionType.QUERY,
            summary="test",
            confidence=0.8,
            urgency=UrgencyLevel.MEDIUM,
        )
        context = ContextMatch(
            has_relevant_context=True,
            best_match_score=0.9,
        )

        # Act
        adjusted = comprehension_service._adjust_priority(5, intent, context)

        # Assert
        assert adjusted == 4  # 5 - 1 for strong context

    def test_priority_clamped_to_valid_range(self, comprehension_service):
        """Test priority stays within 1-10 range."""
        # Arrange
        intent_critical = IntentAnalysis(
            action_type=ActionType.COMMAND,
            summary="test",
            confidence=0.8,
            urgency=UrgencyLevel.CRITICAL,
        )
        intent_none = IntentAnalysis(
            action_type=ActionType.QUERY,
            summary="test",
            confidence=0.8,
            urgency=UrgencyLevel.NONE,
        )
        context_strong = ContextMatch(has_relevant_context=True, best_match_score=0.9)

        # Act
        high = comprehension_service._adjust_priority(10, intent_critical, ContextMatch())
        low = comprehension_service._adjust_priority(1, intent_none, context_strong)

        # Assert
        assert high == 10  # Clamped at 10
        assert low == 1    # Clamped at 1 (1 - 2 - 1 = -2, clamped to 1)


# =============================================================================
# CONFIDENCE SCORING TESTS
# =============================================================================

class TestConfidenceScoring:
    """Tests for overall confidence scoring."""

    def test_high_confidence_all_signals(self, comprehension_service):
        """Test high confidence when all signals are strong."""
        # Arrange
        intent = IntentAnalysis(
            action_type=ActionType.QUERY,
            summary="test",
            confidence=0.95,
        )
        context = ContextMatch(has_relevant_context=True, best_match_score=0.9)
        capabilities = CapabilityMatch(has_capable_agents=True, best_agent_score=0.8)

        # Act
        score = comprehension_service._score_confidence(intent, context, capabilities)

        # Assert
        assert score > 0.7

    def test_low_confidence_weak_signals(self, comprehension_service):
        """Test low confidence when signals are weak."""
        # Arrange
        intent = IntentAnalysis(
            action_type=ActionType.UNKNOWN,
            summary="unclear",
            confidence=0.2,
        )
        context = ContextMatch()
        capabilities = CapabilityMatch()

        # Act
        score = comprehension_service._score_confidence(intent, context, capabilities)

        # Assert
        assert score < 0.5

    def test_confidence_bounded_zero_to_one(self, comprehension_service):
        """Test confidence is always between 0 and 1."""
        # Arrange — extreme inputs
        intent = IntentAnalysis(
            action_type=ActionType.QUERY,
            summary="test",
            confidence=1.0,
        )
        context = ContextMatch(has_relevant_context=True, best_match_score=1.0)
        capabilities = CapabilityMatch(has_capable_agents=True, best_agent_score=1.0)

        # Act
        score = comprehension_service._score_confidence(intent, context, capabilities)

        # Assert
        assert 0.0 <= score <= 1.0


# =============================================================================
# END-TO-END COMPREHENSION FLOW TESTS
# =============================================================================

class TestEndToEndComprehension:
    """End-to-end tests for the full comprehension pipeline."""

    @pytest.mark.asyncio
    async def test_full_comprehension_with_heuristic_fallback(self, comprehension_service, sample_user_input):
        """Test full comprehension pipeline with heuristic fallback when LLM fails."""
        # Arrange — patch LLM to fail, memory to skip, DB to skip
        with patch.object(comprehension_service, "_call_llm_for_intent", new_callable=AsyncMock, side_effect=Exception("LLM unavailable")):
            with patch.object(comprehension_service, "_search_memory_context", new_callable=AsyncMock, return_value=ContextMatch()):
                with patch.object(comprehension_service, "_match_capabilities", new_callable=AsyncMock, return_value=CapabilityMatch(has_capable_agents=True, matched_agent_ids=["a"])):
                    with patch("app.services.comprehension_service.comprehension_repository") as mock_repo:
                        mock_repo.store_comprehension_result = AsyncMock(return_value=uuid4())

                        # Act
                        result = await comprehension_service.comprehend(sample_user_input)

                        # Assert
                        assert isinstance(result, ComprehensionResult)
                        assert result.status == ComprehensionStatus.COMPLETED
                        assert result.intent.confidence == 0.4  # Heuristic
                        assert result.processing_time_ms is not None

    @pytest.mark.asyncio
    async def test_full_comprehension_with_llm(self, comprehension_service, sample_user_input, sample_llm_response):
        """Test full comprehension pipeline with LLM response."""
        # Arrange
        with patch.object(comprehension_service, "_call_llm_for_intent", new_callable=AsyncMock, return_value=sample_llm_response):
            with patch.object(comprehension_service, "_search_memory_context", new_callable=AsyncMock, return_value=ContextMatch(has_relevant_context=True, best_match_score=0.85)):
                with patch.object(comprehension_service, "_match_capabilities", new_callable=AsyncMock, return_value=CapabilityMatch(has_capable_agents=True, matched_agent_ids=["researcher-001"], best_agent_score=0.8)):
                    with patch("app.services.comprehension_service.comprehension_repository") as mock_repo:
                        mock_repo.store_comprehension_result = AsyncMock(return_value=uuid4())

                        # Act
                        result = await comprehension_service.comprehend(sample_user_input)

                        # Assert
                        assert result.status == ComprehensionStatus.COMPLETED
                        assert result.intent.action_type == ActionType.QUERY
                        assert result.intent.confidence == 0.85
                        assert result.suggested_task_type == "research"
                        assert result.confidence > 0.5

    @pytest.mark.asyncio
    async def test_comprehension_failure_returns_failed_result(self, comprehension_service, sample_user_input):
        """Test that catastrophic failure returns a FAILED result instead of raising."""
        # Arrange — everything fails
        with patch.object(comprehension_service, "_parse_intent", new_callable=AsyncMock, side_effect=Exception("Total failure")):
            # Act
            result = await comprehension_service.comprehend(sample_user_input)

            # Assert
            assert result.status == ComprehensionStatus.FAILED
            assert result.confidence == 0.0
            assert result.intent.action_type == ActionType.UNKNOWN


# =============================================================================
# FEEDBACK TESTS
# =============================================================================

class TestFeedback:
    """Tests for the feedback and learning loop."""

    @pytest.mark.asyncio
    async def test_submit_valid_feedback(self, comprehension_service):
        """Test submitting valid feedback."""
        # Arrange
        feedback = ComprehensionFeedback(
            comprehension_id=uuid4(),
            score=0.9,
            correct_action_type="query",
            notes="Spot on!",
            submitted_by="user-1",
        )

        with patch("app.services.comprehension_service.comprehension_repository") as mock_repo:
            mock_repo.store_feedback = AsyncMock(return_value=True)

            # Act
            result = await comprehension_service.submit_feedback(feedback)

            # Assert
            assert result is True
            mock_repo.store_feedback.assert_awaited_once_with(feedback)

    def test_feedback_model_validation(self):
        """Test feedback model validates score bounds."""
        # Arrange & Act
        feedback = ComprehensionFeedback(
            comprehension_id=uuid4(),
            score=0.5,
        )

        # Assert
        assert feedback.score == 0.5

    def test_feedback_rejects_invalid_score(self):
        """Test feedback rejects out-of-range scores."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError):
            ComprehensionFeedback(
                comprehension_id=uuid4(),
                score=1.5,
            )


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_very_long_input(self, comprehension_service):
        """Test handling of very long input text."""
        # Arrange
        long_content = " ".join(["word"] * 10000)
        input_data = ComprehensionInput(
            content=long_content,
            source_type=SourceType.USER,
        )

        # Act
        intent, complexity, _, _, _ = comprehension_service._heuristic_intent_parse(input_data)

        # Assert — should not crash
        assert intent.action_type is not None
        assert complexity.overall <= 1.0

    def test_special_characters_input(self, comprehension_service):
        """Test handling of special characters in input."""
        # Arrange
        input_data = ComprehensionInput(
            content="¿Cómo está? 🤖 <script>alert('xss')</script> SELECT * FROM users; --",
            source_type=SourceType.USER,
        )

        # Act
        intent, _, _, _, _ = comprehension_service._heuristic_intent_parse(input_data)

        # Assert
        assert intent.action_type is not None

    def test_single_word_input(self, comprehension_service):
        """Test handling of single word input."""
        # Arrange
        input_data = ComprehensionInput(
            content="help",
            source_type=SourceType.USER,
        )

        # Act
        intent, complexity, _, _, _ = comprehension_service._heuristic_intent_parse(input_data)

        # Assert
        assert intent is not None
        assert complexity.overall < 0.2

    @pytest.mark.asyncio
    async def test_capability_registry_returns_system_capabilities(self, comprehension_service):
        """Test capability registry always includes system capabilities."""
        # Arrange — DB fails but system caps should still return
        with patch("app.dependencies.get_db_pool", new_callable=AsyncMock, side_effect=Exception("DB error")):
            # Act
            caps = await comprehension_service.get_capability_registry()

            # Assert
            assert len(caps) >= 3  # memory_search, embedding_generation, task_routing
            cap_names = [c.name for c in caps]
            assert "memory_search" in cap_names
            assert "embedding_generation" in cap_names
            assert "task_routing" in cap_names

    @pytest.mark.asyncio
    async def test_health_check(self, comprehension_service):
        """Test health check returns proper structure."""
        # Arrange
        with patch("app.services.comprehension_service.embedding_service") as mock_embed:
            mock_embed.health_check = AsyncMock(return_value=True)

            # Act
            health = await comprehension_service.health_check()

            # Assert
            assert "initialized" in health
            assert "embedding_service" in health
            assert health["initialized"] is True

    def test_all_action_types_in_enum(self):
        """Test that all expected action types exist."""
        # Arrange & Act
        action_types = [at.value for at in ActionType]

        # Assert
        assert "query" in action_types
        assert "command" in action_types
        assert "analysis" in action_types
        assert "creation" in action_types
        assert "monitoring" in action_types
        assert "conversation" in action_types
        assert "planning" in action_types
        assert "unknown" in action_types

    def test_all_handling_modes_in_enum(self):
        """Test that all expected handling modes exist."""
        # Arrange & Act
        modes = [m.value for m in HandlingMode]

        # Assert
        assert "single_agent" in modes
        assert "multi_agent" in modes
        assert "no_agent" in modes
        assert "human_required" in modes
