"""
Tests for OpenTelemetry tracing of the CORE LangGraph pipeline.

Sentinel-value contract: every span test would FAIL if you removed
the get_tracer() / start_as_current_span calls from the node methods.

Coverage:
- setup_telemetry() configures a provider with the injected exporter
- Each pipeline node (comprehension/orchestration/reasoning/evaluation/conversation)
  produces exactly one span with the correct name
- Span attributes are populated (node.name, pipeline.stage)
- Parent-child relationships: node spans are children of an outer root span
- No-op path: when OTEL_EXPORTER_OTLP_ENDPOINT is absent, setup still works
- Error path: exception in a node records an error attribute on the span
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry import trace as otel_trace

from app.core.telemetry import setup_telemetry, get_tracer, reset_telemetry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_otel():
    """Reset the module-level OTel state before and after each test."""
    reset_telemetry()
    yield
    reset_telemetry()


@pytest.fixture
def memory_exporter() -> InMemorySpanExporter:
    """Return a fresh InMemorySpanExporter wired into the global provider."""
    exporter = InMemorySpanExporter()
    setup_telemetry(exporter=exporter)
    return exporter


# ---------------------------------------------------------------------------
# setup_telemetry / get_tracer basics
# ---------------------------------------------------------------------------


class TestSetupTelemetry:
    def test_returns_tracer_provider(self, memory_exporter):
        provider = setup_telemetry(exporter=InMemorySpanExporter())
        assert isinstance(provider, TracerProvider)

    def test_get_tracer_returns_tracer_after_setup(self, memory_exporter):
        tracer = get_tracer()
        assert tracer is not None

    def test_get_tracer_lazy_init_without_setup(self):
        """get_tracer() must initialise lazily even if setup_telemetry was never called."""
        tracer = get_tracer()
        assert tracer is not None

    def test_no_otlp_endpoint_does_not_raise(self):
        """Without OTEL_EXPORTER_OTLP_ENDPOINT, setup_telemetry should not raise."""
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
            provider = setup_telemetry()  # No exporter injected, no endpoint set
        assert isinstance(provider, TracerProvider)

    def test_span_recorded_via_memory_exporter(self, memory_exporter):
        """
        SENTINEL TEST — proves memory exporter is wired correctly.
        A span created by get_tracer() must appear in the exporter.
        Remove the SimpleSpanProcessor → fails.
        """
        tracer = get_tracer()
        with tracer.start_as_current_span("test.sentinel"):
            pass
        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "test.sentinel"


# ---------------------------------------------------------------------------
# Node span names — sentinel tests
# ---------------------------------------------------------------------------


def _make_core_state():
    """Create a minimal COREState for node method calls."""
    from app.models.core_state import COREState

    return COREState(user_input="test input", session_id="test-session")


class TestNodeSpanNames:
    """
    SENTINEL TESTS — each test fails if the corresponding node's
    start_as_current_span call is removed or the span name is changed.
    """

    def test_comprehension_node_creates_span(self, memory_exporter):
        from app.core.langgraph.core_graph_v2 import COREGraph

        graph = COREGraph.__new__(COREGraph)
        mock_agent = MagicMock()
        mock_agent.analyze_intent.return_value = MagicMock(
            type="conversation", confidence=0.9
        )
        graph.comprehension_agent = mock_agent

        state = _make_core_state()
        graph.comprehension_node(state)

        spans = memory_exporter.get_finished_spans()
        span_names = [s.name for s in spans]
        assert (
            "core.comprehension" in span_names
        ), f"Expected 'core.comprehension' span, got: {span_names}"

    def test_orchestration_node_creates_span(self, memory_exporter):
        from app.core.langgraph.core_graph_v2 import COREGraph
        from app.models.core_state import UserIntent

        graph = COREGraph.__new__(COREGraph)
        mock_agent = MagicMock()
        mock_agent.create_plan.return_value = MagicMock(steps=[], revision=1, goal="g")
        graph.orchestration_agent = mock_agent

        state = _make_core_state()
        state.intent = UserIntent(
            type="task", description="do something", confidence=0.9
        )
        graph.orchestration_node(state)

        span_names = [s.name for s in memory_exporter.get_finished_spans()]
        assert (
            "core.orchestration" in span_names
        ), f"Expected 'core.orchestration' span, got: {span_names}"

    def test_reasoning_node_creates_span(self, memory_exporter):
        from app.core.langgraph.core_graph_v2 import COREGraph
        from app.models.core_state import UserIntent, ExecutionPlan

        graph = COREGraph.__new__(COREGraph)
        mock_agent = MagicMock()
        mock_agent.execute_plan.return_value = []
        graph.reasoning_agent = mock_agent

        state = _make_core_state()
        state.intent = UserIntent(type="task", description="t", confidence=0.9)
        state.plan = ExecutionPlan(goal="test goal", steps=[])
        state.config = {"enable_tools": False}
        graph.reasoning_node(state)

        span_names = [s.name for s in memory_exporter.get_finished_spans()]
        assert (
            "core.reasoning" in span_names
        ), f"Expected 'core.reasoning' span, got: {span_names}"

    def test_evaluation_node_creates_span(self, memory_exporter):
        from app.core.langgraph.core_graph_v2 import COREGraph
        from app.models.core_state import UserIntent, ExecutionPlan, EvaluationResult

        graph = COREGraph.__new__(COREGraph)
        mock_agent = MagicMock()
        mock_agent.evaluate_execution.return_value = EvaluationResult(
            overall_status="success",
            confidence=0.9,
            meets_requirements=True,
            quality_score=0.9,
            feedback="good",
            next_action="finalize",
        )
        graph.evaluation_agent = mock_agent

        state = _make_core_state()
        state.intent = UserIntent(type="task", description="t", confidence=0.9)
        state.plan = ExecutionPlan(goal="test goal", steps=[])
        graph.evaluation_node(state)

        span_names = [s.name for s in memory_exporter.get_finished_spans()]
        assert (
            "core.evaluation" in span_names
        ), f"Expected 'core.evaluation' span, got: {span_names}"

    def test_conversation_node_creates_span(self, memory_exporter):
        from app.core.langgraph.core_graph_v2 import COREGraph
        from app.models.core_state import UserIntent

        graph = COREGraph.__new__(COREGraph)

        state = _make_core_state()
        state.intent = UserIntent(
            type="conversation", description="chat", confidence=0.9
        )
        graph.conversation_node(state)

        span_names = [s.name for s in memory_exporter.get_finished_spans()]
        assert (
            "core.conversation" in span_names
        ), f"Expected 'core.conversation' span, got: {span_names}"

    def test_all_five_pipeline_stages_produce_spans(self, memory_exporter):
        """
        SENTINEL TEST — run all five nodes and verify each produces exactly its span.
        Removing any node's tracer call drops it from this list and fails the test.
        """
        from app.core.langgraph.core_graph_v2 import COREGraph
        from app.models.core_state import UserIntent, ExecutionPlan, EvaluationResult

        graph = COREGraph.__new__(COREGraph)
        graph.comprehension_agent = MagicMock()
        graph.comprehension_agent.analyze_intent.return_value = MagicMock(
            type="task", confidence=0.9
        )
        graph.orchestration_agent = MagicMock()
        graph.orchestration_agent.create_plan.return_value = MagicMock(
            steps=[], revision=1, goal="g"
        )
        graph.reasoning_agent = MagicMock()
        graph.reasoning_agent.execute_plan.return_value = []
        graph.evaluation_agent = MagicMock()
        graph.evaluation_agent.evaluate_execution.return_value = EvaluationResult(
            overall_status="success",
            confidence=0.9,
            meets_requirements=True,
            quality_score=0.9,
            feedback="ok",
            next_action="finalize",
        )

        state = _make_core_state()
        state.config = {"enable_tools": False}
        state.plan = ExecutionPlan(goal="test goal", steps=[])
        state.intent = UserIntent(type="task", description="t", confidence=0.9)

        graph.comprehension_node(state)
        graph.orchestration_node(state)
        graph.reasoning_node(state)
        graph.evaluation_node(state)
        graph.conversation_node(state)

        span_names = {s.name for s in memory_exporter.get_finished_spans()}
        expected = {
            "core.comprehension",
            "core.orchestration",
            "core.reasoning",
            "core.evaluation",
            "core.conversation",
        }
        missing = expected - span_names
        assert not missing, f"Missing spans: {missing} — got: {span_names}"


# ---------------------------------------------------------------------------
# Span attributes
# ---------------------------------------------------------------------------


class TestSpanAttributes:
    def test_comprehension_span_has_node_name_attribute(self, memory_exporter):
        from app.core.langgraph.core_graph_v2 import COREGraph

        graph = COREGraph.__new__(COREGraph)
        graph.comprehension_agent = MagicMock()
        graph.comprehension_agent.analyze_intent.return_value = MagicMock(
            type="conversation", confidence=0.8
        )

        graph.comprehension_node(_make_core_state())

        span = next(
            s
            for s in memory_exporter.get_finished_spans()
            if s.name == "core.comprehension"
        )
        assert span.attributes.get("node.name") == "comprehension"

    def test_orchestration_span_has_pipeline_stage_attribute(self, memory_exporter):
        from app.core.langgraph.core_graph_v2 import COREGraph
        from app.models.core_state import UserIntent

        graph = COREGraph.__new__(COREGraph)
        graph.orchestration_agent = MagicMock()
        graph.orchestration_agent.create_plan.return_value = MagicMock(
            steps=[], revision=1, goal="g"
        )

        state = _make_core_state()
        state.intent = UserIntent(type="task", description="t", confidence=0.9)
        graph.orchestration_node(state)

        span = next(
            s
            for s in memory_exporter.get_finished_spans()
            if s.name == "core.orchestration"
        )
        assert span.attributes.get("pipeline.stage") == "2_orchestration"

    def test_reasoning_span_records_step_count(self, memory_exporter):
        from app.core.langgraph.core_graph_v2 import COREGraph
        from app.models.core_state import UserIntent, ExecutionPlan, PlanStep

        graph = COREGraph.__new__(COREGraph)
        graph.reasoning_agent = MagicMock()
        graph.reasoning_agent.execute_plan.return_value = []

        state = _make_core_state()
        state.config = {"enable_tools": False}
        step = PlanStep(name="Step A", description="do A")
        state.plan = ExecutionPlan(goal="g", steps=[step])

        graph.reasoning_node(state)

        span = next(
            s
            for s in memory_exporter.get_finished_spans()
            if s.name == "core.reasoning"
        )
        assert span.attributes.get("plan.step_count") == 1

    def test_evaluation_span_records_next_action(self, memory_exporter):
        """
        SENTINEL TEST — eval.next_action attribute on the span must match
        the evaluation result. Remove the set_attribute call → attribute absent.
        """
        from app.core.langgraph.core_graph_v2 import COREGraph
        from app.models.core_state import UserIntent, ExecutionPlan, EvaluationResult

        graph = COREGraph.__new__(COREGraph)
        graph.evaluation_agent = MagicMock()
        graph.evaluation_agent.evaluate_execution.return_value = EvaluationResult(
            overall_status="success",
            confidence=0.9,
            meets_requirements=True,
            quality_score=0.9,
            feedback="ok",
            next_action="revise_plan",  # valid Literal value used as sentinel
        )

        state = _make_core_state()
        state.plan = ExecutionPlan(goal="g", steps=[])
        graph.evaluation_node(state)

        span = next(
            s
            for s in memory_exporter.get_finished_spans()
            if s.name == "core.evaluation"
        )
        assert span.attributes.get("eval.next_action") == "revise_plan"


# ---------------------------------------------------------------------------
# Parent-child relationships
# ---------------------------------------------------------------------------


class TestSpanParentChild:
    """
    SENTINEL TESTS — verify that node spans become children of an outer span
    when the caller creates a root span first.
    Remove start_as_current_span from a node → its parent_id becomes None.
    """

    def test_comprehension_span_is_child_of_root_span(self, memory_exporter):
        from app.core.langgraph.core_graph_v2 import COREGraph

        graph = COREGraph.__new__(COREGraph)
        graph.comprehension_agent = MagicMock()
        graph.comprehension_agent.analyze_intent.return_value = MagicMock(
            type="conversation", confidence=0.9
        )

        tracer = get_tracer()
        with tracer.start_as_current_span("core.pipeline.root") as root_span:
            graph.comprehension_node(_make_core_state())
            root_span_ctx = root_span.get_span_context()

        finished = memory_exporter.get_finished_spans()
        root = next((s for s in finished if s.name == "core.pipeline.root"), None)
        child = next((s for s in finished if s.name == "core.comprehension"), None)

        assert root is not None, "Root span not found"
        assert child is not None, "Comprehension span not found"
        assert child.parent is not None, "Comprehension span has no parent"
        assert (
            child.parent.span_id == root.context.span_id
        ), "Comprehension span parent does not match root span"

    def test_orchestration_span_is_child_of_root_span(self, memory_exporter):
        from app.core.langgraph.core_graph_v2 import COREGraph
        from app.models.core_state import UserIntent

        graph = COREGraph.__new__(COREGraph)
        graph.orchestration_agent = MagicMock()
        graph.orchestration_agent.create_plan.return_value = MagicMock(
            steps=[], revision=1, goal="g"
        )

        state = _make_core_state()
        state.intent = UserIntent(type="task", description="t", confidence=0.9)

        tracer = get_tracer()
        with tracer.start_as_current_span("core.pipeline.root"):
            graph.orchestration_node(state)

        finished = memory_exporter.get_finished_spans()
        root = next((s for s in finished if s.name == "core.pipeline.root"), None)
        child = next((s for s in finished if s.name == "core.orchestration"), None)

        assert child is not None
        assert child.parent is not None
        assert child.parent.span_id == root.context.span_id

    def test_independent_node_calls_have_no_parent(self, memory_exporter):
        """When called without an outer span, node spans have no parent."""
        from app.core.langgraph.core_graph_v2 import COREGraph

        graph = COREGraph.__new__(COREGraph)
        graph.comprehension_agent = MagicMock()
        graph.comprehension_agent.analyze_intent.return_value = MagicMock(
            type="conversation", confidence=0.9
        )

        graph.comprehension_node(_make_core_state())

        finished = memory_exporter.get_finished_spans()
        child = next((s for s in finished if s.name == "core.comprehension"), None)
        assert child is not None
        # No outer context → no parent
        assert child.parent is None


# ---------------------------------------------------------------------------
# Error recording
# ---------------------------------------------------------------------------


class TestSpanErrorRecording:
    def test_comprehension_exception_sets_error_attribute(self, memory_exporter):
        """
        SENTINEL TEST — when the comprehension agent raises, the span must
        record the exception. Remove span.record_exception → error attr absent.
        """
        from app.core.langgraph.core_graph_v2 import COREGraph

        graph = COREGraph.__new__(COREGraph)
        graph.comprehension_agent = MagicMock()
        graph.comprehension_agent.analyze_intent.side_effect = RuntimeError(
            "SENTINEL_ERROR"
        )

        graph.comprehension_node(_make_core_state())

        spans = memory_exporter.get_finished_spans()
        comp_span = next((s for s in spans if s.name == "core.comprehension"), None)
        assert comp_span is not None
        assert comp_span.attributes.get("error") is True

    def test_reasoning_no_plan_sets_error_attribute(self, memory_exporter):
        """Reasoning node with no plan sets error=True on the span."""
        from app.core.langgraph.core_graph_v2 import COREGraph

        graph = COREGraph.__new__(COREGraph)
        graph.reasoning_agent = MagicMock()

        state = _make_core_state()
        state.config = {"enable_tools": False}
        # No plan set → should trigger error path
        graph.reasoning_node(state)

        spans = memory_exporter.get_finished_spans()
        reason_span = next((s for s in spans if s.name == "core.reasoning"), None)
        assert reason_span is not None
        assert reason_span.attributes.get("error") is True
