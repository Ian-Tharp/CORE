"""
CORE Cognitive Graph v2 - Production Implementation

Implements the full CORE pipeline with proper state management and conditional routing:
START → Comprehension → Orchestration → Reasoning → Evaluation → Conversation → END
"""

import os
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.types import Send

# Maximum number of plan revisions before forcing finalization.
# Prevents infinite revise_plan → orchestration → reasoning → evaluation loops.
_MAX_PLAN_REVISIONS = int(os.getenv("CORE_MAX_PLAN_REVISIONS", "3"))

from app.models.core_state import (
    COREState,
    UserIntent,
    ExecutionPlan,
    PlanStep,
    EvaluationResult,
)
from app.core.agents.comprehension_agent import ComprehensionAgent
from app.core.agents.orchestration_agent import OrchestrationAgent
from app.core.agents.reasoning_agent import ReasoningAgent
from app.core.agents.evaluation_agent import EvaluationAgent
from app.core.telemetry import get_tracer
from app.core.intelligence_log import record_pipeline_run


class COREGraph:
    """
    CORE Cognitive Graph - The main orchestration engine.

    This graph implements the neutral cognitive kernel that processes user input through:
    - Comprehension: Understand intent and retrieve context
    - Orchestration: Plan the execution strategy
    - Reasoning: Execute the plan step-by-step
    - Evaluation: Assess quality and determine next action
    - Conversation: Formulate response to user
    """

    def __init__(self):
        self.graph = None
        self.compiled_graph = None

        # Initialize agents with configurable model
        # Priority: CORE_DEFAULT_MODEL env var > centralized config > hardcoded fallback
        from app.config.models import get_default_model

        ollama_model = os.getenv(
            "CORE_DEFAULT_MODEL", get_default_model("comprehension")
        )

        self.comprehension_agent = ComprehensionAgent(model=ollama_model)
        self.orchestration_agent = OrchestrationAgent(model=ollama_model)
        self.reasoning_agent = ReasoningAgent(model=ollama_model)
        self.evaluation_agent = EvaluationAgent(model=ollama_model)

        # Capability Registry — the system's self-model. Built FROM the live
        # dispatcher so what comprehension reasons over can't drift from what runs.
        from app.core.tools.dispatcher import ToolDispatcher
        from app.core.registry.builtin import build_registry_from_dispatcher

        _workspace = os.getenv("CORE_WORKSPACE_DIR", os.getcwd())
        self.registry = build_registry_from_dispatcher(
            ToolDispatcher(workspace_dir=_workspace)
        )

    def initialize_graph(self) -> None:
        """Build and compile the CORE graph with conditional routing."""
        # Create graph with COREState as the state schema
        self.graph = StateGraph(COREState)

        # Add nodes
        self.graph.add_node("comprehension", self.comprehension_node)
        self.graph.add_node("orchestration", self.orchestration_node)
        self.graph.add_node("reasoning", self.reasoning_node)
        self.graph.add_node("evaluation", self.evaluation_node)
        self.graph.add_node("conversation", self.conversation_node)

        # Define conditional routing logic
        self.graph.set_entry_point("comprehension")

        # Comprehension routes to either Orchestration (task) or Conversation (chat)
        self.graph.add_conditional_edges(
            "comprehension",
            self.route_from_comprehension,
            {
                "orchestration": "orchestration",
                "conversation": "conversation",
            },
        )

        # Orchestration always goes to Reasoning
        self.graph.add_edge("orchestration", "reasoning")

        # Reasoning always goes to Evaluation
        self.graph.add_edge("reasoning", "evaluation")

        # Evaluation routes based on outcome
        self.graph.add_conditional_edges(
            "evaluation",
            self.route_from_evaluation,
            {
                "conversation": "conversation",
                "orchestration": "orchestration",  # Revise plan
                "reasoning": "reasoning",  # Retry step
            },
        )

        # Conversation always ends
        self.graph.add_edge("conversation", END)

        # Compile the graph
        self.compiled_graph = self.graph.compile()

    def get_graph(self):
        """Get the compiled graph for execution."""
        if not self.compiled_graph:
            self.initialize_graph()
        return self.compiled_graph

    # ======================
    # Node Implementations
    # ======================

    def comprehension_node(self, state: COREState) -> COREState:
        """
        Comprehension Node: Understand user intent and retrieve relevant context.

        Determines:
        - Is this a task that needs doing, or just a conversation?
        - What tools/capabilities are needed?
        - Is there ambiguity that needs clarification?
        """
        tracer = get_tracer()
        with tracer.start_as_current_span("core.comprehension") as span:
            span.set_attribute("node.name", "comprehension")
            span.set_attribute("pipeline.stage", "1_comprehension")

            state.add_execution_node("comprehension")

            try:
                # Run comprehension agent
                intent = self.comprehension_agent.analyze_intent(state.user_input)
                state.intent = intent

                span.set_attribute("intent.type", intent.type if intent else "unknown")
                span.set_attribute(
                    "intent.confidence", float(intent.confidence) if intent else 0.0
                )

                # Merge deterministic ambiguity detection into the intent so the
                # gate sees both the model's and the pattern-based ambiguities.
                try:
                    for amb in self.comprehension_agent.detect_ambiguities(
                        state.user_input
                    ):
                        if amb not in intent.ambiguities:
                            intent.ambiguities.append(amb)
                except Exception as amb_exc:  # never block on ambiguity detection
                    import logging as _logging

                    _logging.getLogger(__name__).debug(
                        "ambiguity detection failed (non-critical): %s", amb_exc
                    )

                # Front verifier: tri-state gate over the capability registry.
                # proceed → run the loop; clarify/refuse → answer directly (no spend).
                from app.core.agents.comprehension_gate import decide_gate

                decision = decide_gate(intent, self.registry)
                state.gate_outcome = decision.outcome
                state.gate_message = decision.message or None
                span.set_attribute("gate.outcome", decision.outcome)

            except Exception as e:
                span.record_exception(e)
                span.set_attribute("error", True)
                state.add_error(f"Comprehension failed: {str(e)}")
                # Fallback to conversation mode on error
                state.intent = UserIntent(
                    type="conversation",
                    description="Error in comprehension, falling back to chat",
                    confidence=0.5,
                )

        return state

    def orchestration_node(self, state: COREState) -> COREState:
        """
        Orchestration Node: Create an execution plan with steps, dependencies, and retry policies.

        Produces:
        - List of steps to accomplish the goal
        - Tool selections and parameters
        - Dependency graph between steps
        - Retry policies and HITL checkpoints
        """
        tracer = get_tracer()
        with tracer.start_as_current_span("core.orchestration") as span:
            span.set_attribute("node.name", "orchestration")
            span.set_attribute("pipeline.stage", "2_orchestration")

            state.add_execution_node("orchestration")

            try:
                # If this is a plan revision, increment revision counter
                if state.plan and state.eval_result:
                    state.plan_revisions += 1
                    revision_num = state.plan.revision + 1
                    feedback = state.eval_result.feedback
                else:
                    revision_num = 1
                    feedback = None

                span.set_attribute("plan.revision", revision_num)
                span.set_attribute("plan.revision_count", state.plan_revisions)

                # Run orchestration agent to create/revise plan
                plan = self.orchestration_agent.create_plan(
                    user_input=state.user_input,
                    intent=state.intent,
                    previous_plan=state.plan,
                    evaluation_feedback=feedback,
                    revision=revision_num,
                )

                # --- Phase 2: additive registry grounding ---------------------
                # Ground the draft plan to the Capability Registry: drop invented
                # tools, registry-unknown actions, and schema-invalid params before
                # the plan reaches Reasoning. Report-only for orphaned dependents.
                if plan is not None:
                    from app.core.registry.grounding import ground_plan

                    roots = {e.id.split(".", 1)[0] for e in self.registry.all()}
                    grounded_plan, report = ground_plan(plan, self.registry, roots)
                    plan = grounded_plan
                    if not report.is_clean() or report.orphaned_dependents:
                        state.add_warning(
                            f"grounding dropped {len(report.dropped)} step(s); "
                            f"orphaned dependents: {len(report.orphaned_dependents)}"
                        )
                    state.grounding_report = report.to_dict()
                    span.set_attribute("grounding.dropped", len(report.dropped))

                state.plan = plan
                if plan:
                    span.set_attribute("plan.step_count", len(plan.steps))

            except Exception as e:
                span.record_exception(e)
                span.set_attribute("error", True)
                state.add_error(f"Orchestration failed: {str(e)}")

        return state

    def reasoning_node(self, state: COREState) -> COREState:
        """
        Reasoning Node: Execute plan steps with tool calls and error handling.

        Executes:
        - Each step in the plan sequentially
        - Calls appropriate tools with safety checks
        - Handles retries based on retry policy
        - Produces artifacts and intermediate results
        """
        tracer = get_tracer()
        with tracer.start_as_current_span("core.reasoning") as span:
            span.set_attribute("node.name", "reasoning")
            span.set_attribute("pipeline.stage", "3_reasoning")

            state.add_execution_node("reasoning")

            if not state.plan:
                state.add_error("No plan available for reasoning")
                span.set_attribute("error", True)
                span.set_attribute("error.reason", "no_plan")
                return state

            try:
                # If evaluation requested retry of a specific step, start there
                if state.eval_result and state.eval_result.retry_step_id:
                    start_step_id = state.eval_result.retry_step_id
                else:
                    start_step_id = None

                span.set_attribute("plan.goal", state.plan.goal or "")
                span.set_attribute("plan.step_count", len(state.plan.steps))

                # Execute the plan
                results = self.reasoning_agent.execute_plan(
                    plan=state.plan,
                    start_from_step=start_step_id,
                    enable_tools=state.config.get("enable_tools", True),
                )

                # Update state with results
                if start_step_id:
                    # Replace result for retried step
                    state.step_results = [
                        r for r in state.step_results if r.step_id != start_step_id
                    ]
                state.step_results.extend(results)

                span.set_attribute("steps.executed", len(results))

            except Exception as e:
                span.record_exception(e)
                span.set_attribute("error", True)
                state.add_error(f"Reasoning failed: {str(e)}")

        return state

    def evaluation_node(self, state: COREState) -> COREState:
        """
        Evaluation Node: Assess execution quality and determine next action.

        Evaluates:
        - Did we accomplish the goal?
        - What's the quality/confidence of the output?
        - Should we retry a step, revise the plan, or finalize?
        """
        tracer = get_tracer()
        with tracer.start_as_current_span("core.evaluation") as span:
            span.set_attribute("node.name", "evaluation")
            span.set_attribute("pipeline.stage", "4_evaluation")

            state.add_execution_node("evaluation")

            try:
                # Run evaluation agent
                evaluation = self.evaluation_agent.evaluate_execution(
                    user_input=state.user_input,
                    intent=state.intent,
                    plan=state.plan,
                    step_results=state.step_results,
                )

                state.eval_result = evaluation

                if evaluation:
                    span.set_attribute(
                        "eval.status", evaluation.overall_status or "unknown"
                    )
                    span.set_attribute(
                        "eval.next_action", evaluation.next_action or "unknown"
                    )
                    span.set_attribute(
                        "eval.confidence", float(evaluation.confidence or 0.0)
                    )

            except Exception as e:
                span.record_exception(e)
                span.set_attribute("error", True)
                state.add_error(f"Evaluation failed: {str(e)}")
                # Default to success on evaluation failure to avoid infinite loops
                state.eval_result = EvaluationResult(
                    overall_status="success",
                    confidence=0.5,
                    meets_requirements=True,
                    quality_score=0.5,
                    feedback="Evaluation failed, defaulting to success",
                    next_action="finalize",
                )

        return state

    def conversation_node(self, state: COREState) -> COREState:
        """
        Conversation Node: Formulate final response to user.

        Produces:
        - Natural language response (primary content first!)
        - Minimal metadata (only if debugging enabled or low confidence)
        """
        import logging

        logger = logging.getLogger(__name__)

        tracer = get_tracer()
        with tracer.start_as_current_span("core.conversation") as span:
            span.set_attribute("node.name", "conversation")
            span.set_attribute("pipeline.stage", "5_conversation")

            state.add_execution_node("conversation")
            logger.info(
                f"[CONVERSATION] Starting. step_results={len(state.step_results)}, plan={state.plan is not None}, eval={state.eval_result is not None}"
            )

            try:
                # Front gate asked to clarify or honestly refused — surface that
                # message directly (the O→R→E loop never ran).
                if state.gate_message:
                    state.response = state.gate_message

                # For simple conversations (no task execution)
                elif state.intent and state.intent.type == "conversation":
                    state.response = f"I understand you're saying: {state.user_input}. How can I help?"

                # For task executions
                elif state.plan and state.eval_result:
                    response_parts = []

                    # PRIMARY: Add the actual task outputs first (what the user actually wants!)
                    logger.info(
                        f"[CONVERSATION] Processing {len(state.step_results)} step results"
                    )
                    for result in state.step_results:
                        logger.info(
                            f"[CONVERSATION] Step {result.step_id}: outputs={result.outputs}"
                        )
                        if result.outputs:
                            for key, value in result.outputs.items():
                                if key in ("result", "output", "content"):
                                    logger.info(
                                        f"[CONVERSATION] Found '{key}' output: {str(value)[:100]}"
                                    )
                                    response_parts.append(str(value))
                                elif key not in (
                                    "files_modified",
                                    "query_result",
                                    "rows_affected",
                                    "branch",
                                    "commit_sha",
                                    "files_changed",
                                ):
                                    # Skip technical outputs, include named content
                                    response_parts.append(str(value))

                    # If no outputs found, check logs for meaningful content
                    if not response_parts:
                        for result in state.step_results:
                            if result.logs:
                                # Filter out technical logs
                                meaningful_logs = [
                                    log
                                    for log in result.logs
                                    if not log.startswith(
                                        (
                                            "Executing",
                                            "Tool:",
                                            "Parameters:",
                                            "Model:",
                                            "LLM",
                                        )
                                    )
                                ]
                                response_parts.extend(meaningful_logs)

                    # SECONDARY: Add artifacts only if present (useful info)
                    artifacts = [a for r in state.step_results for a in r.artifacts]
                    if artifacts:
                        response_parts.append("")  # blank line
                        response_parts.append("📎 Generated files:")
                        response_parts.extend([f"  • {a}" for a in artifacts[:5]])

                    # METADATA: Only show if debugging enabled or low confidence
                    show_metadata = state.config.get("show_metadata", False)
                    if show_metadata or state.eval_result.confidence < 0.7:
                        response_parts.append("")
                        completed_steps = [
                            s for s in state.plan.steps if s.status == "completed"
                        ]
                        response_parts.append(
                            f"✓ Completed: {state.plan.goal} "
                            f"({len(completed_steps)} steps, {state.eval_result.quality_score:.0%} quality)"
                        )

                        if state.eval_result.confidence < 0.7:
                            response_parts.append(
                                f"⚠️ Low confidence ({state.eval_result.confidence:.0%}) - please review"
                            )

                    state.response = (
                        "\n".join(response_parts)
                        if response_parts
                        else "Task completed."
                    )
                    logger.info(
                        f"[CONVERSATION] Final response set (len={len(state.response)}): {state.response[:200] if state.response else 'NONE'}"
                    )

                else:
                    state.response = (
                        "Task processed. Check execution history for details."
                    )
                    logger.info(
                        "[CONVERSATION] No plan or eval_result, using default response"
                    )

                # Mark as complete
                state.completed_at = state.updated_at
                logger.info(
                    f"[CONVERSATION] Complete. response={state.response[:100] if state.response else 'NONE'}"
                )

            except Exception as e:
                logger.error(f"[CONVERSATION] Exception: {e}", exc_info=True)
                span.record_exception(e)
                span.set_attribute("error", True)
                state.add_error(f"Conversation failed: {str(e)}")
                state.response = "An error occurred while formulating the response."

            # Intelligence Layer: emit structured pipeline run summary for learning/analysis
            try:
                record_pipeline_run(state)
            except Exception as _intel_exc:
                logger.debug("Intelligence log failed (non-critical): %s", _intel_exc)

        return state

    # ======================
    # Routing Logic
    # ======================

    def route_from_comprehension(
        self, state: COREState
    ) -> Literal["orchestration", "conversation"]:
        """
        Route after comprehension based on the front-gate decision.

        - gate "proceed" on an actionable intent (task/question) → orchestration
        - clarify / refuse / plain conversation → conversation (direct response)
        """
        from app.core.agents.comprehension_routing import route_after_comprehension

        return route_after_comprehension(state.intent, state.gate_outcome)

    def route_from_evaluation(
        self, state: COREState
    ) -> Literal["conversation", "orchestration", "reasoning"]:
        """
        Route after evaluation based on next_action.

        - finalize → conversation (we're done)
        - revise_plan → orchestration (plan needs improvement)
        - retry_step → reasoning (execute failed step again)
        - ask_user → conversation (need human input)
        """
        if not state.eval_result:
            return "conversation"  # Default to conversation on error

        # Loop guard: cap plan revisions to prevent infinite cycles
        if (
            state.eval_result.next_action == "revise_plan"
            and state.plan_revisions >= _MAX_PLAN_REVISIONS
        ):
            import logging

            logging.getLogger(__name__).warning(
                "Max plan revisions (%d) reached — forcing finalization",
                _MAX_PLAN_REVISIONS,
            )
            return "conversation"

        action_routes = {
            "finalize": "conversation",
            "revise_plan": "orchestration",
            "retry_step": "reasoning",
            "ask_user": "conversation",
        }

        return action_routes.get(state.eval_result.next_action, "conversation")


# ======================
# Singleton instance
# ======================

_core_graph_instance: COREGraph | None = None


def get_core_graph() -> COREGraph:
    """Get or create the singleton CORE graph instance."""
    global _core_graph_instance
    if _core_graph_instance is None:
        _core_graph_instance = COREGraph()
        _core_graph_instance.initialize_graph()
    return _core_graph_instance
