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
# Maximum retry_step passes per plan revision before forcing finalization.
# Prevents infinite retry_step → reasoning → evaluation loops.
_MAX_STEP_RETRIES = int(os.getenv("CORE_MAX_STEP_RETRIES", "3"))

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
from app.evaluation.capture import maybe_capture_fault


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

        self.orchestration_agent = OrchestrationAgent(model=ollama_model)
        self.reasoning_agent = ReasoningAgent(model=ollama_model)
        self.evaluation_agent = EvaluationAgent(model=ollama_model)

        # Capability Registry — the system's self-model. Built FROM the live
        # dispatcher so what comprehension reasons over can't drift from what runs.
        from app.core.tools.dispatcher import ToolDispatcher
        from app.core.registry.builtin import build_registry_from_dispatcher

        _workspace = os.getenv("CORE_WORKSPACE_DIR", os.getcwd())
        _dispatcher = ToolDispatcher(workspace_dir=_workspace)
        self.registry = build_registry_from_dispatcher(_dispatcher)
        # The classifier's tool list comes from the same registry the gate
        # checks, so it can't advertise actions CORE doesn't have.
        self.comprehension_agent = ComprehensionAgent(
            model=ollama_model, registry=self.registry
        )
        # Dispatchable tool roots, used by plan grounding to reject invented tools.
        # Sourced from the dispatcher (not registry ids) so that future MCP
        # capabilities registered into the registry can't make a tool the
        # in-process dispatcher cannot run look valid.
        self._dispatcher_tools = set(_dispatcher.available_tools)

        # Phase 3 memory grounding: built lazily on first use, and only when
        # CORE_COMPREHENSION_MEMORY_GROUNDING is on. Tests may inject one.
        self.memory_grounder = None

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

                # Phase 3 (flag, default off): ground the request in memory.
                # Knowledge gaps become ambiguities, so the gate below clarifies.
                self._ground_in_memory(state, intent)

                # Front verifier: tri-state gate over the capability registry.
                # proceed → run the loop; clarify/refuse → answer directly (no spend).
                from app.core.agents.comprehension_gate import decide_gate

                decision = decide_gate(intent, self.registry, text=state.user_input)
                state.gate_outcome = decision.outcome
                state.gate_message = decision.message or None
                span.set_attribute("gate.outcome", decision.outcome)

                # A blocking gate outcome is an unresolved refusal: if anything
                # downstream runs anyway, the reasoning latch starts tripped.
                from app.core.agents.refusal_guard import gate_refusal

                gate_rec = gate_refusal(decision.outcome, decision.message)
                if gate_rec is not None:
                    state.refusal = gate_rec

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

    def _ground_in_memory(self, state: COREState, intent: UserIntent) -> None:
        """Phase 3: feed known facts and knowledge gaps into the intent.

        No-op unless CORE_COMPREHENSION_MEMORY_GROUNDING is on. Never raises.
        """
        from app.core.memory import grounding as memory

        if not memory.memory_grounding_enabled():
            return
        try:
            if self.memory_grounder is None:
                self.memory_grounder = memory.StoreMemoryGrounder(
                    memory.build_memory_store()
                )
            assessment = memory.assess_knowledge(
                state.user_input,
                self.memory_grounder,
                user_id=state.user_id,
                conversation_id=state.conversation_id,
            )
            memory.apply_to_intent(intent, assessment)
            state.memory_grounding = assessment.to_dict()
        except Exception as exc:  # memory must never block comprehension
            state.add_warning(f"memory grounding failed: {exc}")

    def _record_step_checks(self, state: COREState) -> None:
        """Phase 3: record a deterministic check per executed step. Never raises."""
        try:
            from app.evaluation.step_checks import check_steps

            state.step_checks = [
                c.to_dict() for c in check_steps(state.plan, state.step_results)
            ]
        except Exception as exc:
            import logging

            logging.getLogger(__name__).debug(
                "step checks failed (non-critical): %s", exc
            )

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
                # the plan reaches Reasoning. Dependents of dropped steps are
                # cascade-dropped so nothing runs without its prerequisite.
                if plan is not None:
                    from app.core.registry.grounding import ground_plan

                    grounded_plan, report = ground_plan(
                        plan, self.registry, self._dispatcher_tools
                    )
                    plan = grounded_plan
                    if not report.is_clean() or report.orphaned_dependents:
                        state.add_warning(
                            f"grounding dropped {len(report.dropped)} step(s); "
                            f"orphaned dependents: {len(report.orphaned_dependents)}"
                        )
                    state.grounding_report = report.to_dict()
                    span.set_attribute("grounding.dropped", len(report.dropped))

                    # Confirm tier: mark write steps as HITL checkpoints so the
                    # plan reports them. The reasoning guard enforces the hold.
                    from app.core.agents.action_guard import (
                        confirm_writes_enabled,
                        mark_confirmation_steps,
                    )

                    if confirm_writes_enabled():
                        marked = mark_confirmation_steps(plan, self.registry)
                        span.set_attribute("plan.write_checkpoints", marked)

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

                # Abstention guard: one latch per run, seeded with any refusal
                # already on the state (gate or an earlier reasoning pass).
                from app.core.agents.refusal_guard import RefusalLatch, guard_enabled

                latch = (
                    RefusalLatch(existing=state.refusal) if guard_enabled() else None
                )

                # Confirm tier + dependency latch: one guard per pass, seeded
                # with what earlier passes of this run already hold.
                from app.core.agents.action_guard import (
                    ActionGuard,
                    confirm_writes_enabled,
                )

                guard = ActionGuard(
                    registry=self.registry,
                    approved=state.config.get("approved_capabilities") or (),
                    confirm=confirm_writes_enabled(),
                    awaiting=[p.step_id for p in state.pending_confirmations]
                    + [
                        b.step_id
                        for b in state.blocked_actions
                        if b.kind == "awaiting_confirmation"
                    ],
                )

                # Execute the plan
                results = self.reasoning_agent.execute_plan(
                    plan=state.plan,
                    start_from_step=start_step_id,
                    enable_tools=state.config.get("enable_tools", True),
                    refusal_latch=latch,
                    action_guard=guard,
                    prior_results=list(state.step_results) if start_step_id else None,
                )

                # A retry replaces the results (and block records) of every
                # step it re-ran: the retried step and any resumed dependents.
                rerun = {r.step_id for r in results}
                if start_step_id:
                    rerun.add(start_step_id)
                    state.blocked_actions = [
                        b for b in state.blocked_actions if b.step_id not in rerun
                    ]
                    state.pending_confirmations = [
                        p for p in state.pending_confirmations if p.step_id not in rerun
                    ]

                if latch is not None:
                    state.refusal = latch.record
                    state.blocked_actions.extend(latch.blocked)
                    if latch.blocked:
                        state.add_warning(
                            f"abstention guard blocked {len(latch.blocked)} step(s) "
                            f"after an unresolved {state.refusal.source} refusal"
                        )
                    span.set_attribute("refusal.latched", state.refusal is not None)
                    span.set_attribute("refusal.blocked", len(latch.blocked))

                state.blocked_actions.extend(guard.blocked)
                state.pending_confirmations.extend(guard.pending)
                if guard.pending:
                    state.add_warning(
                        f"confirm tier held {len(guard.pending)} step(s) for the "
                        "user's go-ahead: "
                        + ", ".join(p.capability_id for p in guard.pending[:5])
                    )
                span.set_attribute("confirm.held", len(guard.pending))
                span.set_attribute("dependency.blocked", len(guard.blocked))

                # Update state with results
                if start_step_id:
                    order = {s.id: i for i, s in enumerate(state.plan.steps)}
                    kept = [r for r in state.step_results if r.step_id not in rerun]
                    state.step_results = sorted(
                        kept + list(results),
                        key=lambda r: order.get(r.step_id, len(order)),
                    )
                else:
                    state.step_results.extend(results)

                span.set_attribute("steps.executed", len(results))

            except Exception as e:
                span.record_exception(e)
                span.set_attribute("error", True)
                state.add_error(f"Reasoning failed: {str(e)}")

            # Phase 3 per-step evaluation: deterministic, no LLM.
            self._record_step_checks(state)

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
                # Run evaluation agent. Failing per-step checks are passed in
                # only when present, so the clean-run call is unchanged.
                extra = {}
                failing = [c for c in state.step_checks if not c.get("passed", True)]
                if failing:
                    extra["step_checks"] = failing
                evaluation = self.evaluation_agent.evaluate_execution(
                    user_input=state.user_input,
                    intent=state.intent,
                    plan=state.plan,
                    step_results=state.step_results,
                    **extra,
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

            self._verify_claims(state, span)
            self._hold_for_confirmation(state, span)

        return state

    def _hold_for_confirmation(self, state: COREState, span=None) -> None:
        """Held writes need the user, not another pass: turn retry/revise
        into ask_user so the run ends with the confirmation request."""
        ev = state.eval_result
        if not state.pending_confirmations or ev is None:
            return
        if span is not None:
            span.set_attribute("confirm.pending", len(state.pending_confirmations))
        if ev.next_action in ("retry_step", "revise_plan", "finalize"):
            ev.next_action = "ask_user"
            ev.retry_step_id = None
            note = "Waiting for the user to confirm: " + ", ".join(
                p.capability_id for p in state.pending_confirmations[:5]
            )
            ev.feedback = f"{ev.feedback}\n{note}" if ev.feedback else note

    # Confidence ceiling when a recognised action claim has no tool call behind it.
    # Below the conversation node's 0.7 threshold, so the low-confidence notice shows.
    _UNSUPPORTED_CLAIM_CONFIDENCE = 0.5

    def _verify_claims(self, state: COREState, span=None) -> None:
        """Back verifier, deterministic half: check LLM step outputs' action
        claims against the dispatched tool-call trace. Annotates the evaluation
        (never changes next_action, so it cannot cause a loop)."""
        import logging

        try:
            from app.evaluation.claims import verify_state_claims

            report = verify_state_claims(state)
            state.claim_report = report.to_dict()
            bad = report.unsupported
            if span is not None:
                span.set_attribute("claims.checked", len(report.checks))
                span.set_attribute("claims.unsupported", len(bad))
            if not bad:
                return
            state.add_warning(
                f"{len(bad)} action claim(s) not backed by any tool call: "
                + "; ".join(c.text for c in bad[:3])
            )
            ev = state.eval_result
            if ev is not None:
                ev.meets_requirements = False
                ev.confidence = min(ev.confidence, self._UNSUPPORTED_CLAIM_CONFIDENCE)
                note = "Unsupported action claims: " + "; ".join(
                    c.text for c in bad[:3]
                )
                ev.feedback = f"{ev.feedback}\n{note}" if ev.feedback else note
        except Exception as exc:  # verification must never break the run
            logging.getLogger(__name__).debug(
                "claim verification failed (non-critical): %s", exc
            )

    @staticmethod
    def _confirmation_notice(state: COREState) -> list:
        """Lines asking the user to confirm held writes (empty if none)."""
        held = state.pending_confirmations
        if not held:
            return []
        lines = [
            "I need your go-ahead before changing anything. "
            f"I did not run {len(held)} step(s):"
        ]
        for p in held[:5]:
            what = p.capability_id + (f" → {p.target}" if p.target else "")
            lines.append(f"  • {p.step_name or p.step_id} ({what})")
        waiting = [
            b for b in state.blocked_actions if b.kind == "awaiting_confirmation"
        ]
        if waiting:
            lines.append(
                f"{len(waiting)} step(s) that depend on them also waited: "
                + ", ".join(b.step_name or b.step_id for b in waiting[:5])
                + "."
            )
        caps = sorted({p.capability_id for p in held})
        lines.append(
            "To go ahead, send the request again with approval for: "
            + ", ".join(caps)
            + "."
        )
        return lines

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

                    # Notices lead the response, before any partial output:
                    # an unresolved refusal first, then writes held for the
                    # user's go-ahead.
                    notice = []
                    if state.refusal and state.refusal.source == "reasoning":
                        notice.append(
                            "I stopped before finishing: a step declined "
                            f"({state.refusal.reason})."
                        )
                        refused = [
                            b for b in state.blocked_actions if b.kind == "refusal"
                        ]
                        if refused:
                            names = ", ".join(
                                b.step_name or b.step_id for b in refused[:5]
                            )
                            notice.append(
                                f"I did not run the remaining "
                                f"{len(refused)} step(s): {names}."
                            )
                    notice.extend(self._confirmation_notice(state))
                    if notice:
                        response_parts = notice + [""] + response_parts

                    dep_blocked = [
                        b
                        for b in state.blocked_actions
                        if b.kind == "dependency_failed"
                    ]
                    if dep_blocked:
                        response_parts.append("")
                        response_parts.append(
                            "⚠️ Not run: a step they depend on did not succeed: "
                            + ", ".join(
                                b.step_name or b.step_id for b in dep_blocked[:5]
                            )
                        )

                    unsupported = (state.claim_report or {}).get("unsupported") or []
                    if unsupported:
                        response_parts.append("")
                        response_parts.append(
                            "⚠️ Not verified: no tool call backs these statements:"
                        )
                        response_parts.extend(
                            f"  • {c.get('text', '')}" for c in unsupported[:5]
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

            # Evaluation layer: persist faulted runs for the Golden Set loop
            # (docs/evaluation_spec.md 3.C). Off unless CORE_EVAL_CAPTURE_FAULTS
            # is set; maybe_capture_fault never raises, the guard is belt and
            # braces so capture can never break a user request.
            try:
                maybe_capture_fault(state)
            except Exception as _capture_exc:
                logger.warning("Fault capture failed (non-critical): %s", _capture_exc)

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

        # Abstention guard: never retry or re-plan around an unresolved refusal.
        # Only the user can resolve it, so the run ends with the refusal surfaced.
        if state.refusal is not None and state.eval_result.next_action in (
            "retry_step",
            "revise_plan",
        ):
            from app.core.agents.refusal_guard import guard_enabled

            if guard_enabled():
                import logging

                logging.getLogger(__name__).warning(
                    "Unresolved %s refusal — not routing to %s",
                    state.refusal.source,
                    state.eval_result.next_action,
                )
                return "conversation"

        # Confirm tier: a held write is waiting on the user. Retrying holds it
        # again; re-planning could route around it. Either way, ask the user.
        if state.pending_confirmations and state.eval_result.next_action in (
            "retry_step",
            "revise_plan",
        ):
            from app.core.agents.action_guard import confirm_writes_enabled

            if confirm_writes_enabled():
                return "conversation"

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

        # Loop guard: cap step retries since the latest plan (re)creation
        if state.eval_result.next_action == "retry_step":
            history = state.execution_history
            last_orch = (
                len(history) - 1 - history[::-1].index("orchestration")
                if "orchestration" in history
                else -1
            )
            retries_done = history[last_orch + 1 :].count("reasoning") - 1
            if retries_done >= _MAX_STEP_RETRIES:
                import logging

                logging.getLogger(__name__).warning(
                    "Max step retries (%d) reached — forcing finalization",
                    _MAX_STEP_RETRIES,
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
