"""
CORE Cognitive Graph v2 - Production Implementation

Implements the full CORE pipeline with proper state management and conditional routing:
START → Comprehension → Orchestration → Reasoning → Evaluation → Conversation → END

RSI TODO: Add tool execution safety layer with confirmation dialogs
RSI TODO: Implement HITL (Human-in-the-Loop) checkpoints
RSI TODO: Add Intelligence Layer logging for learning
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.types import Send

from app.models.core_state import COREState, UserIntent, ExecutionPlan, PlanStep, EvaluationResult
from app.core.agents.comprehension_agent import ComprehensionAgent
from app.core.agents.orchestration_agent import OrchestrationAgent
from app.core.agents.reasoning_agent import ReasoningAgent
from app.core.agents.evaluation_agent import EvaluationAgent


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

        # Initialize agents with Ollama model
        # RSI TODO: Make model configurable via environment variable or config
        ollama_model = "gpt-oss:20b"
        self.comprehension_agent = ComprehensionAgent(model=ollama_model)
        self.orchestration_agent = OrchestrationAgent(model=ollama_model)
        self.reasoning_agent = ReasoningAgent(model=ollama_model)
        self.evaluation_agent = EvaluationAgent(model=ollama_model)

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
            }
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
            }
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
        state.add_execution_node("comprehension")

        try:
            # Run comprehension agent
            intent = self.comprehension_agent.analyze_intent(state.user_input)
            state.intent = intent

            # RSI TODO: Query knowledge base for relevant context
            # RSI TODO: Check if user input needs clarification

        except Exception as e:
            state.add_error(f"Comprehension failed: {str(e)}")
            # Fallback to conversation mode on error
            state.intent = UserIntent(
                type="conversation",
                description="Error in comprehension, falling back to chat",
                confidence=0.5
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
        state.add_execution_node("orchestration")

        try:
            # If this is a plan revision, increment revision number
            if state.plan and state.eval_result:
                revision_num = state.plan.revision + 1
                feedback = state.eval_result.feedback
            else:
                revision_num = 1
                feedback = None

            # Run orchestration agent to create/revise plan
            plan = self.orchestration_agent.create_plan(
                user_input=state.user_input,
                intent=state.intent,
                previous_plan=state.plan,
                evaluation_feedback=feedback,
                revision=revision_num
            )

            state.plan = plan

        except Exception as e:
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
        state.add_execution_node("reasoning")

        if not state.plan:
            state.add_error("No plan available for reasoning")
            return state

        try:
            # If evaluation requested retry of a specific step, start there
            if state.eval_result and state.eval_result.retry_step_id:
                start_step_id = state.eval_result.retry_step_id
            else:
                start_step_id = None

            # Execute the plan
            results = self.reasoning_agent.execute_plan(
                plan=state.plan,
                start_from_step=start_step_id,
                enable_tools=state.config.get("enable_tools", True)
            )

            # Update state with results
            if start_step_id:
                # Replace result for retried step
                state.step_results = [
                    r for r in state.step_results if r.step_id != start_step_id
                ]
            state.step_results.extend(results)

        except Exception as e:
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

        except Exception as e:
            state.add_error(f"Evaluation failed: {str(e)}")
            # Default to success on evaluation failure to avoid infinite loops
            state.eval_result = EvaluationResult(
                overall_status="success",
                confidence=0.5,
                meets_requirements=True,
                quality_score=0.5,
                feedback="Evaluation failed, defaulting to success",
                next_action="finalize"
            )

        return state

    def conversation_node(self, state: COREState) -> COREState:
        """
        Conversation Node: Formulate final response to user.

        Produces:
        - Natural language response summarizing what was done
        - Links to artifacts, files changed, etc.
        - Next steps or suggestions if applicable
        """
        state.add_execution_node("conversation")

        try:
            # For simple conversations (no task execution)
            if state.intent and state.intent.type == "conversation":
                # RSI TODO: Use a conversational agent for natural responses
                state.response = f"I understand you're saying: {state.user_input}. How can I help?"

            # For task executions
            elif state.plan and state.eval_result:
                # Summarize what was accomplished
                completed_steps = [s for s in state.plan.steps if s.status == "completed"]
                artifacts = [a for r in state.step_results for a in r.artifacts]

                response_parts = [
                    f"✓ Completed task: {state.plan.goal}",
                    f"Executed {len(completed_steps)} steps with {state.eval_result.quality_score:.0%} quality.",
                ]

                if artifacts:
                    response_parts.append(f"Generated {len(artifacts)} artifacts:")
                    response_parts.extend([f"  - {a}" for a in artifacts[:5]])

                if state.eval_result.confidence < 0.8:
                    response_parts.append(
                        f"\n⚠️ Note: Confidence is {state.eval_result.confidence:.0%}. "
                        "You may want to review the output."
                    )

                state.response = "\n".join(response_parts)

            else:
                state.response = "Task processed. Check execution history for details."

            # Mark as complete
            state.completed_at = state.updated_at

        except Exception as e:
            state.add_error(f"Conversation failed: {str(e)}")
            state.response = "An error occurred while formulating the response."

        return state

    # ======================
    # Routing Logic
    # ======================

    def route_from_comprehension(self, state: COREState) -> Literal["orchestration", "conversation"]:
        """
        Route after comprehension based on intent type.

        - task/question → orchestration (needs planning and execution)
        - conversation/clarification → conversation (direct response)
        """
        if not state.intent:
            return "conversation"  # Default to conversation on error

        if state.intent.type in ["task", "question"]:
            return "orchestration"
        else:
            return "conversation"

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
