from typing import Dict, Any
from langgraph.graph import StateGraph, START, END


class CoreGraph:
    """LangGraph representation of CORE pipeline with conditional edges.

    This graph is primarily used for automation later; the playground UI
    uses step-by-step HTTP endpoints. Nodes return the updated state dict.
    """

    def __init__(self) -> None:
        self.graph: StateGraph | None = None

    def intialize_graph(self) -> None:
        g = StateGraph()

        # Register nodes
        g.add_node("Comprehension", self.comprehension_node)
        g.add_node("Orchestration", self.orchestration_node)
        g.add_node("Reasoning", self.reasoning_node)
        g.add_node("Evaluation", self.evaluation_node)
        g.add_node("Conversation", self.conversation_node)

        # Edges with simple conditional routing stored in state["route"]
        g.add_edge(START, "Comprehension")
        g.add_conditional_edges(
            "Comprehension",
            self._route_from_comprehension,
            {"orchestration": "Orchestration", "conversation": "Conversation"},
        )
        g.add_edge("Orchestration", "Reasoning")
        g.add_edge("Reasoning", "Evaluation")
        g.add_conditional_edges(
            "Evaluation",
            self._route_from_evaluation,
            {
                "conversation": "Conversation",
                "orchestration": "Orchestration",
                "reasoning": "Reasoning",
            },
        )
        g.add_edge("Conversation", END)

        self.graph = g.compile()

    def get_graph(self):
        return self.graph

    # --- Node implementations (placeholder logic; see controllers for LLM I/O) ---
    def comprehension_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state.setdefault("route", "orchestration")
        return state

    def orchestration_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state.setdefault("plan", ["Step 1", "Step 2"]) 
        return state

    def reasoning_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state.setdefault("result", "intermediate result")
        return state

    def evaluation_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state.setdefault("evaluation", "SATISFACTORY")
        state.setdefault("next", "conversation")
        return state

    def conversation_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state.setdefault("final", state.get("result", ""))
        return state

    # --- Conditional routing helpers ---
    def _route_from_comprehension(self, state: Dict[str, Any]) -> str:
        return state.get("route", "orchestration")

    def _route_from_evaluation(self, state: Dict[str, Any]) -> str:
        return state.get("next", "conversation")
