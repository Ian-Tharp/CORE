from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END

# Import models from the organized models directory
from app.models import (
    ComprehensionState,
    KnowledgeBaseResult,
    CapabilitiesResult,
    ComprehensionOutput
)


class ComprehensionGraph(BaseModel):
    def __init__(self):
        self.graph = None

    def initialize_graph(self):
        """Initialize the comprehension graph with all nodes and edges"""
        workflow = StateGraph(ComprehensionState)
        
        # Add all nodes
        workflow.add_node("check_input", self.check_input_node)
        workflow.add_node("check_knowledge_base", self.check_knowledge_base_node)
        workflow.add_node("check_capabilities", self.check_capabilities_node)
        workflow.add_node("route_to_conversation", self.route_to_conversation_node)
        workflow.add_node("route_to_orchestration", self.route_to_orchestration_node)
        
        # Define the workflow edges
        workflow.add_edge(START, "check_input")
        workflow.add_edge("check_input", "check_knowledge_base")
        
        # Conditional edge from knowledge base check
        workflow.add_conditional_edges(
            "check_knowledge_base",
            self._should_check_capabilities,
            {
                "check_capabilities": "check_capabilities",
                "route_to_conversation": "route_to_conversation"
            }
        )
        
        # Conditional edge from capabilities check
        workflow.add_conditional_edges(
            "check_capabilities",
            self._route_decision,
            {
                "conversation": "route_to_conversation",
                "orchestration": "route_to_orchestration"
            }
        )
        
        # Final edges to END
        workflow.add_edge("route_to_conversation", END)
        workflow.add_edge("route_to_orchestration", END)
        
        self.graph = workflow.compile()

    def _should_check_capabilities(self, state: ComprehensionState) -> str:
        """Determine if we should check capabilities or route to conversation"""
        if state.get("knowledge_base_result") and state["knowledge_base_result"].get("found_answer"):
            return "route_to_conversation"
        return "check_capabilities"
    
    def _route_decision(self, state: ComprehensionState) -> str:
        """Determine final routing based on capabilities check"""
        if state.get("routing_decision"):
            return state["routing_decision"]
        return "conversation"

    def get_graph(self):
        return self.graph
    
    def check_input_node(self, state: ComprehensionState) -> ComprehensionState:
        # Agent will check the user input and check if it is a command or a query.
        user_input = state.get("user_input", "")
        
        # Basic input classification logic (to be enhanced with actual AI agent)
        if user_input.startswith(("/", "!", "cmd:")):
            input_type = "command"
        else:
            # Default to query, but check if it's actually a conversation based on state
            if state.get("input_type") == "conversation":
                input_type = "conversation"
            else:
                input_type = "query"
        
        return {
            **state,
            "input_type": input_type,
            "context": {**state.get("context", {}), "input_classified": True}
        }

    def check_knowledge_base_node(self, state: ComprehensionState) -> ComprehensionState:
        # Agent will check the user input against the system's knowledge base
        # to see if we can just retrieve answer via RAG/Knowledgebase.
        
        # Create structured knowledge base result
        knowledge_result = KnowledgeBaseResult(
            found_answer=False,  # Would be determined by actual knowledge base search
            confidence=0.0,
            sources=[],
            answer=None
        )
        
        return {
            **state,
            "knowledge_base_result": knowledge_result.model_dump(),
            "context": {**state.get("context", {}), "knowledge_base_checked": True}
        }

    def check_capabilities_node(self, state: ComprehensionState) -> ComprehensionState:
        # Agent will check the user input against the system's list of capabilities
        # to see if we can/should process the user query via the capabilities.
        # Capabilities are the tools that agents have from the agent registry, as well as the system's list of capabilities.
        # This is also the check taking a look at the MCP server registry and available tools that could be dynamically
        # bound to an agent to complete the task.
        
        # Create structured capabilities result
        capabilities_result = CapabilitiesResult(
            can_handle=False,  # Would be determined by actual capabilities matching
            matching_capabilities=[],
            required_tools=[],
            mcp_servers=[],
            confidence=0.0
        )
        
        # Determine routing based on capabilities
        if capabilities_result.can_handle:
            routing_decision = "orchestration"
        else:
            routing_decision = "conversation"
        
        return {
            **state,
            "capabilities_result": capabilities_result.model_dump(),
            "routing_decision": routing_decision,
            "context": {**state.get("context", {}), "capabilities_checked": True}
        }

    def route_to_conversation_node(self, state: ComprehensionState) -> ComprehensionState:
        # Agent will route to the conversation node if the user input is not a command or a query, or it is
        # a query that we cannot process via the knowledge base or capabilities, ultimately returning an
        # OUTSIDE_EXPERTISE enum, which will be sent to the Conversation node to send the final response to the user.
        
        return {
            **state,
            "completed": True,
            "context": {
                **state.get("context", {}), 
                "routed_to": "conversation",
                "reason": ComprehensionOutput.OUTSIDE_EXPERTISE.value
            }
        }

    def route_to_orchestration_node(self, state: ComprehensionState) -> ComprehensionState:
        # Agent/output will complete this cycle of the graph and output to the Orchestration node.
        # The CORE Sidecar will be the main glue between each step of the CORE workflow from the different graphs.
        
        return {
            **state,
            "completed": True,
            "context": {
                **state.get("context", {}), 
                "routed_to": "orchestration",
                "reason": ComprehensionOutput.WITHIN_EXPERTISE.value
            }
        }

