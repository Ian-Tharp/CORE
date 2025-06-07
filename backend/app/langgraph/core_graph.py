from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END



class CoreGraph(BaseModel):
    def __init__(self):
        self.graph = None

    def intialize_graph(self):
        self.graph = StateGraph(
            nodes=[
                START,
                "Comprehension",
                "Orchestration",
                "Reasoning",
                "Evaluation",
                "Conversation",
                END
            ],
            transitions={
                START: "Comprehension",
                "Comprehension": "Orchestration", # Based on user input, go directly to Orchestration
                "Comprehension": "Conversation", # Based on user input, go directly to Conversation
                "Orchestration": "Reasoning", # Pass the plan to Reasoning to execute
                "Reasoning": "Evaluation", # Pass the result(s) to Evaluation to evaluate the task
                "Evaluation": "Conversation", # Successfully complete the task, go to Conversation
                "Evaluation": "Orchestration", # Based on evaluation, revise the plan - go back to Orchestration
                "Evaluation": "Reasoning", # Based on the evaluation, continue execution of the plan
                "Conversation": END
            }
        )
        self.graph.add_node("Comprehension", self.comprehension_node)
        self.graph.add_node("Orchestration", self.orchestration_node)
        self.graph.add_node("Reasoning", self.reasoning_node)
        self.graph.add_node("Evaluation", self.evaluation_node)
        self.graph.add_node("Conversation", self.conversation_node)

        self.graph.compile()

    def get_graph(self):
        return self.graph
    
    def comprehension_node(self, state: dict):
        return state
    
    def orchestration_node(self, state: dict):
        return state
    
    def reasoning_node(self, state: dict):
        return state
    
    def evaluation_node(self, state: dict):
        return state
    
    def conversation_node(self, state: dict):
        return state
