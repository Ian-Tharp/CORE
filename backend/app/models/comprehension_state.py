from pydantic import BaseModel
from typing import Dict, Any, Literal, Optional
from typing_extensions import TypedDict


class ComprehensionState(TypedDict):
    """State for the Comprehension Graph workflow"""
    user_input: str
    input_type: Optional[Literal["command", "query", "conversation"]]
    knowledge_base_result: Optional[Dict[str, Any]]
    capabilities_result: Optional[Dict[str, Any]]
    routing_decision: Optional[Literal["conversation", "orchestration"]]
    context: Dict[str, Any]
    error: Optional[str]
    completed: bool


class KnowledgeBaseResult(BaseModel):
    """Result from knowledge base lookup"""
    found_answer: bool
    confidence: float
    sources: list[str]
    answer: Optional[str]


class CapabilitiesResult(BaseModel):
    """Result from capabilities matching"""
    can_handle: bool
    matching_capabilities: list[str]
    required_tools: list[str]
    mcp_servers: list[str]
    confidence: float


class ComprehensionContext(BaseModel):
    """Context tracking for comprehension workflow"""
    input_classified: bool = False
    knowledge_base_checked: bool = False
    capabilities_checked: bool = False
    routed_to: Literal["conversation", "orchestration"]
    reason: Literal["WITHIN_EXPERTISE", "OUTSIDE_EXPERTISE"]