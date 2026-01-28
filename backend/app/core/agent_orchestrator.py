"""
Agent Orchestrator for CORE

Coordinates sandboxed agent execution with MCP tool selection
and state management as part of the CORE loop.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from uuid import uuid4
from enum import Enum

from pydantic import BaseModel, Field

# Import sandbox components
from app.sandbox import (
    ContainerManager,
    ContainerConfig,
    TrustLevel,
    AgentSecurityConfig,
    TRUST_PRESETS,
    StateManager,
    AgentState,
)

# Note: Import dynamic tool selector from mcp module
import sys
sys.path.append("../mcp")
from mcp.dynamic_tool_selector import DynamicToolSelector, ToolSelection

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Roles for agents in the CORE loop."""
    COMPREHENSION = "comprehension"
    ORCHESTRATION = "orchestration"
    REASONING = "reasoning"
    EVALUATION = "evaluation"
    CONVERSATION = "conversation"
    TOOL_EXECUTOR = "tool_executor"


class TaskStatus(str, Enum):
    """Status of a task execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentTask(BaseModel):
    """A task to be executed by an agent."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: AgentRole
    description: str
    input_data: Dict[str, Any] = Field(default_factory=dict)
    required_capabilities: List[str] = Field(default_factory=list)
    trust_level: TrustLevel = TrustLevel.TRUSTED
    timeout_seconds: int = 300
    parent_task_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: TaskStatus = TaskStatus.PENDING
    
    class Config:
        use_enum_values = True


class TaskResult(BaseModel):
    """Result of a task execution."""
    
    task_id: str
    success: bool
    output: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    execution_time_ms: int = 0
    tools_used: List[str] = Field(default_factory=list)
    artifacts: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class AgentConfig(BaseModel):
    """Configuration for an agent instance."""
    
    role: AgentRole
    name: str
    model: str = "gpt-4"  # Default LLM model
    security: AgentSecurityConfig = Field(default_factory=lambda: TRUST_PRESETS[TrustLevel.TRUSTED])
    mcp_capabilities: List[str] = Field(default_factory=list)
    system_prompt: Optional[str] = None
    max_iterations: int = 10
    
    class Config:
        use_enum_values = True


class AgentOrchestrator:
    """
    Orchestrates agent execution within the CORE loop.
    
    Responsibilities:
    - Spawn and manage sandboxed agent containers
    - Select appropriate MCP tools for tasks
    - Coordinate state between agents
    - Handle task routing and execution
    
    This integrates with the CORE StateGraph as the execution layer
    for agent nodes.
    """
    
    def __init__(
        self,
        container_manager: ContainerManager,
        state_manager: StateManager,
        tool_selector: DynamicToolSelector,
        mcp_registry_url: str = "http://localhost:8000"
    ):
        """
        Initialize the orchestrator.
        
        Args:
            container_manager: Manager for sandboxed containers
            state_manager: Manager for agent state
            tool_selector: Dynamic tool selector
            mcp_registry_url: URL of the MCP registry
        """
        self.container_manager = container_manager
        self.state_manager = state_manager
        self.tool_selector = tool_selector
        self.mcp_registry_url = mcp_registry_url
        
        # Active tasks
        self._tasks: Dict[str, AgentTask] = {}
        
        # Agent configurations by role
        self._agent_configs: Dict[AgentRole, AgentConfig] = {}
        
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
        
        # Initialize default agent configs
        self._setup_default_agents()
    
    def _setup_default_agents(self):
        """Setup default agent configurations for CORE roles."""
        
        self._agent_configs[AgentRole.COMPREHENSION] = AgentConfig(
            role=AgentRole.COMPREHENSION,
            name="comprehension-agent",
            model="gpt-4",
            mcp_capabilities=["file:read", "web:search"],
            system_prompt="""You are the Comprehension Agent in the CORE system.
Your role is to understand user intent, extract key information, and classify requests.
Analyze the input and produce a structured understanding."""
        )
        
        self._agent_configs[AgentRole.ORCHESTRATION] = AgentConfig(
            role=AgentRole.ORCHESTRATION,
            name="orchestration-agent",
            model="gpt-4",
            mcp_capabilities=["*"],  # Full access to plan tool usage
            system_prompt="""You are the Orchestration Agent in the CORE system.
Your role is to create execution plans, coordinate tools, and manage task flow.
Given a comprehension result, produce a step-by-step execution plan."""
        )
        
        self._agent_configs[AgentRole.REASONING] = AgentConfig(
            role=AgentRole.REASONING,
            name="reasoning-agent",
            model="gpt-4",
            security=TRUST_PRESETS[TrustLevel.SANDBOXED],  # Sandboxed for tool execution
            mcp_capabilities=["*"],
            system_prompt="""You are the Reasoning Agent in the CORE system.
Your role is to execute plans using available tools and synthesize results.
Execute the given plan step by step, using tools as needed."""
        )
        
        self._agent_configs[AgentRole.EVALUATION] = AgentConfig(
            role=AgentRole.EVALUATION,
            name="evaluation-agent",
            model="gpt-4",
            mcp_capabilities=[],  # No tools needed for evaluation
            system_prompt="""You are the Evaluation Agent in the CORE system.
Your role is to assess execution results against the original intent.
Determine if the task was successful or needs revision."""
        )
        
        self._agent_configs[AgentRole.CONVERSATION] = AgentConfig(
            role=AgentRole.CONVERSATION,
            name="conversation-agent",
            model="gpt-4",
            mcp_capabilities=[],
            system_prompt="""You are the Conversation Agent in the CORE system.
Your role is to formulate clear, helpful responses to the user.
Synthesize execution results into a coherent response."""
        )
        
        self._agent_configs[AgentRole.TOOL_EXECUTOR] = AgentConfig(
            role=AgentRole.TOOL_EXECUTOR,
            name="tool-executor-agent",
            model="gpt-4",
            security=TRUST_PRESETS[TrustLevel.SANDBOXED],
            mcp_capabilities=["*"],
            system_prompt="""You are the Tool Executor Agent.
Your role is to execute specific tool calls and return results.
Execute the requested tool with the given parameters."""
        )
    
    async def initialize(self):
        """Initialize the orchestrator and its dependencies."""
        logger.info("Initializing AgentOrchestrator...")
        
        # Initialize sub-components
        await self.container_manager.initialize()
        await self.state_manager.initialize()
        
        # Refresh tool cache
        await self.tool_selector.refresh_tool_cache()
        
        logger.info("AgentOrchestrator initialized")
    
    async def shutdown(self):
        """Shutdown the orchestrator."""
        logger.info("Shutting down AgentOrchestrator...")
        
        await self.container_manager.shutdown()
        
        logger.info("AgentOrchestrator shutdown complete")
    
    async def submit_task(
        self,
        role: AgentRole,
        description: str,
        input_data: Dict[str, Any],
        required_capabilities: Optional[List[str]] = None,
        trust_level: Optional[TrustLevel] = None,
        parent_task_id: Optional[str] = None
    ) -> AgentTask:
        """
        Submit a task for execution.
        
        Args:
            role: Agent role to execute the task
            description: Task description
            input_data: Input data for the task
            required_capabilities: Required MCP capabilities
            trust_level: Override trust level for this task
            parent_task_id: ID of parent task (for sub-tasks)
            
        Returns:
            Created AgentTask
        """
        agent_config = self._agent_configs.get(role)
        if not agent_config:
            raise ValueError(f"No agent configured for role: {role}")
        
        task = AgentTask(
            role=role,
            description=description,
            input_data=input_data,
            required_capabilities=required_capabilities or agent_config.mcp_capabilities,
            trust_level=trust_level or agent_config.security.trust_level,
            timeout_seconds=agent_config.security.timeout_seconds,
            parent_task_id=parent_task_id
        )
        
        async with self._lock:
            self._tasks[task.id] = task
        
        logger.info(f"Task submitted: {task.id} for role {role}")
        return task
    
    async def execute_task(self, task_id: str) -> TaskResult:
        """
        Execute a submitted task.
        
        Args:
            task_id: ID of the task to execute
            
        Returns:
            TaskResult with execution outcome
        """
        task = self._tasks.get(task_id)
        if not task:
            return TaskResult(
                task_id=task_id,
                success=False,
                error=f"Task {task_id} not found"
            )
        
        # Update status
        task.status = TaskStatus.RUNNING
        start_time = datetime.utcnow()
        
        try:
            # Get agent configuration
            agent_config = self._agent_configs[task.role]
            
            # Select tools for this task
            tool_selection = await self.tool_selector.select_tools_for_task(
                task_description=task.description,
                explicit_capabilities=task.required_capabilities,
                constraints={"max_latency_ms": task.timeout_seconds * 1000}
            )
            
            # Get or create container for execution
            container_config = ContainerConfig(
                name=f"core-{task.role.value}-{task.id[:8]}",
                security=TRUST_PRESETS[task.trust_level],
                mcp_capabilities=[t.name for t in tool_selection.selected_tools]
            )
            
            container_info = await self.container_manager.get_container(container_config)
            
            try:
                # Store task state
                await self.state_manager.set_state(
                    agent_id=agent_config.name,
                    task_id=task.id,
                    data={
                        "status": "executing",
                        "input": task.input_data,
                        "tools_available": [t.name for t in tool_selection.selected_tools]
                    }
                )
                
                # Execute based on role
                result = await self._execute_role_task(
                    task=task,
                    agent_config=agent_config,
                    container_id=container_info.id,
                    tool_selection=tool_selection
                )
                
                # Update state with result
                await self.state_manager.set_state(
                    agent_id=agent_config.name,
                    task_id=task.id,
                    data={
                        "status": "completed" if result.success else "failed",
                        "output": result.output,
                        "error": result.error
                    }
                )
                
                execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                result.execution_time_ms = execution_time_ms
                
                task.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
                return result
            
            finally:
                # Release container
                await self.container_manager.release_container(
                    container_info.id,
                    return_to_pool=True
                )
        
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            task.status = TaskStatus.FAILED
            
            return TaskResult(
                task_id=task_id,
                success=False,
                error=str(e),
                execution_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
            )
    
    async def _execute_role_task(
        self,
        task: AgentTask,
        agent_config: AgentConfig,
        container_id: str,
        tool_selection: ToolSelection
    ) -> TaskResult:
        """
        Execute a task based on its role.
        
        This is the core execution logic that would typically invoke
        an LLM with the appropriate prompt and tools.
        """
        role = task.role
        
        # Build execution context
        context = {
            "task_id": task.id,
            "role": role.value,
            "description": task.description,
            "input_data": task.input_data,
            "available_tools": [
                {"name": t.name, "description": t.description}
                for t in tool_selection.selected_tools
            ],
            "system_prompt": agent_config.system_prompt
        }
        
        # This is where you would integrate with your LLM service
        # For now, we'll return a placeholder result
        
        if role == AgentRole.COMPREHENSION:
            return await self._execute_comprehension(context, container_id)
        elif role == AgentRole.ORCHESTRATION:
            return await self._execute_orchestration(context, container_id)
        elif role == AgentRole.REASONING:
            return await self._execute_reasoning(context, container_id, tool_selection)
        elif role == AgentRole.EVALUATION:
            return await self._execute_evaluation(context, container_id)
        elif role == AgentRole.CONVERSATION:
            return await self._execute_conversation(context, container_id)
        elif role == AgentRole.TOOL_EXECUTOR:
            return await self._execute_tool_call(context, container_id, tool_selection)
        else:
            return TaskResult(
                task_id=task.id,
                success=False,
                error=f"Unknown role: {role}"
            )
    
    async def _execute_comprehension(
        self,
        context: Dict[str, Any],
        container_id: str
    ) -> TaskResult:
        """Execute comprehension phase."""
        # Placeholder - integrate with actual LLM
        logger.info(f"Executing comprehension for task {context['task_id']}")
        
        # In production, this would:
        # 1. Send user input to LLM with comprehension prompt
        # 2. Parse response into structured understanding
        # 3. Determine if this is a simple query or complex task
        
        return TaskResult(
            task_id=context["task_id"],
            success=True,
            output={
                "intent": "task_execution",  # or "simple_query"
                "entities": [],
                "context_required": True,
                "tools_likely_needed": True
            },
            metadata={"phase": "comprehension"}
        )
    
    async def _execute_orchestration(
        self,
        context: Dict[str, Any],
        container_id: str
    ) -> TaskResult:
        """Execute orchestration phase - create execution plan."""
        logger.info(f"Executing orchestration for task {context['task_id']}")
        
        # In production, this would:
        # 1. Analyze comprehension output
        # 2. Determine required tools and sequence
        # 3. Create step-by-step execution plan
        
        return TaskResult(
            task_id=context["task_id"],
            success=True,
            output={
                "plan": [
                    {"step": 1, "action": "gather_information", "tool": "web_search"},
                    {"step": 2, "action": "process_data", "tool": "python_exec"},
                    {"step": 3, "action": "generate_response", "tool": None}
                ],
                "estimated_steps": 3,
                "tools_required": ["web_search", "python_exec"]
            },
            metadata={"phase": "orchestration"}
        )
    
    async def _execute_reasoning(
        self,
        context: Dict[str, Any],
        container_id: str,
        tool_selection: ToolSelection
    ) -> TaskResult:
        """Execute reasoning phase - run the plan with tools."""
        logger.info(f"Executing reasoning for task {context['task_id']}")
        
        tools_used = []
        
        # In production, this would:
        # 1. Execute each step of the plan
        # 2. Call MCP tools as needed
        # 3. Handle errors and retries
        # 4. Collect and synthesize results
        
        # Example: Execute a simple Python command in the container
        if tool_selection.selected_tools:
            # This would actually call MCP tools
            tools_used = [t.name for t in tool_selection.selected_tools[:3]]
        
        return TaskResult(
            task_id=context["task_id"],
            success=True,
            output={
                "execution_results": [],
                "synthesized_output": "Task completed successfully",
                "intermediate_results": []
            },
            tools_used=tools_used,
            metadata={"phase": "reasoning"}
        )
    
    async def _execute_evaluation(
        self,
        context: Dict[str, Any],
        container_id: str
    ) -> TaskResult:
        """Execute evaluation phase - assess results."""
        logger.info(f"Executing evaluation for task {context['task_id']}")
        
        # In production, this would:
        # 1. Compare results against original intent
        # 2. Check for completeness and correctness
        # 3. Determine if revision is needed
        
        return TaskResult(
            task_id=context["task_id"],
            success=True,
            output={
                "evaluation": "pass",  # or "needs_revision"
                "confidence": 0.85,
                "issues": [],
                "suggestions": [],
                "next_action": "conversation"  # or "orchestration" for revision
            },
            metadata={"phase": "evaluation"}
        )
    
    async def _execute_conversation(
        self,
        context: Dict[str, Any],
        container_id: str
    ) -> TaskResult:
        """Execute conversation phase - generate response."""
        logger.info(f"Executing conversation for task {context['task_id']}")
        
        # In production, this would:
        # 1. Synthesize all results into coherent response
        # 2. Format appropriately for user
        # 3. Include relevant citations/sources
        
        return TaskResult(
            task_id=context["task_id"],
            success=True,
            output={
                "response": "Here is the response to your request...",
                "sources": [],
                "follow_up_suggestions": []
            },
            metadata={"phase": "conversation"}
        )
    
    async def _execute_tool_call(
        self,
        context: Dict[str, Any],
        container_id: str,
        tool_selection: ToolSelection
    ) -> TaskResult:
        """Execute a specific tool call."""
        logger.info(f"Executing tool call for task {context['task_id']}")
        
        tool_name = context["input_data"].get("tool_name")
        tool_params = context["input_data"].get("parameters", {})
        
        # Find the tool
        selected_tool = None
        for tool in tool_selection.selected_tools:
            if tool.name == tool_name:
                selected_tool = tool
                break
        
        if not selected_tool:
            return TaskResult(
                task_id=context["task_id"],
                success=False,
                error=f"Tool {tool_name} not found in selection"
            )
        
        # In production, this would call the actual MCP tool
        # via the MCP client manager
        
        return TaskResult(
            task_id=context["task_id"],
            success=True,
            output={
                "tool_result": {"status": "success"},
                "tool_name": tool_name
            },
            tools_used=[tool_name],
            metadata={"phase": "tool_execution"}
        )
    
    async def execute_core_loop(
        self,
        user_input: str,
        session_id: str,
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Execute the full CORE loop for a user input.
        
        CORE Loop:
        1. Comprehension - Understand intent
        2. Orchestration - Plan execution
        3. Reasoning - Execute with tools
        4. Evaluation - Assess results
        5. (Loop back to Orchestration if needed)
        6. Conversation - Generate response
        
        Args:
            user_input: User's input text
            session_id: Session identifier
            max_iterations: Maximum OODA iterations before forcing completion
            
        Returns:
            Final response and execution metadata
        """
        logger.info(f"Starting CORE loop for session {session_id}")
        
        # Phase 1: Comprehension
        comprehension_task = await self.submit_task(
            role=AgentRole.COMPREHENSION,
            description="Understand user intent and extract key information",
            input_data={"user_input": user_input, "session_id": session_id}
        )
        comprehension_result = await self.execute_task(comprehension_task.id)
        
        if not comprehension_result.success:
            return {
                "success": False,
                "error": comprehension_result.error,
                "phase": "comprehension"
            }
        
        # Check if simple query (skip orchestration/reasoning)
        if comprehension_result.output.get("intent") == "simple_query":
            # Direct to conversation
            conversation_task = await self.submit_task(
                role=AgentRole.CONVERSATION,
                description="Generate response for simple query",
                input_data={
                    "comprehension": comprehension_result.output,
                    "user_input": user_input
                },
                parent_task_id=comprehension_task.id
            )
            conversation_result = await self.execute_task(conversation_task.id)
            
            return {
                "success": conversation_result.success,
                "response": conversation_result.output.get("response"),
                "phase": "conversation",
                "path": "simple_query"
            }
        
        # Complex task - enter OODA loop
        iteration = 0
        orchestration_input = comprehension_result.output
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"CORE loop iteration {iteration}")
            
            # Phase 2: Orchestration
            orchestration_task = await self.submit_task(
                role=AgentRole.ORCHESTRATION,
                description="Create execution plan",
                input_data={
                    "comprehension": orchestration_input,
                    "iteration": iteration
                },
                parent_task_id=comprehension_task.id
            )
            orchestration_result = await self.execute_task(orchestration_task.id)
            
            if not orchestration_result.success:
                return {
                    "success": False,
                    "error": orchestration_result.error,
                    "phase": "orchestration",
                    "iteration": iteration
                }
            
            # Phase 3: Reasoning
            reasoning_task = await self.submit_task(
                role=AgentRole.REASONING,
                description="Execute plan with tools",
                input_data={
                    "plan": orchestration_result.output,
                    "comprehension": comprehension_result.output
                },
                required_capabilities=orchestration_result.output.get("tools_required", []),
                parent_task_id=orchestration_task.id
            )
            reasoning_result = await self.execute_task(reasoning_task.id)
            
            if not reasoning_result.success:
                return {
                    "success": False,
                    "error": reasoning_result.error,
                    "phase": "reasoning",
                    "iteration": iteration
                }
            
            # Phase 4: Evaluation
            evaluation_task = await self.submit_task(
                role=AgentRole.EVALUATION,
                description="Evaluate execution results",
                input_data={
                    "comprehension": comprehension_result.output,
                    "orchestration": orchestration_result.output,
                    "reasoning": reasoning_result.output
                },
                parent_task_id=reasoning_task.id
            )
            evaluation_result = await self.execute_task(evaluation_task.id)
            
            if not evaluation_result.success:
                return {
                    "success": False,
                    "error": evaluation_result.error,
                    "phase": "evaluation",
                    "iteration": iteration
                }
            
            # Check evaluation outcome
            next_action = evaluation_result.output.get("next_action", "conversation")
            
            if next_action == "conversation":
                # Exit loop - proceed to conversation
                break
            elif next_action == "orchestration":
                # Revision needed - loop back
                orchestration_input = {
                    "comprehension": comprehension_result.output,
                    "previous_evaluation": evaluation_result.output,
                    "reasoning_results": reasoning_result.output
                }
                continue
            else:
                # Unknown action - force completion
                logger.warning(f"Unknown evaluation action: {next_action}")
                break
        
        # Phase 5: Conversation (final response)
        conversation_task = await self.submit_task(
            role=AgentRole.CONVERSATION,
            description="Generate final response",
            input_data={
                "comprehension": comprehension_result.output,
                "reasoning": reasoning_result.output,
                "evaluation": evaluation_result.output,
                "user_input": user_input
            },
            parent_task_id=evaluation_task.id
        )
        conversation_result = await self.execute_task(conversation_task.id)
        
        return {
            "success": conversation_result.success,
            "response": conversation_result.output.get("response"),
            "phase": "conversation",
            "path": "complex_task",
            "iterations": iteration,
            "tools_used": reasoning_result.tools_used,
            "metadata": {
                "comprehension_confidence": comprehension_result.output.get("confidence"),
                "evaluation_confidence": evaluation_result.output.get("confidence")
            }
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get status information about the orchestrator."""
        return {
            "active_tasks": len([t for t in self._tasks.values() if t.status == TaskStatus.RUNNING]),
            "total_tasks": len(self._tasks),
            "container_manager": self.container_manager.get_status(),
            "state_manager": self.state_manager.get_status(),
            "tool_selector": self.tool_selector.get_status()
        }
