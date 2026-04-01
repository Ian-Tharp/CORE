"""
Orchestration Agent - Creates execution plans for accomplishing tasks.

Produces:
- List of steps to accomplish the goal
- Tool selections and parameters for each step
- Dependency graph between steps
- Retry policies and HITL checkpoints
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from app.dependencies import get_openai_client_sync
from app.models.core_state import UserIntent, ExecutionPlan, PlanStep
from app.utils.json_repair import safe_json_loads, extract_json_object

logger = logging.getLogger(__name__)

# Token estimates per step type (rough heuristics for cost prediction)
_TOKENS_PER_LLM_STEP = 400
_TOKENS_PER_TOOL_STEP = 100
_HIGH_RISK_TOOLS = {"git", "database"}  # tools that mutate external state


class OrchestrationAgent:
    """
    Orchestration Agent - Second stage of CORE pipeline.

    Creates execution plans by decomposing tasks into actionable steps.
    """

    def __init__(self, model: str = "gpt-oss:20b"):
        # Default to local model for offline-first operation
        self.model = model
        self.system_prompt = self._build_system_prompt()

    # ------------------------------------------------------------------
    # Tool registry integration
    # ------------------------------------------------------------------

    @staticmethod
    def _get_available_tools() -> List[str]:
        """
        Return the list of available tool names from the ToolDispatcher.

        Falls back to a hardcoded list if the dispatcher cannot be imported
        (e.g. during unit tests that stub out the tools package).
        """
        try:
            from app.core.tools.dispatcher import ToolDispatcher
            return ToolDispatcher().available_tools
        except Exception:
            return ["file_operations", "git", "web_research", "database"]

    @staticmethod
    def _tool_descriptions(tools: List[str]) -> str:
        """Return a human-readable bullet list of tool capabilities."""
        _descriptions: Dict[str, str] = {
            "file_operations": "read / write / list / search files within the workspace",
            "git": "read-only git operations: status, diff, log, show, branch, ls-files",
            "web_research": "HTTP GET for external pages, documentation, and references",
            "database": "query CORE data via the REST API (conversations, KB, metrics)",
        }
        lines = []
        for tool in tools:
            desc = _descriptions.get(tool, "custom tool")
            lines.append(f"- **{tool}**: {desc}")
        return "\n".join(lines) if lines else "- (no tools registered)"

    def _build_system_prompt(self) -> str:
        """Build the system prompt using dynamically discovered tools."""
        tools = self._get_available_tools()
        tool_section = self._tool_descriptions(tools)
        return f"""You are the Orchestration layer of the CORE cognitive system.

Your job is to create an execution plan for accomplishing the user's task.

Break down the task into concrete, executable steps. For each step:
1. Name and describe what it does
2. Identify which tool is needed (if any)
3. Specify parameters for the tool
4. List dependencies (which steps must complete first)
5. Determine if human review is needed (HITL checkpoint)

Available Tools:
{tool_section}

Respond in JSON format:
{{
  "goal": "High-level description of what we're trying to accomplish",
  "reasoning": "Why this plan was chosen",
  "steps": [
    {{
      "name": "Step name",
      "description": "What this step does",
      "tool": "tool_name or null",
      "params": {{"param1": "value1"}},
      "dependencies": ["step_1_id"],
      "requires_hitl": false
    }}
  ]
}}

Guidelines:
- Keep steps atomic and focused
- Order steps logically (dependencies first)
- Use HITL sparingly (only for risky operations)
- If the task is simple, use fewer steps (1-3)
- If complex, break down thoroughly (5-10 steps)"""

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def estimate_plan_cost(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """
        Estimate the cost of executing a plan.

        Returns a dict with:
          step_count        — total steps
          tool_step_count   — steps that use a registered tool
          llm_step_count    — steps relying on LLM (no tool or unknown tool)
          estimated_tokens  — rough token budget for execution
          hitl_checkpoints  — steps requiring human approval
          risk_level        — "low" | "medium" | "high" based on tool types
          tool_breakdown    — {tool_name: count}
        """
        available_tools = self._get_available_tools()
        tool_counts: Dict[str, int] = {}
        llm_steps = 0
        hitl_steps = 0

        for step in plan.steps:
            if step.requires_hitl:
                hitl_steps += 1
            if step.tool and step.tool in available_tools:
                tool_counts[step.tool] = tool_counts.get(step.tool, 0) + 1
            else:
                llm_steps += 1

        tool_step_count = sum(tool_counts.values())
        estimated_tokens = (
            llm_steps * _TOKENS_PER_LLM_STEP
            + tool_step_count * _TOKENS_PER_TOOL_STEP
        )

        # Risk: high if any high-risk tools used + HITL required, medium if one condition, low otherwise
        uses_high_risk = bool(_HIGH_RISK_TOOLS.intersection(tool_counts))
        if uses_high_risk and hitl_steps > 0:
            risk_level = "high"
        elif uses_high_risk or hitl_steps > 0:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "step_count": len(plan.steps),
            "tool_step_count": tool_step_count,
            "llm_step_count": llm_steps,
            "estimated_tokens": estimated_tokens,
            "hitl_checkpoints": hitl_steps,
            "risk_level": risk_level,
            "tool_breakdown": tool_counts,
        }

    def create_plan(
        self,
        user_input: str,
        intent: Optional[UserIntent] = None,
        previous_plan: Optional[ExecutionPlan] = None,
        evaluation_feedback: Optional[str] = None,
        revision: int = 1
    ) -> ExecutionPlan:
        """
        Create or revise an execution plan.

        Args:
            user_input: The original user input
            intent: The comprehension result
            previous_plan: Previous plan if this is a revision
            evaluation_feedback: Feedback from evaluation if revising
            revision: Revision number

        Returns:
            ExecutionPlan with steps and metadata
        """
        client = get_openai_client_sync()

        # Build context for the LLM
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add the task
        user_message = f"Task: {user_input}"
        if intent:
            user_message += f"\n\nIntent: {intent.description}"
            user_message += f"\nTools needed: {', '.join(intent.tools_needed) if intent.tools_needed else 'None'}"

        # If revising, include previous plan and feedback
        if previous_plan and evaluation_feedback:
            user_message += f"\n\nPrevious plan (revision {previous_plan.revision}):"
            for step in previous_plan.steps:
                user_message += f"\n- {step.name}: {step.status}"
            user_message += f"\n\nEvaluation feedback: {evaluation_feedback}"
            user_message += "\n\nPlease revise the plan based on this feedback."

        messages.append({"role": "user", "content": user_message})

        try:
            logger.info(f"Orchestration creating plan for: '{user_input}' with model={self.model}")
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.4,  # Some creativity, but mostly deterministic
                # Note: Ollama may not support response_format, removed for compatibility
            )

            # Parse response
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")

            logger.info(f"Orchestration LLM response: {content}")
            
            # Extract and repair JSON from response (handles code fences, trailing commas, etc.)
            extracted = extract_json_object(content)
            if extracted:
                logger.info(f"Orchestration: Extracted JSON object: {extracted[:200]}...")
            else:
                extracted = content
            
            data = safe_json_loads(extracted)
            if data is None:
                raise ValueError(f"Could not parse JSON from response: {content[:200]}...")

            # Build ExecutionPlan
            steps = []
            for i, step_data in enumerate(data.get("steps", [])):
                step = PlanStep(
                    name=step_data.get("name", f"Step {i+1}"),
                    description=step_data.get("description", ""),
                    tool=step_data.get("tool"),
                    params=step_data.get("params", {}),
                    dependencies=step_data.get("dependencies", []),
                    requires_hitl=step_data.get("requires_hitl", False),
                )
                steps.append(step)

            return ExecutionPlan(
                goal=data.get("goal", user_input),
                steps=steps,
                reasoning=data.get("reasoning", ""),
                revision=revision
            )

        except Exception as e:
            # Fallback: create a simple single-step plan
            return ExecutionPlan(
                goal=user_input,
                steps=[
                    PlanStep(
                        name="Execute task",
                        description=f"Error in orchestration: {str(e)}. Attempting direct execution.",
                        tool=None,
                        params={},
                        requires_hitl=False
                    )
                ],
                reasoning="Fallback plan due to orchestration error",
                revision=revision
            )
