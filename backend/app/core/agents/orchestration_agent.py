"""
Orchestration Agent - Creates execution plans for accomplishing tasks.

Produces:
- List of steps to accomplish the goal
- Tool selections and parameters for each step
- Dependency graph between steps
- Retry policies and HITL checkpoints

RSI TODO: Implement dynamic plan revision based on execution feedback
RSI TODO: Add cost estimation for plan execution
RSI TODO: Integrate with tool registry for capability discovery
"""

import json
import logging
from typing import Optional
from datetime import datetime

from app.dependencies import get_openai_client_sync
from app.models.core_state import UserIntent, ExecutionPlan, PlanStep

logger = logging.getLogger(__name__)


class OrchestrationAgent:
    """
    Orchestration Agent - Second stage of CORE pipeline.

    Creates execution plans by decomposing tasks into actionable steps.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt for plan generation."""
        return """You are the Orchestration layer of the CORE cognitive system.

Your job is to create an execution plan for accomplishing the user's task.

Break down the task into concrete, executable steps. For each step:
1. Name and describe what it does
2. Identify which tool is needed (if any)
3. Specify parameters for the tool
4. List dependencies (which steps must complete first)
5. Determine if human review is needed (HITL checkpoint)

Available Tools:
- **file_operations**: read_file, write_file, edit_file, search_files
- **git**: create_branch, commit_changes, create_pr, diff
- **database**: query, get_schema, explain_query
- **web_research**: search, fetch_page, search_docs

Respond in JSON format:
{
  "goal": "High-level description of what we're trying to accomplish",
  "reasoning": "Why this plan was chosen",
  "steps": [
    {
      "name": "Step name",
      "description": "What this step does",
      "tool": "tool_name or null",
      "params": {"param1": "value1"},
      "dependencies": ["step_1_id"],
      "requires_hitl": false
    }
  ]
}

Guidelines:
- Keep steps atomic and focused
- Order steps logically (dependencies first)
- Use HITL sparingly (only for risky operations)
- If the task is simple, use fewer steps (1-3)
- If complex, break down thoroughly (5-10 steps)"""

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
            data = json.loads(content)

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
