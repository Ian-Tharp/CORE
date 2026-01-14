"""
Reasoning Agent - Executes plan steps with tool calls and error handling.

Responsibilities:
- Execute each step in the plan sequentially
- Call appropriate tools with safety checks
- Handle retries based on retry policy
- Produce artifacts and intermediate results

RSI TODO: Implement actual tool execution (file ops, git, database, web)
RSI TODO: Add parallel execution for independent steps
RSI TODO: Integrate with Knowledge Base for context retrieval during execution
"""

import time
from typing import List, Optional
from datetime import datetime

from app.models.core_state import ExecutionPlan, PlanStep, StepResult


class ReasoningAgent:
    """
    Reasoning Agent - Third stage of CORE pipeline.

    Executes the plan created by orchestration, calling tools and managing retries.
    """

    def __init__(self, model: str = "gpt-oss:20b"):
        self.model = model
        # RSI TODO: Initialize tool registry here
        self.tools = {}

    def execute_plan(
        self,
        plan: ExecutionPlan,
        start_from_step: Optional[str] = None,
        enable_tools: bool = True
    ) -> List[StepResult]:
        """
        Execute the plan step by step.

        Args:
            plan: The execution plan to run
            start_from_step: Optional step ID to resume from (for retries)
            enable_tools: Whether to actually call tools (False for dry-run)

        Returns:
            List of StepResult objects for each executed step
        """
        results = []

        # Determine which steps to execute
        if start_from_step:
            # Find the step and execute only that one
            steps_to_execute = [s for s in plan.steps if s.id == start_from_step]
        else:
            # Execute all steps in order
            steps_to_execute = plan.steps

        for step in steps_to_execute:
            result = self._execute_step(step, enable_tools)
            results.append(result)

            # Update step status in the plan
            if result.status == "success":
                step.status = "completed"
            elif result.status == "failure":
                step.status = "failed"
            else:
                step.status = "completed"  # Partial success

        return results

    def _execute_step(self, step: PlanStep, enable_tools: bool) -> StepResult:
        """
        Execute a single plan step.

        For now, this is a stub that simulates execution.
        In the future, this will call actual tools.
        """
        start_time = time.time()

        try:
            # Mark step as running
            step.status = "running"
            step.started_at = datetime.utcnow()

            # RSI TODO: Replace simulation with actual tool execution
            if enable_tools and step.tool:
                # Simulate tool execution
                outputs = self._simulate_tool_call(step.tool, step.params)
                artifacts = self._simulate_artifacts(step.tool)
                logs = [
                    f"Executing {step.name}",
                    f"Tool: {step.tool}",
                    f"Parameters: {step.params}",
                    "Execution completed successfully"
                ]
            else:
                # No tool needed or dry-run mode
                outputs = {"result": f"Simulated execution of: {step.description}"}
                artifacts = []
                logs = [f"Dry-run: Would execute {step.name}"]

            duration = time.time() - start_time

            return StepResult(
                step_id=step.id,
                status="success",
                outputs=outputs,
                artifacts=artifacts,
                logs=logs,
                duration_seconds=duration
            )

        except Exception as e:
            duration = time.time() - start_time

            return StepResult(
                step_id=step.id,
                status="failure",
                outputs={},
                artifacts=[],
                logs=[f"Error executing {step.name}: {str(e)}"],
                error=str(e),
                duration_seconds=duration
            )

    def _simulate_tool_call(self, tool_name: str, params: dict) -> dict:
        """
        Simulate tool execution.

        RSI TODO: Replace with actual tool registry lookup and execution
        """
        # Simulate different tool behaviors
        if tool_name == "file_operations":
            return {
                "files_modified": ["src/components/Header.tsx"],
                "changes_made": "Added login button component"
            }
        elif tool_name == "git":
            return {
                "branch": "feature/add-login-button",
                "commit_sha": "abc123def456",
                "files_changed": 1
            }
        elif tool_name == "database":
            return {
                "query_result": [{"id": 1, "name": "example"}],
                "rows_affected": 1
            }
        else:
            return {"result": f"Executed {tool_name} with params {params}"}

    def _simulate_artifacts(self, tool_name: str) -> List[str]:
        """
        Simulate artifact generation.

        RSI TODO: Return actual file paths, git diffs, etc.
        """
        if tool_name == "file_operations":
            return ["src/components/LoginButton.tsx"]
        elif tool_name == "git":
            return ["git.diff"]
        else:
            return []
