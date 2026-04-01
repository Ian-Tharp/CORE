"""
Reasoning Agent - Executes plan steps with tool calls and error handling.

Responsibilities:
- Execute each step in the plan sequentially
- Call appropriate tools with safety checks
- Handle retries based on retry policy
- Produce artifacts and intermediate results

RSI TODO: Implement actual tool execution (file ops, git, database, web)
RSI TODO: Add parallel execution for independent steps
"""

import os
import time
import logging
from typing import List, Optional
from datetime import datetime

import httpx

from app.dependencies import get_openai_client_sync
from app.models.core_state import ExecutionPlan, PlanStep, StepResult

logger = logging.getLogger(__name__)

_CORE_BASE_URL = os.getenv("CORE_BASE_URL", "http://localhost:8001")
_KB_MIN_SIMILARITY = 0.30


class ReasoningAgent:
    """
    Reasoning Agent - Third stage of CORE pipeline.

    Executes the plan created by orchestration, calling tools and managing retries.
    KB context is fetched per step so each LLM call is grounded in relevant documents.
    """

    def __init__(self, model: str = "gpt-oss:20b"):
        self.model = model
        # RSI TODO: Initialize tool registry here
        self.tools = {}

    # ------------------------------------------------------------------
    # Knowledge base retrieval
    # ------------------------------------------------------------------

    def fetch_step_context(self, step: PlanStep, top_k: int = 4) -> Optional[str]:
        """
        Query the CORE knowledge base for documents relevant to this plan step.

        Uses the step name + description as the search query so context is
        specific to what the step needs to do, not just the original user input.

        Same httpx-sync pattern as ComprehensionAgent — safe to call from a
        thread (asyncio.to_thread) without sharing the main event loop.

        Returns a formatted context block or None if nothing relevant is found.
        """
        query = f"{step.name}: {step.description}"

        api_key = os.getenv("CORE_API_KEY")
        if not api_key:
            try:
                from app.auth import VALID_API_KEYS
                api_key = next(iter(VALID_API_KEYS), None)
            except Exception:
                pass
        if not api_key:
            logger.debug("fetch_step_context: no API key available, skipping KB lookup")
            return None

        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.post(
                    f"{_CORE_BASE_URL}/knowledgebase/semantic-search",
                    json={"query": query, "limit": top_k},
                    headers={"X-API-Key": api_key},
                )
            if resp.status_code != 200:
                logger.debug("KB search returned HTTP %s for step '%s'", resp.status_code, step.name)
                return None

            results: list = resp.json()
            if not results:
                return None

            lines = []
            for r in results[:top_k]:
                similarity = float(r.get("similarity", 0.0))
                if similarity < _KB_MIN_SIMILARITY:
                    continue
                title = r.get("title") or r.get("originalName") or r.get("filename") or "Document"
                description = (r.get("description") or "").strip()
                entry = f"- [{title}] (relevance: {similarity:.2f})"
                if description:
                    entry += f": {description}"
                lines.append(entry)

            if not lines:
                return None

            return "Relevant knowledge base context:\n" + "\n".join(lines)

        except Exception as exc:
            logger.debug("KB lookup failed for step '%s' (non-critical): %s", step.name, exc)
            return None

    # ------------------------------------------------------------------
    # Plan execution
    # ------------------------------------------------------------------

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

        if start_from_step:
            steps_to_execute = [s for s in plan.steps if s.id == start_from_step]
        else:
            steps_to_execute = plan.steps

        for step in steps_to_execute:
            result = self._execute_step(step, enable_tools)
            results.append(result)

            if result.status == "success":
                step.status = "completed"
            elif result.status == "failure":
                step.status = "failed"
            else:
                step.status = "completed"

        return results

    def _execute_step(self, step: PlanStep, enable_tools: bool) -> StepResult:
        """Execute a single plan step."""
        start_time = time.time()

        try:
            step.status = "running"
            step.started_at = datetime.utcnow()

            if not enable_tools:
                outputs = {"result": f"Simulated execution of: {step.description}"}
                artifacts = []
                logs = [f"Dry-run: Would execute {step.name}"]
            elif step.tool and step.tool in ["file_operations", "git", "database", "web_research"]:
                # RSI TODO: Implement actual tool execution
                outputs = self._simulate_tool_call(step.tool, step.params)
                artifacts = self._simulate_artifacts(step.tool)
                logs = [
                    f"Executing {step.name}",
                    f"Tool: {step.tool}",
                    f"Parameters: {step.params}",
                    "Execution completed successfully (simulated)"
                ]
            else:
                logger.info(f"Reasoning executing step via LLM: {step.name}")
                outputs = self._execute_with_llm(step)
                artifacts = []
                logs = [
                    f"Executing {step.name} with LLM",
                    f"Model: {self.model}",
                    "LLM generation completed"
                ]

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

    def _execute_with_llm(self, step: PlanStep) -> dict:
        """
        Execute a step using the LLM, augmented with step-specific KB context.

        Queries the knowledge base using the step name + description as the
        search query, then injects relevant documents into the prompt so the
        LLM has grounded context when generating the step output.
        """
        try:
            client = get_openai_client_sync()

            # RAG: fetch KB context specific to this step's task
            try:
                kb_context = self.fetch_step_context(step)
            except Exception as exc:
                logger.debug("fetch_step_context raised unexpectedly (non-critical): %s", exc)
                kb_context = None

            base_prompt = f"""You are executing a step in a larger plan.

Step: {step.name}
Description: {step.description}

Please complete this step and provide the result. Be concise and direct."""

            if kb_context:
                prompt = base_prompt + f"\n\n{kb_context}"
                logger.debug(
                    "Injected RAG context (%d chars) into reasoning prompt for step '%s'",
                    len(kb_context), step.name
                )
            else:
                prompt = base_prompt

            logger.info(f"LLM prompt for step '{step.name}': {prompt}")

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are the Reasoning layer of CORE. Execute the given step and provide the requested output.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )

            content = response.choices[0].message.content
            logger.info(f"LLM response for step '{step.name}': {content}")

            if not content:
                return {"result": f"Empty LLM response for: {step.description}"}

            return {"result": content}

        except Exception as e:
            logger.error(f"LLM execution failed for step '{step.name}': {e}", exc_info=True)
            return {"result": f"Error executing with LLM: {str(e)}"}

    # ------------------------------------------------------------------
    # Tool simulation stubs
    # ------------------------------------------------------------------

    def _simulate_tool_call(self, tool_name: str, params: dict) -> dict:
        """
        Simulate tool execution.

        RSI TODO: Replace with actual tool registry lookup and execution
        """
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
