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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set
from datetime import datetime

import httpx

from app.dependencies import get_openai_client_sync
from app.models.core_state import ExecutionPlan, PlanStep, StepResult
from app.core.tools.dispatcher import ToolDispatcher

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
        workspace = os.getenv("CORE_WORKSPACE_DIR", os.getcwd())
        api_key = os.getenv("CORE_API_KEY", "")
        self.tool_dispatcher = ToolDispatcher(
            workspace_dir=workspace,
            core_base_url=_CORE_BASE_URL,
            core_api_key=api_key,
        )

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

    # Maximum worker threads for parallel step execution within a wave.
    _MAX_PARALLEL_WORKERS = int(os.getenv("CORE_MAX_PARALLEL_STEPS", "4"))

    def execute_plan(
        self,
        plan: ExecutionPlan,
        start_from_step: Optional[str] = None,
        enable_tools: bool = True,
    ) -> List[StepResult]:
        """
        Execute the plan respecting dependency ordering.

        Steps with no mutual dependencies are executed in parallel within
        each "wave" (topological level).  Steps whose dependency steps failed
        are skipped automatically.

        Args:
            plan: The execution plan to run
            start_from_step: Optional step ID to resume from (for retries)
            enable_tools: Whether to actually call tools (False for dry-run)

        Returns:
            List of StepResult objects in plan order (skipped steps included).
        """
        if start_from_step:
            # Retry mode: single-step, no parallelism needed
            steps_to_execute = [s for s in plan.steps if s.id == start_from_step]
            results = []
            for step in steps_to_execute:
                result = self._execute_step(step, enable_tools)
                self._apply_step_status(step, result)
                results.append(result)
            return results

        return self._execute_with_parallelism(plan.steps, enable_tools)

    def _build_execution_waves(self, steps: List) -> List[List]:
        """
        Topological sort: return ordered list of waves.

        Each wave is a list of steps whose declared dependencies have all
        appeared in earlier waves.  Steps with missing dependency IDs are
        treated as having no dependencies (fail-safe).
        """
        step_by_id: Dict[str, object] = {s.id: s for s in steps}
        completed_ids: Set[str] = set()
        remaining = list(steps)
        waves: List[List] = []

        while remaining:
            # Pick all steps whose deps are satisfied
            ready = [
                s for s in remaining
                if all(dep not in step_by_id or dep in completed_ids
                       for dep in s.dependencies)
            ]
            if not ready:
                # Circular dependency or unresolvable — run the rest sequentially
                logger.warning(
                    "Could not resolve dependencies for %d step(s); running sequentially",
                    len(remaining),
                )
                waves.append(remaining)
                break

            waves.append(ready)
            completed_ids.update(s.id for s in ready)
            remaining = [s for s in remaining if s not in ready]

        return waves

    def _execute_with_parallelism(self, steps: List, enable_tools: bool) -> List[StepResult]:
        """Execute steps wave by wave; run each wave in parallel."""
        waves = self._build_execution_waves(steps)
        results_by_id: Dict[str, StepResult] = {}

        for wave in waves:
            # Determine which steps in this wave have failed deps → skip them
            runnable, skipped = [], []
            for step in wave:
                failed_deps = [
                    dep for dep in step.dependencies
                    if dep in results_by_id and results_by_id[dep].status == "failure"
                ]
                if failed_deps:
                    logger.info(
                        "Skipping step '%s' — dependency %s failed", step.name, failed_deps[0]
                    )
                    skipped.append(step)
                else:
                    runnable.append(step)

            # Record skipped steps
            for step in skipped:
                step.status = "skipped"
                results_by_id[step.id] = StepResult(
                    step_id=step.id,
                    status="failure",
                    outputs={},
                    logs=[f"Skipped: dependency failed"],
                    error="Dependency failed",
                )

            if not runnable:
                continue

            if len(runnable) == 1:
                # No parallelism overhead for single step
                result = self._execute_step(runnable[0], enable_tools)
                self._apply_step_status(runnable[0], result)
                results_by_id[runnable[0].id] = result
            else:
                # Execute wave in parallel
                logger.info("Executing wave of %d independent steps in parallel", len(runnable))
                wave_results: Dict[str, StepResult] = {}
                with ThreadPoolExecutor(
                    max_workers=min(len(runnable), self._MAX_PARALLEL_WORKERS)
                ) as executor:
                    future_to_step = {
                        executor.submit(self._execute_step, step, enable_tools): step
                        for step in runnable
                    }
                    for future in as_completed(future_to_step):
                        step = future_to_step[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = StepResult(
                                step_id=step.id,
                                status="failure",
                                outputs={},
                                logs=[f"Parallel execution error: {exc}"],
                                error=str(exc),
                            )
                        self._apply_step_status(step, result)
                        wave_results[step.id] = result

                results_by_id.update(wave_results)

        # Return results in original step order
        return [results_by_id[s.id] for s in steps if s.id in results_by_id]

    @staticmethod
    def _apply_step_status(step, result: StepResult) -> None:
        """Synchronise step.status with the StepResult outcome."""
        if result.status == "success":
            step.status = "completed"
        elif result.status == "failure":
            step.status = "failed"
        else:
            step.status = "completed"

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
            elif step.tool and step.tool in self.tool_dispatcher.available_tools:
                outputs = self.tool_dispatcher.dispatch(step.tool, step.params)
                artifacts = self.tool_dispatcher.get_artifacts(step.tool, outputs)
                tool_status = outputs.get("status", "success")
                logs = [
                    f"Executing {step.name}",
                    f"Tool: {step.tool}",
                    f"Parameters: {step.params}",
                    f"Tool status: {tool_status}",
                ]
                if tool_status == "error":
                    raise RuntimeError(outputs.get("result", "Tool execution failed"))
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

    # Tool dispatch is now handled by self.tool_dispatcher (ToolDispatcher).
    # See app/core/tools/dispatcher.py for per-tool handler implementations.
