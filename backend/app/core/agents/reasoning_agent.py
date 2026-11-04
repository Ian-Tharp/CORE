"""
Reasoning Agent - Executes plan steps with tool calls and error handling.

Responsibilities:
- Execute each step in the plan sequentially or in parallel waves
- Call appropriate tools (file ops, git, database, web) via ToolDispatcher
- Handle retries based on retry policy
- Produce artifacts and intermediate results
"""

import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set
from datetime import datetime

import httpx

from app.dependencies import get_openai_client_sync
from app.models.core_state import (
    ExecutionPlan,
    PlanStep,
    RefusalRecord,
    StepResult,
)
from app.core.tools.dispatcher import ToolDispatcher
from app.core.agents.refusal_guard import EXCERPT_CHARS, RefusalLatch, detect_refusal
from app.core.agents.action_guard import ActionGuard

# StepResult.error prefixes for steps that did not run (read by retry resume
# and by fault capture). Kept stable: "Dependency failed" predates the latch.
_DEP_FAILED = "Dependency failed"
_AWAITING = "Awaiting confirmation"

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
                logger.debug(
                    "KB search returned HTTP %s for step '%s'",
                    resp.status_code,
                    step.name,
                )
                return None

            results: list = resp.json()
            if not results:
                return None

            lines = []
            for r in results[:top_k]:
                similarity = float(r.get("similarity", 0.0))
                if similarity < _KB_MIN_SIMILARITY:
                    continue
                title = (
                    r.get("title")
                    or r.get("originalName")
                    or r.get("filename")
                    or "Document"
                )
                description = (r.get("description") or "").strip()
                entry = f"- [{title}] (relevance: {similarity:.2f})"
                if description:
                    entry += f": {description}"
                lines.append(entry)

            if not lines:
                return None

            return "Relevant knowledge base context:\n" + "\n".join(lines)

        except Exception as exc:
            logger.debug(
                "KB lookup failed for step '%s' (non-critical): %s", step.name, exc
            )
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
        refusal_latch: Optional[RefusalLatch] = None,
        action_guard: Optional[ActionGuard] = None,
        prior_results: Optional[List[StepResult]] = None,
    ) -> List[StepResult]:
        """
        Execute the plan respecting dependency ordering.

        Steps with no mutual dependencies are executed in parallel within
        each "wave" (topological level). A step whose dependency did not
        succeed (or never ran, in a cycle) is not run: the dependency latch.

        Args:
            plan: The execution plan to run
            start_from_step: Optional step ID to resume from (for retries)
            enable_tools: Whether to actually call tools (False for dry-run)
            refusal_latch: Per-run abstention guard. Once it is tripped (by the
                gate or by a step that declines), remaining steps are blocked
                instead of executed. None disables the guard.
            action_guard: Per-run confirm tier and dependency-block record
                (see action_guard.py). None means no confirm tier; dependency
                blocks are still enforced, just not reported.
            prior_results: Results from earlier passes of this run. Used on a
                retry so the retried step's dependencies are checked, and so
                its dependency-blocked dependents can resume.

        Returns:
            List of StepResult objects in plan order (skipped steps included).
            On a retry: the retried step plus any dependents it resumed.
        """
        guard = action_guard if action_guard is not None else ActionGuard(confirm=False)
        plan_ids = {s.id for s in plan.steps}

        if start_from_step:
            if start_from_step not in plan_ids:
                return []
            prior = {r.step_id: r for r in (prior_results or [])}
            rerun = {start_from_step} | self._resumable_dependents(
                plan.steps, start_from_step, prior
            )
            subset = [s for s in plan.steps if s.id in rerun]
            for step in subset:
                step.status = "pending"
            known = {k: v for k, v in prior.items() if k not in rerun}
            return self._execute_with_parallelism(
                subset, enable_tools, refusal_latch, guard, known, plan_ids
            )

        return self._execute_with_parallelism(
            plan.steps, enable_tools, refusal_latch, guard, {}, plan_ids
        )

    @staticmethod
    def _resumable_dependents(
        steps: List[PlanStep], root_id: str, prior: Dict[str, StepResult]
    ) -> Set[str]:
        """Transitive dependents of ``root_id`` whose last result was a
        dependency block. A retry re-runs them (they are checked again)."""
        found: Set[str] = set()
        frontier = {root_id}
        while frontier:
            nxt = set()
            for s in steps:
                if s.id in found or s.id == root_id:
                    continue
                if not frontier.intersection(s.dependencies):
                    continue
                r = prior.get(s.id)
                if r is not None and (r.error or "").startswith(_DEP_FAILED):
                    found.add(s.id)
                    nxt.add(s.id)
            frontier = nxt
        return found

    def _build_execution_waves(self, steps: List) -> List[List]:
        """
        Topological sort: return ordered list of waves.

        Each wave is a list of steps whose declared dependencies have all
        appeared in earlier waves.  Dependency IDs that are not in the plan are
        ignored. Steps caught in a cycle come back as one final wave.
        """
        step_by_id: Dict[str, object] = {s.id: s for s in steps}
        completed_ids: Set[str] = set()
        remaining = list(steps)
        waves: List[List] = []

        while remaining:
            # Pick all steps whose deps are satisfied
            ready = [
                s
                for s in remaining
                if all(
                    dep not in step_by_id or dep in completed_ids
                    for dep in s.dependencies
                )
            ]
            if not ready:
                # Dependency cycle: hand the rest over as one last wave. The
                # dependency latch then blocks them (no step in a cycle can
                # have its prerequisite done first).
                logger.warning(
                    "Could not resolve dependencies for %d step(s) (cycle); "
                    "they will not run",
                    len(remaining),
                )
                waves.append(remaining)
                break

            waves.append(ready)
            completed_ids.update(s.id for s in ready)
            remaining = [s for s in remaining if s not in ready]

        return waves

    def _execute_with_parallelism(
        self,
        steps: List,
        enable_tools: bool,
        refusal_latch: Optional[RefusalLatch] = None,
        guard: Optional[ActionGuard] = None,
        known_results: Optional[Dict[str, StepResult]] = None,
        plan_ids: Optional[Set[str]] = None,
    ) -> List[StepResult]:
        """Execute steps wave by wave; run each wave in parallel.

        ``known_results`` are results from outside ``steps`` (earlier passes)
        that dependencies are checked against; ``plan_ids`` are every step id
        in the plan (a dependency outside the plan is ignored, as before).
        """
        guard = guard if guard is not None else ActionGuard(confirm=False)
        waves = self._build_execution_waves(steps)
        results_by_id: Dict[str, StepResult] = dict(known_results or {})
        plan_ids = plan_ids if plan_ids is not None else {s.id for s in steps}

        for wave in waves:
            runnable = []
            for step in wave:
                # After a refusal, every remaining step goes to the guard so it
                # is recorded as blocked (not merely dependency-skipped).
                if refusal_latch is not None and refusal_latch.active:
                    runnable.append(step)
                    continue
                blocked = self._dependency_block(step, results_by_id, plan_ids)
                if blocked is None:
                    runnable.append(step)
                    continue
                cause, detail = blocked
                record = guard.block_dependent(step, cause, detail)
                logger.info("Not running step '%s': %s", step.name, record.reason)
                step.status = "skipped"
                awaiting = record.kind == "awaiting_confirmation"
                results_by_id[step.id] = StepResult(
                    step_id=step.id,
                    status="failure",
                    outputs={},
                    logs=[f"Skipped: {record.reason}"],
                    error=(
                        f"{_AWAITING}: waits on step {cause}"
                        if awaiting
                        else f"{_DEP_FAILED}: {cause} {detail}"
                    ),
                )

            if not runnable:
                continue

            if len(runnable) == 1:
                # No parallelism overhead for single step
                result = self._run_step(runnable[0], enable_tools, refusal_latch, guard)
                self._apply_step_status(runnable[0], result)
                results_by_id[runnable[0].id] = result
            else:
                # Execute wave in parallel
                logger.info(
                    "Executing wave of %d independent steps in parallel", len(runnable)
                )
                wave_results: Dict[str, StepResult] = {}
                with ThreadPoolExecutor(
                    max_workers=min(len(runnable), self._MAX_PARALLEL_WORKERS)
                ) as executor:
                    future_to_step = {
                        executor.submit(
                            self._run_step, step, enable_tools, refusal_latch, guard
                        ): step
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
    def _dependency_block(
        step: PlanStep, results: Dict[str, StepResult], plan_ids: Set[str]
    ):
        """Return ``(cause_step_id, detail)`` if ``step`` must not run, else None.

        A dependency in the plan must have a successful result. No result yet
        means it never ran: only possible in a dependency cycle, which the
        wave builder hands over as one wave.
        """
        for dep in step.dependencies:
            if dep not in plan_ids:
                continue  # unknown id: planner noise, treated as satisfied
            r = results.get(dep)
            if r is None:
                return dep, "never ran (dependency cycle)"
            if r.status != "success":
                return dep, "did not succeed"
        return None

    def _run_step(
        self,
        step: PlanStep,
        enable_tools: bool,
        refusal_latch: Optional[RefusalLatch] = None,
        guard: Optional[ActionGuard] = None,
    ) -> StepResult:
        """Run one step behind the abstention guard and the confirm tier.

        Blocked and held steps are never handed to ``_execute_step``. After an
        LLM-executed step, its output is checked for a refusal; a refusal fails
        the step (so dependents skip) and trips the latch for everything after.
        """
        if refusal_latch is not None:
            blocked = refusal_latch.check(step)
            if blocked is not None:
                step.status = "skipped"
                logger.info("Blocked step '%s': %s", step.name, blocked.reason)
                return StepResult(
                    step_id=step.id,
                    status="failure",
                    outputs={},
                    logs=[f"Blocked: {blocked.reason}"],
                    error=f"Blocked: {blocked.reason}",
                )

        # Confirm tier: only for steps the dispatcher would actually run.
        if (
            guard is not None
            and enable_tools
            and step.tool
            and step.tool in self.tool_dispatcher.available_tools
        ):
            held = guard.check_confirmation(step)
            if held is not None:
                step.status = "skipped"
                why = (
                    f"{held.capability_id} is a write"
                    if held.reason == "write"
                    else f"plan asked for a checkpoint on {held.capability_id}"
                )
                logger.info("Holding step '%s' for confirmation: %s", step.name, why)
                return StepResult(
                    step_id=step.id,
                    status="failure",
                    outputs={},
                    logs=[f"Held for confirmation: {why}"],
                    error=f"{_AWAITING}: {held.capability_id}",
                )

        result = self._execute_step(step, enable_tools)

        if (
            refusal_latch is not None
            and enable_tools
            and result.status == "success"
            and result.dispatched_tool is None
        ):
            text = str((result.outputs or {}).get("result", ""))
            phrase = detect_refusal(text)
            if phrase:
                refusal_latch.trip(
                    RefusalRecord(
                        source="reasoning",
                        reason=f"step declined: {phrase!r}",
                        step_id=step.id,
                        excerpt=text[:EXCERPT_CHARS],
                    )
                )
                logger.warning("Step '%s' refused (%r); latching", step.name, phrase)
                result.status = "failure"
                result.error = f"Refused: {phrase}"
                result.logs = list(result.logs) + ["Refusal detected; run latched"]
        return result

    @staticmethod
    def _apply_step_status(step, result: StepResult) -> None:
        """Synchronise step.status with the StepResult outcome."""
        if step.status == "skipped":
            return  # blocked by the abstention guard; keep it visible as skipped
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

            dispatched = None
            if not enable_tools:
                outputs = {"result": f"Simulated execution of: {step.description}"}
                artifacts = []
                logs = [f"Dry-run: Would execute {step.name}"]
            elif step.tool and step.tool in self.tool_dispatcher.available_tools:
                dispatched = step.tool
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
                    "LLM generation completed",
                ]

            duration = time.time() - start_time

            return StepResult(
                step_id=step.id,
                status="success",
                outputs=outputs,
                artifacts=artifacts,
                logs=logs,
                duration_seconds=duration,
                dispatched_tool=dispatched,
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
                duration_seconds=duration,
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
                logger.debug(
                    "fetch_step_context raised unexpectedly (non-critical): %s", exc
                )
                kb_context = None

            base_prompt = f"""You are executing a step in a larger plan.

Step: {step.name}
Description: {step.description}

Please complete this step and provide the result. Be concise and direct."""

            if kb_context:
                prompt = base_prompt + f"\n\n{kb_context}"
                logger.debug(
                    "Injected RAG context (%d chars) into reasoning prompt for step '%s'",
                    len(kb_context),
                    step.name,
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
                raise RuntimeError(f"Empty LLM response for: {step.description}")

            return {"result": content}

        except Exception as e:
            # Fail the step: a swallowed error read as success, so dependents
            # (including writes) ran on an answer that never came.
            logger.error(
                f"LLM execution failed for step '{step.name}': {e}", exc_info=True
            )
            raise RuntimeError(f"LLM execution failed: {e}") from e

    # Tool dispatch is now handled by self.tool_dispatcher (ToolDispatcher).
    # See app/core/tools/dispatcher.py for per-tool handler implementations.
