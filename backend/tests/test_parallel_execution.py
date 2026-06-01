"""
Tests for parallel step execution in ReasoningAgent.

Sentinel-value contract:
- Remove _build_execution_waves() → steps run flat → test_wave_ordering_respects_deps fails
- Remove parallel ThreadPoolExecutor → sequential only → test_independent_steps_run_in_parallel fails
- Remove failed-dep skip logic → downstream steps run despite failure → test_failed_dep_skips_downstream fails
- Remove start_from_step filtering → wrong steps run → test_start_from_step_runs_single_step fails
"""

from __future__ import annotations

import time
import threading
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from app.core.agents.reasoning_agent import ReasoningAgent
from app.models.core_state import ExecutionPlan, PlanStep, StepResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step(name: str, *, tool: str = None, deps: List[str] = None) -> PlanStep:
    step = PlanStep(name=name, description=f"do {name}", tool=tool)
    if deps:
        step.dependencies = deps
    return step


def _plan(*steps: PlanStep) -> ExecutionPlan:
    return ExecutionPlan(goal="test goal", steps=list(steps))


# ---------------------------------------------------------------------------
# _build_execution_waves — topological sort
# ---------------------------------------------------------------------------


class TestBuildExecutionWaves:
    def _agent(self):
        return ReasoningAgent.__new__(ReasoningAgent)

    def test_no_deps_single_wave(self):
        """
        SENTINEL TEST — steps with no dependencies form a single wave.
        Break wave builder → multiple waves → len == 1 fails.
        """
        agent = self._agent()
        s1 = _step("a")
        s2 = _step("b")
        waves = agent._build_execution_waves([s1, s2])
        assert len(waves) == 1
        assert set(s.name for s in waves[0]) == {"a", "b"}

    def test_chain_produces_sequential_waves(self):
        """
        SENTINEL TEST — a → b → c must produce 3 waves of 1 step each.
        Return one giant wave → waves[0] has 3 steps → test fails.
        """
        agent = self._agent()
        a = _step("a")
        b = _step("b", deps=[a.id])
        c = _step("c", deps=[b.id])
        waves = agent._build_execution_waves([a, b, c])
        assert len(waves) == 3
        assert waves[0][0].name == "a"
        assert waves[1][0].name == "b"
        assert waves[2][0].name == "c"

    def test_diamond_dependency_three_waves(self):
        """
        SENTINEL TEST — diamond (a → b,c → d) produces 3 waves.
        Ignore deps → one wave → len == 1 → test fails.
        """
        agent = self._agent()
        a = _step("a")
        b = _step("b", deps=[a.id])
        c = _step("c", deps=[a.id])
        d = _step("d", deps=[b.id, c.id])
        waves = agent._build_execution_waves([a, b, c, d])
        assert len(waves) == 3
        # Wave 0: just a
        assert len(waves[0]) == 1 and waves[0][0].name == "a"
        # Wave 1: b and c (parallel)
        assert len(waves[1]) == 2
        assert {s.name for s in waves[1]} == {"b", "c"}
        # Wave 2: just d
        assert len(waves[2]) == 1 and waves[2][0].name == "d"

    def test_unknown_dep_treated_as_satisfied(self):
        """Steps whose dep ID is not in the plan run immediately (fail-safe)."""
        agent = self._agent()
        s = _step("x", deps=["nonexistent-id"])
        waves = agent._build_execution_waves([s])
        assert len(waves) == 1

    def test_circular_dep_runs_remaining_sequentially(self):
        """Circular deps must not cause an infinite loop; remaining steps run in a single wave."""
        agent = self._agent()
        a = _step("a")
        b = _step("b")
        # Inject mutual dependency post-construction
        a.dependencies = [b.id]
        b.dependencies = [a.id]
        # Should not hang
        waves = agent._build_execution_waves([a, b])
        total_steps = sum(len(w) for w in waves)
        assert total_steps == 2


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------


class TestParallelExecution:
    def _make_agent(self):
        agent = ReasoningAgent.__new__(ReasoningAgent)
        agent.model = "gpt-oss:20b"
        # Minimal tool_dispatcher stub
        dispatcher = MagicMock()
        dispatcher.available_tools = []
        agent.tool_dispatcher = dispatcher
        return agent

    def test_independent_steps_run_in_parallel(self):
        """
        SENTINEL TEST — 4 independent steps must overlap in time.
        Run sequentially → total time ≥ 4×delay → wall time assertion fails.
        """
        delay = 0.06  # 60 ms each
        run_times = {}
        lock = threading.Lock()

        agent = self._make_agent()
        steps = [_step(f"step-{i}") for i in range(4)]
        plan = _plan(*steps)

        original_execute = agent._execute_step

        def _slow_step(step, enable_tools):
            t0 = time.monotonic()
            time.sleep(delay)
            t1 = time.monotonic()
            with lock:
                run_times[step.name] = (t0, t1)
            return StepResult(step_id=step.id, status="success", outputs={}, logs=[])

        with patch.object(agent, "_execute_step", side_effect=_slow_step):
            t_start = time.monotonic()
            results = agent.execute_plan(plan, enable_tools=False)
            wall_time = time.monotonic() - t_start

        # Serial time would be 4 × delay; parallel should be < 2 × delay
        assert (
            wall_time < delay * 3
        ), f"Wall time {wall_time:.3f}s suggests sequential execution (expected < {delay*3:.3f}s)"
        assert len(results) == 4

    def test_dependent_steps_run_sequentially(self):
        """
        SENTINEL TEST — b depends on a; a must finish before b starts.
        Ignore deps → b starts before a finishes → order assertion fails.
        """
        finish_order = []
        lock = threading.Lock()

        agent = self._make_agent()
        a = _step("a")
        b = _step("b", deps=[a.id])
        plan = _plan(a, b)

        def _ordered_step(step, enable_tools):
            time.sleep(0.02)
            with lock:
                finish_order.append(step.name)
            return StepResult(step_id=step.id, status="success", outputs={}, logs=[])

        with patch.object(agent, "_execute_step", side_effect=_ordered_step):
            results = agent.execute_plan(plan, enable_tools=False)

        assert finish_order == ["a", "b"], f"Expected ['a','b'], got {finish_order}"
        assert len(results) == 2

    def test_failed_dep_skips_downstream(self):
        """
        SENTINEL TEST — if step 'a' fails, step 'b' (dep on a) must be skipped.
        Remove skip logic → b executes despite a failing → executed_steps includes b → fails.
        """
        executed_steps = []
        agent = self._make_agent()
        a = _step("a")
        b = _step("b", deps=[a.id])
        plan = _plan(a, b)

        def _maybe_fail(step, enable_tools):
            executed_steps.append(step.name)
            if step.name == "a":
                return StepResult(
                    step_id=step.id,
                    status="failure",
                    outputs={},
                    error="forced fail",
                    logs=[],
                )
            return StepResult(step_id=step.id, status="success", outputs={}, logs=[])

        with patch.object(agent, "_execute_step", side_effect=_maybe_fail):
            results = agent.execute_plan(plan, enable_tools=False)

        assert "a" in executed_steps
        assert "b" not in executed_steps, "b should have been skipped due to failed dep"
        # b should appear in results as a failure/skip
        b_result = next(r for r in results if r.step_id == b.id)
        assert b_result.status == "failure"

    def test_independent_sibling_runs_despite_other_failure(self):
        """
        A step with no dep on the failed step must still execute in the same wave.
        """
        executed_steps = []
        agent = self._make_agent()
        a = _step("a")
        b = _step("b")  # sibling of a, no shared deps
        c = _step("c", deps=[a.id])  # depends on a (which will fail)
        plan = _plan(a, b, c)

        def _maybe_fail(step, enable_tools):
            executed_steps.append(step.name)
            if step.name == "a":
                return StepResult(
                    step_id=step.id, status="failure", outputs={}, error="fail", logs=[]
                )
            return StepResult(step_id=step.id, status="success", outputs={}, logs=[])

        with patch.object(agent, "_execute_step", side_effect=_maybe_fail):
            results = agent.execute_plan(plan, enable_tools=False)

        assert (
            "b" in executed_steps
        ), "Independent sibling b must execute even if a fails"
        assert "c" not in executed_steps, "c must be skipped because its dep (a) failed"
        assert len(results) == 3

    def test_results_returned_in_plan_order(self):
        """
        SENTINEL TEST — results must match plan step order, not execution order.
        Return results in completion order → indices mismatch → test fails.
        """
        agent = self._make_agent()
        # Steps with varied sleep to ensure non-completion-order finishes
        steps = [_step(f"step-{i}") for i in range(5)]
        plan = _plan(*steps)

        call_count = {"n": 0}

        def _ordered_result(step, enable_tools):
            call_count["n"] += 1
            # Vary sleep so execution order differs from plan order
            idx = int(step.name.split("-")[1])
            time.sleep((4 - idx) * 0.01)
            return StepResult(
                step_id=step.id, status="success", outputs={"idx": idx}, logs=[]
            )

        with patch.object(agent, "_execute_step", side_effect=_ordered_result):
            results = agent.execute_plan(plan, enable_tools=False)

        assert len(results) == 5
        for i, (result, step) in enumerate(zip(results, steps)):
            assert (
                result.step_id == step.id
            ), f"Result at index {i} has step_id {result.step_id}, expected {step.id}"

    def test_start_from_step_runs_single_step(self):
        """
        SENTINEL TEST — start_from_step must execute ONLY that step.
        Remove filtering → all steps run → executed_steps has 3 items → test fails.
        """
        executed_steps = []
        agent = self._make_agent()
        a = _step("a")
        b = _step("b")
        c = _step("c")
        plan = _plan(a, b, c)

        def _tracking_step(step, enable_tools):
            executed_steps.append(step.name)
            return StepResult(step_id=step.id, status="success", outputs={}, logs=[])

        with patch.object(agent, "_execute_step", side_effect=_tracking_step):
            results = agent.execute_plan(plan, start_from_step=b.id, enable_tools=False)

        assert executed_steps == [
            "b"
        ], f"Expected only 'b' to run, got {executed_steps}"
        assert len(results) == 1
        assert results[0].step_id == b.id


# ---------------------------------------------------------------------------
# _execute_with_parallelism edge cases
# ---------------------------------------------------------------------------


class TestExecuteWithParallelismEdgeCases:
    def _make_agent(self):
        agent = ReasoningAgent.__new__(ReasoningAgent)
        agent.model = "gpt-oss:20b"
        dispatcher = MagicMock()
        dispatcher.available_tools = []
        agent.tool_dispatcher = dispatcher
        return agent

    def test_empty_plan_returns_empty_list(self):
        agent = self._make_agent()
        plan = _plan()
        results = agent.execute_plan(plan, enable_tools=False)
        assert results == []

    def test_single_step_no_parallelism_overhead(self):
        """Single step must still complete and return a result."""
        agent = self._make_agent()
        s = _step("solo")
        plan = _plan(s)
        results = agent.execute_plan(plan, enable_tools=False)
        assert len(results) == 1
        assert results[0].status == "success"

    def test_step_status_set_to_completed_on_success(self):
        """
        SENTINEL TEST — _apply_step_status must set step.status = 'completed' on success.
        Remove _apply_step_status → step.status stays 'running' → test fails.
        """
        agent = self._make_agent()
        s = _step("x")
        plan = _plan(s)

        with patch.object(
            agent,
            "_execute_step",
            return_value=StepResult(
                step_id=s.id, status="success", outputs={}, logs=[]
            ),
        ):
            agent.execute_plan(plan, enable_tools=False)

        assert s.status == "completed"

    def test_step_status_set_to_failed_on_failure(self):
        """_apply_step_status must set step.status = 'failed' on failure."""
        agent = self._make_agent()
        s = _step("x")
        plan = _plan(s)

        with patch.object(
            agent,
            "_execute_step",
            return_value=StepResult(
                step_id=s.id, status="failure", outputs={}, error="boom", logs=[]
            ),
        ):
            agent.execute_plan(plan, enable_tools=False)

        assert s.status == "failed"

    def test_parallel_exception_captured_as_failure(self):
        """
        SENTINEL TEST — if _execute_step raises during parallel execution,
        the result must be a failure StepResult (not an unhandled exception).
        Remove exception catching in ThreadPoolExecutor block → test raises → test fails.
        """
        agent = self._make_agent()
        steps = [_step("a"), _step("b")]
        plan = _plan(*steps)

        call_count = {"n": 0}

        def _exploding_step(step, enable_tools):
            call_count["n"] += 1
            if step.name == "a":
                raise RuntimeError("unexpected crash")
            return StepResult(step_id=step.id, status="success", outputs={}, logs=[])

        with patch.object(agent, "_execute_step", side_effect=_exploding_step):
            results = agent.execute_plan(plan, enable_tools=False)

        assert len(results) == 2
        a_result = next(r for r in results if r.step_id == steps[0].id)
        assert a_result.status == "failure"
        assert "unexpected crash" in (a_result.error or "")
