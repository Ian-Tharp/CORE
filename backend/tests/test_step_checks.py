"""
Unit tests for Phase 3 per-step evaluation (app/evaluation/step_checks.py) and
how failing checks reach the final evaluation (EvaluationAgent).

Pins:
- each failure kind (no_result, error, partial, no_output, schema_mismatch);
- a healthy step passes; artifacts alone count as output;
- check_steps skips unreached steps and uses the latest result per step;
- evaluate_execution without step_checks is unchanged (no checks block in the
  prompt); with failing checks the prompt names them and the rule-based
  fallback treats them as failed steps.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.core.agents.evaluation_agent import EvaluationAgent
from app.evaluation.step_checks import (
    StepCheck,
    check_step,
    check_steps,
    failing_checks,
)
from app.models.core_state import ExecutionPlan, PlanStep, StepResult, UserIntent

_SCHEMA = {
    "type": "object",
    "properties": {"result": {"type": "string"}},
    "required": ["result"],
}


def _step(name="s", **kw):
    return PlanStep(name=name, description=f"{name} step", **kw)


def _result(step, status="success", outputs=None, **kw):
    return StepResult(
        step_id=step.id,
        status=status,
        outputs={"result": "ok"} if outputs is None else outputs,
        **kw,
    )


class TestCheckStep:
    def test_healthy_step_passes(self):
        step = _step()
        check = check_step(step, _result(step))
        assert check.passed and check.failures == []

    def test_missing_result(self):
        check = check_step(_step(), None)
        assert not check.passed and check.failures == ["no_result"]

    def test_failure_status_is_error(self):
        step = _step()
        check = check_step(
            step, _result(step, status="failure", outputs={}, error="boom")
        )
        assert "error" in check.failures and "no_output" in check.failures
        assert "boom" in check.detail

    def test_error_text_on_success_status_is_error(self):
        step = _step()
        check = check_step(step, _result(step, error="warning-ish"))
        assert check.failures == ["error"]

    def test_partial(self):
        step = _step()
        assert check_step(step, _result(step, status="partial")).failures == ["partial"]

    def test_empty_outputs(self):
        step = _step()
        assert check_step(step, _result(step, outputs={})).failures == ["no_output"]

    def test_outputs_with_only_empty_values(self):
        step = _step()
        check = check_step(step, _result(step, outputs={"result": "", "x": None}))
        assert check.failures == ["no_output"]

    def test_artifacts_count_as_output(self):
        step = _step()
        check = check_step(step, _result(step, outputs={}, artifacts=["out.txt"]))
        assert check.passed

    def test_schema_match(self):
        step = _step(output_schema=_SCHEMA)
        assert check_step(step, _result(step)).passed

    def test_schema_mismatch(self):
        step = _step(output_schema=_SCHEMA)
        check = check_step(step, _result(step, outputs={"result": 42}))
        assert check.failures == ["schema_mismatch"]
        assert "output schema" in check.detail

    def test_schema_missing_required(self):
        step = _step(output_schema=_SCHEMA)
        check = check_step(step, _result(step, outputs={"other": "x"}))
        assert check.failures == ["schema_mismatch"]

    def test_no_schema_declared_means_no_schema_check(self):
        step = _step()
        assert check_step(step, _result(step, outputs={"result": 42})).passed

    def test_to_dict(self):
        step = _step("read")
        d = check_step(step, None).to_dict()
        assert d == {
            "step_id": step.id,
            "step_name": "read",
            "passed": False,
            "failures": ["no_result"],
            "detail": "step produced no result",
        }


class TestCheckSteps:
    def test_none_plan(self):
        assert check_steps(None, []) == []

    def test_unreached_pending_and_skipped_steps_ignored(self):
        a, b, c = _step("a"), _step("b"), _step("c", status="skipped")
        plan = ExecutionPlan(goal="g", steps=[a, b, c])
        checks = check_steps(plan, [_result(a)])
        assert [x.step_id for x in checks] == [a.id]

    def test_step_marked_run_without_result_fails(self):
        a = _step("a", status="completed")
        checks = check_steps(ExecutionPlan(goal="g", steps=[a]), [])
        assert checks[0].failures == ["no_result"]

    def test_latest_result_wins_after_retry(self):
        a = _step("a")
        plan = ExecutionPlan(goal="g", steps=[a])
        first = _result(a, status="failure", outputs={}, error="flaky")
        second = _result(a)
        assert check_steps(plan, [first, second])[0].passed

    def test_failing_checks_accepts_objects_and_dicts(self):
        ok = StepCheck(step_id="1", step_name="a", passed=True)
        bad = StepCheck(step_id="2", step_name="b", passed=False, failures=["error"])
        assert [c["step_id"] for c in failing_checks([ok, bad])] == ["2"]
        assert [
            c["step_id"] for c in failing_checks([ok.to_dict(), bad.to_dict()])
        ] == ["2"]
        assert failing_checks(None) == []


# --------------------------------------------------------------------------
# EvaluationAgent integration
# --------------------------------------------------------------------------


def _llm_response(content):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


_FINALIZE = (
    '{"overall_status": "success", "confidence": 0.9, "meets_requirements": true, '
    '"quality_score": 0.9, "feedback": "ok", "next_action": "finalize"}'
)


def _run_eval(agent, step_checks=None, side_effect=None):
    step = _step("fetch")
    plan = ExecutionPlan(goal="g", steps=[step])
    results = [_result(step, outputs={})]
    intent = UserIntent(type="task", description="d", confidence=0.9)
    client = MagicMock()
    if side_effect:
        client.chat.completions.create.side_effect = side_effect
    else:
        client.chat.completions.create.return_value = _llm_response(_FINALIZE)
    with patch(
        "app.core.agents.evaluation_agent.get_openai_client_sync", return_value=client
    ):
        kwargs = {"step_checks": step_checks} if step_checks is not None else {}
        out = agent.evaluate_execution("q", intent, plan, results, **kwargs)
    return out, client, step


class TestEvaluationAgentStepChecks:
    def test_without_checks_prompt_has_no_checks_block(self):
        _, client, _ = _run_eval(EvaluationAgent())
        user_msg = client.chat.completions.create.call_args.kwargs["messages"][1]
        assert "Deterministic step checks" not in user_msg["content"]

    def test_failing_checks_reach_the_prompt(self):
        agent = EvaluationAgent()
        step = _step("fetch")
        failing = [check_step(step, None).to_dict()]
        _, client, _ = _run_eval(agent, step_checks=failing)
        user_msg = client.chat.completions.create.call_args.kwargs["messages"][1]
        assert "Deterministic step checks FAILED" in user_msg["content"]
        assert "fetch: no_result" in user_msg["content"]

    def test_rule_based_fallback_counts_check_failures(self):
        agent = EvaluationAgent()
        ok_step, bad_step = _step("a"), _step("b")
        results = [_result(ok_step), _result(bad_step, outputs={})]
        checks = [
            check_step(ok_step, results[0]).to_dict(),
            check_step(bad_step, results[1]).to_dict(),
        ]

        # Without checks: both statuses are success → finalize.
        assert agent._rule_based_evaluate(results).next_action == "finalize"

        # With checks: the empty-output step is a failure → retry it.
        out = agent._rule_based_evaluate(results, step_checks=checks)
        assert out.next_action == "retry_step"
        assert out.retry_step_id == bad_step.id
        assert out.quality_score == 0.5

    def test_llm_failure_falls_back_with_checks(self):
        agent = EvaluationAgent()
        step = _step("only")
        plan = ExecutionPlan(goal="g", steps=[step])
        results = [_result(step, outputs={})]  # "success" but empty
        failing = [check_step(step, results[0]).to_dict()]
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("llm down")
        with patch(
            "app.core.agents.evaluation_agent.get_openai_client_sync",
            return_value=client,
        ):
            out = agent.evaluate_execution(
                "q", None, plan, results, step_checks=failing
            )
        # Every step failed its check → the fallback asks for a plan revision.
        assert out.next_action == "revise_plan"
        assert out.meets_requirements is False

    def test_system_prompt_names_retry_vs_revise(self):
        prompt = EvaluationAgent().system_prompt
        assert "Execution flaked" in prompt
        assert "Intent misalignment" in prompt
