"""Fault capture wired into core_graph_v2: gated, fail-safe, one report per fault.

Runs the real compiled LangGraph with stubbed agents (no LLM, no network) and
writes only under pytest's tmp_path.

Sentinel-value contract:
- Drop the enabled check -> flag-off run writes a file -> test fails
- Drop the maybe_capture_fault call from conversation_node -> flag-on run
  writes nothing -> test fails
- Let capture exceptions escape -> unwritable-dir run raises -> test fails
- Log the swallowed failure below WARNING -> caplog assertion fails
- Treat a gate refusal as a fault -> refusal run writes a file -> test fails
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.config.evaluation import (
    BACKEND_ROOT,
    DEFAULT_FAULTS_DIR,
    FaultCaptureConfig,
    get_fault_capture_config,
)
from app.core.agents.comprehension_gate import GateDecision
from app.evaluation import fault_reasons, load_faults, maybe_capture_fault
from app.evaluation.capture import FAULTS_FILENAME
from app.models.core_state import (
    COREState,
    EvaluationResult,
    ExecutionPlan,
    PlanStep,
    StepResult,
    UserIntent,
)

FAULT_INPUT = "SENTINEL_FAULTING_REQUEST"


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("CORE_EVAL_CAPTURE_FAULTS", raising=False)
    monkeypatch.delenv("CORE_EVAL_FAULTS_DIR", raising=False)


@pytest.fixture()
def faults_dir(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "faults"
    monkeypatch.setenv("CORE_EVAL_FAULTS_DIR", str(d))
    return d


def _eval(status: str, next_action: str = "finalize") -> EvaluationResult:
    return EvaluationResult(
        overall_status=status,
        confidence=0.9,
        meets_requirements=status == "success",
        quality_score=0.9 if status == "success" else 0.1,
        feedback="stub",
        next_action=next_action,
    )


def _build_graph(*, step_status: str, eval_status: str, gate: str = "proceed"):
    """Real COREGraph + compiled LangGraph, with every agent stubbed."""
    with patch.multiple(
        "app.core.langgraph.core_graph_v2",
        ComprehensionAgent=MagicMock(),
        OrchestrationAgent=MagicMock(),
        ReasoningAgent=MagicMock(),
        EvaluationAgent=MagicMock(),
    ):
        from app.core.langgraph.core_graph_v2 import COREGraph

        g = COREGraph()

    g.comprehension_agent.analyze_intent.return_value = UserIntent(
        type="task", description="do the thing", confidence=0.9
    )
    g.comprehension_agent.detect_ambiguities.return_value = []
    g.orchestration_agent.create_plan.return_value = ExecutionPlan(
        goal="do the thing",
        steps=[PlanStep(id="s1", name="think", description="llm-only step")],
    )
    g.reasoning_agent.execute_plan.return_value = [
        StepResult(
            step_id="s1",
            status=step_status,
            outputs={"result": "stub output"},
            error="boom" if step_status == "failure" else None,
        )
    ]
    g.evaluation_agent.evaluate_execution.return_value = _eval(eval_status)
    g.initialize_graph()
    g._gate = GateDecision(
        outcome=gate, message="I can't do that." if gate == "refuse" else ""
    )
    return g


def _run(g) -> COREState:
    with patch("app.core.agents.comprehension_gate.decide_gate", return_value=g._gate):
        out = g.get_graph().invoke(COREState(user_input=FAULT_INPUT))
    return COREState(**out) if isinstance(out, dict) else out


def _lines(d: Path) -> list[str]:
    p = d / FAULTS_FILENAME
    if not p.exists():
        return []
    return [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


def test_config_defaults_off_and_under_backend_data_dir():
    cfg = get_fault_capture_config()
    assert cfg.enabled is False
    assert cfg.directory == DEFAULT_FAULTS_DIR
    assert DEFAULT_FAULTS_DIR.is_relative_to(BACKEND_ROOT / "data")
    assert DEFAULT_FAULTS_DIR.parent != BACKEND_ROOT.parent  # never repo root


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
def test_config_truthy_values_enable(monkeypatch, val):
    monkeypatch.setenv("CORE_EVAL_CAPTURE_FAULTS", val)
    assert get_fault_capture_config().enabled is True


@pytest.mark.parametrize("val", ["", "0", "false", "no", "nope"])
def test_config_other_values_disable(monkeypatch, val):
    monkeypatch.setenv("CORE_EVAL_CAPTURE_FAULTS", val)
    assert get_fault_capture_config().enabled is False


def test_config_relative_dir_anchored_to_backend_not_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CORE_EVAL_FAULTS_DIR", "data/custom")
    assert get_fault_capture_config().directory == BACKEND_ROOT / "data" / "custom"


def test_config_absolute_dir_used_as_is(faults_dir):
    assert get_fault_capture_config().directory == faults_dir


# --------------------------------------------------------------------------
# Fault detection
# --------------------------------------------------------------------------


def test_refusal_alone_is_not_a_fault():
    s = COREState(user_input="x", gate_outcome="refuse", gate_message="no")
    assert fault_reasons(s) == []


def test_loop_guard_forced_finalization_is_a_fault():
    s = COREState(
        user_input="x",
        eval_result=_eval("needs_revision", next_action="revise_plan"),
        plan_revisions=3,
    )
    assert fault_reasons(s) == ["unresolved_evaluation"]


def test_reasons_combine():
    s = COREState(
        user_input="x",
        errors=["Reasoning failed: x"],
        eval_result=_eval("failure"),
        step_results=[StepResult(step_id="s1", status="failure")],
    )
    assert fault_reasons(s) == ["errors", "eval_failure", "step_failure"]


# --------------------------------------------------------------------------
# End-to-end through the compiled graph
# --------------------------------------------------------------------------


def test_flag_off_faulting_run_writes_nothing_and_never_calls_capture(faults_dir):
    g = _build_graph(step_status="failure", eval_status="failure")
    with patch("app.evaluation.capture.capture_fault") as spy, patch(
        "app.evaluation.capture.fault_report_from_state"
    ) as build_spy:
        state = _run(g)
    assert state.response  # request still answered
    spy.assert_not_called()
    build_spy.assert_not_called()
    assert not faults_dir.exists()


def test_flag_on_faulting_run_writes_exactly_one_report(faults_dir, monkeypatch):
    monkeypatch.setenv("CORE_EVAL_CAPTURE_FAULTS", "true")
    g = _build_graph(step_status="failure", eval_status="failure")

    state = _run(g)

    assert state.response
    assert len(_lines(faults_dir)) == 1
    [report] = load_faults(faults_dir)
    assert report.run_id == state.run_id
    assert report.input == FAULT_INPUT
    assert report.metadata["source"] == "core_graph_v2"
    assert report.metadata["fault_reasons"] == ["eval_failure", "step_failure"]
    assert report.metadata["eval_status"] == "failure"
    assert any(s.name == "s1" and s.status == "failure" for s in report.steps)
    assert report.actual == state.response


def test_flag_on_clean_run_writes_nothing(faults_dir, monkeypatch):
    monkeypatch.setenv("CORE_EVAL_CAPTURE_FAULTS", "true")
    g = _build_graph(step_status="success", eval_status="success")
    state = _run(g)
    assert state.errors == []
    assert _lines(faults_dir) == []


def test_flag_on_gate_refusal_writes_nothing(faults_dir, monkeypatch):
    monkeypatch.setenv("CORE_EVAL_CAPTURE_FAULTS", "true")
    g = _build_graph(step_status="success", eval_status="success", gate="refuse")
    state = _run(g)
    assert state.response == "I can't do that."
    g.orchestration_agent.create_plan.assert_not_called()
    assert _lines(faults_dir) == []


def test_unwritable_dir_request_succeeds_and_warns(tmp_path, monkeypatch, caplog):
    # A regular file where the directory should be: mkdir fails even as root.
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("occupied")
    monkeypatch.setenv("CORE_EVAL_CAPTURE_FAULTS", "true")
    monkeypatch.setenv("CORE_EVAL_FAULTS_DIR", str(blocker / "faults"))
    g = _build_graph(step_status="failure", eval_status="failure")

    with caplog.at_level(logging.WARNING):
        state = _run(g)

    assert state.response == "stub output"
    assert state.completed_at is not None
    assert not any("capture" in e.lower() for e in state.errors)
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "Fault capture failed" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert state.run_id in warnings[0].getMessage()
    assert blocker.read_text() == "occupied"


def test_unexpected_capture_bug_is_swallowed(faults_dir, monkeypatch, caplog):
    monkeypatch.setenv("CORE_EVAL_CAPTURE_FAULTS", "true")
    g = _build_graph(step_status="failure", eval_status="failure")
    with patch(
        "app.evaluation.capture.fault_report_from_state",
        side_effect=RuntimeError("SENTINEL_BUG"),
    ), caplog.at_level(logging.WARNING):
        state = _run(g)
    assert state.response == "stub output"
    assert any("SENTINEL_BUG" in r.getMessage() for r in caplog.records)
    assert _lines(faults_dir) == []


# --------------------------------------------------------------------------
# maybe_capture_fault in isolation
# --------------------------------------------------------------------------


def test_explicit_config_overrides_env(tmp_path):
    s = COREState(user_input="x", errors=["boom"])
    off = FaultCaptureConfig(enabled=False, directory=tmp_path)
    on = FaultCaptureConfig(enabled=True, directory=tmp_path)
    assert maybe_capture_fault(s, off) is None
    path = maybe_capture_fault(s, on)
    assert path == tmp_path / FAULTS_FILENAME
    assert len(_lines(tmp_path)) == 1
