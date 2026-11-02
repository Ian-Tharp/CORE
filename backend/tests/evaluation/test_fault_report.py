from datetime import datetime, timezone
from types import SimpleNamespace as NS

from app.evaluation import (
    FaultReport,
    capture_fault,
    fault_report_from_state,
    load_faults,
    state_failed,
)
from app.evaluation.models import StepRecord, TokenUsage, ToolCallRecord


def test_fault_report_json_roundtrip():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fr = FaultReport(
        input="book a room",
        steps=[StepRecord(name="reasoning", status="failure", latency_ms=12.5)],
        tool_calls=[ToolCallRecord(name="calendar.create", params={"a": 1}, error="x")],
        expected="booked",
        actual="boom",
        started_at=t0,
        ended_at=t0,
        latency_ms=40.0,
        tokens=TokenUsage(prompt=10, completion=5),
    )
    again = FaultReport.model_validate_json(fr.model_dump_json())
    assert again == fr
    assert again.tokens.total == 15


def test_token_total_none_when_unknown():
    assert TokenUsage().total is None


def _state(**kw):
    base = dict(
        run_id="r1",
        user_input="hello",
        errors=[],
        eval_result=None,
        step_results=[],
        execution_history=["comprehension", "reasoning"],
        plan=None,
        response=None,
        started_at=datetime(2026, 1, 1, 0, 0, 0),
        completed_at=datetime(2026, 1, 1, 0, 0, 2),
    )
    base.update(kw)
    return NS(**base)


def test_state_failed_signals():
    assert not state_failed(_state())
    assert state_failed(_state(errors=["x"]))
    assert state_failed(_state(eval_result=NS(overall_status="failure")))
    assert state_failed(_state(step_results=[NS(status="failure")]))


def test_report_from_state_maps_steps_tools_and_latency():
    res = NS(
        step_id="s1",
        status="failure",
        outputs={"rows": 0},
        error="timeout",
        duration_seconds=1.5,
    )
    plan = NS(
        steps=[
            NS(id="s1", tool="db.query", params={"q": "x"}),
            NS(id="s2", tool=None, params={}),
        ]
    )
    fr = fault_report_from_state(
        _state(step_results=[res], plan=plan, errors=["e"], response="oops"),
        expected="rows",
    )
    assert [s.name for s in fr.steps] == ["comprehension", "reasoning", "s1"]
    assert fr.steps[-1].latency_ms == 1500.0
    assert len(fr.tool_calls) == 1
    tc = fr.tool_calls[0]
    assert (tc.name, tc.params, tc.error, tc.result) == (
        "db.query",
        {"q": "x"},
        "timeout",
        {"rows": 0},
    )
    assert fr.latency_ms == 2000.0
    assert (fr.expected, fr.actual, fr.run_id) == ("rows", "oops", "r1")


def test_report_from_state_tolerates_sparse_state():
    fr = fault_report_from_state(NS(user_input="hi"))
    assert fr.input == "hi" and fr.steps == [] and fr.latency_ms is None


def test_capture_and_load_faults_skips_garbage(tmp_path):
    a, b = FaultReport(input="a"), FaultReport(input="b")
    capture_fault(a, tmp_path)
    p = capture_fault(b, tmp_path)
    with p.open("a") as fh:
        fh.write("{not json\n")
    assert [f.input for f in load_faults(tmp_path)] == ["a", "b"]
    assert load_faults(tmp_path / "missing") == []
