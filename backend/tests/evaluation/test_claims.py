"""Claim verification against the tool-call trace (app/evaluation/claims.py)."""

from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from app.evaluation import (
    extract_claims,
    fault_report_from_state,
    run_cases,
    state_failed,
    trace_from_state,
    verify_claims,
    verify_state_claims,
)
from app.evaluation.models import ClaimsSupported, GoldenCase, RunOutput, ToolCallRecord
from app.models.core_state import (
    COREState,
    EvaluationResult,
    ExecutionPlan,
    PlanStep,
    StepResult,
    UserIntent,
)


def write(path, ok=True):
    return ToolCallRecord(
        name="file_operations",
        params={"action": "write", "path": path, "content": "x"},
        result={"status": "success" if ok else "error", "path": path},
    )


def kinds(text):
    return [(c.kind, c.target) for c in extract_claims(text)]


# ---- extraction ---------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "I wrote the config to `config/app.yaml`.",
            [("file_write", "config/app.yaml")],
        ),
        ("The file notes.txt has been updated.", [("file_write", "notes.txt")]),
        ("Successfully wrote report.md", [("file_write", "report.md")]),
        ("I read README.md.", [("file_inspect", "README.md")]),
        ("I've committed and pushed the fix.", [("git_write", None)]),
        ("I ran the tests and they passed.", [("command", None)]),
        ("I updated the database records.", [("database", None)]),
        (
            "I fetched https://example.com/a and summarized it.",
            [("web_fetch", "https://example.com/a")],
        ),
    ],
)
def test_extracts_action_claims(text, expected):
    assert kinds(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "I will write main.py next.",
        "If I wrote main.py, the tests would pass.",
        "I created a plan with three steps.",
        "I reviewed the options carefully.",
        "The plan has been updated.",
        "Version 2.0 was released.",
        "I haven't written anything yet.",
        "",
    ],
)
def test_ignores_plans_conditionals_and_non_actions(text):
    assert extract_claims(text) == []


# ---- verification ---------------------------------------------------------


def test_write_claim_supported_by_matching_successful_write():
    r = verify_claims("I wrote `app.yaml`.", [write("config/app.yaml")])
    assert r.is_clean()
    assert r.checks[0].evidence.startswith("file_operations write")


def test_write_claim_unsupported_without_call_wrong_path_or_failed_call():
    text = "I saved notes.md."
    assert not verify_claims(text, []).is_clean()
    assert not verify_claims(text, [write("other.md")]).is_clean()
    assert not verify_claims(text, [write("notes.md", ok=False)]).is_clean()
    read = ToolCallRecord(
        name="file_operations", params={"action": "read", "path": "notes.md"}
    )
    assert not verify_claims(text, [read]).is_clean()  # reading is not writing


def test_git_write_never_supported_by_read_only_git():
    status = ToolCallRecord(name="git", params={"action": "status"}, result={})
    assert not verify_claims("I committed the change.", [status]).is_clean()
    assert verify_claims("I ran git status.", [status]).is_clean()
    assert not verify_claims("I ran git log.", [status]).is_clean()


def test_web_claim_matches_url_prefix():
    call = ToolCallRecord(
        name="web_research",
        params={"url": "https://example.com/a?x=1"},
        result={"status": "success", "url": "https://example.com/a?x=1"},
    )
    assert verify_claims("I fetched https://example.com/a.", [call]).is_clean()
    assert not verify_claims("I fetched https://other.org.", [call]).is_clean()


def test_inspect_claim_satisfied_by_list_entries():
    ls = ToolCallRecord(
        name="file_operations",
        params={"action": "list", "path": "."},
        result={"status": "success", "entries": ["src/", "src/main.py"]},
    )
    assert verify_claims("I looked at main.py.", [ls]).is_clean()


# ---- COREState adapters -----------------------------------------------------


def _core_state(llm_text, dispatched=True):
    write_step = PlanStep(
        id="w",
        name="write",
        description="write",
        tool="file_operations",
        params={"action": "write", "path": "out.txt", "content": "hi"},
    )
    think = PlanStep(id="t", name="summarize", description="summarize")
    results = [
        StepResult(
            step_id="w",
            status="success" if dispatched else "failure",
            outputs=(
                {"result": "Wrote 2 chars", "path": "out.txt", "status": "success"}
                if dispatched
                else {}
            ),
            dispatched_tool="file_operations" if dispatched else None,
        ),
        StepResult(step_id="t", status="success", outputs={"result": llm_text}),
    ]
    return COREState(
        user_input="x",
        intent=UserIntent(type="task", description="t", confidence=0.9),
        plan=ExecutionPlan(goal="g", steps=[write_step, think]),
        step_results=results,
    )


def test_trace_contains_only_dispatched_calls():
    assert [c.name for c in trace_from_state(_core_state("ok"))] == ["file_operations"]
    assert trace_from_state(_core_state("ok", dispatched=False)) == []


def test_state_claims_ignore_tool_output_text():
    state = _core_state("Summary done.")
    state.step_results[0].outputs["result"] = "I committed and pushed."  # file content
    assert verify_state_claims(state).checks == []


def test_state_claims_supported_and_unsupported():
    assert verify_state_claims(_core_state("I wrote out.txt.")).is_clean()
    bad = verify_state_claims(_core_state("I wrote out.txt.", dispatched=False))
    assert [c.target for c in bad.unsupported] == ["out.txt"]


# ---- graph wiring -----------------------------------------------------------


def _graph():
    from app.core.langgraph.core_graph_v2 import COREGraph

    return object.__new__(COREGraph)


def test_evaluation_node_downgrades_on_unsupported_claim():
    g = _graph()
    g.evaluation_agent = NS(
        evaluate_execution=lambda **_: EvaluationResult(
            overall_status="success",
            confidence=0.95,
            meets_requirements=True,
            quality_score=0.9,
            feedback="looks good",
        )
    )
    state = g.evaluation_node(_core_state("I committed the fix to main."))
    ev = state.eval_result
    assert not ev.meets_requirements and ev.confidence <= 0.5
    assert ev.next_action == "finalize"  # annotate only, never reroute
    assert "Unsupported action claims" in ev.feedback
    assert state.claim_report["unsupported"][0]["kind"] == "git_write"
    assert state_failed(state)


def test_evaluation_node_clean_claims_leave_eval_alone():
    g = _graph()
    g.evaluation_agent = NS(
        evaluate_execution=lambda **_: EvaluationResult(
            overall_status="success",
            confidence=0.95,
            meets_requirements=True,
            quality_score=0.9,
        )
    )
    state = g.evaluation_node(_core_state("I wrote out.txt."))
    assert state.eval_result.confidence == 0.95
    assert state.claim_report["unsupported"] == []


def test_conversation_flags_unsupported_claims():
    g = _graph()
    state = _core_state("I committed the fix.")
    state.eval_result = EvaluationResult(
        overall_status="success",
        confidence=0.9,
        meets_requirements=True,
        quality_score=0.9,
    )
    state.claim_report = verify_state_claims(state).to_dict()
    with patch("app.core.langgraph.core_graph_v2.record_pipeline_run"):
        out = g.conversation_node(state)
    assert "Not verified" in out.response
    assert "I committed the fix" in out.response


# ---- fault capture + golden evaluator ---------------------------------------


def test_fault_report_skips_undispatched_and_records_claims():
    state = _core_state("I committed it.", dispatched=False)
    state.claim_report = verify_state_claims(state).to_dict()
    fr = fault_report_from_state(state)
    assert fr.tool_calls == []  # planned write never ran
    assert fr.metadata["unsupported_claims"][0]["kind"] == "git_write"


def test_claims_supported_evaluator_uses_run_tool_calls():
    case = GoldenCase(id="c", input="save it", evaluators=[ClaimsSupported()])
    good = run_cases(
        [case], lambda _: RunOutput(output="I saved a.md.", tool_calls=[write("a.md")])
    )
    bad = run_cases([case], lambda _: "I saved a.md.")
    assert good.summary.passed == 1
    assert bad.summary.failed == 1
    assert "unsupported claims" in bad.results[0].outcomes[0].detail
