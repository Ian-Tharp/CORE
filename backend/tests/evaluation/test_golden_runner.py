import pytest

from app.evaluation import RunOutput, arun_cases, run_cases
from app.evaluation.models import Contains, ExactMatch, GoldenCase, MaxLatency


def gc(id_, text, *ev):
    return GoldenCase(id=id_, input=text, evaluators=list(ev))


CASES = [
    gc("ok", "hi", Contains(value="hello")),
    gc("bad", "yo", Contains(value="nope")),
    gc("slow", "z", MaxLatency(ms=50)),
]


def fake(text):
    if text == "z":
        return RunOutput(output="done", latency_ms=500)
    return "hello there"


def test_runs_every_case_and_summarizes():
    run = run_cases(CASES, fake)
    assert [r.case_id for r in run.results] == ["ok", "bad", "slow"]
    s = run.summary
    assert (s.total, s.passed, s.failed, s.errored) == (3, 1, 2, 0)
    assert s.pass_rate == pytest.approx(1 / 3)
    assert s.failed_case_ids == ["bad", "slow"]
    assert s.by_evaluator["contains"] == {"passed": 1, "failed": 1}
    assert run.ended_at >= run.started_at


def test_callable_gets_input_only_and_wall_latency_measured():
    seen = []

    def fn(x):
        seen.append(x)
        return "a"

    run = run_cases([gc("c", "in", ExactMatch(value="a"), MaxLatency(ms=5000))], fn)
    assert seen == ["in"] and run.results[0].passed
    assert run.results[0].latency_ms is not None


def test_exception_is_failed_case_and_run_continues():
    def fn(x):
        if x == "yo":
            raise RuntimeError("kaput")
        return "hello"

    run = run_cases(CASES[:2], fn)
    assert run.results[0].passed
    assert run.results[1].error == "RuntimeError: kaput"
    assert run.summary.errored == 1 and run.summary.failed == 1


def test_bad_return_type_fails_case():
    run = run_cases(CASES[:1], lambda x: 42)
    assert not run.results[0].passed and "int" in run.results[0].error


def test_empty_run():
    s = run_cases([], fake).summary
    assert (s.total, s.pass_rate) == (0, 0.0)


def test_result_json_roundtrip():
    from app.evaluation import RunResult

    run = run_cases(CASES, fake)
    assert RunResult.model_validate_json(run.model_dump_json()) == run


async def test_async_runner_accepts_sync_and_async_callables():
    async def afn(x):
        return "hello"

    assert (await arun_cases(CASES[:1], afn)).summary.passed == 1
    assert (await arun_cases(CASES[:1], lambda x: "hello")).summary.passed == 1
