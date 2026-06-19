"""Regression FLOOR for the offline benchmark — guards CORE verdict quality.

Runs the offline heuristic benchmark (the real `_score_quality` →
`_check_plan_completion` → `_determine_verdict` path, no LLM/DB) over the
hand-labelled seed fixtures and asserts the engine's verdict-accuracy stays
above a conservative floor. If a change to the cognitive scorer or verdict
logic drops accuracy below the floor, this fails at test/commit time —
surfacing eval drift before it ships.

The floor is intentionally LOOSE (a guard, not a target). Current overall is
~0.78; raise the floor only as the harness and fixtures mature. Per-domain
floors are lower because some domains (notably `code`) are deliberately the
weak spot the harness exists to track — see app/eval/FINDINGS_code_drift_*.md.
"""

from __future__ import annotations

from app.eval.cases import load_cases
from app.eval.runner import DEFAULT_FIXTURES_DIR, run_benchmark

# Conservative floors (current: overall ~0.78). Guards against regression.
_OVERALL_FLOOR = 0.6
_PER_DOMAIN_FLOOR = 0.4


def _report():
    cases = load_cases(DEFAULT_FIXTURES_DIR)
    assert len(cases) >= 15, f"expected >=15 seed cases, got {len(cases)}"
    return run_benchmark(cases)


def test_overall_verdict_accuracy_above_floor():
    report = _report()
    overall = report["overall"]["verdict_accuracy"]
    assert overall >= _OVERALL_FLOOR, (
        f"overall verdict-accuracy {overall:.3f} fell below floor "
        f"{_OVERALL_FLOOR}: the cognitive verdict path may have regressed. "
        f"Report: {report['overall']}"
    )


def test_no_domain_collapses_below_floor():
    report = _report()
    weak = {
        d: s["verdict_accuracy"]
        for d, s in report["by_domain"].items()
        if s["verdict_accuracy"] < _PER_DOMAIN_FLOOR
    }
    assert not weak, f"domain(s) below per-domain floor {_PER_DOMAIN_FLOOR}: {weak}"


def test_report_shape_is_sound():
    """Rates are well-formed and per-domain counts reconcile to the total."""
    report = _report()
    assert set(report) >= {"overall", "by_domain", "cases"}
    for block in [report["overall"], *report["by_domain"].values()]:
        for key in ("verdict_accuracy", "false_approve_rate", "false_reject_rate"):
            assert 0.0 <= block[key] <= 1.0, (key, block[key])
    assert sum(s["n"] for s in report["by_domain"].values()) == report["overall"]["n"]
