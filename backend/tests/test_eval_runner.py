"""
Tests for app.eval.runner — the offline benchmark runner.

Strictly offline: no DB, no LLM, no network. The runner replays the real
hand-labelled fixtures under ``tests/fixtures/eval_cases/`` through the engine's
real heuristic scorer + verdict callables and reports accuracy / drift. These
tests assert the report's *structure* and *invariants* — not a particular
accuracy number — because the whole point of the runner is to surface where the
heuristic disagrees with the hand labels. Forcing those agreements here would
defeat the drift signal.

Sentinel-value contract:
- Break the rate denominators (e.g. divide by total instead of per-domain n) →
  per-domain rates leave [0, 1] → the rate-bounds assertion fails.
- Drop a domain from aggregation → per-domain n no longer sums to the total →
  the sum assertion fails.
- Shrink the fixture corpus below 18 cases → coverage drops → the count
  assertion fails.
- Stop computing predicted verdicts (e.g. always "") → no predicted verdict is
  ever a valid Verdict value → the verdict-vocabulary assertion fails.
- Miscount false_approve / false_reject so they exceed n → the
  confusion-bounds assertion fails.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.eval.cases import VALID_VERDICTS, load_cases
from app.eval.runner import run_benchmark

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "eval_cases"

# The corpus is hand-authored: 3 domains x 6 cases.
EXPECTED_CASE_COUNT = 18

_DOMAIN_BLOCK_KEYS = {
    "n",
    "verdict_accuracy",
    "false_approve_rate",
    "false_reject_rate",
    "confusion",
    "predicted_counts",
    "ground_truth_counts",
}


@pytest.fixture(scope="module")
def report() -> dict:
    """Run the benchmark once over the real fixtures for the whole module."""
    cases = load_cases(FIXTURES_DIR)
    return run_benchmark(cases)


# ===========================================================================
# Structure
# ===========================================================================


def test_report_top_level_keys(report: dict) -> None:
    """The report exposes overall, by_domain, and per-case rows."""
    assert set(report.keys()) == {"overall", "by_domain", "cases"}


def test_overall_block_has_expected_keys(report: dict) -> None:
    """The overall block carries the full rate/confusion structure."""
    assert _DOMAIN_BLOCK_KEYS.issubset(report["overall"].keys())


def test_each_domain_block_has_expected_keys(report: dict) -> None:
    """Every per-domain block mirrors the overall structure."""
    assert report["by_domain"], "expected at least one domain"
    for domain, block in report["by_domain"].items():
        assert _DOMAIN_BLOCK_KEYS.issubset(block.keys()), domain


def test_domains_are_the_fixture_domains(report: dict) -> None:
    """The runner discovers exactly the three authored domains."""
    assert set(report["by_domain"].keys()) == {"code", "research", "file_ops"}


# ===========================================================================
# Coverage / counts
# ===========================================================================


def test_ran_all_eighteen_cases(report: dict) -> None:
    """All 18 hand-labelled cases were evaluated."""
    assert report["overall"]["n"] == EXPECTED_CASE_COUNT
    assert len(report["cases"]) == EXPECTED_CASE_COUNT


def test_per_domain_n_sums_to_total(report: dict) -> None:
    """Per-domain case counts partition the overall total exactly."""
    domain_total = sum(block["n"] for block in report["by_domain"].values())
    assert domain_total == report["overall"]["n"] == EXPECTED_CASE_COUNT


# ===========================================================================
# Rate invariants
# ===========================================================================


def _iter_blocks(report: dict):
    yield "overall", report["overall"]
    for domain, block in report["by_domain"].items():
        yield domain, block


@pytest.mark.parametrize(
    "rate_key",
    ["verdict_accuracy", "false_approve_rate", "false_reject_rate"],
)
def test_all_rates_in_unit_interval(report: dict, rate_key: str) -> None:
    """Every reported rate is a probability in [0, 1]."""
    for label, block in _iter_blocks(report):
        value = block[rate_key]
        assert 0.0 <= value <= 1.0, f"{label}.{rate_key} = {value!r}"


def test_confusion_counts_do_not_exceed_n(report: dict) -> None:
    """Misclassification counts can never exceed the block's case count."""
    for label, block in _iter_blocks(report):
        n = block["n"]
        assert sum(block["confusion"].values()) <= n, label
        assert sum(block["predicted_counts"].values()) == n, label
        assert sum(block["ground_truth_counts"].values()) == n, label


# ===========================================================================
# Verdicts were actually produced (the runner did real work)
# ===========================================================================


def test_every_case_has_a_valid_predicted_verdict(report: dict) -> None:
    """Each case row carries a predicted verdict from the engine's vocabulary."""
    for row in report["cases"]:
        assert row["predicted_verdict"] in VALID_VERDICTS, row["id"]
        assert row["ground_truth_verdict"] in VALID_VERDICTS, row["id"]


def test_some_predicted_verdicts_are_produced(report: dict) -> None:
    """The engine produced a non-trivial spread of predictions (did real work)."""
    predicted = {row["predicted_verdict"] for row in report["cases"]}
    assert predicted, "no predictions produced"
    # The corpus spans approve and non-approve ground truths; the engine should
    # likewise not collapse every case to a single verdict.
    assert len(predicted) >= 2, f"engine collapsed all cases to {predicted}"


def test_quality_overall_in_unit_interval(report: dict) -> None:
    """Per-case quality_overall stays on the engine's [0, 1] scale."""
    for row in report["cases"]:
        assert 0.0 <= row["quality_overall"] <= 1.0, row["id"]
