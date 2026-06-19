"""
Tests for app.eval.cases — BenchmarkCase model + load_cases() loader.

Unit-level only: no DB, no model loads, no service imports, no network. These
tests validate the hand-labelled offline fixtures under
``tests/fixtures/eval_cases/`` and the loader that reads them, so the offline
eval harness has a trustworthy, well-formed corpus to replay.

Sentinel-value contract:
- Shrink the fixture corpus below 15 cases → the offline benchmark loses
  coverage → the count assertion fails.
- Introduce a ground_truth_verdict outside {approve, retry, refine, escalate}
  (e.g. a typo'd "approved") → BenchmarkCase's pattern rejects it on load, or
  the verdict-vocabulary assertion fails.
- Author a corpus that is all-approve or all-failure → the harness can no
  longer distinguish good from bad behavior → the "needs ≥1 approve and ≥1
  non-approve" assertions fail.
- Emit an expected_quality_band outside {low, medium, high} → the QualityBand
  enum rejects it on load, or the band-membership assertion fails.
- Drop a required field (intent, final_output, ...) from a fixture →
  model_validate raises and load_cases surfaces it here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.eval.cases import (
    VALID_VERDICTS,
    BenchmarkCase,
    QualityBand,
    load_cases,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "eval_cases"


@pytest.fixture(scope="module")
def cases() -> list[BenchmarkCase]:
    """Load every fixture file once for the module."""
    return load_cases(FIXTURES_DIR)


def test_fixtures_dir_exists():
    """The fixtures directory must exist where the loader expects it."""
    assert FIXTURES_DIR.is_dir(), f"missing fixtures dir: {FIXTURES_DIR}"


def test_loads_at_least_fifteen_cases(cases):
    """
    SENTINEL — the corpus must hold at least 15 cases.
    Trim the fixtures below 15 → coverage drops → this fails.
    """
    assert len(cases) >= 15


def test_all_entries_are_benchmark_cases(cases):
    """load_cases() returns validated BenchmarkCase instances."""
    assert cases  # non-empty
    assert all(isinstance(c, BenchmarkCase) for c in cases)


def test_every_verdict_is_valid(cases):
    """
    SENTINEL — every ground_truth_verdict is a real Verdict value.
    A typo or out-of-vocab verdict → caught by the model on load or here.
    """
    for case in cases:
        assert (
            case.ground_truth_verdict in VALID_VERDICTS
        ), f"{case.id} has invalid verdict {case.ground_truth_verdict!r}"


def test_verdict_vocabulary_matches_engine():
    """The local verdict vocabulary mirrors the engine's Verdict values."""
    assert VALID_VERDICTS == {"approve", "retry", "refine", "escalate"}


def test_has_at_least_one_approve(cases):
    """
    SENTINEL — at least one clearly-good (approve) case exists.
    An all-failure corpus can't validate the approve path → this fails.
    """
    approvals = [c for c in cases if c.ground_truth_verdict == "approve"]
    assert approvals, "corpus has no approve cases"


def test_has_at_least_one_non_approve(cases):
    """
    SENTINEL — at least one clearly-bad (non-approve) case exists.
    An all-approve corpus can't validate retry/refine/escalate → this fails.
    """
    non_approvals = [c for c in cases if c.ground_truth_verdict != "approve"]
    assert non_approvals, "corpus has no non-approve cases"


def test_every_quality_band_is_valid(cases):
    """Each expected_quality_band is one of low | medium | high."""
    valid_bands = {QualityBand.LOW, QualityBand.MEDIUM, QualityBand.HIGH}
    valid_values = {"low", "medium", "high"}
    for case in cases:
        assert case.expected_quality_band in valid_bands
        assert case.expected_quality_band.value in valid_values


def test_domains_are_in_allowed_set(cases):
    """Every case declares a known domain."""
    allowed = {"code", "research", "file_ops"}
    for case in cases:
        assert case.domain in allowed, f"{case.id} bad domain {case.domain!r}"


def test_case_ids_are_unique(cases):
    """Case ids are globally unique across all fixture files."""
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids found"


def test_low_band_cases_are_failure_verdicts(cases):
    """A low expected-quality band should never be labelled approve."""
    for case in cases:
        if case.expected_quality_band is QualityBand.LOW:
            assert (
                case.ground_truth_verdict != "approve"
            ), f"{case.id} is band=low but verdict=approve"


def test_missing_directory_raises(tmp_path):
    """load_cases() raises FileNotFoundError for a non-existent directory."""
    with pytest.raises(FileNotFoundError):
        load_cases(tmp_path / "does-not-exist")
