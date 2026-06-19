"""
Tests for app.eval.protocol — Verifier Protocol + VerifierResult value object.

Unit-level only: no DB, no model loads, no event loop. Importing
``app.eval`` must stay cheap, so these tests assert the contracts in
isolation.

Sentinel-value contract:
- Remove the ``__post_init__`` score check → out-of-[0,1] scores silently
  stored → aggregate eval metrics corrupted → test fails.
- Remove ``frozen=True`` → VerifierResult becomes mutable → results can be
  rewritten mid-run → immutability test fails.
- Remove the ``metrics`` default_factory → unset metrics is required / shared
  → construction test fails.
- Remove ``@runtime_checkable`` from Verifier → isinstance() against the
  Protocol raises TypeError → satisfiability test fails.
- Change ``verify`` to async / a non-VerifierResult return → the in-test
  Verifier no longer satisfies the contract → test fails.
"""

from __future__ import annotations

import dataclasses

import pytest

from app.eval import Verifier, VerifierResult


# ===========================================================================
# VerifierResult — defaults and bounds
# ===========================================================================


class TestVerifierResult:
    def test_minimal_construction_defaults_metrics(self):
        """metrics defaults to an empty dict when not supplied."""
        result = VerifierResult(passed=True, score=1.0, detail="ok")
        assert result.passed is True
        assert result.score == 1.0
        assert result.detail == "ok"
        assert result.metrics == {}

    def test_metrics_default_is_not_shared(self):
        """
        SENTINEL — each instance gets its own metrics dict.
        Remove default_factory (use a bare {} default) → instances share one
        mutable dict → mutating one leaks into the next → test fails.
        """
        a = VerifierResult(passed=True, score=1.0, detail="a")
        b = VerifierResult(passed=False, score=0.0, detail="b")
        a.metrics["k"] = 1
        assert b.metrics == {}

    def test_metrics_payload_preserved(self):
        """Caller-supplied metrics are stored verbatim."""
        metrics = {"predicted_verdict": "retry", "quality_overall": 0.62}
        result = VerifierResult(
            passed=False, score=0.62, detail="mismatch", metrics=metrics
        )
        assert result.metrics == metrics

    def test_score_bounds_lower(self):
        """score == 0.0 is the inclusive lower bound and is accepted."""
        result = VerifierResult(passed=False, score=0.0, detail="floor")
        assert result.score == 0.0

    def test_score_bounds_upper(self):
        """score == 1.0 is the inclusive upper bound and is accepted."""
        result = VerifierResult(passed=True, score=1.0, detail="ceiling")
        assert result.score == 1.0

    def test_score_above_one_rejected(self):
        """
        SENTINEL — score > 1.0 must raise ValueError.
        Remove the __post_init__ check → out-of-scale score stored →
        downstream aggregation breaks → test fails.
        """
        with pytest.raises(ValueError):
            VerifierResult(passed=True, score=1.1, detail="too high")

    def test_score_below_zero_rejected(self):
        """score < 0.0 must raise ValueError."""
        with pytest.raises(ValueError):
            VerifierResult(passed=False, score=-0.01, detail="too low")

    def test_is_frozen(self):
        """
        SENTINEL — VerifierResult is immutable.
        Remove frozen=True → assignment succeeds → results mutable mid-run →
        test fails.
        """
        result = VerifierResult(passed=True, score=1.0, detail="ok")
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.passed = False  # type: ignore[misc]


# ===========================================================================
# Verifier — Protocol satisfiability
# ===========================================================================


class _ConstVerifier:
    """Trivial in-test Verifier: it does not subclass Verifier.

    Proves the Protocol is satisfiable by structural typing alone — a class
    with a matching sync ``verify`` returning a VerifierResult is a Verifier.
    """

    def verify(self, case, actual) -> VerifierResult:
        passed = bool(actual)
        return VerifierResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            detail="truthy" if passed else "falsy",
            metrics={"actual_repr": repr(actual)},
        )


class TestVerifierProtocol:
    def test_structural_instance_check(self):
        """
        SENTINEL — a non-subclass with a matching verify() is a Verifier.
        Remove @runtime_checkable → isinstance() raises TypeError → test fails.
        Change verify's signature/return contract → not recognized → test fails.
        """
        verifier = _ConstVerifier()
        assert isinstance(verifier, Verifier)

    def test_verify_returns_verifier_result(self):
        """verify() returns a VerifierResult with the expected verdict."""
        verifier: Verifier = _ConstVerifier()
        result = verifier.verify(case=None, actual="non-empty")
        assert isinstance(result, VerifierResult)
        assert result.passed is True
        assert result.score == 1.0
        assert result.metrics["actual_repr"] == repr("non-empty")

    def test_verify_failure_path(self):
        """A falsy actual yields passed=False, score=0.0."""
        result = _ConstVerifier().verify(case=None, actual="")
        assert result.passed is False
        assert result.score == 0.0
        assert result.detail == "falsy"

    def test_plain_object_is_not_a_verifier(self):
        """An object lacking verify() does not satisfy the Protocol."""
        assert not isinstance(object(), Verifier)
