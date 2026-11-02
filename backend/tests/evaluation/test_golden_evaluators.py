import pytest

from app.evaluation.evaluators import evaluate
from app.evaluation.models import (
    Contains,
    ExactMatch,
    ForbiddenSubstring,
    MaxLatency,
    Regex,
)


def ok(spec, out="", lat=None):
    return evaluate(spec, out, lat).passed


def test_exact_match():
    assert ok(ExactMatch(value="pong"), "  pong\n")
    assert not ok(ExactMatch(value="pong"), "Pong")
    assert ok(ExactMatch(value="pong", case_sensitive=False), "Pong")
    assert not ok(ExactMatch(value="a", strip=False), "a ")


def test_contains_default_case_insensitive():
    assert ok(Contains(value="Canberra"), "it is canberra.")
    assert not ok(Contains(value="Canberra", case_sensitive=True), "canberra")
    assert not ok(Contains(value="x"), "")


def test_regex_and_invalid_pattern_fails_not_raises():
    assert ok(Regex(pattern=r"\b391\b"), "that is 391.")
    assert not ok(Regex(pattern=r"\b391\b"), "3910")
    assert ok(Regex(pattern="abc", ignore_case=True), "ABC")
    bad = evaluate(Regex(pattern="("), "x", None)
    assert not bad.passed and "invalid regex" in bad.detail


def test_max_latency_boundary_and_missing():
    assert ok(MaxLatency(ms=100), lat=100.0)
    assert not ok(MaxLatency(ms=100), lat=100.1)
    missing = evaluate(MaxLatency(ms=100), "", None)
    assert not missing.passed and "unavailable" in missing.detail


def test_forbidden_substring():
    spec = ForbiddenSubstring(values=["sk-", "secret"])
    assert ok(spec, "all clear")
    out = evaluate(spec, "key is SK-123", None)
    assert not out.passed and "sk-" in out.detail
    assert ok(ForbiddenSubstring(values=["SK-"], case_sensitive=True), "sk-1")


def test_forbidden_requires_values():
    with pytest.raises(ValueError):
        ForbiddenSubstring(values=[])
