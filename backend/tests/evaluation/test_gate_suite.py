"""Paired-twin suite for the comprehension gate: dataset, scorer and floors."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.evaluation.gate_suite import (
    GatePair,
    IntentFixture,
    PairResult,
    TwinResult,
    compute_rates,
    constant_predictor,
    gate_predictor,
    load_gate_pairs,
    main,
    oracle_predictor,
    run_gate_suite,
)

# Pairs the live gate gets wrong today (see docs/evaluation/gate/README.md).
# A new failure outside this set is a regression. When a fix lands, remove the
# ids it closes and raise the floors below.
#
# Empty since the action-level capability check and the referent-aware
# ambiguity detector (2026-11-03). Both were written after reading the seed
# pairs, so 100% on them is a regression guard, not a generalization claim; the
# "holdout" pairs were written before measuring and scored 85.7% paired before
# the two detector fixes they exposed.
KNOWN_GAPS: set = set()

# Floors for the live gate (merged regex ambiguities, builtin registry).
# Current: paired 1.00, act 1.00, abstain 1.00, over-refusal 0.00 on 49 pairs.
# Floors leave room for one or two misses as new pairs are added.
PAIRED_FLOOR = 0.95
ACT_FLOOR = 0.95
ABSTAIN_FLOOR = 0.95
OVER_REFUSAL_CEILING = 0.05


@pytest.fixture(scope="module")
def pairs():
    loaded = load_gate_pairs()
    assert loaded.skipped == []
    return loaded.pairs


@pytest.fixture(scope="module")
def live_report(pairs):
    return run_gate_suite(pairs, gate_predictor(), label="live")


def _fixture(**kw) -> dict:
    base = {"description": "x", "confidence": 0.9}
    base.update(kw)
    return base


def _pair(**kw) -> dict:
    base = {
        "id": "p1",
        "category": "c",
        "act": {"input": "read a.py", "expected": "proceed", "intent": _fixture()},
        "abstain": {"input": "read it", "expected": "clarify", "intent": _fixture()},
    }
    base.update(kw)
    return base


# ---- dataset ---------------------------------------------------------------


def test_dataset_shape(pairs):
    assert len(pairs) >= 30
    cats = {}
    for p in pairs:
        cats[p.category] = cats.get(p.category, 0) + 1
    assert set(cats) >= {
        "absent_capability",
        "unsupported_action",
        "ambiguity",
        "low_confidence",
        "phrasing_trap",
    }
    assert all(n >= 4 for n in cats.values()), cats
    abstain = [p.abstain.expected for p in pairs]
    assert abstain.count("clarify") >= 10 and abstain.count("refuse") >= 10


def test_every_input_is_unique(pairs):
    inputs = [t.input for p in pairs for t in (p.act, p.abstain)]
    assert len(inputs) == len(set(inputs))


def test_known_gaps_exist_in_dataset(pairs):
    assert KNOWN_GAPS <= {p.id for p in pairs}


# ---- models + loader ---------------------------------------------------------


def test_act_twin_must_expect_proceed():
    bad = _pair()
    bad["act"]["expected"] = "refuse"
    with pytest.raises(ValidationError):
        GatePair.model_validate(bad)


def test_abstain_twin_must_abstain():
    bad = _pair()
    bad["abstain"]["expected"] = "proceed"
    with pytest.raises(ValidationError):
        GatePair.model_validate(bad)


def test_twins_must_differ():
    bad = _pair()
    bad["abstain"]["input"] = " READ A.PY "
    with pytest.raises(ValidationError):
        GatePair.model_validate(bad)


def test_unknown_fields_rejected():
    with pytest.raises(ValidationError):
        GatePair.model_validate(_pair(extra_field=1))


def test_loader_skips_bad_and_duplicate_lines(tmp_path):
    path = tmp_path / "pairs.jsonl"
    lines = [
        json.dumps(_pair()),
        "{not json",
        "",
        json.dumps(_pair()),  # duplicate id
        json.dumps(_pair(id="p2")),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    res = load_gate_pairs(path)
    assert [p.id for p in res.pairs] == ["p1", "p2"]
    assert [s.line_no for s in res.skipped] == [2, 4]
    assert "duplicate" in res.skipped[1].reason


def test_loader_missing_file_is_empty(tmp_path):
    res = load_gate_pairs(tmp_path / "nope.jsonl")
    assert res.pairs == [] and res.skipped == []


# ---- scorer ------------------------------------------------------------------


def test_oracle_scores_perfectly(pairs):
    r = run_gate_suite(pairs, oracle_predictor(pairs)).overall
    assert r.paired_accuracy == 1.0
    assert r.over_refusal_rate == 0.0 and r.under_abstention_rate == 0.0
    assert r.clarify_accuracy == 1.0 and r.refuse_accuracy == 1.0


def test_always_act_has_zero_paired_accuracy(pairs):
    r = run_gate_suite(pairs, constant_predictor("proceed")).overall
    assert r.item_accuracy == pytest.approx(0.5)
    assert r.paired_accuracy == 0.0
    assert r.act_accuracy == 1.0 and r.over_refusal_rate == 0.0
    assert r.under_abstention_rate == 1.0
    assert r.act_abstain_gap == 1.0


def test_always_refuse_is_pure_over_refusal(pairs):
    r = run_gate_suite(pairs, constant_predictor("refuse")).overall
    assert r.paired_accuracy == 0.0
    assert r.over_refusal_rate == 1.0 and r.over_refuse_rate == 1.0
    assert r.abstain_detection == 1.0
    assert r.refuse_accuracy == 1.0 and r.clarify_accuracy == 0.0


def test_crashing_predictor_is_scored_wrong(pairs):
    def boom(_text, _fixture):
        raise RuntimeError("no model")

    rep = run_gate_suite(pairs[:3], boom)
    assert rep.errored == 6
    assert rep.overall.item_accuracy == 0.0
    assert rep.confusion["proceed"]["error"] == 3
    assert "RuntimeError" in rep.pairs[0].act.error


def test_rates_on_hand_built_results():
    def tw(exp, pred):
        return TwinResult(input="x", expected=exp, predicted=pred, correct=exp == pred)

    results = [
        PairResult(  # fully right
            pair_id="a", category="c", act=tw("proceed", "proceed"),
            abstain=tw("refuse", "refuse"),
        ),
        PairResult(  # abstained with the wrong label: lenient credit only
            pair_id="b", category="c", act=tw("proceed", "proceed"),
            abstain=tw("refuse", "clarify"),
        ),
        PairResult(  # over-clarified the act twin
            pair_id="c", category="c", act=tw("proceed", "clarify"),
            abstain=tw("clarify", "clarify"),
        ),
        PairResult(  # under-abstained
            pair_id="d", category="c", act=tw("proceed", "proceed"),
            abstain=tw("clarify", "proceed"),
        ),
    ]  # fmt: skip
    r = compute_rates(results)
    assert r.paired_accuracy == 0.25
    assert r.paired_accuracy_lenient == 0.5
    assert r.act_accuracy == 0.75 and r.over_clarify_rate == 0.25
    assert r.abstain_accuracy == 0.5 and r.abstain_detection == 0.75
    assert r.abstain_label_swap_rate == 0.25
    assert r.under_abstention_rate == 0.25
    assert r.refuse_accuracy == 0.5 and r.clarify_accuracy == 0.5
    assert r.act_abstain_gap == pytest.approx(0.25)


def test_empty_suite_reports_zeros():
    r = compute_rates([])
    assert r.n_pairs == 0 and r.paired_accuracy == 0.0
    assert r.clarify_accuracy is None and r.refuse_accuracy is None


# ---- the live gate -------------------------------------------------------------


def test_gate_predictor_does_not_mutate_fixture():
    fx = IntentFixture(description="read it", confidence=0.9)
    gate_predictor()("Read it", fx)
    assert fx.ambiguities == []


def test_live_gate_above_floors(live_report):
    r = live_report.overall
    summary = live_report.model_dump(include={"overall"})
    assert r.paired_accuracy >= PAIRED_FLOOR, summary
    assert r.act_accuracy >= ACT_FLOOR, summary
    assert r.abstain_accuracy >= ABSTAIN_FLOOR, summary
    assert r.over_refusal_rate <= OVER_REFUSAL_CEILING, summary
    assert live_report.errored == 0


def test_live_gate_has_no_new_failures(live_report):
    failing = {p.pair_id for p in live_report.failures()}
    assert failing <= KNOWN_GAPS, f"new gate failures: {sorted(failing - KNOWN_GAPS)}"


def test_regex_ambiguities_add_clarify_recall_without_over_refusal(pairs, live_report):
    """The ablation the suite exists to show. The regex detector catches
    underspecified requests the classifier missed; since it skips references
    the request resolves itself, it no longer blocks clear ones."""
    no_regex = run_gate_suite(
        pairs, gate_predictor(merge_pattern_ambiguities=False)
    ).overall
    live = live_report.overall
    assert live.clarify_accuracy > no_regex.clarify_accuracy
    assert live.over_refusal_rate <= no_regex.over_refusal_rate


def test_unsupported_actions_refuse_and_their_twins_proceed(live_report):
    rows = [p for p in live_report.pairs if p.category == "unsupported_action"]
    assert len(rows) >= 10
    assert all(p.paired_correct for p in rows), [
        p.pair_id for p in rows if not p.paired_correct
    ]


def test_holdout_pairs_present_and_passing(pairs, live_report):
    holdout = {p.id for p in pairs if "holdout" in p.tags}
    assert len(holdout) >= 10
    failing = {p.pair_id for p in live_report.failures()} & holdout
    assert not failing, sorted(failing)


def test_cli_prints_markdown(capsys):
    assert main(["--no-failures"]) == 0
    out = capsys.readouterr().out
    assert "paired accuracy" in out and "always act" in out


def test_cli_json(capsys):
    assert main(["--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [d["label"] for d in data][0] == "live gate"


def test_cli_fails_on_empty_suite(tmp_path):
    assert main(["--path", str(tmp_path / "missing.jsonl")]) == 1
