"""Paired-twin evaluation suite for the comprehension front gate.

Each :class:`GatePair` holds two requests that differ by a small edit: an
**act** twin (clear and feasible, so the gate should ``proceed``) and an
**abstain** twin (ambiguous, so ``clarify``; or impossible, so ``refuse``).
Scoring the twins together catches the two ways a gate fails:

* **over-refusal**: the act twin is blocked (clarify/refuse);
* **under-abstention**: the abstain twin runs the loop anyway (proceed).

A gate that always proceeds scores 50% item accuracy but 0% **paired accuracy**
(both twins right), which is why paired accuracy is the headline number.
The design follows the paired act/abstain protocol used in recent agent
abstention benchmarks, which report that models abstain much less accurately
than they act.

The predictor is injected, as in :mod:`app.evaluation.runner`. It receives the
twin's input text and its :class:`IntentFixture`, never the expected outcome:

* :func:`gate_predictor` runs the real, pure ``decide_gate`` on the fixture
  (what the LLM classifier is assumed to emit), mirroring the live comprehension
  node. Deterministic, offline, used by the tests.
* An end-to-end predictor can ignore the fixture, call the live comprehension
  agent on the text and return its gate outcome. That needs an LLM, so it is
  the caller's job.

No LLM, network or database is used here.

CLI: ``python -m app.evaluation.gate_suite [--path FILE] [--json]``
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.core.agents.comprehension_gate import GateThresholds, decide_gate
from app.core.registry.builtin import _BUILTIN, build_registry_from_dispatcher
from app.core.registry.capability import CapabilityRegistry
from app.models.core_state import UserIntent

GateLabel = Literal["proceed", "clarify", "refuse"]
OUTCOMES: Tuple[str, ...] = ("proceed", "clarify", "refuse")
ERROR = "error"  # predictor raised; always scored as wrong

DEFAULT_SUITE_PATH = (
    Path(__file__).resolve().parents[3] / "docs/evaluation/gate/paired_twins.jsonl"
)

# --------------------------------------------------------------------------
# Dataset models
# --------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IntentFixture(_Strict):
    """What the comprehension classifier is assumed to emit for this input."""

    type: Literal["task", "conversation", "question", "clarification"] = "task"
    description: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    requires_tools: bool = False
    tools_needed: List[str] = Field(default_factory=list)
    ambiguities: List[str] = Field(default_factory=list)

    def to_intent(self) -> UserIntent:
        return UserIntent(**self.model_dump())


class TwinCase(_Strict):
    input: str = Field(min_length=1)
    expected: GateLabel
    intent: IntentFixture
    note: str = ""


class GatePair(_Strict):
    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    act: TwinCase
    abstain: TwinCase
    tags: List[str] = Field(default_factory=list)
    source: Literal["seed", "fault", "manual"] = "manual"

    @model_validator(mode="after")
    def _check_labels(self) -> "GatePair":
        if self.act.expected != "proceed":
            raise ValueError("act twin must expect 'proceed'")
        if self.abstain.expected not in ("clarify", "refuse"):
            raise ValueError("abstain twin must expect 'clarify' or 'refuse'")
        if self.act.input.strip().casefold() == self.abstain.input.strip().casefold():
            raise ValueError("twins must have different inputs")
        return self


@dataclass(frozen=True)
class SkippedPair:
    line_no: int  # 1-based
    reason: str


@dataclass
class PairLoadResult:
    pairs: List[GatePair] = field(default_factory=list)
    skipped: List[SkippedPair] = field(default_factory=list)


def load_gate_pairs(path: str | Path = DEFAULT_SUITE_PATH) -> PairLoadResult:
    """Load pairs from JSONL. Bad or duplicate-id lines are reported, not raised."""
    result = PairLoadResult()
    path = Path(path)
    if not path.exists():
        return result
    seen: Dict[str, int] = {}
    for line_no, raw in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        line = raw.strip()
        if not line:
            continue
        try:
            pair = GatePair.model_validate_json(line)
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(p) for p in first["loc"]) or "<root>"
            result.skipped.append(
                SkippedPair(line_no, f"invalid: {where}: {first['msg']}")
            )
            continue
        if pair.id in seen:
            result.skipped.append(
                SkippedPair(line_no, f"duplicate id of line {seen[pair.id]}")
            )
            continue
        seen[pair.id] = line_no
        result.pairs.append(pair)
    return result


# --------------------------------------------------------------------------
# Predictors
# --------------------------------------------------------------------------

# (input_text, intent_fixture) -> "proceed" | "clarify" | "refuse"
GatePredictor = Callable[[str, IntentFixture], str]


def builtin_registry() -> CapabilityRegistry:
    """Registry as the live dispatcher builds it, with every builtin root."""
    return build_registry_from_dispatcher(SimpleNamespace(available_tools=_BUILTIN))


def gate_predictor(
    registry: Optional[CapabilityRegistry] = None,
    thresholds: Optional[GateThresholds] = None,
    *,
    merge_pattern_ambiguities: bool = True,
) -> GatePredictor:
    """Predictor that runs the real ``decide_gate`` on each twin's fixture.

    With ``merge_pattern_ambiguities`` (the default, as in the live
    ``core_graph_v2.comprehension_node``), the regex ambiguities from
    ``ComprehensionAgent.detect_ambiguities`` are merged into the intent first.
    Turning it off isolates the gate from the regex detector (an ablation).
    """
    reg = registry if registry is not None else builtin_registry()
    detect = None
    if merge_pattern_ambiguities:
        from app.core.agents.comprehension_agent import ComprehensionAgent

        detect = ComprehensionAgent().detect_ambiguities

    def predict(text: str, fixture: IntentFixture) -> str:
        intent = fixture.to_intent()
        if detect is not None:
            for amb in detect(text):
                if amb not in intent.ambiguities:
                    intent.ambiguities.append(amb)
        return decide_gate(intent, reg, thresholds, text=text).outcome

    return predict


def oracle_predictor(pairs: Iterable[GatePair]) -> GatePredictor:
    """Answers from the labels. Only for checking the scorer itself."""
    table = {}
    for p in pairs:
        table[p.act.input] = p.act.expected
        table[p.abstain.input] = p.abstain.expected
    return lambda text, _fixture: table[text]


def constant_predictor(outcome: str) -> GatePredictor:
    """Always returns ``outcome``. Baselines: always-act, always-refuse."""
    return lambda _text, _fixture: outcome


# --------------------------------------------------------------------------
# Results + metrics
# --------------------------------------------------------------------------


class TwinResult(BaseModel):
    input: str
    expected: GateLabel
    predicted: str  # an outcome, or "error"
    correct: bool
    error: Optional[str] = None


class PairResult(BaseModel):
    pair_id: str
    category: str
    tags: List[str] = Field(default_factory=list)
    act: TwinResult
    abstain: TwinResult

    @property
    def paired_correct(self) -> bool:
        return self.act.correct and self.abstain.correct


class Rates(BaseModel):
    """Metrics over a set of pairs. All rates are in [0, 1]; 0.0 when n is 0."""

    n_pairs: int
    paired_accuracy: float  # both twins exactly right
    paired_accuracy_lenient: float  # act right, abstain twin did not proceed
    item_accuracy: float
    act_accuracy: float  # act twins that proceeded
    abstain_accuracy: float  # abstain twins with the exact label
    abstain_detection: float  # abstain twins that did not proceed
    clarify_accuracy: Optional[float] = None  # None when no twin expects it
    refuse_accuracy: Optional[float] = None
    over_refusal_rate: float  # act twins blocked (= 1 - act_accuracy)
    over_clarify_rate: float  # act twins sent to clarify
    over_refuse_rate: float  # act twins refused
    under_abstention_rate: float  # abstain twins that proceeded
    abstain_label_swap_rate: float  # clarify <-> refuse mix-ups
    act_abstain_gap: float  # act_accuracy - abstain_accuracy


class GateSuiteReport(BaseModel):
    label: str = ""
    overall: Rates
    by_category: Dict[str, Rates]
    confusion: Dict[str, Dict[str, int]]  # expected -> predicted -> count
    errored: int
    pairs: List[PairResult]

    def failures(self) -> List[PairResult]:
        return [p for p in self.pairs if not p.paired_correct]


def _ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def compute_rates(results: List[PairResult]) -> Rates:
    n = len(results)
    acts = [r.act for r in results]
    abst = [r.abstain for r in results]

    act_ok = sum(t.correct for t in acts)
    over_clarify = sum(t.predicted == "clarify" for t in acts)
    over_refuse = sum(t.predicted == "refuse" for t in acts)
    abst_ok = sum(t.correct for t in abst)
    detected = sum(t.predicted in ("clarify", "refuse") for t in abst)
    proceeded = sum(t.predicted == "proceed" for t in abst)
    swapped = sum(
        t.predicted in ("clarify", "refuse") and t.predicted != t.expected for t in abst
    )

    def label_acc(label: str) -> Optional[float]:
        group = [t for t in abst if t.expected == label]
        return _ratio(sum(t.correct for t in group), len(group)) if group else None

    act_acc = _ratio(act_ok, n)
    abst_acc = _ratio(abst_ok, n)
    return Rates(
        n_pairs=n,
        paired_accuracy=_ratio(sum(r.paired_correct for r in results), n),
        paired_accuracy_lenient=_ratio(
            sum(
                r.act.correct and r.abstain.predicted in ("clarify", "refuse")
                for r in results
            ),
            n,
        ),
        item_accuracy=_ratio(act_ok + abst_ok, 2 * n),
        act_accuracy=act_acc,
        abstain_accuracy=abst_acc,
        abstain_detection=_ratio(detected, n),
        clarify_accuracy=label_acc("clarify"),
        refuse_accuracy=label_acc("refuse"),
        over_refusal_rate=_ratio(n - act_ok, n),
        over_clarify_rate=_ratio(over_clarify, n),
        over_refuse_rate=_ratio(over_refuse, n),
        under_abstention_rate=_ratio(proceeded, n),
        abstain_label_swap_rate=_ratio(swapped, n),
        act_abstain_gap=act_acc - abst_acc,
    )


def _run_twin(case: TwinCase, fn: GatePredictor) -> TwinResult:
    try:
        predicted = fn(case.input, case.intent.model_copy(deep=True))
    except Exception as exc:  # a crashing predictor is a wrong answer, not a crash
        return TwinResult(
            input=case.input,
            expected=case.expected,
            predicted=ERROR,
            correct=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    predicted = str(predicted)
    return TwinResult(
        input=case.input,
        expected=case.expected,
        predicted=predicted,
        correct=predicted == case.expected,
    )


def run_gate_suite(
    pairs: Iterable[GatePair], fn: GatePredictor, *, label: str = ""
) -> GateSuiteReport:
    results = [
        PairResult(
            pair_id=p.id,
            category=p.category,
            tags=list(p.tags),
            act=_run_twin(p.act, fn),
            abstain=_run_twin(p.abstain, fn),
        )
        for p in pairs
    ]

    by_cat: Dict[str, List[PairResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    confusion: Dict[str, Dict[str, int]] = {
        e: {p: 0 for p in (*OUTCOMES, ERROR)} for e in OUTCOMES
    }
    for r in results:
        for t in (r.act, r.abstain):
            row = confusion[t.expected]
            row[t.predicted] = row.get(t.predicted, 0) + 1

    return GateSuiteReport(
        label=label,
        overall=compute_rates(results),
        by_category={c: compute_rates(rs) for c, rs in sorted(by_cat.items())},
        confusion=confusion,
        errored=sum(t.predicted == ERROR for r in results for t in (r.act, r.abstain)),
        pairs=results,
    )


# --------------------------------------------------------------------------
# Text report + CLI
# --------------------------------------------------------------------------

_HEADLINE = (
    ("paired accuracy", "paired_accuracy"),
    ("paired accuracy (lenient)", "paired_accuracy_lenient"),
    ("act accuracy", "act_accuracy"),
    ("abstain accuracy", "abstain_accuracy"),
    ("  clarify accuracy", "clarify_accuracy"),
    ("  refuse accuracy", "refuse_accuracy"),
    ("over-refusal rate", "over_refusal_rate"),
    ("under-abstention rate", "under_abstention_rate"),
)


def _pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v * 100:5.1f}%"


def format_report(
    reports: List[GateSuiteReport],
    *,
    show_failures: bool = True,
    detail_limit: Optional[int] = None,
) -> str:
    """Markdown summary; several reports side by side (e.g. an ablation).

    Only the first ``detail_limit`` reports (all by default) get the
    per-category, confusion and failure sections; the rest, typically
    baselines, appear in the headline table only.
    """
    labels = [r.label or f"run {i + 1}" for i, r in enumerate(reports)]
    out = [
        f"## Comprehension gate: paired-twin suite ({reports[0].overall.n_pairs} pairs)",
        "",
        "| metric | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(reports),
    ]
    for name, key in _HEADLINE:
        vals = [_pct(getattr(r.overall, key)) for r in reports]
        out.append(f"| {name} | " + " | ".join(vals) + " |")

    for r, lab in list(zip(reports, labels))[:detail_limit]:
        out += ["", f"### {lab}: by category", ""]
        out += [
            "| category | pairs | paired | act | abstain |",
            "|---|---|---|---|---|",
        ]
        for cat, rates in r.by_category.items():
            out.append(
                f"| {cat} | {rates.n_pairs} | {_pct(rates.paired_accuracy)} | "
                f"{_pct(rates.act_accuracy)} | {_pct(rates.abstain_accuracy)} |"
            )
        out += ["", "confusion (rows = expected, cols = predicted):", ""]
        cols = [*OUTCOMES, ERROR]
        out += ["| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for exp, row in r.confusion.items():
            out.append(f"| {exp} | " + " | ".join(str(row[c]) for c in cols) + " |")
        if show_failures and r.failures():
            out += ["", "failures:", ""]
            for p in r.failures():
                for side, t in (("act", p.act), ("abstain", p.abstain)):
                    if not t.correct:
                        out.append(
                            f"- `{p.pair_id}` {side}: expected {t.expected}, "
                            f"got {t.predicted} — {t.input!r}"
                        )
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", default=str(DEFAULT_SUITE_PATH))
    ap.add_argument("--json", action="store_true", help="print JSON, not markdown")
    ap.add_argument("--no-failures", action="store_true")
    args = ap.parse_args(argv)

    loaded = load_gate_pairs(args.path)
    for s in loaded.skipped:
        print(f"skipped line {s.line_no}: {s.reason}", file=sys.stderr)
    if not loaded.pairs:
        print(f"no pairs loaded from {args.path}", file=sys.stderr)
        return 1

    reports = [
        run_gate_suite(loaded.pairs, gate_predictor(), label="live gate"),
        run_gate_suite(
            loaded.pairs,
            gate_predictor(merge_pattern_ambiguities=False),
            label="gate without regex ambiguities",
        ),
        run_gate_suite(loaded.pairs, constant_predictor("proceed"), label="always act"),
    ]
    if args.json:
        print(json.dumps([r.model_dump(mode="json") for r in reports], indent=2))
    else:
        print(
            format_report(reports, show_failures=not args.no_failures, detail_limit=2)
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
