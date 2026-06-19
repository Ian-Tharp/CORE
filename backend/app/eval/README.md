# `app.eval` — Offline evaluation harness

An **offline drift / benchmark harness** for the CORE evaluation engine. It
replays hand-labelled fixtures through the engine's *real* heuristic scorer and
verdict logic and reports how often the engine's verdict matches the human
ground truth — per domain and overall.

## Purpose

The evaluation layer (`app.services.evaluation_service`) decides whether a
task's output should be approved, retried, refined, or escalated. Before this
package existed, that layer was **blind to its own drift**: there was no way to
tell whether a change to the scorer or the thresholds quietly moved verdicts in
the wrong direction. The eval engine graded everyone else but never graded
itself.

This harness closes that gap. It pins the engine's behavior against a small,
hand-labelled corpus and surfaces every disagreement as a number you can watch
over time. The first run already paid for itself: it caught a real logic bug in
`_check_plan_completion` and a coarseness problem in the quality scorer (see
[Current results](#current-results) and the FINDINGS note).

## How it stays offline

The runner drives the engine's **real heuristic callables** directly:

- `_score_quality` — async, but contains only synchronous heuristics (no IO),
  so the runner wraps it in `asyncio.run` to stay synchronous.
- `_check_plan_completion` — sync, pure.
- `_determine_verdict` — sync, pure; the REAL thresholds and verdict rules.

It deliberately **bypasses `evaluation_service.evaluate()`**, which would
persist results to the DB and write procedural memory. By calling the three
pure building blocks in sequence instead, the harness exercises the same path
`evaluate()` takes — minus the DB / LLM / memory writes — so thresholds and
verdict rules are never reimplemented or mocked here. No LLM, no DB, no network.

**Leaf-import discipline.** `import app.eval` (and each submodule) must stay
cheap and IO-free. The package re-exports only the two leaf contracts
(`Verifier`, `VerifierResult`); it does **not** re-export the runner or cases.
Every `app.models.*` / `app.services.*` symbol is imported *lazily inside*
`run_benchmark` (`_predict_case`), never at module scope, so importing the
package never transitively loads the evaluation service → repository → DB driver
chain.

## Components

| File | What it holds |
|------|---------------|
| `protocol.py` | `Verifier` (a `@runtime_checkable` structural Protocol) and `VerifierResult` (a frozen, slotted dataclass value object). Dependency-free leaf — stdlib + typing only. |
| `cases.py` | `BenchmarkCase` (a Pydantic model, `extra="forbid"`), the `QualityBand` enum (`low` / `medium` / `high`), `VALID_VERDICTS`, and `load_cases(directory)`. Leaf — stdlib + pydantic only. |
| `runner.py` | `run_benchmark(cases) -> report` plus the `python -m app.eval.runner` CLI. Maps each case onto an `EvaluationInput` and runs the real scorer + verdict callables. |
| `tests/fixtures/eval_cases/*.json` | The corpus: 18 hand-labelled cases as JSON arrays, one file per domain (`code.json`, `research.json`, `file_ops.json` — 6 cases each). |

> Fixtures live under `backend/tests/fixtures/eval_cases/`, not inside this
> package. The runner resolves that path relative to its own location so the
> CLI works from any working directory.

## Quickstart

From `backend/`:

```bash
uv run python -m app.eval.runner --mode heuristic
```

`--mode` only accepts `heuristic` (the offline path). `--fixtures <dir>`
overrides the default fixtures directory. The CLI prints the full report as
indented, key-sorted JSON. Trimmed shape:

```json
{
  "overall": {
    "n": 18,
    "verdict_accuracy": 0.7777777777777778,
    "false_approve_rate": 0.05555555555555555,
    "false_reject_rate": 0.05555555555555555,
    "confusion": { "approve->retry": 1, "refine->approve": 1, "...": 1 },
    "predicted_counts": { "approve": 6, "escalate": 5, "...": 0 },
    "ground_truth_counts": { "approve": 6, "escalate": 5, "...": 0 }
  },
  "by_domain": {
    "code":     { "n": 6, "verdict_accuracy": 0.5, "...": 0 },
    "file_ops": { "n": 6, "verdict_accuracy": 1.0, "...": 0 },
    "research": { "n": 6, "verdict_accuracy": 0.8333333333333334, "...": 0 }
  },
  "cases": [
    {
      "id": "code-approve-fizzbuzz",
      "domain": "code",
      "predicted_verdict": "retry",
      "ground_truth_verdict": "approve",
      "correct": false,
      "quality_overall": 0.69,
      "completion_rate": 1.0,
      "required_steps_met": true
    }
  ]
}
```

`approve` is the only verdict treated as an approval for the safety rates:
**false_approve** = predicted `approve` but truth ≠ `approve` (waved through bad
work); **false_reject** = truth `approve` but predicted ≠ `approve` (blocked
good work). All `*_rate` and `verdict_accuracy` values are in `[0.0, 1.0]`.

## Current results

Live run (`--mode heuristic`, 18 cases):

| Scope | verdict_accuracy |
|-------|------------------|
| **overall** | **0.778** (14/18) |
| `file_ops` | 1.000 (6/6) |
| `research` | 0.833 (5/6) |
| `code` | 0.500 (3/6) |

The **`code` domain is the weak spot**. Of its three misses, roughly half are
engine and half were fixture-label noise that has since been corrected. The
remaining engine causes are a too-coarse quality scorer (`code-approve-fizzbuzz`
lands at 0.69 against the 0.70 approve gate) and a PARTIAL-step bug in
`_check_plan_completion` that suppresses the REFINE path. The full diagnosis —
case-by-case root causes and ownership — is in
[`FINDINGS_code_drift_2026-06-19.md`](./FINDINGS_code_drift_2026-06-19.md).

## Add a benchmark case

Cases live in `backend/tests/fixtures/eval_cases/`, one JSON **array** per
domain file. Append a new object to the matching file. `BenchmarkCase` fields
(all required unless noted; `extra="forbid"`, so no stray keys):

| Field | Type | Notes |
|-------|------|-------|
| `id` | str | Stable, globally unique. |
| `domain` | str | One of `code` \| `research` \| `file_ops`. |
| `intent` | str | The original user intent / task description. |
| `plan_steps` | list[dict] | Raw `PlanStep` dicts (`description`, `expected_output?`, `required?`). `step_index` is optional — the runner defaults it to position. Defaults to `[]`. |
| `step_results` | list[dict] | Raw `StepResult` dicts (`step_index`, `status`, `output?`, `error?`, `duration_ms?`). Defaults to `[]`. |
| `final_output` | str | The reasoning phase's final output; may be empty to model a broken run. |
| `ground_truth_verdict` | str | Human label: `approve` \| `retry` \| `refine` \| `escalate`. |
| `expected_quality_band` | str | `low` \| `medium` \| `high` (`QualityBand`). |

`plan_steps` / `step_results` are kept as raw dicts so fixtures stay plain JSON;
the runner narrows them to the engine's `PlanStep` / `StepResult` models lazily.
`load_cases` validates every entry on load (it raises if a file is not a JSON
array or any case fails validation), and `test_eval_cases.py` asserts the corpus
holds **≥ 15** cases — keep that floor.

## Implement a Verifier

`Verifier` is a structural Protocol (`@runtime_checkable`). Any class with a
matching method satisfies it — no subclassing required. The contract is a single
**sync, pure** method:

```python
def verify(self, case: BenchmarkCase, actual: Any) -> VerifierResult: ...
```

- **Sync and pure:** same inputs → same result, no IO, no model loads. That is
  what lets the runner be reproducible and loop without an event loop.
- `actual` is intentionally `Any`; each verifier narrows it to the type it
  expects (e.g. a predicted verdict / `EvaluationResult`, or a raw
  `final_output` string) and documents that narrowing.
- Return a `VerifierResult(passed, score, detail, metrics={})`. `score` must be
  in `[0.0, 1.0]` (the dataclass raises on construction otherwise). Do **not**
  raise just because the output was bad — that is `passed=False`; raise only on
  genuinely malformed inputs.

## Known issues / next steps

See [`FINDINGS_code_drift_2026-06-19.md`](./FINDINGS_code_drift_2026-06-19.md)
for the full diagnosis. Highlights:

- **Parked engine bug (Ian decision):** `_check_plan_completion` never sets
  `required_steps_met = False` for a **PARTIAL** required step (only
  FAILED / SKIPPED / NOT_STARTED branches do), so partial work is treated as
  "requirements met" and the REFINE path can't fire. Domain-agnostic. The fix
  changes live verdict semantics and may trip sentinel tests in
  `test_evaluation_service.py`, so it is **parked for a human** rather than
  applied here.
- **#5 (per-dimension confidence) is the principled fix** for scorer coarseness
  — per-dimension confidence plus a confidence-gated LLM re-grade on
  near-threshold cases (0.65–0.75), rather than nudging the approve threshold to
  rescue `code-approve-fizzbuzz`.
- **#8 (per-domain thresholds) is NOT recommended yet.** Most of the "threshold"
  misses dissolve after the PARTIAL fix + the relabel; tuning per-domain
  thresholds on a 6-case-per-domain set would be overfitting.

## Tests

| File | Covers |
|------|--------|
| `tests/test_eval_protocol.py` | `Verifier` Protocol + `VerifierResult` (bounds, immutability, structural satisfiability). |
| `tests/test_eval_cases.py` | `BenchmarkCase`, `load_cases`, `QualityBand`; corpus sentinels (≥ 15 cases, valid verdicts/bands/domains, unique ids). |
| `tests/test_eval_runner.py` | `run_benchmark` report structure & invariants (rates in `[0,1]`, per-domain `n` sums to total, 18 cases). |

Run them from `backend/`:

```bash
uv run python -m pytest tests/test_eval_protocol.py tests/test_eval_cases.py tests/test_eval_runner.py -q
```
