# Eval drift finding: the `code` domain (2026-06-19)

The offline benchmark runner (`app/eval/runner.py`, target #4) replays the 18 hand-labelled
fixtures through the engine's real heuristic scorer + verdict logic. First run flagged the
`code` domain at **33% verdict accuracy** (overall 72.2%). This note diagnoses why, because the
answer decides what to fix. It was produced by an offline analysis (no model): tracing each
mislabelled case through `_score_quality` / `_check_plan_completion` / `_determine_verdict`, plus
a skeptical audit of whether the fixture *labels themselves* were defensible.

## Headline: ~half engine, ~half my fixture labels

The 33% was **not** one failure mode. Of the 4 `code` misses:

| Case | predicted → truth | root cause | owner |
|------|-------------------|------------|-------|
| `code-approve-fizzbuzz` | retry → approve | quality 0.69 vs 0.70 gate; flawless 2-step task scored 1 pt short | **engine** (scorer too coarse) |
| `code-refine-missing-edge-case` | retry → refine | a **PARTIAL** required step left `required_steps_met=True`, suppressing the REFINE path | **engine** (logic bug) |
| `code-retry-shallow-output` | refine → retry | a *not-started* required step still cleared the 0.5 REFINE gate | design question (label defensible, engine defensible) |
| `code-escalate-wrong-algorithm` | retry → escalate | runnable-but-wrong sort, quality 0.46 | **my fixture mislabel** |

## Action taken this session (safe, in-scope)

**Relabelled `code-escalate-wrong-algorithm`: `escalate` → `retry` (band low → medium).**
This corrects *my own* labeling error from earlier tonight — it is test data, not engine behavior.
Rationale: in this codebase ESCALATE is reserved for *unrecoverable* states (retries exhausted,
quality < 0.3, or empty/infra failure — cf. the correctly-labelled `code-escalate-import-error`
at quality 0.0). A runnable-but-wrong algorithm at quality 0.46 with one failed required step is
contractually a **retry**, not an escalate. My original label encoded a human "this is broken, give
up" intuition that the engine's ESCALATE does not mean.

Effect: `code` 33% → **50%**, overall 72.2% → **77.8%**. (`test_eval_cases` + `test_eval_runner`
still green: 25 passed.)

## Parked for Ian (NOT changed — alters live verdict behavior)

### Real engine bug: PARTIAL required steps in `_check_plan_completion`

`_check_plan_completion` sets `required_steps_met = False` only in the FAILED / SKIPPED /
NOT_STARTED branches. A **PARTIAL** status on a *required* step falls through all of them, so
`required_steps_met` keeps its initialized value `True`. Consequences:
- A partially-completed required step is treated as "requirements met."
- The REFINE path (`not required_steps_met and overall >= 0.5`) can never fire for partial work,
  so it collapses to RETRY (this is exactly `code-refine-missing-edge-case`).

This is domain-agnostic — it affects every task type, not just code.

**Proposed fix** (one place, conservative): make "required met" mean *all required steps are
COMPLETED*, e.g.

```python
required_met = all(
    sr.status == StepStatus.COMPLETED
    for step, sr in zip(plan_steps, step_results)
    if step.required
)
```

…or, minimally, add a PARTIAL branch alongside FAILED/SKIPPED/NOT_STARTED that sets
`required_steps_met = False` for required steps.

**Why parked:** this changes live verdict semantics and may trip existing sentinel tests in
`test_evaluation_service.py` that encode the current behavior — those need a human to confirm
whether they assert the bug or the intended contract. Expected effect once fixed: `code-refine-
missing-edge-case` resolves → `code` domain ~**67%** (4/6).

## Recommendation (prioritized)

1. **Fix the PARTIAL bug** (above). Highest value, lowest cost, helps all domains. Re-run the
   benchmark immediately after; the new number tells you whether anything else is worth doing.
2. **Leave `code-retry-shallow-output` as-is** and treat it as a design question: should a
   *not-started* required step force RETRY over REFINE regardless of quality? That's a policy call.
3. **Do NOT build #8 (per-domain thresholds) yet.** Three of the four "threshold" misses dissolve
   after step 1 + the relabel. Per-domain thresholds on a 6-case set would be overfitting.
4. **`code-approve-fizzbuzz` (0.69 vs 0.70)** is the honest signal that the heuristic can't
   recognize obvious correctness. The principled fix is #5 (per-dimension confidence + a
   confidence-gated LLM re-grade on near-threshold cases 0.65–0.75), not nudging APPROVE_QUALITY.
   Treat as P2; decide after step 1.

## Update — PARTIAL fix APPLIED (2026-06-19, post-loop, at Ian's request)

The `_check_plan_completion` PARTIAL fix is now applied: a partially-completed *required* step sets `required_steps_met=False` (parity with FAILED/SKIPPED/NOT_STARTED). All 247 eval + evaluation-engine tests pass (no sentinel-contract breakage); black-clean.

Benchmark effect — **net-neutral on the aggregate, and informative**:
- overall 77.8% → **77.8%** (unchanged)
- code 50% → **66.7%** (`code-refine-missing-edge-case` now correctly REFINE ✓)
- file_ops 100% → **83.3%** (`file_ops-retry-no-confirmation` flipped: was matching RETRY *by accident* because the bug left `required_steps_met=True`; now correctly `required_steps_met=False`, so the q≥0.5 REFINE gate fires → REFINE, which disagrees with its RETRY label)

So the fix is correct (the old match was for the wrong reason) and it **surfaced a real design question, now affecting two cases** (`code-shallow-output`, `file_ops-retry-no-confirmation`):

> **OPEN DESIGN QUESTION for Ian:** when a *required* step is incomplete (partial / not-started) but overall quality ≥ 0.5, should the verdict be REFINE (engine's current rule) or RETRY (both hand labels)? This is a policy call about the REFINE/RETRY boundary, not a bug. Not relabelling the fixtures (RETRY is defensible); flagging for decision. Resolving it (e.g. "an unmet required step forces RETRY regardless of quality") would lift both code and file_ops.

— Atlas (overnight loop, 2026-06-19)
