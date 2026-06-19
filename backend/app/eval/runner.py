"""
Offline benchmark runner for the CORE evaluation engine.

This module replays hand-labelled :class:`~app.eval.cases.BenchmarkCase`
fixtures through the engine's **real** heuristic scorer and verdict logic and
reports how often the predicted verdict matches the human ground truth. It is
the first drift signal for the evaluation layer: where the heuristic and the
hand labels disagree, that gap is surfaced here rather than hidden.

Strictly offline by construction:

- **No LLM.** The verdict comes from the engine's heuristic path
  (``_score_quality`` is async but contains only synchronous heuristics).
- **No DB.** We deliberately bypass :func:`app.services.evaluation_service.evaluate`
  (which persists results and writes procedural memory) and call the three pure
  building blocks directly: ``_score_quality`` → ``_check_plan_completion`` →
  ``_determine_verdict``. These are the REAL engine callables, so thresholds and
  verdict rules are not reimplemented here.
- **No network.**

Leaf-import discipline: every ``app.models.*`` / ``app.services.*`` symbol is
imported *lazily inside* :func:`run_benchmark`, never at module scope. Importing
``app.eval.runner`` therefore stays cheap and IO-free and never transitively
loads ``app.services.evaluation_service`` → ``app.repository`` → the DB driver
chain. The CLI (:func:`main`) and the case loader are the only module-level
imports, and the loader is itself a dependency-light leaf.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from uuid import NAMESPACE_URL, uuid5

from app.eval.cases import BenchmarkCase, load_cases

# Default fixtures live alongside the backend test suite. Resolved relative to
# this file so the CLI works regardless of the current working directory.
#   app/eval/runner.py -> app/ -> backend/  (parents[2])
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURES_DIR = _BACKEND_ROOT / "tests" / "fixtures" / "eval_cases"

# The single verdict we treat as "approve" for the false-approve / false-reject
# accounting. Kept as a literal string (the engine's ``Verdict.APPROVE.value``)
# so this module needs no runtime import of the Verdict enum.
_APPROVE = "approve"


# =============================================================================
# CORE: case -> predicted verdict via the REAL engine callables
# =============================================================================


def _predict_case(case: BenchmarkCase) -> Dict[str, Any]:
    """Run one benchmark case through the real heuristic scorer + verdict.

    All engine imports are performed lazily here (leaf-import discipline). The
    case's raw ``plan_steps`` / ``step_results`` dicts are narrowed to the
    engine's ``PlanStep`` / ``StepResult`` models, an ``EvaluationInput`` is
    constructed to validate the mapping, and then the three pure building
    blocks are called in sequence — exactly the path
    :func:`evaluation_service.evaluate` takes, minus the DB / memory writes.

    Args:
        case: The hand-labelled benchmark case to evaluate.

    Returns:
        A dict with the predicted verdict and the underlying signals:
        ``{"predicted_verdict", "ground_truth_verdict", "quality_overall",
        "completion_rate", "required_steps_met"}``.
    """
    # ── Lazy engine imports (never at module scope) ──────────────────────
    from app.models.evaluation_models import (
        EvaluationInput,
        PlanStep,
        StepResult,
    )
    from app.services.evaluation_service import (
        _check_plan_completion,
        _determine_verdict,
        _score_quality,
    )

    # Narrow raw fixture dicts onto the real engine models.
    plan_steps: List[PlanStep] = [
        PlanStep.model_validate(_with_step_index(i, raw))
        for i, raw in enumerate(case.plan_steps)
    ]
    step_results: List[StepResult] = [
        StepResult.model_validate(raw) for raw in case.step_results
    ]

    # Build an EvaluationInput to validate the full mapping. task_id must be a
    # UUID; fixtures use stable slug ids, so derive a deterministic UUID from
    # the slug (uuid5) rather than requiring fixtures to carry UUIDs.
    evaluation_input = EvaluationInput(
        task_id=uuid5(NAMESPACE_URL, case.id),
        original_intent=case.intent,
        plan_steps=plan_steps,
        step_results=step_results,
        final_output=case.final_output,
        retry_count=0,
        max_retries=3,
    )

    # 1. Score quality. _score_quality is async but purely heuristic (no IO);
    #    asyncio.run keeps the runner / CLI synchronous.
    quality = asyncio.run(
        _score_quality(
            original_intent=evaluation_input.original_intent,
            final_output=evaluation_input.final_output,
            plan_steps=evaluation_input.plan_steps,
            step_results=evaluation_input.step_results,
        )
    )

    # 2. Check plan completion (sync, pure).
    plan_completion = _check_plan_completion(
        plan_steps=evaluation_input.plan_steps,
        step_results=evaluation_input.step_results,
    )

    # 3. Determine verdict (sync, pure) — the REAL thresholds/verdict logic.
    verdict_obj = _determine_verdict(
        quality=quality,
        plan_completion=plan_completion,
        retry_count=evaluation_input.retry_count,
        max_retries=evaluation_input.max_retries,
    )

    return {
        "predicted_verdict": verdict_obj.verdict.value,
        "ground_truth_verdict": case.ground_truth_verdict,
        "quality_overall": quality.overall,
        "completion_rate": plan_completion.completion_rate,
        "required_steps_met": plan_completion.required_steps_met,
    }


def _with_step_index(index: int, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure a plan-step dict carries a ``step_index``.

    Fixture plan steps are authored without an explicit ``step_index`` (they are
    positional), but the engine's ``PlanStep`` requires one and ``StepResult``
    rows reference steps by index. Default the index to the step's position when
    absent, while honoring an explicit value if a fixture provides one.
    """
    if "step_index" in raw:
        return raw
    return {**raw, "step_index": index}


# =============================================================================
# AGGREGATION
# =============================================================================


def _empty_domain_stats() -> Dict[str, Any]:
    """Zeroed per-domain accumulator with the full confusion structure."""
    return {
        "n": 0,
        "correct": 0,
        "false_approve": 0,
        "false_reject": 0,
        # Confusion: count of (predicted -> ground_truth) where they differ,
        # plus a per-verdict predicted/ground-truth tally for inspection.
        "confusion": {},
        "predicted_counts": {},
        "ground_truth_counts": {},
    }


def _finalize_domain(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Turn raw counts into the reported, rate-bearing domain block."""
    n = stats["n"]

    def _rate(num: int) -> float:
        return (num / n) if n else 0.0

    return {
        "n": n,
        "verdict_accuracy": _rate(stats["correct"]),
        "false_approve_rate": _rate(stats["false_approve"]),
        "false_reject_rate": _rate(stats["false_reject"]),
        "confusion": dict(sorted(stats["confusion"].items())),
        "predicted_counts": dict(sorted(stats["predicted_counts"].items())),
        "ground_truth_counts": dict(sorted(stats["ground_truth_counts"].items())),
    }


def run_benchmark(cases: List[BenchmarkCase]) -> Dict[str, Any]:
    """Replay ``cases`` through the real heuristic engine and aggregate results.

    For each case the predicted verdict is computed via the engine's real
    ``_score_quality`` → ``_check_plan_completion`` → ``_determine_verdict``
    callables and compared against ``case.ground_truth_verdict``.

    "Approve" is the only verdict treated as an approval for the safety-oriented
    rates:

    - **false_approve**: predicted == approve but ground truth != approve
      (the engine waved through work a human would not have).
    - **false_reject**: ground truth == approve but predicted != approve
      (the engine blocked work a human would have approved).

    Args:
        cases: The benchmark cases to evaluate. May span multiple domains.

    Returns:
        A report dict::

            {
              "overall": {n, verdict_accuracy, false_approve_rate,
                          false_reject_rate, confusion, predicted_counts,
                          ground_truth_counts},
              "by_domain": {"<domain>": {<same shape as overall>}, ...},
              "cases": [ {id, domain, predicted_verdict,
                          ground_truth_verdict, correct, quality_overall,
                          completion_rate, required_steps_met}, ... ],
            }

        All ``*_rate`` and ``verdict_accuracy`` values are in ``[0.0, 1.0]``.
    """
    overall = _empty_domain_stats()
    by_domain: Dict[str, Dict[str, Any]] = {}
    case_rows: List[Dict[str, Any]] = []

    for case in cases:
        prediction = _predict_case(case)
        predicted = prediction["predicted_verdict"]
        ground_truth = prediction["ground_truth_verdict"]
        correct = predicted == ground_truth

        domain_stats = by_domain.setdefault(case.domain, _empty_domain_stats())

        for stats in (overall, domain_stats):
            stats["n"] += 1
            if correct:
                stats["correct"] += 1
            if predicted == _APPROVE and ground_truth != _APPROVE:
                stats["false_approve"] += 1
            if ground_truth == _APPROVE and predicted != _APPROVE:
                stats["false_reject"] += 1
            stats["predicted_counts"][predicted] = (
                stats["predicted_counts"].get(predicted, 0) + 1
            )
            stats["ground_truth_counts"][ground_truth] = (
                stats["ground_truth_counts"].get(ground_truth, 0) + 1
            )
            if not correct:
                key = f"{ground_truth}->{predicted}"
                stats["confusion"][key] = stats["confusion"].get(key, 0) + 1

        case_rows.append(
            {
                "id": case.id,
                "domain": case.domain,
                "predicted_verdict": predicted,
                "ground_truth_verdict": ground_truth,
                "correct": correct,
                "quality_overall": round(prediction["quality_overall"], 4),
                "completion_rate": round(prediction["completion_rate"], 4),
                "required_steps_met": prediction["required_steps_met"],
            }
        )

    return {
        "overall": _finalize_domain(overall),
        "by_domain": {
            domain: _finalize_domain(stats)
            for domain, stats in sorted(by_domain.items())
        },
        "cases": case_rows,
    }


# =============================================================================
# CLI
# =============================================================================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.runner",
        description=(
            "Offline benchmark runner: replays hand-labelled eval fixtures "
            "through the real heuristic scorer + verdict (no LLM, DB, or "
            "network) and prints an accuracy/drift report as JSON."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["heuristic"],
        default="heuristic",
        help=(
            "Scoring mode. Only 'heuristic' is supported offline; it calls the "
            "engine's pure scorer + verdict callables."
        ),
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=DEFAULT_FIXTURES_DIR,
        help=(
            "Directory of *.json benchmark fixtures "
            f"(default: {DEFAULT_FIXTURES_DIR})."
        ),
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    """CLI entry point.

    Usage::

        python -m app.eval.runner --mode heuristic [--fixtures <dir>]

    Loads the fixtures, runs the benchmark, and prints the report as indented
    JSON. Returns a process exit code (``0`` on success).
    """
    args = _build_parser().parse_args(argv)

    cases = load_cases(args.fixtures)
    report = run_benchmark(cases)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via the CLI
    sys.exit(main())
