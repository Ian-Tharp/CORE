"""
Benchmark cases for the offline evaluation layer.

This module defines :class:`BenchmarkCase` — the hand-labelled fixture record
the offline runner replays against the evaluation engine — and
:func:`load_cases`, a small filesystem loader that reads and validates a
directory of case files.

Import discipline mirrors the rest of ``app.eval``: this module stays a
dependency-light *leaf*. Top-level imports are limited to the stdlib and
``pydantic``. There are **no** ``app.models.*`` / ``app.services.*`` imports
at module scope and no DB / model / network access on import. The runner is
responsible for mapping a :class:`BenchmarkCase` onto the engine's
``EvaluationInput`` (importing those models lazily inside its own module), so
that ``import app.eval.cases`` never transitively loads the evaluation service
or its repository → DB driver chain.

A :class:`BenchmarkCase` is intentionally looser than ``EvaluationInput``:
``plan_steps`` / ``step_results`` are kept as raw ``list[dict]`` so fixtures
can be authored as plain JSON without importing the engine's ``PlanStep`` /
``StepResult`` models here. The runner narrows them to the real models when it
constructs an ``EvaluationInput``.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Union

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# ENUMS
# =============================================================================


class QualityBand(str, Enum):
    """Coarse expected-quality band for a benchmark case.

    Maps onto the engine's overall-quality thresholds (see
    ``EvaluationThresholds``): ``LOW`` is below the retry floor (always
    escalates), ``MEDIUM`` is the retry/refine middle ground, and ``HIGH`` is
    at or above the approve threshold.
    """

    LOW = "low"  # overall < 0.3  → escalate territory
    MEDIUM = "medium"  # 0.3 <= overall < 0.7 → retry / refine territory
    HIGH = "high"  # overall >= 0.7 → approve territory (if plan complete)


# Valid ground-truth verdict strings. Mirrors the engine's ``Verdict`` enum
# values exactly, but is defined locally so this leaf module does not import
# ``app.models.evaluation_models``. Kept as a frozenset for cheap membership
# checks and so the test suite can assert against a single source of truth.
VALID_VERDICTS: "frozenset[str]" = frozenset({"approve", "retry", "refine", "escalate"})


# =============================================================================
# CASE MODEL
# =============================================================================


class BenchmarkCase(BaseModel):
    """A single hand-labelled offline evaluation fixture.

    Each case captures the inputs the evaluation engine would receive for one
    task plus the *ground truth* a human assigned to it. The offline runner
    feeds the inputs through the engine and compares the engine's verdict /
    quality against ``ground_truth_verdict`` and ``expected_quality_band``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Stable, unique identifier for the case")
    domain: str = Field(
        ...,
        description="Task domain: one of 'code', 'research', 'file_ops'",
        pattern="^(code|research|file_ops)$",
    )
    intent: str = Field(..., description="The original user intent / task description")
    plan_steps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Raw planned steps. Each dict matches PlanStep "
            "(description, expected_output?, required?); the runner narrows "
            "these to PlanStep models lazily."
        ),
    )
    step_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Raw step results. Each dict matches StepResult "
            "(step_index, status, output?, error?, duration_ms?); the runner "
            "narrows these to StepResult models lazily."
        ),
    )
    final_output: str = Field(
        ...,
        description=(
            "The final output produced by the reasoning phase. May be empty "
            "to model a broken / no-output run."
        ),
    )
    ground_truth_verdict: str = Field(
        ...,
        description=(
            "Human-labelled correct verdict. One of: approve, retry, refine, "
            "escalate (mirrors Verdict enum values)."
        ),
        pattern="^(approve|retry|refine|escalate)$",
    )
    expected_quality_band: QualityBand = Field(
        ...,
        description="Coarse expected overall-quality band: low | medium | high",
    )


# =============================================================================
# LOADER
# =============================================================================


def load_cases(directory: Union[str, Path]) -> List[BenchmarkCase]:
    """Load and validate every ``*.json`` benchmark file in ``directory``.

    Each file must contain a JSON *array* of case objects; every object is
    validated into a :class:`BenchmarkCase`. Files are read in sorted name
    order so the resulting case list is deterministic across platforms.

    Args:
        directory: Path to the directory holding the ``*.json`` fixture files.

    Returns:
        A flat list of validated :class:`BenchmarkCase` objects, concatenated
        across files in sorted filename order.

    Raises:
        FileNotFoundError: If ``directory`` does not exist or is not a
            directory.
        ValueError: If any file does not contain a JSON array.
        pydantic.ValidationError: If any case object fails validation.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Not a directory: {dir_path}")

    cases: List[BenchmarkCase] = []
    for json_path in sorted(dir_path.glob("*.json")):
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(
                f"Expected a JSON array of cases in {json_path.name}, "
                f"got {type(raw).__name__}"
            )
        for entry in raw:
            cases.append(BenchmarkCase.model_validate(entry))
    return cases
