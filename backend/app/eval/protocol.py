"""
Verifier contracts for the offline evaluation layer.

This module is a dependency-free *leaf*: it defines the structural
``Verifier`` protocol and the ``VerifierResult`` value object and imports
nothing heavy. Top-level imports are limited to ``dataclasses`` and
``typing``; there are no model loads, no DB access, and no ``app.services.*``
imports at module scope. The offline runner and concrete verifiers import
the existing pure scorer functions inside their own modules — never here.

Forward type references (e.g. ``BenchmarkCase``) are guarded behind
``typing.TYPE_CHECKING`` so they cost nothing at runtime and can never
create an import cycle with ``app.eval.cases``. ``from __future__ import
annotations`` ensures every annotation is stored as a string and is never
evaluated on import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Protocol, runtime_checkable

if TYPE_CHECKING:
    # Typing-only import. Never executed at runtime, so there is no import
    # cycle with ``app.eval.cases`` and no cost at module import time.
    from app.eval.cases import BenchmarkCase


@dataclass(frozen=True, slots=True)
class VerifierResult:
    """Outcome of a single verification.

    A small, immutable value object produced by a :class:`Verifier`.
    Deliberately a stdlib dataclass rather than a Pydantic model: it never
    crosses an API or DB boundary (it is an in-process eval artifact), so we
    keep it dependency-free, immutable, and cheap to construct in tight
    runner loops over many benchmark cases.

    Attributes:
        passed: Boolean verdict — did ``actual`` satisfy this verifier?
        score: Continuous quality in ``[0.0, 1.0]``. For a binary validator
            this is typically ``1.0`` on pass / ``0.0`` on fail; for graded
            verifiers it is the heuristic quality (e.g.
            ``QualityScore.overall``). Mirrors the engine's own ``ge=0.0,
            le=1.0`` scoring scale.
        detail: Human-readable, single-paragraph explanation of the outcome.
            Always populated, including on pass (e.g. ``"verdict matched:
            approve"``).
        metrics: Optional bag of extra numeric/structured signals (e.g.
            ``{"predicted_verdict": "retry", "expected_verdict": "approve",
            "quality_overall": 0.62}``). Free-form; consumers must treat it as
            optional and not assume any particular keys are present.
    """

    passed: bool
    score: float
    detail: str
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the score invariant.

        Cheap, pure, IO-free. Raising (rather than clamping) keeps verifier
        bugs loud: a score outside ``[0.0, 1.0]`` indicates the verifier
        produced an out-of-scale value, which would silently corrupt
        aggregate metrics downstream.
        """
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0.0, 1.0], got {self.score!r}")


@runtime_checkable
class Verifier(Protocol):
    """Structural contract for an offline verifier.

    A ``Verifier`` inspects a benchmark ``case`` and an ``actual`` produced
    output and returns a deterministic :class:`VerifierResult`. Verifiers are
    **sync** and **pure**: given the same inputs they must return the same
    result, perform no IO, and load no models. That is what lets the offline
    runner be reproducible and execute many cases without an event loop.

    Implementations are matched structurally (duck typing) — a class need not
    subclass ``Verifier``; it only needs a matching ``verify`` method.
    Decorated ``@runtime_checkable`` so a registry or validation step can use
    ``isinstance(obj, Verifier)`` while the contract itself stays
    import-light.
    """

    def verify(self, case: "BenchmarkCase", actual: Any) -> VerifierResult:
        """Score ``actual`` against the expectations encoded in ``case``.

        Args:
            case: The benchmark case under test. Typed via a ``TYPE_CHECKING``
                forward reference to avoid importing ``app.eval.cases`` at
                runtime. Concrete verifiers read whichever fields they need
                (intent, plan_steps, ground_truth_verdict,
                expected_quality_band, ...).
            actual: The output being judged. Intentionally ``Any`` — different
                verifiers expect different runtime types (e.g. a predicted
                ``Verdict`` / ``EvaluationResult`` for verdict comparison, or a
                raw ``final_output`` string for output validation). Each
                concrete verifier narrows ``actual`` to the type it expects and
                documents that narrowing.

        Returns:
            A :class:`VerifierResult`. The method must not raise merely because
            "the output was bad" — that case is ``passed=False``. Raise only on
            genuinely malformed inputs.
        """
        ...
