"""
Tests for the output-validation gate wiring into the verdict path (target #7).

These tests exercise ``evaluate()``'s integration with ``validate_output``
behind the ``ENABLE_OUTPUT_VALIDATION`` feature flag. They prove three things:

  (a) Flag OFF (explicitly disabled): an APPROVE-worthy case with deliberately
      broken output still returns APPROVE — the gate is a strict no-op.
  (b) Flag ON + failing validation on an otherwise-APPROVE case → ESCALATE.
  (c) Flag ON + passing validation → verdict unchanged (APPROVE).

The verdict is pinned to APPROVE by patching ``_determine_verdict`` so the
tests isolate the *wiring* (not the heuristic scorer). Repositories are mocked
with AsyncMock (no DB), and ``validate_output`` is patched at the service layer
for the flag-ON cases (no real validation dependency, no network/LLM).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

import app.services.evaluation_service as eval_service
from app.models.evaluation_models import (
    EvaluationInput,
    EvaluationVerdict,
    Verdict,
)
from app.services.evaluation_service import evaluate
from app.services.output_validator import CheckResult, ValidationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _approve_verdict() -> EvaluationVerdict:
    """An APPROVE verdict, as ``_determine_verdict`` would return on success."""
    return EvaluationVerdict(
        verdict=Verdict.APPROVE,
        reasoning="Output quality (0.90) meets threshold and all required steps completed.",
        confidence=0.9,
        suggested_improvements=[],
    )


def _make_input(final_output: str, task_category: str | None = None) -> EvaluationInput:
    """Minimal evaluation input; metadata carries the task category if given."""
    metadata = {}
    if task_category is not None:
        metadata["task_category"] = task_category
    return EvaluationInput(
        task_id=uuid4(),
        original_intent="Write a python function that adds two numbers.",
        plan_steps=[],
        step_results=[],
        final_output=final_output,
        agent_id=uuid4(),
        model_used="phi3:mini",
        retry_count=0,
        max_retries=3,
        metadata=metadata,
    )


def _mock_repos():
    """Patch the repository + memory layers so evaluate() touches no DB."""
    return (
        patch("app.services.evaluation_service.evaluation_repository"),
        patch("app.services.evaluation_service.memory_repository"),
    )


# Deliberately broken "code" output: an unbalanced markdown fence — would fail
# the output validator's fence_balance check, but is otherwise rich prose so a
# (mocked) APPROVE verdict is plausible.
BROKEN_CODE_OUTPUT = (
    "Here is the function you asked for:\n\n"
    "```python\n"
    "def add(a, b):\n"
    "    return a + b\n"
    "\n"  # NOTE: missing closing fence → unbalanced
)


# ---------------------------------------------------------------------------
# (a) Flag OFF → strict no-op
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_off_broken_output_still_approves(monkeypatch):
    """Flag OFF (explicit): broken output that would fail validation still APPROVES."""
    # Force the flag OFF for this test regardless of the (now ON-by-default) value.
    monkeypatch.setattr(eval_service, "ENABLE_OUTPUT_VALIDATION", False)

    repo_patch, mem_patch = _mock_repos()
    with repo_patch as mock_repo, mem_patch as mock_mem, patch(
        "app.services.evaluation_service._determine_verdict",
        return_value=_approve_verdict(),
    ), patch(
        "app.services.output_validator.validate_output",
        side_effect=AssertionError(
            "validate_output must NOT be called when flag is OFF"
        ),
    ):
        mock_repo.store_evaluation = AsyncMock(return_value=uuid4())
        mock_mem.create_procedural_memory = AsyncMock(return_value=uuid4())

        result = await evaluate(_make_input(BROKEN_CODE_OUTPUT, task_category="code"))

    assert result.verdict.verdict == Verdict.APPROVE


# ---------------------------------------------------------------------------
# (b) Flag ON + failing validation → downgrade to ESCALATE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_on_failing_validation_downgrades_to_escalate(monkeypatch):
    """Flag ON + validation FAILS on an otherwise-APPROVE case → ESCALATE."""
    monkeypatch.setattr(eval_service, "ENABLE_OUTPUT_VALIDATION", True)

    failing = ValidationResult(
        passed=False,
        checks=[
            CheckResult(name="nonempty", passed=True),
            CheckResult(
                name="fence_balance",
                passed=False,
                detail="unbalanced markdown code fences (3 fence markers)",
            ),
        ],
    )

    repo_patch, mem_patch = _mock_repos()
    with repo_patch as mock_repo, mem_patch as mock_mem, patch(
        "app.services.evaluation_service._determine_verdict",
        return_value=_approve_verdict(),
    ), patch(
        "app.services.output_validator.validate_output", return_value=failing
    ) as mock_validate:
        mock_repo.store_evaluation = AsyncMock(return_value=uuid4())
        mock_mem.create_procedural_memory = AsyncMock(return_value=uuid4())

        result = await evaluate(_make_input(BROKEN_CODE_OUTPUT, task_category="code"))

    # Verdict downgraded.
    assert result.verdict.verdict == Verdict.ESCALATE
    # Reasoning extended with the failure + failed check name.
    assert "validation failed" in result.verdict.reasoning.lower()
    assert "fence_balance" in result.verdict.reasoning
    # Confidence stays sensible (within bounds).
    assert 0.0 <= result.verdict.confidence <= 1.0
    # Validator was actually invoked with the final output + category.
    mock_validate.assert_called_once()
    called_args = mock_validate.call_args.args
    assert called_args[0] == BROKEN_CODE_OUTPUT
    assert called_args[1] == "code"


# ---------------------------------------------------------------------------
# (c) Flag ON + passing validation → verdict unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_on_passing_validation_keeps_approve(monkeypatch):
    """Flag ON + validation PASSES → verdict stays APPROVE."""
    monkeypatch.setattr(eval_service, "ENABLE_OUTPUT_VALIDATION", True)

    passing = ValidationResult(
        passed=True,
        checks=[
            CheckResult(name="nonempty", passed=True),
            CheckResult(name="max_length", passed=True),
        ],
    )

    repo_patch, mem_patch = _mock_repos()
    with repo_patch as mock_repo, mem_patch as mock_mem, patch(
        "app.services.evaluation_service._determine_verdict",
        return_value=_approve_verdict(),
    ), patch(
        "app.services.output_validator.validate_output", return_value=passing
    ) as mock_validate:
        mock_repo.store_evaluation = AsyncMock(return_value=uuid4())
        mock_mem.create_procedural_memory = AsyncMock(return_value=uuid4())

        result = await evaluate(
            _make_input("A clean, valid answer.", task_category="research")
        )

    assert result.verdict.verdict == Verdict.APPROVE
    mock_validate.assert_called_once()


# ---------------------------------------------------------------------------
# (d) Flag ON + human-feedback override → never downgraded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_on_human_override_skips_gate(monkeypatch):
    """Flag ON but a human override in metadata → gate is skipped, APPROVE stands."""
    monkeypatch.setattr(eval_service, "ENABLE_OUTPUT_VALIDATION", True)

    inp = _make_input(BROKEN_CODE_OUTPUT, task_category="code")
    inp.metadata["human_feedback_override"] = True

    repo_patch, mem_patch = _mock_repos()
    with repo_patch as mock_repo, mem_patch as mock_mem, patch(
        "app.services.evaluation_service._determine_verdict",
        return_value=_approve_verdict(),
    ), patch(
        "app.services.output_validator.validate_output",
        side_effect=AssertionError("gate must be skipped on human override"),
    ):
        mock_repo.store_evaluation = AsyncMock(return_value=uuid4())
        mock_mem.create_procedural_memory = AsyncMock(return_value=uuid4())

        result = await evaluate(inp)

    assert result.verdict.verdict == Verdict.APPROVE
