"""Tests for app.services.output_validator — advisory output-validation gate.

Unit-level only: no DB, no model loads, no network. The validator is pure and
deterministic, so these assert the checks in isolation.

Sentinel-flavored contract:
- Remove the nonempty check -> empty output passes -> test fails.
- Remove the ast.parse syntax check -> "def f(:" passes -> test fails.
- Remove the fence-balance check -> a single fence passes -> test fails.
- Remove the path-safety check -> "../etc/passwd" passes for file ops -> fails.
- Make validate_output able to raise -> the never-raises test fails.
"""

from __future__ import annotations

import pytest

from app.services.output_validator import (
    ValidationResult,
    validate_output,
)


def _names_failed(result: ValidationResult):
    return {c.name for c in result.failures}


class TestGenericChecks:
    def test_valid_prose_passes_generic(self):
        result = validate_output("A concise, complete answer.", "task")
        assert result.passed is True
        assert result.failures == []

    def test_empty_fails(self):
        result = validate_output("", "task")
        assert result.passed is False
        assert "nonempty" in _names_failed(result)

    def test_whitespace_only_fails(self):
        result = validate_output("   \n\t ", "code")
        assert result.passed is False
        assert "nonempty" in _names_failed(result)

    def test_overlong_output_fails(self):
        result = validate_output("x" * 100_001, "conversation")
        assert result.passed is False
        assert "max_length" in _names_failed(result)


class TestCodeChecks:
    def test_valid_python_passes(self):
        result = validate_output("def f():\n    return 1\n", "code")
        assert result.passed is True

    def test_valid_fenced_python_passes(self):
        result = validate_output("Here:\n```python\nx = 1\nprint(x)\n```\n", "code")
        assert result.passed is True

    def test_broken_python_fails_syntax(self):
        result = validate_output("def f(:\n    pass\n", "code")
        assert result.passed is False
        assert "python_syntax" in _names_failed(result)

    def test_unbalanced_fence_fails(self):
        # One opening fence, no closing fence.
        result = validate_output("```python\nx = 1\n", "code")
        assert result.passed is False
        assert "fence_balance" in _names_failed(result)

    def test_prose_in_code_category_not_failed_for_syntax(self):
        # Conservative: prose with no code-looking lines is not parsed.
        result = validate_output(
            "The function should sum the list and return the total.", "code"
        )
        assert result.passed is True


class TestPathSafety:
    def test_parent_traversal_fails_for_file_management(self):
        result = validate_output("wrote ../etc/passwd", "file_management")
        assert result.passed is False
        assert "path_safety" in _names_failed(result)

    def test_absolute_path_fails_for_file_management(self):
        result = validate_output(
            "saved output to /etc/passwd successfully", "file_management"
        )
        assert result.passed is False
        assert "path_safety" in _names_failed(result)

    def test_file_ops_alias_runs_path_safety(self):
        result = validate_output("touched ../secrets", "file_ops")
        assert result.passed is False
        assert "path_safety" in _names_failed(result)

    def test_relative_workspace_path_passes(self):
        result = validate_output(
            "created output/result.txt in the workspace", "file_management"
        )
        assert result.passed is True

    def test_path_safety_not_run_for_research(self):
        # research is not a path category, so a stray "/etc/passwd" mention is
        # not flagged (no rule-based checks beyond generic).
        result = validate_output(
            "The file /etc/passwd is referenced in the docs.", "research"
        )
        assert result.passed is True


class TestRobustness:
    def test_unknown_category_runs_generic_only(self):
        assert validate_output("hello", "banana").passed is True
        empty = validate_output("", "banana")
        assert empty.passed is False
        assert _names_failed(empty) == {"nonempty"}

    def test_none_category_ok(self):
        assert validate_output("some answer", None).passed is True

    @pytest.mark.parametrize("junk", ["", "   ", "\x00\x01", "```", "().,;"])
    def test_never_raises_on_junk(self, junk):
        result = validate_output(junk, "code")
        assert isinstance(result, ValidationResult)

    def test_returns_validation_result_type(self):
        assert isinstance(validate_output("x", "task"), ValidationResult)
