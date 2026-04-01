"""
Tests for task_models.py — Task bounds, defaults, constants, and routing score validation.

Sentinel-value contract:
- Remove Task.priority ge=1 → priority=0 stored → routing skips critical tasks → test fails
- Remove Task.priority le=10 → priority=100 stored → sort order broken → test fails
- Remove Task.status default → task created with None status → routing crashes → test fails
- Remove TaskAssignment.confidence_score bounds → agent stores 1.5 confidence → trust math breaks → test fails
- Remove TaskRoutingScore bounds → scores outside 0-1 → normalisation fails → test fails
"""

from __future__ import annotations

import pytest
from uuid import UUID
from pydantic import ValidationError

from app.models.task_models import (
    AgentResponse,
    AgentTaskMetrics,
    Task,
    TaskAssignment,
    TaskMetrics,
    TaskRoutingScore,
    TaskStatus,
    TaskType,
)
from uuid import uuid4


# ===========================================================================
# Task — priority bounds and defaults
# ===========================================================================

class TestTask:
    def _min_task(self, **overrides) -> dict:
        base = {"task_type": "research"}
        base.update(overrides)
        return base

    def test_priority_below_1_rejected(self):
        """
        SENTINEL — priority < 1 must raise ValidationError.
        Remove ge=1 → priority=0 stored → high-priority tasks missed → test fails.
        """
        with pytest.raises(ValidationError):
            Task(**self._min_task(priority=0))

    def test_priority_above_10_rejected(self):
        """
        SENTINEL — priority > 10 must raise ValidationError.
        Remove le=10 → priority=100 stored → sort order meaningless → test fails.
        """
        with pytest.raises(ValidationError):
            Task(**self._min_task(priority=11))

    def test_priority_boundaries_accepted(self):
        """Boundary values 1 and 10 must be accepted."""
        low = Task(**self._min_task(priority=1))
        high = Task(**self._min_task(priority=10))
        assert low.priority == 1
        assert high.priority == 10

    def test_default_priority_is_5(self):
        """
        SENTINEL — default priority must be 5 (middle of range).
        Default to 1 → unset tasks compete as lowest priority → test fails.
        """
        task = Task(**self._min_task())
        assert task.priority == 5

    def test_default_status_is_queued(self):
        """
        SENTINEL — default status must be 'queued'.
        Default None → routing engine skips task → test fails.
        """
        task = Task(**self._min_task())
        assert task.status == "queued"

    def test_default_human_override_false(self):
        """
        SENTINEL — human_override defaults to False.
        Default True → all tasks flagged as manually overridden → audit trail wrong → test fails.
        """
        task = Task(**self._min_task())
        assert task.human_override is False

    def test_id_auto_generated(self):
        """id must be auto-generated UUID."""
        task = Task(**self._min_task())
        assert isinstance(task.id, UUID)

    def test_required_fields_set(self):
        """task_type is required — omitting must raise ValidationError."""
        with pytest.raises(ValidationError):
            Task()

    def test_required_capabilities_default_empty(self):
        """required_capabilities must default to empty list."""
        task = Task(**self._min_task())
        assert task.required_capabilities == []

    def test_payload_default_empty(self):
        """payload must default to empty dict."""
        task = Task(**self._min_task())
        assert task.payload == {}


# ===========================================================================
# TaskAssignment — confidence_score bounds
# ===========================================================================

class TestTaskAssignment:
    def _min_assignment(self, **overrides) -> dict:
        base = {
            "task_id": uuid4(),
            "agent_id": uuid4(),
            "agent_response": "accept",
            "confidence_score": 0.9,
        }
        base.update(overrides)
        return base

    def test_confidence_score_above_1_rejected(self):
        """
        SENTINEL — confidence_score > 1.0 must raise ValidationError.
        Remove le=1.0 → agent stores 1.5 confidence → trust normalisation breaks → test fails.
        """
        with pytest.raises(ValidationError):
            TaskAssignment(**self._min_assignment(confidence_score=1.1))

    def test_confidence_score_below_0_rejected(self):
        """confidence_score < 0.0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            TaskAssignment(**self._min_assignment(confidence_score=-0.1))

    def test_confidence_score_boundary_accepted(self):
        """0.0 and 1.0 must be accepted."""
        low = TaskAssignment(**self._min_assignment(confidence_score=0.0))
        high = TaskAssignment(**self._min_assignment(confidence_score=1.0))
        assert low.confidence_score == 0.0
        assert high.confidence_score == 1.0


# ===========================================================================
# TaskRoutingScore — all score bounds
# ===========================================================================

class TestTaskRoutingScore:
    def _min_score(self, **overrides) -> dict:
        base = {
            "agent_id": uuid4(),
            "total_score": 0.8,
            "capability_match_score": 0.9,
            "load_score": 0.7,
            "trust_score": 0.8,
            "model_preference_score": 0.6,
            "task_type_performance_score": 0.7,
        }
        base.update(overrides)
        return base

    def test_total_score_above_1_rejected(self):
        """
        SENTINEL — total_score > 1.0 must raise ValidationError.
        Remove le=1.0 → total > 1 breaks normalisation across agents → test fails.
        """
        with pytest.raises(ValidationError):
            TaskRoutingScore(**self._min_score(total_score=1.1))

    def test_capability_match_score_below_0_rejected(self):
        """capability_match_score < 0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            TaskRoutingScore(**self._min_score(capability_match_score=-0.1))

    def test_load_score_above_1_rejected(self):
        """load_score > 1.0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            TaskRoutingScore(**self._min_score(load_score=1.5))

    def test_trust_score_below_0_rejected(self):
        """trust_score < 0.0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            TaskRoutingScore(**self._min_score(trust_score=-1.0))

    def test_valid_routing_score_accepted(self):
        """A fully valid routing score must be accepted."""
        score = TaskRoutingScore(**self._min_score())
        assert score.total_score == pytest.approx(0.8)
        assert isinstance(score.agent_id, UUID)


# ===========================================================================
# TaskStatus constants
# ===========================================================================

class TestTaskStatusConstants:
    def test_queued_value(self):
        """
        SENTINEL — QUEUED must equal 'queued'.
        Change to 'pending' → routing engine uses wrong string → tasks never dispatched → test fails.
        """
        assert TaskStatus.QUEUED == "queued"

    def test_completed_value(self):
        """COMPLETED must equal 'completed'."""
        assert TaskStatus.COMPLETED == "completed"

    def test_failed_value(self):
        """FAILED must equal 'failed'."""
        assert TaskStatus.FAILED == "failed"

    def test_running_value(self):
        """RUNNING must equal 'running'."""
        assert TaskStatus.RUNNING == "running"

    def test_assigned_value(self):
        """ASSIGNED must equal 'assigned'."""
        assert TaskStatus.ASSIGNED == "assigned"


# ===========================================================================
# AgentResponse constants
# ===========================================================================

class TestAgentResponseConstants:
    def test_accept_value(self):
        """
        SENTINEL — ACCEPT must equal 'accept'.
        Change string → routing can't detect accepted tasks → assigned counter wrong → test fails.
        """
        assert AgentResponse.ACCEPT == "accept"

    def test_refuse_value(self):
        """REFUSE must equal 'refuse'."""
        assert AgentResponse.REFUSE == "refuse"

    def test_suggest_alternative_value(self):
        """SUGGEST_ALTERNATIVE must equal 'suggest_alternative'."""
        assert AgentResponse.SUGGEST_ALTERNATIVE == "suggest_alternative"


# ===========================================================================
# TaskType constants
# ===========================================================================

class TestTaskTypeConstants:
    def test_research_value(self):
        """
        SENTINEL — RESEARCH must equal 'research'.
        Change string → model router looks up wrong key → wrong model selected → test fails.
        """
        assert TaskType.RESEARCH == "research"

    def test_code_value(self):
        """CODE must equal 'code'."""
        assert TaskType.CODE == "code"

    def test_analysis_value(self):
        """ANALYSIS must equal 'analysis'."""
        assert TaskType.ANALYSIS == "analysis"
