"""
Tests for the pure plan-grounding engine (app/core/registry/grounding.py) and
its internal schema validator (_schema_validate.validate_params).

Grounding is the Phase-2 back-half of orchestration's self-grounding: a draft
ExecutionPlan goes in, a plan whose steps are all registry-backed + schema-valid
comes out, plus a report of what was dropped and why. It is PURE — no LLM, no IO,
never raises on a bad step (bad steps become DroppedStep entries).
"""

from __future__ import annotations

import tempfile

import pytest

from app.models.core_state import ExecutionPlan, PlanStep
from app.core.tools.dispatcher import ToolDispatcher
from app.core.registry.builtin import build_registry_from_dispatcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _registry():
    """A real registry built from a real dispatcher (anti-drift), tmp workspace."""
    return build_registry_from_dispatcher(
        ToolDispatcher(workspace_dir=tempfile.gettempdir())
    )


def _roots(registry):
    return {e.id.split(".", 1)[0] for e in registry.all()}


def _plan(*steps, revision=1):
    return ExecutionPlan(goal="test goal", steps=list(steps), revision=revision)


# ---------------------------------------------------------------------------
# STEP 1/2 — validate_params
# ---------------------------------------------------------------------------


class TestValidateParams:
    def test_empty_schema_accepts_any_params(self):
        from app.core.registry._schema_validate import validate_params

        # git.* schema is {} → must accept anything (vacuously valid)
        assert validate_params({}, {}) is None
        assert validate_params({"anything": 1, "foo": "bar"}, {}) is None

    def test_required_key_missing_is_rejected(self):
        from app.core.registry._schema_validate import validate_params

        schema = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        }
        err = validate_params({"path": "x"}, schema)
        assert err is not None and isinstance(err, str)

    def test_all_required_present_is_accepted(self):
        from app.core.registry._schema_validate import validate_params

        schema = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        }
        assert validate_params({"path": "x", "content": "hello"}, schema) is None

    def test_wrong_type_is_rejected(self):
        from app.core.registry._schema_validate import validate_params

        schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        err = validate_params({"path": 123}, schema)
        assert err is not None

    def test_extra_unknown_key_is_accepted(self):
        from app.core.registry._schema_validate import validate_params

        # Schemas are non-strict (no additionalProperties:false) → extra keys OK.
        schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        assert validate_params({"path": "x", "extra": "ignored"}, schema) is None


# ---------------------------------------------------------------------------
# STEP 3/4 — capability_id_for_step (root.action reconciliation)
# ---------------------------------------------------------------------------


class TestCapabilityIdForStep:
    def test_file_operations_uses_action(self):
        from app.core.registry.grounding import capability_id_for_step

        assert (
            capability_id_for_step("file_operations", {"action": "read"})
            == "file_operations.read"
        )

    def test_file_operations_default_action_is_list(self):
        from app.core.registry.grounding import capability_id_for_step

        # Dispatcher default for file_operations is "list".
        assert capability_id_for_step("file_operations", {}) == "file_operations.list"

    def test_git_default_action_is_status(self):
        from app.core.registry.grounding import capability_id_for_step

        assert capability_id_for_step("git", {}) == "git.status"

    def test_web_research_is_fixed_to_fetch(self):
        from app.core.registry.grounding import capability_id_for_step

        # web_research has no params['action']; it reads a url → fixed capability.
        assert (
            capability_id_for_step("web_research", {"url": "http://x"})
            == "web_research.fetch"
        )

    def test_database_is_fixed_to_query(self):
        from app.core.registry.grounding import capability_id_for_step

        assert (
            capability_id_for_step("database", {"endpoint": "/kb"}) == "database.query"
        )


# ---------------------------------------------------------------------------
# STEP 5-9 — ground_plan
# ---------------------------------------------------------------------------


class TestGroundPlanHappyPath:
    def test_clean_plan_survives_intact(self):
        from app.core.registry.grounding import ground_plan

        reg = _registry()
        read_step = PlanStep(
            name="read",
            description="read a file",
            tool="file_operations",
            params={"action": "read", "path": "main.py"},
        )
        git_step = PlanStep(
            name="status",
            description="git status",
            tool="git",
            params={"action": "status"},
        )
        web_step = PlanStep(
            name="fetch",
            description="fetch a page",
            tool="web_research",
            params={"url": "https://example.com"},
        )
        llm_step = PlanStep(name="think", description="reason", tool=None)
        plan = _plan(read_step, git_step, web_step, llm_step, revision=2)

        grounded, report = ground_plan(plan, reg, _roots(reg))

        assert report.is_clean()
        assert report.dropped == []
        ids = {s.id for s in grounded.steps}
        assert ids == {read_step.id, git_step.id, web_step.id, llm_step.id}
        assert set(report.grounded_step_ids) == ids
        # tool=None step survives unconditionally
        assert llm_step.id in report.grounded_step_ids

    def test_revision_and_identity_preserved_and_no_mutation(self):
        from app.core.registry.grounding import ground_plan

        reg = _registry()
        step = PlanStep(
            name="read",
            description="read",
            tool="file_operations",
            params={"action": "read", "path": "main.py"},
        )
        plan = _plan(step, revision=3)
        original_step_count = len(plan.steps)

        grounded, report = ground_plan(plan, reg, _roots(reg))

        assert grounded.revision == 3
        assert grounded.id == plan.id
        assert grounded.goal == plan.goal
        # Input plan object NOT mutated.
        assert grounded is not plan
        assert len(plan.steps) == original_step_count

    def test_git_step_passes_empty_schema_trivially(self):
        from app.core.registry.grounding import ground_plan

        reg = _registry()
        # git.diff schema is {} — must accept regardless of extra params.
        step = PlanStep(
            name="diff",
            description="diff",
            tool="git",
            params={"action": "diff", "args": ["HEAD"]},
        )
        plan = _plan(step)
        grounded, report = ground_plan(plan, reg, _roots(reg))
        assert report.is_clean()
        assert len(grounded.steps) == 1


class TestGroundPlanInventedTool:
    def test_invented_tool_dropped(self):
        from app.core.registry.grounding import ground_plan

        reg = _registry()
        good = PlanStep(
            name="read",
            description="read",
            tool="file_operations",
            params={"action": "read", "path": "main.py"},
        )
        bad = PlanStep(
            name="browse",
            description="open browser",
            tool="browser",
            params={"url": "x"},
        )
        plan = _plan(good, bad)
        grounded, report = ground_plan(plan, reg, _roots(reg))

        surviving = {s.id for s in grounded.steps}
        assert good.id in surviving
        assert bad.id not in surviving
        assert len(report.dropped) == 1
        d = report.dropped[0]
        assert d.step_id == bad.id
        assert d.reason == "invented_tool"
        assert d.tool == "browser"


class TestGroundPlanUnknownAction:
    def test_real_root_unknown_action_dropped(self):
        from app.core.registry.grounding import ground_plan

        reg = _registry()
        # git.branch is dispatcher-safe (allow-listed) but has NO capability entry.
        step = PlanStep(
            name="branch",
            description="list branches",
            tool="git",
            params={"action": "branch"},
        )
        plan = _plan(step)
        grounded, report = ground_plan(plan, reg, _roots(reg))

        assert grounded.steps == []
        assert len(report.dropped) == 1
        d = report.dropped[0]
        assert d.reason == "unknown_action"
        assert d.capability_id == "git.branch"


class TestGroundPlanInvalidParams:
    def test_missing_required_param_dropped(self):
        from app.core.registry.grounding import ground_plan

        reg = _registry()
        # file_operations.write requires 'content'; omit it.
        step = PlanStep(
            name="write",
            description="write",
            tool="file_operations",
            params={"action": "write", "path": "x"},
        )
        plan = _plan(step)
        grounded, report = ground_plan(plan, reg, _roots(reg))

        assert grounded.steps == []
        assert len(report.dropped) == 1
        d = report.dropped[0]
        assert d.reason == "invalid_params"
        assert d.capability_id == "file_operations.write"

    def test_action_key_not_validated_against_schema(self):
        from app.core.registry.grounding import ground_plan

        reg = _registry()
        # 'action' is not part of any schema; presence must not trip validation.
        step = PlanStep(
            name="read",
            description="read",
            tool="file_operations",
            params={"action": "read", "path": "main.py"},
        )
        plan = _plan(step)
        grounded, report = ground_plan(plan, reg, _roots(reg))
        assert report.is_clean()

    def test_list_without_path_is_grounded(self):
        # file_operations.list defaults path to '.' in the dispatcher, so a list
        # step with no path is VALID and must NOT be dropped. The registry schema
        # must not be stricter than the handler it describes.
        from app.core.registry.grounding import ground_plan

        reg = _registry()
        step = PlanStep(
            name="list",
            description="list workspace root",
            tool="file_operations",
            params={"action": "list"},
        )
        plan = _plan(step)
        grounded, report = ground_plan(plan, reg, _roots(reg))
        assert report.is_clean()
        assert len(grounded.steps) == 1


class TestGroundPlanOrphanedDependents:
    def test_dependent_on_dropped_step_is_cascade_dropped(self):
        # If a step is dropped, steps that depend on it cannot run correctly, so
        # they are cascade-dropped too — not left to silently run without their
        # prerequisite (reasoning treats unknown dep ids as satisfied). They are
        # also surfaced in orphaned_dependents.
        from app.core.registry.grounding import ground_plan

        reg = _registry()
        a = PlanStep(
            name="A",
            description="invented",
            tool="browser",
            params={"url": "x"},
        )
        b = PlanStep(
            name="B",
            description="read",
            tool="file_operations",
            params={"action": "read", "path": "main.py"},
            dependencies=[a.id],
        )
        plan = _plan(a, b)
        grounded, report = ground_plan(plan, reg, _roots(reg))

        surviving = {s.id for s in grounded.steps}
        assert a.id not in surviving  # invented tool dropped
        assert b.id not in surviving  # cascade-dropped (depended on A)
        assert b.id in report.orphaned_dependents
        reasons = {d.step_id: d.reason for d in report.dropped}
        assert reasons[a.id] == "invented_tool"
        assert reasons[b.id] == "orphaned_dependency"

    def test_transitive_cascade_drop(self):
        # C -> B -> A(dropped): both B and C cascade-drop to fixpoint.
        from app.core.registry.grounding import ground_plan

        reg = _registry()
        a = PlanStep(name="A", description="invented", tool="browser", params={})
        b = PlanStep(
            name="B",
            description="read",
            tool="file_operations",
            params={"action": "read", "path": "x"},
            dependencies=[a.id],
        )
        c = PlanStep(
            name="C",
            description="read",
            tool="file_operations",
            params={"action": "read", "path": "y"},
            dependencies=[b.id],
        )
        plan = _plan(a, b, c)
        grounded, report = ground_plan(plan, reg, _roots(reg))
        assert grounded.steps == []
        assert set(report.orphaned_dependents) == {b.id, c.id}

    def test_to_dict_shape(self):
        from app.core.registry.grounding import ground_plan

        reg = _registry()
        bad = PlanStep(
            name="bad",
            description="invented",
            tool="shell",
            params={},
        )
        plan = _plan(bad)
        _, report = ground_plan(plan, reg, _roots(reg))
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["dropped"] and d["dropped"][0]["reason"] == "invented_tool"
        assert "grounded_step_ids" in d
        assert "orphaned_dependents" in d
