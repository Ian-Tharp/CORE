"""
Behavior tests for OrchestrationAgent (the "O" in C/O/R/E).

Covers plan parsing from LLM output (fenced / raw / bare JSON), intent
injection into the prompt, and the fallback single-step plan for every
failure mode (empty content, unparseable JSON, LLM exception, malformed
step shapes). The LLM boundary (``get_openai_client_sync``) is always mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.agents.orchestration_agent import OrchestrationAgent
from app.models.core_state import ExecutionPlan, PlanStep, UserIntent

_CLIENT_PATH = "app.core.agents.orchestration_agent.get_openai_client_sync"


def _client_returning(content):
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=content))]
    )
    return client


def _run(content, **kwargs):
    agent = OrchestrationAgent(model="test-model")
    client = _client_returning(content)
    with patch(_CLIENT_PATH, return_value=client):
        plan = agent.create_plan(kwargs.pop("user_input", "do the task"), **kwargs)
    return plan, client


def _user_message(client) -> str:
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    return next(m["content"] for m in messages if m["role"] == "user")


def _assert_fallback(plan: ExecutionPlan, goal: str = "do the task"):
    assert plan.goal == goal
    assert plan.reasoning == "Fallback plan due to orchestration error"
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.name == "Execute task"
    assert step.tool is None
    assert step.requires_hitl is False
    assert step.description.startswith("Error in orchestration: ")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestPlanParsing:
    def test_full_plan_is_mapped_to_steps(self):
        content = (
            "Here is the plan:\n```json\n"
            '{"goal": "Ship it", "reasoning": "because", "steps": ['
            '{"name": "Read", "description": "read file", "tool": "file_operations",'
            ' "params": {"action": "read", "path": "a.py"}, "dependencies": [],'
            ' "requires_hitl": false},'
            '{"name": "Push", "description": "inspect", "tool": "git",'
            ' "params": {"action": "status"}, "dependencies": ["s1"],'
            ' "requires_hitl": true}'
            "]}\n```\nDone."
        )
        plan, client = _run(content, revision=3)

        assert plan.goal == "Ship it"
        assert plan.reasoning == "because"
        assert plan.revision == 3
        assert [s.name for s in plan.steps] == ["Read", "Push"]
        assert plan.steps[0].tool == "file_operations"
        assert plan.steps[0].params == {"action": "read", "path": "a.py"}
        assert plan.steps[1].dependencies == ["s1"]
        assert plan.steps[1].requires_hitl is True

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "test-model"
        assert kwargs["messages"][0]["role"] == "system"

    def test_missing_step_fields_get_defaults(self):
        plan, _ = _run('{"steps": [{}, {"name": "Named"}]}', user_input="goal X")
        assert plan.goal == "goal X"  # defaults to user input
        assert plan.reasoning == ""
        assert [s.name for s in plan.steps] == ["Step 1", "Named"]
        first = plan.steps[0]
        assert first.description == ""
        assert first.tool is None
        assert first.params == {}
        assert first.dependencies == []
        assert first.requires_hitl is False

    def test_bare_json_without_braces_falls_through_to_raw_content(self):
        """No ``{`` in output → extraction yields None → raw content parsed."""
        plan, _ = _run("[]")
        # A JSON list has no .get → handled by the fallback, not raised.
        _assert_fallback(plan)

    def test_trailing_comma_json_is_repaired(self):
        plan, _ = _run('{"goal": "g", "steps": [{"name": "A",},],}')
        assert plan.goal == "g"
        assert [s.name for s in plan.steps] == ["A"]

    def test_plan_may_reference_tool_not_in_registry(self):
        """Unknown tools are kept verbatim; cost estimation treats them as LLM steps."""
        agent = OrchestrationAgent()
        client = _client_returning('{"steps": [{"name": "x", "tool": "teleport"}]}')
        with patch(_CLIENT_PATH, return_value=client):
            plan = agent.create_plan("t")
        assert plan.steps[0].tool == "teleport"
        cost = agent.estimate_plan_cost(plan)
        assert cost["llm_step_count"] == 1
        assert cost["tool_breakdown"] == {}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


class TestPromptContext:
    def test_intent_description_and_tools_are_injected(self):
        intent = UserIntent(
            type="task",
            description="SENTINEL_INTENT",
            confidence=0.9,
            tools_needed=["git", "web_research"],
        )
        _, client = _run('{"steps": []}', intent=intent)
        msg = _user_message(client)
        assert "Task: do the task" in msg
        assert "Intent: SENTINEL_INTENT" in msg
        assert "Tools needed: git, web_research" in msg

    def test_intent_without_tools_says_none(self):
        intent = UserIntent(type="question", description="d", confidence=0.5)
        _, client = _run('{"steps": []}', intent=intent)
        assert "Tools needed: None" in _user_message(client)

    def test_feedback_without_previous_plan_is_not_injected(self):
        _, client = _run('{"steps": []}', evaluation_feedback="IGNORED_FEEDBACK")
        assert "IGNORED_FEEDBACK" not in _user_message(client)


# ---------------------------------------------------------------------------
# Failure modes → fallback plan
# ---------------------------------------------------------------------------


class TestFallbackPlan:
    @pytest.mark.parametrize("content", [None, ""])
    def test_empty_llm_content_yields_fallback(self, content):
        plan, _ = _run(content, revision=2)
        _assert_fallback(plan)
        assert plan.revision == 2
        assert "Empty response from LLM" in plan.steps[0].description

    def test_unparseable_json_yields_fallback(self):
        plan, _ = _run("I cannot produce a plan right now, sorry.")
        _assert_fallback(plan)
        assert "Could not parse JSON" in plan.steps[0].description

    def test_llm_exception_is_captured_not_propagated(self):
        agent = OrchestrationAgent()
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("LLM down")
        with patch(_CLIENT_PATH, return_value=client):
            plan = agent.create_plan("do the task", revision=4)
        _assert_fallback(plan)
        assert plan.revision == 4
        assert "LLM down" in plan.steps[0].description

    def test_malformed_step_params_yields_fallback(self):
        """``params`` must be a dict; a string fails PlanStep validation."""
        plan, _ = _run('{"steps": [{"name": "a", "params": "not-a-dict"}]}')
        _assert_fallback(plan)

    def test_non_dict_step_yields_fallback(self):
        plan, _ = _run('{"steps": ["just a string"]}')
        _assert_fallback(plan)


class TestRevisionAndCost:
    def test_revision_prompt_lists_previous_steps_and_feedback(self):
        prev = ExecutionPlan(
            goal="g",
            revision=1,
            steps=[PlanStep(name="Old step", description="d", status="failed")],
        )
        _, client = _run(
            '{"steps": []}', previous_plan=prev, evaluation_feedback="FIX_IT"
        )
        msg = _user_message(client)
        assert "Previous plan (revision 1):" in msg
        assert "- Old step: failed" in msg
        assert "Evaluation feedback: FIX_IT" in msg

    def test_cost_high_risk_with_hitl(self):
        plan = ExecutionPlan(
            goal="g",
            steps=[
                PlanStep(
                    name="a", description="", tool=" Database ", requires_hitl=True
                ),
                PlanStep(name="b", description="", tool="file_operations"),
                PlanStep(name="c", description="", tool=None),
            ],
        )
        cost = OrchestrationAgent().estimate_plan_cost(plan)
        assert cost["risk_level"] == "high"
        assert cost["tool_breakdown"] == {"database": 1, "file_operations": 1}
        assert cost["llm_step_count"] == 1
        assert cost["hitl_checkpoints"] == 1
        assert cost["estimated_tokens"] == 400 + 2 * 100

    @pytest.mark.parametrize(
        "tool,hitl,expected",
        [
            ("git", False, "medium"),
            (None, True, "medium"),
            ("web_research", False, "low"),
        ],
    )
    def test_cost_risk_levels(self, tool, hitl, expected):
        plan = ExecutionPlan(
            goal="g",
            steps=[PlanStep(name="a", description="", tool=tool, requires_hitl=hitl)],
        )
        assert OrchestrationAgent().estimate_plan_cost(plan)["risk_level"] == expected


# ---------------------------------------------------------------------------
# Tool discovery fallback
# ---------------------------------------------------------------------------


class TestToolDiscovery:
    def test_dispatcher_failure_falls_back_to_default_tools(self):
        with patch(
            "app.core.tools.dispatcher.ToolDispatcher",
            side_effect=RuntimeError("boom"),
        ):
            tools = OrchestrationAgent._get_available_tools()
            agent = OrchestrationAgent()
        assert tools == ["file_operations", "git", "web_research", "database"]
        for t in tools:
            assert f"**{t}**" in agent.system_prompt

    def test_custom_tool_gets_generic_description(self):
        out = OrchestrationAgent._tool_descriptions(["my_tool"])
        assert out == "- **my_tool**: custom tool"
