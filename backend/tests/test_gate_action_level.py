"""
Action-level capability checks in the comprehension gate.

A present tool root does not mean the requested action is present: CORE's git is
read-only, so "push main to origin" must refuse even though "git" is registered.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.core.agents.comprehension_agent import ComprehensionAgent
from app.core.agents.comprehension_gate import (
    decide_gate,
    resolve_tools,
    unsupported_operations,
)
from app.core.registry.actions import GATED_OPERATIONS, requested_operations
from app.core.registry.builtin import _BUILTIN, build_registry_from_dispatcher
from app.core.registry.capability import CapabilityEntry, CapabilityRegistry
from app.models.core_state import UserIntent


@pytest.fixture(scope="module")
def registry() -> CapabilityRegistry:
    return build_registry_from_dispatcher(SimpleNamespace(available_tools=_BUILTIN))


def _intent(description: str, tools, confidence: float = 0.92) -> UserIntent:
    return UserIntent(
        type="task",
        description=description,
        confidence=confidence,
        requires_tools=bool(tools),
        tools_needed=list(tools),
    )


# ---- requested_operations -----------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Push main to origin", ["push"]),
        ("Commit my staged changes", ["commit"]),
        ("Delete docs/README.md", ["delete"]),
        ("Fill in and submit the signup form", ["submit"]),
        ("Read main.py, then delete it", ["delete"]),
        ("Can you drop the table?", ["drop"]),
        ("I want to rebase onto develop", ["rebase"]),
        ("please wipe scratch/", ["wipe"]),
    ],
)
def test_gated_verbs_in_verb_position(text, expected):
    assert requested_operations(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Show commit a1b2c3d with git show",  # noun after a verb
        "Show the git log for commits after 2026-10-01",
        "Read docs/README.md",  # the dot in a path is not a sentence break
        "Fetch https://example.com/post",  # 'post' inside a URL
        "Show branches and tags",  # plural nouns are not stemmed
        "Run git status and send me the diff",  # run/send are not gated
        "",
    ],
)
def test_no_gated_verbs(text):
    assert requested_operations(text) == []


def test_gated_verbs_are_lowercase_single_words():
    assert all(v == v.lower() and " " not in v for v in GATED_OPERATIONS)


# ---- resolve_tools ------------------------------------------------------------


def test_resolve_roots_ids_and_absent(registry):
    present, missing = resolve_tools(
        ["git", "file_operations.read", "Web-Research", "email", "git.push"],
        registry,
    )
    assert present == ["git", "file_operations", "web_research"]
    assert missing == ["email", "git.push"]


def test_resolve_is_exact_not_lexical(registry):
    # "web_search" shares words with web_research.fetch and file_operations.search,
    # which the old lexical check accepted. CORE cannot search the web.
    assert registry.has_capability("web_search")  # the lexical check still would
    assert resolve_tools(["web_search"], registry) == ([], ["web_search"])


def test_unsupported_operations_respects_served_verbs(registry):
    assert unsupported_operations(["git"], "Push main", registry) == ["git.push"]
    # update/edit on a file is served by file_operations.write
    assert unsupported_operations(["file_operations"], "Update a.md", registry) == []
    # ... but not on the database
    assert unsupported_operations(["database"], "Update row 4", registry) == [
        "database.update"
    ]
    # any named root that serves the verb is enough
    assert (
        unsupported_operations(["git", "file_operations"], "Edit a.md", registry) == []
    )
    assert unsupported_operations([], "Push main", registry) == []


def test_entry_verbs_default_to_the_action_segment():
    e = CapabilityEntry(id="git.blame", name="Git blame", description="blame")
    assert e.root == "git" and e.action_verbs() == ("blame",)
    assert CapabilityEntry(id="solo", name="s", description="d").action_verbs() == ()


def test_registry_roots_and_for_root(registry):
    assert registry.roots() == {"file_operations", "git", "web_research", "database"}
    assert {e.id for e in registry.for_root("git")} == {
        "git.status",
        "git.diff",
        "git.log",
        "git.show",
    }


# ---- decide_gate --------------------------------------------------------------


@pytest.mark.parametrize(
    "text,tools,missing",
    [
        ("Push main to origin", ["git"], ["git.push"]),
        ("Commit my staged changes", ["git"], ["git.commit"]),
        ("Delete docs/README.md", ["file_operations"], ["file_operations.delete"]),
        ("Drop the conversations table", ["database"], ["database.drop"]),
        ("Send my branch upstream", ["git.push"], ["git.push"]),  # id, no verb
        ("Search the web for FastAPI", ["web_search"], ["web_search"]),
    ],
)
def test_missing_action_on_present_root_refuses(registry, text, tools, missing):
    d = decide_gate(_intent(text, tools), registry, text=text)
    assert d.outcome == "refuse"
    assert d.missing == missing


@pytest.mark.parametrize(
    "text,tools",
    [
        ("Show the git log for main", ["git"]),
        ("Show commit a1b2c3d with git show", ["git"]),
        ("Show the git diff", ["git.diff"]),
        ("Update docs/README.md to mention the suite", ["file_operations"]),
        ("Fetch https://example.com/status", ["web_research"]),
    ],
)
def test_served_action_proceeds(registry, text, tools):
    assert decide_gate(_intent(text, tools), registry, text=text).outcome == "proceed"


def test_raw_text_is_checked_when_the_description_paraphrases(registry):
    intent = _intent("Sync the branch with the remote", ["git"])
    assert decide_gate(intent, registry).outcome == "proceed"
    d = decide_gate(intent, registry, text="push main to origin")
    assert d.outcome == "refuse" and d.missing == ["git.push"]


def test_refusal_lists_what_the_root_can_do(registry):
    d = decide_gate(_intent("Push main", ["git"]), registry, text="Push main")
    assert "git.push" in d.message
    for name in ("Git status", "Git diff", "Git log", "Git show"):
        assert name in d.message


def test_absent_tool_still_refuses_before_action_check(registry):
    d = decide_gate(_intent("Email a file", ["file_operations", "email"]), registry)
    assert d.outcome == "refuse" and d.missing == ["email"]


def test_low_confidence_clarifies_before_action_check(registry):
    d = decide_gate(_intent("push?", ["git"], confidence=0.3), registry, text="push?")
    assert d.outcome == "clarify"


def test_no_tools_named_skips_the_action_check(registry):
    d = decide_gate(_intent("Delete nothing", []), registry, text="Delete nothing")
    assert d.outcome == "proceed"


# ---- classifier prompt from the registry ----------------------------------------


def test_prompt_lists_registry_actions_only(registry):
    prompt = ComprehensionAgent(registry=registry).system_prompt
    for e in registry.all():
        assert e.id in prompt
    # the old hardcoded list advertised actions the dispatcher cannot run
    assert "create PR" not in prompt
    assert "Branch, commit" not in prompt
    assert "git.push" in prompt  # as the example of naming an unlisted action


def test_prompt_follows_a_smaller_registry():
    reg = CapabilityRegistry()
    reg.register(
        CapabilityEntry(id="git.log", name="Git log", description="Show history")
    )
    prompt = ComprehensionAgent(registry=reg).system_prompt
    assert "git.log" in prompt and "file_operations.read" not in prompt


def test_default_agent_uses_every_builtin_capability():
    agent = ComprehensionAgent()
    assert agent.registry.roots() == set(_BUILTIN)


# ---- graph wiring -------------------------------------------------------------


def test_graph_gate_sees_raw_input():
    from app.core.langgraph.core_graph_v2 import COREGraph
    from app.models.core_state import COREState

    graph = COREGraph()
    assert graph.comprehension_agent.registry is graph.registry
    intent = _intent("Sync the branch with the remote", ["git"])
    state = COREState(user_input="push main to origin")
    with patch.object(
        graph.comprehension_agent,
        "analyze_intent",
        side_effect=lambda *_a, **_k: intent.model_copy(deep=True),
    ):
        state = graph.comprehension_node(state)
    assert state.gate_outcome == "refuse"
    assert graph.route_from_comprehension(state) == "conversation"
