"""
Tests for the Capability Registry — the single source of truth describing what
the system can do. Comprehension queries it for capability *presence*;
orchestration sequences from it.

These are pure/offline: no model, no network.
"""

from __future__ import annotations

import pytest

from app.core.registry.capability import CapabilityEntry, CapabilityRegistry


def _entry(id_, name, desc, **kw):
    return CapabilityEntry(
        id=id_,
        name=name,
        description=desc,
        params_schema=kw.get("params_schema", {}),
        side_effects=kw.get("side_effects", "none"),
        source=kw.get("source", "builtin"),
        examples=kw.get("examples", []),
    )


def _registry() -> CapabilityRegistry:
    reg = CapabilityRegistry()
    reg.register(
        _entry(
            "file_operations.read",
            "Read file",
            "Read the contents of a file in the workspace",
            side_effects="read",
            examples=["read main.py"],
        )
    )
    reg.register(
        _entry(
            "git.diff",
            "Git diff",
            "Show the git diff of changes in the repository",
            side_effects="read",
        )
    )
    reg.register(
        _entry(
            "web_research.fetch",
            "Fetch URL",
            "Fetch external web content over HTTP",
            side_effects="network",
            examples=["fetch a documentation page"],
        )
    )
    return reg


def test_register_get_all():
    reg = _registry()
    assert reg.get("git.diff").name == "Git diff"
    assert reg.get("nonexistent") is None
    assert len(reg.all()) == 3


def test_register_rejects_duplicate_id():
    reg = CapabilityRegistry()
    reg.register(_entry("a.b", "X", "desc"))
    with pytest.raises(ValueError):
        reg.register(_entry("a.b", "Y", "other"))


def test_search_finds_relevant_capability_by_content_overlap():
    reg = _registry()
    results = reg.search("open and read a file from disk", k=3)
    assert results
    assert results[0].id == "file_operations.read"


def test_search_ranks_better_match_first():
    reg = _registry()
    results = reg.search("show me the diff in the git repository", k=3)
    assert results[0].id == "git.diff"


def test_search_returns_empty_on_capability_absence():
    # No email/calendar tool exists → the registry must say "absent", which is
    # what lets comprehension honestly refuse instead of hallucinating a plan.
    reg = _registry()
    assert reg.search("send an email to my boss", k=3) == []


def test_search_ignores_stopwords():
    # A query of only stopwords must not spuriously match everything.
    reg = _registry()
    assert reg.search("the a an of to and", k=3) == []


def test_has_capability_presence_check():
    reg = _registry()
    assert reg.has_capability("read a file") is True
    assert reg.has_capability("send an email") is False
