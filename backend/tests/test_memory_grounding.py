"""
Unit tests for the Phase 3 memory-grounding component (app/core/memory/grounding.py).

Pins:
- cue detection fires on strong "you should know this" phrases only (a bare
  "my"/"our" is not a cue, so self-contained requests don't clarify);
- StoreMemoryGrounder scores lexically over an InMemoryStore, scopes by user,
  falls back to the global namespace, respects min_score / k, and never raises;
- assess_knowledge marks a cue as a gap only when no hit covers its topic;
- apply_to_intent feeds gaps into ambiguities (actionable intents only) and
  memory into context_retrieved;
- the feature flag is off by default.
"""

from __future__ import annotations

import pytest
from langgraph.store.memory import InMemoryStore

from app.core.agents.comprehension_gate import decide_gate
from app.core.memory.grounding import (
    KnowledgeAssessment,
    MemoryHit,
    NullMemoryGrounder,
    StoreMemoryGrounder,
    apply_to_intent,
    assess_knowledge,
    build_memory_store,
    detect_knowledge_dependencies,
    memory_grounding_enabled,
)
from app.core.registry.capability import CapabilityEntry, CapabilityRegistry
from app.models.core_state import UserIntent


@pytest.fixture()
def grounder():
    return StoreMemoryGrounder(InMemoryStore())


def _phrases(text):
    return [c.phrase for c in detect_knowledge_dependencies(text)]


# --------------------------------------------------------------------------
# Flag
# --------------------------------------------------------------------------


class TestFlag:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv("CORE_COMPREHENSION_MEMORY_GROUNDING", raising=False)
        assert memory_grounding_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
    def test_truthy_values_enable(self, monkeypatch, value):
        monkeypatch.setenv("CORE_COMPREHENSION_MEMORY_GROUNDING", value)
        assert memory_grounding_enabled() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "off", "nope"])
    def test_other_values_disable(self, monkeypatch, value):
        monkeypatch.setenv("CORE_COMPREHENSION_MEMORY_GROUNDING", value)
        assert memory_grounding_enabled() is False


# --------------------------------------------------------------------------
# Cue detection
# --------------------------------------------------------------------------


class TestDetectKnowledgeDependencies:
    @pytest.mark.parametrize(
        "text, phrase",
        [
            ("deploy the service like last time", "like last time"),
            ("format it the usual way", "the usual"),
            ("as we discussed, ship it", "as we discussed"),
            ("remember my deadline?", "remember my"),
            ("what did you recommend yesterday", None),
        ],
    )
    def test_recall_cues(self, text, phrase):
        found = _phrases(text)
        if phrase is None:
            assert found == []
        else:
            assert phrase in found

    def test_preference_cue_carries_topic(self):
        cues = detect_knowledge_dependencies("scaffold an app with my preferred stack")
        assert len(cues) == 1
        assert "stack" in cues[0].topic

    def test_personal_fact_cue_carries_topic(self):
        cues = detect_knowledge_dependencies("email my manager the report")
        assert [c.phrase for c in cues] == ["my manager"]
        assert cues[0].topic == frozenset({"manager"})

    def test_agreed_on_cue_carries_topic(self):
        cues = detect_knowledge_dependencies("use the schema we agreed on yesterday")
        assert cues and "yesterday" in cues[0].topic

    @pytest.mark.parametrize(
        "text",
        [
            "fix my code in main.py",
            "list the files in our repo root",
            "what is the capital of France?",
            "",
            "   ",
        ],
    )
    def test_self_contained_requests_have_no_cues(self, text):
        assert detect_knowledge_dependencies(text) == []

    def test_overlapping_matches_are_not_duplicated(self):
        # "like last time" contains "last time": only the longer phrase counts.
        assert _phrases("do it like last time") == ["like last time"]

    def test_deterministic(self):
        text = "as we discussed, deploy like last time with my usual config"
        assert _phrases(text) == _phrases(text)


# --------------------------------------------------------------------------
# StoreMemoryGrounder
# --------------------------------------------------------------------------


class TestStoreMemoryGrounder:
    def test_empty_store_returns_nothing(self, grounder):
        assert grounder.ground("deploy the api service") == []

    def test_returns_relevant_fact_with_score(self, grounder):
        grounder.remember("The api service deploys to staging via helm", user_id="u1")
        grounder.remember("User likes green tea", user_id="u1")

        hits = grounder.ground("deploy the api service", user_id="u1")

        assert len(hits) == 1
        hit = hits[0]
        assert isinstance(hit, MemoryHit)
        assert "helm" in hit.content
        assert hit.kind == "fact"
        assert 0.0 < hit.score <= 1.0

    def test_stemming_matches_inflections(self, grounder):
        grounder.remember("deployed the billing service", user_id="u1")
        assert grounder.ground("deploy billing", user_id="u1")

    def test_cue_words_do_not_score(self, grounder):
        grounder.remember("last time we ate pizza", user_id="u1")
        # Only cue words in the query → no scoring tokens → no hits.
        assert grounder.ground("like last time", user_id="u1") == []

    def test_inflected_cue_words_are_not_scored(self, grounder):
        from app.core.memory.grounding import _query_tokens

        tokens = _query_tokens("as we discussed, use my preferred stack as decided")
        assert tokens == {"use", "stack"}
        grounder.remember("I preferred tea, we discussed and decided it")
        assert grounder.ground("as we discussed, my preferred stack") == []

    def test_hits_ranked_best_first_and_capped_at_k(self, grounder):
        grounder.remember("api service", user_id="u1", key="a")
        grounder.remember("api service deploy helm", user_id="u1", key="b")
        grounder.remember("api", user_id="u1", key="c")

        hits = grounder.ground("deploy api service helm", user_id="u1", k=2)

        assert [h.key for h in hits] == ["b", "a"]

    def test_min_score_filters_weak_matches(self):
        g = StoreMemoryGrounder(InMemoryStore(), min_score=0.6)
        g.remember("api", user_id="u1")
        assert g.ground("deploy api service helm", user_id="u1") == []

    def test_user_scoping(self, grounder):
        grounder.remember("u2 deploys with terraform", user_id="u2")
        assert grounder.ground("deploy with terraform", user_id="u1") == []
        assert grounder.ground("deploy with terraform", user_id="u2")

    def test_global_facts_visible_to_every_user(self, grounder):
        grounder.remember("company stack is fastapi and angular")  # global
        assert grounder.ground("what stack do we use", user_id="anyone")
        assert grounder.ground("what stack do we use")  # anonymous

    def test_episode_kind_preserved(self, grounder):
        grounder.remember("ran migration 042 on staging", kind="episode", user_id="u1")
        hits = grounder.ground("rerun the staging migration", user_id="u1")
        assert hits and hits[0].kind == "episode"

    def test_store_errors_mean_no_hits(self):
        class Broken:
            def search(self, *a, **kw):
                raise RuntimeError("db down")

        assert StoreMemoryGrounder(Broken()).ground("deploy api") == []

    def test_store_semantic_score_is_used_when_present(self):
        class Item:
            def __init__(self):
                self.namespace = ("core_memory", "global", "fact")
                self.key = "k"
                self.value = {"content": "unrelated words entirely"}
                self.score = 0.9

        class Scored:
            def search(self, *a, **kw):
                return [Item()]

        hits = StoreMemoryGrounder(Scored()).ground("deploy api")
        assert hits and hits[0].score == pytest.approx(0.9)

    def test_k_zero_and_null_grounder(self, grounder):
        grounder.remember("api service", user_id="u1")
        assert grounder.ground("api service", user_id="u1", k=0) == []
        assert NullMemoryGrounder().ground("anything") == []


# --------------------------------------------------------------------------
# assess_knowledge
# --------------------------------------------------------------------------


class TestAssessKnowledge:
    def test_cue_without_memory_is_a_gap(self, grounder):
        a = assess_knowledge("deploy the api like last time", grounder, user_id="u1")
        assert [g.phrase for g in a.gaps] == ["like last time"]
        assert a.has_gaps

    def test_cue_covered_by_relevant_memory(self, grounder):
        grounder.remember("deployed the api with helm to staging", user_id="u1")
        a = assess_knowledge("deploy the api like last time", grounder, user_id="u1")
        assert a.gaps == []
        assert a.hits

    def test_topic_cue_needs_topic_in_memory(self, grounder):
        grounder.remember(
            "my preferred editor is vim, scaffold apps fast", user_id="u1"
        )
        a = assess_knowledge(
            "scaffold an app with my preferred stack", grounder, user_id="u1"
        )
        # A hit exists ("scaffold", "app") but nothing about a stack.
        assert a.hits
        assert [g.phrase for g in a.gaps] == ["my preferred stack"]

        grounder.remember("preferred stack: fastapi + angular", user_id="u1")
        a = assess_knowledge(
            "scaffold an app with my preferred stack", grounder, user_id="u1"
        )
        assert a.gaps == []

    def test_no_cues_no_gaps_even_with_empty_memory(self, grounder):
        a = assess_knowledge("read main.py and summarise it", grounder)
        assert a.cues == [] and a.gaps == []

    def test_to_dict_shape(self, grounder):
        grounder.remember("api deploys via helm", user_id="u1")
        d = assess_knowledge(
            "deploy api like last time", grounder, user_id="u1"
        ).to_dict()
        assert set(d) == {"hits", "cues", "gaps"}
        assert d["cues"] == ["like last time"]
        assert d["hits"][0]["content"] == "api deploys via helm"


# --------------------------------------------------------------------------
# apply_to_intent + the existing gate
# --------------------------------------------------------------------------


def _intent(**kw):
    base = dict(
        type="task",
        description="deploy the api service",
        confidence=0.95,
        requires_tools=False,
    )
    base.update(kw)
    return UserIntent(**base)


def _registry():
    r = CapabilityRegistry()
    r.register(
        CapabilityEntry(id="deploy.run", name="deploy", description="deploy a service")
    )
    return r


class TestApplyToIntent:
    def test_gap_becomes_ambiguity_and_gate_clarifies(self, grounder):
        intent = _intent()
        assert decide_gate(intent, _registry()).outcome == "proceed"

        apply_to_intent(
            intent, assess_knowledge("deploy the api like last time", grounder)
        )

        decision = decide_gate(intent, _registry())
        assert decision.outcome == "clarify"
        assert "like last time" in decision.message

    def test_non_actionable_intent_gets_no_ambiguity(self, grounder):
        intent = _intent(type="conversation")
        apply_to_intent(intent, assess_knowledge("chat like last time", grounder))
        assert intent.ambiguities == []

    def test_hits_appended_to_context(self, grounder):
        grounder.remember("api deploys via helm", user_id="u1")
        intent = _intent(context_retrieved="KB context")
        apply_to_intent(intent, assess_knowledge("deploy api", grounder, user_id="u1"))
        assert intent.context_retrieved.startswith("KB context")
        assert "api deploys via helm" in intent.context_retrieved

    def test_empty_assessment_changes_nothing(self):
        intent = _intent()
        before = intent.model_dump()
        apply_to_intent(intent, KnowledgeAssessment())
        assert intent.model_dump() == before

    def test_none_intent_is_noop(self):
        apply_to_intent(None, KnowledgeAssessment())

    def test_ambiguity_not_duplicated(self, grounder):
        intent = _intent()
        a = assess_knowledge("deploy like last time", grounder)
        apply_to_intent(intent, a)
        apply_to_intent(intent, a)
        assert len(intent.ambiguities) == 1


class TestBuildMemoryStore:
    def test_default_is_in_memory(self, monkeypatch):
        monkeypatch.delenv("CORE_MEMORY_STORE", raising=False)
        assert isinstance(build_memory_store(), InMemoryStore)

    def test_postgres_without_dsn_falls_back(self, monkeypatch):
        monkeypatch.setenv("CORE_MEMORY_STORE", "postgres")
        monkeypatch.delenv("CORE_MEMORY_STORE_DSN", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert isinstance(build_memory_store(), InMemoryStore)

    def test_postgres_unavailable_falls_back(self, monkeypatch):
        # Unreachable DSN (or the optional package missing) must not raise.
        monkeypatch.setenv("CORE_MEMORY_STORE", "postgres")
        monkeypatch.setenv(
            "CORE_MEMORY_STORE_DSN",
            "postgresql://x:y@127.0.0.1:1/none?connect_timeout=1",
        )
        assert isinstance(build_memory_store(), InMemoryStore)
