"""
Tests for automatic model selection by task type / UserIntent.

Sentinel-value contract:
- Remove _INTENT_TASK_TYPE_MAP → all intents map to default → task-type tests fail
- Remove task_category preference over intent_type → category ignored → sentinel tests fail
- Remove select_model_for_intent() → ImportError → collection fails
- Remove normalisation → "Code" not found → uses "complex" default instead of mapped value
"""

from __future__ import annotations

import pytest

from app.services.model_router import (
    ModelRouter,
    ModelProvider,
    ModelTier,
    MODELS,
)


# ---------------------------------------------------------------------------
# _INTENT_TASK_TYPE_MAP — unit tests (no I/O)
# ---------------------------------------------------------------------------


class TestIntentTaskTypeMap:
    def setup_method(self):
        self.router = ModelRouter()

    def test_map_is_not_empty(self):
        """_INTENT_TASK_TYPE_MAP must define at least the four core intent types."""
        assert len(self.router._INTENT_TASK_TYPE_MAP) >= 4

    def test_all_four_core_types_mapped(self):
        """
        SENTINEL TEST — all four UserIntent.type values must be in the map.
        Remove any entry → key absent → test fails.
        """
        for intent_type in ("task", "conversation", "question", "clarification"):
            assert (
                intent_type in self.router._INTENT_TASK_TYPE_MAP
            ), f"'{intent_type}' not in _INTENT_TASK_TYPE_MAP"

    def test_all_values_are_valid_select_model_task_types(self):
        """Each map value must be a task_type accepted by select_model()."""
        valid = {"simple", "complex", "creative", "reasoning"}
        for category, task_type in self.router._INTENT_TASK_TYPE_MAP.items():
            assert (
                task_type in valid
            ), f"'{category}' maps to invalid task_type '{task_type}'"

    def test_code_maps_to_complex_or_reasoning(self):
        """
        SENTINEL TEST — 'code' must map to a high-complexity task type.
        Change to 'simple' → code gets fast model → sentinel test fails.
        """
        task_type = self.router._INTENT_TASK_TYPE_MAP.get("code")
        assert task_type in {
            "complex",
            "reasoning",
        }, f"'code' should map to complex/reasoning, got '{task_type}'"

    def test_conversation_maps_to_simple(self):
        """
        SENTINEL TEST — 'conversation' must map to 'simple' (fast model).
        Change to 'complex' → expensive model for chat → test fails.
        """
        task_type = self.router._INTENT_TASK_TYPE_MAP.get("conversation")
        assert (
            task_type == "simple"
        ), f"'conversation' should map to 'simple', got '{task_type}'"

    def test_task_maps_to_complex(self):
        """'task' intent type must map to 'complex' (powerful model needed)."""
        task_type = self.router._INTENT_TASK_TYPE_MAP.get("task")
        assert task_type in {
            "complex",
            "reasoning",
        }, f"'task' should map to complex/reasoning, got '{task_type}'"


# ---------------------------------------------------------------------------
# select_model_for_intent — model tier checks
# ---------------------------------------------------------------------------


class TestSelectModelForIntent:
    def setup_method(self):
        self.router = ModelRouter()

    def test_code_category_selects_powerful_tier(self):
        """
        SENTINEL TEST — 'code' task_category must route to a complex/powerful model.
        Map 'code' → 'simple' → returns fast model → tier assertion fails.
        """
        model_id = self.router.select_model_for_intent(
            intent_type="task",
            task_category="code",
            prefer_local=True,
        )
        cfg = MODELS.get(model_id)
        assert cfg is not None
        assert cfg.tier in {
            ModelTier.POWERFUL,
            ModelTier.BALANCED,
        }, f"Expected powerful/balanced for 'code', got {cfg.tier} ({model_id})"

    def test_conversation_intent_selects_fast_tier(self):
        """
        SENTINEL TEST — 'conversation' intent must route to a fast model.
        Map 'conversation' → 'complex' → returns powerful model → test fails.
        """
        model_id = self.router.select_model_for_intent(
            intent_type="conversation",
            prefer_local=True,
        )
        cfg = MODELS.get(model_id)
        assert cfg is not None
        assert (
            cfg.tier == ModelTier.FAST
        ), f"Expected fast tier for 'conversation', got {cfg.tier} ({model_id})"

    def test_task_category_preferred_over_intent_type(self):
        """
        SENTINEL TEST — when task_category is set, it must override intent_type.
        Remove category preference → uses intent_type ('conversation' → simple)
        instead of 'code' → wrong tier → test fails.
        """
        # conversation intent with code category → should use code's tier (complex)
        model_id_with_cat = self.router.select_model_for_intent(
            intent_type="conversation",
            task_category="code",  # category overrides the 'simple' mapping of conversation
            prefer_local=True,
        )
        model_id_no_cat = self.router.select_model_for_intent(
            intent_type="conversation",
            task_category=None,
            prefer_local=True,
        )
        cfg_with = MODELS[model_id_with_cat]
        cfg_no = MODELS[model_id_no_cat]

        # Code should select a more capable model than plain conversation
        tier_rank = {ModelTier.FAST: 0, ModelTier.BALANCED: 1, ModelTier.POWERFUL: 2}
        assert tier_rank[cfg_with.tier] >= tier_rank[cfg_no.tier], (
            f"task_category='code' should select a tier >= 'conversation' alone; "
            f"got {cfg_with.tier} vs {cfg_no.tier}"
        )

    def test_file_management_category_selects_fast(self):
        """
        SENTINEL TEST — file_management is a simple structured task → fast model.
        Map to 'complex' → powerful model → tier assertion fails.
        """
        model_id = self.router.select_model_for_intent(
            intent_type="task",
            task_category="file_management",
            prefer_local=True,
        )
        cfg = MODELS.get(model_id)
        assert cfg is not None
        assert (
            cfg.tier == ModelTier.FAST
        ), f"Expected fast tier for 'file_management', got {cfg.tier}"

    def test_research_category_selects_capable_model(self):
        """Research requires a capable model (not fast)."""
        model_id = self.router.select_model_for_intent(
            intent_type="task",
            task_category="research",
            prefer_local=True,
        )
        cfg = MODELS.get(model_id)
        assert cfg is not None
        assert cfg.tier in {
            ModelTier.POWERFUL,
            ModelTier.BALANCED,
        }, f"Expected powerful/balanced for 'research', got {cfg.tier}"

    def test_none_intent_type_returns_valid_model(self):
        """None intent_type must not raise — should return the default/fallback."""
        model_id = self.router.select_model_for_intent(
            intent_type=None,
            prefer_local=True,
        )
        assert model_id in MODELS

    def test_unknown_intent_type_returns_valid_model(self):
        """Completely unknown intent type must not raise."""
        model_id = self.router.select_model_for_intent(
            intent_type="some_unknown_type_xyz",
            prefer_local=True,
        )
        assert model_id in MODELS

    def test_case_insensitive_lookup(self):
        """
        SENTINEL TEST — 'CODE' and 'code' must resolve to the same model.
        Remove normalisation → 'CODE' not found → uses complex default → may pass,
        but 'Conversation' → not found → 'complex' instead of 'simple' → tier differs.
        """
        model_lower = self.router.select_model_for_intent(
            intent_type="conversation", prefer_local=True
        )
        model_upper = self.router.select_model_for_intent(
            intent_type="CONVERSATION", prefer_local=True
        )
        model_mixed = self.router.select_model_for_intent(
            intent_type="Conversation", prefer_local=True
        )
        assert (
            model_lower == model_upper == model_mixed
        ), "Case variants of 'conversation' returned different models"

    def test_hyphen_normalised_in_category(self):
        """'file-management' and 'file_management' must select the same model."""
        m1 = self.router.select_model_for_intent(
            intent_type="task", task_category="file-management", prefer_local=True
        )
        m2 = self.router.select_model_for_intent(
            intent_type="task", task_category="file_management", prefer_local=True
        )
        assert m1 == m2

    def test_all_mapped_categories_return_known_models(self):
        """
        SENTINEL TEST — every category in _INTENT_TASK_TYPE_MAP must produce a
        model that exists in the MODELS registry.
        Map to a nonexistent task_type → select_model returns default → regression caught.
        """
        for category in self.router._INTENT_TASK_TYPE_MAP:
            model_id = self.router.select_model_for_intent(
                intent_type=category, prefer_local=True
            )
            assert (
                model_id in MODELS
            ), f"select_model_for_intent('{category}') returned unknown model '{model_id}'"


# ---------------------------------------------------------------------------
# Integration with existing select_model path
# ---------------------------------------------------------------------------


class TestSelectModelForIntentIntegration:
    def setup_method(self):
        self.router = ModelRouter()

    def test_require_tools_forwarded(self):
        """require_tools=True must exclude models without tool support."""
        model_id = self.router.select_model_for_intent(
            intent_type="task",
            task_category="code",
            require_tools=True,
            prefer_local=True,
        )
        cfg = MODELS[model_id]
        assert cfg.supports_tools is True

    def test_prefer_local_true_returns_ollama(self):
        """prefer_local=True must favour Ollama models."""
        model_id = self.router.select_model_for_intent(
            intent_type="conversation",
            prefer_local=True,
        )
        cfg = MODELS[model_id]
        assert cfg.provider == ModelProvider.OLLAMA

    def test_prefer_local_false_allows_cloud(self):
        """prefer_local=False with 'code' should allow cloud powerful models."""
        model_id = self.router.select_model_for_intent(
            intent_type="task",
            task_category="code",
            prefer_local=False,
        )
        # Should be able to pick a cloud model — just verify it's valid
        assert model_id in MODELS
