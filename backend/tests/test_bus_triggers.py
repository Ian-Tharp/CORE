"""
Tests for BusTriggerService — rule management, pattern matching, evaluation, execution.

Sentinel-value contract:
- Remove regex matching in TriggerRule.matches → patterns never fire → test fails
- Remove disabled-rule skip in evaluate_message → disabled triggers fire → test fails
- Remove error isolation in execute_triggers → one bad trigger kills others → test fails
- Remove _extract_text topic combination → topic-based triggers never match → test fails
- Remove default trigger loading → no auto-council on "deliberate" → test fails
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.bus_models import BusMessage, MessageType
from app.services.bus_triggers import (
    BusTriggerService,
    TriggerAction,
    TriggerRule,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_service() -> BusTriggerService:
    return BusTriggerService()


def _make_message(text: str = "", topic: str | None = None) -> BusMessage:
    payload = {"text": text} if text else {}
    return BusMessage(
        sender_id="test-agent",
        message_type=MessageType.BROADCAST,
        topic=topic,
        payload=payload,
    )


def _make_rule(
    name: str = "test",
    pattern: str = r"\bsentinel\b",
    action: TriggerAction = TriggerAction.COUNCIL_SESSION,
    enabled: bool = True,
) -> TriggerRule:
    return TriggerRule(name=name, pattern=pattern, action=action, enabled=enabled)


# ===========================================================================
# TriggerRule.matches — pure pattern logic
# ===========================================================================


class TestTriggerRuleMatches:
    def test_matching_text_returns_true(self):
        """
        SENTINEL — a matching text must return True.
        Remove re.search → always returns False → test fails.
        """
        rule = _make_rule(pattern=r"\bdeliberate\b")
        assert rule.matches("please deliberate on this")

    def test_non_matching_text_returns_false(self):
        """Text that doesn't match the pattern must return False."""
        rule = _make_rule(pattern=r"\bdeliberate\b")
        assert not rule.matches("just do it")

    def test_match_is_case_insensitive(self):
        """
        SENTINEL — regex matching must be case-insensitive (re.IGNORECASE).
        Remove IGNORECASE → "DELIBERATE" doesn't match → test fails.
        """
        rule = _make_rule(pattern=r"\bdeliberate\b")
        assert rule.matches("DELIBERATE on this")

    def test_invalid_regex_returns_false(self):
        """Invalid regex pattern must return False (not raise)."""
        rule = TriggerRule(
            name="bad", pattern="[invalid", action=TriggerAction.COUNCIL_SESSION
        )
        assert rule.matches("anything") is False


# ===========================================================================
# BusTriggerService — CRUD
# ===========================================================================


class TestTriggerCRUD:
    def test_defaults_loaded_on_init(self):
        """
        SENTINEL — default triggers (deliberate, brainstorm) must be present on init.
        Remove _load_defaults → no auto-reactions to bus messages → test fails.
        """
        svc = _make_service()
        rules = svc.list_triggers()
        patterns = [r.pattern for r in rules]
        assert any("deliberate" in p for p in patterns)
        assert any("brainstorm" in p or "ideate" in p for p in patterns)

    def test_register_trigger_returns_id(self):
        """register_trigger must return the rule's id."""
        svc = _make_service()
        rule = _make_rule()
        returned_id = svc.register_trigger(rule)
        assert returned_id == rule.id

    def test_registered_trigger_is_retrievable(self):
        """A registered trigger must be findable by id."""
        svc = _make_service()
        rule = _make_rule(name="SENTINEL_RULE")
        svc.register_trigger(rule)
        found = svc.get_trigger(rule.id)
        assert found is not None
        assert found.name == "SENTINEL_RULE"

    def test_invalid_regex_raises_on_register(self):
        """Registering a trigger with invalid regex must raise ValueError."""
        svc = _make_service()
        rule = TriggerRule(
            name="bad", pattern="[invalid", action=TriggerAction.COUNCIL_SESSION
        )
        with pytest.raises(ValueError, match="[Ii]nvalid regex"):
            svc.register_trigger(rule)

    def test_remove_existing_trigger_returns_true(self):
        """
        SENTINEL — removing an existing trigger must return True.
        Return False always → callers think deletion failed → test fails.
        """
        svc = _make_service()
        rule = _make_rule()
        svc.register_trigger(rule)
        assert svc.remove_trigger(rule.id) is True
        assert svc.get_trigger(rule.id) is None

    def test_remove_nonexistent_trigger_returns_false(self):
        """Removing a non-existent trigger must return False."""
        svc = _make_service()
        assert svc.remove_trigger("nonexistent-id") is False

    def test_register_overwrites_existing_trigger(self):
        """Re-registering with same id must overwrite (upsert)."""
        svc = _make_service()
        rule_v1 = _make_rule(name="v1")
        rule_v2 = TriggerRule(
            id=rule_v1.id,
            name="v2",
            pattern=r"\bsentinel\b",
            action=TriggerAction.COUNCIL_SESSION,
        )
        svc.register_trigger(rule_v1)
        svc.register_trigger(rule_v2)
        found = svc.get_trigger(rule_v1.id)
        assert found.name == "v2"


# ===========================================================================
# _extract_text — text extraction
# ===========================================================================


class TestExtractText:
    def test_topic_included_in_text(self):
        """
        SENTINEL — topic must be included in the matchable text.
        Remove topic from _extract_text → topic-based triggers never match → test fails.
        """
        svc = _make_service()
        msg = _make_message(topic="SENTINEL_TOPIC")
        text = svc._extract_text(msg)
        assert "SENTINEL_TOPIC" in text

    def test_payload_text_included(self):
        """payload['text'] must be included in the matchable text."""
        svc = _make_service()
        msg = _make_message(text="SENTINEL_BODY")
        text = svc._extract_text(msg)
        assert "SENTINEL_BODY" in text

    def test_payload_content_included(self):
        """payload['content'] must also be included."""
        svc = _make_service()
        msg = BusMessage(
            sender_id="a",
            message_type=MessageType.BROADCAST,
            payload={"content": "SENTINEL_CONTENT"},
        )
        text = svc._extract_text(msg)
        assert "SENTINEL_CONTENT" in text

    def test_empty_message_returns_empty_string(self):
        """Message with no text fields must return empty string (not crash)."""
        svc = _make_service()
        msg = BusMessage(sender_id="a", message_type=MessageType.BROADCAST)
        text = svc._extract_text(msg)
        assert text == ""


# ===========================================================================
# evaluate_message — dry-run matching
# ===========================================================================


class TestEvaluateMessage:
    @pytest.mark.asyncio
    async def test_matching_trigger_appears_in_evaluation(self):
        """
        SENTINEL — a matching trigger must appear in matched_triggers.
        Remove rule.matches() call → evaluation always empty → test fails.
        """
        svc = _make_service()
        rule = _make_rule(pattern=r"\bsentinel\b", name="SENTINEL_TRIGGER")
        svc.register_trigger(rule)
        msg = _make_message(text="please sentinel this")
        eval_result = await svc.evaluate_message(msg)
        matched_names = [t["trigger_name"] for t in eval_result.matched_triggers]
        assert "SENTINEL_TRIGGER" in matched_names

    @pytest.mark.asyncio
    async def test_non_matching_trigger_not_in_evaluation(self):
        """A trigger that doesn't match must NOT appear in evaluation."""
        svc = _make_service()
        # Start fresh by clearing defaults
        svc._triggers.clear()
        rule = _make_rule(pattern=r"\bsentinel\b")
        svc.register_trigger(rule)
        msg = _make_message(text="something else entirely")
        eval_result = await svc.evaluate_message(msg)
        assert len(eval_result.matched_triggers) == 0

    @pytest.mark.asyncio
    async def test_disabled_trigger_is_skipped(self):
        """
        SENTINEL — disabled triggers must not appear in evaluation results.
        Remove the `if not rule.enabled: continue` check → disabled triggers fire → test fails.
        """
        svc = _make_service()
        svc._triggers.clear()
        rule = _make_rule(pattern=r"\bsentinel\b", enabled=False, name="DISABLED")
        svc.register_trigger(rule)
        msg = _make_message(text="sentinel this")
        eval_result = await svc.evaluate_message(msg)
        matched_names = [t["trigger_name"] for t in eval_result.matched_triggers]
        assert "DISABLED" not in matched_names

    @pytest.mark.asyncio
    async def test_total_triggers_checked_is_accurate(self):
        """total_triggers_checked must equal the number of registered triggers."""
        svc = _make_service()
        svc._triggers.clear()
        for i in range(3):
            svc.register_trigger(_make_rule(name=f"rule-{i}", pattern=rf"\bword{i}\b"))
        msg = _make_message(text="irrelevant")
        eval_result = await svc.evaluate_message(msg)
        assert eval_result.total_triggers_checked == 3


# ===========================================================================
# execute_triggers — action dispatch and error isolation
# ===========================================================================


class TestExecuteTriggers:
    @pytest.mark.asyncio
    async def test_no_matching_triggers_returns_empty_list(self):
        """If no triggers match, execute_triggers must return []."""
        svc = _make_service()
        svc._triggers.clear()
        msg = _make_message(text="nothing matches")
        results = await svc.execute_triggers(msg)
        assert results == []

    @pytest.mark.asyncio
    async def test_matching_trigger_produces_result(self):
        """
        SENTINEL — a matching trigger must produce a TriggerResult in the output.
        Remove result appending → matched triggers silently ignored → test fails.
        """
        svc = _make_service()
        svc._triggers.clear()
        rule = _make_rule(pattern=r"\bsentinel\b", name="SENTINEL_RULE")
        svc.register_trigger(rule)

        with patch.object(
            svc, "_execute_action", new=AsyncMock(return_value={"session_id": "s-1"})
        ):
            results = await svc.execute_triggers(_make_message(text="sentinel fires"))

        assert len(results) == 1
        assert results[0].trigger_name == "SENTINEL_RULE"
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_failed_trigger_records_error_not_raises(self):
        """
        SENTINEL — if an action raises, result must have success=False with error message.
        Remove try/except → one bad trigger kills all subsequent triggers → test fails.
        """
        svc = _make_service()
        svc._triggers.clear()
        rule = _make_rule(pattern=r"\bsentinel\b", name="FAILING_RULE")
        svc.register_trigger(rule)

        with patch.object(
            svc,
            "_execute_action",
            new=AsyncMock(side_effect=RuntimeError("SENTINEL_ERROR")),
        ):
            results = await svc.execute_triggers(_make_message(text="sentinel fires"))

        assert len(results) == 1
        assert results[0].success is False
        assert "SENTINEL_ERROR" in results[0].error

    @pytest.mark.asyncio
    async def test_one_failed_trigger_does_not_block_others(self):
        """
        SENTINEL — a failing trigger must not prevent subsequent triggers from running.
        Remove error isolation → second trigger never runs → test fails.
        """
        svc = _make_service()
        svc._triggers.clear()

        rule1 = _make_rule(name="FAIL_RULE", pattern=r"\bsentinel\b")
        rule2 = _make_rule(
            name="OK_RULE", pattern=r"\bsentinel\b", action=TriggerAction.CATALYST_RUN
        )
        svc.register_trigger(rule1)
        svc.register_trigger(rule2)

        call_count = 0

        async def _mixed_execute(rule, message, text):
            nonlocal call_count
            call_count += 1
            if rule.name == "FAIL_RULE":
                raise RuntimeError("first trigger fails")
            return {"result": "ok"}

        with patch.object(svc, "_execute_action", side_effect=_mixed_execute):
            results = await svc.execute_triggers(_make_message(text="sentinel fires"))

        assert call_count == 2  # Both rules attempted
        assert len(results) == 2
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]
        assert len(successes) == 1
        assert len(failures) == 1
