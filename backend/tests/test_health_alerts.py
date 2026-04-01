"""
Tests for health failure alerting webhooks in CORE.

Sentinel-value contract:
- Remove fire_health_alert() call from get_full_health() → alert never fires → test fails
- Remove health.degraded / health.unhealthy from WebhookEvent → enum missing → test fails
- Remove failed_checks from payload → payload lacks check details → test fails
- Remove exception guard → webhook failure crashes health endpoint → test fails
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.webhook_service import WebhookEvent
from app.core.health import HealthStatus, fire_health_alert


# ---------------------------------------------------------------------------
# WebhookEvent — new health alert events
# ---------------------------------------------------------------------------

class TestHealthWebhookEvents:
    def test_health_degraded_event_exists(self):
        """
        SENTINEL TEST — WebhookEvent.HEALTH_DEGRADED must exist.
        Remove it → AttributeError → test fails.
        """
        assert hasattr(WebhookEvent, "HEALTH_DEGRADED")
        assert WebhookEvent.HEALTH_DEGRADED.value == "health.degraded"

    def test_health_unhealthy_event_exists(self):
        """
        SENTINEL TEST — WebhookEvent.HEALTH_UNHEALTHY must exist.
        Remove it → AttributeError → test fails.
        """
        assert hasattr(WebhookEvent, "HEALTH_UNHEALTHY")
        assert WebhookEvent.HEALTH_UNHEALTHY.value == "health.unhealthy"

    def test_health_events_are_webhook_event_instances(self):
        assert isinstance(WebhookEvent.HEALTH_DEGRADED, WebhookEvent)
        assert isinstance(WebhookEvent.HEALTH_UNHEALTHY, WebhookEvent)


# ---------------------------------------------------------------------------
# fire_health_alert unit tests
# ---------------------------------------------------------------------------

class TestFireHealthAlert:
    def _make_health_result(self, status: str = "degraded") -> dict:
        return {
            "status": status,
            "timestamp": "2026-04-01T12:00:00",
            "summary": {"total": 3, "healthy": 2, "degraded": 1, "unhealthy": 0},
            "checks": [
                {"name": "database", "status": "healthy"},
                {"name": "redis", "status": "degraded"},
                {"name": "ollama", "status": "healthy"},
            ],
        }

    @pytest.mark.asyncio
    async def test_degraded_fires_health_degraded_event(self):
        """
        SENTINEL TEST — DEGRADED status must fire HEALTH_DEGRADED event.
        Fire HEALTH_UNHEALTHY instead → event value wrong → test fails.
        """
        mock_service = MagicMock()
        mock_service.fire = AsyncMock()

        with patch("app.services.webhook_service.get_webhook_service", return_value=mock_service):
            await fire_health_alert(HealthStatus.DEGRADED, self._make_health_result("degraded"))

        mock_service.fire.assert_called_once()
        call_kwargs = mock_service.fire.call_args[1]
        assert call_kwargs["event"] == WebhookEvent.HEALTH_DEGRADED

    @pytest.mark.asyncio
    async def test_unhealthy_fires_health_unhealthy_event(self):
        """
        SENTINEL TEST — UNHEALTHY status must fire HEALTH_UNHEALTHY event.
        Fire HEALTH_DEGRADED instead → event value wrong → test fails.
        """
        mock_service = MagicMock()
        mock_service.fire = AsyncMock()

        with patch("app.services.webhook_service.get_webhook_service", return_value=mock_service):
            await fire_health_alert(HealthStatus.UNHEALTHY, self._make_health_result("unhealthy"))

        call_kwargs = mock_service.fire.call_args[1]
        assert call_kwargs["event"] == WebhookEvent.HEALTH_UNHEALTHY

    @pytest.mark.asyncio
    async def test_payload_contains_status(self):
        """
        SENTINEL TEST — webhook payload must contain the overall status.
        Remove status from payload → monitoring cannot determine severity → test fails.
        """
        mock_service = MagicMock()
        mock_service.fire = AsyncMock()

        with patch("app.services.webhook_service.get_webhook_service", return_value=mock_service):
            await fire_health_alert(HealthStatus.DEGRADED, self._make_health_result("degraded"))

        payload = mock_service.fire.call_args[1]["payload"]
        assert payload["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_payload_contains_failed_checks(self):
        """
        SENTINEL TEST — payload must include checks that are NOT healthy.
        Remove failed_checks filtering → empty list → test fails.
        """
        mock_service = MagicMock()
        mock_service.fire = AsyncMock()

        health_result = self._make_health_result("degraded")
        with patch("app.services.webhook_service.get_webhook_service", return_value=mock_service):
            await fire_health_alert(HealthStatus.DEGRADED, health_result)

        payload = mock_service.fire.call_args[1]["payload"]
        # Only the degraded 'redis' check should be in failed_checks
        failed = payload["failed_checks"]
        assert len(failed) == 1
        assert failed[0]["name"] == "redis"

    @pytest.mark.asyncio
    async def test_payload_contains_summary(self):
        """Payload must include the health summary dict."""
        mock_service = MagicMock()
        mock_service.fire = AsyncMock()

        with patch("app.services.webhook_service.get_webhook_service", return_value=mock_service):
            await fire_health_alert(HealthStatus.DEGRADED, self._make_health_result())

        payload = mock_service.fire.call_args[1]["payload"]
        assert "summary" in payload
        assert payload["summary"]["degraded"] == 1

    @pytest.mark.asyncio
    async def test_fire_failure_does_not_raise(self):
        """
        SENTINEL TEST — if the webhook service raises, fire_health_alert must NOT propagate.
        Remove the try/except → test raises RuntimeError → test fails.
        """
        mock_service = MagicMock()
        mock_service.fire = AsyncMock(side_effect=RuntimeError("webhook down"))

        with patch("app.services.webhook_service.get_webhook_service", return_value=mock_service):
            # Must not raise
            await fire_health_alert(HealthStatus.UNHEALTHY, self._make_health_result("unhealthy"))

    @pytest.mark.asyncio
    async def test_import_failure_does_not_raise(self):
        """If webhook_service cannot be imported, alert silently fails."""
        with patch("app.services.webhook_service.get_webhook_service",
                   side_effect=RuntimeError("import failed")):
            # Must not raise
            await fire_health_alert(HealthStatus.DEGRADED, self._make_health_result())


# ---------------------------------------------------------------------------
# get_full_health fires alert on non-healthy status
# ---------------------------------------------------------------------------

class TestGetFullHealthFiresAlert:
    def _ok_check(self, name: str):
        from app.core.health import HealthCheck, HealthStatus
        return AsyncMock(return_value=HealthCheck(name=name, status=HealthStatus.HEALTHY))

    def _failing_check(self, name: str):
        from app.core.health import HealthCheck, HealthStatus
        return AsyncMock(return_value=HealthCheck(name=name, status=HealthStatus.UNHEALTHY, message="down"))

    @pytest.mark.asyncio
    async def test_unhealthy_check_triggers_alert(self):
        """
        SENTINEL TEST — when any check is UNHEALTHY, an alert task must be created.
        Remove asyncio.create_task call → no alert → spy not called → test fails.
        """
        from app.core import health as health_mod

        with patch("app.core.health.check_database", self._failing_check("database")), \
             patch("app.core.health.check_ollama", self._ok_check("ollama")), \
             patch("app.core.health.check_redis", self._ok_check("redis")), \
             patch("app.core.health.check_websocket_manager", self._ok_check("ws")), \
             patch("app.core.health.check_engine_state", self._ok_check("engine")), \
             patch("app.core.health.asyncio.create_task") as task_spy:

            result = await health_mod.get_full_health()

        assert task_spy.called, "asyncio.create_task was not called for unhealthy state"
        assert result["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_all_healthy_does_not_trigger_alert(self):
        """
        SENTINEL TEST — when all checks pass, NO alert must be created.
        Trigger alert always → create_task called → test fails.
        """
        from app.core import health as health_mod

        with patch("app.core.health.check_database", self._ok_check("db")), \
             patch("app.core.health.check_ollama", self._ok_check("ollama")), \
             patch("app.core.health.check_redis", self._ok_check("redis")), \
             patch("app.core.health.check_websocket_manager", self._ok_check("ws")), \
             patch("app.core.health.check_engine_state", self._ok_check("engine")), \
             patch("app.core.health.asyncio.create_task") as task_spy:

            result = await health_mod.get_full_health()

        task_spy.assert_not_called()
        assert result["status"] == "healthy"
