"""
Tests for AuditService — fire-and-forget audit logging facade.

Sentinel-value contract:
- Remove try/except in log() → DB errors propagate to callers → test fails
- Remove ip extraction from request.client → IP never logged → test fails
- Remove correlation_id extraction from request.state → correlation lost → test fails
- Remove None return on failure → callers see exception instead of None → test fails
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.audit_service import AuditService


# ===========================================================================
# Helpers
# ===========================================================================


def _make_service() -> AuditService:
    return AuditService()


def _make_request(
    ip: str = "127.0.0.1", correlation_id: str | None = None
) -> MagicMock:
    """Build a minimal FastAPI Request-like mock."""
    req = MagicMock()
    req.client = MagicMock()
    req.client.host = ip
    req.state = MagicMock()
    req.state.correlation_id = correlation_id
    return req


# ===========================================================================
# log() — core behaviour
# ===========================================================================


class TestAuditLog:
    @pytest.mark.asyncio
    async def test_success_returns_event_id(self):
        """
        SENTINEL — successful log must return the UUID from the repository.
        Return None always → callers can't confirm the event was recorded → test fails.
        """
        svc = _make_service()
        with patch(
            "app.services.audit_service.repo.record",
            new=AsyncMock(return_value="SENTINEL_UUID"),
        ):
            result = await svc.log(actor="ian", action="test.action")
        assert result == "SENTINEL_UUID"

    @pytest.mark.asyncio
    async def test_db_error_returns_none_not_raises(self):
        """
        SENTINEL — repository failure must be swallowed and return None.
        Remove try/except → DB outage crashes API endpoint → test fails.
        """
        svc = _make_service()
        with patch(
            "app.services.audit_service.repo.record",
            new=AsyncMock(side_effect=RuntimeError("DB down")),
        ):
            result = await svc.log(actor="ian", action="test.action")
        assert result is None

    @pytest.mark.asyncio
    async def test_ip_extracted_from_request_client(self):
        """
        SENTINEL — IP must be extracted from request.client.host.
        Skip extraction → IP is None in every audit log → test fails.
        """
        svc = _make_service()
        req = _make_request(ip="SENTINEL_IP")
        captured = {}

        async def _fake_record(**kwargs):
            captured.update(kwargs)
            return "uuid-1"

        with patch("app.services.audit_service.repo.record", side_effect=_fake_record):
            await svc.log(actor="ian", action="login", request=req)

        assert captured["ip_address"] == "SENTINEL_IP"

    @pytest.mark.asyncio
    async def test_correlation_id_extracted_from_request_state(self):
        """
        SENTINEL — correlation_id must be extracted from request.state.
        Skip extraction → distributed traces broken in audit log → test fails.
        """
        svc = _make_service()
        req = _make_request(correlation_id="SENTINEL_CORR_ID")
        captured = {}

        async def _fake_record(**kwargs):
            captured.update(kwargs)
            return "uuid-1"

        with patch("app.services.audit_service.repo.record", side_effect=_fake_record):
            await svc.log(actor="ian", action="login", request=req)

        assert captured["correlation_id"] == "SENTINEL_CORR_ID"

    @pytest.mark.asyncio
    async def test_no_request_passes_none_ip_and_correlation(self):
        """Without a request, ip_address and correlation_id must be None."""
        svc = _make_service()
        captured = {}

        async def _fake_record(**kwargs):
            captured.update(kwargs)
            return "uuid-1"

        with patch("app.services.audit_service.repo.record", side_effect=_fake_record):
            await svc.log(actor="ian", action="api_key.create")

        assert captured["ip_address"] is None
        assert captured["correlation_id"] is None

    @pytest.mark.asyncio
    async def test_request_without_client_uses_none_ip(self):
        """
        SENTINEL — when request.client is None, ip_address must be None (no crash).
        Remove `if request.client` check → AttributeError on .host → test fails.
        """
        svc = _make_service()
        req = MagicMock()
        req.client = None
        req.state = MagicMock()
        req.state.correlation_id = None
        captured = {}

        async def _fake_record(**kwargs):
            captured.update(kwargs)
            return "uuid-1"

        with patch("app.services.audit_service.repo.record", side_effect=_fake_record):
            await svc.log(actor="ian", action="login", request=req)

        assert captured["ip_address"] is None

    @pytest.mark.asyncio
    async def test_all_fields_forwarded_to_repo(self):
        """
        SENTINEL — every log() parameter must be forwarded to repo.record.
        Drop detail → audit events lose metadata → test fails.
        """
        svc = _make_service()
        captured = {}

        async def _fake_record(**kwargs):
            captured.update(kwargs)
            return "uuid-1"

        with patch("app.services.audit_service.repo.record", side_effect=_fake_record):
            await svc.log(
                actor="SENTINEL_ACTOR",
                action="SENTINEL_ACTION",
                resource_type="SENTINEL_TYPE",
                resource_id="SENTINEL_ID",
                detail={"key": "SENTINEL_DETAIL"},
                outcome="failure",
            )

        assert captured["actor"] == "SENTINEL_ACTOR"
        assert captured["action"] == "SENTINEL_ACTION"
        assert captured["resource_type"] == "SENTINEL_TYPE"
        assert captured["resource_id"] == "SENTINEL_ID"
        assert captured["detail"] == {"key": "SENTINEL_DETAIL"}
        assert captured["outcome"] == "failure"

    @pytest.mark.asyncio
    async def test_default_outcome_is_success(self):
        """Default outcome must be 'success' if not provided."""
        svc = _make_service()
        captured = {}

        async def _fake_record(**kwargs):
            captured.update(kwargs)
            return "uuid-1"

        with patch("app.services.audit_service.repo.record", side_effect=_fake_record):
            await svc.log(actor="ian", action="login")

        assert captured["outcome"] == "success"


# ===========================================================================
# get_events / get_summary / prune — delegation to repo
# ===========================================================================


class TestAuditReadHelpers:
    @pytest.mark.asyncio
    async def test_get_events_delegates_to_repo(self):
        """
        SENTINEL — get_events must call repo.get_events with kwargs.
        Return [] always → callers see empty audit history → test fails.
        """
        svc = _make_service()
        sentinel_events = [{"event_id": "SENTINEL_EVENT"}]
        with patch(
            "app.services.audit_service.repo.get_events",
            new=AsyncMock(return_value=sentinel_events),
        ) as spy:
            result = await svc.get_events(actor="ian", limit=10)
        spy.assert_called_once_with(actor="ian", limit=10)
        assert result == sentinel_events

    @pytest.mark.asyncio
    async def test_get_summary_delegates_to_repo(self):
        """get_summary must forward hours to repo.get_summary."""
        svc = _make_service()
        sentinel_summary = {"total": 42}
        with patch(
            "app.services.audit_service.repo.get_summary",
            new=AsyncMock(return_value=sentinel_summary),
        ) as spy:
            result = await svc.get_summary(hours=48)
        spy.assert_called_once_with(hours=48)
        assert result == sentinel_summary

    @pytest.mark.asyncio
    async def test_prune_delegates_to_repo(self):
        """prune must call repo.prune_old_events with keep_days."""
        svc = _make_service()
        with patch(
            "app.services.audit_service.repo.prune_old_events",
            new=AsyncMock(return_value=5),
        ) as spy:
            result = await svc.prune(keep_days=30)
        spy.assert_called_once_with(keep_days=30)
        assert result == 5
