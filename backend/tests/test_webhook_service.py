"""
Tests for webhook service and repository.

Covers:
- webhook_repository: CRUD, delivery recording, stats, pruning
- WebhookRegistration: event matching, serialization
- WebhookService: register, fire, deliver, persistence integration
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_pool():
    """Create a mock asyncpg pool with async context manager."""
    conn = AsyncMock()
    pool = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = ctx
    return pool, conn


# ===========================================================================
# webhook_repository unit tests
# ===========================================================================


@pytest.mark.asyncio
class TestWebhookRepositoryCreate:

    @patch("app.repository.webhook_repository.get_db_pool")
    async def test_create_returns_id(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        mock_pool_fn.return_value = pool

        from app.repository.webhook_repository import create_registration

        result = await create_registration(
            webhook_id="wh-001",
            name="test",
            url="https://example.com/hook",
            events=["run.completed"],
        )
        assert result == "wh-001"
        conn.execute.assert_awaited_once()

    @patch("app.repository.webhook_repository.get_db_pool")
    async def test_create_passes_correct_params(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        mock_pool_fn.return_value = pool

        from app.repository.webhook_repository import create_registration

        await create_registration(
            webhook_id="wh-002",
            name="my-hook",
            url="https://example.com/cb",
            events=["run.started", "run.failed"],
            secret="s3cr3t",
            headers={"X-Custom": "val"},
            max_retries=5,
        )

        args = conn.execute.call_args[0]
        assert "INSERT INTO webhook_registrations" in args[0]
        assert args[1] == "wh-002"
        assert args[2] == "my-hook"
        assert args[3] == "https://example.com/cb"
        assert args[4] == ["run.started", "run.failed"]
        assert args[5] == "s3cr3t"
        assert json.loads(args[6]) == {"X-Custom": "val"}
        assert args[7] == 5

    @patch("app.repository.webhook_repository.get_db_pool")
    async def test_create_handles_db_error(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        conn.execute.side_effect = Exception("connection lost")
        mock_pool_fn.return_value = pool

        from app.repository.webhook_repository import create_registration

        result = await create_registration(
            webhook_id="wh-err",
            name="test",
            url="https://example.com",
            events=["run.completed"],
        )
        assert result is None


@pytest.mark.asyncio
class TestWebhookRepositoryGet:

    @patch("app.repository.webhook_repository.get_db_pool")
    async def test_get_returns_dict(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        mock_pool_fn.return_value = pool

        now = datetime.now(timezone.utc)
        conn.fetchrow.return_value = {
            "id": "wh-001",
            "name": "test",
            "url": "https://example.com",
            "events": ["run.completed"],
            "secret": None,
            "headers": "{}",
            "max_retries": 3,
            "is_active": True,
            "delivery_count": 5,
            "failure_count": 1,
            "last_delivery_at": now,
            "last_error": None,
            "created_at": now,
            "updated_at": now,
        }

        from app.repository.webhook_repository import get_registration

        result = await get_registration("wh-001")
        assert result["id"] == "wh-001"
        assert result["events"] == ["run.completed"]
        assert result["delivery_count"] == 5

    @patch("app.repository.webhook_repository.get_db_pool")
    async def test_get_returns_none_for_missing(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        conn.fetchrow.return_value = None
        mock_pool_fn.return_value = pool

        from app.repository.webhook_repository import get_registration

        result = await get_registration("nonexistent")
        assert result is None


@pytest.mark.asyncio
class TestWebhookRepositoryList:

    @patch("app.repository.webhook_repository.get_db_pool")
    async def test_list_returns_all(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        mock_pool_fn.return_value = pool

        now = datetime.now(timezone.utc)
        conn.fetch.return_value = [
            {
                "id": "wh-001", "name": "a", "url": "https://a.com",
                "events": ["run.completed"], "secret": None, "headers": "{}",
                "max_retries": 3, "is_active": True, "delivery_count": 0,
                "failure_count": 0, "last_delivery_at": None, "last_error": None,
                "created_at": now, "updated_at": now,
            },
        ]

        from app.repository.webhook_repository import list_registrations

        result = await list_registrations()
        assert len(result) == 1
        assert result[0]["id"] == "wh-001"

    @patch("app.repository.webhook_repository.get_db_pool")
    async def test_list_active_only(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        mock_pool_fn.return_value = pool
        conn.fetch.return_value = []

        from app.repository.webhook_repository import list_registrations

        await list_registrations(active_only=True)
        query = conn.fetch.call_args[0][0]
        assert "is_active" in query


@pytest.mark.asyncio
class TestWebhookRepositoryUpdate:

    @patch("app.repository.webhook_repository.get_db_pool")
    async def test_update_fields(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        conn.execute.return_value = "UPDATE 1"
        mock_pool_fn.return_value = pool

        from app.repository.webhook_repository import update_registration

        result = await update_registration("wh-001", name="new-name", is_active=False)
        assert result is True
        query = conn.execute.call_args[0][0]
        assert "name" in query
        assert "is_active" in query

    @patch("app.repository.webhook_repository.get_db_pool")
    async def test_update_ignores_unknown_fields(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        mock_pool_fn.return_value = pool

        from app.repository.webhook_repository import update_registration

        result = await update_registration("wh-001", nonexistent_field="nope")
        assert result is False
        conn.execute.assert_not_awaited()


@pytest.mark.asyncio
class TestWebhookRepositoryDelete:

    @patch("app.repository.webhook_repository.get_db_pool")
    async def test_delete_returns_true(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        conn.execute.return_value = "DELETE 1"
        mock_pool_fn.return_value = pool

        from app.repository.webhook_repository import delete_registration

        result = await delete_registration("wh-001")
        assert result is True

    @patch("app.repository.webhook_repository.get_db_pool")
    async def test_delete_returns_false_when_missing(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        conn.execute.return_value = "DELETE 0"
        mock_pool_fn.return_value = pool

        from app.repository.webhook_repository import delete_registration

        result = await delete_registration("nonexistent")
        assert result is False


@pytest.mark.asyncio
class TestWebhookRepositoryCounters:

    @patch("app.repository.webhook_repository.get_db_pool")
    async def test_increment_delivery_count(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        mock_pool_fn.return_value = pool

        from app.repository.webhook_repository import increment_delivery_count

        await increment_delivery_count("wh-001")
        query = conn.execute.call_args[0][0]
        assert "delivery_count = delivery_count + 1" in query

    @patch("app.repository.webhook_repository.get_db_pool")
    async def test_increment_failure_count(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        mock_pool_fn.return_value = pool

        from app.repository.webhook_repository import increment_failure_count

        await increment_failure_count("wh-001", "HTTP 500")
        query = conn.execute.call_args[0][0]
        assert "failure_count = failure_count + 1" in query


@pytest.mark.asyncio
class TestWebhookRepositoryDeliveries:

    @patch("app.repository.webhook_repository.get_db_pool")
    async def test_record_delivery(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        mock_pool_fn.return_value = pool

        from app.repository.webhook_repository import record_delivery

        result = await record_delivery(
            delivery_id="del-001",
            webhook_id="wh-001",
            event="run.completed",
            payload={"data": "test"},
            attempts=1,
            status_code=200,
            delivered_at=datetime.now(timezone.utc),
        )
        assert result == "del-001"

    @patch("app.repository.webhook_repository.get_db_pool")
    async def test_get_deliveries_with_filters(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        conn.fetch.return_value = []
        mock_pool_fn.return_value = pool

        from app.repository.webhook_repository import get_deliveries

        await get_deliveries(webhook_id="wh-001", event_filter="run.completed")
        query = conn.fetch.call_args[0][0]
        assert "webhook_id" in query
        assert "event" in query

    @patch("app.repository.webhook_repository.get_db_pool")
    async def test_prune_old_deliveries(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        conn.execute.return_value = "DELETE 42"
        mock_pool_fn.return_value = pool

        from app.repository.webhook_repository import prune_old_deliveries

        result = await prune_old_deliveries(keep_days=7)
        assert result == 42

    @patch("app.repository.webhook_repository.get_db_pool")
    async def test_delivery_stats(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        mock_pool_fn.return_value = pool

        conn.fetchval.side_effect = [100, 95]
        conn.fetch.return_value = [
            {"event": "run.completed", "cnt": 80},
            {"event": "run.failed", "cnt": 20},
        ]

        from app.repository.webhook_repository import get_delivery_stats

        stats = await get_delivery_stats(hours=12)
        assert stats["total_deliveries"] == 100
        assert stats["successful"] == 95
        assert stats["by_event"]["run.completed"] == 80


# ===========================================================================
# WebhookRegistration unit tests
# ===========================================================================


class TestWebhookRegistrationMatching:

    def test_matches_exact_event(self):
        from app.services.webhook_service import WebhookRegistration, WebhookEvent

        reg = WebhookRegistration(
            url="https://example.com",
            events=[WebhookEvent.RUN_COMPLETED],
        )
        assert reg.matches_event(WebhookEvent.RUN_COMPLETED) is True
        assert reg.matches_event(WebhookEvent.RUN_FAILED) is False

    def test_matches_wildcard(self):
        from app.services.webhook_service import WebhookRegistration, WebhookEvent

        reg = WebhookRegistration(
            url="https://example.com",
            events=[WebhookEvent.WILDCARD],
        )
        assert reg.matches_event(WebhookEvent.RUN_COMPLETED) is True
        assert reg.matches_event(WebhookEvent.RUN_FAILED) is True
        assert reg.matches_event(WebhookEvent.NODE_STARTED) is True

    def test_matches_multiple_events(self):
        from app.services.webhook_service import WebhookRegistration, WebhookEvent

        reg = WebhookRegistration(
            url="https://example.com",
            events=[WebhookEvent.RUN_STARTED, WebhookEvent.RUN_FAILED],
        )
        assert reg.matches_event(WebhookEvent.RUN_STARTED) is True
        assert reg.matches_event(WebhookEvent.RUN_FAILED) is True
        assert reg.matches_event(WebhookEvent.RUN_COMPLETED) is False

    def test_inactive_matches_nothing(self):
        from app.services.webhook_service import WebhookRegistration, WebhookEvent

        reg = WebhookRegistration(
            url="https://example.com",
            events=[WebhookEvent.WILDCARD],
        )
        reg.is_active = False
        assert reg.matches_event(WebhookEvent.RUN_COMPLETED) is False

    def test_to_dict_roundtrip(self):
        from app.services.webhook_service import WebhookRegistration, WebhookEvent

        reg = WebhookRegistration(
            url="https://example.com",
            events=[WebhookEvent.RUN_COMPLETED, WebhookEvent.RUN_FAILED],
            name="test-hook",
            secret="mysecret",
        )
        d = reg.to_dict()
        assert d["url"] == "https://example.com"
        assert "run.completed" in d["events"]
        assert d["name"] == "test-hook"

    def test_from_db_dict(self):
        from app.services.webhook_service import WebhookRegistration, WebhookEvent

        data = {
            "id": "wh-db1",
            "name": "from-db",
            "url": "https://db.example.com",
            "events": ["run.completed", "*"],
            "secret": "abc",
            "headers": {"X-Key": "val"},
            "max_retries": 5,
            "is_active": True,
            "delivery_count": 10,
            "failure_count": 2,
            "last_delivery_at": "2026-03-24T12:00:00+00:00",
            "last_error": None,
            "created_at": "2026-03-24T10:00:00+00:00",
        }
        reg = WebhookRegistration.from_db_dict(data)
        assert reg.id == "wh-db1"
        assert WebhookEvent.WILDCARD in reg.events
        assert reg.delivery_count == 10
        assert reg.max_retries == 5

    def test_from_db_dict_handles_unknown_events(self):
        from app.services.webhook_service import WebhookRegistration

        data = {
            "id": "wh-unk",
            "name": "test",
            "url": "https://example.com",
            "events": ["run.completed", "unknown.event"],
            "is_active": True,
        }
        reg = WebhookRegistration.from_db_dict(data)
        assert len(reg.events) == 1  # only the valid one


# ===========================================================================
# WebhookService unit tests
# ===========================================================================


@pytest.mark.asyncio
class TestWebhookServiceRegister:

    @patch("app.services.webhook_service.webhook_repository")
    async def test_register_adds_to_memory_and_db(self, mock_repo):
        from app.services.webhook_service import WebhookService

        mock_repo.create_registration = AsyncMock(return_value="wh-new")
        mock_repo.list_registrations = AsyncMock(return_value=[])

        svc = WebhookService()
        svc._http_client = AsyncMock()

        reg = await svc.register(
            url="https://example.com/hook",
            events=["run.completed"],
            secret="s3cr3t",
            name="my-hook",
        )

        assert reg.id in svc.webhooks
        assert reg.name == "my-hook"
        assert reg.secret == "s3cr3t"
        mock_repo.create_registration.assert_awaited_once()

    @patch("app.services.webhook_service.webhook_repository")
    async def test_register_multiple(self, mock_repo):
        from app.services.webhook_service import WebhookService

        mock_repo.create_registration = AsyncMock(return_value="ok")
        mock_repo.list_registrations = AsyncMock(return_value=[])

        svc = WebhookService()
        svc._http_client = AsyncMock()

        await svc.register(url="https://a.com", events=["run.started"])
        await svc.register(url="https://b.com", events=["run.failed"])

        assert len(svc.webhooks) == 2


@pytest.mark.asyncio
class TestWebhookServiceUnregister:

    @patch("app.services.webhook_service.webhook_repository")
    async def test_unregister_removes_from_memory_and_db(self, mock_repo):
        from app.services.webhook_service import WebhookService, WebhookRegistration, WebhookEvent

        mock_repo.delete_registration = AsyncMock(return_value=True)

        svc = WebhookService()
        reg = WebhookRegistration(
            url="https://example.com",
            events=[WebhookEvent.RUN_COMPLETED],
        )
        svc.webhooks[reg.id] = reg

        result = await svc.unregister(reg.id)
        assert result is True
        assert reg.id not in svc.webhooks
        mock_repo.delete_registration.assert_awaited_once_with(reg.id)

    @patch("app.services.webhook_service.webhook_repository")
    async def test_unregister_nonexistent(self, mock_repo):
        from app.services.webhook_service import WebhookService

        svc = WebhookService()
        result = await svc.unregister("nonexistent")
        assert result is False


@pytest.mark.asyncio
class TestWebhookServiceUpdate:

    @patch("app.services.webhook_service.webhook_repository")
    async def test_update_changes_fields(self, mock_repo):
        from app.services.webhook_service import WebhookService, WebhookRegistration, WebhookEvent

        mock_repo.update_registration = AsyncMock(return_value=True)

        svc = WebhookService()
        reg = WebhookRegistration(
            url="https://old.com",
            events=[WebhookEvent.RUN_COMPLETED],
            name="old-name",
        )
        svc.webhooks[reg.id] = reg

        updated = await svc.update_webhook(reg.id, name="new-name", url="https://new.com")
        assert updated.name == "new-name"
        assert updated.url == "https://new.com"
        mock_repo.update_registration.assert_awaited_once()

    @patch("app.services.webhook_service.webhook_repository")
    async def test_update_nonexistent(self, mock_repo):
        from app.services.webhook_service import WebhookService

        svc = WebhookService()
        result = await svc.update_webhook("nonexistent", name="nope")
        assert result is None


@pytest.mark.asyncio
class TestWebhookServiceFire:

    @patch("app.services.webhook_service.webhook_repository")
    async def test_fire_queues_matching_webhooks(self, mock_repo):
        from app.services.webhook_service import WebhookService, WebhookRegistration, WebhookEvent

        svc = WebhookService()
        svc._delivery_queue = AsyncMock()
        svc._delivery_queue.put = AsyncMock()

        matching = WebhookRegistration(
            url="https://a.com",
            events=[WebhookEvent.RUN_COMPLETED],
        )
        non_matching = WebhookRegistration(
            url="https://b.com",
            events=[WebhookEvent.RUN_FAILED],
        )
        svc.webhooks[matching.id] = matching
        svc.webhooks[non_matching.id] = non_matching

        await svc.fire(WebhookEvent.RUN_COMPLETED, {"data": "test"})

        # Only the matching webhook should be queued
        assert svc._delivery_queue.put.await_count == 1

    @patch("app.services.webhook_service.webhook_repository")
    async def test_fire_wildcard_matches_all(self, mock_repo):
        from app.services.webhook_service import WebhookService, WebhookRegistration, WebhookEvent

        svc = WebhookService()
        svc._delivery_queue = AsyncMock()
        svc._delivery_queue.put = AsyncMock()

        wildcard = WebhookRegistration(
            url="https://wildcard.com",
            events=[WebhookEvent.WILDCARD],
        )
        svc.webhooks[wildcard.id] = wildcard

        await svc.fire(WebhookEvent.NODE_STARTED, {"node": "test"})
        assert svc._delivery_queue.put.await_count == 1

    @patch("app.services.webhook_service.webhook_repository")
    async def test_fire_skips_inactive(self, mock_repo):
        from app.services.webhook_service import WebhookService, WebhookRegistration, WebhookEvent

        svc = WebhookService()
        svc._delivery_queue = AsyncMock()
        svc._delivery_queue.put = AsyncMock()

        inactive = WebhookRegistration(
            url="https://inactive.com",
            events=[WebhookEvent.WILDCARD],
        )
        inactive.is_active = False
        svc.webhooks[inactive.id] = inactive

        await svc.fire(WebhookEvent.RUN_COMPLETED, {})
        assert svc._delivery_queue.put.await_count == 0

    @patch("app.services.webhook_service.webhook_repository")
    async def test_fire_payload_includes_metadata(self, mock_repo):
        from app.services.webhook_service import WebhookService, WebhookRegistration, WebhookEvent

        svc = WebhookService()
        svc._delivery_queue = AsyncMock()
        svc._delivery_queue.put = AsyncMock()

        hook = WebhookRegistration(
            url="https://example.com",
            events=[WebhookEvent.RUN_COMPLETED],
        )
        svc.webhooks[hook.id] = hook

        await svc.fire(WebhookEvent.RUN_COMPLETED, {"key": "val"}, run_id="run-123")

        call_args = svc._delivery_queue.put.call_args[0][0]
        _, delivery = call_args
        assert delivery.payload["event"] == "run.completed"
        assert delivery.payload["run_id"] == "run-123"
        assert "timestamp" in delivery.payload
        assert delivery.payload["key"] == "val"


@pytest.mark.asyncio
class TestWebhookServiceDeliver:

    @patch("app.services.webhook_service.webhook_repository")
    async def test_successful_delivery(self, mock_repo):
        from app.services.webhook_service import (
            WebhookService, WebhookRegistration, WebhookDelivery, WebhookEvent,
        )

        mock_repo.increment_delivery_count = AsyncMock()
        mock_repo.record_delivery = AsyncMock()

        svc = WebhookService()

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = "OK"
        mock_response.is_success = True

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        svc._http_client = mock_client

        webhook = WebhookRegistration(
            url="https://example.com",
            events=[WebhookEvent.RUN_COMPLETED],
            secret="test-secret",
        )
        delivery = WebhookDelivery(
            webhook_id=webhook.id,
            event=WebhookEvent.RUN_COMPLETED,
            payload={"test": True},
        )

        await svc._deliver(webhook, delivery)

        assert delivery.delivered_at is not None
        assert delivery.status_code == 200
        assert webhook.delivery_count == 1
        mock_repo.increment_delivery_count.assert_awaited_once()
        mock_repo.record_delivery.assert_awaited_once()

    @patch("app.services.webhook_service.webhook_repository")
    async def test_failed_delivery_retries(self, mock_repo):
        from app.services.webhook_service import (
            WebhookService, WebhookRegistration, WebhookDelivery, WebhookEvent,
        )

        mock_repo.increment_failure_count = AsyncMock()
        mock_repo.record_delivery = AsyncMock()

        svc = WebhookService()

        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.is_success = False

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        svc._http_client = mock_client

        webhook = WebhookRegistration(
            url="https://example.com",
            events=[WebhookEvent.RUN_COMPLETED],
            max_retries=2,
        )
        delivery = WebhookDelivery(
            webhook_id=webhook.id,
            event=WebhookEvent.RUN_COMPLETED,
            payload={"test": True},
        )

        # Patch sleep to avoid waiting
        with patch("app.services.webhook_service.asyncio.sleep", new_callable=AsyncMock):
            await svc._deliver(webhook, delivery)

        assert delivery.delivered_at is None
        assert delivery.attempts == 2
        assert webhook.failure_count == 1
        mock_repo.increment_failure_count.assert_awaited_once()

    @patch("app.services.webhook_service.webhook_repository")
    async def test_delivery_includes_hmac_signature(self, mock_repo):
        import hashlib
        import hmac as hmac_mod
        from app.services.webhook_service import (
            WebhookService, WebhookRegistration, WebhookDelivery, WebhookEvent,
        )

        mock_repo.increment_delivery_count = AsyncMock()
        mock_repo.record_delivery = AsyncMock()

        svc = WebhookService()

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = "OK"
        mock_response.is_success = True

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        svc._http_client = mock_client

        secret = "my-secret-key"
        webhook = WebhookRegistration(
            url="https://example.com",
            events=[WebhookEvent.RUN_COMPLETED],
            secret=secret,
        )
        payload = {"test": True}
        delivery = WebhookDelivery(
            webhook_id=webhook.id,
            event=WebhookEvent.RUN_COMPLETED,
            payload=payload,
        )

        await svc._deliver(webhook, delivery)

        # Verify the HMAC header was sent
        call_kwargs = mock_client.post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        assert "X-Webhook-Signature" in headers

        expected_sig = hmac_mod.new(
            secret.encode(),
            json.dumps(payload).encode(),
            hashlib.sha256,
        ).hexdigest()
        assert headers["X-Webhook-Signature"] == f"sha256={expected_sig}"

    @patch("app.services.webhook_service.webhook_repository")
    async def test_delivery_exception_retries(self, mock_repo):
        from app.services.webhook_service import (
            WebhookService, WebhookRegistration, WebhookDelivery, WebhookEvent,
        )

        mock_repo.increment_failure_count = AsyncMock()
        mock_repo.record_delivery = AsyncMock()

        svc = WebhookService()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=ConnectionError("refused"))
        svc._http_client = mock_client

        webhook = WebhookRegistration(
            url="https://unreachable.com",
            events=[WebhookEvent.RUN_COMPLETED],
            max_retries=2,
        )
        delivery = WebhookDelivery(
            webhook_id=webhook.id,
            event=WebhookEvent.RUN_COMPLETED,
            payload={},
        )

        with patch("app.services.webhook_service.asyncio.sleep", new_callable=AsyncMock):
            await svc._deliver(webhook, delivery)

        assert delivery.error == "refused"
        assert webhook.failure_count == 1


@pytest.mark.asyncio
class TestWebhookServiceStart:

    @patch("app.services.webhook_service.webhook_repository")
    async def test_start_loads_from_db(self, mock_repo):
        from app.services.webhook_service import WebhookService

        mock_repo.list_registrations = AsyncMock(return_value=[
            {
                "id": "wh-loaded",
                "name": "from-db",
                "url": "https://loaded.com",
                "events": ["run.completed"],
                "secret": None,
                "headers": {},
                "max_retries": 3,
                "is_active": True,
                "delivery_count": 42,
                "failure_count": 3,
                "last_delivery_at": None,
                "last_error": None,
                "created_at": "2026-03-24T10:00:00+00:00",
            },
        ])

        svc = WebhookService()

        # Patch to prevent actual background task
        with patch("asyncio.create_task"):
            await svc.start()

        assert "wh-loaded" in svc.webhooks
        assert svc.webhooks["wh-loaded"].delivery_count == 42

    @patch("app.services.webhook_service.webhook_repository")
    async def test_start_handles_db_failure(self, mock_repo):
        from app.services.webhook_service import WebhookService

        mock_repo.list_registrations = AsyncMock(side_effect=Exception("db down"))

        svc = WebhookService()

        with patch("asyncio.create_task"):
            await svc.start()  # should not raise

        assert len(svc.webhooks) == 0


@pytest.mark.asyncio
class TestWebhookServiceStats:

    @patch("app.services.webhook_service.webhook_repository")
    async def test_get_stats(self, mock_repo):
        from app.services.webhook_service import WebhookService, WebhookRegistration, WebhookEvent

        mock_repo.get_delivery_stats = AsyncMock(return_value={
            "window_hours": 24,
            "total_deliveries": 100,
            "successful": 95,
            "failed": 5,
            "success_rate": 0.95,
            "by_event": {},
        })

        svc = WebhookService()
        active = WebhookRegistration(url="https://a.com", events=[WebhookEvent.RUN_COMPLETED])
        inactive = WebhookRegistration(url="https://b.com", events=[WebhookEvent.RUN_FAILED])
        inactive.is_active = False
        svc.webhooks[active.id] = active
        svc.webhooks[inactive.id] = inactive

        stats = await svc.get_stats()
        assert stats["registered_webhooks"] == 2
        assert stats["active_webhooks"] == 1
        assert stats["last_24h"]["total_deliveries"] == 100


@pytest.mark.asyncio
class TestWebhookServicePrune:

    @patch("app.services.webhook_service.webhook_repository")
    async def test_prune_delegates_to_repo(self, mock_repo):
        from app.services.webhook_service import WebhookService

        mock_repo.prune_old_deliveries = AsyncMock(return_value=15)

        svc = WebhookService()
        result = await svc.prune_deliveries(keep_days=7)
        assert result == 15
        mock_repo.prune_old_deliveries.assert_awaited_once_with(7)


# ===========================================================================
# WebhookEvent enum tests
# ===========================================================================


class TestWebhookEvent:

    def test_all_events_are_strings(self):
        from app.services.webhook_service import WebhookEvent

        for event in WebhookEvent:
            assert isinstance(event.value, str)

    def test_wildcard_exists(self):
        from app.services.webhook_service import WebhookEvent

        assert WebhookEvent.WILDCARD.value == "*"

    def test_event_from_string(self):
        from app.services.webhook_service import WebhookEvent

        assert WebhookEvent("run.completed") == WebhookEvent.RUN_COMPLETED
        assert WebhookEvent("*") == WebhookEvent.WILDCARD


# ===========================================================================
# Global helper tests
# ===========================================================================


class TestGlobalHelpers:

    def test_get_webhook_service_singleton(self):
        from app.services import webhook_service

        # Reset
        webhook_service._webhook_service = None
        svc1 = webhook_service.get_webhook_service()
        svc2 = webhook_service.get_webhook_service()
        assert svc1 is svc2
        webhook_service._webhook_service = None  # cleanup