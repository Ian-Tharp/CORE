"""
Audit Repository & Service — Unit Tests

Covers:
- Table creation
- Event recording and retrieval
- Filtering (actor, action, resource, outcome, time range)
- Summary aggregation
- Pruning
- AuditService facade (including request extraction)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_row(**overrides):
    """Build a dict-like asyncpg Record stub."""
    defaults = {
        "id": uuid.uuid4(),
        "timestamp": datetime.now(timezone.utc),
        "actor": "test-actor",
        "action": "test.action",
        "resource_type": "test_resource",
        "resource_id": "res-123",
        "detail": json.dumps({"key": "value"}),
        "ip_address": "127.0.0.1",
        "correlation_id": "corr-abc",
        "outcome": "success",
    }
    defaults.update(overrides)

    class Row(dict):
        def __getitem__(self, key):
            return dict.__getitem__(self, key)

    return Row(defaults)


def _mock_pool():
    """Return (pool, conn) mocks wired for ``async with pool.acquire()``."""
    conn = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire.return_value = ctx

    # transaction context manager
    tx = AsyncMock()
    tx.__aenter__ = AsyncMock(return_value=tx)
    tx.__aexit__ = AsyncMock(return_value=False)
    conn.transaction.return_value = tx

    return pool, conn


# ===========================================================================
# REPOSITORY TESTS
# ===========================================================================


class TestEnsureAuditTables:
    @pytest.mark.asyncio
    async def test_creates_table_and_indexes(self):
        pool, conn = _mock_pool()

        # The ensure function uses ``async with conn.transaction()``
        # which needs a proper async-context-manager mock.
        tx_cm = AsyncMock()
        tx_cm.__aenter__ = AsyncMock(return_value=tx_cm)
        tx_cm.__aexit__ = AsyncMock(return_value=False)
        conn.transaction = MagicMock(return_value=tx_cm)

        with patch("app.repository.audit_repository.get_db_pool", return_value=pool):
            from app.repository import audit_repository

            await audit_repository.ensure_audit_tables()

        # Should run CREATE TABLE + 5 indexes = at least 6 execute calls
        assert conn.execute.call_count >= 6
        sql_calls = [str(c) for c in conn.execute.call_args_list]
        combined = " ".join(sql_calls)
        assert "audit_log" in combined
        assert "idx_audit_log_timestamp" in combined


class TestRecordEvent:
    @pytest.mark.asyncio
    async def test_record_returns_uuid(self):
        pool, conn = _mock_pool()
        with patch("app.repository.audit_repository.get_db_pool", return_value=pool):
            from app.repository import audit_repository

            event_id = await audit_repository.record(
                actor="admin",
                action="api_key.create",
                resource_type="api_key",
                resource_id="my-key",
                detail={"permissions": ["*"]},
                ip_address="10.0.0.1",
                correlation_id="req-xyz",
                outcome="success",
            )

        assert event_id is not None
        uuid.UUID(event_id)  # validates format
        conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_handles_db_error(self):
        pool, conn = _mock_pool()
        conn.execute.side_effect = Exception("connection lost")
        with patch("app.repository.audit_repository.get_db_pool", return_value=pool):
            from app.repository import audit_repository

            result = await audit_repository.record(actor="admin", action="test")
        assert result is None

    @pytest.mark.asyncio
    async def test_record_defaults_detail_to_empty(self):
        pool, conn = _mock_pool()
        with patch("app.repository.audit_repository.get_db_pool", return_value=pool):
            from app.repository import audit_repository

            await audit_repository.record(actor="x", action="y")

        call_args = conn.execute.call_args
        # The 7th positional arg (index 5) is the detail JSON
        assert call_args[0][6] == json.dumps({})


class TestGetEvents:
    @pytest.mark.asyncio
    async def test_returns_empty_on_no_results(self):
        pool, conn = _mock_pool()
        conn.fetch.return_value = []
        with patch("app.repository.audit_repository.get_db_pool", return_value=pool):
            from app.repository import audit_repository

            events = await audit_repository.get_events()
        assert events == []

    @pytest.mark.asyncio
    async def test_returns_formatted_events(self):
        pool, conn = _mock_pool()
        conn.fetch.return_value = [_fake_row()]
        with patch("app.repository.audit_repository.get_db_pool", return_value=pool):
            from app.repository import audit_repository

            events = await audit_repository.get_events()

        assert len(events) == 1
        evt = events[0]
        assert evt["actor"] == "test-actor"
        assert evt["action"] == "test.action"
        assert isinstance(evt["detail"], dict)

    @pytest.mark.asyncio
    async def test_filters_by_actor(self):
        pool, conn = _mock_pool()
        conn.fetch.return_value = []
        with patch("app.repository.audit_repository.get_db_pool", return_value=pool):
            from app.repository import audit_repository

            await audit_repository.get_events(actor="vigil")

        sql = conn.fetch.call_args[0][0]
        assert "actor = $1" in sql

    @pytest.mark.asyncio
    async def test_filters_by_action(self):
        pool, conn = _mock_pool()
        conn.fetch.return_value = []
        with patch("app.repository.audit_repository.get_db_pool", return_value=pool):
            from app.repository import audit_repository

            await audit_repository.get_events(action="api_key.create")

        sql = conn.fetch.call_args[0][0]
        assert "action = $1" in sql

    @pytest.mark.asyncio
    async def test_filters_by_resource(self):
        pool, conn = _mock_pool()
        conn.fetch.return_value = []
        with patch("app.repository.audit_repository.get_db_pool", return_value=pool):
            from app.repository import audit_repository

            await audit_repository.get_events(
                resource_type="webhook", resource_id="wh-1"
            )

        sql = conn.fetch.call_args[0][0]
        assert "resource_type = $1" in sql
        assert "resource_id = $2" in sql

    @pytest.mark.asyncio
    async def test_filters_by_outcome(self):
        pool, conn = _mock_pool()
        conn.fetch.return_value = []
        with patch("app.repository.audit_repository.get_db_pool", return_value=pool):
            from app.repository import audit_repository

            await audit_repository.get_events(outcome="denied")

        sql = conn.fetch.call_args[0][0]
        assert "outcome = $1" in sql

    @pytest.mark.asyncio
    async def test_filters_by_time_range(self):
        pool, conn = _mock_pool()
        conn.fetch.return_value = []
        since = datetime.now(timezone.utc) - timedelta(hours=1)
        until = datetime.now(timezone.utc)
        with patch("app.repository.audit_repository.get_db_pool", return_value=pool):
            from app.repository import audit_repository

            await audit_repository.get_events(since=since, until=until)

        sql = conn.fetch.call_args[0][0]
        assert "timestamp >=" in sql
        assert "timestamp <=" in sql

    @pytest.mark.asyncio
    async def test_combined_filters(self):
        pool, conn = _mock_pool()
        conn.fetch.return_value = []
        with patch("app.repository.audit_repository.get_db_pool", return_value=pool):
            from app.repository import audit_repository

            await audit_repository.get_events(
                actor="admin", action="webhook.create", outcome="success"
            )

        sql = conn.fetch.call_args[0][0]
        assert "actor = $1" in sql
        assert "action = $2" in sql
        assert "outcome = $3" in sql

    @pytest.mark.asyncio
    async def test_handles_db_error(self):
        pool, conn = _mock_pool()
        conn.fetch.side_effect = Exception("timeout")
        with patch("app.repository.audit_repository.get_db_pool", return_value=pool):
            from app.repository import audit_repository

            events = await audit_repository.get_events()
        assert events == []


class TestCountEvents:
    @pytest.mark.asyncio
    async def test_returns_count(self):
        pool, conn = _mock_pool()
        conn.fetchval.return_value = 42
        with patch("app.repository.audit_repository.get_db_pool", return_value=pool):
            from app.repository import audit_repository

            count = await audit_repository.count_events()
        assert count == 42

    @pytest.mark.asyncio
    async def test_returns_zero_on_error(self):
        pool, conn = _mock_pool()
        conn.fetchval.side_effect = Exception("boom")
        with patch("app.repository.audit_repository.get_db_pool", return_value=pool):
            from app.repository import audit_repository

            count = await audit_repository.count_events()
        assert count == 0

    @pytest.mark.asyncio
    async def test_filters_by_actor_and_since(self):
        pool, conn = _mock_pool()
        conn.fetchval.return_value = 5
        since = datetime.now(timezone.utc) - timedelta(hours=2)
        with patch("app.repository.audit_repository.get_db_pool", return_value=pool):
            from app.repository import audit_repository

            await audit_repository.count_events(actor="vigil", since=since)

        sql = conn.fetchval.call_args[0][0]
        assert "actor = $1" in sql
        assert "timestamp >= $2" in sql


class TestGetSummary:
    @pytest.mark.asyncio
    async def test_returns_summary_structure(self):
        pool, conn = _mock_pool()
        conn.fetchval.return_value = 10
        conn.fetch.side_effect = [
            [
                {"action": "api_key.create", "cnt": 7},
                {"action": "webhook.delete", "cnt": 3},
            ],
            [{"actor": "admin", "cnt": 10}],
            [{"outcome": "success", "cnt": 9}, {"outcome": "failure", "cnt": 1}],
        ]
        with patch("app.repository.audit_repository.get_db_pool", return_value=pool):
            from app.repository import audit_repository

            summary = await audit_repository.get_summary(hours=24)

        assert summary["period_hours"] == 24
        assert summary["total_events"] == 10
        assert summary["by_action"]["api_key.create"] == 7
        assert summary["by_actor"]["admin"] == 10
        assert summary["by_outcome"]["success"] == 9

    @pytest.mark.asyncio
    async def test_handles_db_error(self):
        pool, conn = _mock_pool()
        conn.fetchval.side_effect = Exception("nope")
        with patch("app.repository.audit_repository.get_db_pool", return_value=pool):
            from app.repository import audit_repository

            summary = await audit_repository.get_summary()
        assert summary["total_events"] == 0


class TestPruneOldEvents:
    @pytest.mark.asyncio
    async def test_returns_deleted_count(self):
        pool, conn = _mock_pool()
        conn.execute.return_value = "DELETE 15"
        with patch("app.repository.audit_repository.get_db_pool", return_value=pool):
            from app.repository import audit_repository

            deleted = await audit_repository.prune_old_events(keep_days=30)
        assert deleted == 15

    @pytest.mark.asyncio
    async def test_handles_db_error(self):
        pool, conn = _mock_pool()
        conn.execute.side_effect = Exception("disk full")
        with patch("app.repository.audit_repository.get_db_pool", return_value=pool):
            from app.repository import audit_repository

            deleted = await audit_repository.prune_old_events()
        assert deleted == 0


# ===========================================================================
# SERVICE TESTS
# ===========================================================================


class TestAuditService:
    @pytest.mark.asyncio
    async def test_log_extracts_request_context(self):
        with patch("app.services.audit_service.repo") as mock_repo:
            mock_repo.record = AsyncMock(return_value="evt-id")

            # Build a fake Request
            request = MagicMock()
            request.client.host = "192.168.1.1"
            request.state.correlation_id = "corr-999"

            from app.services.audit_service import AuditService

            svc = AuditService()
            result = await svc.log(
                actor="admin",
                action="api_key.create",
                resource_type="api_key",
                resource_id="k1",
                request=request,
            )

            assert result == "evt-id"
            mock_repo.record.assert_called_once()
            call_kwargs = mock_repo.record.call_args.kwargs
            assert call_kwargs["ip_address"] == "192.168.1.1"
            assert call_kwargs["correlation_id"] == "corr-999"

    @pytest.mark.asyncio
    async def test_log_without_request(self):
        with patch("app.services.audit_service.repo") as mock_repo:
            mock_repo.record = AsyncMock(return_value="evt-id")

            from app.services.audit_service import AuditService

            svc = AuditService()
            result = await svc.log(actor="system", action="startup")

            call_kwargs = mock_repo.record.call_args.kwargs
            assert call_kwargs["ip_address"] is None
            assert call_kwargs["correlation_id"] is None

    @pytest.mark.asyncio
    async def test_log_never_raises(self):
        with patch("app.services.audit_service.repo") as mock_repo:
            mock_repo.record = AsyncMock(side_effect=Exception("db down"))

            from app.services.audit_service import AuditService

            svc = AuditService()
            result = await svc.log(actor="x", action="y")
            assert result is None  # graceful fallback, no exception

    @pytest.mark.asyncio
    async def test_log_passes_outcome(self):
        with patch("app.services.audit_service.repo") as mock_repo:
            mock_repo.record = AsyncMock(return_value="evt-id")

            from app.services.audit_service import AuditService

            svc = AuditService()
            await svc.log(actor="x", action="y", outcome="denied")

            call_kwargs = mock_repo.record.call_args.kwargs
            assert call_kwargs["outcome"] == "denied"

    @pytest.mark.asyncio
    async def test_get_events_delegates(self):
        with patch("app.services.audit_service.repo") as mock_repo:
            mock_repo.get_events = AsyncMock(return_value=[{"id": "1"}])

            from app.services.audit_service import AuditService

            svc = AuditService()
            events = await svc.get_events(actor="admin")

            mock_repo.get_events.assert_called_once_with(actor="admin")
            assert len(events) == 1

    @pytest.mark.asyncio
    async def test_get_summary_delegates(self):
        with patch("app.services.audit_service.repo") as mock_repo:
            mock_repo.get_summary = AsyncMock(return_value={"total_events": 5})

            from app.services.audit_service import AuditService

            svc = AuditService()
            summary = await svc.get_summary(hours=48)

            mock_repo.get_summary.assert_called_once_with(hours=48)
            assert summary["total_events"] == 5

    @pytest.mark.asyncio
    async def test_prune_delegates(self):
        with patch("app.services.audit_service.repo") as mock_repo:
            mock_repo.prune_old_events = AsyncMock(return_value=10)

            from app.services.audit_service import AuditService

            svc = AuditService()
            deleted = await svc.prune(keep_days=60)

            mock_repo.prune_old_events.assert_called_once_with(keep_days=60)
            assert deleted == 10

    @pytest.mark.asyncio
    async def test_log_with_detail(self):
        with patch("app.services.audit_service.repo") as mock_repo:
            mock_repo.record = AsyncMock(return_value="evt-id")

            from app.services.audit_service import AuditService

            svc = AuditService()
            await svc.log(
                actor="admin",
                action="api_key.create",
                detail={"permissions": ["*"], "description": "test"},
            )

            call_kwargs = mock_repo.record.call_args.kwargs
            assert call_kwargs["detail"]["permissions"] == ["*"]
