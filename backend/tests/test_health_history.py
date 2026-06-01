"""
Tests for health history tracking.

Covers:
- health_repository: record, query, summary, prune
- /health/history endpoints
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# health_repository unit tests
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


@pytest.mark.asyncio
class TestRecordSnapshot:

    @patch("app.repository.health_repository.get_db_pool")
    async def test_record_returns_uuid(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        mock_pool_fn.return_value = pool

        from app.repository.health_repository import record_snapshot

        result = await record_snapshot(
            overall_status="healthy",
            services={"database": {"status": "healthy"}},
            total_latency_ms=42.5,
            summary={"total_services": 1, "healthy": 1},
        )

        assert result is not None
        # Should be a valid UUID string
        uuid.UUID(result)
        conn.execute.assert_awaited_once()

    @patch("app.repository.health_repository.get_db_pool")
    async def test_record_stores_correct_data(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        mock_pool_fn.return_value = pool

        from app.repository.health_repository import record_snapshot

        services = {"db": {"status": "healthy", "latency_ms": 5.0}}
        summary = {"total_services": 1, "healthy": 1, "unhealthy": 0}

        await record_snapshot("degraded", services, 100.0, summary)

        args = conn.execute.call_args[0]
        assert "INSERT INTO health_snapshots" in args[0]
        assert args[2] == "degraded"  # overall_status
        assert json.loads(args[3])["db"]["status"] == "healthy"

    @patch("app.repository.health_repository.get_db_pool")
    async def test_record_handles_db_error_gracefully(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        conn.execute.side_effect = Exception("connection lost")
        mock_pool_fn.return_value = pool

        from app.repository.health_repository import record_snapshot

        result = await record_snapshot("healthy", {}, 0.0, {})
        assert result is None  # Should not raise


@pytest.mark.asyncio
class TestGetHistory:

    @patch("app.repository.health_repository.get_db_pool")
    async def test_returns_list(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        now = datetime.now(timezone.utc)
        conn.fetch.return_value = [
            {
                "id": "abc-123",
                "overall_status": "healthy",
                "services": json.dumps({"db": {"status": "healthy"}}),
                "total_latency_ms": 10.0,
                "summary": json.dumps({"healthy": 1}),
                "created_at": now,
            }
        ]
        mock_pool_fn.return_value = pool

        from app.repository.health_repository import get_history

        rows = await get_history(limit=10)
        assert len(rows) == 1
        assert rows[0]["overall_status"] == "healthy"
        assert rows[0]["id"] == "abc-123"

    @patch("app.repository.health_repository.get_db_pool")
    async def test_limit_capped_at_500(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        conn.fetch.return_value = []
        mock_pool_fn.return_value = pool

        from app.repository.health_repository import get_history

        await get_history(limit=9999)
        # The LIMIT param (second-to-last) should be 500
        args = conn.fetch.call_args[0]
        # params are positional: last two are limit and offset
        assert args[-2] == 500

    @patch("app.repository.health_repository.get_db_pool")
    async def test_status_filter_applied(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        conn.fetch.return_value = []
        mock_pool_fn.return_value = pool

        from app.repository.health_repository import get_history

        await get_history(status_filter="unhealthy")
        query = conn.fetch.call_args[0][0]
        assert "overall_status = $1" in query

    @patch("app.repository.health_repository.get_db_pool")
    async def test_since_and_until_filters(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        conn.fetch.return_value = []
        mock_pool_fn.return_value = pool

        from app.repository.health_repository import get_history

        now = datetime.now(timezone.utc)
        await get_history(since=now - timedelta(hours=1), until=now)
        query = conn.fetch.call_args[0][0]
        assert "created_at >=" in query
        assert "created_at <=" in query

    @patch("app.repository.health_repository.get_db_pool")
    async def test_handles_jsonb_services(self, mock_pool_fn):
        """Services may come back as dict (JSONB) or str depending on driver."""
        pool, conn = _make_mock_pool()
        conn.fetch.return_value = [
            {
                "id": "x",
                "overall_status": "healthy",
                "services": {"db": {"status": "healthy"}},  # Already dict
                "total_latency_ms": 5.0,
                "summary": {"healthy": 1},
                "created_at": datetime.now(timezone.utc),
            }
        ]
        mock_pool_fn.return_value = pool

        from app.repository.health_repository import get_history

        rows = await get_history()
        assert rows[0]["services"]["db"]["status"] == "healthy"

    @patch("app.repository.health_repository.get_db_pool")
    async def test_handles_db_error(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        conn.fetch.side_effect = Exception("timeout")
        mock_pool_fn.return_value = pool

        from app.repository.health_repository import get_history

        result = await get_history()
        assert result == []


@pytest.mark.asyncio
class TestGetSnapshot:

    @patch("app.repository.health_repository.get_db_pool")
    async def test_returns_snapshot(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        conn.fetchrow.return_value = {
            "id": "snap-1",
            "overall_status": "degraded",
            "services": json.dumps({}),
            "total_latency_ms": 20.0,
            "summary": json.dumps({"degraded": 1}),
            "created_at": datetime.now(timezone.utc),
        }
        mock_pool_fn.return_value = pool

        from app.repository.health_repository import get_snapshot

        result = await get_snapshot("snap-1")
        assert result is not None
        assert result["overall_status"] == "degraded"

    @patch("app.repository.health_repository.get_db_pool")
    async def test_returns_none_for_missing(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        conn.fetchrow.return_value = None
        mock_pool_fn.return_value = pool

        from app.repository.health_repository import get_snapshot

        result = await get_snapshot("nonexistent")
        assert result is None


@pytest.mark.asyncio
class TestGetStatusSummary:

    @patch("app.repository.health_repository.get_db_pool")
    async def test_computes_uptime_pct(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        conn.fetch.return_value = [
            {"overall_status": "healthy", "cnt": 90},
            {"overall_status": "degraded", "cnt": 10},
        ]
        conn.fetchval.side_effect = [50.0, 100]  # avg_latency, total
        mock_pool_fn.return_value = pool

        from app.repository.health_repository import get_status_summary

        result = await get_status_summary(hours=24)
        assert result["uptime_pct"] == 90.0
        assert result["avg_latency_ms"] == 50.0
        assert result["total_checks"] == 100

    @patch("app.repository.health_repository.get_db_pool")
    async def test_handles_zero_checks(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        conn.fetch.return_value = []
        conn.fetchval.side_effect = [None, 0]
        mock_pool_fn.return_value = pool

        from app.repository.health_repository import get_status_summary

        result = await get_status_summary(hours=1)
        assert result["uptime_pct"] == 0.0
        assert result["total_checks"] == 0

    @patch("app.repository.health_repository.get_db_pool")
    async def test_handles_db_error(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        conn.fetch.side_effect = Exception("db down")
        mock_pool_fn.return_value = pool

        from app.repository.health_repository import get_status_summary

        result = await get_status_summary()
        assert result["total_checks"] == 0


@pytest.mark.asyncio
class TestPruneOldSnapshots:

    @patch("app.repository.health_repository.get_db_pool")
    async def test_returns_deleted_count(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        conn.execute.return_value = "DELETE 42"
        mock_pool_fn.return_value = pool

        from app.repository.health_repository import prune_old_snapshots

        deleted = await prune_old_snapshots(keep_days=7)
        assert deleted == 42

    @patch("app.repository.health_repository.get_db_pool")
    async def test_handles_db_error(self, mock_pool_fn):
        pool, conn = _make_mock_pool()
        conn.execute.side_effect = Exception("permission denied")
        mock_pool_fn.return_value = pool

        from app.repository.health_repository import prune_old_snapshots

        deleted = await prune_old_snapshots()
        assert deleted == 0


# ---------------------------------------------------------------------------
# Integration: health aggregator records snapshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHealthAggregatorRecording:

    @patch("app.repository.health_repository.record_snapshot")
    @patch("app.services.health_aggregator.check_system_resources")
    @patch("app.services.health_aggregator.check_task_queue")
    @patch("app.services.health_aggregator.check_bus_queue")
    @patch("app.services.health_aggregator.check_engine_state")
    @patch("app.services.health_aggregator.check_websocket_manager")
    @patch("app.services.health_aggregator.check_vector_db")
    @patch("app.services.health_aggregator.check_ollama")
    @patch("app.services.health_aggregator.check_redis")
    @patch("app.services.health_aggregator.check_database")
    async def test_comprehensive_health_records_snapshot(
        self,
        mock_db,
        mock_redis,
        mock_ollama,
        mock_vector,
        mock_ws,
        mock_engine,
        mock_bus,
        mock_tasks,
        mock_system,
        mock_record,
    ):
        from app.services.health_aggregator import (
            ServiceHealth,
            HealthStatus,
            get_comprehensive_health,
        )

        healthy = ServiceHealth(
            name="test", status=HealthStatus.HEALTHY, latency_ms=1.0
        )
        for m in [
            mock_db,
            mock_redis,
            mock_ollama,
            mock_vector,
            mock_ws,
            mock_engine,
            mock_bus,
            mock_tasks,
            mock_system,
        ]:
            m.return_value = healthy

        mock_record.return_value = "snap-id"

        import asyncio

        result = await get_comprehensive_health()
        # Give fire-and-forget task a chance to run
        await asyncio.sleep(0.05)

        assert result["status"] == "healthy"
        mock_record.assert_awaited_once()
        call_kwargs = mock_record.call_args[1]
        assert call_kwargs["overall_status"] == "healthy"


# ---------------------------------------------------------------------------
# Controller endpoint tests (FastAPI TestClient-style with mocks)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHealthHistoryEndpoints:

    @patch("app.repository.health_repository.get_history")
    async def test_history_endpoint_returns_snapshots(self, mock_history):
        mock_history.return_value = [
            {
                "id": "a",
                "overall_status": "healthy",
                "created_at": "2026-03-09T12:00:00",
            }
        ]

        # Simulate calling the endpoint function directly
        from app.controllers.health import health_history

        result = await health_history(
            limit=10, offset=0, status_filter=None, since=None, until=None
        )
        assert result["count"] == 1
        assert result["snapshots"][0]["id"] == "a"

    @patch("app.repository.health_repository.get_status_summary")
    async def test_summary_endpoint(self, mock_summary):
        mock_summary.return_value = {
            "window_hours": 24,
            "total_checks": 100,
            "by_status": {"healthy": 95, "degraded": 5},
            "avg_latency_ms": 42.0,
            "uptime_pct": 95.0,
        }

        from app.controllers.health import health_summary

        result = await health_summary(hours=24)
        assert result["uptime_pct"] == 95.0

    @patch("app.repository.health_repository.get_snapshot")
    async def test_snapshot_detail_found(self, mock_snap):
        mock_snap.return_value = {"id": "x", "overall_status": "healthy"}

        from app.controllers.health import health_snapshot_detail

        result = await health_snapshot_detail("x")
        assert result["id"] == "x"

    @patch("app.repository.health_repository.get_snapshot")
    async def test_snapshot_detail_not_found(self, mock_snap):
        mock_snap.return_value = None

        from app.controllers.health import health_snapshot_detail

        with pytest.raises(Exception) as exc_info:
            await health_snapshot_detail("missing")
        assert (
            "404" in str(exc_info.value.status_code)
            or exc_info.value.status_code == 404
        )

    @patch("app.repository.health_repository.prune_old_snapshots")
    async def test_prune_endpoint(self, mock_prune):
        mock_prune.return_value = 15

        from app.controllers.health import prune_health_history

        result = await prune_health_history(keep_days=7)
        assert result["deleted"] == 15
        assert result["keep_days"] == 7
