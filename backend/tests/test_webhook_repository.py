"""Behavior tests for app.repository.webhook_repository.

The asyncpg boundary is mocked by patching ``get_db_pool`` in the repository
module; no live Postgres is used.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.repository import webhook_repository as repo

TS = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def _make_pool(conn):
    pool = MagicMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = ctx
    return pool


@pytest.fixture
def conn():
    c = MagicMock()
    c.execute = AsyncMock(return_value="UPDATE 1")
    c.fetchrow = AsyncMock(return_value=None)
    c.fetch = AsyncMock(return_value=[])
    c.fetchval = AsyncMock(return_value=0)
    return c


@pytest.fixture
def pool_patch(conn):
    with patch.object(
        repo, "get_db_pool", AsyncMock(return_value=_make_pool(conn))
    ) as p:
        yield p


@pytest.fixture
def broken_pool():
    with patch.object(
        repo, "get_db_pool", AsyncMock(side_effect=RuntimeError("db down"))
    ) as p:
        yield p


def _reg_row(**overrides):
    row = {
        "id": "wh-1",
        "name": "hook",
        "url": "https://example.test/hook",
        "events": ("pipeline.completed", "pipeline.failed"),
        "secret": "s3cr3t",
        "headers": {"X-A": "1"},
        "max_retries": 3,
        "is_active": True,
        "delivery_count": 4,
        "failure_count": 1,
        "last_delivery_at": TS,
        "last_error": None,
        "created_at": TS,
        "updated_at": TS,
    }
    row.update(overrides)
    return row


def _delivery_row(**overrides):
    row = {
        "id": "d-1",
        "webhook_id": "wh-1",
        "event": "pipeline.completed",
        "payload": {"k": "v"},
        "attempts": 1,
        "status_code": 200,
        "response_body": "ok",
        "error": None,
        "delivered_at": TS,
        "created_at": TS,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# create_registration
# ---------------------------------------------------------------------------


async def test_create_registration_serializes_headers_and_returns_id(conn, pool_patch):
    result = await repo.create_registration(
        "wh-1", "hook", "https://x", ["a"], secret="s", headers={"H": "v"}
    )
    assert result == "wh-1"
    args = conn.execute.await_args.args
    assert "INSERT INTO webhook_registrations" in args[0]
    assert args[1:6] == ("wh-1", "hook", "https://x", ["a"], "s")
    assert json.loads(args[6]) == {"H": "v"}
    assert args[7] == 3
    assert isinstance(args[8], datetime)


async def test_create_registration_defaults_headers_to_empty_object(conn, pool_patch):
    await repo.create_registration("wh-1", "hook", "https://x", ["a"])
    assert conn.execute.await_args.args[6] == "{}"


async def test_create_registration_db_error_returns_none(broken_pool):
    assert await repo.create_registration("wh-1", "n", "u", []) is None


# ---------------------------------------------------------------------------
# get_registration / _row_to_dict
# ---------------------------------------------------------------------------


async def test_get_registration_not_found_returns_none(conn, pool_patch):
    conn.fetchrow.return_value = None
    assert await repo.get_registration("missing") is None
    assert conn.fetchrow.await_args.args[1] == "missing"


async def test_get_registration_maps_row(conn, pool_patch):
    conn.fetchrow.return_value = _reg_row()
    result = await repo.get_registration("wh-1")
    assert result["id"] == "wh-1"
    assert result["events"] == ["pipeline.completed", "pipeline.failed"]
    assert result["headers"] == {"X-A": "1"}
    assert result["last_delivery_at"] == TS.isoformat()
    assert result["created_at"] == TS.isoformat()
    assert result["updated_at"] == TS.isoformat()


async def test_get_registration_decodes_headers_str_and_null_timestamps(
    conn, pool_patch
):
    conn.fetchrow.return_value = _reg_row(
        headers='{"Authorization": "Bearer t"}',
        last_delivery_at=None,
        created_at=None,
        updated_at=None,
    )
    result = await repo.get_registration("wh-1")
    assert result["headers"] == {"Authorization": "Bearer t"}
    assert result["last_delivery_at"] is None
    assert result["created_at"] is None
    assert result["updated_at"] is None


async def test_get_registration_db_error_returns_none(broken_pool):
    assert await repo.get_registration("wh-1") is None


# ---------------------------------------------------------------------------
# list_registrations
# ---------------------------------------------------------------------------


async def test_list_registrations_no_filter(conn, pool_patch):
    conn.fetch.return_value = [_reg_row(), _reg_row(id="wh-2")]
    result = await repo.list_registrations(limit=10, offset=5)
    assert [r["id"] for r in result] == ["wh-1", "wh-2"]
    query, *params = conn.fetch.await_args.args
    assert "WHERE" not in query
    assert "LIMIT $1 OFFSET $2" in query
    assert params == [10, 5]


async def test_list_registrations_active_only_and_limit_capped(conn, pool_patch):
    await repo.list_registrations(active_only=True, limit=10_000)
    query, *params = conn.fetch.await_args.args
    assert "WHERE is_active = $1" in query
    assert "LIMIT $2 OFFSET $3" in query
    assert params == [True, 500, 0]


async def test_list_registrations_empty(conn, pool_patch):
    conn.fetch.return_value = []
    assert await repo.list_registrations() == []


async def test_list_registrations_db_error_returns_empty_list(broken_pool):
    assert await repo.list_registrations() == []


# ---------------------------------------------------------------------------
# update_registration
# ---------------------------------------------------------------------------


async def test_update_registration_no_allowed_fields_skips_db(pool_patch):
    assert await repo.update_registration("wh-1", bogus=1, name=None) is False
    pool_patch.assert_not_awaited()


async def test_update_registration_builds_set_clause(conn, pool_patch):
    conn.execute.return_value = "UPDATE 1"
    ok = await repo.update_registration(
        "wh-1", name="new", headers={"A": "b"}, is_active=False, junk="x"
    )
    assert ok is True
    query, *params = conn.execute.await_args.args
    assert "name = $1" in query
    assert "headers = $2::jsonb" in query
    assert "is_active = $3" in query
    assert "updated_at = $4" in query
    assert "WHERE id = $5" in query
    assert "junk" not in query
    assert params[0] == "new"
    assert json.loads(params[1]) == {"A": "b"}
    assert params[2] is False
    assert isinstance(params[3], datetime)
    assert params[4] == "wh-1"


async def test_update_registration_not_found_returns_false(conn, pool_patch):
    conn.execute.return_value = "UPDATE 0"
    assert await repo.update_registration("missing", name="x") is False


async def test_update_registration_db_error_returns_false(broken_pool):
    assert await repo.update_registration("wh-1", name="x") is False


# ---------------------------------------------------------------------------
# delete_registration
# ---------------------------------------------------------------------------


async def test_delete_registration_success(conn, pool_patch):
    conn.execute.return_value = "DELETE 1"
    assert await repo.delete_registration("wh-1") is True
    assert conn.execute.await_args.args[1] == "wh-1"


async def test_delete_registration_not_found(conn, pool_patch):
    conn.execute.return_value = "DELETE 0"
    assert await repo.delete_registration("missing") is False


async def test_delete_registration_db_error_returns_false(broken_pool):
    assert await repo.delete_registration("wh-1") is False


# ---------------------------------------------------------------------------
# counters
# ---------------------------------------------------------------------------


async def test_increment_delivery_count(conn, pool_patch):
    assert await repo.increment_delivery_count("wh-1") is None
    args = conn.execute.await_args.args
    assert "delivery_count = delivery_count + 1" in args[0]
    assert args[1] == "wh-1"
    assert isinstance(args[2], datetime)


async def test_increment_delivery_count_swallows_db_error(broken_pool):
    assert await repo.increment_delivery_count("wh-1") is None


async def test_increment_failure_count_truncates_error(conn, pool_patch):
    await repo.increment_failure_count("wh-1", "e" * 5000)
    args = conn.execute.await_args.args
    assert "failure_count = failure_count + 1" in args[0]
    assert args[1] == "wh-1"
    assert args[2] == "e" * 1000


async def test_increment_failure_count_swallows_db_error(broken_pool):
    assert await repo.increment_failure_count("wh-1", "boom") is None


# ---------------------------------------------------------------------------
# record_delivery
# ---------------------------------------------------------------------------


async def test_record_delivery_truncates_and_serializes(conn, pool_patch):
    result = await repo.record_delivery(
        "d-1",
        "wh-1",
        "pipeline.completed",
        {"a": [1, 2]},
        attempts=2,
        status_code=500,
        response_body="r" * 900,
        error="x" * 2000,
        delivered_at=TS,
    )
    assert result == "d-1"
    args = conn.execute.await_args.args
    assert args[1:4] == ("d-1", "wh-1", "pipeline.completed")
    assert json.loads(args[4]) == {"a": [1, 2]}
    assert args[5] == 2
    assert args[6] == 500
    assert args[7] == "r" * 500
    assert args[8] == "x" * 1000
    assert args[9] == TS


async def test_record_delivery_optional_fields_none(conn, pool_patch):
    await repo.record_delivery("d-1", "wh-1", "ev", {})
    args = conn.execute.await_args.args
    assert args[6:10] == (None, None, None, None)


async def test_record_delivery_db_error_returns_none(broken_pool):
    assert await repo.record_delivery("d-1", "wh-1", "ev", {}) is None


# ---------------------------------------------------------------------------
# get_deliveries / _delivery_to_dict
# ---------------------------------------------------------------------------


async def test_get_deliveries_with_all_filters(conn, pool_patch):
    conn.fetch.return_value = [
        _delivery_row(),
        _delivery_row(id="d-2", payload='{"k": 2}', delivered_at=None, created_at=None),
    ]
    result = await repo.get_deliveries(
        webhook_id="wh-1", event_filter="pipeline.completed", limit=999, offset=3
    )
    query, *params = conn.fetch.await_args.args
    assert "webhook_id = $1 AND event = $2" in query
    assert "LIMIT $3 OFFSET $4" in query
    assert params == ["wh-1", "pipeline.completed", 500, 3]

    assert result[0]["payload"] == {"k": "v"}
    assert result[0]["delivered_at"] == TS.isoformat()
    assert result[0]["created_at"] == TS.isoformat()
    assert result[1]["payload"] == {"k": 2}
    assert result[1]["delivered_at"] is None
    assert result[1]["created_at"] is None


async def test_get_deliveries_event_filter_only(conn, pool_patch):
    await repo.get_deliveries(event_filter="ev")
    query, *params = conn.fetch.await_args.args
    assert "WHERE event = $1" in query
    assert "webhook_id" not in query
    assert params == ["ev", 50, 0]


async def test_get_deliveries_no_filters_empty(conn, pool_patch):
    assert await repo.get_deliveries() == []
    query, *params = conn.fetch.await_args.args
    assert "WHERE" not in query
    assert params == [50, 0]


async def test_get_deliveries_db_error_returns_empty_list(broken_pool):
    assert await repo.get_deliveries(webhook_id="wh-1") == []


# ---------------------------------------------------------------------------
# prune_old_deliveries
# ---------------------------------------------------------------------------


async def test_prune_old_deliveries_parses_count(conn, pool_patch):
    conn.execute.return_value = "DELETE 17"
    assert await repo.prune_old_deliveries(keep_days=7) == 17
    assert conn.execute.await_args.args[1] == 7


async def test_prune_old_deliveries_unparseable_status_returns_zero(conn, pool_patch):
    conn.execute.return_value = "DELETE"
    assert await repo.prune_old_deliveries() == 0


async def test_prune_old_deliveries_db_error_returns_zero(broken_pool):
    assert await repo.prune_old_deliveries() == 0


# ---------------------------------------------------------------------------
# get_delivery_stats
# ---------------------------------------------------------------------------


async def test_get_delivery_stats_computes_rates(conn, pool_patch):
    conn.fetchval.side_effect = [8, 6]
    conn.fetch.return_value = [
        {"event": "pipeline.completed", "cnt": 5},
        {"event": "pipeline.failed", "cnt": 3},
    ]
    stats = await repo.get_delivery_stats(hours=12)
    assert stats == {
        "window_hours": 12,
        "total_deliveries": 8,
        "successful": 6,
        "failed": 2,
        "success_rate": 0.75,
        "by_event": {"pipeline.completed": 5, "pipeline.failed": 3},
    }
    for call in conn.fetchval.await_args_list:
        assert call.args[1] == 12


async def test_get_delivery_stats_null_counts_no_division(conn, pool_patch):
    conn.fetchval.side_effect = [None, None]
    conn.fetch.return_value = []
    stats = await repo.get_delivery_stats()
    assert stats["total_deliveries"] == 0
    assert stats["successful"] == 0
    assert stats["failed"] == 0
    assert stats["success_rate"] == 0.0
    assert stats["by_event"] == {}
    assert stats["window_hours"] == 24


async def test_get_delivery_stats_db_error_returns_zeroed(broken_pool):
    assert await repo.get_delivery_stats(hours=6) == {
        "window_hours": 6,
        "total_deliveries": 0,
        "successful": 0,
        "failed": 0,
        "success_rate": 0.0,
        "by_event": {},
    }
