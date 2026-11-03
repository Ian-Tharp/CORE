"""
Behavior tests for ``app/repository/run_repository.py``.

The asyncpg pool is replaced with a MagicMock whose ``acquire()`` async context
manager yields a mock connection, so we can assert on the SQL + parameters and
on row -> dict mapping without a live Postgres.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from app.repository import run_repository as repo

RUN_ID = "3f2b1c4d-0000-4000-8000-000000000001"


@pytest.fixture
def conn(monkeypatch):
    c = MagicMock()
    c.fetchrow = AsyncMock(return_value=None)
    c.fetch = AsyncMock(return_value=[])
    c.fetchval = AsyncMock(return_value=0)
    c.execute = AsyncMock(return_value="UPDATE 1")

    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=c)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(repo, "get_db_pool", AsyncMock(return_value=pool))
    return c


def _sql(call) -> str:
    return " ".join(call.args[0].split())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_row_to_dict_serializes_datetimes_and_uuids(self):
        row = {
            "run_id": UUID(RUN_ID),
            "created_at": datetime(2026, 3, 4, 5, 6, 7),
            "status": "running",
            "state": {"a": 1},
            "completed_at": None,
        }
        out = repo._row_to_dict(row)
        assert out == {
            "run_id": RUN_ID,
            "created_at": "2026-03-04T05:06:07",
            "status": "running",
            "state": {"a": 1},
            "completed_at": None,
        }

    def test_serialize_falls_back_to_str(self):
        out = json.loads(
            repo._serialize({"t": datetime(2026, 1, 1), "u": UUID(RUN_ID)})
        )
        assert out == {"t": "2026-01-01 00:00:00", "u": RUN_ID}


# ---------------------------------------------------------------------------
# create / get
# ---------------------------------------------------------------------------


class TestCreateRun:
    async def test_inserts_and_returns_mapped_row(self, conn):
        conn.fetchrow.return_value = {
            "run_id": UUID(RUN_ID),
            "user_id": "u1",
            "input_text": "hi",
            "status": "running",
            "created_at": datetime(2026, 1, 1),
        }

        out = await repo.create_run(
            run_id=RUN_ID, user_id="u1", input_text="hi", state={"k": "v"}
        )

        assert out["run_id"] == RUN_ID
        assert out["created_at"] == "2026-01-01T00:00:00"
        call = conn.fetchrow.await_args
        assert "INSERT INTO core_runs" in _sql(call)
        assert "'running'" in _sql(call)
        assert call.args[1:] == (RUN_ID, "u1", "hi", '{"k": "v"}')

    async def test_without_state_passes_null(self, conn):
        conn.fetchrow.return_value = {"run_id": RUN_ID}

        await repo.create_run(run_id=RUN_ID, input_text="hi")

        assert conn.fetchrow.await_args.args[1:] == (RUN_ID, None, "hi", None)

    async def test_db_error_propagates(self, conn):
        conn.fetchrow.side_effect = RuntimeError("unique violation")
        with pytest.raises(RuntimeError, match="unique violation"):
            await repo.create_run(run_id=RUN_ID, input_text="hi")


class TestGetRun:
    async def test_found(self, conn):
        conn.fetchrow.return_value = {
            "run_id": UUID(RUN_ID),
            "state": '{"user_input": "x"}',
            "completed_at": datetime(2026, 5, 5, 5, 5),
        }

        out = await repo.get_run(RUN_ID)

        assert out == {
            "run_id": RUN_ID,
            "state": '{"user_input": "x"}',  # JSON string left untouched
            "completed_at": "2026-05-05T05:05:00",
        }
        call = conn.fetchrow.await_args
        assert _sql(call) == "SELECT * FROM core_runs WHERE run_id = $1::uuid"
        assert call.args[1] == RUN_ID

    async def test_not_found_returns_none(self, conn):
        conn.fetchrow.return_value = None
        assert await repo.get_run(RUN_ID) is None


# ---------------------------------------------------------------------------
# save_run (upsert used by the engine controller)
# ---------------------------------------------------------------------------


class TestSaveRun:
    async def test_running_snapshot(self, conn):
        state = {
            "run_id": RUN_ID,
            "user_id": "u1",
            "user_input": "question",
            "current_node": "START",
            "errors": [],
            "completed_at": None,
            "started_at": datetime(2026, 1, 1),
        }

        await repo.save_run(state)

        call = conn.execute.await_args
        sql = _sql(call)
        assert "INSERT INTO core_runs" in sql
        assert "ON CONFLICT (run_id) DO UPDATE" in sql
        run_id, user_id, input_text, status, state_json, error, completed = call.args[
            1:
        ]
        assert (run_id, user_id, input_text, status, error, completed) == (
            RUN_ID,
            "u1",
            "question",
            "running",
            None,
            None,
        )
        assert json.loads(state_json)["started_at"] == "2026-01-01 00:00:00"

    async def test_completed_snapshot(self, conn):
        done_at = datetime(2026, 1, 1, 1)
        await repo.save_run(
            {
                "run_id": RUN_ID,
                "user_input": "q",
                "current_node": "conversation",
                "errors": ["non-fatal warning"],
                "completed_at": done_at,
            }
        )
        args = conn.execute.await_args.args
        assert args[4] == "completed"
        assert args[7] == done_at

    async def test_end_node_counts_as_completed(self, conn):
        await repo.save_run(
            {"run_id": RUN_ID, "user_input": "q", "current_node": "END"}
        )
        assert conn.execute.await_args.args[4] == "completed"

    async def test_errors_mark_failed(self, conn):
        await repo.save_run(
            {
                "run_id": RUN_ID,
                "user_input": "q",
                "current_node": "START",
                "errors": ["Execution failed: a", "b"],
            }
        )
        args = conn.execute.await_args.args
        assert args[4] == "failed"
        assert args[6] == "Execution failed: a; b"

    async def test_missing_input_defaults_to_empty(self, conn):
        await repo.save_run({"run_id": RUN_ID, "user_input": None})
        assert conn.execute.await_args.args[3] == ""

    async def test_engine_state_dump_roundtrip(self, conn):
        from app.models.core_state import COREState

        state = COREState(user_input="hello", user_id="u2")
        await repo.save_run(state.model_dump())

        args = conn.execute.await_args.args
        assert args[1] == state.run_id
        assert args[3] == "hello"
        assert json.loads(args[5])["user_input"] == "hello"

    async def test_db_error_propagates(self, conn):
        conn.execute.side_effect = ConnectionError("db gone")
        with pytest.raises(ConnectionError):
            await repo.save_run({"run_id": RUN_ID, "user_input": "q"})


# ---------------------------------------------------------------------------
# updates
# ---------------------------------------------------------------------------


class TestUpdates:
    async def test_update_run_status(self, conn):
        await repo.update_run_status(RUN_ID, "cancelled")
        call = conn.execute.await_args
        assert "SET status = $2" in _sql(call)
        assert call.args[1:] == (RUN_ID, "cancelled")

    async def test_update_run_state(self, conn):
        await repo.update_run_state(RUN_ID, {"current_node": "reasoning"})
        call = conn.execute.await_args
        assert "SET state = $2::jsonb" in _sql(call)
        assert call.args[1:] == (RUN_ID, '{"current_node": "reasoning"}')

    async def test_complete_run_with_state(self, conn):
        await repo.complete_run(RUN_ID, {"response": "ok"}, state={"s": 1})
        call = conn.execute.await_args
        sql = _sql(call)
        assert "status = 'completed'" in sql
        assert "COALESCE($3::jsonb, state)" in sql
        assert call.args[1:] == (RUN_ID, '{"response": "ok"}', '{"s": 1}')

    async def test_complete_run_without_state_keeps_existing(self, conn):
        await repo.complete_run(RUN_ID, {"response": "ok"})
        assert conn.execute.await_args.args[3] is None

    async def test_fail_run(self, conn):
        await repo.fail_run(RUN_ID, "kaboom")
        call = conn.execute.await_args
        assert "status = 'failed'" in _sql(call)
        assert call.args[1:] == (RUN_ID, "kaboom")


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------


class TestListRuns:
    async def test_no_filters_uses_default_pagination(self, conn):
        conn.fetch.return_value = [
            {"run_id": UUID(RUN_ID), "created_at": datetime(2026, 1, 1)},
        ]

        out = await repo.list_runs()

        assert out == [{"run_id": RUN_ID, "created_at": "2026-01-01T00:00:00"}]
        call = conn.fetch.await_args
        sql = _sql(call)
        assert "WHERE" not in sql
        assert "ORDER BY created_at DESC LIMIT $1 OFFSET $2" in sql
        assert call.args[1:] == (50, 0)

    async def test_user_filter_only(self, conn):
        await repo.list_runs(user_id="u1", limit=5)
        call = conn.fetch.await_args
        sql = _sql(call)
        assert "WHERE user_id = $1 ORDER BY" in sql
        assert "LIMIT $2 OFFSET $3" in sql
        assert call.args[1:] == ("u1", 5, 0)

    async def test_status_filter_only(self, conn):
        await repo.list_runs(status="failed")
        call = conn.fetch.await_args
        assert "WHERE status = $1" in _sql(call)
        assert call.args[1:] == ("failed", 50, 0)

    async def test_both_filters_and_offset(self, conn):
        await repo.list_runs(user_id="u1", status="completed", limit=10, offset=20)
        call = conn.fetch.await_args
        sql = _sql(call)
        assert "WHERE user_id = $1 AND status = $2" in sql
        assert "LIMIT $3 OFFSET $4" in sql
        assert call.args[1:] == ("u1", "completed", 10, 20)

    async def test_empty_result(self, conn):
        conn.fetch.return_value = []
        assert await repo.list_runs(user_id="nobody") == []


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------


class TestEvents:
    async def test_log_event_with_data(self, conn):
        await repo.log_event(RUN_ID, "node_start", "comprehension", {"x": 1})
        call = conn.execute.await_args
        assert "INSERT INTO core_run_events" in _sql(call)
        assert call.args[1:] == (RUN_ID, "node_start", "comprehension", '{"x": 1}')

    async def test_log_event_without_data(self, conn):
        await repo.log_event(RUN_ID, "error")
        assert conn.execute.await_args.args[1:] == (RUN_ID, "error", None, None)

    async def test_log_event_swallows_db_errors(self, conn):
        conn.execute.side_effect = RuntimeError("fk violation")
        # Must not raise: event logging never breaks a run
        assert await repo.log_event(RUN_ID, "node_start") is None

    async def test_get_run_events(self, conn):
        conn.fetch.return_value = [
            {
                "event_id": UUID(int=1),
                "run_id": UUID(RUN_ID),
                "event_type": "node_start",
                "data": None,
                "created_at": datetime(2026, 1, 1),
            }
        ]

        out = await repo.get_run_events(RUN_ID, limit=10)

        assert out[0]["event_id"] == str(UUID(int=1))
        assert out[0]["run_id"] == RUN_ID
        assert out[0]["created_at"] == "2026-01-01T00:00:00"
        call = conn.fetch.await_args
        assert "ORDER BY created_at ASC" in _sql(call)
        assert call.args[1:] == (RUN_ID, 10)

    async def test_get_run_events_default_limit(self, conn):
        assert await repo.get_run_events(RUN_ID) == []
        assert conn.fetch.await_args.args[1:] == (RUN_ID, 200)


# ---------------------------------------------------------------------------
# stats & cleanup
# ---------------------------------------------------------------------------


class TestStats:
    async def test_aggregates(self, conn):
        conn.fetchval.side_effect = [10, 7, 2, 3]

        out = await repo.get_run_stats()

        assert out == {
            "total_runs": 10,
            "completed_runs": 7,
            "failed_runs": 2,
            "success_rate": 0.7,
            "runs_last_hour": 3,
        }
        assert conn.fetchval.await_count == 4

    async def test_empty_table_and_null_counts(self, conn):
        conn.fetchval.side_effect = [None, None, None, None]

        out = await repo.get_run_stats()

        assert out["total_runs"] == 0
        assert out["success_rate"] == 0
        assert "error" not in out

    async def test_db_error_returns_zeroed_stats(self, conn):
        conn.fetchval.side_effect = RuntimeError("relation does not exist")

        out = await repo.get_run_stats()

        assert out == {
            "total_runs": 0,
            "completed_runs": 0,
            "failed_runs": 0,
            "success_rate": 0,
            "runs_last_hour": 0,
            "error": "relation does not exist",
        }


class TestCleanup:
    async def test_returns_deleted_count(self, conn):
        conn.fetch.return_value = [{"run_id": "a"}, {"run_id": "b"}]

        assert await repo.cleanup_old_runs(7) == 2

        call = conn.fetch.await_args
        assert "DELETE FROM core_runs" in _sql(call)
        assert "make_interval(days => $1)" in _sql(call)
        assert call.args[1] == 7

    async def test_default_days_and_nothing_deleted(self, conn):
        assert await repo.cleanup_old_runs() == 0
        assert conn.fetch.await_args.args[1] == 30

    async def test_pool_unavailable_propagates(self, monkeypatch):
        monkeypatch.setattr(
            repo, "get_db_pool", AsyncMock(side_effect=RuntimeError("no pool"))
        )
        with pytest.raises(RuntimeError, match="no pool"):
            await repo.cleanup_old_runs()
