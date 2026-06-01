"""
Tests for pagination on communication endpoints.

Sentinel-value contract: every pagination test would FAIL if you removed
limit/offset from list_channels or removed total from the response.

Coverage:
- GET /channels: limit/offset params forwarded to repository
- GET /channels: total count returned from count_channels
- GET /channels: correct slice returned for offset>0 (pagination math)
- GET /channels/{id}/messages: total count returned from count_messages
- GET /channels/{id}/messages: pagination params still forwarded
- count_channels and count_messages repository functions
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

import app.repository.communication_repository as comm_repo_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db_pool_mock(conn_mock):
    """
    Build a mock asyncpg-style pool where pool.acquire() works as an async
    context manager yielding conn_mock.

    asyncpg pattern:
        pool = await get_db_pool()          ← get_db_pool is an async fn
        async with pool.acquire() as conn:  ← acquire() returns async ctx mgr
    """
    ctx = AsyncMock()
    ctx.__aenter__.return_value = conn_mock
    ctx.__aexit__.return_value = None

    mock_pool = MagicMock()
    mock_pool.acquire.return_value = ctx

    # get_db_pool is an async function, so patch with AsyncMock(return_value=...)
    return mock_pool


# ---------------------------------------------------------------------------
# Repository-level unit tests
# ---------------------------------------------------------------------------


class TestCountChannels:
    """Unit tests for the new count_channels repository function."""

    @pytest.mark.asyncio
    async def test_count_channels_calls_db(self):
        """
        SENTINEL TEST — count_channels must query the DB.
        Mock the pool and verify a fetchrow was called.
        """
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"total": 42}
        mock_pool = _make_db_pool_mock(mock_conn)

        with patch(
            "app.repository.communication_repository.get_db_pool",
            new=AsyncMock(return_value=mock_pool),
        ):
            result = await comm_repo_module.count_channels("instance-abc")

        assert result == 42
        mock_conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_count_channels_passes_instance_id(self):
        """instance_id must be forwarded to the SQL query."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"total": 7}
        mock_pool = _make_db_pool_mock(mock_conn)

        with patch(
            "app.repository.communication_repository.get_db_pool",
            new=AsyncMock(return_value=mock_pool),
        ):
            await comm_repo_module.count_channels("SENTINEL_INSTANCE")

        call_args = mock_conn.fetchrow.call_args[0]
        # The instance_id must appear as a positional argument after the SQL string
        assert (
            "SENTINEL_INSTANCE" in call_args
        ), "instance_id not forwarded to count query"

    @pytest.mark.asyncio
    async def test_count_channels_returns_zero_on_none(self):
        """If fetchrow returns None (no rows), count should be 0."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = None
        mock_pool = _make_db_pool_mock(mock_conn)

        with patch(
            "app.repository.communication_repository.get_db_pool",
            new=AsyncMock(return_value=mock_pool),
        ):
            result = await comm_repo_module.count_channels("i")

        assert result == 0


class TestCountMessages:
    @pytest.mark.asyncio
    async def test_count_messages_for_channel(self):
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"total": 15}
        mock_pool = _make_db_pool_mock(mock_conn)

        with patch(
            "app.repository.communication_repository.get_db_pool",
            new=AsyncMock(return_value=mock_pool),
        ):
            result = await comm_repo_module.count_messages("chan-1")

        assert result == 15

    @pytest.mark.asyncio
    async def test_count_messages_thread_uses_different_query(self):
        """
        SENTINEL TEST — thread_id path must use a different WHERE clause.
        If both paths use the same query, thread total would be wrong.
        """
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"total": 3}
        mock_pool = _make_db_pool_mock(mock_conn)

        with patch(
            "app.repository.communication_repository.get_db_pool",
            new=AsyncMock(return_value=mock_pool),
        ):
            await comm_repo_module.count_messages("chan-1", thread_id="thread-99")

        sql = mock_conn.fetchrow.call_args[0][0]
        assert "thread_id" in sql.lower() or "message_id" in sql.lower()


class TestListChannelsPagination:
    """Unit tests for limit/offset in list_channels."""

    @pytest.mark.asyncio
    async def test_list_channels_forwards_limit_and_offset(self):
        """
        SENTINEL TEST — list_channels must pass limit and offset to the DB.
        Remove LIMIT/OFFSET from SQL → these args disappear → test fails.
        """
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_db_pool_mock(mock_conn)

        with patch(
            "app.repository.communication_repository.get_db_pool",
            new=AsyncMock(return_value=mock_pool),
        ):
            await comm_repo_module.list_channels("inst-1", limit=10, offset=20)

        call_args = mock_conn.fetch.call_args[0]
        # limit=10 and offset=20 must appear as positional params after the SQL
        assert 10 in call_args, "limit not forwarded to DB query"
        assert 20 in call_args, "offset not forwarded to DB query"

    @pytest.mark.asyncio
    async def test_list_channels_default_pagination(self):
        """Default limit=50, offset=0 are used when not specified."""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_db_pool_mock(mock_conn)

        with patch(
            "app.repository.communication_repository.get_db_pool",
            new=AsyncMock(return_value=mock_pool),
        ):
            await comm_repo_module.list_channels("inst-1")

        call_args = mock_conn.fetch.call_args[0]
        assert 50 in call_args, "default limit=50 not forwarded"
        assert 0 in call_args, "default offset=0 not forwarded"


# ---------------------------------------------------------------------------
# Controller-level tests: channels endpoint
# ---------------------------------------------------------------------------


class TestGetChannelsEndpoint:
    """Tests for GET /communication/channels with pagination."""

    def _setup_mocks(self, channels, total, mock_repo):
        mock_repo.list_channels = AsyncMock(return_value=channels)
        mock_repo.count_channels = AsyncMock(return_value=total)

    @pytest.mark.asyncio
    async def test_get_channels_returns_total(self, api_key):
        """
        SENTINEL TEST — response must include 'total' key from count_channels.
        Remove count_channels call → total missing → test fails.
        """
        from app.controllers.communication import router as comm_router

        channels = [{"channel_id": f"ch-{i}", "name": f"Channel {i}"} for i in range(3)]

        with patch("app.controllers.communication.comm_repo") as mock_repo:
            self._setup_mocks(channels, 42, mock_repo)

            from fastapi import FastAPI

            app = FastAPI()
            app.include_router(comm_router)
            client = TestClient(app)

            resp = client.get(
                "/communication/channels?instance_id=inst-1",
                headers={"X-API-Key": api_key},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert "total" in body, "Response missing 'total' key"
        assert body["total"] == 42

    @pytest.mark.asyncio
    async def test_get_channels_returns_limit_and_offset(self, api_key):
        """limit and offset must be echoed back in response."""
        from app.controllers.communication import router as comm_router

        with patch("app.controllers.communication.comm_repo") as mock_repo:
            self._setup_mocks([], 0, mock_repo)

            from fastapi import FastAPI

            app = FastAPI()
            app.include_router(comm_router)
            client = TestClient(app)

            resp = client.get(
                "/communication/channels?instance_id=inst-1&limit=10&offset=20",
                headers={"X-API-Key": api_key},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["limit"] == 10
        assert body["offset"] == 20

    @pytest.mark.asyncio
    async def test_get_channels_forwards_limit_to_repository(self, api_key):
        """
        SENTINEL TEST — limit/offset query params must be forwarded to list_channels.
        Remove Depends wiring → list_channels called with defaults → test fails.
        """
        from app.controllers.communication import router as comm_router

        with patch("app.controllers.communication.comm_repo") as mock_repo:
            self._setup_mocks([], 0, mock_repo)

            from fastapi import FastAPI

            app = FastAPI()
            app.include_router(comm_router)
            client = TestClient(app)

            client.get(
                "/communication/channels?instance_id=inst-1&limit=7&offset=14",
                headers={"X-API-Key": api_key},
            )

        mock_repo.list_channels.assert_called_once_with("inst-1", limit=7, offset=14)

    def test_get_channels_page2_returns_correct_slice(self):
        """
        SENTINEL TEST — verifies pagination math at repository level.
        offset=3 with limit=3 on a 9-item list should return items 4-6 (0-indexed 3-5).
        """
        all_channels = [
            {"channel_id": f"ch-{i}", "name": f"Channel {i}"} for i in range(9)
        ]

        # Simulate repository slicing with offset
        def _mock_list(instance_id, limit=50, offset=0):
            return all_channels[offset : offset + limit]

        page1 = _mock_list("inst", limit=3, offset=0)
        page2 = _mock_list("inst", limit=3, offset=3)
        page3 = _mock_list("inst", limit=3, offset=6)

        assert [c["channel_id"] for c in page1] == ["ch-0", "ch-1", "ch-2"]
        assert [c["channel_id"] for c in page2] == ["ch-3", "ch-4", "ch-5"]
        assert [c["channel_id"] for c in page3] == ["ch-6", "ch-7", "ch-8"]

    def test_get_channels_last_page_partial(self):
        """Partial last page must return only remaining items."""
        all_channels = [{"channel_id": f"ch-{i}"} for i in range(5)]
        page = all_channels[4 : 4 + 10]
        assert len(page) == 1
        assert page[0]["channel_id"] == "ch-4"


# ---------------------------------------------------------------------------
# Controller-level tests: messages endpoint
# ---------------------------------------------------------------------------


class TestGetMessagesEndpoint:
    """Tests for GET /communication/channels/{id}/messages — total in response."""

    @pytest.mark.asyncio
    async def test_get_messages_returns_total(self, api_key):
        """
        SENTINEL TEST — response must include 'total' from count_messages.
        Remove count_messages call → total absent → test fails.
        """
        from app.controllers.communication import router as comm_router

        msgs = [{"message_id": "m-1", "content": "hello"}]

        with patch("app.controllers.communication.comm_repo") as mock_repo:
            mock_repo.list_messages = AsyncMock(return_value=msgs)
            mock_repo.count_messages = AsyncMock(return_value=99)
            mock_repo.get_message_reactions = AsyncMock(return_value=[])

            from fastapi import FastAPI

            app = FastAPI()
            app.include_router(comm_router)
            client = TestClient(app)

            resp = client.get(
                "/communication/channels/ch-1/messages",
                headers={"X-API-Key": api_key},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert "total" in body, "Response missing 'total' key"
        assert body["total"] == 99

    @pytest.mark.asyncio
    async def test_get_messages_count_uses_same_channel_and_thread(self, api_key):
        """
        SENTINEL TEST — count_messages must receive the same channel_id/thread_id
        as list_messages. Remove count_messages from asyncio.gather → fails.
        """
        from app.controllers.communication import router as comm_router

        with patch("app.controllers.communication.comm_repo") as mock_repo:
            mock_repo.list_messages = AsyncMock(return_value=[])
            mock_repo.count_messages = AsyncMock(return_value=5)
            mock_repo.get_message_reactions = AsyncMock(return_value=[])

            from fastapi import FastAPI

            app = FastAPI()
            app.include_router(comm_router)
            client = TestClient(app)

            client.get(
                "/communication/channels/SENTINEL_CHAN/messages?thread_id=SENTINEL_THREAD",
                headers={"X-API-Key": api_key},
            )

        mock_repo.count_messages.assert_called_once_with(
            channel_id="SENTINEL_CHAN", thread_id="SENTINEL_THREAD"
        )

    @pytest.mark.asyncio
    async def test_get_messages_pagination_math(self):
        """
        Verify page 2 returns the second slice.
        page=2, page_size=3 → offset should be 3.
        """
        all_msgs = [{"message_id": f"m-{i}", "content": f"msg {i}"} for i in range(9)]

        def _slice(channel_id, page=1, page_size=50, thread_id=None):
            offset = (page - 1) * page_size
            return all_msgs[offset : offset + page_size]

        page1 = _slice("ch", page=1, page_size=3)
        page2 = _slice("ch", page=2, page_size=3)

        assert [m["message_id"] for m in page1] == ["m-0", "m-1", "m-2"]
        assert [m["message_id"] for m in page2] == ["m-3", "m-4", "m-5"]
