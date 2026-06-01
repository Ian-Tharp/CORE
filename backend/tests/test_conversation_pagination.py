"""
Tests for conversation repository pagination and search (P5-conversations).

Sentinel-value contract:
- Removing count_conversations() breaks test_total_in_response
- Removing search param breaks test_search_filter_passed_to_repository
- Removing pagination metadata breaks test_response_includes_pagination_keys
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tests.test_communication_pagination import _make_db_pool_mock


# ---------------------------------------------------------------------------
# count_conversations — unit tests
# ---------------------------------------------------------------------------


class TestCountConversations:
    @pytest.mark.asyncio
    async def test_count_returns_integer(self):
        """count_conversations() must return the DB scalar directly."""
        from app.repository.conversation_repository import count_conversations

        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 42
        mock_pool = _make_db_pool_mock(mock_conn)

        with patch(
            "app.repository.conversation_repository.get_db_pool",
            new=AsyncMock(return_value=mock_pool),
        ):
            result = await count_conversations()

        assert result == 42

    @pytest.mark.asyncio
    async def test_count_with_search_passes_ilike_pattern(self):
        """
        SENTINEL TEST — count_conversations(search='foo') must issue a LIKE/ILIKE query.
        Remove the search branch → fetchval called without %foo% → test fails.
        """
        from app.repository.conversation_repository import count_conversations

        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 3
        mock_pool = _make_db_pool_mock(mock_conn)

        with patch(
            "app.repository.conversation_repository.get_db_pool",
            new=AsyncMock(return_value=mock_pool),
        ):
            result = await count_conversations(search="SENTINEL_SEARCH")

        assert result == 3
        call_args = mock_conn.fetchval.call_args
        # The query and args must contain the ILIKE pattern
        assert any("SENTINEL_SEARCH" in str(a) for a in call_args[0])

    @pytest.mark.asyncio
    async def test_count_without_search_omits_where_clause(self):
        """count_conversations() without search must use the simple COUNT(*)."""
        from app.repository.conversation_repository import count_conversations

        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 10
        mock_pool = _make_db_pool_mock(mock_conn)

        with patch(
            "app.repository.conversation_repository.get_db_pool",
            new=AsyncMock(return_value=mock_pool),
        ):
            await count_conversations()

        sql = mock_conn.fetchval.call_args[0][0]
        assert "ILIKE" not in sql.upper()


# ---------------------------------------------------------------------------
# list_conversations — search param
# ---------------------------------------------------------------------------


class TestListConversationsSearch:
    @pytest.mark.asyncio
    async def test_search_param_injected_into_query(self):
        """
        SENTINEL TEST — list_conversations(search='SENTINEL') must pass %SENTINEL%
        to the DB fetch call. Remove search → query called without pattern → test fails.
        """
        from app.repository.conversation_repository import list_conversations

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_db_pool_mock(mock_conn)

        with patch(
            "app.repository.conversation_repository.get_db_pool",
            new=AsyncMock(return_value=mock_pool),
        ):
            await list_conversations(search="SENTINEL_TITLE")

        call_args = mock_conn.fetch.call_args[0]
        assert any("SENTINEL_TITLE" in str(a) for a in call_args)

    @pytest.mark.asyncio
    async def test_no_search_uses_plain_query(self):
        """Without search, query must not contain ILIKE."""
        from app.repository.conversation_repository import list_conversations

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_db_pool_mock(mock_conn)

        with patch(
            "app.repository.conversation_repository.get_db_pool",
            new=AsyncMock(return_value=mock_pool),
        ):
            await list_conversations()

        sql = mock_conn.fetch.call_args[0][0]
        assert "ILIKE" not in sql.upper()

    @pytest.mark.asyncio
    async def test_pagination_offset_math_page2(self):
        """
        SENTINEL TEST — page=2, page_size=10 → OFFSET=10 in DB call.
        Break offset math → DB called with wrong offset → test fails.
        """
        from app.repository.conversation_repository import list_conversations

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool = _make_db_pool_mock(mock_conn)

        with patch(
            "app.repository.conversation_repository.get_db_pool",
            new=AsyncMock(return_value=mock_pool),
        ):
            await list_conversations(page=2, page_size=10)

        call_args = mock_conn.fetch.call_args[0]
        # Args are (sql, limit, offset): limit=10, offset=10
        assert 10 in call_args  # limit
        assert 10 in call_args  # offset (same value coincidentally, but both present)
        # More specifically: the second numeric arg is offset = (page-1)*page_size = 10
        numeric_args = [a for a in call_args[1:] if isinstance(a, int)]
        assert numeric_args[1] == 10  # offset


# ---------------------------------------------------------------------------
# GET /conversations/ endpoint — pagination metadata in response
# ---------------------------------------------------------------------------


class TestConversationsEndpointPagination:
    def _make_app(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.controllers.conversations import router

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_response_includes_pagination_keys(self):
        """
        SENTINEL TEST — response must contain conversations, total, page, page_size.
        Remove any key → assertion fails → proves the field is required.
        """
        client = self._make_app()

        with (
            patch(
                "app.repository.conversation_repository.get_db_pool",
                new=AsyncMock(return_value=_make_db_pool_mock(_conn_returning([], 0))),
            ),
        ):
            resp = client.get("/conversations/")

        assert resp.status_code == 200
        body = resp.json()
        assert "conversations" in body
        assert "total" in body
        assert "page" in body
        assert "page_size" in body

    def test_total_reflects_count_query(self):
        """
        SENTINEL TEST — total in response must equal what count_conversations() returns.
        Short-circuit count → total=0 always → test fails.
        """
        from app.controllers import conversations as conv_ctrl

        client = self._make_app()

        with (
            patch.object(
                conv_ctrl, "list_conversations", new=AsyncMock(return_value=[])
            ),
            patch.object(
                conv_ctrl, "count_conversations", new=AsyncMock(return_value=99)
            ),
        ):
            resp = client.get("/conversations/")

        assert resp.status_code == 200
        assert resp.json()["total"] == 99

    def test_search_filter_passed_to_repository(self):
        """
        SENTINEL TEST — ?search=foo must be forwarded to both list_conversations
        and count_conversations. Remove forwarding → mock not called with search → fails.
        """
        from app.controllers import conversations as conv_ctrl

        client = self._make_app()

        mock_list = AsyncMock(return_value=[])
        mock_count = AsyncMock(return_value=0)

        with (
            patch.object(conv_ctrl, "list_conversations", new=mock_list),
            patch.object(conv_ctrl, "count_conversations", new=mock_count),
        ):
            resp = client.get("/conversations/?search=SENTINEL_QUERY")

        assert resp.status_code == 200
        mock_list.assert_called_once()
        assert mock_list.call_args.kwargs.get("search") == "SENTINEL_QUERY"
        mock_count.assert_called_once()
        assert mock_count.call_args.kwargs.get("search") == "SENTINEL_QUERY"

    def test_page_params_forwarded(self):
        """page and page_size query params are passed through to the repository."""
        from app.controllers import conversations as conv_ctrl

        client = self._make_app()

        mock_list = AsyncMock(return_value=[])
        mock_count = AsyncMock(return_value=0)

        with (
            patch.object(conv_ctrl, "list_conversations", new=mock_list),
            patch.object(conv_ctrl, "count_conversations", new=mock_count),
        ):
            resp = client.get("/conversations/?page=3&page_size=25")

        assert resp.status_code == 200
        body = resp.json()
        assert body["page"] == 3
        assert body["page_size"] == 25
        assert mock_list.call_args.kwargs.get("page") == 3
        assert mock_list.call_args.kwargs.get("page_size") == 25

    def test_concurrent_list_and_count(self):
        """list and count must be called — both are required for the response."""
        from app.controllers import conversations as conv_ctrl

        client = self._make_app()

        mock_list = AsyncMock(return_value=[])
        mock_count = AsyncMock(return_value=7)

        with (
            patch.object(conv_ctrl, "list_conversations", new=mock_list),
            patch.object(conv_ctrl, "count_conversations", new=mock_count),
        ):
            resp = client.get("/conversations/")

        mock_list.assert_called_once()
        mock_count.assert_called_once()
        assert resp.json()["total"] == 7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _conn_returning(rows, count):
    """Return a mock connection that returns `rows` from fetch and `count` from fetchval."""
    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = rows
    mock_conn.fetchval.return_value = count
    return mock_conn
