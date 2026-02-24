"""
Council Repository Tests

Tests for the council_repository module, focusing on count_sessions.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.council_models import SessionStatus
from app.repository.council_repository import count_sessions


@pytest.fixture
def mock_db():
    """Create a mock database pool with async context manager for acquire."""
    conn = AsyncMock()
    mock_acquire_ctx = MagicMock()
    mock_acquire_ctx.__aenter__ = AsyncMock(return_value=conn)
    mock_acquire_ctx.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.acquire.return_value = mock_acquire_ctx
    return pool, conn


class TestCountSessions:
    """Tests for the count_sessions function."""

    @pytest.mark.asyncio
    @patch("app.repository.council_repository.get_db_pool", new_callable=AsyncMock)
    async def test_count_all_sessions(self, mock_get_pool, mock_db):
        pool, conn = mock_db
        mock_get_pool.return_value = pool
        conn.fetchval.return_value = 42

        result = await count_sessions()

        assert result == 42
        conn.fetchval.assert_called_once()
        query = conn.fetchval.call_args[0][0]
        assert "COUNT(*)" in query
        assert "WHERE" not in query

    @pytest.mark.asyncio
    @patch("app.repository.council_repository.get_db_pool", new_callable=AsyncMock)
    async def test_count_sessions_with_status_filter(self, mock_get_pool, mock_db):
        pool, conn = mock_db
        mock_get_pool.return_value = pool
        conn.fetchval.return_value = 7

        result = await count_sessions(status=SessionStatus.DELIBERATING)

        assert result == 7
        conn.fetchval.assert_called_once()
        query = conn.fetchval.call_args[0][0]
        assert "WHERE" in query
        assert conn.fetchval.call_args[0][1] == "deliberating"

    @pytest.mark.asyncio
    @patch("app.repository.council_repository.get_db_pool", new_callable=AsyncMock)
    async def test_count_sessions_returns_zero(self, mock_get_pool, mock_db):
        pool, conn = mock_db
        mock_get_pool.return_value = pool
        conn.fetchval.return_value = 0

        result = await count_sessions()

        assert result == 0
