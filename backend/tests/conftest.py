"""
Shared pytest configuration and fixtures for the CORE backend test suite.

Provides:
- pytest-asyncio auto mode (configured via pyproject.toml)
- Common mock fixtures for repositories
- Warning suppression for known deprecations
"""

import warnings
import pytest
from unittest.mock import AsyncMock, MagicMock

# ---------------------------------------------------------------------------
# Suppress known deprecation warnings from third-party code
# ---------------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message=r".*datetime\.datetime\.utcnow\(\) is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*class-based `config` is deprecated.*",
    category=DeprecationWarning,
)


# ---------------------------------------------------------------------------
# Auth fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def api_key():
    """Test API key for authenticated endpoints."""
    from app.auth import add_api_key
    key = "test-api-key-fixture"
    add_api_key(key)
    yield key


# ---------------------------------------------------------------------------
# Common mock fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_db_pool():
    """Mock asyncpg database pool."""
    pool = AsyncMock()
    pool.acquire.return_value.__aenter__ = AsyncMock()
    pool.acquire.return_value.__aexit__ = AsyncMock()
    return pool


@pytest.fixture
def mock_agent_repo():
    """Mock agent repository with common operations."""
    repo = MagicMock()
    repo.get_instance_by_container_id = AsyncMock(return_value=None)
    repo.update_instance = AsyncMock(return_value=True)
    repo.update_instance_status = AsyncMock(return_value=True)
    repo.update_heartbeat = AsyncMock(return_value=True)
    repo.create_instance = AsyncMock()
    repo.increment_task_completed = AsyncMock(return_value=True)
    repo.increment_task_refused = AsyncMock(return_value=True)
    return repo


@pytest.fixture
def mock_bus_repo():
    """Mock bus repository with common operations."""
    repo = MagicMock()
    repo.save_message = AsyncMock(return_value=True)
    repo.create_delivery_receipt = AsyncMock(return_value=True)
    repo.get_message = AsyncMock(return_value=None)
    repo.get_delivery_receipts = AsyncMock(return_value=[])
    repo.get_correlated_messages = AsyncMock(return_value=[])
    return repo
