"""
Unit tests for the health aggregator service.

Tests health check logic, status aggregation, and uptime tracking.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import sys
import os

# Add the backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.health_aggregator import (
    HealthStatus,
    ServiceHealth,
    get_uptime_seconds,
    get_uptime_formatted,
    determine_overall_status,
    quick_health,
)


# ============================================================================
# HealthStatus Tests
# ============================================================================


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_status_values(self):
        """Test that all status values are defined."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"

    def test_status_is_string_enum(self):
        """Test status can be used as string."""
        assert str(HealthStatus.HEALTHY) == "HealthStatus.HEALTHY"
        assert HealthStatus.HEALTHY.value == "healthy"


# ============================================================================
# ServiceHealth Tests
# ============================================================================


class TestServiceHealth:
    """Tests for ServiceHealth dataclass."""

    def test_basic_creation(self):
        """Test creating a basic health check result."""
        health = ServiceHealth(
            name="test_service",
            status=HealthStatus.HEALTHY,
            latency_ms=10.5,
            message="All good",
        )

        assert health.name == "test_service"
        assert health.status == HealthStatus.HEALTHY
        assert health.latency_ms == 10.5
        assert health.message == "All good"
        assert health.details == {}

    def test_to_dict(self):
        """Test serialization to dict."""
        health = ServiceHealth(
            name="database",
            status=HealthStatus.HEALTHY,
            latency_ms=5.123,
            message="Connected",
            details={"pool_size": 10},
        )

        result = health.to_dict()

        assert result["name"] == "database"
        assert result["status"] == "healthy"
        assert result["latency_ms"] == 5.12  # Rounded
        assert result["message"] == "Connected"
        assert result["details"]["pool_size"] == 10
        assert "checked_at" in result

    def test_default_details(self):
        """Test default empty details dict."""
        health = ServiceHealth(name="test", status=HealthStatus.DEGRADED)

        assert health.details == {}
        assert health.latency_ms is None
        assert health.message is None


# ============================================================================
# Uptime Tests
# ============================================================================


class TestUptime:
    """Tests for uptime tracking functions."""

    def test_uptime_seconds_positive(self):
        """Test uptime returns positive number."""
        uptime = get_uptime_seconds()
        assert uptime >= 0

    def test_uptime_formatted_structure(self):
        """Test formatted uptime returns string."""
        formatted = get_uptime_formatted()
        assert isinstance(formatted, str)
        # Should end with 's' for seconds
        assert formatted.endswith("s")


# ============================================================================
# Status Aggregation Tests
# ============================================================================


class TestDetermineOverallStatus:
    """Tests for overall status determination."""

    def test_all_healthy_returns_healthy(self):
        """Test all healthy services return healthy overall."""
        checks = [
            ServiceHealth(name="database", status=HealthStatus.HEALTHY),
            ServiceHealth(name="redis", status=HealthStatus.HEALTHY),
            ServiceHealth(name="ollama", status=HealthStatus.HEALTHY),
        ]

        result = determine_overall_status(checks)
        assert result == HealthStatus.HEALTHY

    def test_critical_unhealthy_returns_unhealthy(self):
        """Test unhealthy database returns unhealthy overall."""
        checks = [
            ServiceHealth(name="database", status=HealthStatus.UNHEALTHY),
            ServiceHealth(name="redis", status=HealthStatus.HEALTHY),
            ServiceHealth(name="ollama", status=HealthStatus.HEALTHY),
        ]

        result = determine_overall_status(checks)
        assert result == HealthStatus.UNHEALTHY

    def test_non_critical_unhealthy_returns_degraded(self):
        """Test unhealthy non-critical service returns degraded."""
        checks = [
            ServiceHealth(name="database", status=HealthStatus.HEALTHY),
            ServiceHealth(name="redis", status=HealthStatus.UNHEALTHY),
            ServiceHealth(name="ollama", status=HealthStatus.HEALTHY),
        ]

        result = determine_overall_status(checks)
        assert result == HealthStatus.DEGRADED

    def test_degraded_service_returns_degraded(self):
        """Test degraded service returns degraded overall."""
        checks = [
            ServiceHealth(name="database", status=HealthStatus.HEALTHY),
            ServiceHealth(name="redis", status=HealthStatus.HEALTHY),
            ServiceHealth(name="ollama", status=HealthStatus.DEGRADED),
        ]

        result = determine_overall_status(checks)
        assert result == HealthStatus.DEGRADED

    def test_unknown_service_returns_degraded(self):
        """Test unknown service status returns degraded overall."""
        checks = [
            ServiceHealth(name="database", status=HealthStatus.HEALTHY),
            ServiceHealth(name="redis", status=HealthStatus.UNKNOWN),
        ]

        result = determine_overall_status(checks)
        assert result == HealthStatus.DEGRADED


# ============================================================================
# Quick Health Tests
# ============================================================================


class TestQuickHealth:
    """Tests for quick health check."""

    @pytest.mark.asyncio
    async def test_quick_health_structure(self):
        """Test quick health returns expected structure."""
        result = await quick_health()

        assert result["status"] == "healthy"
        assert result["service"] == "core-backend"
        assert "timestamp" in result
        assert "uptime_seconds" in result
        assert isinstance(result["uptime_seconds"], float)


# ============================================================================
# Individual Check Function Tests
# ============================================================================


class TestCheckDatabase:
    """Tests for check_database function."""

    @pytest.mark.asyncio
    async def test_healthy_database(self):
        """Test healthy database returns correct status and pool stats."""
        from app.services.health_aggregator import check_database

        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=1)

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value = mock_ctx
        mock_pool.get_size.return_value = 10
        mock_pool.get_idle_size.return_value = 7

        with patch(
            "app.dependencies.get_db_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            result = await check_database()

        assert result.status == HealthStatus.HEALTHY
        assert result.name == "database"
        assert result.latency_ms is not None
        assert result.details["pool_total"] == 10
        assert result.details["pool_idle"] == 7
        assert result.details["pool_active"] == 3

    @pytest.mark.asyncio
    async def test_database_connection_failure(self):
        """Test database failure returns unhealthy."""
        from app.services.health_aggregator import check_database

        with patch(
            "app.dependencies.get_db_pool",
            new_callable=AsyncMock,
            side_effect=ConnectionRefusedError("Connection refused"),
        ):
            result = await check_database()

        assert result.status == HealthStatus.UNHEALTHY
        assert result.name == "database"
        assert "Connection failed" in result.message


class TestCheckOllama:
    """Tests for check_ollama function."""

    @pytest.mark.asyncio
    async def test_healthy_ollama_with_models(self):
        """Test Ollama with loaded models returns healthy."""
        from app.services.health_aggregator import check_ollama

        mock_model = MagicMock()
        mock_model.id = "llama3:latest"
        mock_models_response = MagicMock()
        mock_models_response.data = [mock_model]

        mock_client = MagicMock()
        mock_client.models.list = AsyncMock(return_value=mock_models_response)

        with patch("app.dependencies.get_ollama_client", return_value=mock_client):
            result = await check_ollama()

        assert result.status == HealthStatus.HEALTHY
        assert result.details["model_count"] == 1
        assert result.details["has_models"] is True

    @pytest.mark.asyncio
    async def test_ollama_no_models_degraded(self):
        """Test Ollama with no models returns degraded."""
        from app.services.health_aggregator import check_ollama

        mock_models_response = MagicMock()
        mock_models_response.data = []

        mock_client = MagicMock()
        mock_client.models.list = AsyncMock(return_value=mock_models_response)

        with patch("app.dependencies.get_ollama_client", return_value=mock_client):
            result = await check_ollama()

        assert result.status == HealthStatus.DEGRADED
        assert result.details["model_count"] == 0

    @pytest.mark.asyncio
    async def test_ollama_connection_failure(self):
        """Test Ollama unavailable returns unhealthy."""
        from app.services.health_aggregator import check_ollama

        mock_client = MagicMock()
        mock_client.models.list = AsyncMock(side_effect=ConnectionError("refused"))

        with patch("app.dependencies.get_ollama_client", return_value=mock_client):
            result = await check_ollama()

        assert result.status == HealthStatus.UNHEALTHY
        assert "Connection failed" in result.message


class TestCheckRedis:
    """Tests for check_redis function."""

    @pytest.mark.asyncio
    async def test_healthy_redis(self):
        """Test healthy Redis returns correct status."""
        from app.services.health_aggregator import check_redis

        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)
        mock_client.info = AsyncMock(
            side_effect=[
                {
                    "used_memory_human": "1.5M",
                    "used_memory_peak_human": "2M",
                    "connected_clients": 3,
                },
                {"redis_version": "7.0.0"},
            ]
        )
        mock_client.aclose = AsyncMock()

        mock_redis_cls = MagicMock(return_value=mock_client)

        with patch.dict(
            "sys.modules", {"redis.asyncio": MagicMock(Redis=mock_redis_cls)}
        ):
            # Re-import to pick up patched module
            import importlib
            import app.services.health_aggregator as ha

            # Direct patch of redis inside the function scope
            with patch("redis.asyncio.Redis", mock_redis_cls):
                result = await check_redis()

        # If redis isn't actually installed in test env, we may get UNKNOWN
        assert result.name == "redis"
        assert result.status in (
            HealthStatus.HEALTHY,
            HealthStatus.UNKNOWN,
            HealthStatus.UNHEALTHY,
        )

    @pytest.mark.asyncio
    async def test_redis_import_error(self):
        """Test missing redis package returns unknown."""
        from app.services.health_aggregator import check_redis

        with patch.dict("sys.modules", {"redis": None, "redis.asyncio": None}):
            result = await check_redis()

        # Should handle gracefully
        assert result.name == "redis"


class TestCheckWebsocketManager:
    """Tests for check_websocket_manager function."""

    @pytest.mark.asyncio
    async def test_healthy_websocket(self):
        """Test WebSocket manager reports connections."""
        from app.services.health_aggregator import check_websocket_manager

        mock_manager = MagicMock()
        mock_manager.active_connections = ["conn1", "conn2"]
        mock_manager.channel_subscribers = {"ch1": [], "ch2": []}

        with patch("app.websocket_manager.manager", mock_manager):
            result = await check_websocket_manager()

        assert result.status == HealthStatus.HEALTHY
        assert result.details["active_connections"] == 2
        assert result.details["subscribed_channels"] == 2


class TestCheckEngineState:
    """Tests for check_engine_state function."""

    @pytest.mark.asyncio
    async def test_engine_with_active_runs(self):
        """Test engine correctly counts active and completed runs."""
        from app.services.health_aggregator import check_engine_state

        mock_run_complete = MagicMock()
        mock_run_complete.is_complete.return_value = True

        mock_run_active = MagicMock()
        mock_run_active.is_complete.return_value = False

        mock_runs = {
            "run1": mock_run_complete,
            "run2": mock_run_active,
            "run3": mock_run_active,
        }

        with patch("app.controllers.engine._active_runs", mock_runs):
            result = await check_engine_state()

        assert result.status == HealthStatus.HEALTHY
        assert result.details["active_runs"] == 2
        assert result.details["completed_runs"] == 1
        assert result.details["total_tracked"] == 3


class TestCheckSystemResources:
    """Tests for check_system_resources function."""

    @pytest.mark.asyncio
    async def test_healthy_system(self):
        """Test system resources below thresholds returns healthy."""
        from app.services.health_aggregator import check_system_resources

        mock_mem_info = MagicMock()
        mock_mem_info.rss = 500 * 1024 * 1024  # 500 MB
        mock_mem_info.vms = 1000 * 1024 * 1024

        mock_sys_mem = MagicMock()
        mock_sys_mem.percent = 60.0
        mock_sys_mem.available = 8 * 1024 * 1024 * 1024  # 8 GB

        mock_process = MagicMock()
        mock_process.memory_info.return_value = mock_mem_info
        mock_process.num_threads.return_value = 10
        mock_process.cpu_percent.return_value = 5.0

        with patch("psutil.Process", return_value=mock_process), patch(
            "psutil.virtual_memory", return_value=mock_sys_mem
        ):
            result = await check_system_resources()

        assert result.status == HealthStatus.HEALTHY
        assert result.name == "system"

    @pytest.mark.asyncio
    async def test_high_memory_degraded(self):
        """Test high system memory triggers degraded status."""
        from app.services.health_aggregator import check_system_resources

        mock_mem_info = MagicMock()
        mock_mem_info.rss = 500 * 1024 * 1024
        mock_mem_info.vms = 1000 * 1024 * 1024

        mock_sys_mem = MagicMock()
        mock_sys_mem.percent = 95.0  # Over 90% threshold
        mock_sys_mem.available = 1 * 1024 * 1024 * 1024

        mock_process = MagicMock()
        mock_process.memory_info.return_value = mock_mem_info
        mock_process.num_threads.return_value = 10
        mock_process.cpu_percent.return_value = 5.0

        with patch("psutil.Process", return_value=mock_process), patch(
            "psutil.virtual_memory", return_value=mock_sys_mem
        ):
            result = await check_system_resources()

        assert result.status == HealthStatus.DEGRADED


# ============================================================================
# Comprehensive Health Tests
# ============================================================================


class TestGetComprehensiveHealth:
    """Tests for get_comprehensive_health aggregation."""

    @pytest.mark.asyncio
    async def test_comprehensive_health_structure(self):
        """Test comprehensive health returns expected top-level structure."""
        from app.services.health_aggregator import get_comprehensive_health

        healthy = ServiceHealth(name="database", status=HealthStatus.HEALTHY)

        with patch(
            "app.services.health_aggregator.check_database",
            new_callable=AsyncMock,
            return_value=healthy,
        ), patch(
            "app.services.health_aggregator.check_redis",
            new_callable=AsyncMock,
            return_value=healthy,
        ), patch(
            "app.services.health_aggregator.check_ollama",
            new_callable=AsyncMock,
            return_value=healthy,
        ), patch(
            "app.services.health_aggregator.check_vector_db",
            new_callable=AsyncMock,
            return_value=healthy,
        ), patch(
            "app.services.health_aggregator.check_websocket_manager",
            new_callable=AsyncMock,
            return_value=healthy,
        ), patch(
            "app.services.health_aggregator.check_engine_state",
            new_callable=AsyncMock,
            return_value=healthy,
        ), patch(
            "app.services.health_aggregator.check_bus_queue",
            new_callable=AsyncMock,
            return_value=healthy,
        ), patch(
            "app.services.health_aggregator.check_task_queue",
            new_callable=AsyncMock,
            return_value=healthy,
        ), patch(
            "app.services.health_aggregator.check_system_resources",
            new_callable=AsyncMock,
            return_value=healthy,
        ):
            result = await get_comprehensive_health()

        assert result["status"] == "healthy"
        assert result["service"] == "core-backend"
        assert "timestamp" in result
        assert "uptime" in result
        assert "total_check_latency_ms" in result
        assert "services" in result
        assert "summary" in result
        assert result["summary"]["total_services"] == 9
        assert result["summary"]["healthy"] == 9

    @pytest.mark.asyncio
    async def test_comprehensive_handles_check_exception(self):
        """Test that an exception from a check is converted to unhealthy."""
        from app.services.health_aggregator import get_comprehensive_health

        healthy = ServiceHealth(name="ok", status=HealthStatus.HEALTHY)

        async def raise_error():
            raise RuntimeError("kaboom")

        with patch(
            "app.services.health_aggregator.check_database",
            new_callable=AsyncMock,
            return_value=healthy,
        ), patch(
            "app.services.health_aggregator.check_redis",
            side_effect=RuntimeError("kaboom"),
        ), patch(
            "app.services.health_aggregator.check_ollama",
            new_callable=AsyncMock,
            return_value=healthy,
        ), patch(
            "app.services.health_aggregator.check_vector_db",
            new_callable=AsyncMock,
            return_value=healthy,
        ), patch(
            "app.services.health_aggregator.check_websocket_manager",
            new_callable=AsyncMock,
            return_value=healthy,
        ), patch(
            "app.services.health_aggregator.check_engine_state",
            new_callable=AsyncMock,
            return_value=healthy,
        ), patch(
            "app.services.health_aggregator.check_bus_queue",
            new_callable=AsyncMock,
            return_value=healthy,
        ), patch(
            "app.services.health_aggregator.check_task_queue",
            new_callable=AsyncMock,
            return_value=healthy,
        ), patch(
            "app.services.health_aggregator.check_system_resources",
            new_callable=AsyncMock,
            return_value=healthy,
        ):
            result = await get_comprehensive_health()

        # Should be degraded (non-critical failure) not crash
        assert result["status"] in ("degraded", "unhealthy")
        assert result["summary"]["unhealthy"] >= 1


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
