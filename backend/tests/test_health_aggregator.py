"""
Unit tests for the health aggregator service.

Tests health check logic, status aggregation, and uptime tracking.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import sys
import os

# Add the backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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
            message="All good"
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
            details={"pool_size": 10}
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
        health = ServiceHealth(
            name="test",
            status=HealthStatus.DEGRADED
        )
        
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
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
