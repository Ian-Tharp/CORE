import pytest
import time
from unittest.mock import Mock, patch

from app.services.metrics_service import EmbeddingMetrics


@pytest.fixture
def mock_redis():
    """Mock Redis for testing without requiring real Redis."""
    with patch("app.services.metrics_service.redis") as mock:
        mock_client = Mock()
        mock.from_url.return_value = mock_client
        mock_client.ping.return_value = True
        mock_client.setex = Mock()
        mock_client.pipeline.return_value = mock_client
        mock_client.lpush = Mock()
        mock_client.ltrim = Mock()
        mock_client.expire = Mock()
        mock_client.execute = Mock()
        mock_client.lrange.return_value = []
        yield mock_client


def test_metrics_disabled_without_redis():
    """Test that metrics gracefully handle missing Redis."""
    with patch("app.services.metrics_service._REDIS_AVAILABLE", False):
        metrics = EmbeddingMetrics()
        assert not metrics._enabled

        # Should not raise exceptions
        metrics.record_embedding_latency("batch", "test-model", 5, 100.0, 1000)
        result = metrics.get_recent_metrics()
        assert result == []


def test_record_embedding_latency(mock_redis):
    """Test recording embedding performance metrics."""
    metrics = EmbeddingMetrics()

    metrics.record_embedding_latency(
        operation="batch",
        model="nomic-embed-text",
        text_count=10,
        latency_ms=250.0,
        total_chars=5000,
    )

    # Verify Redis interactions
    mock_redis.setex.assert_called_once()
    mock_redis.pipeline.assert_called_once()


def test_performance_summary_empty():
    """Test performance summary with no data."""
    with patch("app.services.metrics_service._REDIS_AVAILABLE", False):
        metrics = EmbeddingMetrics()
        summary = metrics.get_performance_summary()

        expected = {
            "period_hours": 24,
            "total_operations": 0,
            "avg_latency_ms": 0,
            "total_texts_embedded": 0,
            "operations_by_type": {},
            "models_used": [],
        }
        assert summary == expected


def test_convenience_functions():
    """Test that convenience functions don't raise exceptions."""
    from app.services import metrics_service

    # Should not raise even if Redis unavailable
    metrics_service.record_embedding_performance("test", "model", 1, 100.0)
    result = metrics_service.get_embedding_metrics()
    assert isinstance(result, list)

    summary = metrics_service.get_embedding_summary()
    assert isinstance(summary, dict)


# ===========================================================================
# get_performance_summary — pure aggregation math (sentinels)
# ===========================================================================


def _make_metric(
    operation="single", model="test-model", text_count=1, latency_ms=100.0
):
    """Build a minimal metric dict as returned by get_recent_metrics."""
    return {
        "timestamp": 1700000000,
        "operation": operation,
        "model": model,
        "text_count": text_count,
        "latency_ms": latency_ms,
    }


def test_performance_summary_avg_latency_computed_correctly():
    """
    SENTINEL — avg_latency_ms must be mean of all latencies.
    Use sum instead of mean → wrong result → test fails.
    """
    with patch("app.services.metrics_service._REDIS_AVAILABLE", False):
        metrics = EmbeddingMetrics()
        sample = [_make_metric(latency_ms=100.0), _make_metric(latency_ms=200.0)]
        with patch.object(metrics, "get_recent_metrics", return_value=sample):
            summary = metrics.get_performance_summary()
    assert summary["avg_latency_ms"] == pytest.approx(150.0)


def test_performance_summary_total_texts_summed():
    """
    SENTINEL — total_texts_embedded must be sum of all text_count values.
    Return len(metrics) → text count wrong for batch ops → test fails.
    """
    with patch("app.services.metrics_service._REDIS_AVAILABLE", False):
        metrics = EmbeddingMetrics()
        sample = [
            _make_metric(text_count=10),
            _make_metric(text_count=25),
            _make_metric(text_count=5),
        ]
        with patch.object(metrics, "get_recent_metrics", return_value=sample):
            summary = metrics.get_performance_summary()
    assert summary["total_texts_embedded"] == 40


def test_performance_summary_operations_grouped_by_type():
    """
    SENTINEL — operations_by_type must count how many ops of each type.
    Skip grouping → all ops in one bucket → test fails.
    """
    with patch("app.services.metrics_service._REDIS_AVAILABLE", False):
        metrics = EmbeddingMetrics()
        sample = [
            _make_metric(operation="single"),
            _make_metric(operation="single"),
            _make_metric(operation="batch"),
        ]
        with patch.object(metrics, "get_recent_metrics", return_value=sample):
            summary = metrics.get_performance_summary()
    ops = summary["operations_by_type"]
    assert ops["single"] == 2
    assert ops["batch"] == 1


def test_performance_summary_models_used_deduplicated():
    """
    SENTINEL — models_used must contain unique model names only.
    Return all model strings → duplicates present → test fails.
    """
    with patch("app.services.metrics_service._REDIS_AVAILABLE", False):
        metrics = EmbeddingMetrics()
        sample = [
            _make_metric(model="model-A"),
            _make_metric(model="model-A"),
            _make_metric(model="model-B"),
        ]
        with patch.object(metrics, "get_recent_metrics", return_value=sample):
            summary = metrics.get_performance_summary()
    assert sorted(summary["models_used"]) == ["model-A", "model-B"]


def test_performance_summary_total_operations_is_count():
    """total_operations must equal the number of metrics returned."""
    with patch("app.services.metrics_service._REDIS_AVAILABLE", False):
        metrics = EmbeddingMetrics()
        sample = [_make_metric() for _ in range(7)]
        with patch.object(metrics, "get_recent_metrics", return_value=sample):
            summary = metrics.get_performance_summary()
    assert summary["total_operations"] == 7


def test_performance_summary_last_operation_is_most_recent():
    """last_operation must be the timestamp of the first (newest) entry."""
    with patch("app.services.metrics_service._REDIS_AVAILABLE", False):
        metrics = EmbeddingMetrics()
        sample = [
            {**_make_metric(), "timestamp": 1700001000},
            {**_make_metric(), "timestamp": 1700000000},
        ]
        with patch.object(metrics, "get_recent_metrics", return_value=sample):
            summary = metrics.get_performance_summary()
    assert summary["last_operation"] == 1700001000


# ===========================================================================
# InstanceManager._calculate_cpu_percent — pure arithmetic
# ===========================================================================


def test_calculate_cpu_percent_valid_stats():
    """
    SENTINEL — valid CPU stats must produce a non-zero percentage.
    Return None always → CPU graphs always 0 → test fails.
    """
    from app.services.instance_manager import InstanceManager

    mgr = object.__new__(InstanceManager)

    cpu_stats = {
        "cpu_usage": {
            "total_usage": 200_000_000,  # 0.2s in nanoseconds
            "percpu_usage": [100_000_000, 100_000_000],  # 2 CPUs
        },
        "system_cpu_usage": 10_000_000_000,  # 10s system time
        "precpu_stats": {
            "cpu_usage": {"total_usage": 100_000_000},
            "system_cpu_usage": 5_000_000_000,
        },
    }
    result = mgr._calculate_cpu_percent(cpu_stats)
    assert result is not None
    assert result > 0


def test_calculate_cpu_percent_returns_none_on_zero_system_delta():
    """
    SENTINEL — zero system_delta must return None (not divide by zero).
    Remove ZeroDivisionError guard → ZeroDivisionError raised → test fails.
    """
    from app.services.instance_manager import InstanceManager

    mgr = object.__new__(InstanceManager)

    cpu_stats = {
        "cpu_usage": {"total_usage": 200, "percpu_usage": [100, 100]},
        "system_cpu_usage": 1000,
        "precpu_stats": {
            "cpu_usage": {"total_usage": 100},
            "system_cpu_usage": 1000,  # same → delta = 0
        },
    }
    result = mgr._calculate_cpu_percent(cpu_stats)
    assert result is None


def test_calculate_cpu_percent_returns_none_on_missing_keys():
    """Missing keys must return None (not raise KeyError)."""
    from app.services.instance_manager import InstanceManager

    mgr = object.__new__(InstanceManager)
    result = mgr._calculate_cpu_percent({})
    assert result is None
