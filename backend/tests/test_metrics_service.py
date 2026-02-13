import pytest
import time
from unittest.mock import Mock, patch

from app.services.metrics_service import EmbeddingMetrics


@pytest.fixture
def mock_redis():
    """Mock Redis for testing without requiring real Redis."""
    with patch('app.services.metrics_service.redis') as mock:
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
    with patch('app.services.metrics_service._REDIS_AVAILABLE', False):
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
        total_chars=5000
    )
    
    # Verify Redis interactions
    mock_redis.setex.assert_called_once()
    mock_redis.pipeline.assert_called_once()
    

def test_performance_summary_empty():
    """Test performance summary with no data."""
    with patch('app.services.metrics_service._REDIS_AVAILABLE', False):
        metrics = EmbeddingMetrics()
        summary = metrics.get_performance_summary()
        
        expected = {
            "period_hours": 24,
            "total_operations": 0,
            "avg_latency_ms": 0,
            "total_texts_embedded": 0,
            "operations_by_type": {},
            "models_used": []
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