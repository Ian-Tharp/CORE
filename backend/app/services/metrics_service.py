from __future__ import annotations

import json
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

try:
    import redis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


class EmbeddingMetrics:
    """Tracks and stores embedding performance metrics."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self._redis = None
        self._enabled = _REDIS_AVAILABLE
        if not _REDIS_AVAILABLE:
            logger.warning("Redis not available - embedding metrics disabled")
    
    def _get_redis(self):
        """Lazy Redis connection."""
        if not self._enabled:
            return None
        if self._redis is None:
            try:
                self._redis = redis.from_url(self.redis_url, decode_responses=True)
                # Test connection
                self._redis.ping()
            except Exception as e:
                logger.error("Failed to connect to Redis: %s", e)
                self._enabled = False
                return None
        return self._redis
    
    def record_embedding_latency(
        self, 
        operation: str,  # "single", "batch", "document"
        model: str,
        text_count: int,
        latency_ms: float,
        total_chars: Optional[int] = None
    ):
        """Record embedding performance data."""
        redis_client = self._get_redis()
        if not redis_client:
            return
        
        timestamp = int(time.time())
        metric_data = {
            "timestamp": timestamp,
            "operation": operation,
            "model": model,
            "text_count": text_count,
            "latency_ms": latency_ms,
            "chars_per_sec": total_chars / (latency_ms / 1000) if total_chars and latency_ms > 0 else None,
            "texts_per_sec": text_count / (latency_ms / 1000) if latency_ms > 0 else None
        }
        
        try:
            # Store in time-series with TTL (30 days)
            key = f"embedding_metrics:{timestamp}"
            redis_client.setex(key, 30 * 24 * 3600, json.dumps(metric_data))
            
            # Also store in rolling aggregates
            hour_key = f"embedding_hourly:{timestamp // 3600}"
            pipeline = redis_client.pipeline()
            pipeline.lpush(hour_key, json.dumps(metric_data))
            pipeline.ltrim(hour_key, 0, 999)  # Keep last 1000 entries per hour
            pipeline.expire(hour_key, 7 * 24 * 3600)  # 7 days TTL
            pipeline.execute()
            
        except Exception as e:
            logger.error("Failed to record embedding metric: %s", e)
    
    def get_recent_metrics(self, hours: int = 24, operation: Optional[str] = None) -> List[Dict]:
        """Get recent embedding metrics."""
        redis_client = self._get_redis()
        if not redis_client:
            return []
        
        try:
            current_hour = int(time.time()) // 3600
            metrics = []
            
            for h in range(hours):
                hour_key = f"embedding_hourly:{current_hour - h}"
                raw_data = redis_client.lrange(hour_key, 0, -1)
                
                for entry in raw_data:
                    try:
                        data = json.loads(entry)
                        if operation is None or data.get("operation") == operation:
                            metrics.append(data)
                    except Exception:
                        continue
            
            # Sort by timestamp descending
            metrics.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            return metrics
        
        except Exception as e:
            logger.error("Failed to retrieve metrics: %s", e)
            return []
    
    def get_performance_summary(self, hours: int = 24) -> Dict:
        """Get aggregate performance statistics."""
        metrics = self.get_recent_metrics(hours)
        if not metrics:
            return {
                "period_hours": hours,
                "total_operations": 0,
                "avg_latency_ms": 0,
                "total_texts_embedded": 0,
                "operations_by_type": {},
                "models_used": []
            }
        
        total_latency = sum(m.get("latency_ms", 0) for m in metrics)
        total_texts = sum(m.get("text_count", 0) for m in metrics)
        operations_by_type = {}
        models = set()
        
        for m in metrics:
            op_type = m.get("operation", "unknown")
            operations_by_type[op_type] = operations_by_type.get(op_type, 0) + 1
            if m.get("model"):
                models.add(m["model"])
        
        return {
            "period_hours": hours,
            "total_operations": len(metrics),
            "avg_latency_ms": total_latency / len(metrics) if metrics else 0,
            "total_texts_embedded": total_texts,
            "operations_by_type": operations_by_type,
            "models_used": sorted(list(models)),
            "last_operation": metrics[0].get("timestamp") if metrics else None
        }


# Global metrics instance
_metrics = EmbeddingMetrics()


def record_embedding_performance(operation: str, model: str, text_count: int, latency_ms: float, total_chars: Optional[int] = None):
    """Convenience function to record embedding performance."""
    _metrics.record_embedding_latency(operation, model, text_count, latency_ms, total_chars)


def get_embedding_metrics(hours: int = 24, operation: Optional[str] = None) -> List[Dict]:
    """Convenience function to get recent metrics."""
    return _metrics.get_recent_metrics(hours, operation)


def get_embedding_summary(hours: int = 24) -> Dict:
    """Convenience function to get performance summary."""
    return _metrics.get_performance_summary(hours)