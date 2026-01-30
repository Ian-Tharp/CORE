"""
CORE Health Check Aggregation

Provides comprehensive health checks for all CORE services:
- Backend API
- PostgreSQL database
- Redis cache
- Ollama LLM
- WebSocket connections

RSI TODO: Add custom health checks for MCP servers
RSI TODO: Add health history tracking
RSI TODO: Add alerting webhooks for failures
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

from app.dependencies import get_db_pool, get_ollama_client
import logging

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthCheck:
    """Individual health check result."""
    
    def __init__(
        self,
        name: str,
        status: HealthStatus,
        latency_ms: Optional[float] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.status = status
        self.latency_ms = latency_ms
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


async def check_database() -> HealthCheck:
    """Check PostgreSQL database health."""
    start = time.time()
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            # Simple query to verify connection
            result = await conn.fetchval("SELECT 1")
            
            # Get connection pool stats
            pool_size = pool.get_size()
            pool_free = pool.get_idle_size()
        
        latency = (time.time() - start) * 1000
        
        return HealthCheck(
            name="database",
            status=HealthStatus.HEALTHY,
            latency_ms=latency,
            message="PostgreSQL connected",
            details={
                "pool_size": pool_size,
                "pool_free": pool_free,
                "pool_used": pool_size - pool_free
            }
        )
    except Exception as e:
        latency = (time.time() - start) * 1000
        logger.error(f"Database health check failed: {e}")
        return HealthCheck(
            name="database",
            status=HealthStatus.UNHEALTHY,
            latency_ms=latency,
            message=f"Database error: {str(e)}"
        )


async def check_ollama() -> HealthCheck:
    """Check Ollama LLM service health."""
    start = time.time()
    try:
        client = get_ollama_client()
        
        # List available models
        models = await client.models.list()
        model_names = [m.id for m in models.data] if models.data else []
        
        latency = (time.time() - start) * 1000
        
        return HealthCheck(
            name="ollama",
            status=HealthStatus.HEALTHY,
            latency_ms=latency,
            message="Ollama connected",
            details={
                "available_models": model_names[:5],  # First 5 models
                "model_count": len(model_names)
            }
        )
    except Exception as e:
        latency = (time.time() - start) * 1000
        logger.error(f"Ollama health check failed: {e}")
        return HealthCheck(
            name="ollama",
            status=HealthStatus.UNHEALTHY,
            latency_ms=latency,
            message=f"Ollama error: {str(e)}"
        )


async def check_redis() -> HealthCheck:
    """Check Redis cache health."""
    start = time.time()
    try:
        import redis.asyncio as redis
        import os
        
        redis_host = os.getenv("REDIS_HOST", "redis")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        
        client = redis.Redis(host=redis_host, port=redis_port)
        
        # Ping Redis
        await client.ping()
        
        # Get info
        info = await client.info("memory")
        
        await client.close()
        
        latency = (time.time() - start) * 1000
        
        return HealthCheck(
            name="redis",
            status=HealthStatus.HEALTHY,
            latency_ms=latency,
            message="Redis connected",
            details={
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0)
            }
        )
    except ImportError:
        return HealthCheck(
            name="redis",
            status=HealthStatus.UNKNOWN,
            message="Redis client not installed"
        )
    except Exception as e:
        latency = (time.time() - start) * 1000
        logger.error(f"Redis health check failed: {e}")
        return HealthCheck(
            name="redis",
            status=HealthStatus.UNHEALTHY,
            latency_ms=latency,
            message=f"Redis error: {str(e)}"
        )


async def check_websocket_manager() -> HealthCheck:
    """Check WebSocket connection manager health."""
    try:
        from app.websocket_manager import manager
        
        connection_count = len(manager.active_connections)
        channel_count = len(manager.channel_subscribers)
        
        return HealthCheck(
            name="websocket",
            status=HealthStatus.HEALTHY,
            message="WebSocket manager running",
            details={
                "active_connections": connection_count,
                "subscribed_channels": channel_count
            }
        )
    except Exception as e:
        logger.error(f"WebSocket health check failed: {e}")
        return HealthCheck(
            name="websocket",
            status=HealthStatus.UNHEALTHY,
            message=f"WebSocket error: {str(e)}"
        )


async def check_engine_state() -> HealthCheck:
    """Check CORE engine state."""
    try:
        from app.controllers.engine import _active_runs
        
        total_runs = len(_active_runs)
        
        # Count by status
        completed = sum(1 for r in _active_runs.values() if r.is_complete())
        running = total_runs - completed
        
        return HealthCheck(
            name="engine",
            status=HealthStatus.HEALTHY,
            message="CORE engine ready",
            details={
                "active_runs": running,
                "completed_runs": completed,
                "total_in_memory": total_runs
            }
        )
    except Exception as e:
        logger.error(f"Engine health check failed: {e}")
        return HealthCheck(
            name="engine",
            status=HealthStatus.DEGRADED,
            message=f"Engine check error: {str(e)}"
        )


async def get_full_health() -> Dict[str, Any]:
    """
    Run all health checks and return aggregated status.
    
    Returns:
        Dict with overall status and individual check results
    """
    # Run all checks concurrently
    checks = await asyncio.gather(
        check_database(),
        check_ollama(),
        check_redis(),
        check_websocket_manager(),
        check_engine_state(),
        return_exceptions=True
    )
    
    # Process results
    results = []
    for check in checks:
        if isinstance(check, Exception):
            results.append(HealthCheck(
                name="unknown",
                status=HealthStatus.UNHEALTHY,
                message=str(check)
            ))
        else:
            results.append(check)
    
    # Determine overall status
    statuses = [c.status for c in results]
    
    if all(s == HealthStatus.HEALTHY for s in statuses):
        overall = HealthStatus.HEALTHY
    elif any(s == HealthStatus.UNHEALTHY for s in statuses):
        overall = HealthStatus.UNHEALTHY
    else:
        overall = HealthStatus.DEGRADED
    
    return {
        "status": overall.value,
        "timestamp": datetime.utcnow().isoformat(),
        "checks": [c.to_dict() for c in results],
        "summary": {
            "total": len(results),
            "healthy": sum(1 for s in statuses if s == HealthStatus.HEALTHY),
            "degraded": sum(1 for s in statuses if s == HealthStatus.DEGRADED),
            "unhealthy": sum(1 for s in statuses if s == HealthStatus.UNHEALTHY)
        }
    }


async def quick_health() -> Dict[str, str]:
    """
    Quick health check for load balancer probes.
    
    Returns:
        Simple status dict
    """
    try:
        # Just check if the app is responsive
        return {
            "status": "healthy",
            "service": "core-backend",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception:
        return {"status": "unhealthy"}
