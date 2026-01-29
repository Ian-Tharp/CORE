"""
Health Check Controller

Provides comprehensive health check endpoints for monitoring.
Aggregates status from all CORE components.

Endpoints:
- GET /health - Quick health check
- GET /health/deep - Deep health check with all service status
- GET /health/ready - Readiness probe for k8s
- GET /health/live - Liveness probe for k8s
- GET /health/metrics - Basic metrics

Services Monitored:
- PostgreSQL database (connection pool)
- Redis cache
- Ollama LLM service
- Vector DB (pgvector)
- WebSocket manager
- CORE engine
"""

import time
import logging
from typing import Dict, Any
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status

from app.services.health_aggregator import (
    get_comprehensive_health,
    quick_health,
    check_database,
    get_uptime_seconds,
    get_uptime_formatted,
    HealthStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, Any]:
    """
    Quick health check endpoint.
    
    Returns 200 OK if the service is running.
    Used for basic load balancer health checks.
    
    Response:
        - status: "healthy"
        - service: "core-backend"
        - timestamp: ISO timestamp
        - uptime_seconds: float
    """
    return await quick_health()


@router.get("/deep", status_code=status.HTTP_200_OK)
async def deep_health_check() -> Dict[str, Any]:
    """
    Deep health check with comprehensive service status.
    
    Checks all dependent services and returns aggregate status.
    Returns 200 if system is healthy/degraded, 503 if unhealthy.
    
    Services Checked:
        - database: PostgreSQL with pool stats
        - redis: Cache connectivity and memory
        - ollama: LLM service and available models
        - vector_db: pgvector extension status
        - websocket: Connection manager state
        - engine: CORE engine runs
    
    Response:
        - status: "healthy" | "degraded" | "unhealthy"
        - service: "core-backend"
        - timestamp: ISO timestamp
        - uptime: {seconds, formatted}
        - total_check_latency_ms: float
        - services: {name: {status, latency_ms, message, details}}
        - summary: {total_services, healthy, degraded, unhealthy, unknown}
    """
    result = await get_comprehensive_health()
    
    # Return 503 if unhealthy
    if result["status"] == HealthStatus.UNHEALTHY.value:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=result
        )
    
    return result


@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_probe() -> Dict[str, str]:
    """
    Kubernetes readiness probe.
    
    Returns 200 if service is ready to receive traffic.
    Checks critical dependency (database) before accepting traffic.
    """
    db_status = await check_database()
    
    if db_status.status != HealthStatus.HEALTHY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "reason": "database unavailable",
                "details": db_status.message
            }
        )
    
    return {"status": "ready"}


@router.get("/live", status_code=status.HTTP_200_OK)
async def liveness_probe() -> Dict[str, str]:
    """
    Kubernetes liveness probe.
    
    Returns 200 if service is alive.
    Should only fail if service needs restart (memory leak, deadlock, etc).
    """
    return {"status": "alive"}


@router.get("/metrics", status_code=status.HTTP_200_OK)
async def metrics() -> Dict[str, Any]:
    """
    Basic metrics endpoint.
    
    Returns operational metrics for monitoring.
    For Prometheus integration, consider prometheus-fastapi-instrumentator.
    
    Response:
        - timestamp: ISO timestamp
        - uptime: {seconds, formatted}
        - memory: {rss_mb, vms_mb, percent}
        - cpu: {percent, num_threads}
        - connections: open file descriptors/connections
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory = process.memory_info()
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime": {
            "seconds": round(get_uptime_seconds(), 2),
            "formatted": get_uptime_formatted()
        },
        "memory": {
            "rss_mb": round(memory.rss / 1024 / 1024, 2),
            "vms_mb": round(memory.vms / 1024 / 1024, 2),
            "percent": round(process.memory_percent(), 2)
        },
        "cpu": {
            "percent": process.cpu_percent(),
            "num_threads": process.num_threads()
        },
        "connections": len(process.connections())
    }
