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
- Inter-Agent Communication Bus (queue depths)
- Task Routing Engine (queue depth, success rates)
- System resources (memory, CPU)
"""

import time
import logging
from typing import Dict, Any
from datetime import datetime, timezone
from typing import Optional as OptionalType

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.security import require_role, Role
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
async def deep_health_check(
    api_key: dict = Depends(require_role(Role.AGENT)),
) -> Dict[str, Any]:
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
        - bus: Inter-agent message queue depths, subscriptions, agents
        - task_queue: Task routing queue depth, success rates
        - system: Host memory, CPU, process resource usage

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
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=result
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
                "details": db_status.message,
            },
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
            "formatted": get_uptime_formatted(),
        },
        "memory": {
            "rss_mb": round(memory.rss / 1024 / 1024, 2),
            "vms_mb": round(memory.vms / 1024 / 1024, 2),
            "percent": round(process.memory_percent(), 2),
        },
        "cpu": {"percent": process.cpu_percent(), "num_threads": process.num_threads()},
        "connections": len(process.connections()),
    }


@router.get("/history", status_code=status.HTTP_200_OK)
async def health_history(
    limit: int = Query(50, ge=1, le=500, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    status_filter: OptionalType[str] = Query(
        None, alias="status", description="Filter by status"
    ),
    since: OptionalType[str] = Query(None, description="ISO timestamp lower bound"),
    until: OptionalType[str] = Query(None, description="ISO timestamp upper bound"),
) -> Dict[str, Any]:
    """
    Health check history.

    Returns stored health snapshots for trend analysis and incident review.
    Each deep health check is automatically recorded.

    Query parameters:
        - limit: Max results (1-500, default 50)
        - offset: Pagination offset
        - status: Filter by overall status (healthy/degraded/unhealthy)
        - since: ISO timestamp lower bound
        - until: ISO timestamp upper bound
    """
    from app.repository import health_repository

    since_dt = datetime.fromisoformat(since) if since else None
    until_dt = datetime.fromisoformat(until) if until else None

    snapshots = await health_repository.get_history(
        limit=limit,
        offset=offset,
        status_filter=status_filter,
        since=since_dt,
        until=until_dt,
    )

    return {
        "count": len(snapshots),
        "limit": limit,
        "offset": offset,
        "snapshots": snapshots,
    }


@router.get("/history/summary", status_code=status.HTTP_200_OK)
async def health_summary(
    hours: int = Query(24, ge=1, le=720, description="Lookback window in hours"),
) -> Dict[str, Any]:
    """
    Aggregated health summary over a time window.

    Returns uptime percentage, check counts by status, and average latency.
    """
    from app.repository import health_repository

    return await health_repository.get_status_summary(hours=hours)


@router.get("/history/{snapshot_id}", status_code=status.HTTP_200_OK)
async def health_snapshot_detail(snapshot_id: str) -> Dict[str, Any]:
    """
    Get a single health snapshot by ID.
    """
    from app.repository import health_repository

    snapshot = await health_repository.get_snapshot(snapshot_id)
    if not snapshot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Snapshot {snapshot_id} not found",
        )
    return snapshot


@router.delete("/history/prune", status_code=status.HTTP_200_OK)
async def prune_health_history(
    keep_days: int = Query(30, ge=1, le=365, description="Days of history to keep"),
) -> Dict[str, Any]:
    """
    Prune old health snapshots.

    Deletes snapshots older than keep_days. Useful for maintenance.
    """
    from app.repository import health_repository

    deleted = await health_repository.prune_old_snapshots(keep_days=keep_days)
    return {"deleted": deleted, "keep_days": keep_days}
