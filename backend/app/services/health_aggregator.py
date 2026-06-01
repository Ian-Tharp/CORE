"""
Health Aggregator Service

Centralized health monitoring for all CORE services.
Provides unified health checks with response timing, uptime tracking,
and status aggregation.

Services monitored:
- PostgreSQL database (with pool stats)
- Redis cache
- Ollama LLM service
- Vector DB (pgvector extension)
- WebSocket manager
- CORE engine state
- Inter-Agent Communication Bus (message queue depths, subscriptions)
- Task Routing Engine (queue depth, success rates)
- System resources (memory, CPU)
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# Track service start time for uptime calculation
_SERVICE_START_TIME: float = time.time()


class HealthStatus(str, Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceHealth:
    """Individual service health check result."""

    name: str
    status: HealthStatus
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "latency_ms": round(self.latency_ms, 2) if self.latency_ms else None,
            "message": self.message,
            "details": self.details,
            "checked_at": self.checked_at.isoformat(),
        }


def get_uptime_seconds() -> float:
    """Get service uptime in seconds."""
    return time.time() - _SERVICE_START_TIME


def get_uptime_formatted() -> str:
    """Get formatted uptime string."""
    seconds = get_uptime_seconds()
    days, remainder = divmod(int(seconds), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)

    if days > 0:
        return f"{days}d {hours}h {minutes}m {secs}s"
    elif hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


async def check_database() -> ServiceHealth:
    """Check PostgreSQL database health with pool statistics."""
    from app.dependencies import get_db_pool

    start = time.time()
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")

            # Get pool stats
            pool_size = pool.get_size()
            pool_free = pool.get_idle_size()
            pool_used = pool_size - pool_free

        latency = (time.time() - start) * 1000

        return ServiceHealth(
            name="database",
            status=HealthStatus.HEALTHY,
            latency_ms=latency,
            message="PostgreSQL connected",
            details={
                "pool_total": pool_size,
                "pool_idle": pool_free,
                "pool_active": pool_used,
                "pool_utilization": (
                    f"{(pool_used / pool_size * 100):.1f}%" if pool_size > 0 else "0%"
                ),
            },
        )
    except Exception as e:
        latency = (time.time() - start) * 1000
        logger.error(f"Database health check failed: {e}")
        return ServiceHealth(
            name="database",
            status=HealthStatus.UNHEALTHY,
            latency_ms=latency,
            message=f"Connection failed: {str(e)[:100]}",
        )


async def check_redis() -> ServiceHealth:
    """Check Redis cache health."""
    start = time.time()
    try:
        import redis.asyncio as redis

        redis_host = os.getenv("REDIS_HOST", "redis")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))

        client = redis.Redis(host=redis_host, port=redis_port, socket_timeout=5)

        # Ping and get basic info
        await client.ping()
        info = await client.info("memory")
        server_info = await client.info("server")

        await client.aclose()

        latency = (time.time() - start) * 1000

        return ServiceHealth(
            name="redis",
            status=HealthStatus.HEALTHY,
            latency_ms=latency,
            message="Redis connected",
            details={
                "used_memory": info.get("used_memory_human", "unknown"),
                "peak_memory": info.get("used_memory_peak_human", "unknown"),
                "redis_version": server_info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
            },
        )
    except ImportError:
        return ServiceHealth(
            name="redis",
            status=HealthStatus.UNKNOWN,
            message="Redis client not installed",
            latency_ms=(time.time() - start) * 1000,
        )
    except Exception as e:
        latency = (time.time() - start) * 1000
        logger.error(f"Redis health check failed: {e}")
        return ServiceHealth(
            name="redis",
            status=HealthStatus.UNHEALTHY,
            latency_ms=latency,
            message=f"Connection failed: {str(e)[:100]}",
        )


async def check_ollama() -> ServiceHealth:
    """Check the active local LLM provider (Ollama or LM Studio).

    The shared local client (`get_ollama_client`) is provider-aware and points at
    whichever local provider `CORE_LOCAL_PROVIDER` selects, so this check reports
    the one that is actually in use and names it accordingly.
    """
    from app.dependencies import get_ollama_client, get_local_provider

    is_lmstudio = get_local_provider() == "lmstudio"
    name = "lmstudio" if is_lmstudio else "ollama"
    label = "LM Studio" if is_lmstudio else "Ollama"

    start = time.time()
    try:
        client = get_ollama_client()
        models = await client.models.list()
        model_names = [m.id for m in models.data] if models.data else []

        latency = (time.time() - start) * 1000

        # Degraded if no models loaded
        status = HealthStatus.HEALTHY if model_names else HealthStatus.DEGRADED

        return ServiceHealth(
            name=name,
            status=status,
            latency_ms=latency,
            message=f"{label} connected, {len(model_names)} models",
            details={
                "provider": name,
                "model_count": len(model_names),
                "available_models": model_names[:5],  # First 5
                "has_models": len(model_names) > 0,
            },
        )
    except Exception as e:
        latency = (time.time() - start) * 1000
        logger.error(f"{label} health check failed: {e}")
        return ServiceHealth(
            name=name,
            status=HealthStatus.UNHEALTHY,
            latency_ms=latency,
            message=f"Connection failed: {str(e)[:100]}",
        )


async def check_vector_db() -> ServiceHealth:
    """Check pgvector extension health for vector search."""
    from app.dependencies import get_db_pool

    start = time.time()
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            # Check if pgvector extension is installed
            ext_check = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
            )

            if not ext_check:
                return ServiceHealth(
                    name="vector_db",
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=(time.time() - start) * 1000,
                    message="pgvector extension not installed",
                )

            # Get vector extension version
            version = await conn.fetchval(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )

            # Count indexed chunks
            chunk_count = await conn.fetchval(
                "SELECT COUNT(*) FROM kb_chunks WHERE embedding_vec IS NOT NULL"
            )

        latency = (time.time() - start) * 1000

        return ServiceHealth(
            name="vector_db",
            status=HealthStatus.HEALTHY,
            latency_ms=latency,
            message="pgvector operational",
            details={
                "extension_version": version,
                "indexed_chunks": chunk_count or 0,
                "vector_dimensions": 3072,
            },
        )
    except Exception as e:
        latency = (time.time() - start) * 1000
        logger.error(f"Vector DB health check failed: {e}")
        return ServiceHealth(
            name="vector_db",
            status=HealthStatus.DEGRADED,
            latency_ms=latency,
            message=f"Check failed: {str(e)[:100]}",
        )


async def check_websocket_manager() -> ServiceHealth:
    """Check WebSocket connection manager health."""
    start = time.time()
    try:
        from app.websocket_manager import manager

        connection_count = len(manager.active_connections)
        channel_count = len(manager.channel_subscribers)

        latency = (time.time() - start) * 1000

        return ServiceHealth(
            name="websocket",
            status=HealthStatus.HEALTHY,
            latency_ms=latency,
            message="WebSocket manager active",
            details={
                "active_connections": connection_count,
                "subscribed_channels": channel_count,
            },
        )
    except Exception as e:
        latency = (time.time() - start) * 1000
        logger.error(f"WebSocket health check failed: {e}")
        return ServiceHealth(
            name="websocket",
            status=HealthStatus.DEGRADED,
            latency_ms=latency,
            message=f"Check failed: {str(e)[:100]}",
        )


async def check_engine_state() -> ServiceHealth:
    """Check CORE engine state."""
    start = time.time()
    try:
        from app.controllers.engine import _active_runs

        total_runs = len(_active_runs)
        completed = sum(1 for r in _active_runs.values() if r.is_complete())
        running = total_runs - completed

        latency = (time.time() - start) * 1000

        return ServiceHealth(
            name="engine",
            status=HealthStatus.HEALTHY,
            latency_ms=latency,
            message="CORE engine ready",
            details={
                "active_runs": running,
                "completed_runs": completed,
                "total_tracked": total_runs,
            },
        )
    except Exception as e:
        latency = (time.time() - start) * 1000
        logger.error(f"Engine health check failed: {e}")
        return ServiceHealth(
            name="engine",
            status=HealthStatus.DEGRADED,
            latency_ms=latency,
            message=f"Check failed: {str(e)[:100]}",
        )


async def check_bus_queue() -> ServiceHealth:
    """Check Inter-Agent Communication Bus health and queue depths."""
    from app.dependencies import get_db_pool

    start = time.time()
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            # Total messages
            total_messages = (
                await conn.fetchval("SELECT COUNT(*) FROM bus_messages") or 0
            )

            # Messages by priority
            priority_rows = await conn.fetch(
                "SELECT priority, COUNT(*) AS cnt FROM bus_messages GROUP BY priority"
            )
            by_priority = {row["priority"]: row["cnt"] for row in priority_rows}

            # Active subscriptions
            subscription_count = (
                await conn.fetchval("SELECT COUNT(*) FROM bus_subscriptions") or 0
            )

            # Registered external agents
            agent_count = (
                await conn.fetchval("SELECT COUNT(*) FROM bus_external_agents") or 0
            )

        latency = (time.time() - start) * 1000

        return ServiceHealth(
            name="bus",
            status=HealthStatus.HEALTHY,
            latency_ms=latency,
            message=f"Bus operational, {total_messages} messages",
            details={
                "total_messages": total_messages,
                "messages_by_priority": by_priority,
                "active_subscriptions": subscription_count,
                "registered_agents": agent_count,
            },
        )
    except Exception as e:
        latency = (time.time() - start) * 1000
        logger.error(f"Bus health check failed: {e}")
        return ServiceHealth(
            name="bus",
            status=HealthStatus.DEGRADED,
            latency_ms=latency,
            message=f"Check failed: {str(e)[:100]}",
        )


async def check_task_queue() -> ServiceHealth:
    """Check Task Routing Engine health, queue depth, and success rates."""
    from app.dependencies import get_db_pool

    start = time.time()
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            # Queue depth (queued tasks)
            queued = (
                await conn.fetchval(
                    "SELECT COUNT(*) FROM tasks WHERE status = 'queued'"
                )
                or 0
            )

            # Running tasks
            running = (
                await conn.fetchval(
                    "SELECT COUNT(*) FROM tasks WHERE status = 'running'"
                )
                or 0
            )

            # Completed / failed / refused counts
            completed = (
                await conn.fetchval(
                    "SELECT COUNT(*) FROM tasks WHERE status = 'completed'"
                )
                or 0
            )
            failed = (
                await conn.fetchval(
                    "SELECT COUNT(*) FROM tasks WHERE status = 'failed'"
                )
                or 0
            )
            refused = (
                await conn.fetchval(
                    "SELECT COUNT(*) FROM tasks WHERE status = 'refused'"
                )
                or 0
            )

            total = queued + running + completed + failed + refused
            success_rate = round(completed / total * 100, 1) if total > 0 else 0.0

        latency = (time.time() - start) * 1000

        # Degraded if queue is backing up (arbitrary threshold: 100)
        status = HealthStatus.DEGRADED if queued > 100 else HealthStatus.HEALTHY

        return ServiceHealth(
            name="task_queue",
            status=status,
            latency_ms=latency,
            message=f"Task queue: {queued} queued, {running} running",
            details={
                "queue_depth": queued,
                "running": running,
                "completed": completed,
                "failed": failed,
                "refused": refused,
                "total": total,
                "success_rate_pct": success_rate,
            },
        )
    except Exception as e:
        latency = (time.time() - start) * 1000
        logger.error(f"Task queue health check failed: {e}")
        return ServiceHealth(
            name="task_queue",
            status=HealthStatus.DEGRADED,
            latency_ms=latency,
            message=f"Check failed: {str(e)[:100]}",
        )


async def check_system_resources() -> ServiceHealth:
    """Check host system resources (memory, CPU, disk)."""
    start = time.time()
    try:
        import psutil

        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        sys_mem = psutil.virtual_memory()

        latency = (time.time() - start) * 1000

        # Degraded if system memory > 90% or process RSS > 2 GB
        mem_pressure = sys_mem.percent > 90 or (mem.rss / (1024**3)) > 2
        status = HealthStatus.DEGRADED if mem_pressure else HealthStatus.HEALTHY

        return ServiceHealth(
            name="system",
            status=status,
            latency_ms=latency,
            message=f"System mem {sys_mem.percent}%, process {mem.rss / (1024**2):.0f} MB",
            details={
                "process_rss_mb": round(mem.rss / (1024**2), 2),
                "process_vms_mb": round(mem.vms / (1024**2), 2),
                "process_threads": process.num_threads(),
                "system_memory_percent": sys_mem.percent,
                "system_memory_available_mb": round(sys_mem.available / (1024**2), 2),
                "cpu_percent": process.cpu_percent(),
            },
        )
    except ImportError:
        return ServiceHealth(
            name="system",
            status=HealthStatus.UNKNOWN,
            latency_ms=(time.time() - start) * 1000,
            message="psutil not installed",
        )
    except Exception as e:
        latency = (time.time() - start) * 1000
        logger.error(f"System resource check failed: {e}")
        return ServiceHealth(
            name="system",
            status=HealthStatus.DEGRADED,
            latency_ms=latency,
            message=f"Check failed: {str(e)[:100]}",
        )


def determine_overall_status(checks: List[ServiceHealth]) -> HealthStatus:
    """
    Determine overall system health from individual checks.

    Rules:
    - All healthy = HEALTHY
    - Any critical service (database) unhealthy = UNHEALTHY
    - Any non-critical degraded/unhealthy = DEGRADED
    """
    critical_services = {"database"}

    statuses = {c.name: c.status for c in checks}

    # Check critical services first
    for service in critical_services:
        if service in statuses and statuses[service] == HealthStatus.UNHEALTHY:
            return HealthStatus.UNHEALTHY

    # Check if all healthy
    if all(s == HealthStatus.HEALTHY for s in statuses.values()):
        return HealthStatus.HEALTHY

    # Any unhealthy or degraded means degraded overall
    return HealthStatus.DEGRADED


async def get_comprehensive_health() -> Dict[str, Any]:
    """
    Run all health checks and return comprehensive status.

    Returns aggregated health with:
    - Overall status (healthy/degraded/unhealthy)
    - Individual service checks with response times
    - Timestamp and uptime
    - Summary statistics
    """
    start = time.time()

    # Run all checks concurrently
    results = await asyncio.gather(
        check_database(),
        check_redis(),
        check_ollama(),
        check_vector_db(),
        check_websocket_manager(),
        check_engine_state(),
        check_bus_queue(),
        check_task_queue(),
        check_system_resources(),
        return_exceptions=True,
    )

    # Process results, handle exceptions
    checks: List[ServiceHealth] = []
    for result in results:
        if isinstance(result, Exception):
            checks.append(
                ServiceHealth(
                    name="unknown",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check exception: {str(result)[:100]}",
                )
            )
        else:
            checks.append(result)

    # Calculate overall status
    overall = determine_overall_status(checks)

    # Build response
    total_latency = (time.time() - start) * 1000

    # Record snapshot for health history (fire-and-forget, never blocks response)
    try:
        from app.repository import health_repository

        asyncio.ensure_future(
            health_repository.record_snapshot(
                overall_status=overall.value,
                services={c.name: c.to_dict() for c in checks},
                total_latency_ms=round(total_latency, 2),
                summary={
                    "total_services": len(checks),
                    "healthy": sum(
                        1 for c in checks if c.status == HealthStatus.HEALTHY
                    ),
                    "degraded": sum(
                        1 for c in checks if c.status == HealthStatus.DEGRADED
                    ),
                    "unhealthy": sum(
                        1 for c in checks if c.status == HealthStatus.UNHEALTHY
                    ),
                    "unknown": sum(
                        1 for c in checks if c.status == HealthStatus.UNKNOWN
                    ),
                },
            )
        )
    except Exception:
        pass  # Never let history recording break health checks

    return {
        "status": overall.value,
        "service": "core-backend",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime": {
            "seconds": round(get_uptime_seconds(), 2),
            "formatted": get_uptime_formatted(),
        },
        "total_check_latency_ms": round(total_latency, 2),
        "services": {c.name: c.to_dict() for c in checks},
        "summary": {
            "total_services": len(checks),
            "healthy": sum(1 for c in checks if c.status == HealthStatus.HEALTHY),
            "degraded": sum(1 for c in checks if c.status == HealthStatus.DEGRADED),
            "unhealthy": sum(1 for c in checks if c.status == HealthStatus.UNHEALTHY),
            "unknown": sum(1 for c in checks if c.status == HealthStatus.UNKNOWN),
        },
    }


async def quick_health() -> Dict[str, Any]:
    """Quick health check for load balancer probes."""
    return {
        "status": "healthy",
        "service": "core-backend",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": round(get_uptime_seconds(), 2),
    }
