"""
CORE Health Check Aggregation

Provides comprehensive health checks for all CORE services:
- Backend API
- PostgreSQL database
- Redis cache
- Ollama LLM
- WebSocket connections
- Alerting webhooks for degraded/unhealthy states
- MCP server availability (binary / Docker command reachability)

DONE: Health history tracking (health_repository + /health/history endpoints)
DONE: MCP server health checks (check_mcp_servers)
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
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


def _load_mcp_config() -> Dict[str, Any]:
    """
    Load MCP server configuration from .mcp.json.

    Searches:
    1. Path in CORE_MCP_CONFIG env var
    2. Project root (four levels above this file's location)
    3. Current working directory

    Returns an empty ``{"mcpServers": {}}`` dict when no file is found.
    """
    candidates = []

    env_path = os.getenv("CORE_MCP_CONFIG")
    if env_path:
        candidates.append(Path(env_path))

    # Standard project root (backend/app/core/health.py → ../../../../.mcp.json)
    candidates.append(Path(__file__).resolve().parents[3] / ".mcp.json")
    candidates.append(Path.cwd() / ".mcp.json")

    for path in candidates:
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.debug("Failed to parse MCP config at %s: %s", path, exc)

    return {"mcpServers": {}}


async def check_mcp_servers() -> HealthCheck:
    """
    Check reachability of all configured MCP servers.

    For each server in .mcp.json / CORE_MCP_CONFIG, this check verifies that
    the ``command`` binary is discoverable on the system (via absolute path or
    PATH look-up).  It does **not** start the servers — it only confirms the
    pre-requisite executables are present.

    Status rules:
    - No servers configured  → HEALTHY  (trivially OK)
    - All commands reachable → HEALTHY
    - Some commands missing  → DEGRADED
    - Cannot load config     → DEGRADED
    """
    try:
        config = _load_mcp_config()
        servers = config.get("mcpServers", {})

        if not servers:
            return HealthCheck(
                name="mcp_servers",
                status=HealthStatus.HEALTHY,
                message="No MCP servers configured",
                details={"server_count": 0},
            )

        server_statuses: Dict[str, str] = {}
        missing: List[str] = []

        for name, cfg in servers.items():
            command = cfg.get("command", "")
            if not command:
                server_statuses[name] = "no_command"
                missing.append(name)
                continue

            cmd_path = Path(command)
            if cmd_path.is_absolute():
                reachable = cmd_path.is_file()
            else:
                reachable = shutil.which(command) is not None

            if reachable:
                server_statuses[name] = "reachable"
            else:
                server_statuses[name] = "command_not_found"
                missing.append(name)

        if missing:
            return HealthCheck(
                name="mcp_servers",
                status=HealthStatus.DEGRADED,
                message=f"{len(missing)} MCP server(s) have missing commands: {', '.join(missing)}",
                details={
                    "server_count": len(servers),
                    "missing_count": len(missing),
                    "servers": server_statuses,
                },
            )

        return HealthCheck(
            name="mcp_servers",
            status=HealthStatus.HEALTHY,
            message=f"All {len(servers)} MCP server(s) reachable",
            details={
                "server_count": len(servers),
                "servers": server_statuses,
            },
        )

    except Exception as exc:
        logger.error("MCP server health check failed: %s", exc)
        return HealthCheck(
            name="mcp_servers",
            status=HealthStatus.DEGRADED,
            message=f"MCP config error: {exc}",
        )


async def fire_health_alert(overall: HealthStatus, health_result: Dict[str, Any]) -> None:
    """
    Fire a webhook alert when the overall health status is degraded or unhealthy.

    Uses the global webhook service so any registered ``health.degraded`` or
    ``health.unhealthy`` subscriber receives the payload.  Failures are logged
    and swallowed — alerting must never block or crash the health endpoint.

    Args:
        overall: The aggregated health status (DEGRADED or UNHEALTHY).
        health_result: The full health result dict to include in the payload.
    """
    try:
        from app.services.webhook_service import get_webhook_service, WebhookEvent
        service = get_webhook_service()

        event = (
            WebhookEvent.HEALTH_UNHEALTHY
            if overall == HealthStatus.UNHEALTHY
            else WebhookEvent.HEALTH_DEGRADED
        )

        await service.fire(
            event=event,
            payload={
                "status": overall.value,
                "timestamp": health_result.get("timestamp"),
                "summary": health_result.get("summary", {}),
                "failed_checks": [
                    c for c in health_result.get("checks", [])
                    if c.get("status") != HealthStatus.HEALTHY.value
                ],
            },
        )
        logger.debug("Health alert fired for status '%s'", overall.value)
    except Exception as exc:
        logger.warning("Health alert webhook failed (non-critical): %s", exc)


async def get_full_health() -> Dict[str, Any]:
    """
    Run all health checks and return aggregated status.

    Fires a webhook alert (non-blocking) when overall status is degraded or
    unhealthy so external monitoring systems can react.

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
        check_mcp_servers(),
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

    health_result = {
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

    # Fire alerting webhook when health is not fully OK (non-blocking)
    if overall != HealthStatus.HEALTHY:
        asyncio.create_task(fire_health_alert(overall, health_result))

    return health_result


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
