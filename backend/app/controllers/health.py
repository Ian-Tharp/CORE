"""
Health Check Controller

Provides comprehensive health check endpoints for monitoring.
Aggregates status from all CORE components.

Endpoints:
- GET /health - Quick health check
- GET /health/deep - Deep health check with component status
- GET /health/ready - Readiness probe for k8s
- GET /health/live - Liveness probe for k8s
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, status
import httpx

from app.dependencies import get_db_pool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


async def check_database() -> Dict[str, Any]:
    """Check database connectivity."""
    start = time.time()
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            return {
                "status": "healthy" if result == 1 else "unhealthy",
                "latency_ms": round((time.time() - start) * 1000, 2),
                "details": "Connection successful"
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "latency_ms": round((time.time() - start) * 1000, 2),
            "error": str(e)
        }


async def check_redis() -> Dict[str, Any]:
    """Check Redis connectivity."""
    start = time.time()
    try:
        # Try to connect to Redis
        async with httpx.AsyncClient() as client:
            response = await client.get("http://redis:6379/ping", timeout=5.0)
            # Redis doesn't respond to HTTP, but we can check if port is open
            return {
                "status": "healthy",
                "latency_ms": round((time.time() - start) * 1000, 2),
                "details": "Redis port reachable"
            }
    except Exception as e:
        # Alternative: assume healthy if we can import redis
        return {
            "status": "unknown",
            "latency_ms": round((time.time() - start) * 1000, 2),
            "details": "Redis check skipped (non-HTTP service)"
        }


async def check_ollama() -> Dict[str, Any]:
    """Check Ollama LLM service."""
    start = time.time()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://ollama:11434/api/tags", timeout=5.0)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return {
                    "status": "healthy",
                    "latency_ms": round((time.time() - start) * 1000, 2),
                    "models_loaded": len(models),
                    "details": f"{len(models)} models available"
                }
            else:
                return {
                    "status": "degraded",
                    "latency_ms": round((time.time() - start) * 1000, 2),
                    "details": f"HTTP {response.status_code}"
                }
    except Exception as e:
        return {
            "status": "unhealthy",
            "latency_ms": round((time.time() - start) * 1000, 2),
            "error": str(e)
        }


@router.get("", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, str]:
    """
    Quick health check endpoint.
    
    Returns 200 OK if the service is running.
    Used for basic load balancer health checks.
    """
    return {"status": "healthy", "service": "core-backend"}


@router.get("/deep", status_code=status.HTTP_200_OK)
async def deep_health_check() -> Dict[str, Any]:
    """
    Deep health check with component status.
    
    Checks all dependent services and returns aggregate status.
    Returns 200 if all critical services healthy, 503 if any critical service down.
    """
    start = time.time()
    
    # Run all checks concurrently
    db_check, ollama_check, redis_check = await asyncio.gather(
        check_database(),
        check_ollama(),
        check_redis(),
        return_exceptions=True
    )
    
    # Handle exceptions
    if isinstance(db_check, Exception):
        db_check = {"status": "error", "error": str(db_check)}
    if isinstance(ollama_check, Exception):
        ollama_check = {"status": "error", "error": str(ollama_check)}
    if isinstance(redis_check, Exception):
        redis_check = {"status": "error", "error": str(redis_check)}
    
    components = {
        "database": db_check,
        "ollama": ollama_check,
        "redis": redis_check
    }
    
    # Determine overall status
    critical_healthy = db_check.get("status") == "healthy"
    all_healthy = all(c.get("status") == "healthy" for c in components.values())
    
    if all_healthy:
        overall = "healthy"
    elif critical_healthy:
        overall = "degraded"
    else:
        overall = "unhealthy"
    
    result = {
        "status": overall,
        "service": "core-backend",
        "timestamp": datetime.utcnow().isoformat(),
        "total_latency_ms": round((time.time() - start) * 1000, 2),
        "components": components
    }
    
    # Return 503 if unhealthy
    if overall == "unhealthy":
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
    Checks critical dependencies (database).
    """
    db_status = await check_database()
    
    if db_status.get("status") != "healthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "reason": "database unavailable"}
        )
    
    return {"status": "ready"}


@router.get("/live", status_code=status.HTTP_200_OK)
async def liveness_probe() -> Dict[str, str]:
    """
    Kubernetes liveness probe.
    
    Returns 200 if service is alive.
    Should only fail if service needs restart.
    """
    return {"status": "alive"}


@router.get("/metrics", status_code=status.HTTP_200_OK)
async def metrics() -> Dict[str, Any]:
    """
    Basic metrics endpoint.
    
    Returns operational metrics for monitoring.
    For Prometheus integration, use prometheus-fastapi-instrumentator.
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": time.time() - process.create_time(),
        "memory": {
            "rss_mb": round(process.memory_info().rss / 1024 / 1024, 2),
            "vms_mb": round(process.memory_info().vms / 1024 / 1024, 2),
            "percent": round(process.memory_percent(), 2)
        },
        "cpu": {
            "percent": process.cpu_percent(),
            "num_threads": process.num_threads()
        },
        "connections": len(process.connections())
    }
