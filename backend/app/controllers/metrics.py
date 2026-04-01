"""
Prometheus metrics endpoint.

Exposes GET /metrics returning Prometheus text exposition format (0.0.4).
Standard scrape target for Prometheus / Grafana Agent / VictoriaMetrics.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from app.core.middleware import get_metrics

router = APIRouter(tags=["metrics"])

_PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics() -> PlainTextResponse:
    """
    Prometheus-compatible metrics endpoint.

    Returns request counters, error rates, latency averages, per-endpoint
    and per-status-code counts in Prometheus text exposition format.

    Intended as a scrape target; no authentication required so Prometheus
    can reach it without credentials.
    """
    payload = get_metrics().to_prometheus()
    return PlainTextResponse(content=payload, media_type=_PROMETHEUS_CONTENT_TYPE)
