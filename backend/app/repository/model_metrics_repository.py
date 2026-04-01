"""
Model Metrics Repository

Persists per-request model performance data and provides
aggregation queries for p50/p95 latency and error rates.

Table schema:
    model_metrics(
        id          SERIAL PRIMARY KEY,
        model_id    TEXT NOT NULL,
        latency_ms  DOUBLE PRECISION NOT NULL,
        ttfb_ms     DOUBLE PRECISION,
        token_count INTEGER,
        success     BOOLEAN NOT NULL,
        created_at  TIMESTAMPTZ DEFAULT NOW()
    )
"""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Optional

from app.dependencies import get_db_pool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

async def ensure_model_metrics_tables() -> None:
    """Create model_metrics table if it doesn't exist."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_metrics (
                id          SERIAL PRIMARY KEY,
                model_id    TEXT NOT NULL,
                latency_ms  DOUBLE PRECISION NOT NULL,
                ttfb_ms     DOUBLE PRECISION,
                token_count INTEGER,
                success     BOOLEAN NOT NULL,
                created_at  TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_model_metrics_model_id
                ON model_metrics (model_id);
            CREATE INDEX IF NOT EXISTS idx_model_metrics_created_at
                ON model_metrics (created_at DESC);
            """
        )


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

async def record_metric(
    model_id: str,
    latency_ms: float,
    success: bool,
    ttfb_ms: Optional[float] = None,
    token_count: Optional[int] = None,
) -> None:
    """Insert a single model request metric row."""
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO model_metrics (model_id, latency_ms, ttfb_ms, token_count, success)
                VALUES ($1, $2, $3, $4, $5)
                """,
                model_id, latency_ms, ttfb_ms, token_count, success,
            )
    except Exception as exc:
        # Non-critical — log but don't surface to caller
        logger.warning("Failed to record model metric for %s: %s", model_id, exc)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

async def get_raw_metrics(model_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch all metric rows, optionally filtered by model_id.

    Returns list of dicts with keys: model_id, latency_ms, ttfb_ms, token_count, success.
    """
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        if model_id:
            rows = await conn.fetch(
                """
                SELECT model_id, latency_ms, ttfb_ms, token_count, success
                FROM model_metrics
                WHERE model_id = $1
                ORDER BY created_at DESC
                """,
                model_id,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT model_id, latency_ms, ttfb_ms, token_count, success
                FROM model_metrics
                ORDER BY created_at DESC
                """
            )
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Aggregation (pure Python, DB-independent, fully testable)
# ---------------------------------------------------------------------------

def _percentile(sorted_values: List[float], p: float) -> float:
    """
    Nearest-rank percentile (ceiling method, 1-based).

    Example: p95 of [10, 20, 30, 40, 50]:
        rank = ceil(0.95 * 5) = ceil(4.75) = 5 → sorted[4] = 50
    """
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    rank = math.ceil(p / 100.0 * n)
    rank = max(1, min(rank, n))
    return sorted_values[rank - 1]


def aggregate_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute per-model p50/p95 latency and error rate from raw records.

    Args:
        records: List of dicts with keys: model_id, latency_ms, success

    Returns:
        {
            "models": {
                "<model_id>": {
                    "request_count": int,
                    "success_count": int,
                    "error_count": int,
                    "error_rate": float,   # 0.0–1.0
                    "p50_latency_ms": float,
                    "p95_latency_ms": float,
                }
            },
            "total_requests": int
        }
    """
    by_model: Dict[str, Dict[str, Any]] = {}

    for r in records:
        mid = r["model_id"]
        if mid not in by_model:
            by_model[mid] = {"latencies": [], "success": 0, "error": 0}
        by_model[mid]["latencies"].append(float(r["latency_ms"]))
        if r["success"]:
            by_model[mid]["success"] += 1
        else:
            by_model[mid]["error"] += 1

    result: Dict[str, Any] = {}
    for mid, data in by_model.items():
        sorted_lat = sorted(data["latencies"])
        total = data["success"] + data["error"]
        result[mid] = {
            "request_count": total,
            "success_count": data["success"],
            "error_count": data["error"],
            "error_rate": round(data["error"] / total, 4) if total else 0.0,
            "p50_latency_ms": round(_percentile(sorted_lat, 50), 2),
            "p95_latency_ms": round(_percentile(sorted_lat, 95), 2),
        }

    return {
        "models": result,
        "total_requests": len(records),
    }


async def get_model_stats(model_id: Optional[str] = None) -> Dict[str, Any]:
    """Fetch raw metrics and aggregate them."""
    records = await get_raw_metrics(model_id=model_id)
    return aggregate_metrics(records)
