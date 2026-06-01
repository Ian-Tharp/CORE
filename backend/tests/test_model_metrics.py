"""
Tests for model performance tracking (P6).

Sentinel-value contract: every aggregation test would FAIL if you removed
the _percentile() calculation or the aggregate_metrics() function.

Coverage:
- _percentile(): nearest-rank method correctness
- aggregate_metrics(): p50/p95, error_rate, request_count from known records
- record_metric(): DB insert called with correct args
- get_model_stats(): wires raw fetch + aggregate
- GET /admin/metrics/models: endpoint returns correct structure
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.repository.model_metrics_repository import (
    _percentile,
    aggregate_metrics,
)


# ---------------------------------------------------------------------------
# _percentile — unit tests
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_p95_of_five_values_equals_last(self):
        """
        SENTINEL TEST — the spec example: p95 of [10,20,30,40,50] = 50.
        Change method to linear interpolation → returns 48 → fails.
        """
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert _percentile(data, 95) == 50.0

    def test_p50_of_five_values(self):
        """p50 of [10,20,30,40,50] → rank=ceil(2.5)=3 → value=30."""
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert _percentile(data, 50) == 30.0

    def test_p100_returns_maximum(self):
        data = [5.0, 10.0, 15.0]
        assert _percentile(data, 100) == 15.0

    def test_p0_is_clamped_to_minimum(self):
        """p=0 → rank=ceil(0)=0 → clamped to 1 → first value."""
        data = [5.0, 10.0, 15.0]
        assert _percentile(data, 0) == 5.0

    def test_empty_data_returns_zero(self):
        assert _percentile([], 95) == 0.0

    def test_single_element(self):
        assert _percentile([42.0], 50) == 42.0
        assert _percentile([42.0], 95) == 42.0

    def test_p95_ten_values(self):
        """
        p95 of [10,20,...,100]:
        rank = ceil(0.95 * 10) = ceil(9.5) = 10 → sorted[9] = 100.
        """
        data = [float(i * 10) for i in range(1, 11)]  # [10,20,...,100]
        assert _percentile(data, 95) == 100.0

    def test_p50_even_count(self):
        """p50 of [1,2,3,4]: rank=ceil(2)=2 → value=2."""
        data = [1.0, 2.0, 3.0, 4.0]
        assert _percentile(data, 50) == 2.0


# ---------------------------------------------------------------------------
# aggregate_metrics — unit tests with known data
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    def _records(
        self, model_id: str, latencies: list, error_indices: set = None
    ) -> list:
        """Build records list — error_indices are 0-based indices that are failures."""
        error_indices = error_indices or set()
        return [
            {
                "model_id": model_id,
                "latency_ms": lat,
                "success": i not in error_indices,
            }
            for i, lat in enumerate(latencies)
        ]

    def test_p95_and_p50_match_spec_example(self):
        """
        SENTINEL TEST — the spec: p95 of [10,20,30,40,50] = 50, p50 = 30.
        Removing aggregate_metrics or changing _percentile breaks this.
        """
        records = self._records("gpt-oss:20b", [10.0, 20.0, 30.0, 40.0, 50.0])
        result = aggregate_metrics(records)
        model = result["models"]["gpt-oss:20b"]
        assert model["p95_latency_ms"] == 50.0
        assert model["p50_latency_ms"] == 30.0

    def test_request_count(self):
        records = self._records("m1", [10.0, 20.0, 30.0])
        result = aggregate_metrics(records)
        assert result["models"]["m1"]["request_count"] == 3

    def test_error_rate_all_success(self):
        records = self._records("m1", [10.0, 20.0])
        result = aggregate_metrics(records)
        assert result["models"]["m1"]["error_rate"] == 0.0
        assert result["models"]["m1"]["error_count"] == 0

    def test_error_rate_all_failure(self):
        records = self._records("m1", [10.0, 20.0], error_indices={0, 1})
        result = aggregate_metrics(records)
        assert result["models"]["m1"]["error_rate"] == 1.0
        assert result["models"]["m1"]["success_count"] == 0

    def test_error_rate_half(self):
        """2 successes, 2 failures → error_rate = 0.5."""
        records = self._records("m1", [10.0, 20.0, 30.0, 40.0], error_indices={0, 1})
        result = aggregate_metrics(records)
        assert result["models"]["m1"]["error_rate"] == 0.5

    def test_multiple_models_are_separated(self):
        """
        SENTINEL TEST — records from model A must not bleed into model B's stats.
        """
        records = self._records("model-A", [100.0, 200.0, 300.0]) + self._records(
            "model-B", [10.0, 20.0, 30.0]
        )
        result = aggregate_metrics(records)
        assert "model-A" in result["models"]
        assert "model-B" in result["models"]
        # p50 of A should be ~200, B should be ~20
        assert result["models"]["model-A"]["p50_latency_ms"] == 200.0
        assert result["models"]["model-B"]["p50_latency_ms"] == 20.0

    def test_total_requests_is_sum_across_all_models(self):
        records = self._records("model-A", [10.0, 20.0]) + self._records(
            "model-B", [30.0, 40.0, 50.0]
        )
        result = aggregate_metrics(records)
        assert result["total_requests"] == 5

    def test_empty_records_returns_empty_models(self):
        result = aggregate_metrics([])
        assert result["models"] == {}
        assert result["total_requests"] == 0

    def test_p95_large_dataset(self):
        """p95 of 20 values [5,10,...,100]: rank=ceil(0.95*20)=19 → value=95."""
        latencies = [float(i * 5) for i in range(1, 21)]  # [5,10,...,100]
        records = self._records("big-model", latencies)
        result = aggregate_metrics(records)
        assert result["models"]["big-model"]["p95_latency_ms"] == 95.0

    def test_success_count_is_tracked(self):
        records = self._records("m1", [10.0, 20.0, 30.0], error_indices={2})
        result = aggregate_metrics(records)
        assert result["models"]["m1"]["success_count"] == 2
        assert result["models"]["m1"]["error_count"] == 1


# ---------------------------------------------------------------------------
# record_metric — DB write unit tests
# ---------------------------------------------------------------------------


class TestRecordMetric:
    @pytest.mark.asyncio
    async def test_record_metric_inserts_correct_values(self):
        """
        SENTINEL TEST — record_metric must call DB execute with correct args.
        Remove the INSERT → execute not called → test fails.
        """
        from app.repository.model_metrics_repository import record_metric
        from tests.test_communication_pagination import _make_db_pool_mock

        mock_conn = AsyncMock()
        mock_pool = _make_db_pool_mock(mock_conn)

        with patch(
            "app.repository.model_metrics_repository.get_db_pool",
            new=AsyncMock(return_value=mock_pool),
        ):
            await record_metric(
                model_id="SENTINEL_MODEL",
                latency_ms=123.45,
                success=True,
                token_count=500,
            )

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        # Verify key values are in the positional arguments
        assert "SENTINEL_MODEL" in call_args
        assert 123.45 in call_args
        assert True in call_args

    @pytest.mark.asyncio
    async def test_record_metric_failure_path(self):
        """Failure metrics (success=False) are also persisted."""
        from app.repository.model_metrics_repository import record_metric
        from tests.test_communication_pagination import _make_db_pool_mock

        mock_conn = AsyncMock()
        mock_pool = _make_db_pool_mock(mock_conn)

        with patch(
            "app.repository.model_metrics_repository.get_db_pool",
            new=AsyncMock(return_value=mock_pool),
        ):
            await record_metric(
                model_id="m1",
                latency_ms=999.0,
                success=False,
            )

        call_args = mock_conn.execute.call_args[0]
        assert False in call_args

    @pytest.mark.asyncio
    async def test_record_metric_swallows_db_exception(self):
        """DB failures must not propagate — metric is best-effort."""
        from app.repository.model_metrics_repository import record_metric

        with patch(
            "app.repository.model_metrics_repository.get_db_pool",
            new=AsyncMock(side_effect=RuntimeError("db down")),
        ):
            # Should not raise
            await record_metric("m1", 100.0, success=True)


# ---------------------------------------------------------------------------
# get_model_stats — integration of fetch + aggregate
# ---------------------------------------------------------------------------


class TestGetModelStats:
    @pytest.mark.asyncio
    async def test_get_model_stats_returns_aggregation(self):
        """
        SENTINEL TEST — get_model_stats must fetch records and aggregate them.
        Mock get_raw_metrics with known data and verify p95.
        """
        from app.repository.model_metrics_repository import get_model_stats

        fake_records = [
            {"model_id": "test-model", "latency_ms": float(v), "success": True}
            for v in [10, 20, 30, 40, 50]
        ]

        with patch(
            "app.repository.model_metrics_repository.get_raw_metrics",
            new=AsyncMock(return_value=fake_records),
        ):
            result = await get_model_stats()

        assert "test-model" in result["models"]
        assert result["models"]["test-model"]["p95_latency_ms"] == 50.0

    @pytest.mark.asyncio
    async def test_get_model_stats_passes_model_id_filter(self):
        """model_id filter is forwarded to get_raw_metrics."""
        from app.repository.model_metrics_repository import get_model_stats

        with patch(
            "app.repository.model_metrics_repository.get_raw_metrics",
            new=AsyncMock(return_value=[]),
        ) as mock_fetch:
            await get_model_stats(model_id="SENTINEL_FILTER")

        mock_fetch.assert_called_once_with(model_id="SENTINEL_FILTER")


# ---------------------------------------------------------------------------
# Admin endpoint — GET /admin/metrics/models
# ---------------------------------------------------------------------------


class TestModelMetricsEndpoint:
    def test_get_model_stats_route_exists(self):
        """
        SENTINEL TEST — GET /admin/metrics/models route must be registered.
        Remove the endpoint → route absent → test fails.
        """
        from app.controllers.admin import router as admin_router

        routes = {(r.path, m) for r in admin_router.routes for m in r.methods}
        assert (
            "/admin/metrics/models",
            "GET",
        ) in routes, "GET /admin/metrics/models route not found in admin router"

    @pytest.mark.asyncio
    async def test_endpoint_calls_get_model_stats(self):
        """
        SENTINEL TEST — the endpoint must call get_model_stats from the repository.
        Patch the repository function at the import site inside the endpoint.
        """
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.controllers.admin import router as admin_router
        from app.core.security import get_api_key
        from app.auth import require_api_key
        import app.repository.model_metrics_repository as mmr

        fake_stats = {
            "models": {"test-model": {"p50_latency_ms": 30.0, "p95_latency_ms": 50.0}},
            "total_requests": 5,
        }

        original_fn = mmr.get_model_stats
        mmr.get_model_stats = AsyncMock(return_value=fake_stats)

        async def _bypass_api_key():
            return {"name": "test", "permissions": ["*"]}

        async def _bypass_require_api_key():
            return "test-key"

        try:
            app_instance = FastAPI()
            app_instance.include_router(admin_router)
            app_instance.dependency_overrides[get_api_key] = _bypass_api_key
            app_instance.dependency_overrides[require_api_key] = _bypass_require_api_key
            client = TestClient(app_instance)
            resp = client.get("/admin/metrics/models")
        finally:
            mmr.get_model_stats = original_fn

        assert resp.status_code == 200
        body = resp.json()
        assert "models" in body
        assert body["models"]["test-model"]["p95_latency_ms"] == 50.0
