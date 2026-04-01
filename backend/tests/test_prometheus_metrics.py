"""
Tests for Prometheus metrics endpoint (middleware RSI TODO).

Sentinel-value contract:
- Removing to_prometheus() breaks all format tests
- Removing the /metrics route breaks test_route_exists
- Recording 0 requests but returning a non-zero total breaks test_counters_reflect_recorded_data
"""

from __future__ import annotations

import pytest
from app.core.middleware import RequestMetrics


# ---------------------------------------------------------------------------
# RequestMetrics.to_prometheus() — unit tests
# ---------------------------------------------------------------------------

class TestToPrometheus:
    def _metrics_with_requests(self) -> RequestMetrics:
        """Build a metrics instance with two known requests recorded."""
        m = RequestMetrics()
        m.record(path="/api/test", method="GET", status=200, duration_ms=10.0)
        m.record(path="/api/test", method="GET", status=200, duration_ms=30.0)
        m.record(path="/api/other", method="POST", status=500, duration_ms=50.0)
        return m

    def test_output_is_string(self):
        m = RequestMetrics()
        assert isinstance(m.to_prometheus(), str)

    def test_ends_with_newline(self):
        """Prometheus format requires a trailing newline."""
        m = RequestMetrics()
        assert m.to_prometheus().endswith("\n")

    def test_contains_help_lines(self):
        """Every metric must have a # HELP line."""
        m = RequestMetrics()
        output = m.to_prometheus()
        assert "# HELP core_http_requests_total" in output

    def test_contains_type_lines(self):
        """Every metric must have a # TYPE line."""
        m = RequestMetrics()
        output = m.to_prometheus()
        assert "# TYPE core_http_requests_total counter" in output

    def test_counters_reflect_recorded_data(self):
        """
        SENTINEL TEST — total_requests must reflect calls to record().
        Remove _metrics.record() call in middleware → count stays 0 → test fails.
        """
        m = self._metrics_with_requests()
        output = m.to_prometheus()
        assert "core_http_requests_total 3" in output

    def test_error_counter_reflects_failures(self):
        """
        SENTINEL TEST — error count must equal 5xx/4xx records.
        Remove error counting → error_requests stays 0 → test fails.
        """
        m = self._metrics_with_requests()
        output = m.to_prometheus()
        assert "core_http_errors_total 1" in output

    def test_per_status_labels_present(self):
        """
        SENTINEL TEST — per-status lines must include status label.
        Remove by_status loop → status="200" line absent → test fails.
        """
        m = self._metrics_with_requests()
        output = m.to_prometheus()
        assert 'core_http_requests_by_status{status="200"}' in output
        assert 'core_http_requests_by_status{status="500"}' in output

    def test_per_endpoint_labels_present(self):
        """
        SENTINEL TEST — per-endpoint lines must include endpoint label.
        Remove top_endpoints loop → endpoint label absent → test fails.
        """
        m = self._metrics_with_requests()
        output = m.to_prometheus()
        assert "core_endpoint_request_count" in output
        assert "GET /api/test" in output

    def test_uptime_metric_present(self):
        m = RequestMetrics()
        output = m.to_prometheus()
        assert "core_uptime_seconds" in output

    def test_zero_requests_still_valid_format(self):
        """Empty metrics must still produce parseable Prometheus format."""
        m = RequestMetrics()
        output = m.to_prometheus()
        assert "core_http_requests_total 0" in output
        assert "core_http_errors_total 0" in output


# ---------------------------------------------------------------------------
# GET /metrics endpoint
# ---------------------------------------------------------------------------

class TestMetricsEndpoint:
    def _make_client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.controllers.metrics import router

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_route_exists(self):
        """
        SENTINEL TEST — GET /metrics route must be registered.
        Remove the endpoint → 404 → test fails.
        """
        client = self._make_client()
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_content_type_is_prometheus(self):
        """
        SENTINEL TEST — Content-Type must be text/plain; version=0.0.4.
        Wrong content type → Prometheus scraper rejects it.
        """
        client = self._make_client()
        resp = client.get("/metrics")
        assert "text/plain" in resp.headers["content-type"]
        assert "0.0.4" in resp.headers["content-type"]

    def test_response_body_contains_help_lines(self):
        """Response body must be valid Prometheus text format."""
        client = self._make_client()
        resp = client.get("/metrics")
        assert "# HELP" in resp.text
        assert "# TYPE" in resp.text

    def test_endpoint_calls_to_prometheus(self):
        """
        SENTINEL TEST — the endpoint must call to_prometheus() on the metrics instance.
        Replace with hardcoded string → sentinel not found → test fails.
        """
        from app.core import middleware as mw
        from unittest.mock import patch

        client = self._make_client()

        sentinel_output = "# HELP SENTINEL_METRIC test\nSENTINEL_METRIC 1\n"
        with patch.object(mw._metrics, "to_prometheus", return_value=sentinel_output):
            resp = client.get("/metrics")

        assert "SENTINEL_METRIC" in resp.text

    def test_no_auth_required(self):
        """
        /metrics must be publicly accessible (no API key).
        Prometheus scrapers cannot send auth headers by default.
        """
        client = self._make_client()
        # No auth headers — must still return 200
        resp = client.get("/metrics")
        assert resp.status_code == 200
