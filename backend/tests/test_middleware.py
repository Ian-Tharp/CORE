"""
Tests for CORE middleware — RequestMetrics unit tests.

Covers recording, statistics, error-rate tracking, reset,
and top-endpoint ordering.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from app.core.middleware import RequestMetrics


class TestRequestMetricsRecord:
    """Basic recording of requests."""

    def test_initial_state(self):
        m = RequestMetrics()
        assert m.total_requests == 0
        assert m.error_requests == 0
        assert m.total_duration_ms == 0.0
        assert m.requests_by_path == {}
        assert m.requests_by_status == {}

    def test_single_success(self):
        m = RequestMetrics()
        m.record("/api/foo", "GET", 200, 12.5)
        assert m.total_requests == 1
        assert m.error_requests == 0
        assert m.total_duration_ms == 12.5

    def test_single_error(self):
        m = RequestMetrics()
        m.record("/api/bar", "POST", 500, 50.0)
        assert m.total_requests == 1
        assert m.error_requests == 1

    def test_400_counts_as_error(self):
        m = RequestMetrics()
        m.record("/x", "GET", 400, 1.0)
        assert m.error_requests == 1

    def test_399_not_error(self):
        m = RequestMetrics()
        m.record("/x", "GET", 399, 1.0)
        assert m.error_requests == 0

    def test_multiple_records_accumulate(self):
        m = RequestMetrics()
        m.record("/a", "GET", 200, 10.0)
        m.record("/b", "POST", 201, 20.0)
        m.record("/a", "GET", 500, 30.0)
        assert m.total_requests == 3
        assert m.error_requests == 1
        assert m.total_duration_ms == 60.0

    def test_path_key_includes_method(self):
        m = RequestMetrics()
        m.record("/api", "GET", 200, 5.0)
        m.record("/api", "POST", 200, 5.0)
        assert "GET /api" in m.requests_by_path
        assert "POST /api" in m.requests_by_path

    def test_status_tracking(self):
        m = RequestMetrics()
        m.record("/a", "GET", 200, 1.0)
        m.record("/b", "GET", 200, 1.0)
        m.record("/c", "GET", 404, 1.0)
        assert m.requests_by_status == {"200": 2, "404": 1}


class TestRequestMetricsGetStats:
    """Statistics computation."""

    def test_empty_stats(self):
        m = RequestMetrics()
        stats = m.get_stats()
        assert stats["total_requests"] == 0
        assert stats["error_rate"] == 0
        assert stats["avg_duration_ms"] == 0
        assert stats["top_endpoints"] == []

    def test_error_rate_calculation(self):
        m = RequestMetrics()
        m.record("/a", "GET", 200, 10.0)
        m.record("/b", "GET", 500, 10.0)
        stats = m.get_stats()
        assert stats["error_rate"] == pytest.approx(0.5)

    def test_avg_duration(self):
        m = RequestMetrics()
        m.record("/a", "GET", 200, 10.0)
        m.record("/b", "GET", 200, 30.0)
        stats = m.get_stats()
        assert stats["avg_duration_ms"] == pytest.approx(20.0)

    def test_requests_per_second_positive(self):
        m = RequestMetrics()
        # Push start_time back 10 seconds
        m.start_time = datetime.utcnow() - timedelta(seconds=10)
        m.record("/a", "GET", 200, 1.0)
        m.record("/b", "GET", 200, 1.0)
        stats = m.get_stats()
        # ~0.2 rps, allow tolerance
        assert 0.15 < stats["requests_per_second"] < 0.25

    def test_top_endpoints_sorted_by_count(self):
        m = RequestMetrics()
        m.record("/rare", "GET", 200, 1.0)
        for _ in range(5):
            m.record("/popular", "GET", 200, 1.0)
        for _ in range(3):
            m.record("/medium", "GET", 200, 1.0)
        stats = m.get_stats()
        paths = [e["path"] for e in stats["top_endpoints"]]
        assert paths == ["GET /popular", "GET /medium", "GET /rare"]

    def test_top_endpoints_limited_to_10(self):
        m = RequestMetrics()
        for i in range(15):
            m.record(f"/ep{i}", "GET", 200, 1.0)
        stats = m.get_stats()
        assert len(stats["top_endpoints"]) == 10

    def test_endpoint_error_rate(self):
        m = RequestMetrics()
        m.record("/flaky", "GET", 200, 1.0)
        m.record("/flaky", "GET", 500, 1.0)
        stats = m.get_stats()
        flaky = [e for e in stats["top_endpoints"] if e["path"] == "GET /flaky"][0]
        assert flaky["error_rate"] == pytest.approx(0.5)

    def test_endpoint_avg_ms(self):
        m = RequestMetrics()
        m.record("/slow", "GET", 200, 100.0)
        m.record("/slow", "GET", 200, 200.0)
        stats = m.get_stats()
        slow = [e for e in stats["top_endpoints"] if e["path"] == "GET /slow"][0]
        assert slow["avg_ms"] == pytest.approx(150.0)


class TestRequestMetricsReset:
    """Reset clears all state."""

    def test_reset_clears_counters(self):
        m = RequestMetrics()
        m.record("/a", "GET", 200, 10.0)
        m.record("/b", "POST", 500, 20.0)
        m.reset()
        assert m.total_requests == 0
        assert m.error_requests == 0
        assert m.total_duration_ms == 0.0
        assert m.requests_by_path == {}
        assert m.requests_by_status == {}

    def test_reset_refreshes_start_time(self):
        m = RequestMetrics()
        old_start = m.start_time
        import time
        time.sleep(0.01)
        m.reset()
        assert m.start_time >= old_start
