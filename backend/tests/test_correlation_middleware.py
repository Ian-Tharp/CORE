"""Tests for consolidated correlation ID middleware.

Verifies that CorrelationIDMiddleware provides a single source of truth for
request tracing IDs and that RequestContextMiddleware properly reuses them.
"""

import pytest
import uuid
from unittest.mock import MagicMock
from starlette.requests import Request
from starlette.datastructures import State
from starlette.testclient import TestClient
from fastapi import FastAPI

from app.middleware.correlation import (
    CorrelationIDMiddleware,
    CORRELATION_HEADER,
    REQUEST_ID_HEADER,
)
from app.middleware.logging import get_correlation_id
from app.core.middleware import RequestContextMiddleware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_app(*middleware_classes) -> FastAPI:
    """Build a minimal FastAPI app with the given middleware."""
    app = FastAPI()

    @app.get("/echo")
    async def echo(request: Request):
        return {
            "correlation_id": getattr(request.state, "correlation_id", None),
            "request_id": getattr(request.state, "request_id", None),
        }

    for cls in middleware_classes:
        app.add_middleware(cls)

    return app


# ---------------------------------------------------------------------------
# CorrelationIDMiddleware
# ---------------------------------------------------------------------------

class TestCorrelationIDMiddleware:
    """Tests for the CorrelationIDMiddleware."""

    def setup_method(self):
        self.app = _build_app(CorrelationIDMiddleware)
        self.client = TestClient(self.app)

    def test_generates_correlation_id_when_no_header(self):
        resp = self.client.get("/echo")
        assert resp.status_code == 200
        assert CORRELATION_HEADER in resp.headers
        assert REQUEST_ID_HEADER in resp.headers
        assert len(resp.headers[CORRELATION_HEADER]) == 8

    def test_reuses_incoming_correlation_id(self):
        incoming_id = "trace-abc-123"
        resp = self.client.get("/echo", headers={CORRELATION_HEADER: incoming_id})
        assert resp.status_code == 200
        assert resp.headers[CORRELATION_HEADER] == incoming_id
        assert resp.headers[REQUEST_ID_HEADER] == incoming_id
        body = resp.json()
        assert body["correlation_id"] == incoming_id
        assert body["request_id"] == incoming_id

    def test_sets_both_state_attributes(self):
        resp = self.client.get("/echo")
        body = resp.json()
        assert body["correlation_id"] is not None
        assert body["correlation_id"] == body["request_id"]

    def test_different_requests_get_different_ids(self):
        ids = set()
        for _ in range(10):
            resp = self.client.get("/echo")
            ids.add(resp.headers[CORRELATION_HEADER])
        assert len(ids) == 10

    def test_empty_header_generates_new_id(self):
        resp = self.client.get("/echo", headers={CORRELATION_HEADER: ""})
        assert resp.status_code == 200
        assert len(resp.headers[CORRELATION_HEADER]) == 8


# ---------------------------------------------------------------------------
# RequestContextMiddleware reads X-Correlation-ID header
# ---------------------------------------------------------------------------

class TestRequestContextReusesCorrelationID:
    """Verify RequestContextMiddleware reads the incoming header."""

    def setup_method(self):
        self.app = _build_app(RequestContextMiddleware, CorrelationIDMiddleware)
        self.client = TestClient(self.app)

    def test_incoming_header_propagates_through_both_middlewares(self):
        incoming_id = "external-trace-42"
        resp = self.client.get("/echo", headers={CORRELATION_HEADER: incoming_id})
        body = resp.json()
        # Both middlewares should read the same incoming header
        assert body["correlation_id"] == incoming_id
        assert body["request_id"] == incoming_id

    def test_response_time_header_present(self):
        resp = self.client.get("/echo")
        assert "X-Response-Time" in resp.headers

    def test_without_incoming_header_both_generate_ids(self):
        """Without incoming header, both generate IDs (possibly different)."""
        resp = self.client.get("/echo")
        body = resp.json()
        # Both should have values set
        assert body["correlation_id"] is not None
        assert body["request_id"] is not None


# ---------------------------------------------------------------------------
# get_correlation_id helper
# ---------------------------------------------------------------------------

class TestGetCorrelationIdHelper:

    def test_returns_correlation_id_when_set(self):
        request = MagicMock(spec=Request)
        request.state = MagicMock()
        request.state.correlation_id = "abc123"
        request.state.request_id = "def456"
        assert get_correlation_id(request) == "abc123"

    def test_returns_request_id_when_correlation_missing(self):
        request = MagicMock(spec=Request)
        state = State()
        state.request_id = "req-789"
        request.state = state
        assert get_correlation_id(request) == "req-789"

    def test_generates_fallback_when_nothing_set(self):
        request = MagicMock(spec=Request)
        request.state = State()
        result = get_correlation_id(request)
        assert result is not None
        assert len(result) == 8


# ---------------------------------------------------------------------------
# Integration: full middleware stack
# ---------------------------------------------------------------------------

class TestFullMiddlewareStack:

    def setup_method(self):
        from app.core.middleware import ErrorHandlerMiddleware, MetricsMiddleware
        app = FastAPI()

        @app.get("/ok")
        async def ok(request: Request):
            return {"id": request.state.correlation_id}

        @app.get("/fail")
        async def fail():
            raise ValueError("boom")

        app.add_middleware(ErrorHandlerMiddleware)
        app.add_middleware(RequestContextMiddleware)
        app.add_middleware(MetricsMiddleware)
        app.add_middleware(CorrelationIDMiddleware)

        self.app = app
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def test_successful_request_has_correlation_headers(self):
        resp = self.client.get("/ok")
        assert resp.status_code == 200
        assert CORRELATION_HEADER in resp.headers
        assert "X-Response-Time" in resp.headers

    def test_error_response_includes_request_id(self):
        resp = self.client.get("/fail")
        assert resp.status_code == 500
        body = resp.json()
        assert "request_id" in body
        # Should have a non-empty request_id
        assert len(body["request_id"]) > 0

    def test_distributed_trace_preserved_through_full_stack(self):
        trace_id = "distributed-trace-99"
        resp = self.client.get("/ok", headers={CORRELATION_HEADER: trace_id})
        assert resp.status_code == 200
        assert resp.headers[CORRELATION_HEADER] == trace_id
        assert resp.json()["id"] == trace_id

    def test_error_with_distributed_trace(self):
        trace_id = "error-trace-77"
        resp = self.client.get("/fail", headers={CORRELATION_HEADER: trace_id})
        assert resp.status_code == 500
        body = resp.json()
        assert body["request_id"] == trace_id