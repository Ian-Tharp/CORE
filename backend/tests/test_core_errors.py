"""
Tests for CORE error utilities — exception hierarchy, response serialisation,
handle_core_errors decorator, and core_exception_handler global handler.

Sentinel-value contract:
- Remove code=f"{resource_type.upper()}_NOT_FOUND" → NotFoundError code is wrong → test fails
- Remove field injection in ValidationError → field lost from details → test fails
- Remove retry_after from RateLimitError message → callers don't know when to retry → test fails
- Remove run_id/node from ExecutionError details → debugging context lost → test fails
- Remove COREError branch in handle_core_errors → COREError raises 500 instead of its status → test fails
- Remove HTTPException re-raise → HTTPException becomes 500 → test fails
- Remove correlation_id extraction → correlation missing from error response → test fails
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import HTTPException, status, Request

from app.utils.errors import (
    COREError,
    NotFoundError,
    ValidationError as COREValidationError,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    ServiceUnavailableError,
    ExecutionError,
    ErrorResponse,
    handle_core_errors,
    core_exception_handler,
)


# ===========================================================================
# COREError — base exception
# ===========================================================================


class TestCOREError:
    def test_defaults(self):
        """Default code is CORE_ERROR, status 500."""
        err = COREError("something went wrong")
        assert err.message == "something went wrong"
        assert err.code == "CORE_ERROR"
        assert err.status_code == 500
        assert err.details == {}

    def test_custom_fields(self):
        """Custom code, status, and details are stored correctly."""
        err = COREError(
            "bad input", code="BAD_INPUT", status_code=400, details={"field": "x"}
        )
        assert err.code == "BAD_INPUT"
        assert err.status_code == 400
        assert err.details == {"field": "x"}

    def test_is_exception(self):
        """COREError is an Exception and can be raised/caught."""
        with pytest.raises(COREError, match="test error"):
            raise COREError("test error")

    def test_to_response_includes_all_fields(self):
        """
        SENTINEL — to_response must populate error, code, details.
        Return empty ErrorResponse → callers lose error context → test fails.
        """
        err = COREError(
            "SENTINEL_MSG", code="SENTINEL_CODE", details={"key": "SENTINEL_DETAIL"}
        )
        resp = err.to_response(correlation_id="SENTINEL_CORR")
        assert isinstance(resp, ErrorResponse)
        assert resp.error == "SENTINEL_MSG"
        assert resp.code == "SENTINEL_CODE"
        assert resp.details == {"key": "SENTINEL_DETAIL"}
        assert resp.correlation_id == "SENTINEL_CORR"

    def test_to_response_no_correlation_id(self):
        """to_response without correlation_id has None for that field."""
        err = COREError("msg")
        resp = err.to_response()
        assert resp.correlation_id is None


# ===========================================================================
# NotFoundError
# ===========================================================================


class TestNotFoundError:
    def test_code_derived_from_resource_type(self):
        """
        SENTINEL — code must be '{TYPE}_NOT_FOUND' from resource_type.
        Hardcode code='NOT_FOUND' → all 404s look the same → test fails.
        """
        err = NotFoundError("Agent", "agent-42")
        assert err.code == "AGENT_NOT_FOUND"

    def test_code_uppercased(self):
        """Resource type in code must be upper-cased."""
        err = NotFoundError("knowledge_item", "kid-1")
        assert err.code == "KNOWLEDGE_ITEM_NOT_FOUND"

    def test_status_is_404(self):
        """NotFoundError must have HTTP 404 status."""
        err = NotFoundError("Channel", "ch-1")
        assert err.status_code == 404

    def test_message_includes_type_and_id(self):
        """Message must contain both the resource type and ID."""
        err = NotFoundError("SENTINEL_TYPE", "SENTINEL_ID")
        assert "SENTINEL_TYPE" in err.message
        assert "SENTINEL_ID" in err.message

    def test_auto_details_include_resource_type_and_id(self):
        """
        SENTINEL — details must include resource_type and resource_id automatically.
        Omit auto-details → callers can't identify which resource is missing → test fails.
        """
        err = NotFoundError("Agent", "agent-99")
        assert err.details["resource_type"] == "Agent"
        assert err.details["resource_id"] == "agent-99"

    def test_custom_details_override_auto(self):
        """Explicit details override the auto-generated ones."""
        err = NotFoundError("Agent", "agent-1", details={"custom": "SENTINEL"})
        assert err.details == {"custom": "SENTINEL"}


# ===========================================================================
# ValidationError (COREValidationError)
# ===========================================================================


class TestCOREValidationError:
    def test_code_is_validation_error(self):
        """Code must be VALIDATION_ERROR."""
        err = COREValidationError("bad field")
        assert err.code == "VALIDATION_ERROR"

    def test_status_is_400(self):
        """ValidationError must have HTTP 400 status."""
        err = COREValidationError("invalid")
        assert err.status_code == 400

    def test_field_in_details_when_provided(self):
        """
        SENTINEL — field name must appear in details.
        Skip field injection → callers can't tell which field is invalid → test fails.
        """
        err = COREValidationError("too long", field="SENTINEL_FIELD")
        assert err.details.get("field") == "SENTINEL_FIELD"

    def test_no_field_results_in_empty_details(self):
        """Without field, details must be empty dict."""
        err = COREValidationError("generic error")
        assert err.details == {}


# ===========================================================================
# AuthenticationError / AuthorizationError
# ===========================================================================


class TestAuthErrors:
    def test_authentication_error_defaults(self):
        """AuthenticationError defaults to 401 with AUTHENTICATION_ERROR code."""
        err = AuthenticationError()
        assert err.status_code == 401
        assert err.code == "AUTHENTICATION_ERROR"

    def test_authorization_error_defaults(self):
        """AuthorizationError defaults to 403 with AUTHORIZATION_ERROR code."""
        err = AuthorizationError()
        assert err.status_code == 403
        assert err.code == "AUTHORIZATION_ERROR"


# ===========================================================================
# RateLimitError
# ===========================================================================


class TestRateLimitError:
    def test_message_includes_retry_after(self):
        """
        SENTINEL — message must include the retry_after seconds value.
        Omit it → callers don't know when to retry → test fails.
        """
        err = RateLimitError(retry_after=42)
        assert "42" in err.message

    def test_status_is_429(self):
        """RateLimitError must have HTTP 429 status."""
        err = RateLimitError()
        assert err.status_code == 429

    def test_details_include_retry_after(self):
        """
        SENTINEL — details must include retry_after for programmatic access.
        Omit details → clients can't auto-back-off → test fails.
        """
        err = RateLimitError(retry_after=120)
        assert err.details["retry_after"] == 120

    def test_code_is_rate_limit_exceeded(self):
        """Code must be RATE_LIMIT_EXCEEDED."""
        err = RateLimitError()
        assert err.code == "RATE_LIMIT_EXCEEDED"


# ===========================================================================
# ServiceUnavailableError
# ===========================================================================


class TestServiceUnavailableError:
    def test_default_message_includes_service_name(self):
        """Default message must include the service name."""
        err = ServiceUnavailableError("SENTINEL_SERVICE")
        assert "SENTINEL_SERVICE" in err.message

    def test_status_is_503(self):
        """ServiceUnavailableError must have HTTP 503 status."""
        err = ServiceUnavailableError("db")
        assert err.status_code == 503

    def test_custom_message_overrides_default(self):
        """Explicit message must be used instead of the default."""
        err = ServiceUnavailableError("db", message="SENTINEL_CUSTOM_MSG")
        assert "SENTINEL_CUSTOM_MSG" in err.message

    def test_details_include_service_name(self):
        """Auto-details must include service name for debugging."""
        err = ServiceUnavailableError("redis")
        assert err.details["service"] == "redis"


# ===========================================================================
# ExecutionError
# ===========================================================================


class TestExecutionError:
    def test_run_id_in_details(self):
        """
        SENTINEL — run_id must appear in details.
        Omit run_id → can't correlate error to specific pipeline run → test fails.
        """
        err = ExecutionError("failed", run_id="SENTINEL_RUN_ID")
        assert err.details["run_id"] == "SENTINEL_RUN_ID"

    def test_node_in_details(self):
        """
        SENTINEL — node name must appear in details.
        Omit node → can't tell which graph node failed → test fails.
        """
        err = ExecutionError("failed", node="SENTINEL_NODE")
        assert err.details["node"] == "SENTINEL_NODE"

    def test_both_run_id_and_node(self):
        """Both run_id and node can coexist in details."""
        err = ExecutionError("failed", run_id="run-1", node="reasoning")
        assert err.details["run_id"] == "run-1"
        assert err.details["node"] == "reasoning"

    def test_status_is_500(self):
        """ExecutionError must have HTTP 500 status."""
        err = ExecutionError("failed")
        assert err.status_code == 500

    def test_code_is_execution_error(self):
        """Code must be EXECUTION_ERROR."""
        err = ExecutionError("failed")
        assert err.code == "EXECUTION_ERROR"


# ===========================================================================
# handle_core_errors decorator
# ===========================================================================


class TestHandleCoreErrors:
    @pytest.mark.asyncio
    async def test_core_error_converts_to_http_exception(self):
        """
        SENTINEL — COREError must become HTTPException with correct status.
        Remove COREError branch → COREError raises 500 regardless of status_code → test fails.
        """

        @handle_core_errors
        async def endpoint():
            raise NotFoundError("Agent", "agent-1")

        with pytest.raises(HTTPException) as exc_info:
            await endpoint()

        assert exc_info.value.status_code == 404
        assert "AGENT_NOT_FOUND" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_http_exception_passes_through(self):
        """
        SENTINEL — HTTPException must be re-raised unchanged.
        Catch HTTPException → callers lose the original 403 status → test fails.
        """

        @handle_core_errors
        async def endpoint():
            raise HTTPException(status_code=403, detail="forbidden")

        with pytest.raises(HTTPException) as exc_info:
            await endpoint()

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_unexpected_exception_becomes_500(self):
        """Unexpected Exception must become a 500 HTTPException."""

        @handle_core_errors
        async def endpoint():
            raise RuntimeError("unexpected")

        with pytest.raises(HTTPException) as exc_info:
            await endpoint()

        assert exc_info.value.status_code == 500
        assert "INTERNAL_ERROR" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_success_returns_value(self):
        """Decorator must not interfere with successful functions."""

        @handle_core_errors
        async def endpoint():
            return {"ok": True}

        result = await endpoint()
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_correlation_id_extracted_from_request_arg(self):
        """
        SENTINEL — correlation_id from request.state must appear in error detail.
        Skip extraction → correlation_id is None in every error response → test fails.
        """
        req = MagicMock(spec=Request)
        req.state.correlation_id = "SENTINEL_CORR_ID"

        @handle_core_errors
        async def endpoint(request: Request):
            raise COREError("test", code="TEST_CODE")

        with pytest.raises(HTTPException) as exc_info:
            await endpoint(req)

        # correlation_id appears in the detail dict
        detail = exc_info.value.detail
        assert detail.get("correlation_id") == "SENTINEL_CORR_ID"

    @pytest.mark.asyncio
    async def test_core_error_detail_includes_code_and_message(self):
        """Error detail dict must include error message and code."""

        @handle_core_errors
        async def endpoint():
            raise COREError("SENTINEL_MESSAGE", code="SENTINEL_CODE")

        with pytest.raises(HTTPException) as exc_info:
            await endpoint()

        detail = exc_info.value.detail
        assert detail.get("error") == "SENTINEL_MESSAGE"
        assert detail.get("code") == "SENTINEL_CODE"


# ===========================================================================
# core_exception_handler
# ===========================================================================


class TestCoreExceptionHandler:
    @pytest.mark.asyncio
    async def test_returns_json_response_with_correct_status(self):
        """
        SENTINEL — handler must return JSONResponse with exc.status_code.
        Return 500 always → HTTP clients see wrong status → test fails.
        """
        req = MagicMock(spec=Request)
        req.state.correlation_id = None
        exc = NotFoundError("Agent", "agent-1")

        resp = await core_exception_handler(req, exc)

        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_response_body_includes_error_and_code(self):
        """Response body must include error message and code."""
        req = MagicMock(spec=Request)
        req.state.correlation_id = None
        exc = COREError("SENTINEL_ERR", code="SENTINEL_CD")

        resp = await core_exception_handler(req, exc)

        import json

        body = json.loads(resp.body)
        assert body["error"] == "SENTINEL_ERR"
        assert body["code"] == "SENTINEL_CD"

    @pytest.mark.asyncio
    async def test_correlation_id_included_when_available(self):
        """
        SENTINEL — correlation_id from request.state must appear in JSON body.
        Skip extraction → distributed tracing broken → test fails.
        """
        req = MagicMock(spec=Request)
        req.state.correlation_id = "SENTINEL_TRACE"
        exc = COREError("err")

        resp = await core_exception_handler(req, exc)

        import json

        body = json.loads(resp.body)
        assert body.get("correlation_id") == "SENTINEL_TRACE"
