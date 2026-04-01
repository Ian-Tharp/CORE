"""
Tests for app/core/logging_config.py — JSONFormatter and setup_logging.

Sentinel-value contract:
- Remove timestamp key from JSONFormatter → structured log has no time → alerting breaks → test fails
- Remove exception formatting → errors log without stack traces → debugging impossible → test fails
- Remove correlation_id injection → request tracing lost → test fails
- Remove production JSON formatter → structured logs corrupted in prod → test fails
- Remove noisy-library suppression → httpx DEBUG floods logs → test fails
"""

from __future__ import annotations

import json
import logging
import pytest
from unittest.mock import patch

from app.core.logging_config import JSONFormatter, setup_logging


# ===========================================================================
# Helpers
# ===========================================================================

def _make_record(
    message: str = "SENTINEL_MESSAGE",
    level: int = logging.INFO,
    name: str = "SENTINEL_LOGGER",
    **extra
) -> logging.LogRecord:
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname="test_file.py",
        lineno=42,
        msg=message,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


# ===========================================================================
# JSONFormatter.format — required keys
# ===========================================================================

class TestJSONFormatterRequiredKeys:
    def setup_method(self):
        self.fmt = JSONFormatter()

    def test_output_is_valid_json(self):
        """
        SENTINEL — format() must return parseable JSON.
        Return plain text → JSON consumer crashes → test fails.
        """
        record = _make_record()
        output = self.fmt.format(record)
        parsed = json.loads(output)  # raises if not valid JSON
        assert isinstance(parsed, dict)

    def test_timestamp_present(self):
        """
        SENTINEL — output must include 'timestamp'.
        Omit it → log aggregator can't order events → test fails.
        """
        record = _make_record()
        parsed = json.loads(self.fmt.format(record))
        assert "timestamp" in parsed

    def test_level_present(self):
        """
        SENTINEL — output must include 'level'.
        Omit it → alerting rules can't filter by severity → test fails.
        """
        record = _make_record(level=logging.WARNING)
        parsed = json.loads(self.fmt.format(record))
        assert parsed["level"] == "WARNING"

    def test_logger_name_present(self):
        """output must include 'logger' with the logger's name."""
        record = _make_record(name="SENTINEL_LOGGER_NAME")
        parsed = json.loads(self.fmt.format(record))
        assert parsed["logger"] == "SENTINEL_LOGGER_NAME"

    def test_message_present(self):
        """
        SENTINEL — output must include 'message'.
        Omit it → log line has no content → test fails.
        """
        record = _make_record(message="SENTINEL_LOG_MSG")
        parsed = json.loads(self.fmt.format(record))
        assert parsed["message"] == "SENTINEL_LOG_MSG"

    def test_module_present(self):
        """output must include 'module'."""
        record = _make_record()
        parsed = json.loads(self.fmt.format(record))
        assert "module" in parsed

    def test_function_present(self):
        """output must include 'function'."""
        record = _make_record()
        parsed = json.loads(self.fmt.format(record))
        assert "function" in parsed

    def test_line_present(self):
        """output must include 'line'."""
        record = _make_record()
        parsed = json.loads(self.fmt.format(record))
        assert "line" in parsed
        assert isinstance(parsed["line"], int)


# ===========================================================================
# JSONFormatter — optional extra fields
# ===========================================================================

class TestJSONFormatterOptionalFields:
    def setup_method(self):
        self.fmt = JSONFormatter()

    def test_correlation_id_included_when_present(self):
        """
        SENTINEL — correlation_id on the record must appear in JSON output.
        Skip correlation_id → distributed tracing fails → test fails.
        """
        record = _make_record(correlation_id="SENTINEL_CORR_ID")
        parsed = json.loads(self.fmt.format(record))
        assert parsed["correlation_id"] == "SENTINEL_CORR_ID"

    def test_request_id_included_when_present(self):
        """
        SENTINEL — request_id on the record must appear in JSON output.
        Skip request_id → per-request log correlation lost → test fails.
        """
        record = _make_record(request_id="SENTINEL_REQ_ID")
        parsed = json.loads(self.fmt.format(record))
        assert parsed["request_id"] == "SENTINEL_REQ_ID"

    def test_no_correlation_id_key_when_absent(self):
        """When record has no correlation_id, the key must be absent (no None noise)."""
        record = _make_record()
        parsed = json.loads(self.fmt.format(record))
        assert "correlation_id" not in parsed

    def test_exception_included_when_exc_info(self):
        """
        SENTINEL — exception info must be included in 'exception' key.
        Omit exc_info → stack traces lost from JSON logs → test fails.
        """
        try:
            raise ValueError("SENTINEL_EXCEPTION_MSG")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="something failed",
            args=(),
            exc_info=exc_info,
        )
        parsed = json.loads(self.fmt.format(record))
        assert "exception" in parsed
        assert "SENTINEL_EXCEPTION_MSG" in parsed["exception"]


# ===========================================================================
# setup_logging — environment-driven configuration
# ===========================================================================

class TestSetupLogging:
    def test_production_uses_json_formatter(self):
        """
        SENTINEL — CORE_ENV=production must install JSONFormatter on the root handler.
        Use text formatter in production → structured log pipeline gets plain text → test fails.
        """
        with patch.dict("os.environ", {"CORE_ENV": "production", "CORE_LOG_LEVEL": "INFO"}):
            setup_logging()

        root = logging.getLogger()
        assert len(root.handlers) > 0
        assert isinstance(root.handlers[0].formatter, JSONFormatter)

        # Restore default for test isolation
        root.handlers.clear()

    def test_development_uses_text_formatter(self):
        """
        SENTINEL — CORE_ENV != 'production' must NOT install JSONFormatter.
        Force JSON in dev → human-readable logs lost → developer experience broken → test fails.
        """
        with patch.dict("os.environ", {"CORE_ENV": "development", "CORE_LOG_LEVEL": "DEBUG"}):
            setup_logging()

        root = logging.getLogger()
        assert len(root.handlers) > 0
        assert not isinstance(root.handlers[0].formatter, JSONFormatter)

        root.handlers.clear()

    def test_log_level_set_from_env(self):
        """
        SENTINEL — CORE_LOG_LEVEL must control the root logger level.
        Ignore env var → always DEBUG → noisy production logs → test fails.
        """
        with patch.dict("os.environ", {"CORE_ENV": "development", "CORE_LOG_LEVEL": "WARNING"}):
            setup_logging()

        assert logging.getLogger().level == logging.WARNING
        logging.getLogger().handlers.clear()

    def test_noisy_libraries_suppressed(self):
        """
        SENTINEL — httpx, httpcore, asyncio loggers must be set to WARNING after setup.
        Leave at DEBUG → megabytes of HTTP wire-level logs in dev → test fails.
        """
        with patch.dict("os.environ", {"CORE_ENV": "development", "CORE_LOG_LEVEL": "DEBUG"}):
            setup_logging()

        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING
        assert logging.getLogger("asyncio").level == logging.WARNING

        logging.getLogger().handlers.clear()

    def test_existing_handlers_cleared(self):
        """
        SENTINEL — setup_logging must clear existing handlers before adding new one.
        Append only → duplicate handlers → every log message printed multiple times → test fails.
        """
        root = logging.getLogger()
        # Manually add a handler before calling setup
        dummy = logging.StreamHandler()
        root.addHandler(dummy)

        with patch.dict("os.environ", {"CORE_ENV": "development", "CORE_LOG_LEVEL": "INFO"}):
            setup_logging()

        # Must be exactly one handler (the new one)
        assert len(root.handlers) == 1

        root.handlers.clear()
