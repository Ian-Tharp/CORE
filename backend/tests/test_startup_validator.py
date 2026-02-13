"""
Tests for the startup configuration validator.

Validates that the startup validator correctly identifies missing required
env vars, format issues, and security misconfigurations.
"""

import os
from unittest.mock import patch

import pytest

from app.config.startup_validator import (
    ConfigIssue,
    Severity,
    _mask_value,
    _validate_bool,
    _validate_port,
    _validate_url,
    validate_startup_config,
)


# ─── Unit tests for individual validators ────────────────────────────────


class TestValidatePort:
    def test_valid_port(self):
        assert _validate_port("8080") is None

    def test_valid_port_boundaries(self):
        assert _validate_port("1") is None
        assert _validate_port("65535") is None

    def test_port_zero(self):
        assert _validate_port("0") is not None

    def test_port_too_high(self):
        assert _validate_port("70000") is not None

    def test_port_not_a_number(self):
        assert _validate_port("abc") is not None

    def test_port_negative(self):
        assert _validate_port("-1") is not None


class TestValidateUrl:
    def test_valid_http_url(self):
        assert _validate_url("http://localhost:11434") is None

    def test_valid_https_url(self):
        assert _validate_url("https://api.example.com/v1") is None

    def test_missing_scheme(self):
        assert _validate_url("localhost:11434") is not None

    def test_missing_host(self):
        assert _validate_url("http://") is not None


class TestValidateBool:
    @pytest.mark.parametrize("value", ["true", "false", "True", "FALSE", "1", "0", "yes", "no"])
    def test_valid_booleans(self, value):
        assert _validate_bool(value) is None

    @pytest.mark.parametrize("value", ["maybe", "2", "on", "off"])
    def test_invalid_booleans(self, value):
        assert _validate_bool(value) is not None


class TestMaskValue:
    def test_short_value_fully_masked(self):
        assert _mask_value("abc") == "****"

    def test_long_value_partially_masked(self):
        assert _mask_value("sk-1234567890") == "sk-1****"

    def test_exact_boundary(self):
        assert _mask_value("abcd") == "****"
        assert _mask_value("abcde") == "abcd****"


# ─── Integration tests for full validation ───────────────────────────────


class TestValidateStartupConfig:
    """Test the full validation pipeline with controlled env vars."""

    def test_clean_config_no_errors(self):
        """With reasonable defaults, no errors should be produced."""
        env = {
            "DB_HOST": "localhost",
            "DB_PORT": "5432",
            "DB_NAME": "core_db",
            "DB_USER": "core_user",
            "DB_PASSWORD": "secret",
            "REDIS_HOST": "localhost",
            "REDIS_PORT": "6379",
            "OLLAMA_BASE_URL": "http://localhost:11434",
            "CORE_API_KEY": "test-key-12345678",
            "CORE_ENV": "development",
        }
        with patch.dict(os.environ, env, clear=False):
            issues = validate_startup_config()

        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_invalid_port_produces_warning(self):
        """An invalid port value should produce a warning."""
        env = {
            "DB_PORT": "not-a-port",
        }
        with patch.dict(os.environ, env, clear=False):
            issues = validate_startup_config()

        port_issues = [i for i in issues if "DB_PORT" in i.message]
        assert len(port_issues) > 0
        assert port_issues[0].severity == Severity.WARN

    def test_invalid_url_produces_warning(self):
        """An invalid URL should produce a warning."""
        env = {
            "OLLAMA_BASE_URL": "not-a-url",
        }
        with patch.dict(os.environ, env, clear=False):
            issues = validate_startup_config()

        url_issues = [i for i in issues if "OLLAMA_BASE_URL" in i.message]
        assert len(url_issues) > 0

    def test_auth_disabled_in_production_is_error(self):
        """Auth disabled in production should be flagged as an error."""
        env = {
            "CORE_AUTH_DISABLED": "true",
            "CORE_ENV": "production",
        }
        with patch.dict(os.environ, env, clear=False):
            issues = validate_startup_config()

        security_errors = [
            i for i in issues
            if i.severity == Severity.ERROR and i.category == "security"
        ]
        assert len(security_errors) > 0

    def test_auth_disabled_in_dev_no_error(self):
        """Auth disabled in development should not be an error."""
        env = {
            "CORE_AUTH_DISABLED": "true",
            "CORE_ENV": "development",
        }
        with patch.dict(os.environ, env, clear=False):
            issues = validate_startup_config()

        security_errors = [
            i for i in issues
            if i.severity == Severity.ERROR and i.category == "security"
        ]
        assert len(security_errors) == 0

    def test_rate_limit_disabled_in_production_warns(self):
        """Rate limiting disabled in production should warn."""
        env = {
            "CORE_RATE_LIMIT_DISABLED": "true",
            "CORE_ENV": "production",
        }
        with patch.dict(os.environ, env, clear=False):
            issues = validate_startup_config()

        rate_issues = [
            i for i in issues
            if i.category == "security" and "rate" in i.message.lower()
        ]
        assert len(rate_issues) > 0
