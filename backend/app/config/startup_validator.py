"""
Startup Configuration Validator

Validates environment variables and service configuration at startup.
Produces clear, actionable log output so misconfigurations are caught
early instead of surfacing as cryptic runtime errors.

Usage (called from lifespan in main.py):
    from app.config.startup_validator import validate_startup_config

    issues = validate_startup_config()
    # issues is a list of (level, message) tuples already logged
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ─── Severity levels ────────────────────────────────────────────────────


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


@dataclass
class ConfigIssue:
    """A single configuration issue found during validation."""

    severity: Severity
    category: str
    message: str


# ─── Environment variable definitions ───────────────────────────────────


@dataclass
class EnvVarSpec:
    """Specification for an environment variable."""

    name: str
    description: str
    required: bool = False
    default: Optional[str] = None
    validator: Optional[Callable[[str], Optional[str]]] = (
        None  # returns error msg or None
    )
    category: str = "general"
    sensitive: bool = False  # if True, value is masked in logs


def _validate_port(value: str) -> Optional[str]:
    """Validate that a value is a valid port number."""
    try:
        port = int(value)
        if not (1 <= port <= 65535):
            return f"Port {port} out of range (1-65535)"
    except ValueError:
        return f"'{value}' is not a valid port number"
    return None


def _validate_url(value: str) -> Optional[str]:
    """Validate that a value looks like a URL."""
    try:
        parsed = urlparse(value)
        if not parsed.scheme or not parsed.netloc:
            return f"'{value}' is not a valid URL (missing scheme or host)"
    except Exception:
        return f"'{value}' could not be parsed as a URL"
    return None


def _validate_bool(value: str) -> Optional[str]:
    """Validate that a value is a recognizable boolean."""
    if value.lower() not in ("true", "false", "1", "0", "yes", "no"):
        return f"'{value}' is not a recognizable boolean (use true/false)"
    return None


# All environment variables the application uses, grouped by category
ENV_VARS: List[EnvVarSpec] = [
    # ── Database ─────────────────────────────────────────────────────
    EnvVarSpec(
        name="DB_HOST",
        description="PostgreSQL host",
        default="postgres",
        category="database",
    ),
    EnvVarSpec(
        name="DB_PORT",
        description="PostgreSQL port",
        default="5432",
        validator=_validate_port,
        category="database",
    ),
    EnvVarSpec(
        name="DB_NAME",
        description="PostgreSQL database name",
        default="core_db",
        category="database",
    ),
    EnvVarSpec(
        name="DB_USER",
        description="PostgreSQL user",
        default="core_user",
        category="database",
    ),
    EnvVarSpec(
        name="DB_PASSWORD",
        description="PostgreSQL password",
        default="core_password",
        sensitive=True,
        category="database",
    ),
    # ── Redis ────────────────────────────────────────────────────────
    EnvVarSpec(
        name="REDIS_HOST",
        description="Redis host",
        default="redis",
        category="redis",
    ),
    EnvVarSpec(
        name="REDIS_PORT",
        description="Redis port",
        default="6379",
        validator=_validate_port,
        category="redis",
    ),
    # ── Ollama (local LLM) ──────────────────────────────────────────
    EnvVarSpec(
        name="OLLAMA_BASE_URL",
        description="Ollama service URL",
        default="http://ollama:11434",
        validator=_validate_url,
        category="ollama",
    ),
    # ── Authentication ──────────────────────────────────────────────
    EnvVarSpec(
        name="CORE_API_KEY",
        description="Primary API key for CORE endpoints",
        required=False,  # auto-generated if missing
        sensitive=True,
        category="auth",
    ),
    EnvVarSpec(
        name="CORE_API_KEY_2",
        description="Secondary API key (optional)",
        sensitive=True,
        category="auth",
    ),
    EnvVarSpec(
        name="CORE_AUTH_DISABLED",
        description="Disable API key auth (dev only)",
        validator=_validate_bool,
        category="auth",
    ),
    EnvVarSpec(
        name="CORE_RATE_LIMIT_DISABLED",
        description="Disable rate limiting (dev only)",
        validator=_validate_bool,
        category="auth",
    ),
    # ── Cloud API keys (optional) ───────────────────────────────────
    EnvVarSpec(
        name="OPENAI_API_KEY",
        description="OpenAI API key (for cloud models)",
        sensitive=True,
        category="cloud_apis",
    ),
    EnvVarSpec(
        name="ANTHROPIC_API_KEY",
        description="Anthropic API key (for cloud models)",
        sensitive=True,
        category="cloud_apis",
    ),
    # ── Application ─────────────────────────────────────────────────
    EnvVarSpec(
        name="CORE_ENV",
        description="Environment (development/staging/production)",
        default="development",
        category="application",
    ),
    EnvVarSpec(
        name="CORE_ALLOWED_ORIGINS",
        description="CORS allowed origins (comma-separated)",
        default="http://localhost:4200,http://localhost:8001",
        category="application",
    ),
    EnvVarSpec(
        name="PYTHONPATH",
        description="Python module search path",
        category="application",
    ),
]


def _mask_value(value: str) -> str:
    """Mask a sensitive value, showing only first 4 chars."""
    if len(value) <= 4:
        return "****"
    return value[:4] + "****"


def validate_startup_config() -> List[ConfigIssue]:
    """
    Validate all known configuration at startup.

    Checks environment variables, validates formats, and logs a clear
    configuration summary. Returns a list of issues found.

    Call this early in the application lifespan (before connecting to
    services) so operators see problems immediately.
    """
    issues: List[ConfigIssue] = []

    logger.info("=" * 60)
    logger.info("CORE Startup Configuration Validator")
    logger.info("=" * 60)

    # ── Group env vars by category for readable output ───────────────
    categories: Dict[str, List[EnvVarSpec]] = {}
    for spec in ENV_VARS:
        categories.setdefault(spec.category, []).append(spec)

    for category, specs in categories.items():
        logger.info("── %s ──", category.upper().replace("_", " "))

        for spec in specs:
            value = os.getenv(spec.name)
            effective = value or spec.default

            if value is None and spec.required:
                issue = ConfigIssue(
                    severity=Severity.ERROR,
                    category=category,
                    message=f"Required env var {spec.name} is not set ({spec.description})",
                )
                issues.append(issue)
                logger.error(
                    "  ✗ %s — MISSING (required): %s", spec.name, spec.description
                )
                continue

            if value is None and spec.default is None:
                # Optional, not set, no default — just note it
                logger.info("  · %s — not set (optional)", spec.name)
                continue

            # Validate format if a validator is provided
            if effective and spec.validator:
                error = spec.validator(effective)
                if error:
                    issue = ConfigIssue(
                        severity=Severity.WARN,
                        category=category,
                        message=f"{spec.name}: {error}",
                    )
                    issues.append(issue)
                    logger.warning(
                        "  ⚠ %s = %s — %s",
                        spec.name,
                        _mask_value(effective) if spec.sensitive else effective,
                        error,
                    )
                    continue

            # Show the effective value
            if spec.sensitive:
                display = _mask_value(effective) if effective else "(empty)"
            else:
                display = effective or "(empty)"

            source = "env" if value is not None else "default"
            logger.info("  ✓ %s = %s [%s]", spec.name, display, source)

    # ── Security warnings ────────────────────────────────────────────
    logger.info("── SECURITY CHECKS ──")

    auth_disabled = os.getenv("CORE_AUTH_DISABLED", "").lower() in ("true", "1", "yes")
    rate_limit_disabled = os.getenv("CORE_RATE_LIMIT_DISABLED", "").lower() in (
        "true",
        "1",
        "yes",
    )
    env = os.getenv("CORE_ENV", "development")

    if auth_disabled:
        if env == "production":
            issue = ConfigIssue(
                severity=Severity.ERROR,
                category="security",
                message="CORE_AUTH_DISABLED=true in production — API is unprotected!",
            )
            issues.append(issue)
            logger.error("  ✗ Authentication DISABLED in production!")
        else:
            logger.warning("  ⚠ Authentication disabled (CORE_AUTH_DISABLED=true)")

    if rate_limit_disabled:
        if env == "production":
            issue = ConfigIssue(
                severity=Severity.WARN,
                category="security",
                message="CORE_RATE_LIMIT_DISABLED=true in production",
            )
            issues.append(issue)
            logger.warning("  ⚠ Rate limiting DISABLED in production")
        else:
            logger.info("  · Rate limiting disabled (dev mode)")

    if not os.getenv("CORE_API_KEY"):
        logger.warning(
            "  ⚠ No CORE_API_KEY set — auto-generated key in use (not persisted)"
        )

    has_cloud = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
    if has_cloud:
        logger.info("  ✓ Cloud API keys configured")
    else:
        logger.info("  · No cloud API keys — using local models only")

    if not auth_disabled and not rate_limit_disabled:
        logger.info("  ✓ Auth and rate limiting enabled")

    # ── Summary ──────────────────────────────────────────────────────
    errors = sum(1 for i in issues if i.severity == Severity.ERROR)
    warnings = sum(1 for i in issues if i.severity == Severity.WARN)

    logger.info("── SUMMARY ──")
    logger.info("  Environment: %s", env)

    if errors:
        logger.error(
            "  %d error(s), %d warning(s) — fix errors before production use",
            errors,
            warnings,
        )
    elif warnings:
        logger.warning(
            "  %d warning(s) — review before production use",
            warnings,
        )
    else:
        logger.info("  ✓ All configuration checks passed")

    logger.info("=" * 60)

    return issues
