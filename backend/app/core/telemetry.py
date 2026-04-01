"""
CORE OpenTelemetry Instrumentation

Provides span-based tracing for the LangGraph cognitive pipeline.
Exports to OTLP HTTP if OTEL_EXPORTER_OTLP_ENDPOINT is set;
otherwise creates spans locally (no export).

Implementation note: module-level TracerProvider is used directly rather
than the global OTel API so that tests can inject a fresh provider per run
without hitting the "TracerProvider may only be set once" restriction.

Usage:
    from app.core.telemetry import get_tracer, setup_telemetry

    # At startup (optional — auto-initialises lazily if skipped):
    setup_telemetry()

    # In any node:
    tracer = get_tracer()
    with tracer.start_as_current_span("core.comprehension") as span:
        span.set_attribute("node.name", "comprehension")
        ...
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
)

logger = logging.getLogger(__name__)

_SERVICE_NAME = "core-backend"
_TRACER_NAME = "core.pipeline"

# Module-level provider and tracer (lazily initialised, replaceable for tests)
_provider: Optional[TracerProvider] = None
_tracer: Optional[trace.Tracer] = None


def setup_telemetry(exporter: Optional[SpanExporter] = None) -> TracerProvider:
    """
    Initialise OpenTelemetry tracing.

    Uses the module-level provider (not the global OTel singleton) so tests
    can inject a fresh InMemorySpanExporter per run without hitting the
    "TracerProvider may only be set once" restriction.

    Args:
        exporter: Optional span exporter — used in tests to inject an
                  InMemorySpanExporter. In production, auto-detects from
                  OTEL_EXPORTER_OTLP_ENDPOINT env var.

    Returns:
        The configured TracerProvider.
    """
    global _provider, _tracer

    from opentelemetry.sdk.resources import Resource

    resource = Resource.create({"service.name": _SERVICE_NAME})
    provider = TracerProvider(resource=resource)

    if exporter is not None:
        # Injected exporter (tests use SimpleSpanProcessor for synchronous export)
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        logger.debug("OTel: using injected exporter (%s)", type(exporter).__name__)
    else:
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )
                otlp = OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
                provider.add_span_processor(BatchSpanProcessor(otlp))
                logger.info("OTel: OTLP exporter configured → %s", endpoint)
            except Exception as exc:
                logger.warning("OTel: failed to configure OTLP exporter: %s", exc)
        else:
            logger.debug(
                "OTel: OTEL_EXPORTER_OTLP_ENDPOINT not set — spans created but not exported"
            )

    # Store on the module — do NOT call trace.set_tracer_provider() to avoid
    # the global singleton restriction that prevents resetting between tests.
    _provider = provider
    _tracer = provider.get_tracer(_TRACER_NAME)
    return provider


def get_tracer() -> trace.Tracer:
    """Return the module tracer, initialising lazily if needed."""
    global _tracer
    if _tracer is None:
        setup_telemetry()
    return _tracer


def get_provider() -> Optional[TracerProvider]:
    """Return the current provider (useful for tests / teardown)."""
    return _provider


def reset_telemetry() -> None:
    """
    Reset module state. Used in tests to ensure a clean provider per test.
    Does NOT touch the global OTel singleton.
    """
    global _provider, _tracer
    _provider = None
    _tracer = None
