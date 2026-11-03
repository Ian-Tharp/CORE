"""
Behavior tests for app/core/telemetry.py exporter selection.

No network: the OTLP exporter class is replaced with a fake (or made to fail),
and BatchSpanProcessor is never pointed at a real endpoint.
"""

from __future__ import annotations

import builtins
import logging
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import app.core.telemetry as telemetry

OTLP_MOD = "opentelemetry.exporter.otlp.proto.http.trace_exporter"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    telemetry.reset_telemetry()
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    yield
    provider = telemetry.get_provider()
    if provider is not None:
        provider.shutdown()
    telemetry.reset_telemetry()


def _processors(provider: TracerProvider):
    return list(provider._active_span_processor._span_processors)


class TestOtlpExporter:
    def test_endpoint_configures_batch_otlp_exporter_with_traces_path(
        self, monkeypatch, caplog
    ):
        import importlib

        mod = importlib.import_module(OTLP_MOD)
        fake_cls = MagicMock(return_value=InMemorySpanExporter())
        monkeypatch.setattr(mod, "OTLPSpanExporter", fake_cls)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318/")

        with caplog.at_level(logging.INFO, logger="app.core.telemetry"):
            provider = telemetry.setup_telemetry()

        # trailing slash stripped, /v1/traces appended
        fake_cls.assert_called_once_with(endpoint="http://collector:4318/v1/traces")
        procs = _processors(provider)
        assert len(procs) == 1
        assert type(procs[0]).__name__ == "BatchSpanProcessor"
        assert "OTLP exporter configured" in caplog.text
        assert telemetry.get_provider() is provider

    def test_exporter_construction_failure_degrades_to_local_spans(
        self, monkeypatch, caplog
    ):
        import importlib

        mod = importlib.import_module(OTLP_MOD)
        monkeypatch.setattr(
            mod, "OTLPSpanExporter", MagicMock(side_effect=RuntimeError("bad url"))
        )
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")

        with caplog.at_level(logging.WARNING, logger="app.core.telemetry"):
            provider = telemetry.setup_telemetry()

        assert isinstance(provider, TracerProvider)
        assert _processors(provider) == []
        assert "failed to configure OTLP exporter: bad url" in caplog.text
        # tracing still works (spans created, just not exported)
        with telemetry.get_tracer().start_as_current_span("s") as span:
            assert span.is_recording()

    def test_otlp_package_not_installed_degrades_gracefully(self, monkeypatch, caplog):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == OTLP_MOD:
                raise ImportError("No module named 'opentelemetry.exporter.otlp'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")

        with caplog.at_level(logging.WARNING, logger="app.core.telemetry"):
            provider = telemetry.setup_telemetry()

        assert _processors(provider) == []
        assert "failed to configure OTLP exporter" in caplog.text


class TestProviderLifecycle:
    def test_get_provider_is_none_until_lazy_init(self):
        assert telemetry.get_provider() is None
        tracer = telemetry.get_tracer()
        assert tracer is not None
        assert isinstance(telemetry.get_provider(), TracerProvider)

    def test_no_endpoint_means_no_processors(self):
        provider = telemetry.setup_telemetry()
        assert _processors(provider) == []

    def test_injected_exporter_takes_precedence_over_endpoint(self, monkeypatch):
        import importlib

        mod = importlib.import_module(OTLP_MOD)
        fake_cls = MagicMock()
        monkeypatch.setattr(mod, "OTLPSpanExporter", fake_cls)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
        exporter = InMemorySpanExporter()

        telemetry.setup_telemetry(exporter=exporter)
        with telemetry.get_tracer().start_as_current_span("injected"):
            pass

        fake_cls.assert_not_called()
        spans = exporter.get_finished_spans()
        assert [s.name for s in spans] == ["injected"]
        assert spans[0].resource.attributes["service.name"] == "core-backend"

    def test_reset_clears_state_and_resetup_gives_fresh_provider(self):
        first = telemetry.setup_telemetry()
        telemetry.reset_telemetry()
        assert telemetry.get_provider() is None
        telemetry.get_tracer()
        assert telemetry.get_provider() is not first
