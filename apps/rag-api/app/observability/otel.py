"""OTLP spans for the actual search/context path, alongside optional JSONL traces."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps

from opentelemetry import trace
from opentelemetry.propagate import extract
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from . import metrics

tracer = trace.get_tracer("spec-rag", "0.1.0")


def configure(endpoint: str | None = None):
    if os.getenv("OTEL_SDK_DISABLED", "").lower() == "true" or not (
        endpoint or os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    ):
        return None
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({
        "service.name": "spec-rag", "service.version": "0.1.0",
        "openinference.project.name": "embedded-assistant",
    }), shutdown_on_exit=False)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    return provider


@contextmanager
def observe(name: str, kind: str = "CHAIN", **attributes) -> Iterator[Span]:
    with metrics.operation(name), tracer.start_as_current_span(
        name, attributes={"openinference.span.kind": kind, **attributes},
        record_exception=False, set_status_on_exception=False,
    ) as span:
        try:
            yield span
        except asyncio.CancelledError:
            span.set_attribute("agent.outcome", "cancelled")
            raise
        except Exception as exc:
            span.set_attribute("error.type", type(exc).__name__)
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise


def observed(name: str, kind: str = "CHAIN"):
    def decorate(function):
        @wraps(function)
        async def wrapped(*args, **kwargs):
            with observe(name, kind):
                return await function(*args, **kwargs)
        return wrapped
    return decorate


async def trace_request(request, call_next):
    # Health probes must not fill the Agent trace list with monitoring traffic.
    if request.url.path.endswith(("/health", "/metrics")):
        return await call_next(request)
    with tracer.start_as_current_span(
        "rag.http", context=extract(request.headers), kind=SpanKind.SERVER,
        record_exception=False, set_status_on_exception=False,
        attributes={"http.request.method": request.method, "openinference.span.kind": "CHAIN"},
    ) as span:
        try:
            response = await call_next(request)
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise
        span.set_attribute("http.response.status_code", response.status_code)
        route = request.scope.get("route")
        if route is not None:
            span.set_attribute("http.route", route.path)
        if response.status_code >= 400:
            span.set_status(Status(StatusCode.ERROR, f"http_{response.status_code}"))
        return response
