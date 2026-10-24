"""OpenTelemetry observation of the existing runtime; no execution policy.

DSH session-telemetry separates capture from SDK-owned delivery. OpenHands'
Telemetry/RootSpan keep request accounting inside a span's lifetime and attach
detached spans at async entry points. See docs/observability.md for pinned sources.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from . import metrics
from .contracts import (
    CompactionWarning,
    ContextCompacted,
    MemoryRecalled,
    ProviderRetryScheduled,
    ProviderRetryStarted,
    StepFinished,
    StepStarted,
)

tracer = trace.get_tracer("embedded-agent", "0.1.0")


def configure():
    """Enable the optional SDK only when an OTLP destination was configured."""
    if os.getenv("OTEL_SDK_DISABLED", "").lower() == "true" or not (
        os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    ):
        return None
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({
        "service.name": "embedded-agent", "service.version": "0.1.0",
        "openinference.project.name": "embedded-assistant",
    }), shutdown_on_exit=False)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    return provider


def current_trace_id() -> str | None:
    context = trace.get_current_span().get_span_context()
    return f"{context.trace_id:032x}" if context.is_valid else None


def failure(span: Span, code: str) -> None:
    span.set_attribute("error.type", code)
    span.set_status(Status(StatusCode.ERROR, code))


@contextmanager
def observe(name: str, kind: str = "CHAIN", **attributes) -> Iterator[Span]:
    with tracer.start_as_current_span(
        name, attributes={"openinference.span.kind": kind, **attributes},
        record_exception=False, set_status_on_exception=False,
    ) as span:
        try:
            yield span
        except asyncio.CancelledError:
            span.set_attribute("agent.outcome", "cancelled")
            raise
        except Exception as exc:
            failure(span, type(exc).__name__)
            raise


def observed(name: str, kind: str = "CHAIN"):
    """For ordinary async functions; streaming generators attach per iteration."""
    def decorate(function):
        @wraps(function)
        async def wrapped(*args, **kwargs):
            with observe(name, kind):
                return await function(*args, **kwargs)
        return wrapped
    return decorate


def record_usage(span: Span, usage) -> None:
    attributes = {
        "llm.token_count.prompt": usage.input_tokens,
        "llm.token_count.completion": usage.output_tokens,
        "llm.token_count.total": usage.total_tokens,
        "gen_ai.usage.input_tokens": usage.input_tokens,
        "gen_ai.usage.output_tokens": usage.output_tokens,
        "llm.token_count.prompt_details.cache_read": usage.cached_input_tokens,
        "gen_ai.usage.cache_read.input_tokens": usage.cached_input_tokens,
        "gen_ai.usage.uncached_input_tokens": usage.uncached_input_tokens,
        "llm.token_count.completion_details.reasoning": usage.reasoning_tokens,
    }
    span.set_attributes({key: value for key, value in attributes.items() if value is not None})


def record_event(event) -> None:
    """Record lifecycle facts, without exporting messages, memory or tool bodies."""
    metrics.LAST_EVENT.set(time.time())
    if isinstance(event, ContextCompacted):
        metrics.COMPACTIONS.inc()
    elif isinstance(event, ProviderRetryScheduled):
        metrics.RETRIES.labels(event.code or "unknown").inc()
    if not isinstance(event, (
        StepStarted, StepFinished, MemoryRecalled, ContextCompacted, CompactionWarning,
        ProviderRetryScheduled, ProviderRetryStarted,
    )):
        return
    attributes = {"agent.turn": event.turn, "agent.step": event.step}
    if isinstance(event, MemoryRecalled):
        attributes["memory.item_count"] = len(event.items)
    elif isinstance(event, ContextCompacted):
        attributes.update({"context.tokens_before": event.context_tokens,
                           "context.compacted_messages": event.compacted_messages})
        if event.context_tokens_after is not None:
            attributes["context.tokens_after"] = event.context_tokens_after
    elif isinstance(event, CompactionWarning):
        attributes.update({"context.tokens": event.context_tokens,
                           "context.threshold": event.threshold_tokens,
                           "context.attempts": event.attempts})
    elif isinstance(event, ProviderRetryScheduled):
        attributes.update({"retry.attempt": event.attempt, "retry.delay_seconds": event.delay,
                           "retry.error_code": event.code})
    elif isinstance(event, ProviderRetryStarted):
        attributes["retry.attempt"] = event.attempt
    trace.get_current_span().add_event(type(event).__name__, attributes)
