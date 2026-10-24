"""Process-local Prometheus metrics and structured operational logs.

Task/trace IDs belong in logs, never in metric labels. Counters are updated at
execution boundaries and do not depend on enabling the optional tracing SDK.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path

from opentelemetry import trace
from prometheus_client import Counter, Gauge, Histogram

TASKS = Counter("agent_tasks", "Finished tasks, with independent delivery check status.",
                ["status", "verification"])
TASK_SECONDS = Histogram("agent_task_duration_seconds", "Full task and delivery check duration.",
                         buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1800, 3600))
ACTIVE_TASKS = Gauge("agent_tasks_active", "Tasks currently executing, including delivery checks.")
TASK_STARTED = Gauge("agent_task_started_timestamp_seconds", "Active task start; zero when idle.")
LAST_EVENT = Gauge("agent_last_event_timestamp_seconds", "Last observed runtime progress event.")
TOOL_CALLS = Counter("agent_tool_calls", "Actual dispatched tool calls.", ["tool", "outcome"])
TOOL_SECONDS = Histogram("agent_tool_duration_seconds", "Actual tool execution time.", ["tool"],
                         buckets=(.01, .1, .5, 1, 5, 15, 60, 300, 1800))
ACTIVE_TOOLS = Gauge("agent_tools_active", "Dispatched tools currently running.", ["tool"])
MODEL_CALLS = Counter("agent_model_requests", "Actual model attempts, including retries.",
                     ["model", "purpose", "outcome", "code"])
MODEL_SECONDS = Histogram("agent_model_duration_seconds", "Full streaming request duration.",
                          ["model", "purpose"], buckets=(.1, .5, 1, 3, 10, 30, 60, 120, 300))
MODEL_ACTIVE = Gauge("agent_model_requests_active", "Streaming model requests in progress.")
FIRST_EVENT = Histogram("agent_model_first_event_seconds", "Time to first parsed stream event.",
                        ["model"], buckets=(.1, .5, 1, 3, 10, 30, 60))
FIRST_TEXT = Histogram("agent_model_first_text_seconds", "Time to first answer text.",
                       ["model"], buckets=(.1, .5, 1, 3, 10, 30, 60, 120))
TOKENS = Counter("agent_model_tokens", "Reported usage; cache counts are subsets of input.",
                 ["model", "purpose", "type"])
USAGE_REPORTS = Counter("agent_model_usage_reports", "Requests with reported usage.", ["model"])
RETRIES = Counter("agent_model_retries", "Retries scheduled by runtime policy.", ["code"])
COMPACTIONS = Counter("agent_context_compactions", "Successfully applied context compactions.")
# 熔断：标签只放依赖键，不放任务/会话 ID。状态 0=closed 1=open 2=half_open。
CIRCUIT_STATE = Gauge("agent_circuit_state",
                      "Dependency circuit state: 0 closed, 1 open, 2 half-open.",
                      ["dependency"])
CIRCUIT_TRANSITIONS = Counter("agent_circuit_transitions_total",
                              "Dependency circuit state transitions.", ["dependency", "to"])
CIRCUIT_REJECTED = Counter("agent_circuit_rejected_total",
                           "Requests rejected before any network call by an open circuit.",
                           ["dependency"])
CIRCUIT_RECOVERY = Histogram("agent_circuit_recovery_seconds",
                             "Seconds from open to closed, for request-driven probes.",
                             ["dependency"], buckets=(1, 5, 15, 30, 60, 120, 300, 900))
#: 逻辑请求（进入过重试的请求）的终局，每条只记一次；分子分母口径见设计文档 §2.6。
RETRY_OUTCOMES = Counter("agent_retry_outcomes_total",
                         "Logical requests that entered retry, by terminal outcome.",
                         ["dependency", "outcome"])

logger = logging.getLogger("embedded-agent.operations")
logger.propagate = False


def configure_logging(path: str | None) -> None:
    if not path or logger.handlers:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(destination, maxBytes=10_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def log(event: str, **fields) -> None:
    context = trace.get_current_span().get_span_context()
    logger.info(json.dumps({
        "timestamp": time.time(), "service": "embedded-agent", "event": event,
        "trace_id": f"{context.trace_id:032x}" if context.is_valid else None, **fields,
    }, ensure_ascii=False))


@dataclass
class ToolMeasurement:
    outcome: str = "ok"


@contextmanager
def tool_call(name: str) -> Iterator[ToolMeasurement]:
    measurement = ToolMeasurement()
    started = time.perf_counter()
    ACTIVE_TOOLS.labels(name).inc()
    try:
        yield measurement
    except asyncio.CancelledError:
        measurement.outcome = "cancelled"
        raise
    except Exception:
        measurement.outcome = "error"
        raise
    finally:
        seconds = time.perf_counter() - started
        ACTIVE_TOOLS.labels(name).dec()
        TOOL_SECONDS.labels(name).observe(seconds)
        TOOL_CALLS.labels(name, measurement.outcome).inc()
        if measurement.outcome != "ok":
            log("tool.finished", tool=name, outcome=measurement.outcome, seconds=seconds)


class ModelMeasurement:
    def __init__(self, model: str, purpose: str) -> None:
        self.model, self.purpose = model, purpose
        self.started = time.perf_counter()
        self.outcome, self.code = "interrupted", ""
        self.usage = None
        MODEL_ACTIVE.inc()

    def finish(self) -> None:
        seconds = time.perf_counter() - self.started
        MODEL_ACTIVE.dec()
        MODEL_SECONDS.labels(self.model, self.purpose).observe(seconds)
        MODEL_CALLS.labels(self.model, self.purpose, self.outcome, self.code).inc()
        if self.usage is not None:
            USAGE_REPORTS.labels(self.model).inc()
            for kind, value in {
                "input": self.usage.input_tokens, "output": self.usage.output_tokens,
                "cache_read": self.usage.cached_input_tokens,
                "cache_miss": self.usage.uncached_input_tokens,
                "reasoning": self.usage.reasoning_tokens,
            }.items():
                if value is not None:
                    TOKENS.labels(self.model, self.purpose, kind).inc(value)
        log("model.finished", model=self.model, purpose=self.purpose, outcome=self.outcome,
            code=self.code, seconds=seconds)
