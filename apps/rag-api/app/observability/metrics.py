"""Prometheus service/ingestion metrics independent of optional trace export."""

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

OPERATIONS = Counter("rag_operations", "Completed RAG operations.", ["operation", "outcome"])
DURATION = Histogram("rag_operation_duration_seconds", "RAG operation duration.", ["operation"],
                     buckets=(.01, .05, .1, .25, .5, 1, 2, 5, 10, 30, 60, 120))
ACTIVE = Gauge("rag_operations_active", "RAG operations in progress.", ["operation"])
HITS = Histogram("rag_search_hits", "Canonical hits per successful search.",
                 buckets=(0, 1, 3, 5, 8, 10, 20, 50))
DEGRADED = Counter("rag_degradations", "Searches with a degraded retrieval stage.", ["stage"])
DEPENDENCY = Gauge("rag_dependency_ready", "Last health check result, not active inference.",
                   ["dependency"])
HEALTH_CHECKED = Gauge("rag_health_checked_timestamp_seconds", "Last completed dependency check.")
INGEST_QUEUED = Gauge("rag_ingest_queued", "Upload jobs waiting for a worker.")
INGEST_ACTIVE = Gauge("rag_ingest_active", "Upload jobs currently being processed.")
INGEST_WORKERS = Gauge("rag_ingest_workers", "Running upload worker tasks.")
INGEST_JOBS = Counter("rag_ingest_jobs", "Finished upload jobs.", ["outcome", "stage"])
INGEST_SECONDS = Histogram("rag_ingest_duration_seconds", "Full upload processing duration.",
                           buckets=(1, 5, 15, 60, 300, 900, 3600))
logger = logging.getLogger("spec-rag.operations")
logger.propagate = False


def configure_logging(path: str | Path | None) -> None:
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
        "timestamp": time.time(), "service": "spec-rag", "event": event,
        "trace_id": f"{context.trace_id:032x}" if context.is_valid else None, **fields,
    }, ensure_ascii=False))


@dataclass
class Measurement:
    outcome: str = "ok"


@contextmanager
def operation(name: str) -> Iterator[Measurement]:
    measurement = Measurement()
    started = time.perf_counter()
    ACTIVE.labels(name).inc()
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
        ACTIVE.labels(name).dec()
        DURATION.labels(name).observe(seconds)
        OPERATIONS.labels(name, measurement.outcome).inc()
        log("operation.finished", operation=name, outcome=measurement.outcome, seconds=seconds)
