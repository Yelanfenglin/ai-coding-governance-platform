"""In-memory query tracing with an optional JSONL sink.

PR5 is intentionally observation-only: recording never changes the answer, the
selected hits, the checks, or the degradation state.  A sink failure is ignored.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class TraceStage:
    name: str
    duration_ms: int
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueryTrace:
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    stages: list[TraceStage] = field(default_factory=list)

    def record(self, name: str, started: float, **data: Any) -> None:
        self.stages.append(
            TraceStage(
                name=name,
                duration_ms=max(0, round((time.perf_counter() - started) * 1000)),
                data=data,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "started_at": self.started_at,
            "stages": [
                {
                    "name": stage.name,
                    "duration_ms": stage.duration_ms,
                    **stage.data,
                }
                for stage in self.stages
            ],
        }


class TraceSink:
    """Append-only JSONL sink; never raises into the request path."""

    def __init__(self, path: Path | str | None) -> None:
        self.path = Path(path) if path else None

    def write(self, trace: QueryTrace) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
        except Exception:
            # Tracing is advisory; a failed sink must not fail a query.
            return


@contextmanager
def span(trace: QueryTrace, name: str, **data: Any) -> Iterator[None]:
    """Record a stage even when the wrapped block raises."""

    started = time.perf_counter()
    try:
        yield
    finally:
        trace.record(name, started, **data)
