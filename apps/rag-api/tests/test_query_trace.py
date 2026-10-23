from __future__ import annotations

import json
import time
from pathlib import Path

from app.observability.trace import QueryTrace, TraceSink, span


def test_query_trace_records_stages_and_counts() -> None:
    trace = QueryTrace()
    started = time.perf_counter()
    trace.record("retrieve", started, hits=7)
    trace.record("generate", started, called=True, answer_chars=12)
    payload = trace.to_dict()
    assert payload["trace_id"] == trace.trace_id
    assert [stage["name"] for stage in payload["stages"]] == ["retrieve", "generate"]
    assert payload["stages"][0]["hits"] == 7
    assert payload["stages"][1]["answer_chars"] == 12


def test_trace_sink_appends_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "queries.jsonl"
    trace = QueryTrace()
    trace.record("retrieve", time.perf_counter(), hits=1)
    TraceSink(path).write(trace)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["trace_id"] == trace.trace_id
    assert rows[0]["stages"][0]["name"] == "retrieve"


def test_trace_sink_is_fail_soft() -> None:
    # A directory can never be opened as a JSONL file; tracing must not raise.
    TraceSink(Path.cwd()).write(QueryTrace())


def test_span_records_even_when_block_raises() -> None:
    trace = QueryTrace()
    try:
        with span(trace, "select", selected=3):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert trace.to_dict()["stages"][0]["selected"] == 3
