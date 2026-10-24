from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from opentelemetry.propagate import inject
from opentelemetry.trace import StatusCode

from app.dependencies import get_health_checker, get_rag_service
from app.main import create_app
from app.observability import otel
from app.rag_service import RagQueryService, RagUnavailableError
from app.routers.health import HealthChecker
from app.schemas import SearchHit


@pytest.fixture
def spans(monkeypatch):
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(otel, "tracer", provider.get_tracer("test"))
    yield exporter
    provider.shutdown()


def service_with_retrieval(retrieve):
    service = RagQueryService.__new__(RagQueryService)
    service._settings = SimpleNamespace(query_top_k_max=50)
    service._retrieve = retrieve
    return service


def test_search_continues_the_agent_trace_and_records_canonical_evidence(spans):
    hit = SearchHit(chunk_id=uuid4(), document_id=uuid4(), product="M8P515", version="V1.0",
                    section_path="8.7", page_start=12, page_end=13, kind="TEXT",
                    text="PRIVATE_MANUAL_TEXT", score=0.8, vector_score=0.8)

    async def retrieve(_):
        return [hit]

    application = create_app(SimpleNamespace(api_prefix="/api/rag", api_key=""))
    application.dependency_overrides[get_rag_service] = lambda: service_with_retrieval(retrieve)

    async def request():
        with otel.observe("agent.task") as parent:
            headers = {}
            inject(headers)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(application),
                                         base_url="http://rag") as client:
                response = await client.get("/api/rag/search", params={
                    "question": "PRIVATE_QUERY", "product": "M8P515", "version": "V1.0",
                }, headers=headers)
            return response, parent.get_span_context()

    response, parent = asyncio.run(request())
    assert response.status_code == 200 and response.json()[0]["text"] == hit.text
    records = spans.get_finished_spans()
    http = next(s for s in records if s.name == "rag.http")
    search = next(s for s in records if s.name == "rag.search")
    assert http.context.trace_id == parent.trace_id and http.parent.span_id == parent.span_id
    assert search.parent.span_id == http.context.span_id
    assert search.attributes["retrieval.documents.0.document.id"] == str(hit.chunk_id)
    assert search.attributes["rag.hit_count"] == 1
    assert "PRIVATE" not in str([dict(s.attributes) for s in records])


def test_search_failure_and_empty_results_have_different_outcomes(spans):
    calls = 0

    async def retrieve(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RagUnavailableError("retrieval unavailable")
        return []

    application = create_app(SimpleNamespace(api_prefix="/api/rag", api_key=""))
    application.dependency_overrides[get_rag_service] = lambda: service_with_retrieval(retrieve)

    async def requests():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(application),
                                     base_url="http://rag") as client:
            failed = await client.get("/api/rag/search?question=clock")
            empty = await client.get("/api/rag/search?question=clock")
            return failed, empty

    failed, empty = asyncio.run(requests())
    assert failed.status_code == 503 and empty.status_code == 200
    searches = [s for s in spans.get_finished_spans() if s.name == "rag.search"]
    assert searches[0].status.status_code == StatusCode.ERROR
    assert searches[1].status.status_code != StatusCode.ERROR
    assert searches[1].attributes["rag.hit_count"] == 0


def test_health_preserves_not_ready_and_does_not_create_model_traces(spans):
    settings = SimpleNamespace(api_prefix="/api/rag", api_key="")
    application = create_app(settings)
    application.dependency_overrides[get_health_checker] = lambda: HealthChecker(settings=settings)

    async def request():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(application),
                                     base_url="http://rag") as client:
            return await client.get("/api/rag/health")

    response = asyncio.run(request())
    assert response.status_code == 200 and response.json()["ready"] is False
    assert spans.get_finished_spans() == ()
