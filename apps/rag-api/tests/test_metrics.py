from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from prometheus_client import REGISTRY

from app.dependencies import get_health_checker
from app.main import _UploadTaskRunner, create_app
from app.rag_service import RagQueryService, RagUnavailableError
from app.routers.health import HealthChecker
from app.schemas import SearchParams


def sample(name, labels=None):
    return REGISTRY.get_sample_value(name, labels or {}) or 0


def test_empty_search_is_success_and_failure_is_not_an_empty_hit():
    service = RagQueryService.__new__(RagQueryService)
    service._settings = SimpleNamespace(query_top_k_max=50)

    async def retrieve(_):
        return []

    service._retrieve = retrieve
    error_before = sample("rag_operations_total", {"operation": "rag.search", "outcome": "error"})
    empty_before = sample("rag_search_hits_bucket", {"le": "0.0"})
    params = SearchParams(question="PRIVATE_QUERY", top_k=5)
    assert asyncio.run(service.search(params)) == []

    async def unavailable(_):
        raise RagUnavailableError("unavailable")

    service._retrieve = unavailable
    with pytest.raises(RagUnavailableError):
        asyncio.run(service.search(params))
    assert sample("rag_search_hits_bucket", {"le": "0.0"}) - empty_before == 1
    assert sample("rag_operations_total", {
        "operation": "rag.search", "outcome": "error",
    }) - error_before == 1
    assert sample("rag_operations_active", {"operation": "rag.search"}) == 0


def test_multiple_app_factories_have_metrics_and_false_health_stays_false():
    async def request():
        for _ in range(2):
            settings = SimpleNamespace(api_prefix="/api/rag", api_key="")
            application = create_app(settings)
            application.dependency_overrides[get_health_checker] = lambda current=settings: (
                HealthChecker(settings=current)
            )
            async with httpx.AsyncClient(transport=httpx.ASGITransport(application),
                                         base_url="http://rag") as client:
                result = await client.get("/api/rag/health")
                assert result.status_code == 200 and result.json()["ready"] is False
                await client.get("/api/rag/search")  # Query validation error, not a model call.
                scraped = await client.get("/metrics")
                assert scraped.status_code == 200
                assert 'rag_dependency_ready{dependency="postgres"} 0.0' in scraped.text
                assert 'handler="/api/rag/search"' in scraped.text
                assert 'handler="/api/rag/health"' not in scraped.text
                assert 'handler="/metrics"' not in scraped.text

    asyncio.run(request())


def test_swallowed_ingest_failure_is_still_recorded_as_failure(tmp_path):
    async def unavailable(_):
        raise RuntimeError("PRIVATE_DATABASE_ERROR")

    async def set_status(*args, **kwargs):
        pass

    runner = _UploadTaskRunner(
        repository=SimpleNamespace(get_document=unavailable, set_status=set_status),
        provider=None, store=None, storage=SimpleNamespace(), settings=SimpleNamespace(),
    )
    labels = {"outcome": "error", "stage": "load"}
    before = sample("rag_ingest_jobs_total", labels)
    asyncio.run(runner._process(uuid4(), tmp_path / "missing.pdf"))
    assert sample("rag_ingest_jobs_total", labels) - before == 1
    assert sample("rag_ingest_active") == 0
