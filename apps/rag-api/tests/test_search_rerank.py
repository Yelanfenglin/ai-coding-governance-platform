from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from app.dependencies import get_rag_service
from app.main import create_app
from app.rag_service import RagQueryService
from app.schemas import SearchHit, SearchParams


class Reranker:
    def __init__(self, mode="rank"):
        self.mode = mode
        self.calls = []

    async def rerank(self, question, candidates, top_n):
        self.calls.append((question, candidates, top_n))
        if self.mode == "error":
            raise RuntimeError("upstream unavailable")
        if self.mode == "cancel":
            raise asyncio.CancelledError
        if self.mode == "empty":
            return []
        return [
            SimpleNamespace(id=hit.id, rerank_score=1 - index / 100,
                            degraded=self.mode == "degraded")
            for index, hit in enumerate(reversed(candidates))
        ][:top_n]


def fixture(*, mode="rank", max_k=50, empty=False):
    settings = SimpleNamespace(api_prefix="/api/rag", api_key="", query_top_k_max=max_k,
                               reranker_enabled=mode != "off", context_top_n=1)
    reranker = Reranker(mode)
    service = RagQueryService(settings=settings, reranker=reranker)
    document_id = uuid4()
    hits = [SearchHit(
        chunk_id=uuid4(), document_id=document_id, product="M8P5170", version="V1.6",
        section_path=f"5.{index}", page_start=index + 1, page_end=index + 1,
        kind="TABLE", text=f"canonical evidence {index}", score=1 - index / 10,
        vector_score=1 - index / 10,
    ) for index in range(6)]
    requests = []

    async def retrieve(request):
        requests.append(request)
        return [] if empty else hits[:request.top_k]

    service._retrieve = retrieve
    app = create_app(settings)
    app.dependency_overrides[get_rag_service] = lambda: service
    return app, service, hits, requests, reranker


async def search(app, **params):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app),
                                 base_url="http://rag") as client:
        return await client.get("/api/rag/search", params={"question": "interrupt gates", **params})


def test_agent_search_can_promote_evidence_outside_the_return_budget():
    app, _, hits, requests, reranker = fixture()
    response = asyncio.run(search(app, product="M8P5170", version="1.06", kind="TABLE",
                                  top_k=2, candidate_k=6, rerank=True))
    assert response.status_code == 200
    result = response.json()
    assert [hit["chunk_id"] for hit in result] == [str(hit.chunk_id) for hit in hits[-2:][::-1]]
    assert requests[0].top_k == 6
    assert requests[0].product == "M8P5170" and requests[0].version == "V1.6"
    assert requests[0].kind.value == "TABLE"
    assert reranker.calls[0][2] == 2  # Independent of generation context_top_n=1.
    for received, original in zip(result, hits[-2:][::-1], strict=True):
        for key in ("document_id", "product", "version", "text", "page_start", "section_path"):
            assert str(received[key]) == str(getattr(original, key))
        assert received["vector_score"] == original.vector_score
        assert received["rerank_score"] is not None


def test_search_default_does_not_add_rerank_calls_or_widen_recall():
    app, _, hits, requests, reranker = fixture()
    response = asyncio.run(search(app, top_k=2))
    assert [hit["chunk_id"] for hit in response.json()] == [str(hit.chunk_id) for hit in hits[:2]]
    assert requests[0].top_k == 2
    assert reranker.calls == []


@pytest.mark.parametrize("mode", ["error", "empty", "degraded"])
def test_reranker_failure_keeps_canonical_evidence_and_score_provenance(mode):
    app, _, hits, _, _ = fixture(mode=mode)
    response = asyncio.run(search(app, top_k=2, candidate_k=6, rerank=True))
    assert response.status_code == 200
    assert [hit["chunk_id"] for hit in response.json()] == [str(hit.chunk_id) for hit in hits[:2]]
    assert all(hit["rerank_score"] is None for hit in response.json())


@pytest.mark.parametrize("params", [
    {"top_k": 3, "candidate_k": 2, "rerank": True},
    {"candidate_k": 30},
    {"candidate_k": 51, "rerank": True},
])
def test_invalid_budgets_fail_before_retrieval(params):
    app, _, _, requests, reranker = fixture()
    response = asyncio.run(search(app, **params))
    assert response.status_code == 422
    assert requests == [] and reranker.calls == []


def test_unconfigured_reranker_is_explicit_before_retrieval():
    app, _, _, requests, reranker = fixture(mode="off")
    response = asyncio.run(search(app, rerank=True))
    assert response.status_code == 422
    assert "not configured" in response.json()["detail"]
    assert requests == [] and reranker.calls == []


def test_default_candidate_budget_respects_server_ceiling():
    app, _, _, requests, _ = fixture(max_k=10)
    assert asyncio.run(search(app, top_k=2, rerank=True)).status_code == 200
    assert requests[0].top_k == 10
    requests.clear()
    assert asyncio.run(search(app, top_k=2, candidate_k=11, rerank=True)).status_code == 422
    assert requests == []


def test_empty_search_does_not_call_reranker():
    app, _, _, _, reranker = fixture(empty=True)
    response = asyncio.run(search(app, rerank=True))
    assert response.status_code == 200 and response.json() == []
    assert reranker.calls == []


def test_cancellation_is_not_downgraded_to_success():
    _, service, _, _, _ = fixture(mode="cancel")
    params = SearchParams(question="interrupt gates", top_k=2, rerank=True)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(service.search(params))
    assert params.top_k == 2
