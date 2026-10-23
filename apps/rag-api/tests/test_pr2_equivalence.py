from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from test_api_routers import _LLM, _settings

from app.generation.citations import build_citation
from app.rag_service import RagQueryService
from app.schemas import QueryChecks, QueryRequest, SearchHit

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests/fixtures/pr2_equivalence/golden.json"
QUERY_CONTRACT = ROOT / "tests/fixtures/pr2_equivalence/query_contract.json"
GOLDEN_SHA256 = "0b96d94ed56f6f7b112c36de84cf0ceeb31738916cc731c6324f74a2dbc173c0"
QUERY_CASES = (
    "normal",
    "detector_failure",
    "generation_disabled",
    "generation_unavailable",
    "ordered_degradation",
    "citation_failure",
)
pytestmark = pytest.mark.filterwarnings("ignore:Field name .* shadows")


def _fixture() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _service() -> RagQueryService:
    service = RagQueryService.__new__(RagQueryService)
    service._settings = SimpleNamespace(**_fixture()["settings"])
    return service


def _citations(service, payload):
    return [
        service._citation(SearchHit.model_validate(item), item["text"])
        for item in payload["hits"]
    ]


def test_fixture_is_frozen_and_covers_required_edges() -> None:
    assert hashlib.sha256(GOLDEN.read_bytes()).hexdigest() == GOLDEN_SHA256
    payload = _fixture()
    hits = payload["hits"]
    assert hits[0]["rerank_score"] == 0.0
    assert hits[0]["page_start"] != hits[0]["page_end"]
    assert payload["expected_checks"]["evidence"]
    assert payload["expected_checks"]["answer"]


def test_citation_adapter_matches_frozen_golden_field_by_field() -> None:
    payload = _fixture()
    service = _service()
    actual = _citations(service, payload)
    expected = payload["expected_citations"]
    assert [item.model_dump(mode="json") for item in actual] == expected


def test_query_checks_adapter_matches_frozen_golden_and_order() -> None:
    payload = _fixture()
    service = _service()
    citations = _citations(service, payload)
    actual = service._query_checks(citations, payload["generation_return"]["answer"])
    assert actual.model_dump(mode="json") == payload["expected_checks"]


def test_detector_exception_keeps_fail_soft_empty_checks(monkeypatch) -> None:
    payload = _fixture()
    service = _service()
    citations = _citations(service, payload)

    def boom(_citations):
        raise RuntimeError("detector exploded")

    monkeypatch.setattr("app.rag_service.detect_findings", boom)
    actual = service._query_checks(citations, payload["generation_return"]["answer"])
    assert actual == QueryChecks()


def test_citation_construction_exception_is_preserved() -> None:
    with pytest.raises(AttributeError):
        build_citation(object(), "text")  # type: ignore[arg-type]
    with pytest.raises(AttributeError):
        RagQueryService._citation(object(), "text")  # type: ignore[arg-type]


def query_case_actual(case: str) -> dict:
    """Exercise real query/selection/context/adapters with a fixed retrieval boundary.

    Reuse the existing API tests' settings and LLM stub. The response expectations
    are explicit contracts added after extraction, not an older query recording.
    """

    payload = _fixture()
    contract = json.loads(QUERY_CONTRACT.read_text(encoding="utf-8"))
    hits = [SearchHit.model_validate(item) for item in payload["hits"]]
    if contract["duplicate_first_hit"]:
        hits.append(hits[0].model_copy(deep=True))
    llm = _LLM()

    async def fixed_generation(question, contexts):
        if case == "generation_unavailable":
            raise RuntimeError("generation fixture failure")
        result = await _LLM.generate(llm, question, contexts)
        result.answer = payload["generation_return"]["answer"]
        result.model_id = payload["generation_return"]["model_id"]
        result.degraded = case == "ordered_degradation"
        return result

    llm.generate = AsyncMock(side_effect=fixed_generation)
    settings = {
        **payload["settings"],
        "planner_enabled": False,
        "context_neighbor_window": 0,
        "llm_enabled": case != "generation_disabled",
        "reranker_enabled": case == "ordered_degradation",
        "generation_failure_mode": "retrieval-only",
    }
    reranker = SimpleNamespace(rerank=AsyncMock(side_effect=RuntimeError("rerank fixture failure")))
    service = RagQueryService(llm_provider=llm, reranker=reranker, settings=_settings(**settings))
    service._retrieve_queries = AsyncMock(return_value=hits)
    detector_patch = (
        patch("app.rag_service.detect_findings", side_effect=RuntimeError("detector exploded"))
        if case == "detector_failure"
        else nullcontext()
    )
    citation_patch = (
        patch(
            "app.generation.citations.Citation",
            side_effect=AttributeError("citation fixture failure"),
        )
        if case == "citation_failure"
        else nullcontext()
    )
    with (
        patch.object(service, "_build_contexts", wraps=service._build_contexts) as context_builder,
        detector_patch,
        citation_patch,
    ):
        try:
            response = asyncio.run(service.query(QueryRequest.model_validate(contract["request"])))
        except AttributeError as exc:
            actual = {"error": {"type": type(exc).__name__, "message": str(exc)}}
        else:
            actual = {"response": response.model_dump(mode="json", exclude={"latency_ms"})}
    actual["selected_chunk_ids"] = [
        str(hit.chunk_id) for hit in context_builder.call_args.args[0]
    ]
    actual["generation_calls"] = llm.generate.await_count
    return actual


@pytest.mark.parametrize("case", QUERY_CASES)
def test_query_response_matches_explicit_contract(case: str) -> None:
    contract = json.loads(QUERY_CONTRACT.read_text(encoding="utf-8"))
    assert contract["provenance"]["adapter_golden_sha256"] == GOLDEN_SHA256
    assert query_case_actual(case) == contract["cases"][case]
