from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from uuid import UUID

import httpx
import pytest

from app.model_ai.reranker import (
    DashScopeReranker,
    NoopReranker,
    RerankerConfigurationError,
    RerankerError,
    RerankInput,
    create_reranker,
)

API_KEY = "dashscope-test-key-that-must-not-leak"


class _SecretStr:
    """Small pydantic.SecretStr-compatible test double."""

    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "SecretStr('**********')"


@dataclass
class _Settings:
    dashscope_api_key: object = _SecretStr(API_KEY)
    reranker_enabled: bool = True
    reranker_model: str = "gte-rerank-v2"
    reranker_failure_mode: str = "degrade"
    hf_timeout_seconds: float = 0.1
    hf_max_retries: int = 0


def _candidates() -> tuple[list[RerankInput], list[object]]:
    chunks = [object(), object(), object()]
    return (
        [
            RerankInput("a", chunks[0], "alpha", 0.91),
            RerankInput("b", chunks[1], "beta", 0.82),
            RerankInput("c", chunks[2], "gamma", 0.73),
        ],
        chunks,
    )


def test_noop_and_disabled_factory_preserve_original_order_and_binding() -> None:
    async def scenario() -> None:
        candidates, chunks = _candidates()
        reranker = create_reranker(object())

        assert isinstance(reranker, NoopReranker)
        results = await reranker.rerank("question", candidates, top_n=2)

        assert [result.id for result in results] == ["a", "b"]
        assert [result.rerank_score for result in results] == [0.91, 0.82]
        assert [result.original_index for result in results] == [0, 1]
        assert results[0].chunk is chunks[0]
        assert results[1].chunk is chunks[1]
        assert not any(result.degraded for result in results)

    asyncio.run(scenario())


def test_structural_service_candidates_keep_uuid_and_original_object() -> None:
    @dataclass(frozen=True)
    class ServiceCandidate:
        candidate_id: UUID
        text: str
        vector_score: float

    async def scenario() -> None:
        candidate = ServiceCandidate(
            candidate_id=UUID("00000000-0000-0000-0000-000000000001"),
            text="canonical chunk",
            vector_score=0.75,
        )
        results = await NoopReranker().rerank(
            "question",
            [candidate],  # type: ignore[list-item]
            top_n=1,
        )

        assert results[0].id == candidate.candidate_id
        assert results[0].candidate_id == candidate.candidate_id
        assert results[0].chunk is candidate
        assert results[0].rerank_score == 0.75

    asyncio.run(scenario())


def test_dashscope_200_maps_indexes_to_original_chunks() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["Authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "output": {
                    "results": [
                        {"index": 0, "relevance_score": 0.41},
                        {"index": 2, "relevance_score": 0.97},
                    ]
                }
            },
        )

    async def scenario() -> None:
        candidates, chunks = _candidates()
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            reranker = DashScopeReranker(_Settings(), client=client)
            results = await reranker.rerank("  which document?  ", candidates, top_n=2)

        assert [result.id for result in results] == ["c", "a"]
        assert [result.original_index for result in results] == [2, 0]
        assert results[0].chunk is chunks[2]
        assert results[1].chunk is chunks[0]
        assert [result.rerank_score for result in results] == [0.97, 0.41]
        assert not any(result.degraded for result in results)

    asyncio.run(scenario())

    assert captured["url"] == (
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/"
        "text-rerank/text-rerank"
    )
    assert captured["authorization"] == f"Bearer {API_KEY}"
    assert captured["body"] == {
        "model": "gte-rerank-v2",
        "input": {
            "query": "which document?",
            "documents": ["alpha", "beta", "gamma"],
        },
        "parameters": {"top_n": 2},
    }


def test_dashscope_429_retries_with_backoff() -> None:
    calls = 0
    delays: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0.5"}, json={"message": API_KEY})
        return httpx.Response(
            200,
            json={"output": {"results": [{"index": 1, "relevance_score": 0.88}]}},
        )

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    async def scenario() -> None:
        candidates, chunks = _candidates()
        settings = _Settings(hf_max_retries=1)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = DashScopeReranker(settings, client=client, sleep=fake_sleep)
            results = await reranker.rerank("question", candidates, top_n=1)
        assert results[0].id == "b"
        assert results[0].chunk is chunks[1]

    asyncio.run(scenario())

    assert calls == 2
    assert delays == [0.5]


def test_401_explicitly_degrades_without_losing_original_binding(
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": f"rejected {API_KEY}"})

    async def scenario() -> None:
        candidates, chunks = _candidates()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = DashScopeReranker(_Settings(), client=client)
            results = await reranker.rerank("question", candidates, top_n=2)

        assert [result.id for result in results] == ["a", "b"]
        assert [result.original_index for result in results] == [0, 1]
        assert results[0].chunk is chunks[0]
        assert results[1].chunk is chunks[1]
        assert all(result.degraded for result in results)
        assert {result.degradation_reason for result in results} == {"http_401"}

    asyncio.run(scenario())

    output = capsys.readouterr()
    assert API_KEY not in output.out
    assert API_KEY not in output.err


def test_timeout_retries_then_degrades_in_original_order() -> None:
    calls = 0
    delays: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout(f"timeout contains {API_KEY}", request=request)

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    async def scenario() -> None:
        candidates, chunks = _candidates()
        settings = _Settings(hf_max_retries=1)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = DashScopeReranker(settings, client=client, sleep=fake_sleep)
            results = await reranker.rerank("question", candidates, top_n=3)

        assert [result.id for result in results] == ["a", "b", "c"]
        assert [result.chunk for result in results] == chunks
        assert all(result.degradation_reason == "timeout" for result in results)

    asyncio.run(scenario())

    assert calls == 2
    assert delays == [0.25]


def test_error_mode_and_configuration_errors_never_expose_key() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"upstream echoed {API_KEY}")

    async def scenario() -> None:
        candidates, _ = _candidates()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = DashScopeReranker(_Settings(), client=client, failure_mode="error")
            with pytest.raises(RerankerError) as captured:
                await reranker.rerank("question", candidates, top_n=1)
        assert API_KEY not in str(captured.value)
        assert API_KEY not in repr(captured.value)

    asyncio.run(scenario())

    settings = _Settings(dashscope_api_key=API_KEY)
    with pytest.raises(RerankerConfigurationError) as captured:
        DashScopeReranker(settings)
    assert API_KEY not in str(captured.value)
