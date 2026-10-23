"""Embedding provider tests; every DashScope call uses ``MockTransport``."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from pydantic import SecretStr

from app.model_ai import (
    DashScopeEmbeddingError,
    DashScopeEmbeddingProvider,
    DeterministicHashEmbedding,
    EmbeddingProvider,
    create_embedding_provider,
    embedding_fingerprint,
)

_API_KEY = "test-only-dashscope-key"
_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"


def _run(coroutine):
    return asyncio.run(coroutine)


async def _embed_and_close(
    provider: DashScopeEmbeddingProvider, texts: list[str]
) -> list[list[float]]:
    try:
        return await provider.embed(texts)
    finally:
        await provider.aclose()


def _provider(
    handler,
    *,
    dimension: int = 2,
    batch_size: int = 16,
    max_retries: int = 0,
    backoff_base_s: float = 0.0,
) -> DashScopeEmbeddingProvider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return DashScopeEmbeddingProvider(
        api_key=SecretStr(_API_KEY),
        dimension=dimension,
        batch_size=batch_size,
        max_retries=max_retries,
        backoff_base_s=backoff_base_s,
        client=client,
    )


def _success(request: httpx.Request, vectors: list[list[float]]) -> httpx.Response:
    return httpx.Response(
        200,
        request=request,
        json={
            "object": "list",
            "model": "text-embedding-v3",
            "data": [
                {"object": "embedding", "index": index, "embedding": vector}
                for index, vector in enumerate(vectors)
            ],
        },
    )


def _float32_digest(vector: list[float]) -> str:
    payload = struct.pack(f"<{len(vector)}f", *vector)
    return hashlib.sha256(payload).hexdigest()


def test_deterministic_hash_dimension_normalization_and_canonicalization() -> None:
    provider = DeterministicHashEmbedding()
    vectors = _run(provider.embed(["ＡＢＣ   Foo", "abc foo", "中文规格 PWM1CR"]))

    assert isinstance(provider, EmbeddingProvider)
    assert provider.dimension == 1024
    assert provider.fingerprint == embedding_fingerprint(provider)
    assert provider.fingerprint == "deterministic-hash:rag-hash-v1:1024"
    assert vectors[0] == vectors[1]
    assert len(vectors) == 3
    for vector in vectors:
        assert len(vector) == 1024
        assert np.linalg.norm(np.asarray(vector, dtype=np.float64)) == pytest.approx(1.0)
        assert all(float(np.float32(component)) == component for component in vector)

    assert _run(provider.embed([])) == []
    _run(provider.aclose())


@pytest.mark.parametrize("text", ["", "  \t\r\n  "])
def test_deterministic_hash_rejects_empty_text(text: str) -> None:
    provider = DeterministicHashEmbedding()
    with pytest.raises(ValueError, match="must not be empty"):
        _run(provider.embed([text]))


def test_deterministic_hash_rejects_a_single_string_argument() -> None:
    provider = DeterministicHashEmbedding()
    with pytest.raises(TypeError, match="sequence of strings"):
        _run(provider.embed("not-a-sequence-for-this-api"))


def test_deterministic_hash_is_replayable_across_processes() -> None:
    rag_root = Path(__file__).resolve().parents[1]
    script = """
import asyncio
import hashlib
import struct
from app.model_ai import DeterministicHashEmbedding

vector = asyncio.run(DeterministicHashEmbedding().embed(["跨进程 PWM1CR replay"]))[0]
payload = struct.pack(f"<{len(vector)}f", *vector)
print(hashlib.sha256(payload).hexdigest())
"""
    digests: list[str] = []
    for python_hash_seed in ("1", "987654"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = python_hash_seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=rag_root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        digests.append(completed.stdout.strip())

    local = _run(DeterministicHashEmbedding().embed(["跨进程 PWM1CR replay"]))[0]
    assert digests == [_float32_digest(local), _float32_digest(local)]


def test_dashscope_openai_request_batching_order_and_normalization() -> None:
    observed: list[dict[str, object]] = []
    source_vectors = {
        "one": [3.0, 0.0, 0.0],
        "two": [0.0, 4.0, 0.0],
        "three": [0.0, 0.0, 5.0],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        observed.append(
            {
                "url": str(request.url),
                "authorization_ok": hmac.compare_digest(
                    request.headers.get("Authorization", ""), f"Bearer {_API_KEY}"
                ),
                "content_type": request.headers.get("Content-Type"),
                "body": body,
            }
        )
        data = [
            {
                "object": "embedding",
                "index": index,
                "embedding": source_vectors[text],
            }
            for index, text in reversed(list(enumerate(body["input"])))
        ]
        return httpx.Response(
            200,
            request=request,
            json={"object": "list", "data": data, "model": body["model"]},
        )

    provider = _provider(handler, dimension=3, batch_size=2)
    assert provider.model_id == "text-embedding-v3"
    assert provider.fingerprint == "text-embedding-v3:3"
    vectors = _run(_embed_and_close(provider, ["one", "two", "three"]))

    assert vectors == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert len(observed) == 2
    assert [entry["body"]["input"] for entry in observed] == [["one", "two"], ["three"]]
    for entry in observed:
        assert entry["url"] == _ENDPOINT
        assert entry["authorization_ok"] is True
        assert entry["content_type"] == "application/json"
        assert entry["body"]["model"] == "text-embedding-v3"
        assert entry["body"]["dimensions"] == 3


@pytest.mark.parametrize("first_status", [429, 503])
def test_dashscope_retries_429_and_5xx_with_exponential_backoff(
    first_status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("app.model_ai.embeddings.asyncio.sleep", fake_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(first_status, request=request, json={"error": "retry"})
        return _success(request, [[3.0, 4.0]])

    provider = _provider(handler, max_retries=2, backoff_base_s=0.25)
    result = _run(_embed_and_close(provider, ["retry me"]))

    assert result[0] == pytest.approx([0.6, 0.8])
    assert calls == 3
    assert delays == [0.25, 0.5]


def test_dashscope_retries_timeout() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("synthetic timeout", request=request)
        return _success(request, [[1.0, 0.0]])

    provider = _provider(handler, max_retries=1)
    assert _run(_embed_and_close(provider, ["retry timeout"])) == [[1.0, 0.0]]
    assert calls == 2


def test_dashscope_401_is_not_retried_and_never_exposes_key() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            401,
            request=request,
            json={"error": {"message": f"credential {_API_KEY} rejected"}},
        )

    provider = _provider(handler, max_retries=3)
    with pytest.raises(DashScopeEmbeddingError) as captured:
        _run(_embed_and_close(provider, ["unauthorized"]))

    assert calls == 1
    assert "status 401" in str(captured.value)
    assert _API_KEY not in str(captured.value)
    assert _API_KEY not in repr(provider)


def test_dashscope_final_timeout_never_exposes_transport_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(f"timeout containing {_API_KEY}", request=request)

    provider = _provider(handler)
    with pytest.raises(DashScopeEmbeddingError) as captured:
        _run(_embed_and_close(provider, ["timeout"]))

    assert "timed out" in str(captured.value)
    assert _API_KEY not in str(captured.value)


@pytest.mark.parametrize(
    ("texts", "data"),
    [
        (["one"], []),
        (["one"], [{"index": 0, "embedding": [1.0]}]),
        (["one"], [{"index": 0, "embedding": [1e39, 0.0]}]),
        (["one"], [{"index": 0, "embedding": [0.0, 0.0]}]),
        (
            ["one", "two"],
            [
                {"index": 0, "embedding": [1.0, 0.0]},
                {"index": 0, "embedding": [0.0, 1.0]},
            ],
        ),
    ],
)
def test_dashscope_rejects_malformed_embedding_responses(
    texts: list[str], data: list[dict[str, object]]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, json={"object": "list", "data": data})

    provider = _provider(handler)
    with pytest.raises(DashScopeEmbeddingError):
        _run(_embed_and_close(provider, texts))


def test_dashscope_rejects_invalid_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, content=b"not-json")

    provider = _provider(handler)
    with pytest.raises(DashScopeEmbeddingError, match="not valid JSON"):
        _run(_embed_and_close(provider, ["bad response"]))


def test_create_embedding_provider_selects_fixed_provider_and_fingerprint() -> None:
    deterministic = create_embedding_provider(
        SimpleNamespace(embedding_provider="deterministic", embedding_dimension=1024)
    )
    assert isinstance(deterministic, DeterministicHashEmbedding)
    assert embedding_fingerprint(deterministic) == "deterministic-hash:rag-hash-v1:1024"

    dashscope = create_embedding_provider(
        SimpleNamespace(
            embedding_provider="dashscope",
            embedding_model="text-embedding-v3",
            embedding_dimension=1024,
            embedding_batch_size=16,
            dashscope_api_key=SecretStr(_API_KEY),
        )
    )
    assert isinstance(dashscope, DashScopeEmbeddingProvider)
    assert embedding_fingerprint(dashscope) == "text-embedding-v3:1024"
    _run(dashscope.aclose())


def test_create_embedding_provider_never_falls_back() -> None:
    with pytest.raises(TypeError, match="SecretStr"):
        create_embedding_provider(
            SimpleNamespace(embedding_provider="dashscope", dashscope_api_key=None)
        )
    with pytest.raises(NotImplementedError):
        create_embedding_provider(SimpleNamespace(embedding_provider="hf"))
    with pytest.raises(ValueError, match="unsupported"):
        create_embedding_provider(SimpleNamespace(embedding_provider="unknown"))
