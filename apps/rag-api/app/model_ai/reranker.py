"""DashScope reranking adapters with citation-safe candidate mapping."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable
from uuid import UUID

import httpx

DASHSCOPE_RERANK_BASE_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)
DEFAULT_RERANKER_MODEL = "gte-rerank-v2"

type RerankerFailureMode = Literal["error", "degrade"]
type Sleep = Callable[[float], Awaitable[None]]


class SecretValue(Protocol):
    """The subset of pydantic.SecretStr used by these adapters."""

    def get_secret_value(self) -> str: ...


class RerankerSettings(Protocol):
    """Settings fields consumed by :class:`DashScopeReranker`."""

    dashscope_api_key: SecretValue
    reranker_model: str
    hf_timeout_seconds: float
    hf_max_retries: int


@dataclass(frozen=True, slots=True)
class RerankInput:
    """A rerankable document and its opaque canonical chunk binding.

    ``chunk`` is never reconstructed from the remote response. The response's
    integer index is resolved back to this exact object, which prevents a score
    or citation from being attached to a different chunk.
    """

    id: str | UUID
    chunk: object
    text: str
    rerank_score: float

    @property
    def candidate_id(self) -> str | UUID:
        return self.id

    @property
    def chunk_id(self) -> str | UUID:
        return self.id

    @property
    def vector_score(self) -> float:
        return self.rerank_score


@dataclass(frozen=True, slots=True)
class RerankResult:
    """A ranked candidate whose id/chunk pair came from one input entry."""

    id: str | UUID
    chunk: object
    rerank_score: float
    original_index: int
    degraded: bool = False
    degradation_reason: str | None = None

    @property
    def index(self) -> int:
        return self.original_index

    @property
    def candidate_id(self) -> str | UUID:
        return self.id


@runtime_checkable
class Reranker(Protocol):
    @property
    def model_id(self) -> str: ...

    async def rerank(
        self,
        question: str,
        candidates: Sequence[RerankInput],
        top_n: int,
    ) -> list[RerankResult]: ...


class RerankerError(RuntimeError):
    """A safe-to-report reranker failure.

    Messages contain only a bounded failure category. In particular, neither
    request headers nor response bodies are copied into the exception.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"DashScope rerank failed ({reason})")


class RerankerConfigurationError(RerankerError):
    """Raised when an enabled reranker has invalid or missing settings."""


def _setting(settings: object, name: str, default: object) -> object:
    if isinstance(settings, Mapping):
        return settings.get(name, default)
    return getattr(settings, name, default)


def _secret_from_settings(settings: object) -> str:
    secret = _setting(settings, "dashscope_api_key", None)
    getter = getattr(secret, "get_secret_value", None)
    if not callable(getter):
        raise RerankerConfigurationError("api_key_not_configured") from None
    try:
        value = getter()
    except Exception:
        readable = False
        value = None
    else:
        readable = True
    if not readable:
        raise RerankerConfigurationError("api_key_not_readable") from None
    if not isinstance(value, str) or not value or value != value.strip():
        raise RerankerConfigurationError("api_key_not_configured") from None
    return value


def _positive_float(value: object, *, reason: str) -> float:
    if isinstance(value, bool):
        raise RerankerConfigurationError(reason)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise RerankerConfigurationError(reason) from None
    if not math.isfinite(parsed) or parsed <= 0:
        raise RerankerConfigurationError(reason)
    return parsed


def _non_negative_float(value: object, *, reason: str) -> float:
    if isinstance(value, bool):
        raise RerankerConfigurationError(reason)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise RerankerConfigurationError(reason) from None
    if not math.isfinite(parsed) or parsed < 0:
        raise RerankerConfigurationError(reason)
    return parsed


def _non_negative_int(value: object, *, reason: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RerankerConfigurationError(reason)
    return value


def _failure_mode(value: object) -> RerankerFailureMode:
    if value not in {"error", "degrade"}:
        raise RerankerConfigurationError("invalid_failure_mode")
    return value


_MISSING = object()


def _member(value: object, *names: str, default: object = _MISSING) -> object:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _normalize_candidate(candidate: object, index: int) -> RerankInput:
    if isinstance(candidate, RerankInput):
        return candidate

    candidate_id = _member(candidate, "id", "candidate_id", "chunk_id")
    text = _member(candidate, "text", "content")
    score = _member(candidate, "rerank_score", "vector_score", "score")
    if candidate_id is _MISSING or text is _MISSING or score is _MISSING:
        raise TypeError(f"candidate at index {index} does not satisfy RerankInput")
    chunk = _member(candidate, "chunk", default=candidate)
    return RerankInput(
        id=candidate_id,  # type: ignore[arg-type]
        chunk=chunk,
        text=text,  # type: ignore[arg-type]
        rerank_score=score,  # type: ignore[arg-type]
    )


def _prepare_request(
    question: str,
    candidates: Sequence[RerankInput],
    top_n: int,
) -> tuple[str, tuple[RerankInput, ...], int]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer")

    prepared = tuple(
        _normalize_candidate(candidate, index) for index, candidate in enumerate(candidates)
    )
    for candidate in prepared:
        if not isinstance(candidate.id, (str, UUID)) or not str(candidate.id):
            raise ValueError("candidate id must be a non-empty string or UUID")
        if not isinstance(candidate.text, str) or not candidate.text.strip():
            raise ValueError("candidate text must be a non-empty string")
        score = candidate.rerank_score
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("candidate rerank_score must be finite")
        if not math.isfinite(float(score)):
            raise ValueError("candidate rerank_score must be finite")
    return question.strip(), prepared, min(top_n, len(prepared))


def _original_order_results(
    candidates: Sequence[RerankInput],
    limit: int,
    *,
    degraded: bool,
    reason: str | None = None,
) -> list[RerankResult]:
    return [
        RerankResult(
            id=candidate.id,
            chunk=candidate.chunk,
            rerank_score=float(candidate.rerank_score),
            original_index=index,
            degraded=degraded,
            degradation_reason=reason,
        )
        for index, candidate in enumerate(candidates[:limit])
    ]


class NoopReranker:
    """A disabled reranker that keeps the retrieval order and scores."""

    @property
    def model_id(self) -> str:
        return "noop"

    async def rerank(
        self,
        question: str,
        candidates: Sequence[RerankInput],
        top_n: int,
    ) -> list[RerankResult]:
        _, prepared, limit = _prepare_request(question, candidates, top_n)
        return _original_order_results(prepared, limit, degraded=False)


class DashScopeReranker:
    """Call DashScope's native text-rerank API with bounded retries."""

    def __init__(
        self,
        settings: RerankerSettings | object,
        *,
        client: httpx.AsyncClient | None = None,
        failure_mode: RerankerFailureMode | None = None,
        timeout_s: float | None = None,
        max_retries: int | None = None,
        backoff_base_s: float = 0.25,
        max_backoff_s: float = 8.0,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        model = _setting(settings, "reranker_model", DEFAULT_RERANKER_MODEL)
        if not isinstance(model, str) or not model.strip():
            raise RerankerConfigurationError("invalid_model")
        self._model = model.strip()
        self._api_key = _secret_from_settings(settings)

        configured_timeout = _setting(
            settings,
            "dashscope_timeout_seconds",
            _setting(settings, "hf_timeout_seconds", 30.0),
        )
        configured_retries = _setting(
            settings,
            "dashscope_max_retries",
            _setting(settings, "hf_max_retries", 3),
        )
        configured_failure_mode = _setting(settings, "reranker_failure_mode", "degrade")

        self._timeout_s = _positive_float(
            configured_timeout if timeout_s is None else timeout_s,
            reason="invalid_timeout",
        )
        self._max_retries = _non_negative_int(
            configured_retries if max_retries is None else max_retries,
            reason="invalid_max_retries",
        )
        self._failure_mode = _failure_mode(
            configured_failure_mode if failure_mode is None else failure_mode
        )
        self._backoff_base_s = _non_negative_float(
            backoff_base_s,
            reason="invalid_backoff",
        )
        self._max_backoff_s = _non_negative_float(
            max_backoff_s,
            reason="invalid_backoff",
        )
        self._sleep = sleep
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def endpoint(self) -> str:
        return DASHSCOPE_RERANK_BASE_URL

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def rerank(
        self,
        question: str,
        candidates: Sequence[RerankInput],
        top_n: int,
    ) -> list[RerankResult]:
        normalized_question, prepared, limit = _prepare_request(
            question,
            candidates,
            top_n,
        )
        if not prepared:
            return []

        body = {
            "model": self._model,
            "input": {
                "query": normalized_question,
                "documents": [candidate.text for candidate in prepared],
            },
            "parameters": {"top_n": limit},
        }
        try:
            payload = await self._post_json(body)
            return self._map_results(payload, prepared, limit)
        except RerankerError as error:
            if self._failure_mode == "degrade":
                return _original_order_results(
                    prepared,
                    limit,
                    degraded=True,
                    reason=error.reason,
                )
            raise error from None

    async def _post_json(self, body: dict[str, object]) -> object:
        for attempt in range(self._max_retries + 1):
            response: httpx.Response | None = None
            try:
                response = await self._client.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                    timeout=self._timeout_s,
                )
            except httpx.TimeoutException:
                error = RerankerError("timeout")
                retryable = True
            except Exception:
                error = RerankerError("transport_error")
                retryable = True
            else:
                if 200 <= response.status_code < 300:
                    try:
                        payload = response.json()
                    except (TypeError, ValueError):
                        payload = _MISSING
                    if payload is _MISSING:
                        raise RerankerError("invalid_json") from None
                    return payload
                error = RerankerError(f"http_{response.status_code}")
                retryable = response.status_code in {408, 425, 429} or (
                    500 <= response.status_code < 600
                )

            if not retryable or attempt >= self._max_retries:
                raise error from None
            await self._sleep(self._retry_delay(attempt, response))

        raise RerankerError("retry_exhausted")  # pragma: no cover

    def _retry_delay(self, attempt: int, response: httpx.Response | None) -> float:
        delay = self._backoff_base_s * (2**attempt)
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                with suppress(ValueError):
                    delay = max(delay, float(retry_after))
        return min(delay, self._max_backoff_s)

    @staticmethod
    def _map_results(
        payload: object,
        candidates: Sequence[RerankInput],
        limit: int,
    ) -> list[RerankResult]:
        if not isinstance(payload, dict):
            raise RerankerError("invalid_response")
        output = payload.get("output")
        if not isinstance(output, dict):
            raise RerankerError("invalid_response")
        raw_results = output.get("results")
        if not isinstance(raw_results, list):
            raise RerankerError("invalid_response")

        mapped: list[RerankResult] = []
        seen_indexes: set[int] = set()
        for raw_result in raw_results:
            if not isinstance(raw_result, dict):
                raise RerankerError("invalid_response")
            index = raw_result.get("index")
            score = raw_result.get("relevance_score")
            if isinstance(index, bool) or not isinstance(index, int):
                raise RerankerError("invalid_response")
            if index < 0 or index >= len(candidates) or index in seen_indexes:
                raise RerankerError("invalid_response")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise RerankerError("invalid_response")
            numeric_score = float(score)
            if not math.isfinite(numeric_score):
                raise RerankerError("invalid_response")

            candidate = candidates[index]
            seen_indexes.add(index)
            mapped.append(
                RerankResult(
                    id=candidate.id,
                    chunk=candidate.chunk,
                    rerank_score=numeric_score,
                    original_index=index,
                )
            )

        if len(mapped) < limit:
            raise RerankerError("incomplete_response")
        mapped.sort(key=lambda item: (-item.rerank_score, item.original_index))
        return mapped[:limit]


def create_reranker(
    settings: RerankerSettings | object,
    *,
    client: httpx.AsyncClient | None = None,
    sleep: Sleep = asyncio.sleep,
) -> Reranker:
    """Create the configured reranker; missing enablement remains fail-closed."""

    if not bool(_setting(settings, "reranker_enabled", False)):
        return NoopReranker()
    return DashScopeReranker(settings, client=client, sleep=sleep)
