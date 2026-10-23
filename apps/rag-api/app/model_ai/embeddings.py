"""Embedding providers for the RAG service.

The provider selected for an index is intentionally fixed.  Callers must use
``embedding_fingerprint`` when binding an index so deterministic and remote
vectors can never be mixed by a request-level fallback.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

import httpx
import numpy as np
from pydantic import SecretStr

if TYPE_CHECKING:
    from app.settings import Settings


DEFAULT_DASHSCOPE_BASE = "dashscope.aliyuncs.com"
DEFAULT_DASHSCOPE_MODEL = "text-embedding-v3"
DEFAULT_EMBEDDING_DIMENSION = 1024
DEFAULT_EMBEDDING_BATCH_SIZE = 16

_ASCII_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.ASCII)
_WHITESPACE_RE = re.compile(r"\s+")
_MISSING = object()


class EmbeddingProviderError(RuntimeError):
    """Base error for provider failures with credential-safe messages."""


class DashScopeEmbeddingError(EmbeddingProviderError):
    """A sanitized DashScope request or response failure."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Stable async interface implemented by all embedding providers."""

    @property
    def model_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def aclose(self) -> None: ...


def embedding_fingerprint(provider: EmbeddingProvider) -> str:
    """Return the model-and-dimension identity that must be stored per index."""

    return f"{provider.model_id}:{provider.dimension}"


def _validate_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def _validated_texts(texts: Sequence[str]) -> list[str]:
    if isinstance(texts, (str, bytes)):
        raise TypeError("texts must be a sequence of strings, not a single string")

    values = list(texts)
    for index, text in enumerate(values):
        if not isinstance(text, str):
            raise TypeError(f"text at index {index} must be a string")
        if not _normalize_text(text):
            raise ValueError(f"text at index {index} must not be empty")
    return values


class DeterministicHashEmbedding:
    """Replayable lexical embedding based on signed feature hashing.

    Unicode is NFKC/case-folded before extracting character 3-5 grams and
    complete ASCII tokens.  BLAKE2b, keyed by a digest of ``seed``, supplies a
    stable bucket and sign; Python's process-randomized hash is never used.
    """

    def __init__(self, dimension: int = 1024, seed: str = "rag-hash-v1") -> None:
        self._dimension = _validate_positive_int(dimension, "dimension")
        if not isinstance(seed, str) or not seed:
            raise ValueError("seed must be a non-empty string")
        self._seed = seed
        self._digest_key = hashlib.sha256(seed.encode("utf-8")).digest()

    @property
    def model_id(self) -> str:
        return f"deterministic-hash:{self._seed}"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def fingerprint(self) -> str:
        return embedding_fingerprint(self)

    def _features(self, normalized: str) -> Sequence[str]:
        features: list[str] = []
        padded = f"^{normalized}$"
        for width in (3, 4, 5):
            features.extend(
                f"char-{width}:{padded[offset : offset + width]}"
                for offset in range(max(0, len(padded) - width + 1))
            )
        features.extend(f"ascii-token:{token}" for token in _ASCII_TOKEN_RE.findall(normalized))
        return features

    def _feature_digest(self, feature: str) -> bytes:
        return hashlib.blake2b(
            feature.encode("utf-8"),
            digest_size=16,
            key=self._digest_key,
        ).digest()

    def _embed_one(self, normalized: str) -> list[float]:
        vector = np.zeros(self._dimension, dtype=np.float32)
        for feature in self._features(normalized):
            digest = self._feature_digest(feature)
            bucket = int.from_bytes(digest[:8], byteorder="little") % self._dimension
            sign = np.float32(1.0 if digest[8] & 1 else -1.0)
            vector[bucket] += sign

        norm = float(np.linalg.norm(vector.astype(np.float64)))
        if norm == 0.0:
            # A very small dimension can make all signed features cancel.  A
            # deterministic fallback bucket keeps every accepted text usable.
            digest = self._feature_digest(f"nonzero:{normalized}")
            bucket = int.from_bytes(digest[:8], byteorder="little") % self._dimension
            vector[bucket] = np.float32(1.0 if digest[8] & 1 else -1.0)
            norm = 1.0

        normalized_vector = (vector.astype(np.float64) / norm).astype(np.float32)
        return normalized_vector.tolist()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        values = _validated_texts(texts)
        return [self._embed_one(_normalize_text(text)) for text in values]

    async def aclose(self) -> None:
        """No-op close for interface symmetry with remote providers."""


def _secret_value(secret: object) -> str:
    getter = getattr(secret, "get_secret_value", None)
    if not callable(getter):
        raise TypeError("DashScope API key must be provided as SecretStr")
    value = getter()
    if not isinstance(value, str) or not value.strip():
        raise ValueError("DashScope API key is not configured")
    return value


def _dashscope_endpoint(base: str) -> str:
    if not isinstance(base, str) or not base.strip():
        raise ValueError("DashScope base must be a non-empty HTTPS host")

    candidate = base.strip()
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    parts = urlsplit(candidate)
    if (
        parts.scheme.lower() != "https"
        or not parts.netloc
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise ValueError("DashScope base must be an HTTPS host without credentials or query data")

    path = parts.path.rstrip("/")
    if path not in ("", "/", "/compatible-mode/v1", "/compatible-mode/v1/embeddings"):
        raise ValueError("DashScope base contains an unsupported path")

    origin = f"https://{parts.netloc}"
    if path == "/compatible-mode/v1/embeddings":
        return f"{origin}{path}"
    if path == "/compatible-mode/v1":
        return f"{origin}{path}/embeddings"
    return f"{origin}/compatible-mode/v1/embeddings"


class DashScopeEmbeddingProvider:
    """DashScope OpenAI-compatible asynchronous embedding client."""

    def __init__(
        self,
        api_key: SecretStr,
        model: str = DEFAULT_DASHSCOPE_MODEL,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        base: str = DEFAULT_DASHSCOPE_BASE,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        *,
        client: httpx.AsyncClient | None = None,
        backoff_base_s: float = 0.5,
    ) -> None:
        self._api_key = _secret_value(api_key)
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        self._model = model.strip()
        self._dimension = _validate_positive_int(dimension, "dimension")
        self._batch_size = _validate_positive_int(batch_size, "batch_size")
        if (
            not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be a positive finite number")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if (
            not isinstance(backoff_base_s, (int, float))
            or not math.isfinite(backoff_base_s)
            or backoff_base_s < 0
        ):
            raise ValueError("backoff_base_s must be a non-negative finite number")

        self._endpoint = _dashscope_endpoint(base)
        self._timeout_s = float(timeout_s)
        self._max_retries = max_retries
        self._backoff_base_s = float(backoff_base_s)
        self._client = client if client is not None else httpx.AsyncClient()
        self._closed = False

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def fingerprint(self) -> str:
        return embedding_fingerprint(self)

    async def _backoff(self, attempt: int) -> None:
        delay = min(60.0, self._backoff_base_s * (2**attempt))
        await asyncio.sleep(delay)

    async def _request_batch(self, batch: list[str]) -> list[list[float]]:
        request_body = {
            "model": self._model,
            "input": batch,
            "dimensions": self._dimension,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(
                    self._endpoint,
                    json=request_body,
                    headers=headers,
                    timeout=self._timeout_s,
                )
            except httpx.TimeoutException:
                if attempt < self._max_retries:
                    await self._backoff(attempt)
                    continue
                raise DashScopeEmbeddingError(
                    f"DashScope embeddings request timed out after {attempt + 1} attempts"
                ) from None
            except httpx.TransportError:
                if attempt < self._max_retries:
                    await self._backoff(attempt)
                    continue
                raise DashScopeEmbeddingError(
                    f"DashScope embeddings network request failed after {attempt + 1} attempts"
                ) from None

            status = response.status_code
            if 200 <= status < 300:
                return self._parse_response(response, len(batch))

            retryable = status == 429 or 500 <= status < 600
            if retryable and attempt < self._max_retries:
                await self._backoff(attempt)
                continue

            attempts = attempt + 1
            raise DashScopeEmbeddingError(
                "DashScope embeddings request failed "
                f"with status {status} after {attempts} attempts"
            )

        raise DashScopeEmbeddingError("DashScope embeddings request failed")

    def _parse_response(self, response: httpx.Response, expected_count: int) -> list[list[float]]:
        try:
            payload: Any = response.json()
        except ValueError:
            raise DashScopeEmbeddingError(
                "DashScope embeddings response is not valid JSON"
            ) from None

        if not isinstance(payload, Mapping):
            raise DashScopeEmbeddingError("DashScope embeddings response must be an object")
        data = payload.get("data")
        if not isinstance(data, list):
            raise DashScopeEmbeddingError("DashScope embeddings response data must be a list")
        if len(data) != expected_count:
            raise DashScopeEmbeddingError(
                "DashScope embeddings response count does not match the request"
            )

        ordered: list[list[float] | None] = [None] * expected_count
        for item in data:
            if not isinstance(item, Mapping):
                raise DashScopeEmbeddingError(
                    "DashScope embeddings response item must be an object"
                )
            index = item.get("index")
            if type(index) is not int or not 0 <= index < expected_count:
                raise DashScopeEmbeddingError("DashScope embeddings response has an invalid index")
            if ordered[index] is not None:
                raise DashScopeEmbeddingError("DashScope embeddings response has a duplicate index")
            ordered[index] = self._normalize_returned_vector(item.get("embedding"), index)

        if any(vector is None for vector in ordered):
            raise DashScopeEmbeddingError("DashScope embeddings response is missing an index")
        return [vector for vector in ordered if vector is not None]

    def _normalize_returned_vector(self, value: object, index: int) -> list[float]:
        if not isinstance(value, list) or len(value) != self._dimension:
            raise DashScopeEmbeddingError(
                f"DashScope embedding at index {index} has an invalid dimension"
            )
        if any(
            isinstance(component, bool)
            or not isinstance(component, (int, float))
            or not math.isfinite(float(component))
            for component in value
        ):
            raise DashScopeEmbeddingError(
                f"DashScope embedding at index {index} contains a non-finite value"
            )

        with np.errstate(over="ignore", invalid="ignore"):
            vector = np.asarray(value, dtype=np.float32)
        if not bool(np.isfinite(vector).all()):
            raise DashScopeEmbeddingError(
                f"DashScope embedding at index {index} cannot be represented as float32"
            )
        norm = float(np.linalg.norm(vector.astype(np.float64)))
        if not math.isfinite(norm) or norm == 0.0:
            raise DashScopeEmbeddingError(
                f"DashScope embedding at index {index} has zero or invalid norm"
            )

        normalized = (vector.astype(np.float64) / norm).astype(np.float32)
        if not bool(np.isfinite(normalized).all()):
            raise DashScopeEmbeddingError(
                f"DashScope embedding at index {index} failed normalization"
            )
        return normalized.tolist()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self._closed:
            raise DashScopeEmbeddingError("DashScope embedding provider is closed")
        values = _validated_texts(texts)
        embeddings: list[list[float]] = []
        for offset in range(0, len(values), self._batch_size):
            batch = values[offset : offset + self._batch_size]
            embeddings.extend(await self._request_batch(batch))
        return embeddings

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            await self._client.aclose()


def _first_setting(settings: object, *names: str, default: object = _MISSING) -> object:
    for name in names:
        value = getattr(settings, name, _MISSING)
        if value is not _MISSING and value is not None:
            return value
    if default is _MISSING:
        raise AttributeError(f"required setting is missing: {names[0]}")
    return default


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Build exactly one process/index-level provider; never silently fallback."""

    provider_value = _first_setting(settings, "embedding_provider", default="dashscope")
    if not isinstance(provider_value, str):
        raise ValueError("embedding_provider must be a string")
    provider_name = provider_value.strip().lower()
    dimension = _first_setting(
        settings,
        "embedding_dimension",
        default=DEFAULT_EMBEDDING_DIMENSION,
    )

    if provider_name == "deterministic":
        seed = _first_setting(settings, "embedding_hash_seed", default="rag-hash-v1")
        return DeterministicHashEmbedding(dimension=dimension, seed=seed)  # type: ignore[arg-type]

    if provider_name == "dashscope":
        api_key = _first_setting(settings, "dashscope_api_key", "embedding_api_key", default=None)
        return DashScopeEmbeddingProvider(
            api_key=api_key,  # type: ignore[arg-type]
            model=_first_setting(
                settings,
                "embedding_model",
                default=DEFAULT_DASHSCOPE_MODEL,
            ),  # type: ignore[arg-type]
            dimension=dimension,  # type: ignore[arg-type]
            batch_size=_first_setting(
                settings,
                "embedding_batch_size",
                default=DEFAULT_EMBEDDING_BATCH_SIZE,
            ),  # type: ignore[arg-type]
            base=_first_setting(
                settings,
                "dashscope_base",
                "dashscope_base_url",
                default=DEFAULT_DASHSCOPE_BASE,
            ),  # type: ignore[arg-type]
            timeout_s=_first_setting(
                settings,
                "embedding_timeout_seconds",
                "dashscope_timeout_seconds",
                "hf_timeout_seconds",
                default=30.0,
            ),  # type: ignore[arg-type]
            max_retries=_first_setting(
                settings,
                "embedding_max_retries",
                "dashscope_max_retries",
                "hf_max_retries",
                default=3,
            ),  # type: ignore[arg-type]
        )

    if provider_name == "hf":
        raise NotImplementedError("the hf embedding provider is not implemented")
    raise ValueError("unsupported embedding provider")


__all__ = [
    "DashScopeEmbeddingError",
    "DashScopeEmbeddingProvider",
    "DeterministicHashEmbedding",
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "create_embedding_provider",
    "embedding_fingerprint",
]
