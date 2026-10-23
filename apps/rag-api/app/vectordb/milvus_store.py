"""Milvus-backed vector index for the RAG service.

The public API deliberately accepts typed records and search scopes only.  Milvus
expressions are assembled in this module from UUIDs and enums, never from caller-
supplied expression strings.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Final
from uuid import UUID

from app.filter_translator import translate_filters
from app.query_filters import FilterCondition
from app.scalar_metadata import (
    PACKAGE_FIELD_MAX_COUNT,
    PACKAGE_TOKEN_MAX_CHARS,
    canonical_token,
)

PYMILVUS_VERSION: Final = "2.6.17"
DEFAULT_MILVUS_URI: Final = "http://127.0.0.1:19531"
DEFAULT_COLLECTION_NAME: Final = "rag_specs"
V2_COLLECTION_NAME: Final = "rag_specs_v2"
DEFAULT_DIMENSION: Final = 1024
DEFAULT_BATCH_SIZE: Final = 128
DEFAULT_SEARCH_EF: Final = 64
HNSW_M: Final = 16
HNSW_EF_CONSTRUCTION: Final = 200
VECTOR_INDEX_NAME: Final = "vector_hnsw"
VECTOR_FIELD_NAME: Final = "vector"
METRIC_TYPE: Final = "COSINE"

_COLLECTION_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TEST_COLLECTION_NAME_RE = re.compile(r"^rag_specs_test_[0-9a-f]{32}$")
_MAX_SEARCH_LIMIT = 16_384


class KnowledgeChunkKind(str, Enum):  # noqa: UP042 - frozen cross-layer contract
    """Stable scalar values shared by PostgreSQL and Milvus."""

    TEXT = "TEXT"
    TABLE = "TABLE"
    VISUAL = "VISUAL"
    MIXED = "MIXED"
    CODE = "CODE"


def _chunk_kind(value: object) -> KnowledgeChunkKind:
    if isinstance(value, KnowledgeChunkKind):
        return value
    # The canonical DB/API enums live in separate layers.  Accept only an Enum
    # carrying one of the frozen scalar values; arbitrary strings remain invalid.
    if isinstance(value, Enum):
        try:
            return KnowledgeChunkKind(str(value.value))
        except ValueError:
            pass
    raise TypeError("kind must be a KnowledgeChunkKind enum value")


@dataclass(frozen=True)
class VectorRecord:
    id: str
    vector: Sequence[float]
    document_id: UUID
    product: str
    version: str
    kind: KnowledgeChunkKind
    semantic_type: str
    chunk_index: int
    page_start: int
    page_end: int
    section_path: str
    text: str
    embedding_model_id: str
    section_id: str = ""
    register: str = ""
    packages: tuple[str, ...] = ()


@dataclass(frozen=True)
class VectorSearchScope:
    document_ids: Sequence[UUID]
    kinds: Sequence[KnowledgeChunkKind] = ()

    def __post_init__(self) -> None:
        if isinstance(self.document_ids, (str, bytes)):
            raise TypeError("document_ids must be a sequence of UUID values")
        if isinstance(self.kinds, (str, bytes)):
            raise TypeError("kinds must be a sequence of KnowledgeChunkKind values")

        document_ids = tuple(self.document_ids)
        kinds = tuple(self.kinds)
        if any(not isinstance(document_id, UUID) for document_id in document_ids):
            raise TypeError("document_ids must contain UUID values only")
        normalized_kinds = tuple(_chunk_kind(kind) for kind in kinds)

        # frozen=True protects attributes, while tuples also protect their contents.
        object.__setattr__(self, "document_ids", document_ids)
        object.__setattr__(self, "kinds", normalized_kinds)


@dataclass(frozen=True)
class VectorHit:
    id: str
    score: float
    document_id: UUID
    product: str
    version: str
    kind: KnowledgeChunkKind
    semantic_type: str
    chunk_index: int
    page_start: int
    page_end: int
    section_path: str
    text: str
    embedding_model_id: str

    @property
    def embedding_id(self) -> str:
        """Compatibility name used by the ingest/query plan."""

        return self.id


@dataclass(frozen=True)
class DependencyHealth:
    ready: bool
    detail: str

    @property
    def ok(self) -> bool:
        return self.ready

    @property
    def healthy(self) -> bool:
        return self.ready


class MilvusStoreError(RuntimeError):
    """Base class for vector-store contract failures."""


class MilvusConfigurationError(MilvusStoreError):
    """The local store configuration is unsafe or unsupported."""


class MilvusSchemaMismatchError(MilvusStoreError):
    """An existing collection or index differs from the frozen contract."""


class EmbeddingModelMismatchError(MilvusStoreError):
    """The collection contains vectors from another embedding fingerprint."""


class VectorCountMismatchError(MilvusStoreError):
    """A Strong-consistency verification observed an unexpected row count."""


class MilvusStoreClosedError(MilvusStoreError):
    """An operation was attempted after shutdown began."""


@dataclass(frozen=True)
class _FieldSpec:
    name: str
    data_type: str
    is_primary: bool = False
    dim: int | None = None
    max_length: int | None = None
    element_type: str | None = None
    max_capacity: int | None = None


@dataclass(frozen=True)
class _IndexSpec:
    field_name: str = VECTOR_FIELD_NAME
    index_name: str = VECTOR_INDEX_NAME
    index_type: str = "HNSW"
    metric_type: str = METRIC_TYPE
    m: int = HNSW_M
    ef_construction: int = HNSW_EF_CONSTRUCTION


def _collection_field_specs(
    dimension: int, collection_name: str = DEFAULT_COLLECTION_NAME
) -> tuple[_FieldSpec, ...]:
    """Return a dependency-free schema blueprint for validation and tests."""

    fields = (
        _FieldSpec("id", "VARCHAR", is_primary=True, max_length=512),
        _FieldSpec(VECTOR_FIELD_NAME, "FLOAT_VECTOR", dim=dimension),
        _FieldSpec("document_id", "VARCHAR", max_length=36),
        _FieldSpec("product", "VARCHAR", max_length=256),
        _FieldSpec("version", "VARCHAR", max_length=128),
        _FieldSpec("kind", "VARCHAR", max_length=16),
        _FieldSpec("semantic_type", "VARCHAR", max_length=128),
        _FieldSpec("chunk_index", "INT64"),
        _FieldSpec("page_start", "INT64"),
        _FieldSpec("page_end", "INT64"),
        _FieldSpec("section_path", "VARCHAR", max_length=4096),
        _FieldSpec("text", "VARCHAR", max_length=65_535),
        _FieldSpec("embedding_model_id", "VARCHAR", max_length=512),
    )
    if collection_name == V2_COLLECTION_NAME:
        fields += (
            _FieldSpec("section_id", "VARCHAR", max_length=64),
            _FieldSpec("register", "VARCHAR", max_length=256),
            # Package is a set, so it is an ARRAY: membership is expressed with
            # array_contains(), which a delimited VARCHAR cannot do safely
            # (measured: bare infix LIKE matched "SOP2" against "SSOP28").
            _FieldSpec(
                "packages",
                "ARRAY",
                element_type="VARCHAR",
                max_length=PACKAGE_TOKEN_MAX_CHARS,
                max_capacity=PACKAGE_FIELD_MAX_COUNT,
            ),
        )
    return fields


def _index_spec() -> _IndexSpec:
    return _IndexSpec()


def _build_collection_schema(
    client: Any,
    data_type: Any,
    dimension: int,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Any:
    """Build a PyMilvus schema; ``data_type`` is injectable for offline tests."""

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    for field in _collection_field_specs(dimension, collection_name):
        kwargs: dict[str, Any] = {
            "field_name": field.name,
            "datatype": getattr(data_type, field.data_type),
        }
        if field.is_primary:
            kwargs["is_primary"] = True
        if field.dim is not None:
            kwargs["dim"] = field.dim
        if field.max_length is not None:
            kwargs["max_length"] = field.max_length
        if field.element_type is not None:
            kwargs["element_type"] = getattr(data_type, field.element_type)
        if field.max_capacity is not None:
            kwargs["max_capacity"] = field.max_capacity
        schema.add_field(**kwargs)
    return schema


def _build_index_params(client: Any) -> Any:
    """Build the exact HNSW/COSINE index parameters from plan section 3.7."""

    spec = _index_spec()
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=spec.field_name,
        index_name=spec.index_name,
        index_type=spec.index_type,
        metric_type=spec.metric_type,
        params={"M": spec.m, "efConstruction": spec.ef_construction},
    )
    return index_params


def _data_type_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.upper()
    if isinstance(value, str):
        return value.rsplit(".", maxsplit=1)[-1].upper()
    if isinstance(value, int):
        try:
            from pymilvus import DataType

            return DataType(value).name.upper()
        except (ImportError, TypeError, ValueError):
            return str(value)
    return str(value).rsplit(".", maxsplit=1)[-1].upper()


def _integer(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise MilvusSchemaMismatchError(f"{label} is not an integer") from exc


def _validate_collection_description(
    description: Mapping[str, Any], dimension: int, collection_name: str
) -> None:
    """Fail closed when an existing collection differs from the contract."""

    if not isinstance(description, Mapping):
        raise MilvusSchemaMismatchError("collection description is not a mapping")
    if description.get("collection_name") != collection_name:
        raise MilvusSchemaMismatchError(
            "collection name resolves to a different physical collection"
        )
    if "enable_dynamic_field" not in description or bool(
        description["enable_dynamic_field"]
    ):
        raise MilvusSchemaMismatchError("dynamic fields must be disabled")
    if "auto_id" not in description or bool(description["auto_id"]):
        raise MilvusSchemaMismatchError("auto_id must be disabled")
    if description.get("struct_array_fields"):
        raise MilvusSchemaMismatchError("unexpected struct-array fields are present")

    raw_fields = description.get("fields")
    if not isinstance(raw_fields, Sequence) or isinstance(raw_fields, (str, bytes)):
        raise MilvusSchemaMismatchError("collection fields are missing")

    actual_fields: dict[str, Mapping[str, Any]] = {}
    for raw_field in raw_fields:
        if not isinstance(raw_field, Mapping) or not isinstance(raw_field.get("name"), str):
            raise MilvusSchemaMismatchError("collection contains an invalid field description")
        name = raw_field["name"]
        if name in actual_fields:
            raise MilvusSchemaMismatchError(f"duplicate field: {name}")
        actual_fields[name] = raw_field

    expected_fields = {
        field.name: field
        for field in _collection_field_specs(dimension, collection_name)
    }
    if actual_fields.keys() != expected_fields.keys():
        missing = sorted(expected_fields.keys() - actual_fields.keys())
        extra = sorted(actual_fields.keys() - expected_fields.keys())
        raise MilvusSchemaMismatchError(
            f"field set differs (missing={missing}, extra={extra})"
        )

    for name, expected in expected_fields.items():
        actual = actual_fields[name]
        if _data_type_name(actual.get("type")) != expected.data_type:
            raise MilvusSchemaMismatchError(f"field {name!r} has the wrong data type")
        if bool(actual.get("is_primary", False)) != expected.is_primary:
            raise MilvusSchemaMismatchError(f"field {name!r} has the wrong primary-key flag")
        if bool(actual.get("nullable", False)):
            raise MilvusSchemaMismatchError(f"field {name!r} must not be nullable")
        if bool(actual.get("is_dynamic", False)):
            raise MilvusSchemaMismatchError(f"field {name!r} must not be dynamic")

        params = actual.get("params") or {}
        if not isinstance(params, Mapping):
            raise MilvusSchemaMismatchError(f"field {name!r} params are invalid")
        if expected.dim is not None and _integer(params.get("dim"), f"{name}.dim") != expected.dim:
            raise MilvusSchemaMismatchError(f"field {name!r} has the wrong dimension")
        if expected.max_length is not None and _integer(
            params.get("max_length"), f"{name}.max_length"
        ) != expected.max_length:
            raise MilvusSchemaMismatchError(f"field {name!r} has the wrong max_length")
        if expected.element_type is not None:
            actual_element = _data_type_name(
                actual.get("element_type", params.get("element_type"))
            )
            if actual_element != expected.element_type:
                raise MilvusSchemaMismatchError(f"field {name!r} has the wrong element_type")
        if expected.max_capacity is not None and _integer(
            params.get("max_capacity"), f"{name}.max_capacity"
        ) != expected.max_capacity:
            raise MilvusSchemaMismatchError(f"field {name!r} has the wrong max_capacity")


def _mapping_params(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, Mapping):
            return parsed
    return {}


def _validate_index_description(description: Mapping[str, Any]) -> None:
    if not isinstance(description, Mapping):
        raise MilvusSchemaMismatchError("vector index description is missing")

    params = _mapping_params(description.get("params"))
    nested = _mapping_params(description.get("index_param"))

    def parameter(name: str) -> Any:
        if name in description:
            return description[name]
        if name in params:
            return params[name]
        return nested.get(name)

    spec = _index_spec()
    if description.get("field_name") != spec.field_name:
        raise MilvusSchemaMismatchError("index is attached to the wrong field")
    if str(parameter("index_type")).upper() != spec.index_type:
        raise MilvusSchemaMismatchError("vector index type must be HNSW")
    if str(parameter("metric_type")).upper() != spec.metric_type:
        raise MilvusSchemaMismatchError("vector index metric must be COSINE")
    if _integer(parameter("M"), "index M") != spec.m:
        raise MilvusSchemaMismatchError("HNSW M does not match the frozen contract")
    if _integer(parameter("efConstruction"), "index efConstruction") != spec.ef_construction:
        raise MilvusSchemaMismatchError(
            "HNSW efConstruction does not match the frozen contract"
        )

    state = description.get("state")
    if state is not None:
        state_name = getattr(state, "name", str(state)).rsplit(".", maxsplit=1)[-1]
        if state_name.casefold() != "finished":
            raise MilvusSchemaMismatchError("vector index is not ready")


def _json_literal(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _build_search_filter(scope: VectorSearchScope, embedding_model_id: str) -> str:
    if not isinstance(scope, VectorSearchScope):
        raise TypeError("scope must be a VectorSearchScope")
    document_ids = sorted({str(document_id) for document_id in scope.document_ids})
    if not document_ids:
        raise ValueError("search scope must contain at least one document_id")

    clauses = [f"document_id in {_json_literal(document_ids)}"]
    kinds = sorted({kind.value for kind in scope.kinds})
    if kinds:
        clauses.append(f"kind in {_json_literal(kinds)}")
    clauses.append(f"embedding_model_id == {_json_literal(embedding_model_id)}")
    return " and ".join(clauses)


def scalar_filter_clause(
    filters: Sequence[FilterCondition], collection_name: str
) -> str | None:
    """Return the pushdown clause for this collection, or ``None``.

    The scalar fields only exist in the v2 collection, so pushdown is refused for
    ``rag_specs``: the same filters keep working as post-retrieval checks there.
    """

    if not filters:
        return None
    if collection_name != V2_COLLECTION_NAME:
        return None
    clause, _remaining = translate_filters(tuple(filters))
    return clause


def _require_uuid(value: UUID, label: str) -> UUID:
    if not isinstance(value, UUID):
        raise TypeError(f"{label} must be a UUID")
    return value


def _require_integer(value: int, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _require_varchar(value: str, label: str, max_bytes: int, *, empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if not empty and not value:
        raise ValueError(f"{label} must not be empty")
    if len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"{label} exceeds the Milvus VARCHAR limit ({max_bytes} UTF-8 bytes)")
    return value


def _require_packages(value: object) -> list[str]:
    """Validate and serialise a package set for the ``ARRAY<VARCHAR>`` field.

    The field is a set by contract, so a bare string is rejected rather than split:
    a caller that still holds the delimited form must pass
    ``scalar_metadata.parse_package_field`` first.
    """

    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("packages must be a sequence of package codes, not a string")
    if not isinstance(value, Sequence):
        raise TypeError("packages must be a sequence of package codes")
    tokens: list[str] = []
    for item in value:
        token = canonical_token(item)
        if not token:
            continue
        if len(token) > PACKAGE_TOKEN_MAX_CHARS:
            raise ValueError(
                f"package code {token!r} exceeds the ARRAY element limit "
                f"({PACKAGE_TOKEN_MAX_CHARS} chars)"
            )
        if token not in tokens:
            tokens.append(token)
    if len(tokens) > PACKAGE_FIELD_MAX_COUNT:
        raise ValueError(
            f"packages exceed the ARRAY capacity ({PACKAGE_FIELD_MAX_COUNT} values)"
        )
    return tokens


def _vector_values(vector: Sequence[float], dimension: int) -> list[float]:
    if isinstance(vector, (str, bytes)):
        raise TypeError("vector must be a numeric sequence")
    try:
        length = len(vector)
    except TypeError as exc:
        raise TypeError("vector must be a sized numeric sequence") from exc
    if length != dimension:
        raise ValueError(f"vector dimension must be {dimension}, got {length}")

    values: list[float] = []
    for position, value in enumerate(vector):
        if isinstance(value, bool):
            raise TypeError(f"vector[{position}] must be numeric")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"vector[{position}] must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"vector[{position}] must be finite")
        values.append(number)
    return values


def _default_client_factory(**kwargs: Any) -> Any:
    try:
        import pymilvus
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise MilvusConfigurationError(
            f"pymilvus=={PYMILVUS_VERSION} is required for MilvusStore"
        ) from exc

    installed_version = getattr(pymilvus, "__version__", None)
    if installed_version != PYMILVUS_VERSION:
        raise MilvusConfigurationError(
            f"pymilvus=={PYMILVUS_VERSION} is required; found {installed_version or 'unknown'}"
        )
    return MilvusClient(**kwargs)


def _pymilvus_data_type() -> Any:
    try:
        import pymilvus
        from pymilvus import DataType
    except ImportError as exc:
        raise MilvusConfigurationError(
            f"pymilvus=={PYMILVUS_VERSION} is required for MilvusStore"
        ) from exc
    if getattr(pymilvus, "__version__", None) != PYMILVUS_VERSION:
        raise MilvusConfigurationError(f"pymilvus=={PYMILVUS_VERSION} is required")
    return DataType


def _setting_value(settings: object, name: str, default: Any = None) -> Any:
    if isinstance(settings, Mapping):
        return settings.get(name, default)
    return getattr(settings, name, default)


def _secret_string(value: object, label: str) -> str:
    if isinstance(value, str):
        return value
    getter = getattr(value, "get_secret_value", None)
    if callable(getter):
        secret = getter()
        if isinstance(secret, str):
            return secret
    raise MilvusConfigurationError(f"{label} must be a string or SecretStr")


def _embedding_fingerprint_from_settings(settings: object, dimension: int) -> str:
    explicit = _setting_value(settings, "embedding_model_id")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    provider = str(_setting_value(settings, "embedding_provider", "deterministic")).strip()
    if provider.casefold() == "deterministic":
        seed = str(_setting_value(settings, "embedding_hash_seed", "rag-hash-v1")).strip()
        if not seed:
            raise MilvusConfigurationError("embedding_hash_seed must not be empty")
        return f"deterministic-hash:{seed}:{dimension}"

    model = str(_setting_value(settings, "embedding_model", "")).strip()
    if not model:
        raise MilvusConfigurationError("embedding_model must not be empty")
    return f"{model}:{dimension}"


def create_milvus_store(settings: object) -> MilvusStore:
    """Construct the store from the ``RAG_`` settings without exposing secrets."""

    dimension = int(_setting_value(settings, "embedding_dimension", DEFAULT_DIMENSION))
    token = _secret_string(_setting_value(settings, "milvus_token", ""), "milvus_token")
    return MilvusStore(
        uri=str(_setting_value(settings, "milvus_uri", DEFAULT_MILVUS_URI)),
        token=token,
        collection_name=str(
            _setting_value(settings, "milvus_collection", DEFAULT_COLLECTION_NAME)
        ),
        dimension=dimension,
        batch_size=int(_setting_value(settings, "milvus_batch_size", DEFAULT_BATCH_SIZE)),
        search_ef=int(_setting_value(settings, "milvus_search_ef", DEFAULT_SEARCH_EF)),
        executor_workers=int(_setting_value(settings, "milvus_executor_workers", 4)),
        embedding_model_id=_embedding_fingerprint_from_settings(settings, dimension),
    )


class MilvusStore:
    """Async facade over the synchronous PyMilvus 2.6.17 client."""

    def __init__(
        self,
        *,
        embedding_model_id: str,
        uri: str = DEFAULT_MILVUS_URI,
        token: str = "",
        collection_name: str = DEFAULT_COLLECTION_NAME,
        dimension: int = DEFAULT_DIMENSION,
        batch_size: int = DEFAULT_BATCH_SIZE,
        search_ef: int = DEFAULT_SEARCH_EF,
        executor_workers: int = 4,
        timeout: float | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not isinstance(uri, str) or not uri:
            raise MilvusConfigurationError("Milvus URI must be a non-empty string")
        if not isinstance(token, str):
            raise MilvusConfigurationError("Milvus token must be a string")
        if not isinstance(collection_name, str) or not _COLLECTION_NAME_RE.fullmatch(
            collection_name
        ):
            raise MilvusConfigurationError("Milvus collection name is invalid")
        if "stockpilot" in collection_name.casefold():
            raise MilvusConfigurationError("the stockpilot collection is outside the RAG boundary")
        if (
            collection_name not in {DEFAULT_COLLECTION_NAME, V2_COLLECTION_NAME}
            and not _TEST_COLLECTION_NAME_RE.fullmatch(collection_name)
        ):
            raise MilvusConfigurationError(
                "Milvus collection must be rag_specs or an isolated randomized test collection"
            )
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise MilvusConfigurationError("embedding dimension must be a positive integer")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise MilvusConfigurationError("Milvus batch size must be a positive integer")
        if isinstance(search_ef, bool) or not isinstance(search_ef, int) or search_ef <= 0:
            raise MilvusConfigurationError("Milvus search ef must be a positive integer")
        if (
            isinstance(executor_workers, bool)
            or not isinstance(executor_workers, int)
            or executor_workers <= 0
        ):
            raise MilvusConfigurationError("executor_workers must be a positive integer")
        if timeout is not None and timeout <= 0:
            raise MilvusConfigurationError("timeout must be positive when configured")

        self._embedding_model_id = _require_varchar(
            embedding_model_id, "embedding_model_id", 512
        )
        self._uri = uri
        self._token = token
        self._collection_name = collection_name
        self._dimension = dimension
        self._batch_size = batch_size
        self._search_ef = search_ef
        self._executor_workers = executor_workers
        self._timeout = timeout
        self._client_factory = client_factory or _default_client_factory

        self._executor = ThreadPoolExecutor(
            max_workers=executor_workers, thread_name_prefix="rag-milvus"
        )
        # No call can enter the executor without a slot, so its work queue stays bounded.
        self._operation_slots = asyncio.BoundedSemaphore(executor_workers)
        self._client_lock = threading.Lock()
        self._ensure_lock = threading.Lock()
        self._close_lock = asyncio.Lock()
        self._state_lock = threading.Lock()
        self._client: Any | None = None
        self._ready = False
        self._closing = False
        self._closed = False

    def __repr__(self) -> str:
        return (
            f"MilvusStore(collection_name={self._collection_name!r}, "
            f"dimension={self._dimension}, closed={self._closed})"
        )

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def embedding_model_id(self) -> str:
        return self._embedding_model_id

    def _assert_available(self) -> None:
        with self._state_lock:
            if self._closed or self._closing:
                raise MilvusStoreClosedError("MilvusStore is closing or closed")

    def _assert_ready(self) -> None:
        self._assert_available()
        if not self._ready:
            raise MilvusStoreError("ensure_collection() must complete before this operation")

    async def _run_sync(self, function: Callable[..., Any], *args: Any) -> Any:
        self._assert_available()
        await self._operation_slots.acquire()
        try:
            self._assert_available()
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(self._executor, partial(function, *args))
        except BaseException:
            self._operation_slots.release()
            raise

        # Release only when the worker really finishes.  Cancellation of an HTTP
        # request must not let shutdown close a client still used by that worker.
        future.add_done_callback(lambda _future: self._operation_slots.release())
        return await asyncio.shield(future)

    def _timeout_kwargs(self) -> dict[str, float]:
        return {} if self._timeout is None else {"timeout": self._timeout}

    def _get_client_sync(self) -> Any:
        with self._client_lock:
            if self._client is None:
                kwargs: dict[str, Any] = {"uri": self._uri, "token": self._token}
                kwargs.update(self._timeout_kwargs())
                self._client = self._client_factory(**kwargs)
            return self._client

    def _describe_and_validate_sync(self, client: Any) -> None:
        description = client.describe_collection(
            collection_name=self._collection_name, **self._timeout_kwargs()
        )
        _validate_collection_description(description, self._dimension, self._collection_name)

        index_names = client.list_indexes(
            collection_name=self._collection_name,
            field_name=VECTOR_FIELD_NAME,
            **self._timeout_kwargs(),
        )
        if not isinstance(index_names, Sequence) or isinstance(index_names, (str, bytes)):
            raise MilvusSchemaMismatchError("vector index list is invalid")
        if len(index_names) != 1:
            raise MilvusSchemaMismatchError("exactly one vector index is required")
        index_description = client.describe_index(
            collection_name=self._collection_name,
            index_name=index_names[0],
            **self._timeout_kwargs(),
        )
        _validate_index_description(index_description)

    def _verify_fingerprint_sync(self, client: Any) -> None:
        mismatches = client.query(
            collection_name=self._collection_name,
            filter=f"embedding_model_id != {_json_literal(self._embedding_model_id)}",
            output_fields=["id", "embedding_model_id"],
            limit=1,
            consistency_level="Strong",
            **self._timeout_kwargs(),
        )
        if mismatches:
            raise EmbeddingModelMismatchError(
                "Milvus collection contains a different embedding_model_id"
            )

    def _ensure_collection_sync(self) -> None:
        with self._ensure_lock:
            client = self._get_client_sync()
            exists = client.has_collection(
                collection_name=self._collection_name, **self._timeout_kwargs()
            )
            if not exists:
                schema = _build_collection_schema(
                    client,
                    _pymilvus_data_type(),
                    self._dimension,
                    self._collection_name,
                )
                index_params = _build_index_params(client)
                client.create_collection(
                    collection_name=self._collection_name,
                    schema=schema,
                    index_params=index_params,
                    consistency_level="Strong",
                    **self._timeout_kwargs(),
                )

            # Validate both newly created and pre-existing collections.  There is
            # intentionally no drop/recreate or schema-repair path here.
            self._describe_and_validate_sync(client)
            client.load_collection(
                collection_name=self._collection_name, **self._timeout_kwargs()
            )
            self._verify_fingerprint_sync(client)

    async def ensure_collection(self) -> None:
        await self._run_sync(self._ensure_collection_sync)
        self._ready = True

    def _record_entity(self, record: VectorRecord) -> dict[str, Any]:
        if not isinstance(record, VectorRecord):
            raise TypeError("records must contain VectorRecord values only")
        if record.embedding_model_id != self._embedding_model_id:
            raise EmbeddingModelMismatchError(
                "record embedding_model_id does not match the configured index fingerprint"
            )
        kind = _chunk_kind(record.kind)

        chunk_index = _require_integer(record.chunk_index, "chunk_index")
        page_start = _require_integer(record.page_start, "page_start", minimum=1)
        page_end = _require_integer(record.page_end, "page_end", minimum=1)
        if page_end < page_start:
            raise ValueError("page_end must be greater than or equal to page_start")

        entity = {
            "id": _require_varchar(record.id, "id", 512),
            VECTOR_FIELD_NAME: _vector_values(record.vector, self._dimension),
            "document_id": str(_require_uuid(record.document_id, "document_id")),
            "product": _require_varchar(record.product, "product", 256),
            "version": _require_varchar(record.version, "version", 128),
            "kind": kind.value,
            "semantic_type": _require_varchar(record.semantic_type, "semantic_type", 128),
            "chunk_index": chunk_index,
            "page_start": page_start,
            "page_end": page_end,
            "section_path": _require_varchar(
                record.section_path, "section_path", 4096, empty=True
            ),
            "text": _require_varchar(record.text, "text", 65_535),
            "embedding_model_id": _require_varchar(
                record.embedding_model_id, "embedding_model_id", 512
            ),
        }
        if self._collection_name == V2_COLLECTION_NAME:
            entity.update(
                {
                    "section_id": _require_varchar(
                        record.section_id, "section_id", 64, empty=True
                    ),
                    "register": _require_varchar(
                        record.register, "register", 256, empty=True
                    ),
                    "packages": _require_packages(record.packages),
                }
            )
        return entity

    def _upsert_sync(self, entities: Sequence[dict[str, Any]]) -> int:
        client = self._get_client_sync()
        self._verify_fingerprint_sync(client)
        total = 0
        for start in range(0, len(entities), self._batch_size):
            batch = list(entities[start : start + self._batch_size])
            result = client.upsert(
                collection_name=self._collection_name,
                data=batch,
                **self._timeout_kwargs(),
            )
            if not isinstance(result, Mapping) or "upsert_count" not in result:
                raise MilvusStoreError("Milvus upsert returned no upsert_count")
            count = int(result["upsert_count"])
            if count != len(batch):
                raise MilvusStoreError(
                    f"Milvus upsert count mismatch: expected {len(batch)}, observed {count}"
                )
            total += count
        self._verify_fingerprint_sync(client)
        return total

    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        self._assert_ready()
        if isinstance(records, (str, bytes)):
            raise TypeError("records must be a sequence of VectorRecord values")
        entities = [self._record_entity(record) for record in records]
        ids = [entity["id"] for entity in entities]
        if len(ids) != len(set(ids)):
            raise ValueError("records contain duplicate vector ids")
        if not entities:
            return 0
        return int(await self._run_sync(self._upsert_sync, entities))

    def _search_sync(
        self, vector: list[float], expression: str, limit: int
    ) -> list[Mapping[str, Any]]:
        client = self._get_client_sync()
        self._verify_fingerprint_sync(client)
        result = client.search(
            collection_name=self._collection_name,
            anns_field=VECTOR_FIELD_NAME,
            data=[vector],
            filter=expression,
            limit=limit,
            output_fields=[
                "id",
                "document_id",
                "product",
                "version",
                "kind",
                "semantic_type",
                "chunk_index",
                "page_start",
                "page_end",
                "section_path",
                "text",
                "embedding_model_id",
            ],
            search_params={
                "metric_type": METRIC_TYPE,
                "params": {"ef": max(self._search_ef, limit)},
            },
            consistency_level="Strong",
            **self._timeout_kwargs(),
        )
        if not isinstance(result, Sequence) or isinstance(result, (str, bytes)) or len(result) != 1:
            raise MilvusStoreError("Milvus search returned an invalid result shape")
        hits = result[0]
        if not isinstance(hits, Sequence) or isinstance(hits, (str, bytes)):
            raise MilvusStoreError("Milvus search returned an invalid hit list")
        return list(hits)

    def _vector_hit(self, raw_hit: Mapping[str, Any]) -> VectorHit:
        if not isinstance(raw_hit, Mapping):
            raise MilvusStoreError("Milvus returned an invalid search hit")
        entity = raw_hit.get("entity") or {}
        if not isinstance(entity, Mapping):
            raise MilvusStoreError("Milvus returned invalid hit metadata")

        def field(name: str) -> Any:
            if name == "id" and name in raw_hit:
                return raw_hit[name]
            return entity.get(name)

        model_id = str(field("embedding_model_id"))
        if model_id != self._embedding_model_id:
            raise EmbeddingModelMismatchError(
                "Milvus search returned a different embedding_model_id"
            )
        try:
            return VectorHit(
                id=str(field("id")),
                score=float(raw_hit["distance"]),
                document_id=UUID(str(field("document_id"))),
                product=str(field("product")),
                version=str(field("version")),
                kind=KnowledgeChunkKind(str(field("kind"))),
                semantic_type=str(field("semantic_type")),
                chunk_index=int(field("chunk_index")),
                page_start=int(field("page_start")),
                page_end=int(field("page_end")),
                section_path=str(field("section_path")),
                text=str(field("text")),
                embedding_model_id=model_id,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MilvusStoreError("Milvus returned incomplete search-hit metadata") from exc

    async def search(
        self,
        vector: Sequence[float],
        scope: VectorSearchScope,
        limit: int,
        *,
        filters: Sequence[FilterCondition] = (),
    ) -> list[VectorHit]:
        if not isinstance(scope, VectorSearchScope):
            raise TypeError("scope must be a VectorSearchScope")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer")
        if not 1 <= limit <= _MAX_SEARCH_LIMIT:
            raise ValueError(f"limit must be between 1 and {_MAX_SEARCH_LIMIT}")
        vector_values = _vector_values(vector, self._dimension)
        if not scope.document_ids:
            # An empty PG-resolved scope must never turn into an unfiltered search.
            return []

        self._assert_ready()
        expression = _build_search_filter(scope, self._embedding_model_id)
        clause = scalar_filter_clause(filters, self._collection_name)
        if clause:
            expression = f"({expression}) and ({clause})"
        raw_hits = await self._run_sync(self._search_sync, vector_values, expression, limit)
        return [self._vector_hit(raw_hit) for raw_hit in raw_hits]

    def _count_filter_sync(self, client: Any, expression: str) -> int:
        rows = client.query(
            collection_name=self._collection_name,
            filter=expression,
            output_fields=["count(*)"],
            consistency_level="Strong",
            **self._timeout_kwargs(),
        )
        if not isinstance(rows, Sequence) or len(rows) != 1 or not isinstance(rows[0], Mapping):
            raise MilvusStoreError("Milvus count query returned an invalid result shape")
        value = rows[0].get("count(*)", rows[0].get("count"))
        if value is None:
            raise MilvusStoreError("Milvus count query returned no count")
        return int(value)

    def _count_document_sync(self, document_id: UUID) -> int:
        client = self._get_client_sync()
        self._verify_fingerprint_sync(client)
        expression = (
            f"document_id == {_json_literal(str(document_id))} and "
            f"embedding_model_id == {_json_literal(self._embedding_model_id)}"
        )
        return self._count_filter_sync(client, expression)

    async def count_document(self, document_id: UUID) -> int:
        self._assert_ready()
        document_id = _require_uuid(document_id, "document_id")
        return int(await self._run_sync(self._count_document_sync, document_id))

    async def verify_count(self, document_id: UUID, expected_count: int) -> int:
        """Strongly read back an ingest write and fail when its count differs."""

        expected_count = _require_integer(expected_count, "expected_count")
        actual_count = await self.count_document(document_id)
        if actual_count != expected_count:
            raise VectorCountMismatchError(
                "document vector count mismatch: "
                f"expected {expected_count}, observed {actual_count}"
            )
        return actual_count

    def _delete_document_sync(self, document_id: UUID) -> None:
        client = self._get_client_sync()
        self._verify_fingerprint_sync(client)
        document_filter = f"document_id == {_json_literal(str(document_id))}"
        client.delete(
            collection_name=self._collection_name,
            filter=document_filter,
            **self._timeout_kwargs(),
        )
        remaining = self._count_filter_sync(client, document_filter)
        if remaining:
            raise VectorCountMismatchError(
                f"document vector delete verification observed {remaining} remaining rows"
            )

    async def delete_document(self, document_id: UUID) -> None:
        self._assert_ready()
        document_id = _require_uuid(document_id, "document_id")
        await self._run_sync(self._delete_document_sync, document_id)

    @staticmethod
    def _load_state_name(load_state: Any) -> str:
        if isinstance(load_state, Mapping):
            load_state = load_state.get("state")
        name = getattr(load_state, "name", None)
        if isinstance(name, str):
            return name
        return str(load_state).rsplit(".", maxsplit=1)[-1]

    def _health_sync(self) -> DependencyHealth:
        client = self._get_client_sync()
        if not client.has_collection(
            collection_name=self._collection_name, **self._timeout_kwargs()
        ):
            return DependencyHealth(False, "collection_missing")
        self._describe_and_validate_sync(client)
        load_state = client.get_load_state(
            collection_name=self._collection_name, **self._timeout_kwargs()
        )
        if self._load_state_name(load_state).casefold() != "loaded":
            return DependencyHealth(False, "collection_not_loaded")
        self._verify_fingerprint_sync(client)
        return DependencyHealth(True, "ready")

    async def health(self) -> DependencyHealth:
        try:
            return await self._run_sync(self._health_sync)
        except Exception as exc:  # readiness reports failures instead of leaking client errors
            return DependencyHealth(False, f"unavailable:{type(exc).__name__}")

    def _close_client_sync(self) -> None:
        with self._client_lock:
            client = self._client
            self._client = None
        if client is not None:
            client.close()

    async def aclose(self) -> None:
        async with self._close_lock:
            with self._state_lock:
                if self._closed:
                    return
                self._closing = True

            acquired_slots = 0
            try:
                # Wait for all worker calls, including calls whose awaiting HTTP
                # coroutine was cancelled, before closing the shared client.
                for _ in range(self._executor_workers):
                    await self._operation_slots.acquire()
                    acquired_slots += 1

                if self._client is not None:
                    loop = asyncio.get_running_loop()
                    await asyncio.shield(
                        loop.run_in_executor(self._executor, self._close_client_sync)
                    )
                self._executor.shutdown(wait=True, cancel_futures=False)
            finally:
                for _ in range(acquired_slots):
                    self._operation_slots.release()
                self._token = ""
                self._ready = False
                with self._state_lock:
                    self._closed = True
                    self._closing = False


__all__ = [
    "DependencyHealth",
    "EmbeddingModelMismatchError",
    "KnowledgeChunkKind",
    "MilvusConfigurationError",
    "MilvusSchemaMismatchError",
    "MilvusStore",
    "MilvusStoreClosedError",
    "MilvusStoreError",
    "VectorCountMismatchError",
    "VectorHit",
    "VectorRecord",
    "VectorSearchScope",
    "create_milvus_store",
]
