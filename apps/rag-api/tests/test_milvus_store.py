from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from app.query_filters import FilterCondition
from app.vectordb.milvus_store import (
    DEFAULT_MILVUS_URI,
    EmbeddingModelMismatchError,
    KnowledgeChunkKind,
    MilvusConfigurationError,
    MilvusSchemaMismatchError,
    MilvusStore,
    VectorCountMismatchError,
    VectorRecord,
    VectorSearchScope,
    _build_collection_schema,
    _build_index_params,
    _build_search_filter,
    _collection_field_specs,
    _validate_collection_description,
    _validate_index_description,
    create_milvus_store,
)

MODEL_ID = "deterministic:test:v1"


def _collection_description(
    *, collection_name: str = "rag_specs", dimension: int = 1024
) -> dict[str, Any]:
    fields: list[dict[str, Any]] = []
    for spec in _collection_field_specs(dimension, collection_name):
        params: dict[str, Any] = {}
        if spec.dim is not None:
            params["dim"] = spec.dim
        if spec.max_length is not None:
            params["max_length"] = spec.max_length
        if spec.max_capacity is not None:
            params["max_capacity"] = spec.max_capacity
        field: dict[str, Any] = {"name": spec.name, "type": spec.data_type, "params": params}
        if spec.element_type is not None:
            field["element_type"] = spec.element_type
        if spec.is_primary:
            field["is_primary"] = True
        fields.append(field)
    return {
        "collection_name": collection_name,
        "auto_id": False,
        "enable_dynamic_field": False,
        "struct_array_fields": [],
        "fields": fields,
    }


def test_v2_schema_is_distinct_and_keeps_existing_fields() -> None:
    old_names = {spec.name for spec in _collection_field_specs(1024, "rag_specs")}
    v2_names = {spec.name for spec in _collection_field_specs(1024, "rag_specs_v2")}
    assert old_names < v2_names
    expected = {"section_id", "register", "packages", "chunk_index", "page_start", "page_end"}
    assert expected <= v2_names
    packages = next(
        spec
        for spec in _collection_field_specs(1024, "rag_specs_v2")
        if spec.name == "packages"
    )
    assert packages.data_type == "ARRAY"
    assert packages.element_type == "VARCHAR"
    assert packages.max_capacity == 8
    _validate_collection_description(
        _collection_description(collection_name="rag_specs_v2"), 1024, "rag_specs_v2"
    )


def _index_description(*, metric_type: str = "COSINE") -> dict[str, Any]:
    return {
        "field_name": "vector",
        "index_name": "vector_hnsw",
        "index_type": "HNSW",
        "metric_type": metric_type,
        "params": {"M": 16, "efConstruction": 200},
        "state": "Finished",
    }


def _record(
    document_id: UUID,
    index: int,
    *,
    kind: KnowledgeChunkKind = KnowledgeChunkKind.TEXT,
    embedding_model_id: str = MODEL_ID,
) -> VectorRecord:
    return VectorRecord(
        id=f"{document_id}:{index}",
        vector=(1.0,) + (0.0,) * 1023,
        document_id=document_id,
        product="ADC",
        version="V1.0",
        kind=kind,
        semantic_type=kind.value.casefold(),
        chunk_index=index,
        page_start=index + 1,
        page_end=index + 1,
        section_path="3 / Registers",
        text=f"chunk {index}",
        embedding_model_id=embedding_model_id,
    )


class _FakeSchema:
    def __init__(self) -> None:
        self.fields: list[dict[str, Any]] = []

    def add_field(self, **kwargs: Any) -> None:
        self.fields.append(kwargs)


class _FakeIndexParams:
    def __init__(self) -> None:
        self.indexes: list[dict[str, Any]] = []

    def add_index(self, **kwargs: Any) -> None:
        self.indexes.append(kwargs)


class _BlueprintClient:
    def __init__(self) -> None:
        self.schema_kwargs: dict[str, Any] | None = None
        self.schema = _FakeSchema()
        self.index_params = _FakeIndexParams()

    def create_schema(self, **kwargs: Any) -> _FakeSchema:
        self.schema_kwargs = kwargs
        return self.schema

    def prepare_index_params(self) -> _FakeIndexParams:
        return self.index_params


class _FakeMilvusClient:
    def __init__(self, *, collection_name: str = "rag_specs") -> None:
        self.collection_name = collection_name
        self.description = _collection_description(collection_name=collection_name)
        self.index_description = _index_description()
        self.rows: dict[str, dict[str, Any]] = {}
        self.upsert_batch_sizes: list[int] = []
        self.search_calls: list[dict[str, Any]] = []
        self.query_calls: list[dict[str, Any]] = []
        self.call_threads: list[int] = []
        self.create_calls = 0
        self.closed = False
        self.loaded = False
        self.fingerprint_mismatch = False
        self.count_override: int | None = None

    def _called(self) -> None:
        self.call_threads.append(threading.get_ident())

    def has_collection(self, **kwargs: Any) -> bool:
        self._called()
        assert kwargs["collection_name"] == self.collection_name
        return True

    def describe_collection(self, **kwargs: Any) -> dict[str, Any]:
        self._called()
        return self.description

    def list_indexes(self, **kwargs: Any) -> list[str]:
        self._called()
        assert kwargs["field_name"] == "vector"
        return ["vector_hnsw"]

    def describe_index(self, **kwargs: Any) -> dict[str, Any]:
        self._called()
        assert kwargs["index_name"] == "vector_hnsw"
        return self.index_description

    def create_collection(self, **kwargs: Any) -> None:
        self._called()
        self.create_calls += 1

    def load_collection(self, **kwargs: Any) -> None:
        self._called()
        self.loaded = True

    def get_load_state(self, **kwargs: Any) -> dict[str, str]:
        self._called()
        return {"state": "Loaded" if self.loaded else "NotLoad"}

    def query(self, **kwargs: Any) -> list[dict[str, Any]]:
        self._called()
        self.query_calls.append(kwargs)
        assert kwargs["consistency_level"] == "Strong"
        if kwargs["output_fields"] == ["id", "embedding_model_id"]:
            if self.fingerprint_mismatch:
                return [{"id": "other:0", "embedding_model_id": "other-model"}]
            return []
        if kwargs["output_fields"] == ["count(*)"]:
            if self.count_override is not None:
                return [{"count(*)": self.count_override}]
            expression = kwargs["filter"]
            matches = [row for row in self.rows.values() if row["document_id"] in expression]
            return [{"count(*)": len(matches)}]
        raise AssertionError(f"unexpected query: {kwargs}")

    def upsert(self, **kwargs: Any) -> dict[str, Any]:
        self._called()
        batch = kwargs["data"]
        self.upsert_batch_sizes.append(len(batch))
        for row in batch:
            self.rows[row["id"]] = dict(row)
        return {"upsert_count": len(batch), "ids": [row["id"] for row in batch]}

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self._called()
        self.search_calls.append(kwargs)
        expression = kwargs["filter"]
        hits: list[dict[str, Any]] = []
        for row in self.rows.values():
            if row["document_id"] not in expression:
                continue
            kind_match = re.search(r"kind in (\[[^]]*])", expression)
            if kind_match and row["kind"] not in json.loads(kind_match.group(1)):
                continue
            entity = {key: value for key, value in row.items() if key != "vector"}
            hits.append({"id": row["id"], "distance": 0.95, "entity": entity})
        return [hits[: kwargs["limit"]]]

    def delete(self, **kwargs: Any) -> dict[str, int]:
        self._called()
        expression = kwargs["filter"]
        deleted = [key for key, row in self.rows.items() if row["document_id"] in expression]
        for key in deleted:
            del self.rows[key]
        return {"delete_count": len(deleted)}

    def close(self) -> None:
        self._called()
        self.closed = True


class _ClientFactory:
    def __init__(self, client: _FakeMilvusClient) -> None:
        self.client = client
        self.kwargs: dict[str, Any] | None = None
        self.thread_id: int | None = None

    def __call__(self, **kwargs: Any) -> _FakeMilvusClient:
        self.kwargs = kwargs
        self.thread_id = threading.get_ident()
        return self.client


def test_vector_contract_is_frozen_and_includes_code() -> None:
    document_id = uuid4()
    record = _record(document_id, 0, kind=KnowledgeChunkKind.CODE)
    source_document_ids = [document_id]
    scope = VectorSearchScope(source_document_ids, [KnowledgeChunkKind.CODE])

    source_document_ids.clear()
    assert scope.document_ids == (document_id,)
    assert scope.kinds == (KnowledgeChunkKind.CODE,)
    assert record.embedding_model_id == MODEL_ID
    with pytest.raises(FrozenInstanceError):
        record.id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError, match="UUID"):
        VectorSearchScope(["not-a-uuid"])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="KnowledgeChunkKind"):
        VectorSearchScope([document_id], ["CODE"])  # type: ignore[list-item]


def test_schema_and_index_builders_match_the_frozen_contract() -> None:
    client = _BlueprintClient()
    data_type = SimpleNamespace(
        VARCHAR="VARCHAR", FLOAT_VECTOR="FLOAT_VECTOR", INT64="INT64"
    )

    schema = _build_collection_schema(client, data_type, 1024)
    index_params = _build_index_params(client)

    assert client.schema_kwargs == {"auto_id": False, "enable_dynamic_field": False}
    assert schema.fields[0] == {
        "field_name": "id",
        "datatype": "VARCHAR",
        "is_primary": True,
        "max_length": 512,
    }
    assert [field["field_name"] for field in schema.fields] == [
        "id",
        "vector",
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
    ]
    assert schema.fields[1]["dim"] == 1024
    assert index_params.indexes == [
        {
            "field_name": "vector",
            "index_name": "vector_hnsw",
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        }
    ]


def test_existing_schema_and_index_are_validated_strictly() -> None:
    description = _collection_description()
    _validate_collection_description(description, 1024, "rag_specs")
    _validate_index_description(_index_description())

    dynamic = _collection_description()
    dynamic["enable_dynamic_field"] = True
    with pytest.raises(MilvusSchemaMismatchError, match="dynamic"):
        _validate_collection_description(dynamic, 1024, "rag_specs")

    wrong_dim = _collection_description()
    next(field for field in wrong_dim["fields"] if field["name"] == "vector")["params"][
        "dim"
    ] = 768
    with pytest.raises(MilvusSchemaMismatchError, match="dimension"):
        _validate_collection_description(wrong_dim, 1024, "rag_specs")

    missing_fingerprint = _collection_description()
    missing_fingerprint["fields"] = [
        field
        for field in missing_fingerprint["fields"]
        if field["name"] != "embedding_model_id"
    ]
    with pytest.raises(MilvusSchemaMismatchError, match="field set"):
        _validate_collection_description(missing_fingerprint, 1024, "rag_specs")

    with pytest.raises(MilvusSchemaMismatchError, match="COSINE"):
        _validate_index_description(_index_description(metric_type="L2"))


def test_filter_builder_accepts_only_typed_scope_and_escapes_fingerprint() -> None:
    first = UUID("11111111-1111-1111-1111-111111111111")
    second = UUID("22222222-2222-2222-2222-222222222222")
    scope = VectorSearchScope(
        [second, first, second], [KnowledgeChunkKind.TEXT, KnowledgeChunkKind.CODE]
    )

    expression = _build_search_filter(scope, 'model" or true')

    assert expression == (
        'document_id in ["11111111-1111-1111-1111-111111111111",'
        '"22222222-2222-2222-2222-222222222222"] and '
        'kind in ["CODE","TEXT"] and embedding_model_id == "model\\" or true"'
    )
    assert "model\" or true" not in expression
    with pytest.raises(ValueError, match="at least one"):
        _build_search_filter(VectorSearchScope([]), MODEL_ID)
    with pytest.raises(TypeError, match="VectorSearchScope"):
        _build_search_filter("document_id == '*'", MODEL_ID)  # type: ignore[arg-type]


def test_store_uses_bounded_executor_and_strong_consistency_offline() -> None:
    async def scenario() -> None:
        main_thread = threading.get_ident()
        client = _FakeMilvusClient()
        factory = _ClientFactory(client)
        store = MilvusStore(
            embedding_model_id=MODEL_ID,
            token="do-not-log-this-token",
            batch_size=2,
            executor_workers=2,
            client_factory=factory,
        )
        document_id = uuid4()
        records = [
            _record(document_id, 0),
            _record(document_id, 1, kind=KnowledgeChunkKind.CODE),
            _record(document_id, 2),
        ]

        await store.ensure_collection()
        assert await store.upsert(records) == 3
        assert client.upsert_batch_sizes == [2, 1]
        assert await store.verify_count(document_id, 3) == 3

        hits = await store.search(
            records[0].vector,
            VectorSearchScope([document_id], [KnowledgeChunkKind.CODE]),
            limit=5,
        )
        assert [hit.id for hit in hits] == [records[1].id]
        assert hits[0].embedding_id == records[1].id
        search_call = client.search_calls[-1]
        assert search_call["search_params"] == {
            "metric_type": "COSINE",
            "params": {"ef": 64},
        }
        assert search_call["consistency_level"] == "Strong"
        assert "document_id in" in search_call["filter"]
        assert "kind in" in search_call["filter"]
        assert "embedding_model_id ==" in search_call["filter"]

        health = await store.health()
        assert health.ready is True
        assert health.detail == "ready"
        await store.delete_document(document_id)
        assert await store.count_document(document_id) == 0

        assert factory.kwargs == {"uri": DEFAULT_MILVUS_URI, "token": "do-not-log-this-token"}
        assert factory.thread_id != main_thread
        assert client.call_threads
        assert all(thread_id != main_thread for thread_id in client.call_threads)
        assert "do-not-log-this-token" not in repr(store)
        assert DEFAULT_MILVUS_URI not in repr(store)

        await store.aclose()
        await store.aclose()
        assert client.closed is True

    asyncio.run(scenario())


def test_search_pushes_package_filter_only_on_v2() -> None:
    async def scenario() -> None:
        document_id = uuid4()
        for collection, expect_push in (("rag_specs_v2", True), ("rag_specs", False)):
            client = _FakeMilvusClient(collection_name=collection)
            store = MilvusStore(
                embedding_model_id=MODEL_ID,
                collection_name=collection,
                client_factory=_ClientFactory(client),
            )
            await store.ensure_collection()
            await store.upsert([replace(_record(document_id, 0), packages=("SOP20",))])
            hits = await store.search(
                (1.0,) + (0.0,) * 1023,
                VectorSearchScope([document_id]),
                5,
                filters=(FilterCondition("package", "eq", "SOP20"),),
            )
            expression = client.search_calls[-1]["filter"]
            assert ('array_contains(packages, "SOP20")' in expression) is expect_push
            assert len(hits) == 1
            await store.aclose()

    asyncio.run(scenario())


def test_empty_scope_never_calls_milvus() -> None:
    async def scenario() -> None:
        client = _FakeMilvusClient()
        store = MilvusStore(
            embedding_model_id=MODEL_ID,
            client_factory=_ClientFactory(client),
        )
        hits = await store.search((0.0,) * 1024, VectorSearchScope([]), limit=5)
        assert hits == []
        assert client.call_threads == []
        await store.aclose()

    asyncio.run(scenario())


def test_fingerprint_mismatches_are_rejected_before_writes_and_at_startup() -> None:
    async def wrong_record_scenario() -> None:
        client = _FakeMilvusClient()
        store = MilvusStore(
            embedding_model_id=MODEL_ID,
            client_factory=_ClientFactory(client),
        )
        await store.ensure_collection()
        with pytest.raises(EmbeddingModelMismatchError, match="record"):
            await store.upsert([_record(uuid4(), 0, embedding_model_id="other-model")])
        assert client.upsert_batch_sizes == []
        await store.aclose()

    async def existing_mismatch_scenario() -> None:
        client = _FakeMilvusClient()
        client.fingerprint_mismatch = True
        store = MilvusStore(
            embedding_model_id=MODEL_ID,
            client_factory=_ClientFactory(client),
        )
        with pytest.raises(EmbeddingModelMismatchError, match="different"):
            await store.ensure_collection()
        assert client.create_calls == 0
        await store.aclose()

    asyncio.run(wrong_record_scenario())
    asyncio.run(existing_mismatch_scenario())


def test_existing_schema_mismatch_fails_without_recreate() -> None:
    async def scenario() -> None:
        client = _FakeMilvusClient()
        next(
            field for field in client.description["fields"] if field["name"] == "vector"
        )["params"]["dim"] = 768
        store = MilvusStore(
            embedding_model_id=MODEL_ID,
            client_factory=_ClientFactory(client),
        )
        with pytest.raises(MilvusSchemaMismatchError, match="dimension"):
            await store.ensure_collection()
        assert client.create_calls == 0
        assert not hasattr(client, "drop_collection")
        await store.aclose()

    asyncio.run(scenario())


def test_verify_count_raises_on_strong_readback_mismatch() -> None:
    async def scenario() -> None:
        client = _FakeMilvusClient()
        store = MilvusStore(
            embedding_model_id=MODEL_ID,
            client_factory=_ClientFactory(client),
        )
        document_id = uuid4()
        await store.ensure_collection()
        await store.upsert([_record(document_id, 0)])
        client.count_override = 0
        with pytest.raises(VectorCountMismatchError, match="expected 1, observed 0"):
            await store.verify_count(document_id, 1)
        await store.aclose()

    asyncio.run(scenario())


def test_stockpilot_collection_name_is_refused() -> None:
    with pytest.raises(MilvusConfigurationError, match="stockpilot"):
        MilvusStore(
            embedding_model_id=MODEL_ID,
            collection_name="stockpilot_vectors",
        )
    with pytest.raises(MilvusConfigurationError, match="rag_specs"):
        MilvusStore(
            embedding_model_id=MODEL_ID,
            collection_name="some_other_project",
        )


def test_settings_factory_derives_the_same_provider_fingerprint() -> None:
    class Secret:
        def get_secret_value(self) -> str:
            return "factory-test-token"

    settings = SimpleNamespace(
        milvus_uri=DEFAULT_MILVUS_URI,
        milvus_token=Secret(),
        milvus_collection="rag_specs",
        milvus_batch_size=128,
        milvus_search_ef=64,
        embedding_provider="deterministic",
        embedding_hash_seed="rag-hash-v1",
        embedding_dimension=1024,
    )

    store = create_milvus_store(settings)

    assert store.collection_name == "rag_specs"
    assert store.dimension == 1024
    assert store.embedding_model_id == "deterministic-hash:rag-hash-v1:1024"
    assert "factory-test-token" not in repr(store)
    asyncio.run(store.aclose())


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RAG_TEST_MILVUS") != "1",
    reason="set RAG_TEST_MILVUS=1 to run against the isolated test Milvus",
)
def test_real_milvus_round_trip() -> None:
    pymilvus = pytest.importorskip("pymilvus")
    if pymilvus.__version__ != "2.6.17":
        pytest.skip("integration test requires pymilvus==2.6.17")

    from pymilvus import MilvusClient

    uri = os.getenv("RAG_MILVUS_URI", DEFAULT_MILVUS_URI)
    token = os.getenv("RAG_MILVUS_TOKEN", "")
    collection_name = f"rag_specs_test_{uuid4().hex}"
    model_id = "integration:deterministic:v1"
    document_id = uuid4()

    async def scenario() -> None:
        store = MilvusStore(
            uri=uri,
            token=token,
            collection_name=collection_name,
            embedding_model_id=model_id,
        )
        record = _record(document_id, 0, embedding_model_id=model_id)
        try:
            await store.ensure_collection()
            assert await store.upsert([record]) == 1
            assert await store.verify_count(document_id, 1) == 1
            hits = await store.search(
                record.vector, VectorSearchScope([document_id]), limit=1
            )
            assert hits and hits[0].id == record.id
            assert (await store.health()).ready is True
            await store.delete_document(document_id)
            assert await store.count_document(document_id) == 0
        finally:
            await store.aclose()

    try:
        asyncio.run(scenario())
    finally:
        # The randomized name is created by this test only.  Never enumerate or
        # drop any other collection on the instance.
        cleanup = MilvusClient(uri=uri, token=token)
        try:
            if collection_name.startswith("rag_specs_test_") and cleanup.has_collection(
                collection_name
            ):
                cleanup.drop_collection(collection_name)
        finally:
            cleanup.close()
