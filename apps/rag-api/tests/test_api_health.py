from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from app.dependencies import get_health_checker
from app.main import _build_runtime, _shutdown_runtime, create_app
from app.routers.health import HealthChecker


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        api_prefix="/api/rag",
        api_key="",
        embedding_dimension=2,
        milvus_collection="rag_specs",
        reranker_enabled=False,
        llm_enabled=False,
    )


def _get(app: object, path: str) -> httpx.Response:
    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(path)

    return asyncio.run(run())


class _Session:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    async def execute(self, statement: object) -> None:
        self.calls.append(str(statement))


class _SessionContext:
    def __init__(self, session: object) -> None:
        self.session = session

    async def __aenter__(self) -> object:
        return self.session

    async def __aexit__(self, *_args: object) -> None:
        return None


class _HealthyStore:
    async def health(self) -> dict[str, object]:
        return {
            "ok": True,
            "collection_exists": True,
            "collection": "rag_specs",
            "loaded": True,
            "dimension": 2,
        }


def test_health_checks_postgres_milvus_and_provider_without_network_probe() -> None:
    calls: list[str] = []
    provider = SimpleNamespace(dimension=2, healthy=True, model_id="embedding-v1")
    checker = HealthChecker(
        session_factory=lambda: _SessionContext(_Session(calls)),
        store=_HealthyStore(),
        embedding_provider=provider,
        settings=_settings(),
    )
    app = create_app(_settings())
    app.dependency_overrides[get_health_checker] = lambda: checker

    response = _get(app, "/api/rag/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is True
    assert payload["postgres"]["ok"] is True
    assert payload["milvus"] == {
        "ok": True,
        "configured": True,
        "detail": None,
        "dimension": 2,
        "collection": "rag_specs",
        "loaded": True,
    }
    assert payload["providers"]["embedding"]["ok"] is True
    assert payload["providers"]["llm"]["configured"] is False
    assert calls == ["SELECT 1"]


class _LeakySession:
    async def execute(self, _statement: object) -> None:
        raise RuntimeError("postgresql://user:password@db/rag_kb?token=top-secret")


class _LeakyStore:
    async def health(self) -> dict[str, object]:
        raise RuntimeError("milvus token=top-secret")


def test_health_is_not_ready_and_never_leaks_dependency_secrets() -> None:
    checker = HealthChecker(
        session_factory=lambda: _SessionContext(_LeakySession()),
        store=_LeakyStore(),
        embedding_provider=SimpleNamespace(dimension=2, healthy=False),
        settings=_settings(),
    )
    app = create_app(_settings())
    app.dependency_overrides[get_health_checker] = lambda: checker

    response = _get(app, "/api/rag/health")

    assert response.status_code == 200
    assert response.json()["ready"] is False
    serialized = response.text.lower()
    assert "password" not in serialized
    assert "top-secret" not in serialized
    assert "postgresql://" not in serialized
    assert response.json()["postgres"]["detail"] == "unavailable"
    assert response.json()["milvus"]["detail"] == "unavailable"


def test_lifespan_composition_order_and_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import storage as storage_module
    from app.db import schema as schema_module
    from app.db import session as session_module
    from app.ingest import tasks as tasks_module
    from app.model_ai import embeddings as embeddings_module
    from app.vectordb import milvus_store as milvus_module

    calls: list[str] = []

    class Engine:
        async def dispose(self) -> None:
            calls.append("engine.close")

    engine = Engine()

    def create_engine(_settings: object) -> Engine:
        calls.append("engine")
        return engine

    async def assert_target(_engine: object) -> None:
        calls.append("database.target")

    async def assert_head(_engine: object) -> None:
        calls.append("database.migration")

    class Provider:
        model_id = "model"
        dimension = 1024

        async def aclose(self) -> None:
            calls.append("provider.close")

    def create_provider(settings: object) -> Provider:
        del settings
        calls.append("provider")
        return Provider()

    class Store:
        def __init__(
            self,
            *,
            embedding_model_id: str,
            uri: str,
            token: str,
            collection_name: str,
            dimension: int,
            batch_size: int,
            search_ef: int,
            executor_workers: int,
        ) -> None:
            del uri, token, dimension, batch_size, search_ef, executor_workers
            assert embedding_model_id == "model:1024"
            assert collection_name == "rag_specs"
            calls.append("store")

        async def ensure_collection(self) -> None:
            calls.append("store.ensure")

        async def aclose(self) -> None:
            calls.append("store.close")

    class Storage:
        def __init__(self, storage_dir: object, *, upload_max_bytes: int) -> None:
            del storage_dir, upload_max_bytes
            calls.append("storage")

    class Runner:
        async def start(self) -> None:
            calls.append("runner.start")

        async def stop(self) -> None:
            calls.append("runner.stop")

    def create_runner(settings: object) -> Runner:
        del settings
        calls.append("runner")
        return Runner()

    monkeypatch.setattr(session_module, "create_async_engine_from", create_engine)
    monkeypatch.setattr(session_module, "create_session_factory", lambda _engine: object())
    monkeypatch.setattr(schema_module, "assert_database_target", assert_target)
    monkeypatch.setattr(schema_module, "assert_migration_head", assert_head)
    monkeypatch.setattr(embeddings_module, "create_embedding_provider", create_provider)
    monkeypatch.setattr(
        embeddings_module,
        "embedding_fingerprint",
        lambda _provider: "model:1024",
    )
    monkeypatch.setattr(milvus_module, "MilvusStore", Store)
    monkeypatch.setattr(storage_module, "StorageManager", Storage)
    monkeypatch.setattr(tasks_module, "create_task_runner", create_runner, raising=False)

    settings = SimpleNamespace(
        embedding_provider="dashscope",
        embedding_model="model",
        embedding_dimension=1024,
        milvus_uri="http://milvus.invalid",
        milvus_token="",
        milvus_collection="rag_specs",
        milvus_batch_size=128,
        milvus_search_ef=64,
        reranker_enabled=False,
        llm_enabled=False,
        storage_dir="storage",
        upload_max_bytes=1_000,
    )
    runtime = asyncio.run(_build_runtime(settings))

    assert calls == [
        "engine",
        "database.target",
        "database.migration",
        "store",
        "store.ensure",
        "provider",
        "storage",
        "runner",
        "runner.start",
    ]

    app = create_app(SimpleNamespace(api_prefix="/api/rag", api_key=""))
    for name, value in runtime.items():
        setattr(app.state, name, value)
    asyncio.run(_shutdown_runtime(app))
    assert calls[-4:] == [
        "runner.stop",
        "provider.close",
        "store.close",
        "engine.close",
    ]
