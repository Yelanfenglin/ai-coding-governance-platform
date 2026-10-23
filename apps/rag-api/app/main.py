"""FastAPI application factory and resource lifecycle."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager, suppress
from contextvars import ContextVar
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import FastAPI

from .document_service import DocumentService
from .rag_service import RagQueryService
from .routers import documents_router, health_router, query_router
from .routers.health import HealthChecker


def _setting(settings: object, name: str, default: Any = None) -> Any:
    if isinstance(settings, Mapping):
        return settings.get(name, default)
    return getattr(settings, name, default)


def _secret_text(value: object) -> str:
    reveal = getattr(value, "get_secret_value", None)
    if callable(reveal):
        value = reveal()
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _configured_embedding_fingerprint(settings: object) -> str:
    provider_name = str(_setting(settings, "embedding_provider", "deterministic")).lower()
    dimension = int(_setting(settings, "embedding_dimension", 1024))
    if provider_name == "deterministic":
        seed = str(_setting(settings, "embedding_hash_seed", "rag-hash-v1"))
        model_id = f"deterministic-hash:{seed}"
    else:
        model_id = str(_setting(settings, "embedding_model", "")).strip()
    if not model_id:
        raise RuntimeError("embedding model fingerprint is not configured")
    return f"{model_id}:{dimension}"


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _resolve_settings(app: FastAPI) -> object:
    override = getattr(app.state, "settings_override", None)
    if override is not None:
        return override
    try:
        settings_module = importlib.import_module("app.settings")
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - integration wiring
        raise RuntimeError("RAG settings module is unavailable") from exc
    get_settings = getattr(settings_module, "get_settings", None)
    if get_settings is not None:
        return get_settings()
    settings_type = settings_module.Settings
    return settings_type()


def _component_target(module: object, names: tuple[str, ...]) -> Any:
    for name in names:
        target = getattr(module, name, None)
        if target is not None:
            return target
    raise RuntimeError(f"runtime component is unavailable: {names[0]}")


async def _construct_component(
    module_name: str,
    names: tuple[str, ...],
    available: Mapping[str, object],
) -> Any:
    """Construct an adapter using its documented factory or named dependencies."""

    module = importlib.import_module(module_name)
    target = _component_target(module, names)
    signature = inspect.signature(target)
    kwargs: dict[str, object] = {}
    settings = available.get("settings")
    missing: list[str] = []
    for parameter in signature.parameters.values():
        if parameter.name in {"self", "cls"}:
            continue
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if parameter.name in available:
            kwargs[parameter.name] = available[parameter.name]
        elif settings is not None and (
            isinstance(settings, Mapping) and parameter.name in settings
        ):
            kwargs[parameter.name] = settings[parameter.name]  # type: ignore[index]
        elif settings is not None and hasattr(settings, parameter.name):
            kwargs[parameter.name] = getattr(settings, parameter.name)
        elif parameter.default is inspect.Parameter.empty:
            missing.append(parameter.name)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"cannot construct {names[0]}; missing dependencies: {joined}")
    return await _maybe_await(target(**kwargs))


class _RepositoryGateway:
    """Give background jobs short sessions and one explicit activation transaction."""

    def __init__(self, session_factory: Any, repository_type: Any) -> None:
        self._session_factory = session_factory
        self._repository_type = repository_type
        self._active_session: ContextVar[object | None] = ContextVar(
            "rag_ingest_session",
            default=None,
        )

    async def _call(
        self,
        name: str,
        *args: object,
        write: bool = False,
        **kwargs: object,
    ) -> Any:
        active = self._active_session.get()
        if active is not None:
            method = getattr(self._repository_type(active), name)
            return await _maybe_await(method(*args, **kwargs))

        async with self._session_factory() as session:
            repository = self._repository_type(session)
            method = getattr(repository, name)
            if write:
                async with session.begin():
                    return await _maybe_await(method(*args, **kwargs))
            return await _maybe_await(method(*args, **kwargs))

    async def get_document(self, document_id: UUID, *, for_update: bool = False) -> object:
        return await self._call("get_document", document_id, for_update=for_update)

    async def set_status(
        self,
        document_id: UUID,
        status: object,
        *,
        error: str | None = None,
    ) -> None:
        await self._call("set_status", document_id, status, error=error, write=True)

    async def replace_chunks(self, document_id: UUID, records: object) -> None:
        await self._call("replace_chunks", document_id, records, write=True)

    async def activate_candidate(
        self,
        document_id: UUID,
        *,
        expected_vector_count: int,
    ) -> object:
        return await self._call(
            "activate_candidate",
            document_id,
            expected_vector_count=expected_vector_count,
            write=True,
        )

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        if self._active_session.get() is not None:
            yield
            return
        async with self._session_factory() as session, session.begin():
            token = self._active_session.set(session)
            try:
                yield
            finally:
                self._active_session.reset(token)

    async def fail_unrecoverable_jobs(self) -> None:
        """Make a restart-visible decision when no durable source path exists."""

        models_module = importlib.import_module("app.db.models")
        repo_module = importlib.import_module("app.db.repo")
        statuses = (
            models_module.DocumentStatus.PENDING,
            models_module.DocumentStatus.PARSING,
        )
        for status_value in statuses:
            while True:
                async with self._session_factory() as session, session.begin():
                    repository = self._repository_type(session)
                    documents = await repository.list_documents(
                        repo_module.DocumentFilters(
                            status=status_value,
                            limit=1_000,
                            offset=0,
                        )
                    )
                    for document in documents:
                        await repository.set_status(
                            document.id,
                            models_module.DocumentStatus.FAILED,
                            error="background source path was not recoverable after restart",
                        )
                if len(documents) < 1_000:
                    break


class _PipelineVectorStore:
    """Attach the process fingerprint to pipeline records before strict Milvus IO."""

    def __init__(self, store: object, embedding_model_id: str) -> None:
        self._store = store
        self._embedding_model_id = embedding_model_id

    async def upsert(self, records: object) -> int:
        store_module = importlib.import_module("app.vectordb.milvus_store")
        mapped = []
        for record in records:  # type: ignore[union-attr]
            if isinstance(record, store_module.VectorRecord):
                mapped.append(record)
                continue
            kind_value = getattr(record.kind, "value", record.kind)
            mapped.append(
                store_module.VectorRecord(
                    id=record.id,
                    vector=record.vector,
                    document_id=record.document_id,
                    product=record.product,
                    version=record.version,
                    kind=store_module.KnowledgeChunkKind(str(kind_value)),
                    semantic_type=record.semantic_type,
                    chunk_index=record.chunk_index,
                    page_start=record.page_start,
                    page_end=record.page_end,
                    section_path=record.section_path,
                    text=record.text,
                    embedding_model_id=self._embedding_model_id,
                )
            )
        return int(await self._store.upsert(mapped))  # type: ignore[attr-defined]

    async def count_document(self, document_id: UUID) -> int:
        return int(await self._store.count_document(document_id))  # type: ignore[attr-defined]

    async def delete_document(self, document_id: UUID) -> None:
        await self._store.delete_document(document_id)  # type: ignore[attr-defined]


class _UploadTaskRunner:
    """Bounded in-process upload runner whose queue contains UUIDs and Paths only."""

    _STOP = object()

    def __init__(
        self,
        *,
        repository: _RepositoryGateway,
        provider: object,
        store: _PipelineVectorStore,
        storage: object,
        settings: object,
    ) -> None:
        self._repository = repository
        self._provider = provider
        self._store = store
        self._storage = storage
        self._settings = settings
        self._concurrency = max(1, int(_setting(settings, "ingest_concurrency", 1)))
        self._queue: asyncio.Queue[object] = asyncio.Queue(maxsize=max(8, self._concurrency * 8))
        self._scheduled: set[UUID] = set()
        self._lock = asyncio.Lock()
        self._workers: list[asyncio.Task[None]] = []
        self._started = False
        self._closed = False

    async def submit_path(self, document_id: UUID, path: Path) -> UUID:
        if not isinstance(document_id, UUID):
            raise TypeError("document_id must be a UUID")
        source = Path(path)
        async with self._lock:
            if self._closed or not self._started:
                raise RuntimeError("background task runner is unavailable")
            if document_id in self._scheduled:
                return document_id
            self._scheduled.add(document_id)
            try:
                await self._queue.put((document_id, source))
            except BaseException:
                self._scheduled.discard(document_id)
                raise
        return document_id

    async def submit(self, document_id: UUID) -> UUID:
        del document_id
        raise RuntimeError("a persisted upload Path is required")

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("background task runner is stopped")
        if self._started:
            return
        await self._repository.fail_unrecoverable_jobs()
        self._workers = [
            asyncio.create_task(self._run(), name=f"rag-upload-worker-{index}")
            for index in range(self._concurrency)
        ]
        self._started = True

    async def stop(self) -> None:
        async with self._lock:
            if self._closed:
                workers = tuple(self._workers)
            else:
                self._closed = True
                workers = tuple(self._workers)
        if not workers:
            return
        try:
            await self._queue.join()
            for _ in workers:
                await self._queue.put(self._STOP)
            await asyncio.gather(*workers)
        except asyncio.CancelledError:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is self._STOP:
                    return
                document_id, path = item  # type: ignore[misc]
                await self._process(document_id, path)
            finally:
                if item is not self._STOP:
                    self._scheduled.discard(document_id)
                self._queue.task_done()

    async def _process(self, document_id: UUID, path: Path) -> None:
        pipeline = None
        try:
            document = await self._repository.get_document(document_id)
            layout_path = await self._layout_path(path)
            pipeline_module = importlib.import_module("app.ingest.pipeline")
            pipeline = pipeline_module.IngestPipeline(
                self._provider,
                self._store,
                self._repository,
                self._storage,
                embedding_batch_size=int(_setting(self._settings, "embedding_batch_size", 16)),
                upsert_batch_size=int(_setting(self._settings, "milvus_batch_size", 128)),
                embedding_dimension=int(_setting(self._settings, "embedding_dimension", 1024)),
            )
            await pipeline.run(
                pipeline_module.IngestCommand(
                    document_id=document_id,
                    pdf_path=path,
                    layout_path=layout_path,
                    product=str(document.product),
                    version=str(document.version),
                    language=str(document.language),
                    title=str(document.title),
                )
            )
        except Exception:
            with suppress(Exception):
                await self._repository.set_status(
                    document_id,
                    "failed",
                    error="background ingest failed",
                )
            self._cleanup_staging_path(path)
        finally:
            if pipeline is not None:
                with suppress(Exception):
                    await pipeline.aclose()

    async def _layout_path(self, pdf_path: Path) -> Path:
        candidates = (
            pdf_path.with_suffix(".layout.json"),
            pdf_path.with_suffix(".json"),
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        if str(_setting(self._settings, "layout_mode", "required")) != "auto":
            raise RuntimeError("a layout artifact is required for uploaded PDFs")

        layout_module = importlib.import_module("scripts.pp_doclayout_all")
        output = candidates[0]
        await asyncio.to_thread(
            layout_module.run,
            pdf_path,
            output,
            model_name=str(_setting(self._settings, "paddle_layout_model", "PP-DocLayoutV3")),
            dpi=int(_setting(self._settings, "render_dpi", 168)),
        )
        return output

    def _cleanup_staging_path(self, path: Path) -> None:
        with suppress(Exception):
            staging_id = UUID(path.parent.name)
            self._storage.cleanup_staging(staging_id)  # type: ignore[attr-defined]


async def _build_runtime(settings: object) -> dict[str, object]:
    """Build resources in the required fail-closed startup order."""

    runtime: dict[str, object] = {"settings": settings}
    try:
        return await _build_runtime_resources(settings, runtime)
    except BaseException:
        await _cleanup_partial_runtime(runtime)
        raise


async def _build_runtime_resources(
    settings: object,
    runtime: dict[str, object],
) -> dict[str, object]:

    session_module = importlib.import_module("app.db.session")
    schema_module = importlib.import_module("app.db.schema")
    create_engine = session_module.create_async_engine_from
    create_factory = session_module.create_session_factory
    assert_database_target = schema_module.assert_database_target
    assert_migration_head = schema_module.assert_migration_head

    engine = create_engine(settings)
    runtime["engine"] = engine
    # Isolation guard and migration head both complete before touching Milvus.
    await assert_database_target(engine)
    await assert_migration_head(engine)
    session_factory = create_factory(engine)
    runtime["session_factory"] = session_factory

    available: dict[str, object] = {
        "settings": settings,
        "engine": engine,
        "session_factory": session_factory,
    }
    configured_fingerprint = _configured_embedding_fingerprint(settings)
    available.update(
        {
            "embedding_model_id": configured_fingerprint,
            "uri": str(_setting(settings, "milvus_uri", "")),
            "token": _secret_text(_setting(settings, "milvus_token", "")),
            "collection_name": str(_setting(settings, "milvus_collection", "rag_specs")),
            "dimension": int(_setting(settings, "embedding_dimension", 1024)),
            "batch_size": int(_setting(settings, "milvus_batch_size", 128)),
            "search_ef": int(_setting(settings, "milvus_search_ef", 64)),
        }
    )
    store = await _construct_component(
        "app.vectordb.milvus_store",
        ("create_milvus_store", "MilvusStore"),
        available,
    )
    runtime.update({"store": store, "milvus_store": store})
    await store.ensure_collection()
    available.update({"store": store, "milvus_store": store})

    embedding_provider = await _construct_component(
        "app.model_ai.embeddings",
        ("create_embedding_provider",),
        available,
    )
    runtime["embedding_provider"] = embedding_provider
    try:
        embeddings_module = importlib.import_module("app.model_ai.embeddings")
        provider_fingerprint = embeddings_module.embedding_fingerprint(embedding_provider)
    except (AttributeError, ImportError, ModuleNotFoundError):
        provider_fingerprint = getattr(
            embedding_provider,
            "fingerprint",
            getattr(embedding_provider, "model_id", None),
        )
    if str(provider_fingerprint) != configured_fingerprint:
        raise RuntimeError("embedding provider fingerprint does not match the index")
    available.update(
        {
            "embedding_provider": embedding_provider,
            "provider": embedding_provider,
        }
    )

    reranker = None
    if bool(_setting(settings, "reranker_enabled", False)):
        reranker = await _construct_component(
            "app.model_ai.reranker",
            ("create_reranker", "HFReranker"),
            available,
        )
    if reranker is not None:
        runtime["reranker"] = reranker
    available["reranker"] = reranker

    llm_provider = None
    if bool(_setting(settings, "llm_enabled", False)):
        llm_provider = await _construct_component(
            "app.model_ai.llm",
            ("create_llm_provider", "create_llm", "HFChatProvider"),
            available,
        )
    if llm_provider is not None:
        runtime["llm_provider"] = llm_provider
    available["llm_provider"] = llm_provider

    planner = None
    if bool(_setting(settings, "planner_enabled", False)):
        from .planning import DashScopePlanner, DeepSeekPlanner

        provider = str(_setting(settings, "llm_provider", "dashscope")).strip().lower()
        model = str(
            _setting(settings, "planner_model", "") or _setting(settings, "llm_model", "")
        )
        if provider == "deepseek":
            planner = DeepSeekPlanner(
                settings.deepseek_api_key.get_secret_value(),
                model=model or "deepseek-chat",
                base_url=str(
                    _setting(settings, "deepseek_base_url", "https://api.deepseek.com")
                ),
                timeout_seconds=float(_setting(settings, "hf_timeout_seconds", 30.0)),
            )
        else:
            planner = DashScopePlanner(
                settings.dashscope_api_key.get_secret_value(),
                model=model or "qwen-plus",
                base_url=str(
                    _setting(settings, "dashscope_base_url", "https://dashscope.aliyuncs.com")
                ),
                timeout_seconds=float(_setting(settings, "hf_timeout_seconds", 30.0)),
            )
        runtime["planner"] = planner
    available["planner"] = planner

    storage = await _construct_component(
        "app.storage",
        ("create_storage_manager", "StorageManager"),
        available,
    )
    runtime.update({"storage": storage, "storage_manager": storage})
    available.update({"storage": storage, "storage_manager": storage})

    task_module = importlib.import_module("app.ingest.tasks")
    if getattr(task_module, "create_task_runner", None) is not None:
        task_runner = await _construct_component(
            "app.ingest.tasks",
            ("create_task_runner",),
            available,
        )
    else:
        repo_module = importlib.import_module("app.db.repo")
        repository_gateway = _RepositoryGateway(session_factory, repo_module.RagRepository)
        pipeline_store = _PipelineVectorStore(store, configured_fingerprint)
        task_runner = _UploadTaskRunner(
            repository=repository_gateway,
            provider=embedding_provider,
            store=pipeline_store,
            storage=storage,
            settings=settings,
        )
    available["task_runner"] = task_runner
    runtime["task_runner"] = task_runner

    rag_service = RagQueryService(
        session_factory=session_factory,
        store=store,
        embedding_provider=embedding_provider,
        reranker=reranker,
        llm_provider=llm_provider,
        planner=planner,
        settings=settings,
    )
    document_service = DocumentService(
        session_factory=session_factory,
        store=store,
        storage=storage,
        task_runner=task_runner,
    )
    health_checker = HealthChecker(
        session_factory=session_factory,
        engine=engine,
        store=store,
        embedding_provider=embedding_provider,
        reranker=reranker,
        llm_provider=llm_provider,
        settings=settings,
    )
    await task_runner.start()

    runtime.update(
        {
            "reranker": reranker,
            "llm_provider": llm_provider,
            "rag_service": rag_service,
            "document_service": document_service,
            "health_checker": health_checker,
        }
    )
    return runtime


async def _cleanup_partial_runtime(runtime: Mapping[str, object]) -> None:
    task_runner = runtime.get("task_runner")
    if task_runner is not None and hasattr(task_runner, "stop"):
        with suppress(Exception):
            await _maybe_await(task_runner.stop())  # type: ignore[attr-defined]

    seen: set[int] = set()
    for name in (
        "llm_provider",
        "reranker",
        "embedding_provider",
        "storage",
        "store",
        "engine",
    ):
        resource = runtime.get(name)
        if resource is None or id(resource) in seen:
            continue
        seen.add(id(resource))
        with suppress(Exception):
            await _close_one(resource)


async def _close_one(resource: object | None) -> None:
    if resource is None:
        return
    for name in ("aclose", "close", "dispose"):
        close = getattr(resource, name, None)
        if close is not None:
            await _maybe_await(close())
            return


async def _shutdown_runtime(app: FastAPI) -> None:
    task_runner = getattr(app.state, "task_runner", None)
    if task_runner is not None:
        stop = getattr(task_runner, "stop", None)
        if stop is not None:
            timeout = float(
                _setting(getattr(app.state, "settings", {}), "shutdown_timeout_seconds", 30)
            )
            with suppress(Exception):
                await asyncio.wait_for(_maybe_await(stop()), timeout=max(0.1, timeout))

    seen: set[int] = set()
    for name in (
        "llm_provider",
        "reranker",
        "embedding_provider",
        "storage",
        "store",
        "engine",
    ):
        resource = getattr(app.state, name, None)
        if resource is None or id(resource) in seen:
            continue
        seen.add(id(resource))
        try:
            await _close_one(resource)
        except Exception:
            continue


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = _resolve_settings(app)
    app.state.settings = settings
    app.state.ready = False
    try:
        runtime = await _build_runtime(settings)
        for name, value in runtime.items():
            setattr(app.state, name, value)
        # No request can reach a serving worker before this point in ASGI lifespan.
        app.state.ready = True
        yield
    finally:
        app.state.ready = False
        await _shutdown_runtime(app)


def _api_prefix(settings: object | None) -> str:
    raw = (
        _setting(settings, "api_prefix", "/api/rag")
        if settings is not None
        else os.getenv("RAG_API_PREFIX", "/api/rag")
    )
    prefix = str(raw).strip()
    if not prefix.startswith("/"):
        raise ValueError("api_prefix must start with '/'")
    if prefix != "/":
        prefix = prefix.rstrip("/")
    return prefix


def create_app(settings: object | None = None) -> FastAPI:
    application = FastAPI(title="RAG API", version="1.0.0", lifespan=lifespan)
    if settings is not None:
        # Makes dependency-overridden, lifespan-free ASGI tests deterministic.
        application.state.settings_override = settings
        application.state.settings = settings
    prefix = _api_prefix(settings)
    application.include_router(documents_router, prefix=prefix)
    application.include_router(query_router, prefix=prefix)
    application.include_router(health_router, prefix=prefix)
    return application


app = create_app()


__all__ = ["app", "create_app", "lifespan"]
