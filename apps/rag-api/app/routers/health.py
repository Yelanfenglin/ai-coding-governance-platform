"""Health and readiness endpoint."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from ..dependencies import get_health_checker, require_api_key
from ..schemas import DependencyHealth, HealthResponse

router = APIRouter(tags=["health"], dependencies=[Depends(require_api_key)])


def _get(value: object, *names: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _setting(settings: object | None, name: str, default: Any) -> Any:
    if settings is None:
        return default
    if isinstance(settings, Mapping):
        return settings.get(name, default)
    return getattr(settings, name, default)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class HealthChecker:
    """Perform cheap local checks; provider health never spends inference quota."""

    def __init__(
        self,
        *,
        session_factory: Any = None,
        engine: Any = None,
        store: Any = None,
        embedding_provider: Any = None,
        reranker: Any = None,
        llm_provider: Any = None,
        settings: object | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._engine = engine
        self._store = store
        self._embedding_provider = embedding_provider
        self._reranker = reranker
        self._llm_provider = llm_provider
        self._settings = settings

    async def check(self) -> HealthResponse:
        postgres = await self._postgres_health()
        milvus = await self._milvus_health()
        providers = self._provider_health()
        ready = postgres.ok and milvus.ok and all(
            health.ok for health in providers.values() if health.configured
        )
        return HealthResponse(
            ready=ready,
            postgres=postgres,
            milvus=milvus,
            providers=providers,
        )

    async def _postgres_health(self) -> DependencyHealth:
        try:
            try:
                from sqlalchemy import text

                statement: object = text("SELECT 1")
            except ImportError:  # pragma: no cover - SQLAlchemy is a runtime dependency
                statement = "SELECT 1"

            if self._session_factory is not None:
                session_context = await _maybe_await(self._session_factory())
                async with session_context as session:
                    await _maybe_await(session.execute(statement))
            elif self._engine is not None:
                connection_context = self._engine.connect()
                async with connection_context as connection:
                    await _maybe_await(connection.execute(statement))
            else:
                return DependencyHealth(ok=False, detail="not configured")
            return DependencyHealth(ok=True)
        except Exception:
            # Exception strings can contain a full SQLAlchemy URL.  The public
            # response intentionally reports only a fixed classification.
            return DependencyHealth(ok=False, detail="unavailable")

    async def _milvus_health(self) -> DependencyHealth:
        if self._store is None:
            return DependencyHealth(ok=False, detail="not configured")
        expected_dimension = int(_setting(self._settings, "embedding_dimension", 1024))
        expected_collection = str(
            _setting(self._settings, "milvus_collection", "rag_specs")
        )
        try:
            health_method = getattr(self._store, "health", None)
            if health_method is None:
                return DependencyHealth(ok=False, detail="health check unsupported")
            result = await _maybe_await(health_method())
            ok = bool(_get(result, "ok", "healthy", "ready", default=False))
            exists = bool(_get(result, "collection_exists", "exists", default=True))
            dimension = _get(
                result,
                "dimension",
                default=_get(
                    self._store,
                    "dimension",
                    "embedding_dimension",
                    default=expected_dimension,
                ),
            )
            collection = _get(
                result,
                "collection",
                "collection_name",
                default=_get(self._store, "collection", "collection_name"),
            )
            loaded = _get(
                result,
                "loaded",
                "is_loaded",
                default=_get(self._store, "loaded", "is_loaded", default=ok),
            )

            detail: str | None = None
            if not exists:
                ok = False
                detail = "collection missing"
            if dimension is not None and int(dimension) != expected_dimension:
                ok = False
                detail = "dimension mismatch"
            if collection is not None and str(collection) != expected_collection:
                ok = False
                detail = "collection mismatch"
            if loaded is False:
                ok = False
                detail = "collection not loaded"
            if not ok and detail is None:
                detail = "unavailable"
            return DependencyHealth(
                ok=ok,
                detail=detail,
                dimension=int(dimension) if dimension is not None else None,
                collection=str(collection) if collection is not None else expected_collection,
                loaded=bool(loaded) if loaded is not None else None,
            )
        except Exception:
            # PyMilvus errors may include a URI/token.  Never return their text.
            return DependencyHealth(
                ok=False,
                detail="unavailable",
                collection=expected_collection,
            )

    def _provider_health(self) -> dict[str, DependencyHealth]:
        expected_dimension = int(_setting(self._settings, "embedding_dimension", 1024))
        embedding = self._provider_status(
            self._embedding_provider,
            configured=True,
            expected_dimension=expected_dimension,
        )
        reranker_configured = bool(_setting(self._settings, "reranker_enabled", False))
        llm_configured = bool(_setting(self._settings, "llm_enabled", False))
        return {
            "embedding": embedding,
            "reranker": self._provider_status(
                self._reranker,
                configured=reranker_configured,
            ),
            "llm": self._provider_status(
                self._llm_provider,
                configured=llm_configured,
            ),
        }

    @staticmethod
    def _provider_status(
        provider: object | None,
        *,
        configured: bool,
        expected_dimension: int | None = None,
    ) -> DependencyHealth:
        if not configured:
            return DependencyHealth(ok=True, configured=False, detail="disabled")
        if provider is None:
            return DependencyHealth(ok=False, configured=True, detail="not configured")

        healthy = _get(provider, "healthy", "available", "last_call_ok", default=True)
        status_value = _get(provider, "status")
        if isinstance(status_value, str) and status_value.lower() in {
            "failed",
            "unavailable",
            "error",
        }:
            healthy = False
        dimension = _get(provider, "dimension")
        detail: str | None = None
        if (
            expected_dimension is not None
            and dimension is not None
            and int(dimension) != expected_dimension
        ):
            healthy = False
            detail = "dimension mismatch"
        if not bool(healthy) and detail is None:
            # Do not expose provider last_error; it can include a bearer token.
            detail = "unavailable"
        return DependencyHealth(
            ok=bool(healthy),
            configured=True,
            detail=detail,
            dimension=int(dimension) if dimension is not None else None,
        )


@router.get("/health", response_model=HealthResponse)
async def health(
    checker: Annotated[HealthChecker, Depends(get_health_checker)],
) -> HealthResponse:
    return await checker.check()


__all__ = ["HealthChecker", "health", "router"]
