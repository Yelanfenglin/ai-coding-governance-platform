"""FastAPI dependencies backed exclusively by ``request.app.state``."""

from __future__ import annotations

import hmac
from typing import Annotated, Any

from fastapi import Header, HTTPException, Request, status


def _state_value(request: Request, *names: str) -> Any:
    for name in names:
        try:
            value = getattr(request.app.state, name)
        except AttributeError:
            continue
        if value is not None:
            return value
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="service dependency is unavailable",
    )


def get_settings(request: Request) -> Any:
    return _state_value(request, "settings")


def get_session_factory(request: Request) -> Any:
    return _state_value(request, "session_factory")


def get_store(request: Request) -> Any:
    return _state_value(request, "store", "milvus_store")


def get_embedding_provider(request: Request) -> Any:
    return _state_value(request, "embedding_provider")


def get_reranker(request: Request) -> Any:
    return _state_value(request, "reranker")


def get_llm_provider(request: Request) -> Any:
    return _state_value(request, "llm_provider")


def get_storage_manager(request: Request) -> Any:
    return _state_value(request, "storage", "storage_manager")


def get_task_runner(request: Request) -> Any:
    return _state_value(request, "task_runner")


def get_document_service(request: Request) -> Any:
    return _state_value(request, "document_service")


def get_rag_service(request: Request) -> Any:
    return _state_value(request, "rag_service")


def get_health_checker(request: Request) -> Any:
    return _state_value(request, "health_checker")


def _configured_api_key(settings: object) -> str:
    value: object
    if isinstance(settings, dict):
        value = settings.get("api_key", "")
    else:
        value = getattr(settings, "api_key", "")
    if value is None:
        return ""
    reveal = getattr(value, "get_secret_value", None)
    if callable(reveal):
        value = reveal()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


async def require_api_key(
    request: Request,
    supplied_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Enforce the optional shared key without logging or early string compare."""

    settings = get_settings(request)
    expected = _configured_api_key(settings)
    if not expected:
        return
    supplied = supplied_key or ""
    if not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid API key",
            headers={"WWW-Authenticate": "APIKey"},
        )


# A descriptive alias is useful to callers that read this as a verifier rather
# than a router dependency.  Both names execute the same constant-time path.
verify_api_key = require_api_key


__all__ = [
    "get_document_service",
    "get_embedding_provider",
    "get_health_checker",
    "get_llm_provider",
    "get_rag_service",
    "get_reranker",
    "get_session_factory",
    "get_settings",
    "get_storage_manager",
    "get_store",
    "get_task_runner",
    "require_api_key",
    "verify_api_key",
]
