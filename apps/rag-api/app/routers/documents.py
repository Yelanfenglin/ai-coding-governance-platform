"""Document upload and lifecycle routes."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, NoReturn
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)

from ..dependencies import get_document_service, get_storage_manager, require_api_key
from ..document_service import (
    DocumentConflictError,
    DocumentDeleteError,
    DocumentFilters,
    DocumentNotFoundError,
    DocumentService,
    DocumentServiceUnavailableError,
    DocumentSubmitCommand,
    DocumentValidationError,
)
from ..schemas import (
    DocumentListResponse,
    DocumentStatusResponse,
    JobAccepted,
    Language,
    normalize_product,
    normalize_version,
)

router = APIRouter(tags=["documents"], dependencies=[Depends(require_api_key)])


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


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _raise_service_http(exc: Exception) -> NoReturn:
    if isinstance(exc, DocumentNotFoundError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from None
    if isinstance(exc, DocumentConflictError):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from None
    if isinstance(exc, (DocumentValidationError, ValueError)):
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from None
    if isinstance(exc, (DocumentDeleteError, DocumentServiceUnavailableError)):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from None
    raise exc


async def _cleanup_failed_upload(storage: object, staging_id: UUID) -> None:
    cleanup = getattr(storage, "cleanup_staging", None)
    if cleanup is None:
        return
    try:
        await _maybe_await(cleanup(staging_id))
    except Exception:
        return


def _upload_http_error(exc: Exception) -> HTTPException:
    name = exc.__class__.__name__.lower()
    safe_message = str(exc).lower()
    if (
        "toolarge" in name
        or "sizelimit" in name
        or ("upload" in name and "exceeds" in safe_message)
    ):
        return HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="uploaded PDF exceeds the configured size limit",
        )
    if isinstance(exc, ValueError) or any(
        marker in name for marker in ("invalidupload", "invalidpdf", "unsupported")
    ):
        return HTTPException(
            status_code=422,
            detail="uploaded file is not a valid PDF",
        )
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="upload storage is unavailable",
    )


@router.post("/documents", status_code=status.HTTP_202_ACCEPTED, response_model=JobAccepted)
async def create_document(
    file: Annotated[UploadFile, File()],
    product: Annotated[str, Form(min_length=1, max_length=128)],
    version: Annotated[str, Form(min_length=1, max_length=64)],
    service: Annotated[DocumentService, Depends(get_document_service)],
    storage: Annotated[object, Depends(get_storage_manager)],
    language: Annotated[Language, Form()] = "zh-CN",
    title: Annotated[str | None, Form(max_length=512)] = None,
) -> JobAccepted:
    # Reject malformed scope before consuming and persisting a potentially large
    # request body.  StorageManager owns size/magic/path containment checks.
    client_name = (file.filename or "").replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    if not client_name.lower().endswith(".pdf"):
        await file.close()
        raise HTTPException(
            status_code=422,
            detail="uploaded file must use the .pdf extension",
        )
    try:
        normalized_product = normalize_product(product)
        normalized_version = normalize_version(version)
    except ValueError as exc:
        await file.close()
        _raise_service_http(exc)

    staging_id = uuid4()
    stored: object
    try:
        stored = await storage.save_upload(file, staging_id)
    except Exception as exc:
        await _cleanup_failed_upload(storage, staging_id)
        raise _upload_http_error(exc) from None
    finally:
        # The background service is invoked only after this await.  No task can
        # retain an UploadFile whose underlying temporary handle is now closed.
        await file.close()

    try:
        path = Path(_get(stored, "path"))
        original_filename = str(
            _get(stored, "original_filename", default=file.filename or "upload.pdf")
        )
        command = DocumentSubmitCommand(
            path=path,
            original_filename=original_filename,
            sha256=str(_get(stored, "sha256")),
            size_bytes=int(_get(stored, "size_bytes")),
            product=normalized_product,
            version=normalized_version,
            language=language,
            title=title,
            staging_id=staging_id,
        )
        return await service.submit(command)
    except (
        DocumentConflictError,
        DocumentDeleteError,
        DocumentNotFoundError,
        DocumentServiceUnavailableError,
        DocumentValidationError,
        TypeError,
        ValueError,
    ) as exc:
        await _cleanup_failed_upload(storage, staging_id)
        _raise_service_http(exc)


@router.get("/documents/{doc_id}/status", response_model=DocumentStatusResponse)
async def document_status(
    doc_id: UUID,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentStatusResponse:
    try:
        return await service.status(doc_id)
    except (DocumentNotFoundError, DocumentServiceUnavailableError) as exc:
        _raise_service_http(exc)


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    service: Annotated[DocumentService, Depends(get_document_service)],
    product: Annotated[str | None, Query()] = None,
    version: Annotated[str | None, Query()] = None,
    current: Annotated[bool | None, Query()] = None,
) -> DocumentListResponse:
    try:
        return await service.list(
            DocumentFilters(product=product, version=version, current=current)
        )
    except (DocumentServiceUnavailableError, ValueError) as exc:
        _raise_service_http(exc)


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    doc_id: UUID,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> Response:
    try:
        await service.delete(doc_id)
    except (
        DocumentConflictError,
        DocumentDeleteError,
        DocumentNotFoundError,
        DocumentServiceUnavailableError,
    ) as exc:
        _raise_service_http(exc)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


__all__ = [
    "create_document",
    "delete_document",
    "document_status",
    "list_documents",
    "router",
]
