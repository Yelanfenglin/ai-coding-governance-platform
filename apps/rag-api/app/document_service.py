"""Document lifecycle rules shared by the HTTP and ingest entry points."""

from __future__ import annotations

import inspect
import re
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from .schemas import (
    DocumentListResponse,
    DocumentResponse,
    DocumentStatus,
    DocumentStatusResponse,
    JobAccepted,
    Language,
    normalize_product,
    normalize_version,
)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_ALLOWED_DELETE_STATUSES = {DocumentStatus.SUPERSEDED, DocumentStatus.FAILED}


class DocumentServiceError(RuntimeError):
    pass


class DocumentNotFoundError(DocumentServiceError):
    pass


class DocumentConflictError(DocumentServiceError):
    pass


class DocumentValidationError(DocumentServiceError):
    pass


class DocumentServiceUnavailableError(DocumentServiceError):
    pass


class DocumentDeleteError(DocumentServiceUnavailableError):
    pass


@dataclass(frozen=True, slots=True)
class DocumentSubmitCommand:
    path: Path
    original_filename: str
    sha256: str
    size_bytes: int
    product: str
    version: str
    language: Language = "zh-CN"
    title: str | None = None
    staging_id: UUID | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "product", normalize_product(self.product))
        object.__setattr__(self, "version", normalize_version(self.version))
        if self.language not in {"zh", "en", "zh-CN"}:
            raise ValueError("language must be zh, en, or zh-CN")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("sha256 must contain 64 hexadecimal characters")
        object.__setattr__(self, "sha256", self.sha256.lower())
        if self.size_bytes < 0:
            raise ValueError("size_bytes must not be negative")
        filename = Path(self.original_filename).name.strip()
        if not filename or "\x00" in filename:
            raise ValueError("original_filename is invalid")
        object.__setattr__(self, "original_filename", filename)
        if self.title is not None:
            title = self.title.strip()
            object.__setattr__(self, "title", title or None)


@dataclass(frozen=True, slots=True)
class DocumentFilters:
    product: str | None = None
    version: str | None = None
    current: bool | None = None

    def __post_init__(self) -> None:
        if self.product is not None:
            object.__setattr__(self, "product", normalize_product(self.product))
        if self.version is not None:
            object.__setattr__(self, "version", normalize_version(self.version))


@dataclass(frozen=True, slots=True)
class DocumentCreateData:
    product: str
    version: str
    language: str
    title: str
    filename: str
    file_sha256: str
    size_bytes: int
    source_path: Path
    product_family: str | None = None
    source_url: str | None = None
    stored_by: str | None = None
    ingest_stage: str = "pending"
    status: DocumentStatus = DocumentStatus.PENDING
    current: bool = False
    progress: int = 0


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


def _document_id(document: object) -> UUID:
    value = _get(document, "id", "document_id")
    return value if isinstance(value, UUID) else UUID(str(value))


def _status(value: object) -> DocumentStatus:
    raw = getattr(value, "value", value)
    return DocumentStatus(str(raw))


class DocumentService:
    """Persist document jobs and enforce replacement/deletion invariants."""

    def __init__(
        self,
        session_factory: Any = None,
        store: Any = None,
        storage: Any = None,
        task_runner: Any = None,
        *,
        repository_factory: Any = None,
        repository: Any = None,
    ) -> None:
        self._session_factory = session_factory
        self._store = store
        self._storage = storage
        self._task_runner = task_runner
        self._repository_factory = repository_factory
        self._repository = repository

    @asynccontextmanager
    async def _repo_scope(self, *, write: bool = False) -> AsyncIterator[Any]:
        if self._repository is not None:
            yield self._repository
            return
        if self._session_factory is None:
            raise DocumentServiceUnavailableError("repository is unavailable")

        session_context = await _maybe_await(self._session_factory())
        async with session_context as session:
            factory = self._repository_factory
            if factory is None:
                try:
                    from .db.repo import RagRepository
                except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - integration
                    raise DocumentServiceUnavailableError("repository is unavailable") from exc
                factory = RagRepository
            repo = factory(session)
            if write and hasattr(session, "begin"):
                transaction = session.begin()
                async with transaction:
                    yield repo
            else:
                yield repo

    async def submit(self, command: DocumentSubmitCommand) -> JobAccepted:
        """Idempotently register a pending candidate, then enqueue its Path."""

        existing: object | None = None
        candidate: object | None = None
        try:
            async with self._repo_scope(write=True) as repo:
                existing = await repo.find_identical(
                    command.product, command.version, command.sha256
                )
                if existing is None:
                    candidate = await repo.create_candidate(
                        DocumentCreateData(
                            product=command.product,
                            version=command.version,
                            language=command.language,
                            title=command.title or Path(command.original_filename).stem,
                            filename=command.original_filename,
                            file_sha256=command.sha256,
                            size_bytes=command.size_bytes,
                            source_path=command.path,
                        )
                    )
        except Exception as exc:
            if self._is_unique_violation(exc):
                existing = await self._find_identical(command)
            else:
                await self._cleanup_staging(command)
                if isinstance(exc, DocumentServiceError):
                    raise
                raise DocumentServiceUnavailableError("document registration failed") from exc

        if existing is not None:
            await self._cleanup_staging(command)
            return self._job_response(existing)
        if candidate is None:
            await self._cleanup_staging(command)
            raise DocumentServiceUnavailableError("document registration failed")

        document_id = _document_id(candidate)
        try:
            await self._submit_task(document_id, command.path)
        except Exception as exc:
            await self._mark_failed(document_id, "task submission failed")
            await self._cleanup_staging(command)
            raise DocumentServiceUnavailableError("document task submission failed") from exc
        return self._job_response(candidate)

    async def _find_identical(self, command: DocumentSubmitCommand) -> object | None:
        try:
            async with self._repo_scope() as repo:
                return await repo.find_identical(
                    command.product, command.version, command.sha256
                )
        except Exception as exc:
            raise DocumentServiceUnavailableError("document registration failed") from exc

    @staticmethod
    def _is_unique_violation(exc: Exception) -> bool:
        current: BaseException | None = exc
        while current is not None:
            if current.__class__.__name__ in {"IntegrityError", "UniqueViolationError"}:
                return True
            current = current.__cause__ or current.__context__
        return False

    async def _submit_task(self, document_id: UUID, path: Path) -> None:
        if self._task_runner is None:
            return
        submit_path = getattr(self._task_runner, "submit_path", None)
        if submit_path is not None:
            await _maybe_await(submit_path(document_id, Path(path)))
            return

        submit = getattr(self._task_runner, "submit", None)
        if submit is None:
            raise DocumentServiceUnavailableError("task runner is unavailable")
        try:
            parameters = tuple(inspect.signature(submit).parameters.values())
        except (TypeError, ValueError):
            parameters = ()
        accepts_path = any(
            parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters
        )
        accepts_path = accepts_path or len(parameters) >= 2
        if accepts_path:
            await _maybe_await(submit(document_id, Path(path)))
        else:
            # Recoverable runners resolve the persisted source path by document id.
            await _maybe_await(submit(document_id))

    async def _mark_failed(self, document_id: UUID, error: str) -> None:
        try:
            async with self._repo_scope(write=True) as repo:
                await repo.set_status(document_id, DocumentStatus.FAILED, error=error)
        except Exception:
            # The original enqueue error remains the primary failure.  Reconcile
            # can recover a pending row if this best-effort transition also fails.
            return

    async def _cleanup_staging(self, command: DocumentSubmitCommand) -> None:
        if self._storage is None:
            return
        cleanup = getattr(self._storage, "cleanup_staging", None)
        if cleanup is None:
            return
        target: object = command.staging_id if command.staging_id is not None else command.path
        try:
            await _maybe_await(cleanup(target))
        except Exception:
            return

    async def status(self, document_id: UUID) -> DocumentStatusResponse:
        document = await self._get_document(document_id)
        return DocumentStatusResponse(
            document_id=_document_id(document),
            status=_status(_get(document, "status")),
            progress=_get(document, "progress"),
            error=_get(document, "error", "parse_error"),
        )

    async def list(self, filters: DocumentFilters) -> DocumentListResponse:
        try:
            async with self._repo_scope() as repo:
                documents = await repo.list_documents(filters)
        except Exception as exc:
            if isinstance(exc, DocumentServiceError):
                raise
            raise DocumentServiceUnavailableError("document listing failed") from exc
        items = [self._document_response(document) for document in documents]
        return DocumentListResponse(items=items, total=len(items))

    async def _get_document(self, document_id: UUID, *, for_update: bool = False) -> object:
        try:
            async with self._repo_scope(write=for_update) as repo:
                document = await repo.get_document(document_id, for_update=for_update)
        except Exception as exc:
            if self._is_not_found(exc):
                raise DocumentNotFoundError("document was not found") from None
            if isinstance(exc, DocumentServiceError):
                raise
            raise DocumentServiceUnavailableError("document lookup failed") from exc
        if document is None:
            raise DocumentNotFoundError("document was not found")
        return document

    @staticmethod
    def _is_not_found(exc: Exception) -> bool:
        return isinstance(exc, (KeyError, LookupError)) or exc.__class__.__name__ in {
            "NoResultFound",
            "NotFoundError",
        }

    async def delete(self, document_id: UUID) -> None:
        """Delete Milvus -> assets -> PG, compensating in reverse on failure."""

        if self._store is None or self._storage is None:
            raise DocumentServiceUnavailableError("document deletion dependencies are unavailable")

        vector_snapshot: object | None = None
        assets_token: object | None = None
        vectors_deleted = False
        assets_deleted = False
        try:
            async with self._repo_scope(write=True) as repo:
                document = await repo.get_document(document_id, for_update=True)
                if document is None:
                    raise DocumentNotFoundError("document was not found")
                status = _status(_get(document, "status"))
                if bool(_get(document, "current", default=False)):
                    raise DocumentConflictError("the current document cannot be deleted")
                if status not in _ALLOWED_DELETE_STATUSES:
                    raise DocumentConflictError(
                        "only superseded or failed documents can be deleted"
                    )

                vector_snapshot = await self._snapshot_vectors(repo, document_id)
                await self._store.delete_document(document_id)
                vectors_deleted = True
                assets_token = await _maybe_await(
                    self._storage.delete_document_assets(document_id)
                )
                assets_deleted = True
                await repo.delete_document_row(document_id)
        except (DocumentNotFoundError, DocumentConflictError):
            raise
        except Exception as exc:
            if self._is_not_found(exc):
                raise DocumentNotFoundError("document was not found") from None
            compensated = await self._compensate_delete(
                document_id=document_id,
                vector_snapshot=vector_snapshot,
                assets_token=assets_token,
                restore_vectors=vectors_deleted,
                restore_assets=assets_deleted,
            )
            if not compensated:
                raise DocumentDeleteError(
                    "document deletion failed and compensation is incomplete"
                ) from exc
            raise DocumentDeleteError("document deletion failed and was compensated") from exc

        finalize = getattr(self._storage, "finalize_document_delete", None)
        if finalize is not None and assets_token is not None:
            try:
                await _maybe_await(finalize(document_id, assets_token))
            except Exception as exc:
                # The authoritative PG row is already gone.  A trash-retention
                # cleanup error is operational, not permission to recreate it.
                raise DocumentDeleteError("document deletion cleanup is incomplete") from exc

    async def _snapshot_vectors(self, repo: object, document_id: UUID) -> object | None:
        for owner, names in (
            (repo, ("get_vector_records_for_document", "vector_records_for_document")),
            (self._store, ("export_document", "snapshot_document")),
        ):
            for name in names:
                method = getattr(owner, name, None)
                if method is not None:
                    return await _maybe_await(method(document_id))
        return None

    async def _compensate_delete(
        self,
        *,
        document_id: UUID,
        vector_snapshot: object | None,
        assets_token: object | None,
        restore_vectors: bool,
        restore_assets: bool,
    ) -> bool:
        succeeded = True
        if restore_assets:
            restore = getattr(self._storage, "restore_document_assets", None)
            if restore is None:
                succeeded = False
            else:
                try:
                    await _maybe_await(restore(document_id, assets_token))
                except Exception:
                    succeeded = False
        if restore_vectors:
            if vector_snapshot is None or not hasattr(self._store, "upsert"):
                succeeded = False
            else:
                try:
                    await self._store.upsert(vector_snapshot)
                except Exception:
                    succeeded = False
        return succeeded

    @staticmethod
    def _job_response(document: object) -> JobAccepted:
        document_id = _document_id(document)
        return JobAccepted(
            job_id=document_id,
            document_id=document_id,
            status=_status(_get(document, "status")),
        )

    @staticmethod
    def _document_response(document: object) -> DocumentResponse:
        filename = str(_get(document, "filename", default=""))
        title = _get(document, "title") or Path(filename).stem
        return DocumentResponse(
            document_id=_document_id(document),
            status=_status(_get(document, "status")),
            progress=_get(document, "progress"),
            error=_get(document, "error", "parse_error"),
            product=str(_get(document, "product")),
            version=str(_get(document, "version")),
            language=_get(document, "language", default="zh-CN"),
            title=str(title),
            filename=filename,
            current=bool(_get(document, "current", default=False)),
            file_sha256=_get(document, "file_sha256", "sha256"),
            size_bytes=_get(document, "size_bytes"),
            created_at=_get(document, "created_at"),
            updated_at=_get(document, "updated_at"),
            superseded_at=_get(document, "superseded_at"),
        )


__all__ = [
    "DocumentConflictError",
    "DocumentCreateData",
    "DocumentDeleteError",
    "DocumentFilters",
    "DocumentNotFoundError",
    "DocumentService",
    "DocumentServiceError",
    "DocumentServiceUnavailableError",
    "DocumentSubmitCommand",
    "DocumentValidationError",
]
