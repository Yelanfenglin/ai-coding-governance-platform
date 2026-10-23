"""Command-line entry points for document ingestion and maintenance.

The CLI is deliberately a thin adapter.  Registration, version selection, and
activation remain in the services shared with HTTP; this module only validates
CLI-shaped input and dispatches to those services.  Runtime dependencies are
loaded lazily so importing this module never opens PostgreSQL, Milvus, or model
provider connections.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

if TYPE_CHECKING:
    # The concrete types live in the single pipeline implementation used by HTTP.
    from app.ingest.pipeline import IngestCommand, IngestPipeline


EXIT_OK = 0
EXIT_ERROR = 1

_SENSITIVE_FIELD_PARTS = (
    "api_key",
    "authorization",
    "credential",
    "database_url",
    "dsn",
    "password",
    "secret",
    "token",
)


class CliError(RuntimeError):
    """An expected, safe-to-display CLI error."""


@dataclass(frozen=True, slots=True)
class RegistrationRequest:
    """Validated local input passed to the shared PostgreSQL registration stage."""

    pdf_path: Path
    layout_path: Path
    product: str
    version: str
    language: str
    title: str


class PipelineProtocol(Protocol):
    async def run(self, command: object) -> object: ...


class RegistrarProtocol(Protocol):
    async def register(self, request: RegistrationRequest) -> object: ...


class RepositoryProtocol(Protocol):
    async def get_document(self, document_id: UUID, *, for_update: bool = False) -> object: ...

    async def list_documents_for_reconcile(self) -> Sequence[object]: ...

    async def count_chunks(self, document_id: UUID) -> int: ...


class VectorStoreProtocol(Protocol):
    async def count_document(self, document_id: UUID) -> int: ...

    async def reindex_embeddings(self, *, repository: object, provider: object) -> object: ...

    async def delete_document(self, document_id: UUID) -> None: ...


class StorageProtocol(Protocol):
    def cleanup_staging(self, staging_id: UUID) -> None: ...

    def delete_document_assets(self, document_id: UUID) -> None: ...


ProviderFactory = Callable[[str], object | Awaitable[object]]
CommandFactory = Callable[..., object]


@dataclass(slots=True)
class CliRuntime:
    """Dependencies owned by the application composition root.

    ``registrar`` should be the same registration/document service used by the
    HTTP upload route.  It may be omitted when ``pipeline`` itself exposes the
    registration adapter.  The CLI never generates a document UUID.
    """

    pipeline: IngestPipeline | PipelineProtocol
    registrar: RegistrarProtocol | object | None = None
    repository: RepositoryProtocol | object | None = None
    store: VectorStoreProtocol | object | None = None
    provider_factory: ProviderFactory | object | None = None
    storage: StorageProtocol | object | None = None
    command_factory: CommandFactory | None = None


def _uuid_argument(value: str) -> UUID:
    try:
        return UUID(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a valid UUID") from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the standard-library argument parser without touching resources."""

    parser = argparse.ArgumentParser(
        prog="python -m app.ingest",
        description="Ingest and maintain the RAG knowledge index.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="register and ingest one PDF")
    run_parser.add_argument("--pdf", type=Path, required=True, help="source PDF path")
    run_parser.add_argument("--layout", type=Path, required=True, help="layout JSON path")
    run_parser.add_argument("--product", required=True, help="authoritative product name")
    run_parser.add_argument("--version", required=True, help="authoritative product version")
    run_parser.add_argument(
        "--language",
        choices=("zh", "en", "zh-CN"),
        default="zh-CN",
        help="document language",
    )
    run_parser.add_argument("--title", help="display title (defaults to the PDF stem)")

    status_parser = commands.add_parser("status", help="show one document's ingest status")
    status_parser.add_argument("--document-id", type=_uuid_argument, required=True)

    commands.add_parser(
        "reconcile",
        help="compare PostgreSQL chunk counts with Milvus document counts",
    )

    reindex_parser = commands.add_parser(
        "reindex-embeddings",
        help="perform a controlled full embedding-index rebuild",
    )
    reindex_parser.add_argument(
        "--provider",
        choices=("dashscope",),
        required=True,
        help="target embedding provider; mixed-provider indexes are forbidden",
    )

    cleanup_parser = commands.add_parser(
        "cleanup",
        help="remove staging and pre-activation residue for a failed/pending document",
    )
    cleanup_parser.add_argument("--document-id", type=_uuid_argument, required=True)

    return parser


@asynccontextmanager
async def runtime_context() -> AsyncIterator[CliRuntime]:
    """Load the application-owned CLI runtime lazily.

    The composition root must expose ``app.dependencies.create_cli_runtime``.
    It may return a :class:`CliRuntime`, an object with equivalent attributes,
    an awaitable, or an async context manager.  Keeping construction there makes
    the CLI and HTTP paths share the exact pipeline/store/provider instances and
    lifecycle checks.
    """

    try:
        dependencies = import_module("app.dependencies")
        factory = dependencies.create_cli_runtime
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        raise CliError(
            "application CLI runtime is unavailable; "
            "app.dependencies.create_cli_runtime() must be configured"
        ) from exc

    resource = factory()
    if inspect.isawaitable(resource):
        resource = await resource

    if hasattr(resource, "__aenter__") and hasattr(resource, "__aexit__"):
        async with resource as value:
            yield _coerce_runtime(value)
        return

    try:
        yield _coerce_runtime(resource)
    finally:
        await _close_if_supported(resource)


async def run_command(args: argparse.Namespace, runtime: CliRuntime | None = None) -> int:
    """Dispatch one parsed command and return a process exit code."""

    try:
        if runtime is not None:
            return await _dispatch(args, runtime)
        async with runtime_context() as loaded_runtime:
            return await _dispatch(args, loaded_runtime)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 - CLI boundary must return an exit code
        # Exception text can contain a DSN, token, or provider response.  The
        # detailed error belongs in redacting application logs, not stdout/stderr.
        print(f"error: {type(exc).__name__}: command failed", file=sys.stderr)
        return EXIT_ERROR


def main(argv: Sequence[str] | None = None, *, runtime: CliRuntime | None = None) -> int:
    """Parse arguments and run the async command in a fresh event loop."""

    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(run_command(args, runtime=runtime))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


async def _dispatch(args: argparse.Namespace, runtime: CliRuntime) -> int:
    match args.command:
        case "run":
            return await _run_ingest(args, runtime)
        case "status":
            return await _show_status(args, runtime)
        case "reconcile":
            return await _reconcile(runtime)
        case "reindex-embeddings":
            return await _reindex_embeddings(args, runtime)
        case "cleanup":
            return await _cleanup(args, runtime)
        case _:
            raise CliError(f"unknown command: {args.command}")


async def _run_ingest(args: argparse.Namespace, runtime: CliRuntime) -> int:
    pdf_path = _existing_file(args.pdf, "PDF")
    layout_path = _existing_file(args.layout, "layout")
    product = _nonempty(args.product, "product")
    version = _nonempty(args.version, "version")
    language = _nonempty(args.language, "language")
    title = _nonempty(args.title, "title") if args.title else pdf_path.stem

    request = RegistrationRequest(
        pdf_path=pdf_path,
        layout_path=layout_path,
        product=product,
        version=version,
        language=language,
        title=title,
    )
    registrar = runtime.registrar or runtime.pipeline
    registration = await _register_document(registrar, request)
    document_id = _registered_document_id(registration)

    command_factory = runtime.command_factory or _load_ingest_command
    command = command_factory(
        document_id=document_id,
        pdf_path=pdf_path,
        layout_path=layout_path,
        product=product,
        version=version,
        language=language,
        title=title,
    )
    result = await runtime.pipeline.run(command)
    payload = {
        "command": "run",
        "document_id": str(document_id),
        "result": _public_value(result),
    }
    _emit(payload)
    return EXIT_ERROR if _status_value(_field(result, "status")) == "failed" else EXIT_OK


async def _show_status(args: argparse.Namespace, runtime: CliRuntime) -> int:
    document: object | None = None
    if runtime.registrar is not None and callable(getattr(runtime.registrar, "status", None)):
        document = await _maybe_await(runtime.registrar.status(args.document_id))
    else:
        repository = _required(runtime.repository, "repository")
        document = await repository.get_document(args.document_id)

    if document is None:
        raise CliError(f"document not found: {args.document_id}")

    safe_fields = (
        "document_id",
        "status",
        "progress",
        "error",
        "parse_error",
        "current",
        "product",
        "version",
    )
    details = {
        name: _public_value(_field(document, name))
        for name in safe_fields
        if _has_field(document, name)
    }
    details.setdefault("document_id", str(args.document_id))
    _emit({"command": "status", "document": details})
    return EXIT_OK


async def _reconcile(runtime: CliRuntime) -> int:
    repository = _required(runtime.repository, "repository")
    store = _required(runtime.store, "vector store")
    list_method = _first_method(
        repository,
        "list_documents_for_reconcile",
        "reconciliation_rows",
    )
    documents = await _collect(await _maybe_await(list_method()))

    rows: list[dict[str, object]] = []
    mismatch_count = 0
    orphan_count = 0
    for document in sorted(documents, key=lambda value: str(_field(value, "document_id", "id"))):
        document_id = _as_uuid(_field(document, "document_id", "id"), "document_id")
        pg_count_value = _field(document, "pg_chunk_count", "chunk_count")
        if pg_count_value is None:
            count_method = _first_method(repository, "count_chunks", "count_document_chunks")
            pg_count_value = await _maybe_await(count_method(document_id))
        pg_count = int(pg_count_value)
        milvus_count = int(await store.count_document(document_id))
        status = _status_value(_field(document, "status"))
        matches = pg_count == milvus_count
        orphan_vectors = bool(milvus_count and status not in {"parsed", "superseded"})
        mismatch_count += int(not matches)
        orphan_count += int(orphan_vectors)
        rows.append(
            {
                "document_id": str(document_id),
                "status": status,
                "pg_chunk_count": pg_count,
                "milvus_count": milvus_count,
                "matches": matches,
                "orphan_vectors": orphan_vectors,
            }
        )

    report = {
        "command": "reconcile",
        "checked": len(rows),
        "mismatch_count": mismatch_count,
        "orphan_vector_documents": orphan_count,
        "documents": rows,
    }
    _emit(report)
    return EXIT_OK if mismatch_count == 0 and orphan_count == 0 else EXIT_ERROR


async def _reindex_embeddings(args: argparse.Namespace, runtime: CliRuntime) -> int:
    if args.provider != "dashscope":
        raise CliError("only the controlled dashscope reindex target is allowed")

    repository = _required(runtime.repository, "repository")
    store = _required(runtime.store, "vector store")
    provider_factory = _required(runtime.provider_factory, "embedding provider factory")
    provider = await _create_provider(provider_factory, args.provider)
    if not callable(getattr(provider, "embed", None)):
        await _close_if_supported(provider)
        raise CliError("embedding provider does not implement embed()")

    try:
        rebuild = _first_method(store, "reindex_embeddings", "rebuild_embeddings")
        report = await _call_reindex(rebuild, repository=repository, provider=provider)
    finally:
        await _close_if_supported(provider)

    payload = {
        "command": "reindex-embeddings",
        "provider": args.provider,
        "model_id": _public_value(_field(provider, "model_id")),
        "report": _public_value(report),
    }
    _emit(payload)
    return EXIT_ERROR if _report_failed(report) else EXIT_OK


async def _cleanup(args: argparse.Namespace, runtime: CliRuntime) -> int:
    repository = _required(runtime.repository, "repository")
    store = _required(runtime.store, "vector store")
    storage = _required(runtime.storage, "storage manager")
    document = await repository.get_document(args.document_id, for_update=True)
    if document is None:
        raise CliError(f"document not found: {args.document_id}")

    status = _status_value(_field(document, "status"))
    current = bool(_field(document, "current") or False)
    if current or status not in {"failed", "pending"}:
        raise CliError(
            "cleanup is restricted to non-current failed/pending documents; "
            f"document status is {status or 'unknown'}"
        )

    # Retry-safe order for pre-activation residue: vectors first, then files.
    # PostgreSQL history/status is intentionally retained for auditability.
    await store.delete_document(args.document_id)
    cleanup_residue = getattr(storage, "cleanup_document_residue", None)
    if callable(cleanup_residue):
        await _maybe_await(cleanup_residue(args.document_id))
        operations = ["milvus_vectors", "document_residue"]
    else:
        await _maybe_await(storage.cleanup_staging(args.document_id))
        await _maybe_await(storage.delete_document_assets(args.document_id))
        operations = ["milvus_vectors", "staging", "document_assets"]

    _emit(
        {
            "command": "cleanup",
            "document_id": str(args.document_id),
            "status_retained": status,
            "cleaned": operations,
        }
    )
    return EXIT_OK


def _load_ingest_command(**values: object) -> IngestCommand:
    """Import the canonical command only when the ``run`` command executes."""

    try:
        pipeline_module = import_module("app.ingest.pipeline")
        command_type = pipeline_module.IngestCommand
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        raise CliError("app.ingest.pipeline.IngestCommand is unavailable") from exc
    return command_type(**values)


def _coerce_runtime(value: object) -> CliRuntime:
    if isinstance(value, CliRuntime):
        return value
    pipeline = _field(value, "pipeline", "ingest_pipeline")
    if pipeline is None:
        raise CliError("CLI runtime does not provide an ingest pipeline")
    return CliRuntime(
        pipeline=pipeline,
        registrar=(
            _field(value, "registrar")
            or _field(value, "document_service")
            or pipeline
        ),
        repository=_field(value, "repository"),
        store=_field(value, "store", "vector_store"),
        provider_factory=_field(value, "provider_factory"),
        storage=_field(value, "storage"),
        command_factory=_field(value, "command_factory"),
    )


def _registered_document_id(value: object) -> UUID:
    candidate = value
    if not isinstance(value, (str, UUID)):
        candidate = _field(value, "document_id", "job_id")
    return _as_uuid(candidate, "registration document_id")


async def _register_document(registrar: object, request: RegistrationRequest) -> object:
    """Delegate identity/version rules to the shared registration service.

    A dedicated ``register_local``/``register`` adapter is preferred.  The
    ``DocumentService.submit`` fallback only prepares its boundary DTO and file
    digest; all idempotency and candidate creation still remain in that service.
    A CLI composition root using this fallback must construct the service
    without an asynchronous task runner because this command invokes the shared
    pipeline directly.
    """

    for name in ("register_local", "register"):
        method = getattr(registrar, name, None)
        if callable(method):
            return await _maybe_await(method(request))

    submit = getattr(registrar, "submit", None)
    if not callable(submit):
        raise CliError(
            f"{type(registrar).__name__} does not implement "
            "register_local(), register(), or submit()"
        )
    submission = await asyncio.to_thread(_document_submit_command, request)
    return await _maybe_await(submit(submission))


def _document_submit_command(request: RegistrationRequest) -> object:
    try:
        service_module = import_module("app.document_service")
        command_type = service_module.DocumentSubmitCommand
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        raise CliError("app.document_service.DocumentSubmitCommand is unavailable") from exc

    digest, size_bytes = _fingerprint_pdf(request.pdf_path)
    return command_type(
        path=request.pdf_path,
        original_filename=request.pdf_path.name,
        sha256=digest,
        size_bytes=size_bytes,
        product=request.product,
        version=request.version,
        language=request.language,
        title=request.title,
    )


def _fingerprint_pdf(path: Path) -> tuple[str, int]:
    """Stream a stable PDF digest without loading the file into memory."""

    before = path.stat()
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as stream:
        prefix = stream.read(5)
        if prefix != b"%PDF-":
            raise CliError("PDF file does not start with %PDF- magic")
        digest.update(prefix)
        size_bytes += len(prefix)
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size_bytes += len(chunk)
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or size_bytes != after.st_size
    ):
        raise CliError("PDF changed while it was being registered")
    return digest.hexdigest(), size_bytes


def _as_uuid(value: object, label: str) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise CliError(f"{label} is not a valid UUID") from exc


def _existing_file(value: Path | str, label: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise CliError(f"{label} file does not exist: {path}")
    return path


def _nonempty(value: object, label: str) -> str:
    normalized = str(value).strip() if value is not None else ""
    if not normalized:
        raise CliError(f"{label} must not be empty")
    return normalized


def _required(value: Any, label: str) -> Any:
    if value is None:
        raise CliError(f"CLI runtime does not provide {label}")
    return value


def _first_method(value: object, *names: str) -> Callable[..., object]:
    for name in names:
        method = getattr(value, name, None)
        if callable(method):
            return method
    joined = "() or ".join(names) + "()"
    raise CliError(f"{type(value).__name__} does not implement {joined}")


async def _maybe_await(value: object) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _collect(value: object) -> list[object]:
    if hasattr(value, "__aiter__"):
        return [item async for item in value]  # type: ignore[union-attr]
    if value is None:
        return []
    return list(value)  # type: ignore[arg-type]


async def _create_provider(factory: object, provider_name: str) -> object:
    create = getattr(factory, "create", None)
    value = (
        create(provider_name)
        if callable(create)
        else factory(provider_name)  # type: ignore[operator]
    )
    return await _maybe_await(value)


async def _call_reindex(
    method: Callable[..., object], *, repository: object, provider: object
) -> object:
    """Call the P1 store extension while tolerating a provider-only adapter."""

    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    names = {parameter.name for parameter in parameters}
    accepts_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    kwargs: dict[str, object] = {"provider": provider}
    if "repository" in names or accepts_keywords:
        kwargs["repository"] = repository
    return await _maybe_await(method(**kwargs))


async def _close_if_supported(value: object) -> None:
    close = getattr(value, "aclose", None)
    if callable(close):
        await _maybe_await(close())


def _field(value: object, *names: str) -> Any:
    if value is None:
        return None
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _has_field(value: object, name: str) -> bool:
    return (isinstance(value, Mapping) and name in value) or hasattr(value, name)


def _status_value(value: object) -> str | None:
    if value is None:
        return None
    raw = value.value if isinstance(value, Enum) else value
    return str(raw).lower()


def _report_failed(report: object) -> bool:
    if _field(report, "success") is False:
        return True
    for name in ("failed", "failed_count", "error_count", "mismatch_count"):
        value = _field(report, name)
        if isinstance(value, bool) and value:
            return True
        if isinstance(value, int | float) and value > 0:
            return True
    return False


def _is_sensitive_field(name: object) -> bool:
    normalized = str(name).lower().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_FIELD_PARTS)


def _public_value(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, UUID | Path):
        return str(value)
    if isinstance(value, Enum):
        return _public_value(value.value)
    if isinstance(value, Mapping):
        return {
            str(key): "[redacted]" if _is_sensitive_field(key) else _public_value(item)
            for key, item in value.items()
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: (
                "[redacted]"
                if _is_sensitive_field(item.name)
                else _public_value(getattr(value, item.name))
            )
            for item in fields(value)
        }
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _public_value(model_dump(mode="json"))
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_public_value(item) for item in value]
    if hasattr(value, "__dict__"):
        return _public_value(
            {
                key: item
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        )
    return str(value)


def _emit(payload: Mapping[str, object]) -> None:
    print(json.dumps(_public_value(payload), ensure_ascii=False, sort_keys=True))


__all__ = [
    "CliRuntime",
    "RegistrationRequest",
    "build_parser",
    "main",
    "run_command",
]
