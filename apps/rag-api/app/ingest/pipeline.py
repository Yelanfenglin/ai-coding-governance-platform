"""Asynchronous parse -> chunk -> embed -> stage -> activate orchestration."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

from app.chunking.chunker import Chunk, build_chunks
from app.chunking.loader import ParsedDocument, load_parsed_document
from app.chunking.units import SemanticUnit, build_semantic_units
from app.ingest.adapter import (
    AuthoritativeDocumentIdentity,
    ChunkMappingError,
    PersistenceChunk,
    map_jsonl,
)
from app.scalar_metadata import ScalarMetadata, scalar_metadata
from app.storage import StorageError, StorageManager
from app.vectordb.milvus_store import KnowledgeChunkKind as VectorChunkKind
from app.vectordb.milvus_store import VectorRecord

EMBEDDING_DIMENSION = 1024
DEFAULT_EMBEDDING_BATCH_SIZE = 16
DEFAULT_UPSERT_BATCH_SIZE = 128

try:
    from app.db.models import DocumentStatus
except (ImportError, ModuleNotFoundError):

    class DocumentStatus(StrEnum):
        """Dependency-light fallback with the canonical persisted values."""

        PENDING = "pending"
        PARSING = "parsing"
        PARSED = "parsed"
        FAILED = "failed"
        SUPERSEDED = "superseded"


class IngestError(RuntimeError):
    """Base class for deterministic ingestion failures."""


class IngestInvariantError(IngestError):
    """A cross-stage count, shape, or artifact invariant failed."""


class IngestStateError(IngestError):
    """The PostgreSQL candidate is not in a runnable state."""


class PipelineConfigurationError(IngestError):
    """Injected adapters do not provide the required transaction contract."""


@dataclass(frozen=True, slots=True)
class IngestCommand:
    document_id: UUID
    pdf_path: Path
    layout_path: Path | None
    product: str
    version: str
    language: str
    title: str


@dataclass(frozen=True, slots=True)
class IngestResult:
    document_id: UUID
    chunk_count: int
    vector_count: int
    status: DocumentStatus


@dataclass(frozen=True, slots=True)
class ParsedArtifact:
    document_id: UUID
    directory: Path
    summary: Mapping[str, Any]


class EmbeddingProvider(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    async def upsert(self, records: Sequence[VectorRecord]) -> int: ...

    async def count_document(self, document_id: UUID) -> int: ...

    async def delete_document(self, document_id: UUID) -> None: ...


class IngestRepository(Protocol):
    async def get_document(self, document_id: UUID, *, for_update: bool = False) -> Any: ...

    async def set_status(
        self, document_id: UUID, status: DocumentStatus, *, error: str | None = None
    ) -> None: ...

    async def replace_chunks(
        self, document_id: UUID, records: Sequence[PersistenceChunk]
    ) -> None: ...

    async def activate_candidate(
        self, document_id: UUID, *, expected_vector_count: int
    ) -> Sequence[UUID]: ...


ParseDocumentFn = Callable[..., Mapping[str, Any]]
LoadDocumentFn = Callable[[Path], ParsedDocument]
BuildUnitsFn = Callable[[ParsedDocument], list[SemanticUnit]]
BuildChunksFn = Callable[[ParsedDocument, list[SemanticUnit]], list[Chunk]]


def parse_document(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    """Lazy bridge to the parser's sole public entry.

    Keeping the import here lets adapter-only and fake-pipeline tests run in a
    minimal offline environment without importing PyMuPDF.
    """

    from app.parser.parse import parse_document as public_parse_document

    return public_parse_document(*args, **kwargs)


class IngestPipeline:
    """The only component allowed to coordinate all ingest persistence stages."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        store: VectorStore,
        repo: IngestRepository,
        storage: StorageManager,
        *,
        embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        upsert_batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
        embedding_dimension: int = EMBEDDING_DIMENSION,
        parse_document_fn: ParseDocumentFn | None = None,
        load_document_fn: LoadDocumentFn | None = None,
        build_units_fn: BuildUnitsFn | None = None,
        build_chunks_fn: BuildChunksFn | None = None,
        worker: Executor | None = None,
    ) -> None:
        if embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be positive")
        if upsert_batch_size <= 0:
            raise ValueError("upsert_batch_size must be positive")
        if embedding_dimension != EMBEDDING_DIMENSION:
            raise ValueError(f"ingest embedding dimension must be {EMBEDDING_DIMENSION}")

        self._provider = provider
        self._store = store
        self._repo = repo
        self._storage = storage
        self._embedding_batch_size = embedding_batch_size
        self._upsert_batch_size = upsert_batch_size
        self._embedding_dimension = embedding_dimension
        self._parse_document = parse_document_fn or parse_document
        self._load_document = load_document_fn or load_parsed_document
        self._build_units = build_units_fn or build_semantic_units
        self._build_chunks = build_chunks_fn or build_chunks
        self._worker = worker or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="rag-ingest-local"
        )
        self._owns_worker = worker is None
        self._closed = False

    async def run(self, command: IngestCommand) -> IngestResult:
        """Execute a replay-safe ingestion attempt for one registered candidate."""

        self._ensure_open()
        command = self._validated_command(command)
        stage = "claim"
        attempt_claimed = False
        try:
            async with self._repository_transaction():
                document = await self._get_document(command.document_id, for_update=True)
                status = _document_status(document)
                if status in {DocumentStatus.PARSED, DocumentStatus.SUPERSEDED}:
                    identity = None
                elif status == DocumentStatus.FAILED:
                    raise IngestStateError(
                        "failed candidates require an explicit repository retry/reset "
                        "before ingestion"
                    )
                elif status not in {DocumentStatus.PENDING, DocumentStatus.PARSING}:
                    raise IngestStateError(
                        f"document is not ingestible from status {status.value!r}"
                    )
                else:
                    identity = _authoritative_identity(document, command)
                    await self._set_status(command.document_id, DocumentStatus.PARSING)
                    attempt_claimed = True

            if identity is None:
                return await self._existing_result(command.document_id, status)

            # A pending/parsing document is never query-visible. Clearing its
            # deterministic ID scope makes retry after a partial stage exact.
            stage = "clear candidate vectors"
            await self._store.delete_document(command.document_id)

            stage = "parse"
            artifact = await self.parse(command)
            stage = "chunk"
            chunk_jsonl = await self.chunk(artifact, identity)
            stage = "adapter"
            records = await self.map_records(identity, chunk_jsonl, artifact.directory)
            self._validate_record_pages(records, artifact.summary)
            stage = "embedding"
            vectors = await self.embed(records)
            stage = "Milvus stage / PostgreSQL activation"
            result = await self.stage_and_activate(
                command.document_id,
                records,
                vectors,
            )
        except (Exception, asyncio.CancelledError) as exc:
            if attempt_claimed:
                await self._compensate(command, stage, exc)
            raise

        # Cleanup is deliberately outside the activation failure boundary: a
        # committed PG current switch must not be rolled back because staging
        # retention cleanup encountered a local filesystem problem.
        self._cleanup_staging_for_path(command.pdf_path)
        return result

    async def parse(self, command: IngestCommand) -> ParsedArtifact:
        """Run the public parser entry on the single-worker executor."""

        if command.layout_path is None:
            raise IngestInvariantError("P0 ingestion requires an explicit layout_path")
        output_dir = self._storage.parsed_dir(command.document_id)
        summary = await self._run_local(
            self._parse_document,
            command.pdf_path,
            command.layout_path,
            output_dir,
        )
        if not isinstance(summary, Mapping):
            raise IngestInvariantError("parse_document must return a summary mapping")
        self._validate_parse_artifact(output_dir, summary)
        return ParsedArtifact(command.document_id, output_dir, dict(summary))

    async def chunk(
        self,
        artifact: ParsedArtifact,
        identity: AuthoritativeDocumentIdentity,
    ) -> Path:
        """Build chunks with PG identity and atomically publish the JSONL audit."""

        return await self._run_local(self._chunk_sync, artifact, identity)

    async def map_records(
        self,
        identity: AuthoritativeDocumentIdentity,
        chunk_jsonl: Path,
        image_base_dir: Path,
    ) -> list[PersistenceChunk]:
        return await self._run_local(
            map_jsonl,
            chunk_jsonl,
            identity,
            storage=self._storage,
            image_base_dir=image_base_dir,
        )

    async def embed(self, records: Sequence[PersistenceChunk]) -> list[VectorRecord]:
        """Embed fixed-size batches and reject all partial or malformed responses."""

        provider_dimension = getattr(self._provider, "dimension", self._embedding_dimension)
        if provider_dimension != self._embedding_dimension:
            raise IngestInvariantError(
                f"embedding provider dimension is {provider_dimension}; "
                f"expected {self._embedding_dimension}"
            )

        all_vectors: list[tuple[float, ...]] = []
        for batch in _batches(records, self._embedding_batch_size):
            raw_vectors = await _await_if_needed(
                self._provider.embed([record.embedding_text for record in batch])
            )
            if not isinstance(raw_vectors, Sequence) or isinstance(
                raw_vectors, (str, bytes, bytearray)
            ):
                raise IngestInvariantError("embedding provider returned a non-sequence batch")
            if len(raw_vectors) != len(batch):
                raise IngestInvariantError(
                    f"embedding provider returned {len(raw_vectors)} vectors for "
                    f"{len(batch)} texts"
                )
            for vector in raw_vectors:
                all_vectors.append(self._validated_vector(vector))

        if len(all_vectors) != len(records):
            raise IngestInvariantError("embedding output count does not match chunk count")

        return [
            VectorRecord(
                id=record.embedding_id,
                vector=vector,
                document_id=record.document_id,
                product=record.product,
                version=record.version,
                kind=VectorChunkKind(record.kind.value),
                semantic_type=record.semantic_type,
                chunk_index=record.chunk_index,
                page_start=record.page_start,
                page_end=record.page_end,
                section_path=record.section_path,
                text=record.text,
                embedding_model_id=self._embedding_model_id(),
                section_id=scalar.section_id,
                register=scalar.register,
                packages=scalar.packages,
            )
            for record, vector, scalar in zip(
                records, all_vectors, self._scalar_metadata(records), strict=True
            )
        ]

    async def stage_and_activate(
        self,
        document_id: UUID,
        chunks: Sequence[PersistenceChunk],
        vectors: Sequence[VectorRecord],
    ) -> IngestResult:
        """Stage invisible vectors, enforce the gate, then atomically switch PG."""

        if not chunks or len(chunks) != len(vectors):
            raise IngestInvariantError("chunk/vector candidates must be non-empty and one-to-one")
        for chunk, vector in zip(chunks, vectors, strict=True):
            if chunk.document_id != document_id or vector.document_id != document_id:
                raise IngestInvariantError("candidate record belongs to another document")
            if chunk.embedding_id != vector.id:
                raise IngestInvariantError("chunk/vector embedding IDs do not align")

        embedding_model_id = self._embedding_model_id()
        store_model_id = getattr(self._store, "embedding_model_id", embedding_model_id)
        if store_model_id != embedding_model_id:
            raise IngestInvariantError(
                "embedding provider fingerprint does not match the Milvus index"
            )

        for batch in _batches(vectors, self._upsert_batch_size):
            await _await_if_needed(self._store.upsert(batch))

        staged_count = await self._count_vectors(document_id)
        if staged_count != len(chunks):
            raise IngestInvariantError(
                f"Milvus count gate failed: expected {len(chunks)}, found {staged_count}"
            )

        async with self._repository_transaction():
            # RagRepository.activate_candidate owns the advisory lock per its
            # contract. A separate hook supports implementations that expose it.
            await self._lock_product_if_exposed(chunks[0].product)
            await _await_if_needed(self._repo.replace_chunks(document_id, chunks))

            verified_count = await self._count_vectors(document_id)
            if verified_count != len(chunks):
                raise IngestInvariantError(
                    "Milvus vector count changed before PostgreSQL activation"
                )
            await _await_if_needed(
                self._repo.activate_candidate(
                    document_id,
                    expected_vector_count=verified_count,
                )
            )

        return IngestResult(
            document_id=document_id,
            chunk_count=len(chunks),
            vector_count=staged_count,
            status=DocumentStatus.PARSED,
        )

    async def aclose(self) -> None:
        """Release only the locally owned worker; injected adapters own themselves."""

        if self._closed:
            return
        self._closed = True
        if self._owns_worker:
            worker = self._worker
            await asyncio.to_thread(worker.shutdown, wait=True, cancel_futures=True)

    def _chunk_sync(
        self,
        artifact: ParsedArtifact,
        identity: AuthoritativeDocumentIdentity,
    ) -> Path:
        loaded = self._load_document(artifact.directory)
        authoritative_document = ParsedDocument(
            doc_id=str(identity.document_id),
            product=identity.product,
            version=identity.version,
            pdf_name=loaded.pdf_name,
            pages=loaded.pages,
        )
        units = self._build_units(authoritative_document)
        chunks = self._build_chunks(authoritative_document, units)
        destination = self._storage.chunk_jsonl(identity.document_id)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                for chunk in chunks:
                    value = chunk.to_dict()
                    stream.write(
                        json.dumps(
                            value,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                    )
                    stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return destination.resolve(strict=True)

    def _validate_parse_artifact(
        self, output_dir: Path, returned_summary: Mapping[str, Any]
    ) -> None:
        summary_path = output_dir / "summary.json"
        pages_path = output_dir / "pages.json"
        if not summary_path.is_file() or not pages_path.is_file():
            raise IngestInvariantError("parser did not publish summary.json and pages.json")
        try:
            disk_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            pages = json.loads(pages_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise IngestInvariantError("parser artifact JSON is unreadable") from exc
        if not isinstance(disk_summary, Mapping) or not isinstance(pages, list):
            raise IngestInvariantError("parser artifact JSON has an invalid top-level shape")

        self._validate_summary_invariants(returned_summary)
        self._validate_summary_invariants(disk_summary)
        page_count = disk_summary.get("pages")
        if isinstance(page_count, bool) or not isinstance(page_count, int) or page_count <= 0:
            raise IngestInvariantError("parser summary pages must be a positive integer")
        if len(pages) != page_count:
            raise IngestInvariantError(
                f"parser page count mismatch: summary={page_count}, artifact={len(pages)}"
            )

    @staticmethod
    def _validate_summary_invariants(summary: Mapping[str, Any]) -> None:
        quality = summary.get("quality")
        if not isinstance(quality, Mapping):
            raise IngestInvariantError("parser summary is missing quality invariants")
        if quality.get("orphan_text_lines") != 0:
            raise IngestInvariantError("parser produced orphan text lines")
        if quality.get("duplicate_owned_lines") != 0:
            raise IngestInvariantError("parser produced duplicate-owned text lines")
        missing = quality.get("missing_snapshot_files")
        if not isinstance(missing, Sequence) or isinstance(missing, (str, bytes, bytearray)):
            raise IngestInvariantError("missing_snapshot_files must be an array")
        if missing:
            raise IngestInvariantError("parser summary reports missing snapshot files")

    @staticmethod
    def _validate_record_pages(
        records: Sequence[PersistenceChunk], summary: Mapping[str, Any]
    ) -> None:
        total_pages = summary.get("pages")
        if isinstance(total_pages, bool) or not isinstance(total_pages, int):
            raise IngestInvariantError("parser summary pages must be an integer")
        if any(record.page_end > total_pages for record in records):
            raise IngestInvariantError("chunk page range exceeds the parsed document")

    def _validated_vector(self, value: Any) -> tuple[float, ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise IngestInvariantError("embedding vector must be a numeric sequence")
        if len(value) != self._embedding_dimension:
            raise IngestInvariantError(
                f"embedding vector has {len(value)} dimensions; "
                f"expected {self._embedding_dimension}"
            )
        try:
            vector = tuple(float(component) for component in value)
        except (TypeError, ValueError) as exc:
            raise IngestInvariantError("embedding vector contains a non-numeric value") from exc
        if not all(math.isfinite(component) for component in vector):
            raise IngestInvariantError("embedding vector contains a non-finite value")
        return vector

    async def _existing_result(
        self, document_id: UUID, status: DocumentStatus
    ) -> IngestResult:
        vector_count = await self._count_vectors(document_id)
        chunk_count = vector_count
        count_chunks = getattr(self._repo, "count_chunks", None)
        if callable(count_chunks):
            chunk_count = int(await _await_if_needed(count_chunks(document_id)))
            if chunk_count != vector_count:
                raise IngestInvariantError(
                    "existing document violates the PG chunk/Milvus vector count invariant"
                )
        return IngestResult(document_id, chunk_count, vector_count, status)

    async def _get_document(self, document_id: UUID, *, for_update: bool = False) -> Any:
        getter = getattr(self._repo, "get_document", None)
        if not callable(getter):
            raise PipelineConfigurationError("repository must expose get_document()")
        document = await _await_if_needed(getter(document_id, for_update=for_update))
        if document is None:
            raise IngestStateError("registered document candidate was not found")
        registered_id = _field(document, "id", document_id)
        try:
            matches = UUID(str(registered_id)) == document_id
        except (TypeError, ValueError, AttributeError) as exc:
            raise IngestStateError("repository returned an invalid document ID") from exc
        if not matches:
            raise IngestStateError("repository returned a different document candidate")
        return document

    async def _set_status(
        self,
        document_id: UUID,
        status: DocumentStatus,
        *,
        error: str | None = None,
    ) -> None:
        setter = getattr(self._repo, "set_status", None)
        if not callable(setter):
            raise PipelineConfigurationError("repository must expose set_status()")
        await _await_if_needed(setter(document_id, status, error=error))

    async def _count_vectors(self, document_id: UUID) -> int:
        value = await _await_if_needed(self._store.count_document(document_id))
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise IngestInvariantError("Milvus count_document returned an invalid count")
        return value

    @staticmethod
    def _scalar_metadata(records: Sequence[PersistenceChunk]) -> list[ScalarMetadata]:
        """Derive the v2 scalar fields with the backfill's contract.

        One implementation only: ``app.scalar_metadata`` owns the rules for
        section_id, register and the package set, so live ingestion can never
        drift from the shadow backfill.
        """

        return [
            scalar_metadata(
                chunk_metadata=record.chunk_metadata,
                text=record.text,
                extra_text=record.embedding_text,
                section_path=record.section_path,
            )
            for record in records
        ]

    def _embedding_model_id(self) -> str:
        fingerprint = getattr(self._provider, "fingerprint", None)
        if callable(fingerprint):
            fingerprint = fingerprint()
        if isinstance(fingerprint, str) and fingerprint.strip():
            return fingerprint.strip()
        model_id = getattr(self._provider, "model_id", None)
        if not isinstance(model_id, str) or not model_id.strip():
            raise PipelineConfigurationError(
                "embedding provider must expose a non-empty model_id or fingerprint"
            )
        return f"{model_id.strip()}:{self._embedding_dimension}"

    async def _lock_product_if_exposed(self, product: str) -> None:
        for name in ("lock_product", "advisory_lock_product", "acquire_product_lock"):
            method = getattr(self._repo, name, None)
            if callable(method):
                await _await_if_needed(method(product))
                return

    @asynccontextmanager
    async def _repository_transaction(self) -> AsyncIterator[None]:
        context: Any = None
        transaction = getattr(self._repo, "transaction", None)
        if callable(transaction):
            context = transaction()
        elif transaction is not None:
            context = transaction
        else:
            session = getattr(self._repo, "session", None)
            begin = getattr(session, "begin", None) if session is not None else None
            if not callable(begin):
                begin = getattr(self._repo, "begin", None)
            if callable(begin):
                context = begin()
        if inspect.isawaitable(context):
            context = await context
        if context is None or not hasattr(context, "__aenter__"):
            raise PipelineConfigurationError(
                "repository must expose an async transaction(), begin(), or session.begin()"
            )
        async with context:
            yield

    async def _compensate(
        self,
        command: IngestCommand,
        stage: str,
        error: BaseException,
    ) -> None:
        # Each compensation is independent; one unavailable backend must not
        # prevent the other durable/local cleanup attempts.
        with suppress(Exception):
            await self._store.delete_document(command.document_id)
        with suppress(Exception):
            await self._run_local(self._storage.delete_document_assets, command.document_id)
        self._cleanup_staging_for_path(command.pdf_path)
        with suppress(Exception):
            async with self._repository_transaction():
                await self._set_status(
                    command.document_id,
                    DocumentStatus.FAILED,
                    error=_safe_error_summary(stage, error),
                )

    def _cleanup_staging_for_path(self, path: Path) -> None:
        try:
            resolved = Path(path).resolve(strict=False)
            staging_root = (self._storage.root / "staging").resolve(strict=False)
            relative = resolved.relative_to(staging_root)
            if len(relative.parts) == 1 and resolved.suffix.casefold() == ".pdf":
                staging_id = UUID(resolved.stem)
            elif len(relative.parts) >= 2:
                # Backward-compatible cleanup for an older UUID-directory layout.
                staging_id = UUID(relative.parts[0])
            else:
                return
            self._storage.cleanup_staging(staging_id)
        except (OSError, RuntimeError, ValueError, StorageError):
            pass

    async def _run_local(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()

        def invoke() -> Any:
            return function(*args, **kwargs)

        future = loop.run_in_executor(self._worker, invoke)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # Executor work cannot be killed safely. Wait for the local writer
            # to quiesce so the caller's compensation cannot race late files.
            with suppress(Exception):
                await future
            raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("IngestPipeline is closed")

    @staticmethod
    def _validated_command(command: IngestCommand) -> IngestCommand:
        if not isinstance(command, IngestCommand):
            raise TypeError("command must be an IngestCommand")
        if not isinstance(command.document_id, UUID):
            raise ValueError("document_id must be a UUID")
        for field_name, limit in (
            ("product", 64),
            ("version", 64),
            ("language", 32),
            ("title", 1024),
        ):
            value = getattr(command, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            if len(value.encode("utf-8")) > limit:
                raise ValueError(f"{field_name} exceeds {limit} UTF-8 bytes")

        pdf_path = Path(command.pdf_path).resolve(strict=True)
        if not pdf_path.is_file():
            raise ValueError("pdf_path must be a regular file")
        with pdf_path.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise ValueError("pdf_path does not start with %PDF- magic")

        layout_path: Path | None = None
        if command.layout_path is not None:
            layout_path = Path(command.layout_path).resolve(strict=True)
            if not layout_path.is_file():
                raise ValueError("layout_path must be a regular file")
        return IngestCommand(
            document_id=command.document_id,
            pdf_path=pdf_path,
            layout_path=layout_path,
            product=command.product.strip(),
            version=command.version.strip(),
            language=command.language.strip(),
            title=command.title.strip(),
        )


def _document_status(document: Any) -> DocumentStatus:
    value = _field(document, "status", None)
    if isinstance(value, Enum):
        value = value.value
    try:
        return DocumentStatus(str(value).lower())
    except ValueError as exc:
        raise IngestStateError(f"repository returned unknown document status: {value!r}") from exc


def _authoritative_identity(
    document: Any, command: IngestCommand
) -> AuthoritativeDocumentIdentity:
    product = _field(document, "product", command.product)
    version = _field(document, "version", command.version)
    return AuthoritativeDocumentIdentity(command.document_id, product, version)


def _field(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _batches(values: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


async def _await_if_needed(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _safe_error_summary(stage: str, error: BaseException) -> str:
    safe_types = (IngestError, ChunkMappingError, StorageError)
    detail = str(error) if isinstance(error, safe_types) else type(error).__name__
    detail = " ".join(detail.split())[:700]
    return f"ingest failed during {stage}: {detail or type(error).__name__}"


__all__ = [
    "DocumentStatus",
    "IngestCommand",
    "IngestError",
    "IngestInvariantError",
    "IngestPipeline",
    "IngestResult",
    "IngestStateError",
    "ParsedArtifact",
    "PipelineConfigurationError",
    "VectorRecord",
]
