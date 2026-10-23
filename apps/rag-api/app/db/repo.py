"""Transaction-neutral SQLAlchemy repository for RAG persistence."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document, DocumentStatus, KnowledgeChunkKind


class DocumentNotFoundError(LookupError):
    pass


class CandidateActivationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DocumentCreateData:
    product: str
    version: str
    language: str
    title: str
    filename: str
    file_sha256: str
    product_family: str | None = None
    source_url: str | None = None
    stored_by: str | None = None
    ingest_stage: str = "pending"
    progress: int = 0


@dataclass(frozen=True, slots=True)
class DocumentFilters:
    product: str | None = None
    version: str | None = None
    current: bool | None = None
    status: DocumentStatus | None = None
    limit: int = 100
    offset: int = 0

    def __post_init__(self) -> None:
        if self.limit <= 0 or self.limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        if self.offset < 0:
            raise ValueError("offset must not be negative")


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    chunk_index: int
    kind: KnowledgeChunkKind
    section_path: str
    section_title: str
    page_start: int
    page_end: int
    text: str
    embedding_id: str
    logical_chunk_id: str
    semantic_type: str
    embedding_text: str
    table_repr: Mapping[str, Any] | None = None
    image_ref: str | None = None
    chunk_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchScope:
    product: str | None = None
    version: str | None = None


class RagRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def find_identical(
        self, product: str, version: str, file_sha256: str
    ) -> Document | None:
        statement = select(Document).where(
            Document.product == product,
            Document.version == version,
            Document.file_sha256 == file_sha256,
        )
        return (await self.session.execute(statement)).scalar_one_or_none()

    async def create_candidate(self, data: DocumentCreateData) -> Document:
        document = Document(
            id=uuid4(),
            product=data.product,
            product_family=getattr(data, "product_family", None),
            version=data.version,
            language=data.language,
            title=data.title,
            filename=data.filename,
            source_url=getattr(data, "source_url", None),
            file_sha256=data.file_sha256,
            status=DocumentStatus.PENDING,
            current=False,
            stored_by=getattr(data, "stored_by", None),
            ingest_stage=getattr(data, "ingest_stage", "pending"),
            progress=getattr(data, "progress", 0),
        )
        self.session.add(document)
        await self.session.flush()
        return document

    async def get_document(self, document_id: UUID, *, for_update: bool = False) -> Document:
        statement = select(Document).where(Document.id == document_id)
        if for_update:
            statement = statement.with_for_update()
        document = (await self.session.execute(statement)).scalar_one_or_none()
        if document is None:
            raise DocumentNotFoundError(f"document {document_id} was not found")
        return document

    async def list_documents(self, filters: DocumentFilters) -> Sequence[Document]:
        statement = select(Document)
        if filters.product is not None:
            statement = statement.where(Document.product == filters.product)
        if filters.version is not None:
            statement = statement.where(Document.version == filters.version)
        if filters.current is not None:
            statement = statement.where(Document.current.is_(filters.current))
        status = getattr(filters, "status", None)
        if status is not None:
            statement = statement.where(Document.status == status)
        statement = statement.order_by(Document.created_at.desc(), Document.id.desc())
        statement = statement.offset(getattr(filters, "offset", 0)).limit(
            getattr(filters, "limit", 100)
        )
        return tuple((await self.session.execute(statement)).scalars().all())

    async def set_status(
        self, document_id: UUID, status: DocumentStatus, *, error: str | None = None
    ) -> None:
        result = await self.session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(status=DocumentStatus(status), parse_error=error, updated_at=func.now())
            .returning(Document.id)
        )
        if result.scalar_one_or_none() is None:
            raise DocumentNotFoundError(f"document {document_id} was not found")

    async def replace_chunks(
        self, document_id: UUID, records: Sequence[ChunkRecord]
    ) -> None:
        indexes = [record.chunk_index for record in records]
        embedding_ids = [record.embedding_id for record in records]
        if len(indexes) != len(set(indexes)):
            raise ValueError("chunk_index values must be unique within a document")
        if len(embedding_ids) != len(set(embedding_ids)):
            raise ValueError("embedding_id values must be unique")

        await self.get_document(document_id, for_update=True)
        await self.session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        self.session.add_all(
            [
                Chunk(
                    id=getattr(record, "id", None) or uuid4(),
                    document_id=document_id,
                    chunk_index=record.chunk_index,
                    kind=KnowledgeChunkKind(record.kind),
                    section_path=record.section_path,
                    section_title=record.section_title,
                    page_start=record.page_start,
                    page_end=record.page_end,
                    text=record.text,
                    table_repr=(
                        dict(record.table_repr) if record.table_repr is not None else None
                    ),
                    image_ref=record.image_ref,
                    embedding_id=record.embedding_id,
                    logical_chunk_id=record.logical_chunk_id,
                    semantic_type=record.semantic_type,
                    embedding_text=record.embedding_text,
                    chunk_metadata=dict(record.chunk_metadata),
                )
                for record in records
            ]
        )
        await self.session.flush()

    async def activate_candidate(
        self, document_id: UUID, *, expected_vector_count: int
    ) -> Sequence[UUID]:
        if not self.session.in_transaction():
            raise CandidateActivationError("activate_candidate requires a caller-owned transaction")
        if expected_vector_count < 0:
            raise CandidateActivationError("expected_vector_count must not be negative")

        candidate = await self.get_document(document_id, for_update=True)
        advisory_key = self._product_advisory_key(candidate.product)
        await self.session.execute(select(func.pg_advisory_xact_lock(advisory_key)))

        await self.session.refresh(candidate, attribute_names=["status", "current"])
        if candidate.status != DocumentStatus.PARSING or candidate.current:
            raise CandidateActivationError(
                "candidate must remain non-current and in parsing status before activation"
            )

        chunk_count = await self.session.scalar(
            select(func.count(Chunk.id)).where(Chunk.document_id == document_id)
        )
        if chunk_count != expected_vector_count:
            raise CandidateActivationError(
                "candidate chunk count does not match the verified vector count"
            )

        superseded_result = await self.session.execute(
            update(Document)
            .where(
                Document.product == candidate.product,
                Document.current.is_(True),
                Document.id != candidate.id,
            )
            .values(
                status=DocumentStatus.SUPERSEDED,
                current=False,
                superseded_at=func.now(),
                updated_at=func.now(),
            )
            .returning(Document.id),
            execution_options={"synchronize_session": False},
        )
        superseded_ids = tuple(superseded_result.scalars().all())

        candidate.status = DocumentStatus.PARSED
        candidate.current = True
        candidate.superseded_at = None
        candidate.parse_error = None
        candidate.ingest_stage = "completed"
        candidate.progress = 100
        await self.session.flush()
        return superseded_ids

    async def resolve_search_documents(self, scope: SearchScope) -> Sequence[Document]:
        if scope.version is None:
            statement = select(Document).where(
                Document.current.is_(True),
                Document.status == DocumentStatus.PARSED,
            )
            if scope.product is not None:
                statement = statement.where(Document.product == scope.product)
            statement = statement.order_by(Document.product, Document.id)
        else:
            successful = select(
                Document.id.label("document_id"),
                func.row_number()
                .over(
                    partition_by=(Document.product, Document.version),
                    order_by=(Document.created_at.desc(), Document.id.desc()),
                )
                .label("revision_rank"),
            ).where(
                Document.version == scope.version,
                Document.status.in_(
                    (DocumentStatus.PARSED, DocumentStatus.SUPERSEDED)
                ),
            )
            if scope.product is not None:
                successful = successful.where(Document.product == scope.product)
            ranked = successful.subquery()
            statement = (
                select(Document)
                .join(ranked, ranked.c.document_id == Document.id)
                .where(ranked.c.revision_rank == 1)
                .order_by(Document.product, Document.id)
            )
        return tuple((await self.session.execute(statement)).scalars().all())

    async def get_chunks_by_embedding_ids(
        self, embedding_ids: Sequence[str]
    ) -> Mapping[str, Chunk]:
        if not embedding_ids:
            return {}
        unique_ids = tuple(dict.fromkeys(embedding_ids))
        statement = select(Chunk).where(Chunk.embedding_id.in_(unique_ids))
        chunks = (await self.session.execute(statement)).scalars().all()
        return {chunk.embedding_id: chunk for chunk in chunks}

    async def list_chunks_for_documents(
        self, document_ids: Sequence[UUID]
    ) -> Sequence[Chunk]:
        """Return every persisted chunk for the given current documents.

        Used by the optional BM25 route to build a scope-local index. PostgreSQL
        remains the evidence source; Milvus may briefly lead it during
        compensation/reconcile, so it is not used for lexical text.
        """

        if not document_ids:
            return ()
        unique_ids = tuple(dict.fromkeys(document_ids))
        statement = select(Chunk).where(Chunk.document_id.in_(unique_ids))
        return tuple((await self.session.execute(statement)).scalars().all())

    async def get_chunk_neighbors(
        self, chunk_id: UUID, radius: int = 1
    ) -> Sequence[tuple[Chunk, Document]]:
        """Return document-adjacent chunks (chunk_index ± radius) with their document."""

        if radius < 0:
            raise ValueError("radius must not be negative")
        center = (
            await self.session.execute(
                select(Chunk.document_id, Chunk.chunk_index).where(Chunk.id == chunk_id)
            )
        ).one_or_none()
        if center is None:
            return ()
        document_id, chunk_index = center
        statement = (
            select(Chunk, Document)
            .join(Document, Document.id == Chunk.document_id)
            .where(
                Chunk.document_id == document_id,
                Chunk.chunk_index.between(
                    max(0, chunk_index - radius), chunk_index + radius
                ),
                Chunk.id != chunk_id,
            )
            .order_by(Chunk.chunk_index)
        )
        return tuple((await self.session.execute(statement)).all())

    async def get_context(self, document_id: UUID, chunk_id: UUID) -> Chunk | None:
        statement = select(Chunk).where(
            Chunk.id == chunk_id,
            Chunk.document_id == document_id,
        )
        return (await self.session.execute(statement)).scalar_one_or_none()

    async def delete_document_row(self, document_id: UUID) -> None:
        result = await self.session.execute(
            delete(Document).where(Document.id == document_id).returning(Document.id)
        )
        if result.scalar_one_or_none() is None:
            raise DocumentNotFoundError(f"document {document_id} was not found")

    @staticmethod
    def _product_advisory_key(product: str) -> int:
        digest = hashlib.blake2b(product.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=True)
