"""SQLAlchemy models for canonical RAG document and chunk data."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ENUM as PGEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import text as sql_text


class DocumentStatus(str, Enum):  # noqa: UP042 - contract requires a str Enum
    PENDING = "pending"
    PARSING = "parsing"
    PARSED = "parsed"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class KnowledgeChunkKind(str, Enum):  # noqa: UP042 - contract requires a str Enum
    TEXT = "TEXT"
    TABLE = "TABLE"
    VISUAL = "VISUAL"
    MIXED = "MIXED"
    CODE = "CODE"


def _enum_values(enum_class: type[Enum]) -> list[str]:
    return [str(member.value) for member in enum_class]


document_status_enum = PGEnum(
    DocumentStatus,
    name="document_status",
    values_callable=_enum_values,
)
knowledge_chunk_kind_enum = PGEnum(
    KnowledgeChunkKind,
    name="knowledge_chunk_kind",
    values_callable=_enum_values,
)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint(
            "product",
            "version",
            "file_sha256",
            name="uq_documents_product_version_file_sha256",
        ),
        CheckConstraint("progress BETWEEN 0 AND 100", name="ck_documents_progress_range"),
        Index("ix_documents_product_version", "product", "version"),
        Index("ix_documents_file_sha256", "file_sha256"),
        Index("ix_documents_status", "status"),
        Index(
            "uq_documents_current_product",
            "product",
            unique=True,
            postgresql_where=sql_text("current IS TRUE"),
        ),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    product: Mapped[str] = mapped_column(String(128), nullable=False)
    product_family: Mapped[str | None] = mapped_column(String(128))
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    language: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    source_url: Mapped[str | None] = mapped_column(String(2048))
    file_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[DocumentStatus] = mapped_column(
        document_status_enum,
        nullable=False,
        default=DocumentStatus.PENDING,
        server_default=DocumentStatus.PENDING.value,
    )
    current: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=sql_text("false"),
    )
    stored_by: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    superseded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    parse_error: Mapped[str | None] = mapped_column(Text)

    ingest_stage: Mapped[str] = mapped_column(
        String(64), nullable=False, default="pending", server_default="pending"
    )
    progress: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=0, server_default=sql_text("0")
    )
    task_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_chunk_index"),
        UniqueConstraint("embedding_id", name="uq_chunks_embedding_id"),
        CheckConstraint(
            "page_start > 0 AND page_end >= page_start",
            name="ck_chunks_valid_page_range",
        ),
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_kind", "kind"),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[KnowledgeChunkKind] = mapped_column(knowledge_chunk_kind_enum, nullable=False)
    section_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    section_title: Mapped[str] = mapped_column(String(512), nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    table_repr: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    image_ref: Mapped[str | None] = mapped_column(String(2048))
    embedding_id: Mapped[str] = mapped_column(String(128), nullable=False)

    logical_chunk_id: Mapped[str] = mapped_column(String(512), nullable=False)
    semantic_type: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=sql_text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document: Mapped[Document] = relationship(back_populates="chunks")
