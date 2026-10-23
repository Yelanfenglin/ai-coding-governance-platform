"""Create documents and chunks.

Revision ID: 0001_documents_chunks
Revises:
Create Date: 2026-08-31
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001_documents_chunks"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


document_status_enum = postgresql.ENUM(
    "pending",
    "parsing",
    "parsed",
    "failed",
    "superseded",
    name="document_status",
    create_type=False,
)
knowledge_chunk_kind_enum = postgresql.ENUM(
    "TEXT",
    "TABLE",
    "VISUAL",
    "MIXED",
    "CODE",
    name="knowledge_chunk_kind",
    create_type=False,
)


def upgrade() -> None:
    bind = op.get_bind()
    postgresql.ENUM(
        "pending",
        "parsing",
        "parsed",
        "failed",
        "superseded",
        name="document_status",
    ).create(bind, checkfirst=True)
    postgresql.ENUM(
        "TEXT",
        "TABLE",
        "VISUAL",
        "MIXED",
        "CODE",
        name="knowledge_chunk_kind",
    ).create(bind, checkfirst=True)

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("product", sa.String(length=128), nullable=False),
        sa.Column("product_family", sa.String(length=128), nullable=True),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("language", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("source_url", sa.String(length=2048), nullable=True),
        sa.Column("file_sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            document_status_enum,
            server_default=sa.text("'pending'::document_status"),
            nullable=False,
        ),
        sa.Column("current", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("stored_by", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("superseded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("parse_error", sa.Text(), nullable=True),
        sa.Column(
            "ingest_stage", sa.String(length=64), server_default="pending", nullable=False
        ),
        sa.Column("progress", sa.SmallInteger(), server_default=sa.text("0"), nullable=False),
        sa.Column("task_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("progress BETWEEN 0 AND 100", name="ck_documents_progress_range"),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
        sa.UniqueConstraint(
            "product",
            "version",
            "file_sha256",
            name="uq_documents_product_version_file_sha256",
        ),
    )
    op.create_index("ix_documents_file_sha256", "documents", ["file_sha256"], unique=False)
    op.create_index("ix_documents_product_version", "documents", ["product", "version"])
    op.create_index("ix_documents_status", "documents", ["status"], unique=False)
    op.create_index(
        "uq_documents_current_product",
        "documents",
        ["product"],
        unique=True,
        postgresql_where=sa.text("current IS TRUE"),
    )

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("kind", knowledge_chunk_kind_enum, nullable=False),
        sa.Column("section_path", sa.String(length=1024), nullable=False),
        sa.Column("section_title", sa.String(length=512), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=False),
        sa.Column("page_end", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("table_repr", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("image_ref", sa.String(length=2048), nullable=True),
        sa.Column("embedding_id", sa.String(length=128), nullable=False),
        sa.Column("logical_chunk_id", sa.String(length=512), nullable=False),
        sa.Column("semantic_type", sa.String(length=64), nullable=False),
        sa.Column("embedding_text", sa.Text(), nullable=False),
        sa.Column(
            "chunk_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "page_start > 0 AND page_end >= page_start",
            name="ck_chunks_valid_page_range",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_chunks_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chunks")),
        sa.UniqueConstraint(
            "document_id", "chunk_index", name="uq_chunks_document_chunk_index"
        ),
        sa.UniqueConstraint("embedding_id", name="uq_chunks_embedding_id"),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"], unique=False)
    op.create_index("ix_chunks_kind", "chunks", ["kind"], unique=False)


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_table("documents")
    bind = op.get_bind()
    postgresql.ENUM(name="knowledge_chunk_kind").drop(bind, checkfirst=True)
    postgresql.ENUM(name="document_status").drop(bind, checkfirst=True)
