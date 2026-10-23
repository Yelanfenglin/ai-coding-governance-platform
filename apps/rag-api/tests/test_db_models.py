from __future__ import annotations

import os
from pathlib import Path

import pytest
from sqlalchemy import CheckConstraint, UniqueConstraint

from app.db.models import Chunk, Document, DocumentStatus, KnowledgeChunkKind
from app.db.schema import assert_database_target, database_health
from app.db.session import create_async_engine_from
from app.settings import Settings


def _unique_columns(table: object) -> set[tuple[str, ...]]:
    return {
        tuple(column.name for column in constraint.columns)
        for constraint in table.constraints  # type: ignore[attr-defined]
        if isinstance(constraint, UniqueConstraint)
    }


def _check_sql(table: object) -> set[str]:
    return {
        str(constraint.sqltext)
        for constraint in table.constraints  # type: ignore[attr-defined]
        if isinstance(constraint, CheckConstraint)
    }


def test_document_columns_enums_and_constraints() -> None:
    table = Document.__table__

    assert set(table.c) >= {
        table.c.id,
        table.c.product,
        table.c.product_family,
        table.c.version,
        table.c.language,
        table.c.title,
        table.c.filename,
        table.c.source_url,
        table.c.file_sha256,
        table.c.status,
        table.c.current,
        table.c.stored_by,
        table.c.created_at,
        table.c.updated_at,
        table.c.superseded_at,
        table.c.parse_error,
        table.c.ingest_stage,
        table.c.progress,
        table.c.task_started_at,
        table.c.heartbeat_at,
    }
    assert table.c.status.type.enums == [status.value for status in DocumentStatus]
    assert ("product", "version", "file_sha256") in _unique_columns(table)
    assert "progress BETWEEN 0 AND 100" in _check_sql(table)


def test_document_indexes_include_partial_current_guard() -> None:
    indexes = {index.name: index for index in Document.__table__.indexes}

    expected_indexes = {
        "ix_documents_product_version",
        "ix_documents_file_sha256",
        "ix_documents_status",
    }
    assert expected_indexes <= set(indexes)
    current_index = indexes["uq_documents_current_product"]
    assert current_index.unique is True
    assert tuple(column.name for column in current_index.columns) == ("product",)
    assert str(current_index.dialect_options["postgresql"]["where"]) == "current IS TRUE"


def test_chunk_columns_constraints_and_reserved_metadata_name() -> None:
    table = Chunk.__table__

    assert {
        "logical_chunk_id",
        "semantic_type",
        "embedding_text",
        "chunk_metadata",
    } <= set(table.c.keys())
    assert table.c.embedding_id.nullable is False
    assert table.c.chunk_metadata.nullable is False
    assert table.c.kind.type.enums == [kind.value for kind in KnowledgeChunkKind]
    assert KnowledgeChunkKind.CODE.value == "CODE"
    assert ("document_id", "chunk_index") in _unique_columns(table)
    assert ("embedding_id",) in _unique_columns(table)
    assert "page_start > 0 AND page_end >= page_start" in _check_sql(table)
    assert "metadata" not in Chunk.__mapper__.attrs
    assert "chunk_metadata" in Chunk.__mapper__.attrs

    foreign_key = next(iter(table.c.document_id.foreign_keys))
    assert foreign_key.target_fullname == "documents.id"
    assert foreign_key.ondelete == "CASCADE"
    assert {index.name for index in table.indexes} >= {"ix_chunks_document_id", "ix_chunks_kind"}


RAG_TEST_PG = os.getenv("RAG_TEST_PG")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    not RAG_TEST_PG,
    reason="set RAG_TEST_PG to an isolated postgresql+psycopg rag_kb test URL",
)
async def test_async_database_target_and_health(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        database_url=RAG_TEST_PG,
        milvus_uri="http://127.0.0.1:19531",
        embedding_provider="deterministic",
        embedding_model="deterministic-hash-v1",
        storage_dir=tmp_path / "storage",
        paddle_cache=tmp_path / "paddle-cache",
    )
    engine = create_async_engine_from(settings)
    try:
        await assert_database_target(engine)
        health = await database_health(engine)
        assert health.ok
    finally:
        await engine.dispose()
