"""Database models, lifecycle helpers, and repository API."""

from app.db.models import Base, Chunk, Document, DocumentStatus, KnowledgeChunkKind
from app.db.repo import (
    CandidateActivationError,
    ChunkRecord,
    DocumentCreateData,
    DocumentFilters,
    DocumentNotFoundError,
    RagRepository,
    SearchScope,
)
from app.db.schema import (
    DatabaseTargetError,
    DependencyHealth,
    MigrationStateError,
    assert_database_target,
    assert_migration_head,
    database_health,
)
from app.db.session import create_async_engine_from, create_session_factory, session_scope

__all__ = [
    "Base",
    "CandidateActivationError",
    "Chunk",
    "ChunkRecord",
    "DatabaseTargetError",
    "DependencyHealth",
    "Document",
    "DocumentCreateData",
    "DocumentFilters",
    "DocumentNotFoundError",
    "DocumentStatus",
    "KnowledgeChunkKind",
    "MigrationStateError",
    "RagRepository",
    "SearchScope",
    "assert_database_target",
    "assert_migration_head",
    "create_async_engine_from",
    "create_session_factory",
    "database_health",
    "session_scope",
]
