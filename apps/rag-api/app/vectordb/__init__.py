"""Typed Milvus vector-store boundary for the RAG service."""

from .milvus_store import (
    DependencyHealth,
    EmbeddingModelMismatchError,
    KnowledgeChunkKind,
    MilvusConfigurationError,
    MilvusSchemaMismatchError,
    MilvusStore,
    MilvusStoreClosedError,
    MilvusStoreError,
    VectorCountMismatchError,
    VectorHit,
    VectorRecord,
    VectorSearchScope,
    create_milvus_store,
)

__all__ = [
    "DependencyHealth",
    "EmbeddingModelMismatchError",
    "KnowledgeChunkKind",
    "MilvusConfigurationError",
    "MilvusSchemaMismatchError",
    "MilvusStore",
    "MilvusStoreClosedError",
    "MilvusStoreError",
    "VectorCountMismatchError",
    "VectorHit",
    "VectorRecord",
    "VectorSearchScope",
    "create_milvus_store",
]
