"""Public ingestion contracts."""

from .adapter import (
    AuthoritativeDocumentIdentity,
    ChunkKind,
    ChunkMappingError,
    PersistenceChunk,
    load_jsonl,
    map_chunk_records,
    map_jsonl,
    map_records,
)
from .pipeline import (
    DocumentStatus,
    IngestCommand,
    IngestError,
    IngestInvariantError,
    IngestPipeline,
    IngestResult,
    IngestStateError,
    ParsedArtifact,
    PipelineConfigurationError,
    VectorRecord,
)

__all__ = [
    "AuthoritativeDocumentIdentity",
    "ChunkKind",
    "ChunkMappingError",
    "DocumentStatus",
    "IngestCommand",
    "IngestError",
    "IngestInvariantError",
    "IngestPipeline",
    "IngestResult",
    "IngestStateError",
    "ParsedArtifact",
    "PersistenceChunk",
    "PipelineConfigurationError",
    "VectorRecord",
    "load_jsonl",
    "map_chunk_records",
    "map_jsonl",
    "map_records",
]
