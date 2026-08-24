"""Chunking layer for the parsed document (retrieval view over the parser tree).

Modules:
    loader    storage/parsed/{doc} -> ParsedDocument
    units     ParsedDocument -> SemanticUnit list
    chunker   SemanticUnit -> chunks (content + embedding_text + metadata)
    metadata  ChunkMetadata schema, chunk_id, token estimation
"""

from .chunker import Chunk, build_chunks  # noqa: F401
from .loader import ParsedDocument, load_parsed_document
from .units import Block, SemanticUnit, build_semantic_units  # noqa: F401

__all__ = [
    "ParsedDocument",
    "load_parsed_document",
    "Block",
    "SemanticUnit",
    "build_semantic_units",
    "Chunk",
    "build_chunks",
]
