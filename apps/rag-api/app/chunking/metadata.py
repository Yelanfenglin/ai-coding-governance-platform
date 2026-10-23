"""Chunk metadata schema, chunk_id generation and token estimation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

SEMANTIC_TYPES = ("text", "register", "table", "figure", "code", "toc", "parent", "revision")
CHUNK_KINDS = ("TEXT", "TABLE", "VISUAL", "MIXED", "CODE")


def estimate_tokens(text: str) -> int:
    """Rough token estimate for soft limits.

    CJK characters count as ~1 token each; other printable non-space characters
    count as ~1 token per 3 characters (English words / code).  Good enough to
    hit 300-800 token targets; not a tokenizer.
    """
    if not text:
        return 0
    cjk = 0
    other = 0
    for ch in text:
        if ch.isspace():
            continue
        if "\u3400" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff" or "\u4dc0" <= ch <= "\u4dff":
            cjk += 1
        else:
            other += 1
    return cjk + (other + 2) // 3


@dataclass
class ChunkMetadata:
    # identity
    doc_id: str = ""
    product: str = ""
    version: str = ""
    pdf_name: str = ""
    chunk_id: str = ""
    parent_id: str = ""
    prev_id: str = ""
    next_id: str = ""
    object_key: str = ""  # readable object name (register etc.), separate from chunk_id
    # location & hierarchy
    section_id: str = ""
    section_path: list[str] = field(default_factory=list)
    heading_level: int = 2
    # semantics & content
    kind: str = ""  # coarse kind: TEXT|TABLE|VISUAL|MIXED|CODE
    semantic_type: str = ""  # SEMANTIC_TYPES
    block_types: list[str] = field(default_factory=list)
    register: str = ""
    table_id: str = ""
    row_group: list[str] = field(default_factory=list)
    instruction_category: str = ""
    # retrieval text (content stays separate in the Chunk)
    embedding_text: str = ""
    title: str = ""
    image_ref: str = ""
    image_description: str | None = None
    figure_id: str = ""
    figure_description_status: str = ""
    figure_uncertainties: list[str] = field(default_factory=list)
    figure_ids: list[str] = field(default_factory=list)
    related_chunk_ids: list[str] = field(default_factory=list)
    figure_type: str = ""
    code_language: str = ""
    code_refs: list[dict] = field(default_factory=list)
    extraction_method: str = ""
    extraction_confidence: float | None = None
    # provenance
    page_start: int = 0
    page_end: int = 0
    source_pages: list[int] = field(default_factory=list)
    bbox: list[float] = field(default_factory=list)
    # quality / control
    structured: bool = True
    atomic: bool = False  # register / complete-example package, not split
    exclude_from_embedding: bool = False
    chars: int = 0
    tokens: int = 0
    embedding_tokens: int = 0
    parser_version: str = ""
    chunker_version: str = "0.1.0"

    def to_dict(self) -> dict:
        return asdict(self)


def build_chunk_id(doc_id: str, section_id: str, unit_type: str, ordinal: int) -> str:
    sec = section_id or "front"
    return f"{doc_id}:{sec}:{unit_type}:{ordinal:03d}"
