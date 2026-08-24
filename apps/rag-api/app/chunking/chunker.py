"""SemanticUnit -> chunks.

Structure-aware chunking: heading boundary, register atomic package, table
row-group packing, semantic overlap (schema/breadcrumb/lead sentence), and
separate content / embedding_text.  No fixed-size splitter and no model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .loader import ParsedDocument
from .metadata import ChunkMetadata, build_chunk_id, estimate_tokens
from .units import SemanticUnit

TARGET = 512
SOFT_MAX = 800
NORMAL_MAX = 1200
SOFT_MIN = 150
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*")

CHUNKER_VERSION = "0.1.0"


@dataclass
class Chunk:
    chunk_id: str = ""
    parent_id: str = ""
    prev_id: str = ""
    next_id: str = ""
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
    content: str = ""
    embedding_text: str = ""
    tokens: int = 0
    chars: int = 0
    exclude_from_embedding: bool = False

    def to_dict(self) -> dict:
        d = self.metadata.to_dict()
        d.update(
            {
                "chunk_id": self.chunk_id,
                "parent_id": self.parent_id,
                "prev_id": self.prev_id,
                "next_id": self.next_id,
                "content": self.content,
                "embedding_text": self.embedding_text,
                "exclude_from_embedding": self.exclude_from_embedding,
                "tokens": self.tokens,
                "chars": self.chars,
            }
        )
        return d


def _breadcrumb(doc: ParsedDocument, unit: SemanticUnit) -> str:
    parts = [f"{doc.product} Version {doc.version}"] if doc.version else [doc.product]
    parts.extend(unit.section_path or ())
    return " > ".join(parts)


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _row_values(row) -> list[str]:
    return [str(c or "") for c in row]


def _render_markdown(schema: list[str], body_rows: list[list[str]]) -> str:
    header = ["| " + " | ".join(_row_values(schema)) + " |"]
    sep = "| " + " | ".join(["---"] * len(schema)) + " |"
    body = ["| " + " | ".join(_row_values(r)) + " |" for r in body_rows]
    return "\n".join(header + [sep] + body)


def _col0_groups(cells: list[dict], n_rows: int, start_row: int) -> list[list[int]]:
    """rowspan-aligned groups driven by col0 origins (a parameter/category family)."""
    by_origin = {(c["row"], c["col"]): c for c in cells}
    groups: list[list[int]] = []
    i = start_row
    while i < n_rows:
        cell = by_origin.get((i, 0))
        if cell is not None:
            span = cell.get("rowspan", 1)
            groups.append(list(range(i, i + span)))
            i += span
        else:
            groups.append([i])
            i += 1
    return groups


def _plan_semantic_groups(raw: dict) -> tuple[list[list[str]], int, list[list[int]]]:
    semantic = raw.get("semantic_rows") or []
    rec = raw.get("semantic_reconstruction") or {}
    provenance = rec.get("row_provenance") or []
    parent_of = {}
    for record in provenance:
        parent_of[record.get("semantic_row")] = record.get("parent_row")
    schema = list(_row_values(semantic[0])) if semantic else []
    body = [_row_values(r) for r in semantic[1:]] if len(semantic) > 1 else []
    # group consecutive semantic rows by parent_row (sub-rows of one outer row)
    body_groups: list[list[int]] = []
    last_parent = None
    for i in range(len(body)):
        parent = parent_of.get(i + 1, i + 1)
        if parent != last_parent or not body_groups:
            body_groups.append([i])
            last_parent = parent
        else:
            body_groups[-1].append(i)
    return schema, len(body), body_groups


def _plan_raw_groups(raw: dict) -> tuple[list[list[str]], int, list[list[int]]]:
    rows = raw.get("rows", [])
    n_rows = len(rows)
    schema = list(_row_values(rows[0])) if rows else []
    groups = _col0_groups(raw.get("cells", []), n_rows, 1)
    return schema, n_rows, groups


def _pack_groups(
    groups: list[list[int]], rows: list[list[str]], schema: list[str]
) -> list[list[list[str]]]:
    """Pack consecutive row-groups into chunks sized to SOFT_MAX tokens.

    Never splits a group (rowspan family stays intact).  A legal tiny group is
    not force-expanded to meet SOFT_MIN.
    """
    packed: list[list[list[str]]] = []
    current: list[list[str]] = []
    tokens = 0
    for group in groups:
        group_tokens = sum(estimate_tokens(" ".join(row)) for row in (rows[r] for r in group))
        if current and tokens + group_tokens > SOFT_MAX:
            packed.append(current)
            current = []
            tokens = 0
        current.extend(rows[r] for r in group)
        tokens += group_tokens
    if current:
        packed.append(current)
    return packed


def _serialize_bits(raw: dict) -> str:
    """Bit table -> readable `Bit7 FIELD` lines for embedding_text."""
    rows = raw.get("rows", [])
    if not rows:
        return ""
    header = list(_row_values(rows[0]))
    field_row = list(_row_values(rows[1])) if len(rows) > 1 else []
    lines = []
    for col in range(1, len(header)):
        bit = header[col]
        name = field_row[col] if col < len(field_row) else ""
        if bit and name:
            lines.append(f"{bit} {name}".strip())
    return "\n".join(lines)


def _schema_serialized(schema: list[str]) -> str:
    return "表头: " + " | ".join(x for x in schema if x)


def _build_from_raw_table(doc: ParsedDocument, unit: SemanticUnit, table_id: str) -> list[Chunk]:
    raw = unit.table_raw
    rows = raw.get("rows", [])
    semantic = raw.get("semantic_rows")
    if semantic:
        schema, _, body_groups = _plan_semantic_groups(raw)
        body_rows = semantic[1:]
        body_indexed = [list(_row_values(r)) for r in body_rows]
        packed = _pack_groups(body_groups, body_indexed, schema)
    else:
        schema, _, body_groups = _plan_raw_groups(raw)
        body_indexed = [list(_row_values(r)) for r in rows[1:]]
        body_groups_idx = [[r - 1 for r in group] for group in body_groups]
        packed = _pack_groups(body_groups_idx, body_indexed, schema)

    chunks = []
    for group_rows in packed:
        content = _render_markdown(schema, group_rows)
        group_label = " | ".join(x for x in (body_indexed[0] if body_indexed else []) if x)[:60]
        meta = _base_meta(doc, unit)
        meta.semantic_type = "table"
        meta.block_types = ["table"]
        meta.table_id = table_id
        meta.row_group = [group_label]
        text = (
            _breadcrumb(doc, unit)
            + "\n"
            + meta.table_id
            + "\n"
            + _schema_serialized(schema)
            + "\n"
            + content
        )
        chunks.append(_finalize(meta, content, text))
    _link_chunks(chunks)
    return chunks


def _build_register(doc: ParsedDocument, unit: SemanticUnit) -> Chunk:
    blocks = unit.blocks
    parts = []
    for block in blocks:
        if block.kind == "heading" or block.kind == "paragraph":
            if block.text:
                parts.append(block.text)
        elif block.kind == "table":
            raw = block.raw
            rows = raw.get("rows", [])
            if rows:
                parts.append(
                    _render_markdown(
                        list(_row_values(rows[0])), [list(_row_values(r)) for r in rows[1:]]
                    )
                )
    content = "\n\n".join(parts)
    bits = _serialize_bits(unit.table_raw) if unit.table_raw else ""
    text = (
        _breadcrumb(doc, unit)
        + "\n类型：寄存器定义\n"
        + (f"寄存器：{unit.register_name}\n" if unit.register_name else "")
    )
    if bits:
        text += "位定义：\n" + bits
    text += "\n" + content
    meta = _base_meta(doc, unit)
    meta.semantic_type = "register"
    meta.atomic = True
    meta.block_types = ["heading", "table", "paragraph"]
    meta.register = unit.register_name or ""
    meta.object_key = unit.register_name or ""
    chunk = _finalize(meta, content, text)
    chunk.metadata.block_types = [
        b.kind for b in blocks if b.kind in ("heading", "paragraph", "table")
    ]
    return chunk


def _build_figure(doc: ParsedDocument, unit: SemanticUnit) -> Chunk:
    block = unit.blocks[0]
    raw = block.raw
    image_ref = raw.get("image_ref", "")
    hint = raw.get("hint", "image")
    annotations = raw.get("annotations", "")
    content = f"![图]({image_ref})"
    if annotations:
        content += f"\n<!-- annotations: {annotations} -->"
    text = _breadcrumb(doc, unit) + f"\n类型：图（{hint}）\n图像：{image_ref}"
    if annotations:
        text += "\n" + _collapse(annotations)
    meta = _base_meta(doc, unit)
    meta.semantic_type = "figure"
    meta.block_types = ["snapshot"]
    meta.image_ref = image_ref
    return _finalize(meta, content, text)


def _build_toc(doc: ParsedDocument, unit: SemanticUnit) -> Chunk:
    text = "\n".join((b.text or "") for b in unit.blocks if b.text)
    meta = _base_meta(doc, unit)
    meta.semantic_type = "toc"
    meta.block_types = ["text"]
    chunk = _finalize(meta, text, "")
    chunk.exclude_from_embedding = True
    return chunk


def _build_text(doc: ParsedDocument, unit: SemanticUnit) -> list[Chunk]:
    paragraphs = [b.text.strip() for b in unit.blocks if b.kind == "paragraph" and b.text.strip()]
    if not paragraphs:
        return []

    # pack paragraphs; a group that exceeds NORMAL_MAX is split at sentence
    # boundaries.  Never crosses a heading (a TextUnit is one section).
    groups: list[list[str]] = []
    current: list[str] = []
    tokens = 0
    for para in paragraphs:
        p_tokens = estimate_tokens(para)
        if current and tokens + p_tokens > SOFT_MAX:
            groups.append(current)
            current = []
            tokens = 0
        current.append(para)
        tokens += p_tokens
    if current:
        groups.append(current)

    packed: list[list[str]] = []
    for group in groups:
        if sum(estimate_tokens(p) for p in group) <= NORMAL_MAX:
            packed.append(group)
            continue
        sentence_group: list[str] = []
        s_tokens = 0
        for sentence in _split_sentences(" ".join(group)):
            st = estimate_tokens(sentence)
            if sentence_group and s_tokens + st > SOFT_MAX:
                packed.append([" ".join(sentence_group)])
                sentence_group = []
                s_tokens = 0
            sentence_group.append(sentence)
            s_tokens += st
        if sentence_group:
            packed.append([" ".join(sentence_group)])

    # semantic overlap: every chunk after the first carries the previous
    # chunk's trailing sentences as a lead-in (same section only).
    chunks: list[Chunk] = []
    for i, group in enumerate(packed):
        if i > 0:
            prev_tail = _tail_sentences(chunks[-1].content, limit=60)
            if prev_tail:
                group = [prev_tail] + group
        chunks.append(_text_chunk(doc, unit, group, i))
    _link_chunks(chunks)
    return chunks


def _tail_sentences(text: str, limit: int) -> str:
    """Keep the last sentences of text, capped at ~limit tokens."""
    take: list[str] = []
    for sentence in reversed(_split_sentences(text)):
        if estimate_tokens(" ".join(take)) + estimate_tokens(sentence) > limit:
            break
        take.insert(0, sentence)
    return " ".join(take).strip()


def _split_sentences(text: str) -> list[str]:
    parts = [p for p in _SENTENCE_SPLIT.split(text) if p.strip()]
    return parts or [text]


def _text_chunk(
    doc: ParsedDocument, unit: SemanticUnit, paragraphs: list[str], index: int
) -> Chunk:
    content = "\n\n".join(paragraphs)
    meta = _base_meta(doc, unit)
    meta.semantic_type = "text"
    meta.block_types = ["paragraph"]
    text = _breadcrumb(doc, unit) + "\n" + content
    return _finalize(meta, content, text)


def _base_meta(doc: ParsedDocument, unit: SemanticUnit) -> ChunkMetadata:
    return ChunkMetadata(
        doc_id=doc.doc_id,
        product=doc.product,
        version=doc.version,
        pdf_name=doc.pdf_name,
        section_id=unit.section_id,
        section_path=list(unit.section_path),
        heading_level=unit.heading_level,
        title=unit.title,
        page_start=unit.page_start,
        page_end=unit.page_end,
        source_pages=list(unit.source_pages),
        parser_version="0.4.0",
        chunker_version=CHUNKER_VERSION,
    )


def _finalize(
    meta: ChunkMetadata, content: str, embedding_text: str, *, exclude: bool = False
) -> Chunk:
    meta.content = content
    meta.embedding_text = embedding_text
    meta.chars = len(content)
    meta.tokens = estimate_tokens(content)
    meta.exclude_from_embedding = exclude
    return Chunk(
        metadata=meta,
        content=content,
        embedding_text=embedding_text,
        tokens=meta.tokens,
        chars=meta.chars,
        exclude_from_embedding=exclude,
    )


def _link_chunks(chunks: list[Chunk]) -> None:
    for i, chunk in enumerate(chunks):
        chunk.prev_id = chunks[i - 1].chunk_id if i > 0 else ""
        chunk.next_id = chunks[i + 1].chunk_id if i < len(chunks) - 1 else ""


def build_chunks(doc: ParsedDocument, units: list[SemanticUnit]) -> list[Chunk]:
    level1: list[Chunk] = []
    counters: dict[tuple[str, str], int] = {}

    def next_ordinal(unit: SemanticUnit, unit_type: str) -> int:
        key = (unit.section_id, unit_type)
        counters[key] = counters.get(key, 0) + 1
        return counters[key]

    for unit in units:
        if unit.unit_type == "toc":
            chunk = _build_toc(doc, unit)
            chunk.chunk_id = build_chunk_id(doc.doc_id, unit.section_id, "toc", 1)
            level1.append(chunk)
        elif unit.unit_type == "register":
            chunk = _build_register(doc, unit)
            chunk.chunk_id = build_chunk_id(
                doc.doc_id, unit.section_id, "register", next_ordinal(unit, "register")
            )
            level1.append(chunk)
        elif unit.unit_type == "figure":
            chunk = _build_figure(doc, unit)
            chunk.chunk_id = build_chunk_id(
                doc.doc_id, unit.section_id, "figure", next_ordinal(unit, "figure")
            )
            level1.append(chunk)
        elif unit.unit_type == "table":
            base = next_ordinal(unit, "table")
            table_id = f"{doc.doc_id}:{unit.section_id}:table:{base:03d}"
            chunks = _build_from_raw_table(doc, unit, table_id)
            for i, ch in enumerate(chunks):
                ch.chunk_id = build_chunk_id(doc.doc_id, unit.section_id, "table", base + i)
            level1.extend(chunks)
        elif unit.unit_type == "text":
            chunks = _build_text(doc, unit)
            if not chunks:
                continue
            for ch in chunks:
                ch.chunk_id = build_chunk_id(
                    doc.doc_id, unit.section_id, "text", next_ordinal(unit, "text")
                )
            _link_chunks(chunks)
            level1.extend(chunks)

    # assign parent_id per section and emit Level-2 parent (container) chunks
    children_by_section: dict[tuple[str, str], list[Chunk]] = {}
    for chunk in level1:
        if chunk.exclude_from_embedding:
            continue
        key = (chunk.metadata.doc_id, chunk.metadata.section_id)
        children_by_section.setdefault(key, []).append(chunk)

    parent_chunks: list[Chunk] = []
    for (doc_id, section_id), children in children_by_section.items():
        first = children[0].metadata
        parent_id = build_chunk_id(doc_id, section_id, "parent", 1)
        parent_meta = ChunkMetadata(
            doc_id=doc_id,
            product=first.product,
            version=first.version,
            pdf_name=first.pdf_name,
            chunk_id=parent_id,
            section_id=section_id,
            section_path=first.section_path,
            heading_level=first.heading_level,
            title=first.title,
            page_start=min(c.metadata.page_start for c in children),
            page_end=max(c.metadata.page_end for c in children),
            source_pages=sorted(set().union(*(c.metadata.source_pages for c in children))),
            semantic_type="parent",
            block_types=["section"],
            parser_version=first.parser_version,
            chunker_version=CHUNKER_VERSION,
        )
        content = first.title
        parent = _finalize(parent_meta, content, "", exclude=True)
        parent.chunk_id = parent_id
        parent.exclude_from_embedding = True
        parent.metadata.content = content
        parent_chunks.append(parent)
        for child in children:
            child.parent_id = parent_id

    return level1 + parent_chunks
