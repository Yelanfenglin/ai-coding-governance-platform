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
from .units import Block, SemanticUnit

TARGET = 512
SOFT_MAX = 800
NORMAL_MAX = 1200
SOFT_MIN = 150
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*")

_LANGUAGE_ALIASES = {
    "asm": "asm",
    "assembly": "asm",
    "bash": "bash",
    "c": "c",
    "c#": "csharp",
    "c++": "cpp",
    "cpp": "cpp",
    "csharp": "csharp",
    "go": "go",
    "golang": "go",
    "html": "html",
    "java": "java",
    "javascript": "javascript",
    "js": "javascript",
    "json": "json",
    "kotlin": "kotlin",
    "lua": "lua",
    "python": "python",
    "py": "python",
    "rust": "rust",
    "sh": "bash",
    "shell": "bash",
    "sql": "sql",
    "typescript": "typescript",
    "ts": "typescript",
    "xml": "xml",
    "yaml": "yaml",
    "yml": "yaml",
}
_LANGUAGE_HINTS = (
    (re.compile(r"(?:c\+\+|cpp|C＋＋)", re.I), "cpp"),
    (re.compile(r"(?:c#|csharp)", re.I), "csharp"),
    (re.compile(r"(?:python|python\s*语言)", re.I), "python"),
    (re.compile(r"(?:javascript|java\s*script|\bjs\b)", re.I), "javascript"),
    (re.compile(r"(?:typescript|\bts\b)", re.I), "typescript"),
    (re.compile(r"(?:rust|rust\s*语言)", re.I), "rust"),
    (re.compile(r"(?:golang|go\s*语言)", re.I), "go"),
    (re.compile(r"(?:汇编|assembly|\basm\b)", re.I), "asm"),
    (re.compile(r"(?:shell|bash|脚本)", re.I), "bash"),
    (re.compile(r"(?:json)", re.I), "json"),
    (re.compile(r"(?:sql)", re.I), "sql"),
    (re.compile(r"(?:java\s*语言|\bjava\b)", re.I), "java"),
    (re.compile(r"(?:c\s*语言|语言\s*[:：]?\s*c\b)", re.I), "c"),
)

CHUNKER_VERSION = "0.3.2"


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

    @property
    def kind(self) -> str:
        return self.metadata.kind

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


def _retrieval_text(title: str, body: str) -> str:
    """Keep local context once; product/version and full paths live in metadata."""
    title = _collapse(title)
    return "\n".join(part for part in (title if title not in body else "", body) if part)


def _label_in_text(label: str, text: str) -> bool:
    # AD1 must not disappear merely because AD10 appears in the description.
    return bool(re.search(r"(?<![A-Za-z0-9_])" + re.escape(label) + r"(?![A-Za-z0-9_])", text))


def _language_from_hint(value: object) -> str:
    text = str(value or "").strip()
    direct = _LANGUAGE_ALIASES.get(text.lower())
    if direct:
        return direct
    for pattern, language in _LANGUAGE_HINTS:
        if pattern.search(text):
            return language
    return ""


def _infer_code_language(raw: dict, code_text: str, unit: SemanticUnit) -> str:
    for key in ("language", "language_hint", "code_language"):
        language = _language_from_hint(raw.get(key))
        if language:
            return language

    context = " ".join(
        str(value or "")
        for value in (
            raw.get("hint"),
            raw.get("caption"),
            raw.get("title"),
            raw.get("description"),
            unit.title,
        )
    )
    language = _language_from_hint(context)
    if language:
        return language

    if re.search(r"^\s*#!.*(?:bash|sh)\b", code_text):
        return "bash"
    if re.search(r"\b(?:fn\s+main|let\s+mut|println!)\b", code_text):
        return "rust"
    if re.search(r"\bpackage\s+main\b|\bfunc\s+\w+\s*\(", code_text):
        return "go"
    if re.search(r"\b(?:def|from|import)\s+\w+|\bNone\b|\bself\.", code_text):
        return "python"
    if re.search(r"\bpublic\s+static\s+void\s+main\b|\bSystem\.out\.", code_text):
        return "java"
    if re.search(r"\busing\s+System\b|\bConsole\.WriteLine\b", code_text):
        return "csharp"
    if re.search(r"\b(?:interface|type)\s+\w+\s*[={]", code_text) and re.search(
        r":\s*(?:string|number|boolean)\b", code_text
    ):
        return "typescript"
    if re.search(r"\b(?:const|let|var)\s+\w+|=>|\bconsole\.log\b", code_text):
        return "javascript"
    if re.search(r"^\s*(?:SELECT|INSERT|UPDATE|DELETE|CREATE)\b", code_text, re.I):
        return "sql"
    if re.search(r"^\s*[\[{]", code_text) and re.search(r'"[^"\n]+"\s*:', code_text):
        return "json"
    if re.search(r"#\s*include\b|\b(?:void|int|char|uint\d*_t)\s+\w+\s*\(", code_text):
        cpp = re.search(r"\bstd::|#\s*include\s*<iostream>|\bclass\s+", code_text)
        return "cpp" if cpp else "c"
    return ""


def _render_fenced_code(code_text: str, language: str) -> str:
    longest = max((len(run) for run in re.findall(r"`+", code_text)), default=0)
    fence = "`" * max(3, longest + 1)
    closing_gap = "" if code_text.endswith(("\n", "\r")) else "\n"
    return f"{fence}{language}\n{code_text}{closing_gap}{fence}"


def _apply_snapshot_metadata(meta: ChunkMetadata, raw: dict, image_ref: str) -> None:
    meta.image_ref = image_ref
    # Never clobber a description that a caller already assigned from the figure
    # description sidecar (the raw block has no image_description of its own).
    if not meta.image_description:
        meta.image_description = raw.get("description") or raw.get("image_description")
    meta.figure_type = str(raw.get("figure_type") or "")
    meta.extraction_method = str(raw.get("extraction_method") or "")
    confidence = raw.get("extraction_confidence")
    meta.extraction_confidence = confidence if isinstance(confidence, (int, float)) else None
    bbox = raw.get("bbox") or []
    meta.bbox = list(bbox) if isinstance(bbox, (list, tuple)) else []


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
    groups: list[list[int]], rows: list[list[str]], schema: list[str], context_tokens: int = 0
) -> list[list[list[str]]]:
    """Pack consecutive row-groups into chunks sized to SOFT_MAX tokens.

    Never splits a group (rowspan family stays intact).  A legal tiny group is
    not force-expanded to meet SOFT_MIN.
    """
    packed: list[list[list[str]]] = []
    current: list[list[str]] = []
    reserved = context_tokens + estimate_tokens(_render_markdown(schema, []))
    tokens = reserved
    for group in groups:
        group_tokens = sum(estimate_tokens(" ".join(row)) for row in (rows[r] for r in group))
        if current and tokens + group_tokens > SOFT_MAX:
            packed.append(current)
            current = []
            tokens = reserved
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


def _build_from_raw_table(doc: ParsedDocument, unit: SemanticUnit, table_id: str) -> list[Chunk]:
    raw = unit.table_raw
    before: list[str] = []
    after: list[str] = []
    seen_table = False
    for block in unit.blocks:
        if block.kind == "table":
            seen_table = True
        elif block.kind == "paragraph" and block.text:
            (after if seen_table else before).append(block.text)
    context_tokens = estimate_tokens("\n\n".join([*before, *after]))
    rows = raw.get("rows", [])
    semantic = raw.get("semantic_rows")
    if semantic:
        schema, _, body_groups = _plan_semantic_groups(raw)
        body_rows = semantic[1:]
        body_indexed = [list(_row_values(r)) for r in body_rows]
        packed = _pack_groups(body_groups, body_indexed, schema, context_tokens)
    else:
        schema, _, body_groups = _plan_raw_groups(raw)
        body_indexed = [list(_row_values(r)) for r in rows[1:]]
        body_groups_idx = [[r - 1 for r in group] for group in body_groups]
        packed = _pack_groups(body_groups_idx, body_indexed, schema, context_tokens)

    chunks = []
    for group_rows in packed:
        content = "\n\n".join([*before, _render_markdown(schema, group_rows), *after])
        group_label = " | ".join(x for x in (group_rows[0] if group_rows else []) if x)[:60]
        meta = _base_meta(doc, unit)
        meta.kind = "TABLE"
        meta.semantic_type = "table"
        meta.block_types = list(dict.fromkeys(block.kind for block in unit.blocks))
        meta.table_id = table_id
        meta.row_group = [group_label]
        text = _retrieval_text(unit.title, content)
        chunks.append(_finalize(meta, content, text))
    _link_chunks(chunks)
    return chunks


def _build_register(doc: ParsedDocument, unit: SemanticUnit) -> Chunk:
    blocks = unit.blocks
    parts = []
    retrieval_parts = []
    code_refs = []
    for block in blocks:
        if block.kind == "heading" or block.kind == "paragraph":
            if block.text:
                parts.append(block.text)
                retrieval_parts.append(block.text)
        elif block.kind == "table":
            raw = block.raw
            rows = raw.get("rows", [])
            if rows:
                table = _render_markdown(
                    list(_row_values(rows[0])), [list(_row_values(r)) for r in rows[1:]]
                )
                parts.append(table)
                retrieval_parts.append(table)
        elif block.kind == "code":
            segment = _code_segment(block, unit)
            parts.append(_render_fenced_code(segment["text"], segment["language"]))
            retrieval_parts.append(segment["text"])
            code_refs.append(segment["code_ref"])
    content = "\n\n".join(parts)
    bits = _serialize_bits(unit.table_raw) if unit.table_raw else ""
    text = (
        _breadcrumb(doc, unit)
        + "\n类型：寄存器定义\n"
        + (f"寄存器：{unit.register_name}\n" if unit.register_name else "")
    )
    if bits:
        text += "位定义：\n" + bits
    text += "\n" + "\n\n".join(retrieval_parts)
    meta = _base_meta(doc, unit)
    meta.kind = "MIXED"
    meta.semantic_type = "register"
    meta.atomic = True
    meta.block_types = ["heading", "table", "paragraph"]
    meta.register = unit.register_name or ""
    meta.object_key = unit.register_name or ""
    meta.code_refs = code_refs
    chunk = _finalize(meta, content, text)
    chunk.metadata.block_types = [
        b.kind for b in blocks if b.kind in ("heading", "paragraph", "table", "code")
    ]
    return chunk


def _build_figure(doc: ParsedDocument, unit: SemanticUnit) -> Chunk:
    block = unit.blocks[0]
    raw = block.raw
    image_ref = raw.get("image_ref", "")
    annotations = raw.get("annotations", "")
    description = block.description or {}
    content = f"![图]({image_ref})"
    if annotations:
        content += f"\n<!-- annotations: {annotations} -->"
    facts = description.get("full") or _collapse(annotations)
    text = _retrieval_text(unit.title, facts)
    labels = description.get("labels") or []
    extra_labels: list[str] = []
    for label in labels:
        if label and label not in extra_labels and not _label_in_text(label, text):
            extra_labels.append(label)
    if extra_labels:
        text += "\n" + "、".join(extra_labels)
    uncertainties = description.get("uncertainties") or []
    meta = _base_meta(doc, unit)
    meta.kind = "VISUAL"
    meta.semantic_type = "figure"
    meta.block_types = ["snapshot"]
    meta.figure_id = block.figure_id
    meta.figure_description_status = (
        "described" if description else str(raw.get("description_status") or "missing")
    )
    meta.figure_uncertainties = list(uncertainties)
    excluded = bool(raw.get("figure_excluded"))
    if excluded:
        meta.exclude_from_embedding = True
    if description.get("full"):
        meta.image_description = description["full"]
    _apply_snapshot_metadata(meta, raw, image_ref)
    if meta.figure_type == "code" and meta.extraction_method == "none":
        meta.atomic = True
        meta.structured = False
        text = _breadcrumb(doc, unit) + "\n类型：待转录代码图"
        if annotations:
            text += "\n" + _collapse(annotations)
    return _finalize(meta, content, text, exclude=excluded)


def _code_segment(block: Block, unit: SemanticUnit) -> dict:
    """An indivisible inline segment; its source stays separate from retrieval text."""
    raw = block.raw
    code_text = raw.get("code_text", "")
    if not isinstance(code_text, str) or not code_text:
        raise ValueError("inline code requires non-empty code_text")
    language = _infer_code_language(raw, code_text, unit)
    return {
        "type": "code", "text": code_text, "language": language,
        "source_pages": [block.page],
        "code_ref": {
            "figure_id": block.figure_id or str(raw.get("code_figure_id") or ""),
            "sidecar_figure_id": str(raw.get("code_figure_id") or ""),
            "image_ref": str(raw.get("image_ref") or ""),
            "page": block.page, "bbox": list(raw.get("bbox") or []),
            "language": language,
        },
    }


def _build_toc(doc: ParsedDocument, unit: SemanticUnit) -> Chunk:
    text = "\n".join((b.text or "") for b in unit.blocks if b.text)
    meta = _base_meta(doc, unit)
    meta.kind = "TEXT"
    meta.semantic_type = "toc"
    meta.block_types = ["text"]
    chunk = _finalize(meta, text, "")
    chunk.exclude_from_embedding = True
    return chunk


def _text_segments(unit: SemanticUnit) -> list[dict]:
    """Ordered paragraphs, recovered code and inline figure descriptions."""
    segments: list[dict] = []
    for block in unit.blocks:
        if block.kind == "paragraph" and block.text and block.text.strip():
            segments.append(
                {"type": "paragraph", "text": block.text.strip(), "source_pages": [block.page]}
            )
        elif block.kind == "code":
            segments.append(_code_segment(block, unit))
        elif block.kind == "figure":
            description = block.description or {}
            segments.append(
                {
                    "type": "figure",
                    "image_ref": block.raw.get("image_ref", ""),
                    "figure_id": block.figure_id,
                    "short": description.get("short", ""),
                    "source_pages": [block.page],
                }
            )
    return segments


def _segment_tokens(segment: dict) -> int:
    if segment["type"] in ("paragraph", "code"):
        return estimate_tokens(segment["text"])
    return estimate_tokens(segment.get("short") or "") + 4


def _build_text(doc: ParsedDocument, unit: SemanticUnit) -> list[Chunk]:
    segments = _text_segments(unit)
    if not any(segment["type"] in ("paragraph", "code") for segment in segments):
        # A figure-only section already has independent figure chunks. There is no
        # prose to annotate, so do not duplicate it as a text wrapper.
        return []

    # pack segments; figures no longer break the flow, and an oversized paragraph run is
    # split at sentence boundaries.  Never crosses a heading (a TextUnit is one section).
    groups: list[list[dict]] = []
    current: list[dict] = []
    tokens = 0
    for segment in segments:
        segment_tokens = _segment_tokens(segment)
        if current and tokens + segment_tokens > SOFT_MAX:
            groups.append(current)
            current = []
            tokens = 0
        current.append(segment)
        tokens += segment_tokens
    if current:
        groups.append(current)

    packed: list[list[dict]] = []
    for group in groups:
        if sum(_segment_tokens(segment) for segment in group) <= NORMAL_MAX:
            packed.append(group)
            continue
        sentence_group: list[dict] = []
        s_tokens = 0
        for segment in group:
            if segment["type"] != "paragraph":
                st = _segment_tokens(segment)
                if sentence_group and s_tokens + st > SOFT_MAX:
                    packed.append(sentence_group)
                    sentence_group = []
                    s_tokens = 0
                sentence_group.append(segment)
                s_tokens += st
                continue
            for sentence in _split_sentences(segment["text"]):
                st = estimate_tokens(sentence)
                if sentence_group and s_tokens + st > SOFT_MAX:
                    packed.append(sentence_group)
                    sentence_group = []
                    s_tokens = 0
                sentence_group.append({**segment, "text": sentence})
                s_tokens += st
        if sentence_group:
            packed.append(sentence_group)

    # semantic overlap: every chunk after the first carries the previous chunk's
    # trailing sentences as a lead-in.  The tail comes from the original paragraphs,
    # never from rendered content, so image references cannot leak into retrieval text.
    chunks: list[Chunk] = []
    previous_paragraphs = ""
    previous_pages: list[int] = []
    for i, group in enumerate(packed):
        group_paragraphs = " ".join(
            segment["text"] for segment in group if segment["type"] == "paragraph"
        )
        group_pages = sorted(
            {page for s in group if s["type"] == "paragraph" for page in s.get("source_pages", [])}
        )
        if i > 0 and previous_paragraphs:
            prev_tail = _tail_sentences(previous_paragraphs, limit=60)
            if (prev_tail and estimate_tokens(prev_tail)
                    + sum(_segment_tokens(s) for s in group) <= NORMAL_MAX):
                group = [
                    {"type": "paragraph", "text": prev_tail, "source_pages": previous_pages},
                    *group,
                ]
        chunks.append(_text_chunk(doc, unit, group, i))
        previous_paragraphs = group_paragraphs
        previous_pages = group_pages
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
    doc: ParsedDocument, unit: SemanticUnit, segments: list[dict], index: int
) -> Chunk:
    content_parts: list[str] = []
    retrieval_parts: list[str] = []
    figure_ids: list[str] = []
    code_refs: list[dict] = []
    for segment in segments:
        if segment["type"] == "paragraph":
            content_parts.append(segment["text"])
            retrieval_parts.append(segment["text"])
            continue
        if segment["type"] == "code":
            content_parts.append(_render_fenced_code(segment["text"], segment["language"]))
            retrieval_parts.append(segment["text"])
            code_refs.append(segment["code_ref"])
            continue
        content_parts.append(f'![图]({segment["image_ref"]})')
        retrieval_parts.append(
            f'[图：{segment["short"]}]' if segment.get("short") else "[图]"
        )
        if segment.get("figure_id"):
            figure_ids.append(segment["figure_id"])
    content = "\n\n".join(content_parts)
    meta = _base_meta(doc, unit)
    meta.kind = "TEXT"
    meta.semantic_type = "text"
    meta.block_types = list(dict.fromkeys(s["type"] for s in segments))
    meta.code_refs = code_refs
    meta.atomic = len(segments) == 1 and segments[0]["type"] == "code"
    meta.figure_ids = sorted(set(figure_ids))
    pages = sorted({page for s in segments for page in s.get("source_pages", [])})
    if pages:
        meta.source_pages = pages
        meta.page_start, meta.page_end = pages[0], pages[-1]
    text = _retrieval_text(unit.title, "\n\n".join(retrieval_parts))
    notice = "".join(content.split())
    reference_only = bool(re.fullmatch(
        r"(?:指令集详细说明请到官网下载[:：].*指令说明"
        r"|注[:：]关于.{1,30}应用请参照DEMO相关例程[。.]?)", notice
    ))
    return _finalize(meta, content, text, exclude=reference_only)


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
    meta.embedding_tokens = estimate_tokens(embedding_text)
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

    # explicit text <-> figure associations (no recall strategy, relations only)
    figure_by_id = {
        chunk.metadata.figure_id: chunk
        for chunk in level1
        if chunk.metadata.semantic_type == "figure" and chunk.metadata.figure_id
    }
    for chunk in level1:
        if chunk.metadata.semantic_type != "text":
            continue
        for figure_id in chunk.metadata.figure_ids:
            figure = figure_by_id.get(figure_id)
            if figure is None:
                continue
            if chunk.chunk_id not in figure.metadata.related_chunk_ids:
                figure.metadata.related_chunk_ids.append(chunk.chunk_id)

    # register packages also own figures in their section; links stay metadata-only so
    # the atomic register package keeps its boundary.
    register_by_section = {
        chunk.metadata.section_id: chunk
        for chunk in level1
        if chunk.metadata.semantic_type == "register"
    }
    for figure in level1:
        if figure.metadata.semantic_type != "figure" or not figure.metadata.figure_id:
            continue
        register = register_by_section.get(figure.metadata.section_id)
        if register is None:
            continue
        if figure.metadata.figure_id not in register.metadata.figure_ids:
            register.metadata.figure_ids.append(figure.metadata.figure_id)
        if register.chunk_id not in figure.metadata.related_chunk_ids:
            figure.metadata.related_chunk_ids.append(register.chunk_id)

    return level1 + parent_chunks
