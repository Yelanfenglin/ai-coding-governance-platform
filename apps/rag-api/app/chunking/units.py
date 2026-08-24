"""Semantic reconstruction: parsed pages -> SemanticUnit objects.

Chunker only sees these abstractions, never the raw PDF.  Rules use structural
signals only (heading level, bit-header row, rowspan groups, page markers,
captions); no document name / page number / chip-field vocabulary enters here.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field

from .loader import ParsedDocument

_BIT_HEADER_CELL_RE = re.compile(r"^Bit\s*(\d+|\[\d+:\d+\])$")
_BIT_DESC_RE = re.compile(r"^Bit\s*(\d+|\[\d+:\d+\])")
_TOC_TITLE = re.compile(r"^\s*(?:目\s*录|目\s*次|contents)\s*$", re.I)

UNIT_TYPE = ("text", "register", "table", "figure", "toc")


@dataclass
class Block:
    kind: str  # heading | paragraph | table | figure
    text: str | None = None
    raw: dict = field(default_factory=dict)
    page: int = 0
    section_id: str = ""
    section_path: tuple[str, ...] = ()


@dataclass
class SemanticUnit:
    unit_type: str  # UNIT_TYPE
    section_id: str = ""
    section_path: tuple[str, ...] = ()
    title: str = ""
    heading_level: int = 2
    blocks: list[Block] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0
    source_pages: list[int] = field(default_factory=list)
    atomic: bool = False
    register_name: str | None = None
    table_id: str | None = None
    table_raw: dict | None = None  # the table block dict (for chunker row grouping)


def _normalize(raw: dict, page: int) -> Block:
    typ = raw.get("type")
    if typ == "text":
        kind = raw.get("kind", "paragraph")
        return Block(
            kind=kind,
            text=raw.get("text", ""),
            raw=raw,
            page=page,
            section_id=raw.get("section_id", ""),
            section_path=tuple(raw.get("section_path", []) or []),
        )
    if typ == "table":
        return Block(
            kind="table",
            raw=raw,
            page=page,
            section_id=raw.get("section_id", ""),
            section_path=tuple(raw.get("section_path", []) or []),
        )
    if typ == "snapshot":
        return Block(
            kind="figure",
            raw=raw,
            page=page,
            section_id=raw.get("section_id", ""),
            section_path=tuple(raw.get("section_path", []) or []),
        )
    raise ValueError(f"unknown block type: {typ}")


def _bit_header_row(row: list) -> bool:
    """A register/bit table header row: col0 empty and the rest are Bit cells."""
    if len(row) < 2:
        return False
    cells = row[1:]
    bit_cells = [c for c in cells if isinstance(c, str) and _BIT_HEADER_CELL_RE.match(c)]
    return len(bit_cells) >= 2


def _is_bit_table(raw: dict) -> bool:
    rows = raw.get("rows", [])
    return bool(rows) and _bit_header_row(rows[0])


def _has_bit_description(blocks: list[Block]) -> bool:
    for block in blocks:
        if (
            block.kind == "paragraph"
            and block.text
            and any(_BIT_DESC_RE.match(line.strip()) for line in block.text.splitlines())
        ):
            return True
    return False


def _is_register_section(blocks: list[Block]) -> bool:
    tables = [b.raw for b in blocks if b.kind == "table"]
    if any(_is_bit_table(raw) for raw in tables):
        return True
    return _has_bit_description(blocks)


def _first_heading(blocks: list[Block]) -> Block | None:
    return next((b for b in blocks if b.kind == "heading"), None)


def _heading_level(section_id: str) -> int:
    if not section_id:
        return 0
    return section_id.count(".") + 2  # 1 -> H2, 1.1 -> H3 ...


def _unit_pages(blocks: list[Block]) -> tuple[int, int, list[int]]:
    pages: list[int] = []
    for block in blocks:
        pages.append(block.page)
        if block.kind == "table":
            for p in block.raw.get("source_pages", []) or []:
                pages.append(p)
    unique = sorted(set(pages))
    return (unique[0], unique[-1], unique) if unique else (0, 0, [])


def _register_name(raw: dict) -> str | None:
    rows = raw.get("rows", [])
    if rows and len(rows[0]) > 0:
        name = (rows[1][0] if len(rows) > 1 else rows[0][0]).strip()
        if name:
            return name
    return None


def build_semantic_units(doc: ParsedDocument) -> list[SemanticUnit]:
    flat: list[Block] = []
    for page in doc.pages:
        for blk in page.get("blocks", []):
            flat.append(_normalize(blk, page.get("page", 0)))

    groups: OrderedDict[str, list[Block]] = OrderedDict()
    for block in flat:
        key = block.section_id or "__front__"
        groups.setdefault(key, []).append(block)

    units: list[SemanticUnit] = []
    for section_id, blocks in groups.items():
        real_sid = "" if section_id == "__front__" else section_id
        units.extend(_build_units(doc, real_sid, blocks))
    return units


def _build_units(doc: ParsedDocument, section_id: str, blocks: list[Block]) -> list[SemanticUnit]:
    heading = _first_heading(blocks)
    title = (
        (heading.text or "").strip()
        if heading
        else (blocks[0].section_path[-1] if blocks[0].section_path else "")
    )
    if not title and blocks:
        title = blocks[-1].section_path[-1] if blocks[-1].section_path else ""
    section_path = blocks[0].section_path if blocks else ()
    page_start, page_end, source_pages = _unit_pages(blocks)
    common = dict(
        section_id=section_id,
        section_path=section_path,
        title=title,
        heading_level=_heading_level(section_id),
        page_start=page_start,
        page_end=page_end,
        source_pages=source_pages or [page_start],
    )

    # TOC
    if heading and _TOC_TITLE.match("".join((heading.text or "").split())):
        return [SemanticUnit(unit_type="toc", blocks=blocks, **common)]

    # Register package (atomic)
    if _is_register_section(blocks):
        table = next((b.raw for b in blocks if b.kind == "table" and _is_bit_table(b.raw)), None)
        return [
            SemanticUnit(
                unit_type="register",
                blocks=blocks,
                atomic=True,
                register_name=_register_name(table) if table else None,
                table_raw=table,
                **common,
            )
        ]

    units: list[SemanticUnit] = []
    text_run: list[Block] = []

    def flush_text() -> None:
        if text_run:
            units.append(SemanticUnit(unit_type="text", blocks=list(text_run), **common))
            text_run.clear()

    for block in blocks:
        if block.kind == "heading":
            continue  # section title already captured
        if block.kind == "paragraph":
            text_run.append(block)
        elif block.kind == "table":
            flush_text()
            units.append(
                SemanticUnit(unit_type="table", blocks=[block], table_raw=block.raw, **common)
            )
        elif block.kind == "figure":
            flush_text()
            units.append(SemanticUnit(unit_type="figure", blocks=[block], **common))
    flush_text()
    return units
