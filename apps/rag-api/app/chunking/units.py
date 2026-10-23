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
from .metadata import estimate_tokens

_BIT_HEADER_CELL_RE = re.compile(r"^Bit\s*(\d+|\[\d+:\d+\])$")
_BIT_DESC_RE = re.compile(r"^Bit\s*(\d+|\[\d+:\d+\])")
_TOC_TITLE = re.compile(r"^\s*(?:目\s*录|目\s*次|contents)\s*$", re.I)
_JOINED_BIT_TITLE = re.compile(
    r"(?<=\S)(?=Bit\s*(?:\d+|\[\d+:\d+\])\s*[A-Za-z_]\w*\s*[:：])"
)
_NOTE_START = re.compile(r"^\s*注(?:意)?\s*[:：]")
_NOTE_ITEM = re.compile(r"^\s*[（(]\d+[）)]")

UNIT_TYPE = ("text", "register", "table", "figure", "toc")


@dataclass
class Block:
    kind: str  # heading | paragraph | table | figure | code
    text: str | None = None
    raw: dict = field(default_factory=dict)
    page: int = 0
    section_id: str = ""
    section_path: tuple[str, ...] = ()
    figure_type: str | None = None
    figure_id: str = ""
    description: dict | None = None


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
    figure_type: str | None = None


def _normalize(raw: dict, page: int) -> Block:
    typ = raw.get("type")
    if typ == "text":
        kind = raw.get("kind", "paragraph")
        text = raw.get("text", "")
        if kind == "paragraph":
            # Recover a missing separator only before an explicit Bit FIELD: title.
            # Code snapshots and the saved parser output remain untouched.
            text = _JOINED_BIT_TITLE.sub("\n\n", text)
        return Block(
            kind=kind,
            text=text,
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
        figure_type = raw.get("figure_type")
        code_text = raw.get("code_text")
        is_code = (
            str(figure_type or "").lower() == "code"
            and isinstance(code_text, str)
            and bool(code_text)
        )
        return Block(
            kind="code" if is_code else "figure",
            raw=raw,
            page=page,
            section_id=raw.get("section_id", ""),
            section_path=tuple(raw.get("section_path", []) or []),
            figure_type=figure_type,
            figure_id=str(raw.get("figure_id") or ""),
            description=raw.get("figure_description"),
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


def build_semantic_units(
    doc: ParsedDocument,
    code_sidecar: dict | None = None,
    figure_descriptions: dict | None = None,
) -> list[SemanticUnit]:
    """Build semantic units.

    ``code_sidecar`` promotes recovered code snapshots; ``figure_descriptions`` attaches
    image descriptions and marks duplicate snapshots so they cannot produce chunks.
    Both are applied in memory only.
    """
    # Description alignment also marks duplicate snapshot bboxes, so it always runs:
    # assign ids against the original snapshot order before a code duplicate is removed.
    from .figure_descriptions import apply_figure_descriptions

    doc, _ = apply_figure_descriptions(doc, figure_descriptions or {})
    if code_sidecar:
        from .code_sidecar import apply_code_sidecar

        # Code promotion wins over any description's diagram_type.
        doc, _ = apply_code_sidecar(doc, code_sidecar)
    flat: list[Block] = []
    for page in doc.pages:
        for blk in page.get("blocks", []):
            if blk.get("skip_block"):
                continue
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

    def snapshot_unit(block: Block, unit_type: str) -> SemanticUnit:
        block_page_start, block_page_end, block_source_pages = _unit_pages([block])
        raw_title = str(block.raw.get("title") or block.raw.get("caption") or "").strip()
        local = dict(common)
        local.update(
            title=raw_title or (block.section_path[-1] if block.section_path else title),
            section_path=block.section_path or section_path,
            page_start=block_page_start,
            page_end=block_page_end,
            source_pages=block_source_pages or [block.page],
        )
        return SemanticUnit(
            unit_type=unit_type,
            blocks=[block],
            figure_type=block.figure_type,
            **local,
        )

    # TOC
    if heading and _TOC_TITLE.match("".join((heading.text or "").split())):
        return [SemanticUnit(unit_type="toc", blocks=blocks, **common)]

    # Recovered code is inline text in the register package. Non-code figures
    # remain independently retrievable and linked to the register package.
    if _is_register_section(blocks):
        register_blocks = [block for block in blocks if block.kind != "figure"]
        table = next(
            (b.raw for b in register_blocks if b.kind == "table" and _is_bit_table(b.raw)),
            None,
        )
        register_unit = SemanticUnit(
            unit_type="register",
            blocks=register_blocks,
            atomic=True,
            register_name=_register_name(table) if table else None,
            table_raw=table,
            **common,
        )
        side_units = [
            snapshot_unit(block, "figure")
            for block in blocks
            if block.kind == "figure"
        ]
        return [register_unit, *side_units]

    units: list[SemanticUnit] = []
    text_run: list[Block] = []
    figure_units: list[SemanticUnit] = []
    table_group: list[SemanticUnit] = []
    note_run = False

    def flush_text() -> None:
        if text_run:
            units.append(SemanticUnit(unit_type="text", blocks=list(text_run), **common))
            text_run.clear()

    for block in blocks:
        if block.kind == "heading":
            continue  # section title already captured
        if block.kind in ("paragraph", "code"):
            is_note = block.kind == "paragraph" and (
                _NOTE_START.match(block.text or "")
                or (note_run and _NOTE_ITEM.match(block.text or ""))
            )
            if table_group and is_note:
                # A trailing legend applies to a consecutive table group. Every
                # row-group chunk later repeats it along with its own table header.
                for table_unit in table_group:
                    table_unit.blocks.append(block)
                    start, end, pages = _unit_pages(table_unit.blocks)
                    table_unit.page_start, table_unit.page_end = start, end
                    table_unit.source_pages = pages
                note_run = True
                continue
            table_group = []
            note_run = False
            text_run.append(block)
        elif block.kind == "table":
            if note_run:
                # A footnote closes its table group. The next table's footnote
                # must not be copied back into an already qualified table.
                table_group = []
            # Move the short contiguous introduction into the table it qualifies.
            # Stop at code/images or a long paragraph; never cross a section.
            prefix: list[Block] = []
            context_tokens = 0
            while text_run and text_run[-1].kind == "paragraph":
                candidate = text_run[-1]
                tokens = estimate_tokens(candidate.text or "")
                if context_tokens + tokens > 150:
                    break
                prefix.insert(0, text_run.pop())
                context_tokens += tokens
            flush_text()
            if not prefix and table_group:
                first_blocks = table_group[0].blocks
                prefix = first_blocks[:next(
                    i for i, item in enumerate(first_blocks) if item.kind == "table"
                )]
            else:
                table_group = []
            table_blocks = [*prefix, block]
            local = dict(common)
            start, end, pages = _unit_pages(table_blocks)
            local.update(page_start=start, page_end=end, source_pages=pages)
            table_unit = SemanticUnit(
                unit_type="table", blocks=table_blocks, table_raw=block.raw, **local
            )
            units.append(table_unit)
            table_group.append(table_unit)
            note_run = False
        elif block.kind == "figure":
            table_group = []
            note_run = False
            # Figures no longer break the text run: the flow keeps A -> figure -> B and
            # the figure also becomes its own retrieval unit.  Excluded figures (e.g. a
            # vendor logo) are left out of the text flow and only kept as an excluded unit.
            if block.raw.get("figure_excluded"):
                figure_units.append(snapshot_unit(block, "figure"))
                continue
            text_run.append(block)
            figure_units.append(snapshot_unit(block, "figure"))
    flush_text()
    units.extend(figure_units)
    return units
