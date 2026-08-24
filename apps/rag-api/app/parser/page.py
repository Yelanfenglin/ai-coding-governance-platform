"""Anchor -> PageIR：唯一 ownership、表格恢复与保守的阅读顺序修复。"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .anchor import Anchor
from .models import BBox
from .native_pdf import NativePage
from .postprocess import normalize_text_block
from .table import Table, reconstruct_table


@dataclass
class TextBlock:
    kind: str  # paragraph | heading
    text: str
    bbox: BBox
    pp_order: int | None = None
    pp_label: str = "text"
    source: str = "pp"
    font_size: float = 0.0


@dataclass
class TableBlock:
    table: Table
    bbox: BBox
    source: str = "pp"
    pp_order: int | None = None


@dataclass
class SnapshotBlock:
    """无法可靠重建的表或视觉区域：保留截图、hint 与图内原生文字。"""

    hint: str
    bbox: BBox
    annotations: str = ""
    pp_order: int | None = None
    source: str = "pp"


Block = TextBlock | TableBlock | SnapshotBlock


@dataclass
class PageIR:
    page_number: int
    width: float
    height: float
    blocks: list[Block] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    body_font_size: float = 0.0


def _anchor_text(anchor: Anchor) -> str:
    """保留原生物理行边界；只清理用于定位的首尾空白。"""
    lines = sorted(anchor.lines, key=lambda line: (line.bbox.y0, line.bbox.x0))
    return "\n".join(line.text.strip() for line in lines if line.text.strip()).strip()


def build_blocks(
    page_number: int,
    native: NativePage,
    anchors: list[Anchor],
) -> PageIR:
    visible_sizes = [line.size for line in native.lines if line.text.strip() and line.size > 0]
    page = PageIR(
        page_number=page_number,
        width=native.width,
        height=native.height,
        body_font_size=statistics.median(visible_sizes) if visible_sizes else 0.0,
    )
    blocks: list[Block] = []
    table_success = 0
    table_recovered = 0
    table_fail = 0
    table_merge_repairs = 0
    table_topology_upgrades = 0
    visual_snapshots = 0

    for anchor in anchors:
        if anchor.kind == "NOISE":
            continue
        if anchor.kind in ("TEXT", "HEADING"):
            text = _anchor_text(anchor)
            if not text:
                continue
            blocks.append(
                TextBlock(
                    kind="heading" if anchor.kind == "HEADING" else "paragraph",
                    text=text,
                    bbox=anchor.bbox,
                    pp_order=anchor.pp_order,
                    pp_label=anchor.pp_label,
                    source=anchor.source,
                    font_size=max((line.size for line in anchor.lines), default=0.0),
                )
            )
            continue
        if anchor.kind == "TABLE":
            table = (
                anchor.table
                if isinstance(anchor.table, Table)
                else reconstruct_table(anchor.bbox, native)
            )
            if table is not None and table.valid:
                blocks.append(
                    TableBlock(
                        table=table,
                        bbox=table.bbox or anchor.bbox,
                        source=anchor.source,
                        pp_order=anchor.pp_order,
                    )
                )
                table_success += 1
                table_merge_repairs += table.split_merges
                if anchor.source == "native_table":
                    table_recovered += 1
                if anchor.table_upgraded:
                    table_topology_upgrades += 1
            else:
                blocks.append(
                    SnapshotBlock(
                        hint="table",
                        bbox=anchor.bbox,
                        annotations=_anchor_text(anchor),
                        pp_order=anchor.pp_order,
                        source=anchor.source,
                    )
                )
                table_fail += 1
            continue
        if anchor.kind == "VISUAL":
            blocks.append(
                SnapshotBlock(
                    hint=anchor.pp_label or "image",
                    bbox=anchor.bbox,
                    annotations=_anchor_text(anchor),
                    pp_order=anchor.pp_order,
                    source=anchor.source,
                )
            )
            visual_snapshots += 1

    split = _detect_two_column_split(blocks, native.width, native.height)
    if split is None:
        page.blocks = _postprocess_text_blocks(_single_column_order(blocks))
        order_mode = "pp_order"
    else:
        page.blocks = _postprocess_text_blocks(_two_column_order(blocks, split, native.width))
        order_mode = "two_column_repair"

    page.stats = {
        "tables_rebuilt": table_success,
        "tables_recovered": table_recovered,
        "tables_snapshot": table_fail,
        "table_merge_repairs": table_merge_repairs,
        "table_topology_upgrades": table_topology_upgrades,
        "visual_snapshots": visual_snapshots,
        "reading_order": order_mode,
    }
    return page


def prepare_table_anchors(native: NativePage, anchors: list[Anchor]) -> None:
    """Resolve PP table anchors before text ownership and visual refinement.

    The frozen classifier remains the first decision.  Only a failed PP table
    is retried with enhanced native separator topology; the same validity rule
    (repeated rows and columns) must still pass.  This upgrades vector tables
    while diagrams with many rectangles remain snapshots.
    """
    for anchor in anchors:
        if anchor.kind != "TABLE" or isinstance(anchor.table, Table):
            continue
        baseline = reconstruct_table(anchor.bbox, native)
        if baseline is not None and baseline.valid:
            anchor.table = baseline
            continue
        enhanced = reconstruct_table(
            anchor.bbox,
            native,
            validate_with_topology=True,
        )
        if enhanced is not None and enhanced.valid:
            anchor.table = enhanced
            anchor.table_upgraded = True
        else:
            anchor.table = baseline


def _postprocess_text_blocks(blocks: list[Block]) -> list[Block]:
    for block in blocks:
        if not isinstance(block, TextBlock):
            continue
        block.kind, block.text = normalize_text_block(block.kind, block.text)
    return blocks


def _single_column_order(blocks: list[Block]) -> list[Block]:
    """正文保留 PP order；无 order 的表/图/自动文本按 y 插入。"""
    ordered = sorted(
        (block for block in blocks if block.pp_order is not None),
        key=lambda block: (block.pp_order, block.bbox.y0),
    )
    unordered = sorted(
        (block for block in blocks if block.pp_order is None),
        key=lambda block: (block.bbox.y0, block.bbox.x0),
    )
    if not ordered:
        return unordered
    result: list[Block] = []
    ui = 0
    for block in ordered:
        while ui < len(unordered) and unordered[ui].bbox.y0 < block.bbox.y0:
            result.append(unordered[ui])
            ui += 1
        result.append(block)
    result.extend(unordered[ui:])
    return result


def _detect_two_column_split(
    blocks: list[Block], page_width: float, page_height: float
) -> float | None:
    """只在证据很强时修复 PP order 的双栏局部错序。

    该检测不重新分块，只检查已有窄文本 anchor 是否形成两个数量充足、
    横向明显分离且纵向重叠的簇。证据不足时完全退回 PP order。
    """
    candidates = [
        block
        for block in blocks
        if isinstance(block, TextBlock)
        and block.bbox.w <= page_width * 0.48
        and page_height * 0.08 <= block.bbox.cy <= page_height * 0.92
    ]
    if len(candidates) < 8:
        return None
    by_x = sorted(candidates, key=lambda block: block.bbox.cx)
    gaps = [by_x[i + 1].bbox.cx - by_x[i].bbox.cx for i in range(len(by_x) - 1)]
    if not gaps:
        return None
    gap_index = max(range(len(gaps)), key=gaps.__getitem__)
    if gaps[gap_index] < page_width * 0.15:
        return None
    left, right = by_x[: gap_index + 1], by_x[gap_index + 1 :]
    if len(left) < 4 or len(right) < 4:
        return None
    if max(block.bbox.cx for block in left) >= page_width * 0.52:
        return None
    if min(block.bbox.cx for block in right) <= page_width * 0.48:
        return None
    overlap_y = min(max(b.bbox.y1 for b in left), max(b.bbox.y1 for b in right)) - max(
        min(b.bbox.y0 for b in left), min(b.bbox.y0 for b in right)
    )
    if overlap_y < page_height * 0.15:
        return None
    return (by_x[gap_index].bbox.cx + by_x[gap_index + 1].bbox.cx) / 2.0


def _two_column_order(blocks: list[Block], split: float, page_width: float) -> list[Block]:
    """跨栏块作为 band 分隔；每个 band 内按左栏到底、再右栏到底。"""
    separators = sorted(
        (
            block
            for block in blocks
            if block.bbox.w >= page_width * 0.60
            or (
                not isinstance(block, TextBlock)
                and block.bbox.w >= page_width * 0.45
                and block.bbox.x0 < split - 8.0
                and block.bbox.x1 > split + 8.0
            )
        ),
        key=lambda block: (block.bbox.y0, block.bbox.x0),
    )
    remaining = [block for block in blocks if all(block is not sep for sep in separators)]
    result: list[Block] = []
    for separator in separators:
        before = [block for block in remaining if block.bbox.y0 < separator.bbox.y0]
        result.extend(_column_major(before, split))
        before_ids = {id(block) for block in before}
        remaining = [block for block in remaining if id(block) not in before_ids]
        result.append(separator)
    result.extend(_column_major(remaining, split))
    return result


def _column_major(blocks: list[Block], split: float) -> list[Block]:
    left = sorted(
        (block for block in blocks if block.bbox.cx < split),
        key=lambda block: (block.bbox.y0, block.bbox.x0, block.pp_order or 10**9),
    )
    right = sorted(
        (block for block in blocks if block.bbox.cx >= split),
        key=lambda block: (block.bbox.y0, block.bbox.x0, block.pp_order or 10**9),
    )
    return left + right
