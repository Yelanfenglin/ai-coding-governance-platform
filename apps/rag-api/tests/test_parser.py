from __future__ import annotations

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import fitz
import pytest

from app.parser.anchor import Anchor
from app.parser.cross_page import stitch_cross_page_tables
from app.parser.document import SectionTracker, link_table_continuations
from app.parser.models import BBox, NativeDrawing, VectorRule
from app.parser.native_pdf import NativeChar, NativeLine, NativePage, NativeSpan, _extract_rules
from app.parser.native_tables import recover_missing_tables
from app.parser.page import (
    PageIR,
    TableBlock,
    TextBlock,
    _detect_two_column_split,
    _two_column_order,
    prepare_table_anchors,
)
from app.parser.parse import _run
from app.parser.postprocess import (
    markdown_heading_level,
    normalize_text_block,
    normalize_toc_text,
    recover_document_headings,
    reflow_text,
)
from app.parser.renderer import _render_block
from app.parser.table import (
    Cell,
    Table,
    _horiz_cover,
    _vert_cover,
    connected_rule_components,
    reconstruct_table,
)
from app.parser.visual import add_recovered_visual_anchors, refine_snapshot_anchors


def _h(y: float, x0: float = 0.0, x1: float = 90.0) -> VectorRule:
    return VectorRule(BBox(x0, y - 0.3, x1, y + 0.3), horizontal=True)


def _v(x: float, y0: float = 0.0, y1: float = 60.0) -> VectorRule:
    return VectorRule(BBox(x - 0.3, y0, x + 0.3, y1), horizontal=False)


def _native_line(text: str, bbox: BBox, line_id: int) -> NativeLine:
    span = NativeSpan(
        text=text,
        bbox=bbox,
        size=8.0,
        block_id=0,
        line_id=line_id,
    )
    return NativeLine(
        spans=[span],
        block_id=0,
        line_id=line_id,
        bbox=bbox,
    )


def test_thin_filled_rectangles_become_rules_without_aspect_ratio_special_case() -> None:
    page = SimpleNamespace(
        get_drawings=lambda: [
            {
                "items": [
                    ("re", fitz.Rect(0.0, 10.0, 18.0, 12.2), 1),
                    ("re", fitz.Rect(30.0, 0.0, 32.2, 18.0), 1),
                    ("re", fitz.Rect(40.0, 0.0, 42.2, 2.2), 1),
                    ("re", fitz.Rect(0.0, 30.0, 20.0, 36.0), 1),
                ]
            }
        ]
    )

    rules = _extract_rules(page)

    assert len(rules) == 2
    assert [rule.horizontal for rule in rules] == [True, False]
    assert [rule.thickness for rule in rules] == pytest.approx([2.2, 2.2])


def test_separator_coverage_uses_union_of_collinear_fragments() -> None:
    horizontal = [_h(20.0, 0.0, 28.0), _h(20.0, 30.0, 60.0)]
    vertical = [_v(30.0, 0.0, 18.0), _v(30.0, 20.0, 40.0)]

    assert _horiz_cover(horizontal, 20.0, 0.0, 60.0)
    assert _vert_cover(vertical, 30.0, 0.0, 40.0)
    assert not _horiz_cover([_h(20.0, 0.0, 25.0), _h(20.0, 31.0, 60.0)], 20.0, 0.0, 60.0)


def test_enhanced_topology_does_not_upgrade_a_frozen_snapshot() -> None:
    rules = [_h(y) for y in (0.0, 20.0, 40.0, 60.0)] + [_v(x) for x in (0.0, 30.0, 60.0, 90.0)]
    native = _native_grid_page(
        rules,
        [
            ["r0c0", "r0c1", "r0c2"],
            ["r1c0", "r1c1", "r1c2"],
            ["r2c0", "r2c1", "r2c2"],
        ],
    )
    native.classification_rules = [_h(0.0), _h(60.0), _v(0.0), _v(90.0)]

    table = reconstruct_table(BBox(0.0, 0.0, 90.0, 60.0), native)

    assert table is not None
    assert (table.n_rows, table.n_cols) == (3, 3)
    assert not table.valid


def test_failed_pp_table_can_be_upgraded_by_enhanced_topology() -> None:
    rules = [_h(y, 0.0, 60.0) for y in (0.0, 20.0, 40.0, 60.0)] + [_v(x) for x in (0.0, 30.0, 60.0)]
    native = _native_grid_page(
        rules,
        [["key", "value"], ["a", "1"], ["b", "2"]],
    )
    native.classification_rules = [
        _h(0.0, 0.0, 60.0),
        _h(60.0, 0.0, 60.0),
        _v(0.0),
        _v(60.0),
    ]
    anchor = Anchor(
        anchor_id=0,
        bbox=BBox(0.0, 0.0, 60.0, 60.0),
        pp_label="table",
        kind="TABLE",
    )

    prepare_table_anchors(native, [anchor])

    assert anchor.table is not None
    assert anchor.table.valid
    assert anchor.table_upgraded
    assert anchor.table.text_rows() == [["key", "value"], ["a", "1"], ["b", "2"]]


def test_snapshot_bbox_expands_to_include_same_band_labels() -> None:
    left = _native_line("left label", BBox(5.0, 30.0, 35.0, 40.0), line_id=0)
    right = _native_line("right label", BBox(65.0, 30.0, 98.0, 40.0), line_id=1)
    top = _native_line("short title", BBox(40.0, 14.0, 60.0, 19.0), line_id=2)
    native = NativePage(
        page_number=1,
        width=100.0,
        height=100.0,
        lines=[left, right, top],
        drawings=[NativeDrawing(BBox(35.0, 20.0, 65.0, 60.0), item_count=1)],
    )
    anchor = Anchor(
        anchor_id=0,
        bbox=BBox(35.0, 22.0, 65.0, 58.0),
        pp_label="image",
        kind="VISUAL",
    )

    refine_snapshot_anchors(native, [anchor])

    assert anchor.bbox.x0 <= 1.0
    assert anchor.bbox.x1 == 100.0
    assert anchor.bbox.y0 <= 11.0
    assert anchor.bbox.y1 >= 63.0


def test_snapshot_expansion_preserves_claimed_context_but_keeps_side_labels() -> None:
    title = _native_line("body sentence", BBox(35.0, 14.0, 65.0, 19.0), line_id=0)
    side = _native_line("pin label", BBox(5.0, 30.0, 35.0, 40.0), line_id=1)
    native = NativePage(
        page_number=1,
        width=100.0,
        height=100.0,
        lines=[title, side],
        drawings=[NativeDrawing(BBox(35.0, 20.0, 65.0, 60.0), item_count=1)],
    )
    text_anchor = Anchor(
        anchor_id=0,
        bbox=title.bbox,
        pp_label="text",
        kind="TEXT",
    )
    visual_anchor = Anchor(
        anchor_id=1,
        bbox=BBox(35.0, 22.0, 65.0, 58.0),
        pp_label="image",
        kind="VISUAL",
    )

    refine_snapshot_anchors(native, [text_anchor, visual_anchor])

    assert visual_anchor.bbox.x0 <= 1.0
    assert visual_anchor.bbox.y0 == 19.5


def test_snapshot_expansion_stops_at_neighbor_midline() -> None:
    native = NativePage(
        page_number=1,
        width=100.0,
        height=100.0,
        lines=[
            _native_line("left outer", BBox(2.0, 30.0, 18.0, 40.0), line_id=0),
            _native_line("left inner", BBox(40.0, 30.0, 48.0, 40.0), line_id=1),
            _native_line("right inner", BBox(52.0, 30.0, 60.0, 40.0), line_id=2),
            _native_line("right outer", BBox(82.0, 30.0, 98.0, 40.0), line_id=3),
        ],
        drawings=[
            NativeDrawing(BBox(20.0, 20.0, 40.0, 60.0), item_count=1),
            NativeDrawing(BBox(60.0, 20.0, 80.0, 60.0), item_count=1),
        ],
    )
    left = Anchor(
        anchor_id=0,
        bbox=BBox(20.0, 22.0, 40.0, 58.0),
        pp_label="chart",
        kind="VISUAL",
    )
    right = Anchor(
        anchor_id=1,
        bbox=BBox(60.0, 22.0, 80.0, 58.0),
        pp_label="chart",
        kind="VISUAL",
    )

    refine_snapshot_anchors(native, [left, right])

    assert left.bbox.x0 == 0.0
    assert left.bbox.x1 == 50.0
    assert right.bbox.x0 == 50.0
    assert right.bbox.x1 == 100.0


def test_unmodeled_vector_diagram_becomes_one_visual_anchor() -> None:
    native = NativePage(
        page_number=1,
        width=100.0,
        height=100.0,
        lines=[
            _native_line("left one", BBox(5.0, 25.0, 35.0, 35.0), line_id=0),
            _native_line("right one", BBox(65.0, 25.0, 95.0, 35.0), line_id=1),
            _native_line("left two", BBox(5.0, 45.0, 35.0, 55.0), line_id=2),
            _native_line("right two", BBox(65.0, 45.0, 95.0, 55.0), line_id=3),
        ],
        drawings=[
            NativeDrawing(BBox(35.0, 20.0, 65.0, 60.0), item_count=1),
            NativeDrawing(BBox(30.0, 24.0, 35.0, 34.0), item_count=1),
            NativeDrawing(BBox(65.0, 24.0, 70.0, 34.0), item_count=1),
            NativeDrawing(BBox(30.0, 44.0, 35.0, 54.0), item_count=1),
            NativeDrawing(BBox(65.0, 44.0, 70.0, 54.0), item_count=1),
        ],
    )

    anchors: list[Anchor] = []
    add_recovered_visual_anchors(native, anchors)

    assert len(anchors) == 1
    assert anchors[0].kind == "VISUAL"
    assert anchors[0].source == "native_visual"
    assert anchors[0].bbox.x0 <= 1.0
    assert anchors[0].bbox.x1 >= 99.0


def test_compact_missing_table_needs_matrix_and_nearby_context() -> None:
    rules = [_h(y, 0.0, 60.0) for y in (0.0, 20.0, 40.0, 60.0)] + [
        _v(x, 0.0, 60.0) for x in (0.0, 30.0, 60.0)
    ]
    native = _native_grid_page(
        rules,
        [["mode", "state"], ["0", "disabled"], ["1", "enabled"]],
    )
    # The frozen classifier misses the middle separator, while enhanced
    # topology still sees a complete 3x2 record matrix.
    native.classification_rules = [
        *[_h(y, 0.0, 60.0) for y in (0.0, 20.0, 40.0, 60.0)],
        _v(0.0, 0.0, 60.0),
        _v(60.0, 0.0, 60.0),
    ]

    without_context = recover_missing_tables(native, [], 100.0, 100.0)
    with_context = recover_missing_tables(
        native,
        [],
        100.0,
        100.0,
        context_bboxes=[BBox(0.0, -20.0, 60.0, -2.0)],
    )

    assert without_context == []
    assert len(with_context) == 1
    assert with_context[0].text_rows() == [
        ["mode", "state"],
        ["0", "disabled"],
        ["1", "enabled"],
    ]


def test_missing_table_recovery_does_not_require_cell_vocabulary() -> None:
    rules = [_h(y) for y in (0.0, 20.0, 40.0, 60.0)] + [_v(x) for x in (0.0, 30.0, 60.0, 90.0)]
    native = _native_grid_page(
        rules,
        [
            ["alpha", "beta", "gamma"],
            ["left", "middle", "right"],
            ["red", "green", "blue"],
        ],
    )

    without_context = recover_missing_tables(native, [], 100.0, 100.0)
    with_context = recover_missing_tables(
        native,
        [],
        100.0,
        100.0,
        context_bboxes=[BBox(0.0, -20.0, 90.0, -2.0)],
    )

    assert without_context == []
    assert len(with_context) == 1
    assert with_context[0].text_rows()[0] == ["alpha", "beta", "gamma"]


def test_table_uses_roi_only_for_missing_outer_vertical_edges() -> None:
    rules = [_h(y) for y in (0.0, 20.0, 40.0, 60.0)] + [_v(x) for x in (30.0, 60.0)]
    lines = []
    for row in range(3):
        for col in range(3):
            x0, y0 = col * 30.0 + 6.0, row * 20.0 + 5.0
            span = NativeSpan(
                text=f"r{row}c{col}",
                bbox=BBox(x0, y0, x0 + 12.0, y0 + 8.0),
                size=8.0,
                block_id=row,
                line_id=col,
            )
            lines.append(
                NativeLine(
                    spans=[span],
                    block_id=row,
                    line_id=col,
                    bbox=span.bbox,
                )
            )
    native = NativePage(page_number=1, width=100.0, height=80.0, lines=lines, rules=rules)
    table = reconstruct_table(BBox(0.0, 0.0, 90.0, 60.0), native)
    assert table is not None and table.valid
    assert (table.n_rows, table.n_cols) == (3, 3)
    assert table.text_rows()[0] == ["r0c0", "r0c1", "r0c2"]


def test_one_native_span_is_distributed_across_distinct_cells_by_character_position() -> None:
    rules = [_h(y) for y in (0.0, 20.0, 40.0, 60.0)] + [_v(x) for x in (0.0, 30.0, 60.0, 90.0)]
    native = _native_grid_page(
        rules,
        [["", "", ""], ["a", "1", "x"], ["b", "2", "y"]],
    )
    characters = [
        NativeChar("L", BBox(5.0, 5.0, 9.0, 13.0)),
        NativeChar("0", BBox(9.0, 5.0, 13.0, 13.0)),
        NativeChar("L", BBox(35.0, 5.0, 39.0, 13.0)),
        NativeChar("1", BBox(39.0, 5.0, 43.0, 13.0)),
        NativeChar("L", BBox(65.0, 5.0, 69.0, 13.0)),
        NativeChar("2", BBox(69.0, 5.0, 73.0, 13.0)),
    ]
    span = NativeSpan(
        text="L0 L1 L2",
        bbox=BBox(5.0, 5.0, 73.0, 13.0),
        size=8.0,
        block_id=1,
        line_id=100,
        characters=characters,
    )
    native.lines.append(NativeLine(spans=[span], block_id=1, line_id=100, bbox=span.bbox))

    table = reconstruct_table(BBox(0.0, 0.0, 90.0, 60.0), native)

    assert table is not None and table.valid
    assert table.text_rows()[0] == ["L0", "L1", "L2"]
    assert [line.text for cell in table.rows[0] for line in cell.physical_lines] == [
        "L0",
        "L1",
        "L2",
    ]


def test_character_positions_do_not_split_a_true_colspan() -> None:
    rules = [_h(y) for y in (0.0, 20.0, 40.0, 60.0)] + [
        _v(0.0),
        _v(30.0, 20.0, 60.0),
        _v(60.0, 20.0, 60.0),
        _v(90.0),
    ]
    native = _native_grid_page(
        rules,
        [["", "", ""], ["a", "1", "x"], ["b", "2", "y"]],
    )
    characters = [
        NativeChar("W", BBox(5.0, 5.0, 10.0, 13.0)),
        NativeChar("I", BBox(35.0, 5.0, 40.0, 13.0)),
        NativeChar("D", BBox(65.0, 5.0, 70.0, 13.0)),
        NativeChar("E", BBox(70.0, 5.0, 75.0, 13.0)),
    ]
    span = NativeSpan(
        text="WIDE",
        bbox=BBox(5.0, 5.0, 75.0, 13.0),
        size=8.0,
        block_id=1,
        line_id=100,
        characters=characters,
    )
    native.lines.append(NativeLine(spans=[span], block_id=1, line_id=100, bbox=span.bbox))

    table = reconstruct_table(BBox(0.0, 0.0, 90.0, 60.0), native)

    assert table is not None and table.valid
    assert table.rows[0][0].colspan == 3
    assert table.text_rows()[0] == ["WIDE", "", ""]
    assert [line.text for line in table.rows[0][0].physical_lines] == ["WIDE"]


def test_rowspan_is_split_when_each_atomic_row_has_its_own_text() -> None:
    rules = [_h(0.0), _h(60.0), _h(20.0, 30.0), _h(40.0, 30.0)] + [
        _v(x) for x in (0.0, 30.0, 60.0, 90.0)
    ]
    native = _native_grid_page(
        rules,
        [
            ["BANK0", "0x000", "RAM"],
            ["BANK1", "0x100", "GPR"],
            ["BANK2", "0x200", "-"],
        ],
    )

    table = reconstruct_table(BBox(0.0, 0.0, 90.0, 60.0), native)

    assert table is not None and table.valid
    assert table.split_merges == 1
    assert table.text_rows() == [
        ["BANK0", "0x000", "RAM"],
        ["BANK1", "0x100", "GPR"],
        ["BANK2", "0x200", "-"],
    ]
    assert [table.rows[row][0].rowspan for row in range(3)] == [1, 1, 1]


def test_rowspan_is_kept_when_text_has_one_anchor() -> None:
    rules = [_h(0.0), _h(60.0), _h(20.0, 30.0), _h(40.0, 30.0)] + [
        _v(x) for x in (0.0, 30.0, 60.0, 90.0)
    ]
    native = _native_grid_page(
        rules,
        [
            ["GROUP", "0x000", "RAM"],
            ["", "0x001", "RAM"],
            ["", "0x002", "RAM"],
        ],
    )

    table = reconstruct_table(BBox(0.0, 0.0, 90.0, 60.0), native)

    assert table is not None and table.valid
    assert table.split_merges == 0
    assert table.rows[0][0].text == "GROUP"
    assert table.rows[0][0].rowspan == 3
    assert table.rows[1][0] is None


def test_two_line_text_remains_one_ambiguous_rowspan() -> None:
    rules = [_h(0.0), _h(40.0), _h(20.0, 30.0)] + [
        _v(x, 0.0, 40.0) for x in (0.0, 30.0, 60.0, 90.0)
    ]
    native = _native_grid_page(
        rules,
        [
            ["START MODE", "low", "0"],
            ["SELECT", "high", "1"],
        ],
    )

    table = reconstruct_table(BBox(0.0, 0.0, 90.0, 40.0), native)

    assert table is not None and table.valid
    assert table.split_merges == 0
    assert table.rows[0][0].rowspan == 2
    assert table.rows[0][0].text == "START MODE SELECT"


def test_sparse_vertical_title_does_not_trigger_rowspan_repair() -> None:
    rules = (
        [_h(0.0), _h(120.0)]
        + [_h(y, 30.0) for y in (20.0, 40.0, 60.0, 80.0, 100.0)]
        + [_v(x, 0.0, 120.0) for x in (0.0, 30.0, 60.0, 90.0)]
    )
    native = _native_grid_page(
        rules,
        [
            ["寄 存", "a0", "b0"],
            ["", "a1", "b1"],
            ["", "a2", "b2"],
            ["器 指 令", "a3", "b3"],
            ["", "a4", "b4"],
            ["", "a5", "b5"],
        ],
    )

    table = reconstruct_table(BBox(0.0, 0.0, 90.0, 120.0), native)

    assert table is not None and table.valid
    assert table.split_merges == 0
    assert table.rows[0][0].rowspan == 6
    assert table.rows[0][0].text == "寄 存 器 指 令"


@pytest.mark.parametrize(
    ("header", "expected_colspans", "expected_text"),
    [
        (["最小", "正常", "最大"], [1, 1, 1], ["最小", "正常", "最大"]),
        (["", "单位（mm）", ""], [3], ["单位（mm）", "", ""]),
    ],
)
def test_colspan_requires_one_text_anchor(
    header: list[str], expected_colspans: list[int], expected_text: list[str]
) -> None:
    rules = [_h(y) for y in (0.0, 20.0, 40.0, 60.0)] + [
        _v(0.0),
        _v(30.0, 20.0, 60.0),
        _v(60.0, 20.0, 60.0),
        _v(90.0),
    ]
    native = _native_grid_page(
        rules,
        [header, ["1", "2", "3"], ["4", "5", "6"]],
    )

    table = reconstruct_table(BBox(0.0, 0.0, 90.0, 60.0), native)

    assert table is not None and table.valid
    assert [cell.colspan for cell in table.rows[0] if cell is not None] == expected_colspans
    assert table.text_rows()[0] == expected_text


def test_repeated_aligned_cell_lines_become_canonical_subrows() -> None:
    native, bbox = _native_aligned_cell_page(body_rows=2, line_count=3)

    table = reconstruct_table(bbox, native)

    assert table is not None and table.valid
    assert table.text_rows() == [
        ["h0", "h1", "h2"],
        ["r0c0s0 r0c0s1 r0c0s2", "r0c1s0 r0c1s1 r0c1s2", "r0c2s0 r0c2s1 r0c2s2"],
        ["r1c0s0 r1c0s1 r1c0s2", "r1c1s0 r1c1s1 r1c1s2", "r1c2s0 r1c2s1 r1c2s2"],
    ]
    assert table.aligned_parent_rows == [1, 2]
    assert table.semantic_text_rows() == [
        ["h0", "h1", "h2"],
        ["r0c0s0", "r0c1s0", "r0c2s0"],
        ["r0c0s1", "r0c1s1", "r0c2s1"],
        ["r0c0s2", "r0c1s2", "r0c2s2"],
        ["r1c0s0", "r1c1s0", "r1c2s0"],
        ["r1c0s1", "r1c1s1", "r1c2s1"],
        ["r1c0s2", "r1c1s2", "r1c2s2"],
    ]
    rendered = _render_block(TableBlock(table=table, bbox=bbox))
    assert "| r0c0s1 | r0c1s1 | r0c2s1 |" in rendered
    assert "r0c0s0 r0c0s1" not in rendered


def test_three_repeated_two_line_rows_are_enough_context() -> None:
    native, bbox = _native_aligned_cell_page(body_rows=3, line_count=2)

    table = reconstruct_table(bbox, native)

    assert table is not None and table.valid
    assert table.aligned_parent_rows == [1, 2, 3]
    assert len(table.aligned_subrows) == 6


def test_aligned_visual_wraps_remain_outer_cells_when_lines_fill_columns() -> None:
    native, bbox = _native_aligned_cell_page(
        body_rows=2,
        line_count=3,
        line_width_ratio=0.93,
    )

    table = reconstruct_table(bbox, native)

    assert table is not None and table.valid
    assert table.aligned_subrows == []
    assert len(table.semantic_text_rows()) == 3


def _native_aligned_cell_page(
    *,
    body_rows: int,
    line_count: int,
    line_width_ratio: float = 0.6,
) -> tuple[NativePage, BBox]:
    column_width = 30.0
    table_width = 90.0
    header_height = 20.0
    line_step = 15.0
    body_row_height = line_count * line_step + 8.0
    table_height = header_height + body_rows * body_row_height
    row_edges = [header_height + row * body_row_height for row in range(body_rows + 1)]
    rules = [_h(0.0), *[_h(edge) for edge in row_edges]] + [
        _v(x, 0.0, table_height) for x in (0.0, 30.0, 60.0, 90.0)
    ]
    lines: list[NativeLine] = []
    line_id = 0
    for col in range(3):
        bbox = BBox(col * column_width + 3.0, 5.0, col * column_width + 13.0, 13.0)
        lines.append(_native_line(f"h{col}", bbox, line_id))
        line_id += 1
    for body_row in range(body_rows):
        row_top = header_height + body_row * body_row_height
        for col in range(3):
            for subrow in range(line_count):
                x0 = col * column_width + 2.0
                y0 = row_top + 3.0 + subrow * line_step
                bbox = BBox(
                    x0,
                    y0,
                    x0 + column_width * line_width_ratio,
                    y0 + 8.0,
                )
                lines.append(
                    _native_line(
                        f"r{body_row}c{col}s{subrow}",
                        bbox,
                        line_id,
                    )
                )
                line_id += 1
    return (
        NativePage(
            page_number=1,
            width=100.0,
            height=table_height + 20.0,
            lines=lines,
            rules=rules,
        ),
        BBox(0.0, 0.0, table_width, table_height),
    )


def _native_grid_page(rules: list[VectorRule], text_rows: list[list[str]]) -> NativePage:
    lines: list[NativeLine] = []
    line_id = 0
    for row, values in enumerate(text_rows):
        for col, value in enumerate(values):
            if not value:
                continue
            x0, y0 = col * 30.0 + 4.0, row * 20.0 + 5.0
            span = NativeSpan(
                text=value,
                bbox=BBox(x0, y0, x0 + 20.0, y0 + 8.0),
                size=8.0,
                block_id=0,
                line_id=line_id,
            )
            lines.append(
                NativeLine(
                    spans=[span],
                    block_id=0,
                    line_id=line_id,
                    bbox=span.bbox,
                )
            )
            line_id += 1
    return NativePage(
        page_number=1,
        width=100.0,
        height=80.0,
        lines=lines,
        rules=rules,
    )


def test_disconnected_rule_grids_remain_separate_components() -> None:
    upper = [_h(y, 0.0, 60.0) for y in (0.0, 20.0, 40.0)] + [
        _v(x, 0.0, 40.0) for x in (0.0, 30.0, 60.0)
    ]
    lower = [_h(y, 0.0, 60.0) for y in (100.0, 120.0, 140.0)] + [
        _v(x, 100.0, 140.0) for x in (0.0, 30.0, 60.0)
    ]
    components = connected_rule_components(upper + lower)
    assert sorted(len(component) for component in components) == [6, 6]


def test_two_column_repair_is_column_major() -> None:
    blocks = [
        TextBlock("heading", "title", BBox(50, 40, 545, 70), 1),
        *[
            TextBlock(
                "paragraph",
                f"L{index}",
                BBox(60, 100 + index * 20, 220, 112 + index * 20),
                index + 2,
            )
            for index in range(4)
        ],
        *[
            TextBlock(
                "paragraph",
                f"R{index}",
                BBox(330, 100 + index * 20, 500, 112 + index * 20),
                index + 6,
            )
            for index in range(4)
        ],
    ]
    split = _detect_two_column_split(blocks, 595.0, 400.0)
    assert split is not None
    ordered = _two_column_order(blocks, split, 595.0)
    assert [block.text for block in ordered] == [
        "title",
        "L0",
        "L1",
        "L2",
        "L3",
        "R0",
        "R1",
        "R2",
        "R3",
    ]


def test_section_tracker_links_only_same_section_edge_tables() -> None:
    tracker = SectionTracker()
    first = tracker.annotate(
        {
            "page": 1,
            "height": 100.0,
            "blocks": [
                {"type": "text", "kind": "heading", "text": "2.1 参数"},
                {"type": "table", "bbox": [0, 40, 90, 95], "n_cols": 4, "block_index": 1},
            ],
        }
    )
    second = tracker.annotate(
        {
            "page": 2,
            "height": 100.0,
            "blocks": [{"type": "table", "bbox": [0, 5, 90, 50], "n_cols": 4, "block_index": 0}],
        }
    )
    link_table_continuations([first, second])
    assert first["blocks"][1]["continues_on_page"] == 2
    assert second["blocks"][0]["continued_from_page"] == 1


def test_section_heading_prevents_cross_section_table_link() -> None:
    tracker = SectionTracker()
    first = tracker.annotate(
        {
            "page": 1,
            "height": 100.0,
            "blocks": [
                {"type": "text", "kind": "heading", "text": "3.1 First"},
                {"type": "table", "bbox": [0, 40, 90, 95], "n_cols": 4, "block_index": 1},
            ],
        }
    )
    second = tracker.annotate(
        {
            "page": 2,
            "height": 100.0,
            "blocks": [
                {"type": "text", "kind": "heading", "text": "3.2 Second"},
                {"type": "table", "bbox": [0, 5, 90, 50], "n_cols": 4, "block_index": 1},
            ],
        }
    )

    link_table_continuations([first, second])

    assert "continues_on_page" not in first["blocks"][1]
    assert "continued_from_page" not in second["blocks"][1]


def test_section_tracker_drops_stale_parent_with_different_prefix() -> None:
    tracker = SectionTracker()
    tracker.annotate(
        {
            "page": 1,
            "blocks": [{"type": "text", "kind": "heading", "text": "10 通讯"}],
        }
    )
    page = tracker.annotate(
        {
            "page": 2,
            "blocks": [{"type": "text", "kind": "heading", "text": "11.3 工作模式"}],
        }
    )
    assert page["blocks"][0]["section_path"] == ["11.3 工作模式"]


def test_numbered_heading_recovery_builds_contextual_markdown_tree() -> None:
    blocks = [
        TextBlock(
            "heading",
            "1 产品简述",
            BBox(0, 0, 100, 30),
            pp_label="paragraph_title",
            font_size=24.0,
        ),
        TextBlock(
            "paragraph",
            "1.1 特性",
            BBox(0, 40, 100, 58),
            font_size=15.0,
        ),
        TextBlock(
            "paragraph",
            "1.1.1 深层标题",
            BBox(0, 65, 100, 83),
            font_size=15.0,
        ),
    ]
    page = PageIR(1, 595.0, 842.0, blocks=blocks, body_font_size=9.0)

    recover_document_headings([page])

    assert [block.kind for block in blocks] == ["heading"] * 3
    assert [markdown_heading_level(block.text, block.kind) for block in blocks] == [2, 3, 4]
    assert [_render_block(block) for block in blocks] == [
        "## 1 产品简述",
        "### 1.1 特性",
        "#### 1.1.1 深层标题",
    ]


def test_root_heading_uses_typography_and_number_sequence_context() -> None:
    page_one = PageIR(
        1,
        595.0,
        842.0,
        blocks=[
            TextBlock(
                "heading",
                "10 串口",
                BBox(0, 0, 100, 30),
                pp_label="paragraph_title",
                font_size=24.0,
            )
        ],
        body_font_size=9.0,
    )
    page_two = PageIR(
        2,
        595.0,
        842.0,
        blocks=[
            TextBlock("paragraph", "11 I2C", BBox(0, 0, 100, 30), font_size=24.0),
            TextBlock(
                "heading",
                "11.1 概述",
                BBox(0, 40, 100, 58),
                pp_label="paragraph_title",
                font_size=15.0,
            ),
        ],
        body_font_size=9.0,
    )

    recover_document_headings([page_one, page_two])

    assert page_two.blocks[0].kind == "heading"
    assert page_two.blocks[1].kind == "heading"


def test_short_heading_needs_extra_context_instead_of_a_hard_length_rule() -> None:
    blocks = [
        TextBlock(
            "heading",
            "1 序",
            BBox(0, 0, 100, 30),
            pp_label="paragraph_title",
            font_size=24.0,
        ),
        TextBlock(
            "heading",
            "1.1 背景",
            BBox(0, 40, 100, 58),
            pp_label="paragraph_title",
            font_size=15.0,
        ),
        TextBlock(
            "heading",
            "2 V",
            BBox(0, 70, 100, 83),
            pp_label="paragraph_title",
            font_size=9.0,
        ),
    ]
    page = PageIR(1, 595.0, 842.0, blocks=blocks, body_font_size=9.0)

    recover_document_headings([page])

    assert [block.kind for block in blocks] == ["heading", "heading", "paragraph"]


def test_numeric_values_and_mode_labels_do_not_enter_heading_tree() -> None:
    blocks = [
        TextBlock(
            "heading",
            "7 定时器0",
            BBox(0, 0, 100, 30),
            pp_label="paragraph_title",
            font_size=24.0,
        ),
        TextBlock("paragraph", "8 位模式", BBox(0, 40, 100, 53), font_size=10.5),
        TextBlock(
            "heading",
            "16 位模式",
            BBox(0, 60, 100, 73),
            pp_label="paragraph_title",
            font_size=10.5,
        ),
        TextBlock("paragraph", "0 = 关闭", BBox(0, 80, 100, 93), font_size=9.0),
        TextBlock(
            "heading",
            "8 定时器1",
            BBox(0, 100, 100, 130),
            pp_label="paragraph_title",
            font_size=24.0,
        ),
    ]
    page = PageIR(1, 595.0, 842.0, blocks=blocks, body_font_size=8.0)

    recover_document_headings([page])

    assert [block.kind for block in blocks] == [
        "heading",
        "paragraph",
        "paragraph",
        "paragraph",
        "heading",
    ]


def test_numbered_heading_recovery_rejects_toc_and_note_box() -> None:
    toc = "2.2.3 系统寄存器定义 ................................ 15"
    assert normalize_text_block("paragraph", toc) == ("paragraph", toc)
    assert normalize_text_block("heading", "注： 烧录接口说明") == (
        "paragraph",
        "注： 烧录接口说明",
    )
    assert normalize_text_block("paragraph", "1.35V") == ("paragraph", "1.35V")


def test_existing_heading_without_space_gets_numbered_level() -> None:
    kind, text = normalize_text_block("heading", "8.12.1功能描述")

    assert (kind, text) == ("heading", "8.12.1 功能描述")
    assert markdown_heading_level(text, kind) == 4


def test_paragraph_reflow_joins_visual_wraps_and_preserves_bullets() -> None:
    source = (
        "● 内部高速RC振荡器（32MHz，可用于系\n"
        "统时钟，定时器）\n"
        "● TC0：具有周期寄存器功能的定时/计数\n"
        "器\n"
        "- 16位模式下，周期寄存器用作低8位\n"
        "计数器"
    )

    assert reflow_text(source) == (
        "● 内部高速RC振荡器（32MHz，可用于系统时钟，定时器）\n"
        "● TC0：具有周期寄存器功能的定时/计数器\n"
        "- 16位模式下，周期寄存器用作低8位计数器"
    )


def test_toc_reflow_keeps_each_entry_on_its_own_line() -> None:
    # Root-level entries ("1 产品简述", "2 中央处理器") are not bullets and do
    # not end in sentence punctuation, so the ordinary reflow joins them onto
    # the previous line.  A TOC must keep one visual line per entry.
    source = (
        "应用注意事项 ......... 5\n"
        "1 产品简述 ........ 6\n"
        "1.1 特性 ........ 6\n"
        "1.3 引脚描述 ..... 8\n"
        "2 中央处理器（CPU） ....... 9\n"
        "2.1 程序存储器 ..... 9"
    )
    assert normalize_toc_text(source) == "\n".join(
        [
            "应用注意事项 ......... 5",
            "1 产品简述 ........ 6",
            "1.1 特性 ........ 6",
            "1.3 引脚描述 ..... 8",
            "2 中央处理器（CPU） ....... 9",
            "2.1 程序存储器 ..... 9",
        ]
    )


def test_toc_reflow_joins_only_a_wrapped_title_line() -> None:
    # A first half of a long title (no trailing page number) is joined to the
    # following line that carries the page number; a second title line that
    # already ends in a page number is never merged with the next entry.
    source = (
        "12.3 An Extremely Long Description Of Some\n"
        "Peripheral Register .......... 88\n"
        "12.4 Another Entry .......... 90"
    )
    assert normalize_toc_text(source) == "\n".join(
        [
            "12.3 An Extremely Long Description Of Some Peripheral Register .......... 88",
            "12.4 Another Entry .......... 90",
        ]
    )


def _plain_table(rows: list[list[str]], x0=10.0, x1=80.0, y0=20.0, y1=60.0) -> Table:
    grid = [
        [Cell(row=r, col=c, text=text, bbox=BBox(x0, y0, x1, y1)) for c, text in enumerate(row)]
        for r, row in enumerate(rows)
    ]
    return Table(
        rows=grid,
        n_rows=len(grid),
        n_cols=len(rows[0]),
        bbox=BBox(x0, y0, x1, y1),
        validated=True,
    )


def _page_with_table(page_number: int, table: Table, lead: list | None = None) -> PageIR:
    blocks = list(lead or []) + [TableBlock(table=table, bbox=table.bbox)]
    return PageIR(page_number, 595.0, 842.0, blocks=blocks, body_font_size=9.0)


def test_cross_page_stitch_merges_matching_table_across_pages() -> None:
    header = ["名称", "类型"]
    page_one = _page_with_table(1, _plain_table([header, ["IOA0", "I/O"], ["IOA1", "I/O"]]))
    page_two = _page_with_table(2, _plain_table([header, ["IOB0", "I/O"], ["IOB1", "I/O"]]))
    stitch_cross_page_tables([page_one, page_two])
    assert page_one.blocks[0].source_pages == [1, 2]
    assert page_one.blocks[0].table.n_rows == 5  # header + 4 data rows
    assert page_two.blocks == []  # fully absorbed
    assert page_one.blocks[0].table.text_rows() == [
        ["名称", "类型"],
        ["IOA0", "I/O"],
        ["IOA1", "I/O"],
        ["IOB0", "I/O"],
        ["IOB1", "I/O"],
    ]


def test_cross_page_stitch_chains_across_three_pages() -> None:
    header = ["符号", "说明"]
    pages = [
        _page_with_table(
            11,
            _plain_table([header, ["A", "x"]]),
            lead=[TextBlock("paragraph", "表1", BBox(0, 0, 100, 14))],
        ),
        _page_with_table(12, _plain_table([header, ["B", "y"]])),
        _page_with_table(13, _plain_table([header, ["C", "z"]])),
    ]
    stitch_cross_page_tables(pages)
    merged = pages[0].blocks[-1]
    assert merged.source_pages == [11, 12, 13]
    assert merged.table.n_rows == 4  # header + A/B/C
    assert [row[0] for row in merged.table.text_rows()] == ["符号", "A", "B", "C"]
    assert pages[1].blocks == []
    assert pages[2].blocks == []


def test_cross_page_stitch_refuses_when_table_is_not_first_body_block() -> None:
    # A heading before page 2's table means it is a fresh table, not a continuation.
    header = ["名称", "类型"]
    page_one = _page_with_table(1, _plain_table([header, ["IOA0", "I/O"]]))
    page_two = _page_with_table(
        2,
        _plain_table([header, ["IOB0", "I/O"]]),
        lead=[TextBlock("heading", "11.1 概述", BBox(0, 0, 100, 30))],
    )
    stitch_cross_page_tables([page_one, page_two])
    assert page_one.blocks[0].source_pages is None
    assert page_one.blocks[0].table.n_rows == 2
    assert len(page_two.blocks) == 2  # not absorbed


def test_cross_page_stitch_refuses_when_header_differs() -> None:
    page_one = _page_with_table(1, _plain_table([["名称", "类型"], ["IOA0", "I/O"]]))
    page_two = _page_with_table(2, _plain_table([["符号", "参数"], ["IDD", "mA"]]))
    stitch_cross_page_tables([page_one, page_two])
    assert page_one.blocks[0].source_pages is None
    assert page_one.blocks[0].table.n_rows == 2
    assert len(page_two.blocks) == 1


@pytest.mark.skipif(
    not os.environ.get("RAG_PARSER_FIXTURE_PDF") or not os.environ.get("RAG_PARSER_FIXTURE_LAYOUT"),
    reason="local licensed PDF fixture is not configured",
)
def test_m8p6272_regression_pages(tmp_path: Path) -> None:
    summary = _run(
        os.environ["RAG_PARSER_FIXTURE_PDF"],
        os.environ["RAG_PARSER_FIXTURE_LAYOUT"],
        tmp_path,
        pages=[6, 7, 8, 9, 10, 14, 15, 23, 30, 38, 47, 49, 54, 57, 64, 65, 71, 76],
    )
    assert summary["quality"]["orphan_text_lines"] == 0
    assert summary["quality"]["duplicate_owned_lines"] == 0
    assert summary["quality"]["missing_snapshot_files"] == []
    assert summary["table_snapshot_candidates"] == summary["table_snapshots"]
    assert summary["implicit_subtable_groups"] == 14
    assert summary["implicit_subrows"] == 48

    pages = {
        page["page"]: page
        for page in json.loads((tmp_path / "pages.json").read_text(encoding="utf-8"))
    }
    all_text = "\n".join(
        block["text"]
        for page in pages.values()
        for block in page["blocks"]
        if block["type"] == "text"
    )
    assert "masses-chip.com" not in all_text

    page7_snapshots = [block for block in pages[7]["blocks"] if block["type"] == "snapshot"]
    page8_snapshots = [block for block in pages[8]["blocks"] if block["type"] == "snapshot"]
    assert len(page7_snapshots) == 2
    assert len(page8_snapshots) == 2
    assert page7_snapshots[0]["bbox"][0] <= 104.0
    assert page7_snapshots[0]["bbox"][2] >= 499.0
    assert page8_snapshots[0]["bbox"][0] <= 96.0
    assert page8_snapshots[0]["bbox"][2] >= 502.0
    assert all("M8P6272" in block["annotations"] for block in page7_snapshots)
    assert all("M8P6272" in block["annotations"] for block in page8_snapshots)

    first_pin_table = _table(pages[9], 16, 3)
    assert first_pin_table["source_pages"] == [9, 10]
    assert first_pin_table["semantic_reconstruction"]["parent_rows"] == list(range(2, 16))
    assert ["IOA6", "I/O", "输入/输出IO，SMT，上拉电阻，变换中断唤醒"] in first_pin_table[
        "semantic_rows"
    ]
    assert ["PWM13", "O", "PWM13 输出"] in first_pin_table["semantic_rows"]

    page6 = [block.get("text") for block in pages[6]["blocks"] if block["type"] == "text"]
    assert page6.index("■ 系统工作模式") < page6.index("● 普通模式")
    assert not any("默认复位\n下拉" in text for text in page6)
    assert not any("用于系\n统时钟" in text for text in page6)
    assert any(
        block["kind"] == "paragraph" and block["text"].startswith("注：")
        for block in pages[7]["blocks"]
        if block["type"] == "text"
    )
    assert any(
        block["kind"] == "heading" and block["text"] == "2.2.3 系统寄存器定义"
        for block in pages[15]["blocks"]
        if block["type"] == "text"
    )
    io_change_table = next(
        block
        for block in pages[23]["blocks"]
        if block["type"] == "table" and block["rows"][1][0] == "IOAICR"
    )
    assert io_change_table["rows"][1][1:3] == ["IOAICR7", "IOAICR6"]
    port_mapping_table = next(
        block
        for block in pages[30]["blocks"]
        if block["type"] == "table" and block["rows"][1][0] == "MPPWM1"
    )
    assert port_mapping_table["rows"][1][:3] == ["MPPWM1", "MPPWMM1", "MPPWMM0"]
    pwm_control_table = next(
        block
        for block in pages[38]["blocks"]
        if block["type"] == "table" and block["rows"][1][0] == "PWM1CR"
    )
    assert pwm_control_table["rows"][1] == [
        "PWM1CR",
        "PWM13POL",
        "PWM12POL",
        "PWM11POL",
        "PWM10POL",
        "PWM13EN",
        "PWM12EN",
        "PWM11EN",
        "PWM10EN",
    ]
    assert _table_shapes(pages[49]) == [(4, 9)]
    assert _table_shapes(pages[54]) == [(4, 9), (3, 2)]
    compact_table = _table(pages[54], 3, 2)
    assert compact_table["rows"][0] == ["CMPOL", "比较器输出状态位"]
    assert compact_table["rows"][1][0] == "0"
    assert compact_table["rows"][2][0] == "1"
    assert not any(
        block["kind"] == "heading" and re.match(r"^\d+\s*=", block["text"])
        for block in pages[54]["blocks"]
        if block["type"] == "text"
    )
    assert _table_shapes(pages[57]) == [(16, 6)]
    # pages 64/65 electrical-characteristics table is stitched into one table.
    assert _table_shapes(pages[64]) == [(31, 8)]
    page65_tables = [block for block in pages[65]["blocks"] if block["type"] == "table"]
    assert [(table["n_rows"], table["n_cols"]) for table in page65_tables] == [(15, 7)]
    electrical_table = _table(pages[64], 31, 8)
    assert electrical_table["source_pages"] == [64, 65]
    assert _cell(electrical_table, 0, 7)["rowspan"] == 2
    assert _cell(electrical_table, 0, 7)["text"].replace(" ", "") == "单位"
    assert [
        (cell["row"], cell["rowspan"], cell["text"])
        for cell in electrical_table["cells"]
        if cell["col"] == 7 and cell["row"] >= 10
    ] == [(11, 2, "uA"), (13, 8, ""), (21, 2, "kΩ"), (23, 2, "kΩ"), (25, 6, "mA")]
    package_table = next(block for block in pages[71]["blocks"] if block["type"] == "table")
    assert package_table["rows"][0] == ["符号", "单位（mm）", "", ""]
    assert package_table["rows"][1] == ["", "最小", "正常", "最大"]
    assert package_table["rows"][2] == ["A", "1.35", "-", "1.75"]
    assert all(
        cell["rowspan"] == 1
        for cell in package_table["cells"]
        if cell["col"] == 0 and cell["row"] >= 2
    )
    assert _table_shapes(pages[76]) == [(34, 6)]
    assert not any(
        "Bit7 Bit6 Bit5" in block.get("text", "")
        for block in pages[49]["blocks"]
        if block["type"] == "text"
    )
    markdown = (tmp_path / "md" / "full.md").read_text(encoding="utf-8")
    assert "\n#### 2.2.3 系统寄存器定义\n" in markdown
    assert "\n## 注： 烧录接口说明\n" not in markdown
    assert "\n## Page " not in markdown


@pytest.mark.skipif(
    not os.environ.get("RAG_PARSER_M9_FIXTURE_PDF")
    or not os.environ.get("RAG_PARSER_M9_FIXTURE_LAYOUT"),
    reason="local licensed M9 PDF fixture is not configured",
)
def test_m9f6820_table_and_heading_regression(tmp_path: Path) -> None:
    _run(
        os.environ["RAG_PARSER_M9_FIXTURE_PDF"],
        os.environ["RAG_PARSER_M9_FIXTURE_LAYOUT"],
        tmp_path,
        pages=[
            9,
            10,
            11,
            12,
            13,
            24,
            28,
            41,
            45,
            72,
            82,
            100,
            119,
            122,
            126,
            134,
            135,
            136,
        ],
    )
    pages = {
        page["page"]: page
        for page in json.loads((tmp_path / "pages.json").read_text(encoding="utf-8"))
    }
    assert sum(page["stats"]["implicit_subtable_groups"] for page in pages.values()) == 18
    assert sum(page["stats"]["implicit_subrows"] for page in pages.values()) == 78
    # pages 11/12/13 pin table is stitched into one table.
    first_pin_table = _table(pages[11], 20, 3)
    assert first_pin_table["source_pages"] == [11, 12, 13]
    assert ["VREF", "A", "AD 外部参考电压输入"] in first_pin_table["semantic_rows"]
    assert ["INT1", "I", "外部中断1"] in first_pin_table["semantic_rows"]
    wrapped_header_table = _table(pages[119], 9, 8)
    assert "semantic_rows" not in wrapped_header_table
    assert pages[41]["stats"]["implicit_subtable_groups"] == 0
    assert pages[134]["stats"]["implicit_subtable_groups"] == 0
    page9_snapshots = [block for block in pages[9]["blocks"] if block["type"] == "snapshot"]
    page10_snapshots = [block for block in pages[10]["blocks"] if block["type"] == "snapshot"]
    assert len(page9_snapshots) == 2
    assert len(page10_snapshots) == 1
    assert page9_snapshots[0]["bbox"][0] <= 97.0
    assert page9_snapshots[0]["bbox"][2] >= 490.0
    assert page10_snapshots[0]["bbox"][0] <= 79.0
    assert page10_snapshots[0]["bbox"][2] >= 478.0

    page72_snapshots = [block for block in pages[72]["blocks"] if block["type"] == "snapshot"]
    assert len(page72_snapshots) == 2
    assert page72_snapshots[1]["source"] == "native_visual"
    assert page72_snapshots[1]["bbox"][0] <= 64.0
    assert page72_snapshots[1]["bbox"][2] >= 534.0

    assert _table_shapes(pages[24]) == [(6, 2), (19, 4)]
    assert pages[24]["stats"]["table_topology_upgrades"] == 1
    assert not any(block["type"] == "snapshot" for block in pages[24]["blocks"])
    address_table = next(
        block for block in pages[24]["blocks"] if block["type"] == "table" and block["n_rows"] == 19
    )
    assert address_table["rows"][0] == ["页", "地址", "用途", "说明"]
    assert address_table["rows"][1][0] == "BANK0"
    assert address_table["rows"][2][0] == ""
    assert address_table["rows"][3][0] == "BANK1"
    assert address_table["rows"][17][0] == "BANK15"
    assert address_table["rows"][18][0] == ""
    register_table = _table(pages[28], 34, 10)
    assert _cell(register_table, 0, 0)["colspan"] == 10
    assert all(cell["rowspan"] == 1 for cell in register_table["cells"] if cell["row"] >= 1)

    io_change_table = _table(pages[45], 6, 9)
    assert io_change_table["rows"][1][2:4] == ["IOAICR6", "IOAICR5"]
    assert io_change_table["rows"][2][2:4] == ["IOBICR6", "IOBICR5"]
    baud_rate_tables = [block for block in pages[82]["blocks"] if block["type"] == "table"]
    assert len(baud_rate_tables) == 2
    assert all(table["rows"][0][0] and table["rows"][0][1] for table in baud_rate_tables)

    option_table = _table(pages[122], 32, 3)
    assert [
        (cell["row"], cell["rowspan"], cell["text"].replace(" ", ""))
        for cell in option_table["cells"]
        if cell["col"] == 0 and cell["rowspan"] > 1
    ] == [
        (1, 3, "振荡器模式"),
        (4, 7, "FCPU"),
        (11, 2, "外部复位端口选择"),
        (13, 2, "启动模式选择"),
        (15, 2, "复位向量"),
        (17, 15, "复位电压选择"),
    ]

    oscillator_table = _table(pages[126], 22, 7)
    assert [_cell(oscillator_table, 0, col)["rowspan"] for col in (0, 4, 5, 6)] == [
        2,
        2,
        2,
        2,
    ]
    assert [
        (_cell(oscillator_table, row, 0)["rowspan"], _cell(oscillator_table, row, 0)["text"])
        for row in (2, 12)
    ] == [
        (10, "晶振驱动选择:高驱动 供电选择:芯片VDD 供电"),
        (10, "晶振驱动选择:高驱动 供电选择:BIAS 供电"),
    ]

    # pages 135/136 instruction table is stitched into one table.
    first_instruction_table = _table(pages[135], 73, 6)
    assert first_instruction_table["source_pages"] == [135, 136]
    assert [
        (
            _cell(first_instruction_table, row, 0)["rowspan"],
            _cell(first_instruction_table, row, 0)["text"],
        )
        for row in (1, 32)
    ] == [(31, "字 节 操 作 类 指 令"), (5, "位操作 指令")]
    assert [
        (
            _cell(first_instruction_table, row, 0)["rowspan"],
            _cell(first_instruction_table, row, 0)["text"],
        )
        for row in (37, 59, 68)
    ] == [
        (22, "控 制 操 作 类 指 令"),
        (9, "立 即 数 操 作 指 令"),
        (5, "存储器操 作指令"),
    ]
    assert any(
        block["kind"] == "heading" and block["text"] == "11.4.1 I2C 控制寄存器"
        for block in pages[100]["blocks"]
        if block["type"] == "text"
    )


@pytest.mark.skipif(
    not os.environ.get("RAG_PARSER_M8P5170_FIXTURE_PDF")
    or not os.environ.get("RAG_PARSER_M8P5170_FIXTURE_LAYOUT"),
    reason="local licensed M8P5170 PDF fixture is not configured",
)
def test_m8p5170_visual_and_table_regression(tmp_path: Path) -> None:
    summary = _run(
        os.environ["RAG_PARSER_M8P5170_FIXTURE_PDF"],
        os.environ["RAG_PARSER_M8P5170_FIXTURE_LAYOUT"],
        tmp_path,
        pages=[7, 8, 12, 27, 28, 29, 39, 40, 41],
    )
    pages = {
        page["page"]: page
        for page in json.loads((tmp_path / "pages.json").read_text(encoding="utf-8"))
    }
    assert summary["implicit_subtable_groups"] == 6
    assert summary["implicit_subrows"] == 24

    pin_table = _table(pages[8], 8, 3)
    assert pin_table["rows"][4] == [
        "IOA[2] TK2 PWM02 CP VREF",
        "I/O A O A A",
        "输入/输出IO、SMT、上拉电阻 触摸按键通道2 PWM02 输出端口 "
        "LVD 外部电压输入口 AD 外部参考电压输入",
    ]
    assert pin_table["semantic_reconstruction"]["parent_rows"] == [2, 3, 4, 5, 6, 7]
    assert len(pin_table["semantic_rows"]) == 26
    assert ["CP", "A", "LVD 外部电压输入口"] in pin_table["semantic_rows"]
    assert ["VREF", "A", "AD 外部参考电压输入"] in pin_table["semantic_rows"]
    assert len(pin_table["semantic_reconstruction"]["row_provenance"]) == 24
    assert pages[12]["stats"]["implicit_subtable_groups"] == 0
    markdown = (tmp_path / "md" / "full.md").read_text(encoding="utf-8")
    assert "| CP | A | LVD 外部电压输入口 |" in markdown
    assert "| IOA[2] TK2 PWM02 CP VREF |" not in markdown

    pin_snapshots = [block for block in pages[7]["blocks"] if block["type"] == "snapshot"]
    assert len(pin_snapshots) == 3
    assert pin_snapshots[0]["source"] == "native_visual"
    assert pin_snapshots[0]["bbox"][0] <= 94.0
    assert pin_snapshots[0]["bbox"][2] >= 529.0
    assert "M8P5170" in pin_snapshots[0]["annotations"]
    assert "T1311" in pin_snapshots[1]["annotations"]
    assert "M8P5170" in pin_snapshots[2]["annotations"]

    assert _table_shapes(pages[27]) == [(4, 9), (5, 2), (9, 2)]
    assert _table_shapes(pages[28]) == [(4, 9), (9, 4), (4, 9)]
    assert _table_shapes(pages[39]) == [(24, 3)]
    assert pages[27]["stats"]["table_topology_upgrades"] == 2
    assert pages[28]["stats"]["table_topology_upgrades"] == 1
    assert pages[39]["stats"]["table_topology_upgrades"] == 1
    assert not any(
        block["type"] == "snapshot"
        for page_number in (27, 28, 39)
        for block in pages[page_number]["blocks"]
    )

    pwm_control_table = next(
        block
        for block in pages[29]["blocks"]
        if block["type"] == "table" and block["rows"][1][0] == "PWM0CR"
    )
    assert pwm_control_table["rows"][1] == [
        "PWM0CR",
        "PWM03EN",
        "PWM03OE",
        "PWM02EN",
        "PWM02OE",
        "PWM01EN",
        "PWM01OE",
        "PWM00EN",
        "PWM00OE",
    ]

    option_table = _table(pages[39], 24, 3)
    assert [
        (_cell(option_table, row, 0)["rowspan"], _cell(option_table, row, 0)["text"])
        for row in (1, 5, 7, 9, 14, 17, 21)
    ] == [
        (4, "CPU 运行 速度选择"),
        (2, "SWP 端口设置"),
        (2, "输出IO 读取设 置"),
        (5, "复位电压选择"),
        (3, "WDT 使能选择"),
        (4, "WDT 溢出时间"),
        (3, "仿真电压 选择"),
    ]

    # pages 40/41 electrical-characteristics table is stitched into one table.
    page40 = _table(pages[40], 25, 8)
    assert page40["source_pages"] == [40, 41]
    assert [
        (cell["row"], cell["rowspan"], cell["text"])
        for cell in page40["cells"]
        if cell["col"] == 7 and cell["row"] >= 2
    ] == [
        (2, 1, "V"),
        (3, 2, "mA"),
        (5, 6, "uA"),
        (11, 2, "uA"),
        (13, 2, "kΩ"),
        (15, 2, "kΩ"),
        (17, 8, "mA"),
    ]
    assert [_cell(page40, row, 3)["rowspan"] for row in (3, 5, 7, 9, 11)] == [2] * 5


def _table_shapes(page: dict) -> list[tuple[int, int]]:
    return [
        (block["n_rows"], block["n_cols"]) for block in page["blocks"] if block["type"] == "table"
    ]


def _table(page: dict, n_rows: int, n_cols: int) -> dict:
    return next(
        block
        for block in page["blocks"]
        if block["type"] == "table" and block["n_rows"] == n_rows and block["n_cols"] == n_cols
    )


def _cell(table: dict, row: int, col: int) -> dict:
    return next(cell for cell in table["cells"] if cell["row"] == row and cell["col"] == col)
