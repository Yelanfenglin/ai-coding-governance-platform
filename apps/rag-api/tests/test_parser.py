from __future__ import annotations

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import fitz
import pytest

from app.parser.anchor import Anchor
from app.parser.document import SectionTracker, link_table_continuations
from app.parser.models import BBox, NativeDrawing, VectorRule
from app.parser.native_pdf import NativeLine, NativePage, NativeSpan, _extract_rules
from app.parser.native_tables import recover_missing_tables
from app.parser.page import (
    PageIR,
    TextBlock,
    _detect_two_column_split,
    _two_column_order,
    prepare_table_anchors,
)
from app.parser.parse import _run
from app.parser.postprocess import (
    markdown_heading_level,
    normalize_text_block,
    recover_document_headings,
    reflow_text,
)
from app.parser.renderer import _render_block
from app.parser.table import (
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


@pytest.mark.skipif(
    not os.environ.get("RAG_PARSER_FIXTURE_PDF") or not os.environ.get("RAG_PARSER_FIXTURE_LAYOUT"),
    reason="local licensed PDF fixture is not configured",
)
def test_m8p6272_regression_pages(tmp_path: Path) -> None:
    summary = _run(
        os.environ["RAG_PARSER_FIXTURE_PDF"],
        os.environ["RAG_PARSER_FIXTURE_LAYOUT"],
        tmp_path,
        pages=[6, 7, 8, 14, 15, 47, 49, 54, 57, 64, 65, 71, 76],
    )
    assert summary["quality"]["orphan_text_lines"] == 0
    assert summary["quality"]["duplicate_owned_lines"] == 0
    assert summary["quality"]["missing_snapshot_files"] == []
    assert summary["table_snapshot_candidates"] == summary["table_snapshots"]

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

    page7_snapshots = [
        block for block in pages[7]["blocks"] if block["type"] == "snapshot"
    ]
    page8_snapshots = [
        block for block in pages[8]["blocks"] if block["type"] == "snapshot"
    ]
    assert len(page7_snapshots) == 2
    assert len(page8_snapshots) == 2
    assert page7_snapshots[0]["bbox"][0] <= 104.0
    assert page7_snapshots[0]["bbox"][2] >= 499.0
    assert page8_snapshots[0]["bbox"][0] <= 96.0
    assert page8_snapshots[0]["bbox"][2] >= 502.0
    assert all("M8P6272" in block["annotations"] for block in page7_snapshots)
    assert all("M8P6272" in block["annotations"] for block in page8_snapshots)

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
    assert _table_shapes(pages[64]) == [(13, 8)]
    page65_tables = [block for block in pages[65]["blocks"] if block["type"] == "table"]
    assert [(table["n_rows"], table["n_cols"]) for table in page65_tables] == [(20, 8), (15, 7)]
    electrical_table = page65_tables[0]
    assert _cell(electrical_table, 0, 7)["rowspan"] == 2
    assert _cell(electrical_table, 0, 7)["text"].replace(" ", "") == "单位"
    assert [
        (cell["row"], cell["rowspan"], cell["text"])
        for cell in electrical_table["cells"]
        if cell["col"] == 7 and cell["row"] >= 10
    ] == [(10, 2, "kΩ"), (12, 2, "kΩ"), (14, 6, "mA")]
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
        pages=[9, 10, 24, 28, 72, 100, 122, 126, 135, 136],
    )
    pages = {
        page["page"]: page
        for page in json.loads((tmp_path / "pages.json").read_text(encoding="utf-8"))
    }
    page9_snapshots = [
        block for block in pages[9]["blocks"] if block["type"] == "snapshot"
    ]
    page10_snapshots = [
        block for block in pages[10]["blocks"] if block["type"] == "snapshot"
    ]
    assert len(page9_snapshots) == 2
    assert len(page10_snapshots) == 1
    assert page9_snapshots[0]["bbox"][0] <= 97.0
    assert page9_snapshots[0]["bbox"][2] >= 490.0
    assert page10_snapshots[0]["bbox"][0] <= 79.0
    assert page10_snapshots[0]["bbox"][2] >= 478.0

    page72_snapshots = [
        block for block in pages[72]["blocks"] if block["type"] == "snapshot"
    ]
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

    first_instruction_table = _table(pages[135], 37, 6)
    assert [
        (
            _cell(first_instruction_table, row, 0)["rowspan"],
            _cell(first_instruction_table, row, 0)["text"],
        )
        for row in (1, 32)
    ] == [(31, "字 节 操 作 类 指 令"), (5, "位操作 指令")]
    second_instruction_table = _table(pages[136], 37, 6)
    assert [
        (
            _cell(second_instruction_table, row, 0)["rowspan"],
            _cell(second_instruction_table, row, 0)["text"],
        )
        for row in (1, 23, 32)
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
    _run(
        os.environ["RAG_PARSER_M8P5170_FIXTURE_PDF"],
        os.environ["RAG_PARSER_M8P5170_FIXTURE_LAYOUT"],
        tmp_path,
        pages=[7, 27, 28, 39, 40, 41],
    )
    pages = {
        page["page"]: page
        for page in json.loads((tmp_path / "pages.json").read_text(encoding="utf-8"))
    }

    pin_snapshots = [
        block for block in pages[7]["blocks"] if block["type"] == "snapshot"
    ]
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

    page40 = _table(pages[40], 13, 8)
    assert [
        (cell["row"], cell["rowspan"], cell["text"])
        for cell in page40["cells"]
        if cell["col"] == 7 and cell["row"] >= 2
    ] == [(2, 1, "V"), (3, 2, "mA"), (5, 6, "uA"), (11, 2, "uA")]
    assert [_cell(page40, row, 3)["rowspan"] for row in (3, 5, 7, 9, 11)] == [2] * 5

    page41 = _table(pages[41], 14, 8)
    assert [
        (cell["row"], cell["rowspan"], cell["text"])
        for cell in page41["cells"]
        if cell["col"] == 7 and cell["row"] >= 2
    ] == [(2, 2, "kΩ"), (4, 2, "kΩ"), (6, 8, "mA")]


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
