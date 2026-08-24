"""Chunking layer unit tests (synthetic documents, no licensed PDFs)."""

from __future__ import annotations

from app.chunking import build_chunks, build_semantic_units
from app.chunking.loader import _doc_meta_from_pdf_name
from app.chunking.metadata import build_chunk_id, estimate_tokens

# ---------------------------------------------------------------------------
# synthetic builders
# ---------------------------------------------------------------------------


def _doc(pages: list[dict], pdf_name: str = "DEMO_SPEC_V1.0.pdf"):
    product, version, doc_id = _doc_meta_from_pdf_name(pdf_name)
    return __import__("app.chunking.loader", fromlist=["ParsedDocument"]).ParsedDocument(
        doc_id=doc_id, product=product, version=version, pdf_name=pdf_name, pages=pages
    )


def _page(number: int, blocks: list[dict]) -> dict:
    return {"page": number, "width": 595.0, "height": 842.0, "stats": {}, "blocks": blocks}


def _text(kind: str, text: str, section_id: str, path: list[str]) -> dict:
    return {
        "type": "text",
        "kind": kind,
        "text": text,
        "section_id": section_id,
        "section_path": path,
        "source": "pp",
        "block_index": 0,
    }


def _table(rows: list[list[str]], section_id: str, path: list[str], spans=None) -> dict:
    cells = []
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            cells.append(
                {"row": r, "col": c, "rowspan": 1, "colspan": 1, "text": cell, "bbox": None}
            )
    if spans:
        for r, c, rs, cs in spans:
            cells.append(
                {
                    "row": r,
                    "col": c,
                    "rowspan": rs,
                    "colspan": cs,
                    "text": rows[r][c] if r < len(rows) and c < len(rows[r]) else "",
                    "bbox": None,
                }
            )
    return {
        "type": "table",
        "section_id": section_id,
        "section_path": path,
        "source": "pp",
        "n_rows": len(rows),
        "n_cols": len(rows[0]),
        "rows": rows,
        "cells": cells,
        "block_index": 0,
    }


def _snapshot(hint: str, annotations: str, section_id: str, path: list[str]) -> dict:
    return {
        "type": "snapshot",
        "hint": hint,
        "annotations": annotations,
        "section_id": section_id,
        "section_path": path,
        "image_ref": "snapshots/p001_s1_x.png",
        "block_index": 0,
    }


def _chunks(doc):
    units = build_semantic_units(doc)
    chunks = build_chunks(doc, units)
    return chunks, units


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_doc_meta_parsed_from_pdf_name() -> None:
    product, version, doc_id = _doc_meta_from_pdf_name("M8P6272_SPEC_V1.10CN.pdf")
    assert product == "M8P6272"
    assert version == "1.10"
    assert doc_id == "m8p6272_spec_v1.10"


def test_estimate_tokens_mixes_cjk_and_ascii() -> None:
    assert estimate_tokens("寄存器 PWM1CR") > estimate_tokens("寄存器")
    assert estimate_tokens("寄存器 PWM1CR") < estimate_tokens("寄存器 PWM1CR " * 3)


def test_chunk_id_format() -> None:
    assert (
        build_chunk_id("demo_spec_v1.0", "8.6", "register", 3) == "demo_spec_v1.0:8.6:register:003"
    )


def test_toc_unit_is_excluded_from_embedding() -> None:
    doc = _doc(
        [
            _page(
                3,
                [
                    _text("heading", "目录", "", []),
                    _text("paragraph", "1 产品简述 .......... 6", "", []),
                ],
            ),
        ]
    )
    chunks, _ = _chunks(doc)
    toc = [c for c in chunks if c.metadata.semantic_type == "toc"]
    assert len(toc) == 1
    assert toc[0].exclude_from_embedding is True
    assert toc[0].embedding_text == ""


def test_register_package_is_atomic_and_keeps_table() -> None:
    doc = _doc(
        [
            _page(
                4,
                [
                    _text(
                        "heading",
                        "7.2 PWM1CR 控制寄存器",
                        "7.2",
                        ["7 定时器", "7.2 PWM1CR 控制寄存器"],
                    ),
                    _table(
                        [
                            ["", "Bit7", "Bit6", "Bit5", "Bit4", "Bit3", "Bit2", "Bit1", "Bit0"],
                            ["PWM1CR", "POL7", "POL6", "POL5", "POL4", "EN3", "EN2", "EN1", "EN0"],
                        ],
                        "7.2",
                        ["7 定时器", "7.2 PWM1CR 控制寄存器"],
                    ),
                    _text(
                        "paragraph",
                        "Bit [7:4] PWM1xPOL： 反向输出使能位",
                        "7.2",
                        ["7 定时器", "7.2 PWM1CR 控制寄存器"],
                    ),
                ],
            ),
        ]
    )
    chunks, units = _chunks(doc)
    registers = [c for c in chunks if c.metadata.semantic_type == "register"]
    assert len(registers) == 1
    reg = registers[0]
    assert reg.metadata.atomic is True
    assert reg.metadata.register == "PWM1CR"
    assert "Bit7" in reg.content
    assert "POL7" in reg.content
    assert "Bit [7:4]" in reg.content
    assert "反向输出使能位" in reg.embedding_text


def test_text_unit_does_not_cross_section_heading() -> None:
    doc = _doc(
        [
            _page(
                5,
                [
                    _text("heading", "7.1 概述", "7.1", ["7 定时器", "7.1 概述"]),
                    _text(
                        "paragraph", "TC1 为带周期寄存器的定时器。", "7.1", ["7 定时器", "7.1 概述"]
                    ),
                    _text("heading", "7.2 寄存器", "7.2", ["7 定时器", "7.2 寄存器"]),
                    _text("paragraph", "寄存器内容是关键。", "7.2", ["7 定时器", "7.2 寄存器"]),
                ],
            ),
        ]
    )
    chunks, units = _chunks(doc)
    texts = [c for c in chunks if c.metadata.semantic_type == "text"]
    assert len(texts) == 2
    for chunk in texts:
        assert "7.2 寄存器" not in chunk.content or chunk.metadata.section_id == "7.2"
        assert (
            "TC1 为带周期寄存器的定时器。" not in chunk.content
            or chunk.metadata.section_id == "7.1"
        )


def test_table_chunk_repeats_schema_header() -> None:
    rows = [
        ["符号", "参数", "最小值", "单位"],
        ["VDD", "电源", "1.8", "V"],
        ["IDD1", "电流", "0.6", "mA"],
        ["IDD2", "电流", "0.9", "mA"],
    ]
    doc = _doc(
        [
            _page(
                6,
                [
                    _text("heading", "14.2 直流特性", "14.2", ["14 参数", "14.2 直流特性"]),
                    _table(rows, "14.2", ["14 参数", "14.2 直流特性"]),
                ],
            ),
        ]
    )
    chunks, _ = _chunks(doc)
    tables = [c for c in chunks if c.metadata.block_types == ["table"]]
    assert tables, "table chunk missing"
    for chunk in tables:
        assert "| 符号 | 参数 | 最小值 | 单位 |" in chunk.content
        assert chunk.metadata.table_id
        assert chunk.metadata.row_group


def test_semantic_rows_grouped_by_parent_row() -> None:
    raw_rows = [
        ["名称", "类型", "说明"],
        ["IOA[0]", "I/O", "输入输出"],
        ["TK0", "A", "触摸"],
        ["IOA[1]", "I/O", "输入输出"],
        ["SWP", "I/O", "编程"],
    ]
    semantic_rows = [
        ["名称", "类型", "说明"],
        ["IOA[0]", "I/O", "输入输出"],
        ["TK0", "A", "触摸"],
        ["IOA[1]", "I/O", "输入输出"],
        ["SWP", "I/O", "编程"],
    ]
    rec = {
        "kind": "aligned_subrows",
        "method": "native_baseline_repetition_v1",
        "parent_rows": [2, 3],
        "row_provenance": [
            {"semantic_row": 2, "parent_row": 2, "subrow": 0, "cell_bboxes": []},
            {"semantic_row": 3, "parent_row": 3, "subrow": 0, "cell_bboxes": []},
        ],
    }
    table = _table(raw_rows, "1.3", ["1 产品", "1.3 引脚描述"])
    table["semantic_rows"] = semantic_rows
    table["semantic_reconstruction"] = rec
    doc = _doc(
        [
            _page(
                7,
                [
                    _text("heading", "1.3 引脚描述", "1.3", ["1 产品", "1.3 引脚描述"]),
                    table,
                ],
            ),
        ]
    )
    chunks, _ = _chunks(doc)
    tables = [c for c in chunks if c.metadata.block_types == ["table"]]
    assert tables
    assert all("| 名称 | 类型 | 说明 |" in c.content for c in tables)


def test_figure_unit_carries_annotations() -> None:
    doc = _doc(
        [
            _page(
                8,
                [
                    _text("heading", "1.5 结构框图", "1.5", ["1 产品", "1.5 结构框图"]),
                    _snapshot("image", "OSC CPU RAM 外设", "1.5", ["1 产品", "1.5 结构框图"]),
                ],
            ),
        ]
    )
    chunks, _ = _chunks(doc)
    figures = [c for c in chunks if c.metadata.semantic_type == "figure"]
    assert len(figures) == 1
    assert figures[0].metadata.image_ref
    assert "OSC CPU RAM 外设" in figures[0].embedding_text
    assert not figures[0].exclude_from_embedding


def test_parent_container_exists_for_every_section() -> None:
    doc = _doc(
        [
            _page(
                9,
                [
                    _text("heading", "2 重启", "2", ["2 重启"]),
                    _text("paragraph", "有关复位方式说明。", "2", ["2 重启"]),
                ],
            ),
        ]
    )
    chunks, _ = _chunks(doc)
    children = [c for c in chunks if not c.exclude_from_embedding]
    assert children
    assert all(c.parent_id for c in children)
    parents = [c for c in chunks if c.metadata.semantic_type == "parent"]
    assert parents
    assert parents[0].exclude_from_embedding is True


def test_cross_page_source_pages_carried_into_metadata() -> None:
    table = _table(
        [["符号", "参数"], ["VDD", "工作电压"]],
        "14.2",
        ["14 参数", "14.2 直流特性"],
    )
    table["source_pages"] = [40, 41]
    doc = _doc(
        [
            _page(
                40,
                [
                    _text("heading", "14.2 直流特性", "14.2", ["14 参数", "14.2 直流特性"]),
                    table,
                ],
            ),
        ]
    )
    chunks, _ = _chunks(doc)
    table_chunks = [c for c in chunks if c.metadata.block_types == ["table"]]
    assert table_chunks
    assert table_chunks[0].metadata.source_pages == [40, 41]
