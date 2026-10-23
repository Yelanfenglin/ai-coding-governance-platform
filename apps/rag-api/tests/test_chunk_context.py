"""Regression cases for dependent captions, conditions and field boundaries."""

from __future__ import annotations

from app.chunking import build_chunks, build_semantic_units
from app.chunking.loader import ParsedDocument


def text(body, section="1.1", *, heading=False, title="1.1 参数"):
    return {
        "type": "text",
        "kind": "heading" if heading else "paragraph",
        "text": body,
        "section_id": section,
        "section_path": ["1 功能", title],
    }


def table(rows, section="1.1"):
    return {
        "type": "table",
        "rows": rows,
        "section_id": section,
        "section_path": ["1 功能", "1.1 参数"],
        "cells": [],
    }


def build(pages):
    doc = ParsedDocument("demo_spec_v1.0", "DEMO", "1.0", "DEMO_SPEC_V1.0.pdf", pages)
    return build_chunks(doc, build_semantic_units(doc))


def test_frequency_caption_stays_with_its_own_table():
    chunks = build(
        [
            {
                "page": 1,
                "blocks": [
                    text("1.1 参数", heading=True),
                    text("波特率 = FOSC/((BRGD+1)×n)"),
                    text("常用波特率设置（FOSC = 16MHz）"),
                    table([["波特率", "BRGD"], ["9600", "0x67"]]),
                    text("常用波特率设置（Fosc = 24MHz）"),
                    table([["波特率", "BRGD"], ["9600", "0x9B"]]),
                ],
            }
        ]
    )
    tables = [c for c in chunks if c.metadata.semantic_type == "table"]
    assert len(tables) == 2
    assert "16MHz" in tables[0].embedding_text and "24MHz" not in tables[0].embedding_text
    assert "24MHz" in tables[1].embedding_text and "16MHz" not in tables[1].embedding_text
    assert not any(c.metadata.semantic_type == "text" for c in chunks)


def test_table_has_boot_restriction_and_inverted_write_note():
    chunks = build(
        [
            {
                "page": 1,
                "blocks": [
                    text("修改配置字区只能在BOOT 中进行，对应地址如下。"),
                    table([["RAM 地址", "对应OPT"], ["0x060", "OPT0L"]]),
                    text("注：对应寄存器一定要取反写入。"),
                ],
            }
        ]
    )
    table_chunk = next(c for c in chunks if c.metadata.semantic_type == "table")
    for view in [table_chunk.content, table_chunk.embedding_text]:
        assert view.index("BOOT") < view.index("OPT0L") < view.index("取反写入")


def test_split_tables_repeat_context_and_preserve_its_source_pages():
    rows = [["名称", "说明"]] + [[f"ROW{i}", "参数说明" * 100] for i in range(6)]
    chunks = build(
        [
            {"page": 1, "blocks": [text("参数适用条件（电源电压 = 3V）")]},
            {"page": 2, "blocks": [table(rows)]},
            {"page": 3, "blocks": [text("注：具体值不做设计保证。")]},
        ]
    )
    tables = [c for c in chunks if c.metadata.semantic_type == "table"]
    assert len(tables) > 1
    for c in tables:
        assert "电源电压 = 3V" in c.embedding_text
        assert "具体值不做设计保证" in c.embedding_text
        assert c.metadata.source_pages == [1, 2, 3]
        assert c.metadata.embedding_tokens <= 1200
    assert sum(c.content.count("ROW") for c in tables) == 6


def test_shared_caption_and_legend_cover_consecutive_tables():
    chunks = build(
        [
            {
                "page": 1,
                "blocks": [
                    text("存储区域读写权限表"),
                    table([["读", "区域"], ["可读", "APP"]]),
                    table([["写", "区域"], ["不可写", "APP"]]),
                    text("注：空白处为未实现地址，读为0，无法写。"),
                ],
            }
        ]
    )
    tables = [c for c in chunks if c.metadata.semantic_type == "table"]
    assert len(tables) == 2
    assert all("存储区域读写权限表" in c.embedding_text for c in tables)
    assert all("未实现地址" in c.embedding_text for c in tables)


def test_context_does_not_cross_a_heading():
    chunks = build(
        [
            {
                "page": 1,
                "blocks": [
                    text("1.1 参数", heading=True),
                    table([["参数", "值"], ["A", "1"]]),
                    text("1.2 独立功能", "1.2", heading=True, title="1.2 独立功能"),
                    text("注：该独立功能支持掉电唤醒。", "1.2", title="1.2 独立功能"),
                ],
            }
        ]
    )
    c = next(c for c in chunks if c.metadata.semantic_type == "table")
    assert "掉电唤醒" not in c.content
    assert any("掉电唤醒" in c.embedding_text for c in chunks)


def test_each_tables_own_note_closes_its_context_group():
    chunks = build(
        [
            {
                "page": 1,
                "blocks": [
                    table([["读", "区域"], ["可读", "APP"]]),
                    text("注：读操作适用于用户区。"),
                    table([["写", "区域"], ["不可写", "APP"]]),
                    text("注：写操作需要解锁。"),
                ],
            }
        ]
    )
    tables = [c for c in chunks if c.metadata.semantic_type == "table"]
    assert "读操作适用于用户区" in tables[0].content
    assert "写操作需要解锁" not in tables[0].content
    assert "写操作需要解锁" in tables[1].content
    assert "读操作适用于用户区" not in tables[1].content


def test_figure_uses_local_title_when_section_number_is_repeated():
    def figure(title, ref):
        return {
            "type": "snapshot",
            "section_id": "1.1",
            "section_path": ["1 功能", title],
            "image_ref": ref,
            "bbox": [0, 0, 100, 100],
            "annotations": title,
        }

    chunks = build(
        [
            {
                "page": 1,
                "blocks": [
                    text("1.1 MSOP10", heading=True, title="1.1 MSOP10"),
                    figure("1.1 MSOP10", "snapshots/a.png"),
                ],
            },
            {
                "page": 2,
                "blocks": [
                    text("1.1 SOP20", heading=True, title="1.1 SOP20"),
                    figure("1.1 SOP20", "snapshots/b.png"),
                ],
            },
        ]
    )
    figures = [c for c in chunks if c.metadata.semantic_type == "figure"]
    assert figures[1].metadata.title == "1.1 SOP20"
    assert "MSOP10" not in figures[1].embedding_text


def test_bit_field_heading_is_separated_but_inline_reference_is_unchanged():
    original = "Bit 2 RXIF：接收标志\n1 = 接收中断Bit 1 TXIF：发送标志\n参考Bit 2 RXIF设置。"
    chunks = build([{"page": 1, "blocks": [text(original)]}])
    c = next(c for c in chunks if c.metadata.semantic_type == "register")
    assert "接收中断\n\nBit 1 TXIF：" in c.content
    assert "参考Bit 2 RXIF设置。" in c.content
    assert "".join(c.content.split()) == "".join(original.split())


def test_reference_only_notice_is_preserved_without_a_vector_but_short_fact_is_kept():
    chunks = build(
        [
            {
                "page": 1,
                "blocks": [
                    text("指令集详细说明请到官网下载：\n\nDEMO 指令说明"),
                    text("1.2 I2C", "1.2", heading=True, title="1.2 I2C"),
                    text("支持高速I2C（400K）Slave。", "1.2", title="1.2 I2C"),
                ],
            }
        ]
    )
    notice = next(c for c in chunks if "官网下载" in c.content)
    fact = next(c for c in chunks if "400K" in c.content)
    assert notice.exclude_from_embedding
    assert not fact.exclude_from_embedding
