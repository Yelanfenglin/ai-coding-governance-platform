"""Text-across-figure merge, figure description chunks and association tests."""

from __future__ import annotations

from app.chunking import build_chunks, build_semantic_units
from app.chunking.figure_descriptions import apply_figure_descriptions
from app.chunking.loader import ParsedDocument, _doc_meta_from_pdf_name

DOC_ID = "demo_spec_v1.0"


def _doc(pages: list[dict]) -> ParsedDocument:
    product, version, doc_id = _doc_meta_from_pdf_name("DEMO_SPEC_V1.0.pdf")
    return ParsedDocument(
        doc_id=doc_id, product=product, version=version, pdf_name="DEMO_SPEC_V1.0.pdf", pages=pages
    )


def _page(number: int, blocks: list[dict]) -> dict:
    return {"page": number, "width": 595.0, "height": 842.0, "stats": {}, "blocks": blocks}


def _heading(text: str, section_id: str, path: list[str]) -> dict:
    return {
        "type": "text",
        "kind": "heading",
        "text": text,
        "section_id": section_id,
        "section_path": path,
        "source": "pp",
        "block_index": 0,
    }


def _paragraph(text: str, section_id: str, path: list[str], index: int = 1) -> dict:
    return {
        "type": "text",
        "kind": "paragraph",
        "text": text,
        "section_id": section_id,
        "section_path": path,
        "source": "pp",
        "block_index": index,
    }


def _figure(
    image_ref: str,
    section_id: str,
    path: list[str],
    *,
    bbox: list[float] | None = None,
    annotations: str = "",
    hint: str = "chart",
    index: int = 2,
) -> dict:
    return {
        "type": "snapshot",
        "hint": hint,
        "image_ref": image_ref,
        "annotations": annotations,
        "bbox": bbox or [60.0, 100.0, 520.0, 300.0],
        "section_id": section_id,
        "section_path": path,
        "block_index": index,
    }


def _table(cell: str, section_id: str, path: list[str], index: int = 1) -> dict:
    return {
        "type": "table",
        "rows": [["", "Bit7", "Bit6"], ["REG", cell, "X"], ["读/写", "R/W", "R/W"]],
        "section_id": section_id,
        "section_path": path,
        "source": "pp",
        "block_index": index,
    }


def _description(figure_id: str, short: str, full: str, labels: list[str]) -> dict:
    return {
        "figure_id": figure_id,
        "description_short": short,
        "description_full": full,
        "visible_labels": labels,
        "diagram_type": "waveform",
        "generation_status": "generated",
        "uncertainties": [],
    }


def _chunks(doc: ParsedDocument, descriptions: dict | None = None):
    units = build_semantic_units(doc, figure_descriptions=descriptions)
    return build_chunks(doc, units), units


def test_text_flow_keeps_order_across_figure_and_uses_short_description():
    path = ["2 章节", "2.2 时序"]
    pages = [
        _page(
            1,
            [
                _heading("2.2 时序", "2.2", path),
                _paragraph("正文 A：计数器开始累加。", "2.2", path, 1),
                _figure("snapshots/p001_s1_chart.png", "2.2", path, index=2),
                _paragraph("正文 B：周期结束时输出翻转。", "2.2", path, 3),
            ],
        )
    ]
    doc = _doc(pages)
    figure_id = f"{DOC_ID}:p001:f1"
    descriptions = {
        figure_id: _description(figure_id, "PWM 周期时序短说明", "完整说明内容", ["TC0CL"])
    }
    chunks, units = _chunks(doc, descriptions)

    texts = [chunk for chunk in chunks if chunk.metadata.semantic_type == "text"]
    assert len(texts) == 1
    chunk = texts[0]
    # retrieval view keeps A -> short description -> B order
    marker = "[图：PWM 周期时序短说明]"
    assert chunk.embedding_text.index("正文 A") < chunk.embedding_text.index(marker)
    assert chunk.embedding_text.index(marker) < chunk.embedding_text.index("正文 B")
    # content keeps the original text and the image reference
    assert "正文 A" in chunk.content and "正文 B" in chunk.content
    assert "![图](snapshots/p001_s1_chart.png)" in chunk.content
    assert "snapshots/" not in chunk.embedding_text
    assert chunk.metadata.figure_ids == [figure_id]

    figures = [c for c in chunks if c.metadata.semantic_type == "figure"]
    assert len(figures) == 1
    figure = figures[0]
    assert figure.metadata.figure_id == figure_id
    assert "完整说明内容" in figure.embedding_text
    assert "TC0CL" in figure.embedding_text
    assert "类型：" not in figure.embedding_text and "标签：" not in figure.embedding_text
    assert "snapshots/" not in figure.embedding_text
    assert chunk.chunk_id in figure.metadata.related_chunk_ids


def test_register_section_no_longer_swallows_figures():
    path = ["3 寄存器", "3.1 CTRL"]
    pages = [
        _page(
            1,
            [
                _heading("3.1 CTRL", "3.1", path),
                _table("EN", "3.1", path, 1),
                _paragraph("Bit7 EN：使能位", "3.1", path, 2),
                _figure("snapshots/p001_s2_image.png", "3.1", path, index=3, hint="image"),
            ],
        )
    ]
    doc = _doc(pages)
    figure_id = f"{DOC_ID}:p001:f1"
    descriptions = {figure_id: _description(figure_id, "框图短说明", "框图完整说明", ["EN"])}
    chunks, _ = _chunks(doc, descriptions)
    kinds = [chunk.metadata.semantic_type for chunk in chunks]
    assert "register" in kinds
    assert "figure" in kinds, "figure inside a register section must still produce a chunk"
    figure = next(chunk for chunk in chunks if chunk.metadata.semantic_type == "figure")
    assert figure.metadata.figure_id == figure_id
    register = next(chunk for chunk in chunks if chunk.metadata.semantic_type == "register")
    # metadata-only association both ways; the atomic register package stays intact
    assert figure_id in register.metadata.figure_ids
    assert register.chunk_id in figure.metadata.related_chunk_ids
    assert "标注" not in register.content or "框图" not in register.content


def test_figure_only_section_does_not_emit_a_duplicate_text_wrapper():
    path = ["1 产品", "1.3 引脚图"]
    doc = _doc([_page(1, [
        _heading("1.3 引脚图", "1.3", path),
        _figure("snapshots/pinout.png", "1.3", path),
    ])])
    fid = f"{DOC_ID}:p001:f1"
    chunks, _ = _chunks(doc, {fid: _description(fid, "引脚图", "各脚列出 IO 与复用功能。", [])})
    assert len([c for c in chunks if c.metadata.semantic_type == "figure"]) == 1
    assert not [c for c in chunks if c.metadata.semantic_type == "text"]
    figure = next(c for c in chunks if c.metadata.semantic_type == "figure")
    assert figure.parent_id


def test_duplicate_figure_bbox_is_skipped():
    path = ["2 章节", "2.1 图"]
    bbox = [60.0, 100.0, 520.0, 300.0]
    pages = [
        _page(
            1,
            [
                _heading("2.1 图", "2.1", path),
                _figure("snapshots/p001_s1_image.png", "2.1", path, bbox=bbox, index=1),
                _figure("snapshots/p001_s2_image.png", "2.1", path, bbox=bbox, index=2),
            ],
        )
    ]
    doc = _doc(pages)
    aligned, report = apply_figure_descriptions(doc, {})
    assert report["duplicates"] == 1
    figures = [
        block
        for page in aligned.pages
        for block in page["blocks"]
        if block.get("type") == "snapshot"
        and (page["page"], tuple(round(v, 1) for v in block["bbox"]))  # keep format stable
    ]
    assert len(figures) == 2  # blocks kept for provenance, second marked skip_block
    assert any(block.get("skip_block") for block in figures)
    chunks, _ = _chunks(doc, {})
    figure_chunks = [c for c in chunks if c.metadata.semantic_type == "figure"]
    assert len(figure_chunks) == 1


def test_undescribed_figure_falls_back_without_path_in_embedding():
    path = ["2 章节", "2.3 波形"]
    pages = [
        _page(
            1,
            [
                _heading("2.3 波形", "2.3", path),
                _paragraph("正文 A。", "2.3", path, 1),
                _figure("snapshots/p001_s3_chart.png", "2.3", path, index=2, annotations="TCK PWM"),
                _paragraph("正文 B。", "2.3", path, 3),
            ],
        )
    ]
    doc = _doc(pages)
    chunks, _ = _chunks(doc, {})
    text = next(chunk for chunk in chunks if chunk.metadata.semantic_type == "text")
    assert text.embedding_text.count("[图]") == 1
    figure = next(chunk for chunk in chunks if chunk.metadata.semantic_type == "figure")
    assert figure.metadata.figure_description_status == "missing"
    assert "TCK PWM" in figure.embedding_text
    assert "snapshots/" not in figure.embedding_text


def test_overlap_between_chunks_never_carries_image_path():
    """Long paragraph + figure + short paragraph: the second chunk's overlap tail must
    come from the original paragraphs, not from rendered content."""
    path = ["2 章节", "2.4 长节"]
    long_paragraph = "".join(f"第{i}句说明寄存器配置与采样时序的关系。" for i in range(55))
    pages = [
        _page(
            1,
            [
                _heading("2.4 长节", "2.4", path),
                _paragraph(long_paragraph, "2.4", path, 1),
                _figure("snapshots/p001_s9_chart.png", "2.4", path, index=2),
                _paragraph("这段正文位于图之后，用来触发第二个块。", "2.4", path, 3),
            ],
        )
    ]
    doc = _doc(pages)
    figure_id = f"{DOC_ID}:p001:f1"
    chunks, _ = _chunks(
        doc,
        {figure_id: _description(figure_id, "波形短说明", "波形完整说明", ["TCK"])},
    )
    texts = [chunk for chunk in chunks if chunk.metadata.semantic_type == "text"]
    assert len(texts) >= 2
    for chunk in texts:
        assert "snapshots/" not in chunk.embedding_text
        assert ".png" not in chunk.embedding_text
    # the overlap lead-in for the later chunk is real paragraph text
    assert "第" in texts[1].embedding_text


def test_failed_description_is_not_counted_as_described():
    path = ["2 章节", "2.5 失败回退"]
    pages = [
        _page(
            1,
            [
                _heading("2.5 失败回退", "2.5", path),
                _paragraph("正文 A。", "2.5", path, 1),
                _figure(
                    "snapshots/p001_s5_image.png",
                    "2.5",
                    path,
                    index=2,
                    annotations="标注文本",
                ),
                _paragraph("正文 B。", "2.5", path, 3),
            ],
        )
    ]
    doc = _doc(pages)
    figure_id = f"{DOC_ID}:p001:f1"
    failed = {
        figure_id: {
            "figure_id": figure_id,
            "generation_status": "failed",
            "description_short": "",
            "description_full": "",
            "visible_labels": [],
            "diagram_type": "",
        }
    }
    aligned, report = apply_figure_descriptions(doc, failed)
    assert report["described"] == 0
    assert report["undescribed"] == 1
    block = next(
        block
        for page in aligned.pages
        for block in page["blocks"]
        if block.get("type") == "snapshot"
    )
    assert block["description_status"] == "failed"
    assert "figure_description" not in block

    chunks, _ = _chunks(doc, failed)
    text = next(chunk for chunk in chunks if chunk.metadata.semantic_type == "text")
    assert "[图]" in text.embedding_text
    figure = next(chunk for chunk in chunks if chunk.metadata.semantic_type == "figure")
    assert figure.metadata.figure_description_status == "failed"
    assert "标注文本" in figure.embedding_text
    assert "snapshots/" not in figure.embedding_text


def test_incomplete_description_requires_both_short_and_full():
    path = ["2 章节", "2.6 半成品"]
    pages = [
        _page(
            1,
            [
                _heading("2.6 半成品", "2.6", path),
                _paragraph("正文 A。", "2.6", path, 1),
                _figure("snapshots/p001_s6_image.png", "2.6", path, index=2),
            ],
        )
    ]
    doc = _doc(pages)
    figure_id = f"{DOC_ID}:p001:f1"
    half = {
        figure_id: {
            "figure_id": figure_id,
            "generation_status": "generated",
            "description_short": "只有短说明",
            "description_full": "",
            "visible_labels": [],
            "diagram_type": "chart",
        }
    }
    aligned, report = apply_figure_descriptions(doc, half)
    assert report["described"] == 0
    block = next(
        block
        for page in aligned.pages
        for block in page["blocks"]
        if block.get("type") == "snapshot"
    )
    assert block["description_status"] == "failed"
    assert block["description_error"] == "incomplete_description"


def test_excluded_figure_is_not_in_text_flow_and_not_embedded():
    path = ["1 产品简述", "1.0 封面"]
    pages = [
        _page(
            1,
            [
                _heading("1.0 封面", "1.0", path),
                _paragraph("正文 A。", "1.0", path, 1),
                _figure("snapshots/p001_s7_image.png", "1.0", path, index=2),
                _paragraph("正文 B。", "1.0", path, 3),
            ],
        )
    ]
    doc = _doc(pages)
    figure_id = f"{DOC_ID}:p001:f1"
    excluded = {
        figure_id: {
            "figure_id": figure_id,
            "excluded": True,
            "exclusion_reason": "vendor logo",
            "generation_status": "excluded",
        }
    }
    chunks, _ = _chunks(doc, excluded)
    text_chunk = next(chunk for chunk in chunks if chunk.metadata.semantic_type == "text")
    assert "[图" not in text_chunk.embedding_text
    assert "![图]" not in text_chunk.content
    figure = next(chunk for chunk in chunks if chunk.metadata.semantic_type == "figure")
    assert figure.metadata.figure_description_status == "excluded"
    assert figure.exclude_from_embedding is True


def test_described_figure_sets_figure_type_and_keeps_image_description():
    """figure_type comes from the description; image_description must survive the
    snapshot metadata pass (regression: it used to be cleared right after assignment)."""
    path = ["2 章节", "2.7 框图"]
    pages = [
        _page(
            1,
            [
                _heading("2.7 框图", "2.7", path),
                _paragraph("正文 A。", "2.7", path, 1),
                _figure("snapshots/p001_s8_image.png", "2.7", path, index=2),
            ],
        )
    ]
    doc = _doc(pages)
    figure_id = f"{DOC_ID}:p001:f1"
    description = _description(figure_id, "短说明。", "长说明：框图结构。", ["A"])
    chunks, _ = _chunks(doc, {figure_id: description})
    figure = next(chunk for chunk in chunks if chunk.metadata.semantic_type == "figure")
    assert figure.metadata.figure_type == "waveform"
    assert figure.metadata.image_description == "长说明：框图结构。"


def test_promoted_code_is_inline_without_reusing_its_figure_description():
    from app.chunking.code_sidecar import CodeRecord

    path = ["2 章节", "2.8 例程"]
    pages = [
        _page(
            1,
            [
                _heading("2.8 例程", "2.8", path),
                _paragraph("正文 A。", "2.8", path, 1),
                _figure(
                    "snapshots/p001_s9_table.png",
                    "2.8",
                    path,
                    annotations="MOVIA 0x55",
                    hint="table",
                    index=2,
                ),
            ],
        )
    ]
    doc = _doc(pages)
    record = CodeRecord(
        figure_id=f"{DOC_ID}:p001:s1",
        doc_id=DOC_ID,
        page=1,
        image_ref="snapshots/p001_s9_table.png",
        status="complete",
        effective_status="complete",
        review_status="",
        duplicate_of=None,
        code_text="MOVIA 0x55",
        source_bbox=(60.0, 100.0, 520.0, 300.0),
        issues=(),
        lines=(),
    )
    figure_id = f"{DOC_ID}:p001:f1"
    description = _description(figure_id, "短说明。", "长说明：汇编示例。", ["MOVIA"])
    description["diagram_type"] = "code_listing"
    units = build_semantic_units(
        doc, code_sidecar={DOC_ID: [record]}, figure_descriptions={figure_id: description}
    )
    chunks = build_chunks(doc, units)
    code = next(chunk for chunk in chunks if chunk.metadata.code_refs)
    assert code.metadata.semantic_type == "text"
    assert code.metadata.code_refs[0]["figure_id"] == figure_id
    assert code.metadata.code_refs[0]["sidecar_figure_id"] == record.figure_id
    assert "正文 A。" in code.content and "MOVIA 0x55" in code.content
    assert "长说明：汇编示例" not in code.embedding_text
    assert not [c for c in chunks if c.metadata.semantic_type == "figure"]


def test_register_keeps_inline_code_between_its_explanations():
    path = ["3 寄存器", "3.2 CTRL"]
    snapshot = _figure("snapshots/code.png", "3.2", path)
    snapshot.update(figure_type="code", code_text="MOVIA 0x55\nMOVAR CTRL", code_language="asm")
    doc = _doc([_page(1, [
        _heading("3.2 CTRL", "3.2", path), _table("EN", "3.2", path),
        _paragraph("写入前先解锁。", "3.2", path), snapshot,
        _paragraph("写入后等待完成。", "3.2", path),
    ])])
    chunks, _ = _chunks(doc)
    register = next(c for c in chunks if c.metadata.semantic_type == "register")
    for text in (register.content, register.embedding_text):
        assert text.index("写入前") < text.index("MOVIA") < text.index("写入后")
    assert len(register.metadata.code_refs) == 1
    assert register.metadata.atomic is True
    assert not [c for c in chunks if c.metadata.semantic_type in ("figure", "code")]


def test_review_notes_stay_in_metadata_without_retrieval_boilerplate():
    path = ["18 电性参数", "18.5 系统时钟"]
    doc = _doc([_page(1, [
        _heading("18.5 系统时钟", "18.5", path),
        _figure("snapshots/clock.png", "18.5", path),
    ])])
    fid = f"{DOC_ID}:p001:f1"
    description = _description(fid, "频率温度曲线", "图中包含 5V 和 3V 两条曲线。", ["5V", "3V"])
    description["uncertainties"] = ["当前裁图未包含测试条件，需结合原页读取。"]
    chunks, _ = _chunks(doc, {fid: description})
    figure = next(c for c in chunks if c.metadata.semantic_type == "figure")
    assert description["uncertainties"][0] not in figure.embedding_text
    assert "5V 和 3V 两条曲线" in figure.embedding_text
    assert figure.metadata.figure_uncertainties == description["uncertainties"]


def test_figure_keeps_qualifiers_and_distinct_labels_without_repeating_labels():
    path = ["2 章节", "2.3 时序"]
    doc = _doc([_page(1, [
        _heading("2.3 时序", "2.3", path),
        _figure("snapshots/timing.png", "2.3", path),
    ])])
    fid = f"{DOC_ID}:p001:f1"
    full = "AD10 输出在计数归零时翻转；仅表示边沿位置，未说明互补关系。"
    description = _description(fid, "输出时序", full, ["AD10", "AD1", "PWM00A/B", "AD1"])
    chunks, _ = _chunks(doc, {fid: description})
    figure = next(c for c in chunks if c.metadata.semantic_type == "figure")
    assert full in figure.embedding_text
    assert figure.embedding_text.count("AD10") == 1
    assert figure.embedding_text.endswith("AD1、PWM00A/B")
    assert figure.metadata.image_description == full
    assert figure.metadata.product == "DEMO" and figure.metadata.version == "1.0"


def test_removing_code_duplicate_keeps_later_figure_identity():
    from dataclasses import replace

    from app.chunking.code_sidecar import CodeRecord

    path = ["2 章节", "2.9 示例"]
    doc = _doc([_page(1, [
        _heading("2.9 示例", "2.9", path),
        _figure("snapshots/code.png", "2.9", path),
        _figure("snapshots/duplicate.png", "2.9", path),
        _figure("snapshots/timing.png", "2.9", path, bbox=[60, 400, 520, 600]),
    ])])
    code = CodeRecord(
        figure_id=f"{DOC_ID}:p001:s1", doc_id=DOC_ID, page=1,
        image_ref="snapshots/code.png", status="complete", effective_status="complete",
        review_status="", duplicate_of=None, code_text="ORG 0000H", source_bbox=(60, 100, 520, 300),
        issues=(), lines=(),
    )
    duplicate = replace(
        code, image_ref="snapshots/duplicate.png", figure_id=f"{DOC_ID}:p001:s2",
        duplicate_of=code.figure_id, status="no-fit", effective_status="no-fit", code_text="",
    )
    fid = f"{DOC_ID}:p001:f3"
    units = build_semantic_units(doc, {DOC_ID: [code, duplicate]},
                                 {fid: _description(fid, "时序短说明", "时序完整说明", ["PWM"])})
    chunks = build_chunks(doc, units)
    figure = next(c for c in chunks if c.metadata.semantic_type == "figure")
    assert figure.metadata.figure_id == fid
    assert figure.metadata.image_ref == "snapshots/timing.png"
    assert figure.metadata.image_description == "时序完整说明"
    assert "duplicate.png" not in "\n".join(c.content for c in chunks)


def test_split_text_pages_follow_constituents_instead_of_entire_section():
    path = ["2 章节", "2.10 说明"]
    doc = _doc([
        _page(1, [
            _heading("2.10 说明", "2.10", path), _paragraph("前" * 760 + "。", "2.10", path),
        ]),
        _page(2, [_paragraph("后" * 760 + "。", "2.10", path)]),
    ])
    chunks, _ = _chunks(doc)
    texts = [c for c in chunks if c.metadata.semantic_type == "text"]
    assert len(texts) == 2
    assert [c.metadata.source_pages for c in texts] == [[1], [2]]
