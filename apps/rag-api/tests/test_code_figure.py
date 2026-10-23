"""Offline R1 coverage for native-text code figures and CODE chunks."""

from __future__ import annotations

from app.chunking import ParsedDocument, build_chunks, build_semantic_units
from app.parser.anchor import Anchor
from app.parser.models import BBox
from app.parser.native_pdf import NativeLine, NativePage, NativeSpan
from app.parser.page import SnapshotBlock
from app.parser.visual import annotate_code_figure, refine_snapshot_anchors


def _native_line(
    text: str,
    line_id: int,
    *,
    x: float = 20.0,
    y: float | None = None,
    font: str = "Courier New",
) -> NativeLine:
    y0 = 20.0 + line_id * 12.0 if y is None else y
    indentation = len(text) - len(text.lstrip(" "))
    visible_x = x + indentation * 5.0
    visible_width = max(len(text.strip()), 1) * 5.0
    bbox = BBox(visible_x, y0, visible_x + visible_width, y0 + 8.0)
    span = NativeSpan(
        text=text,
        bbox=bbox,
        font=font,
        size=9.0,
        block_id=0,
        line_id=line_id,
        span_id=0,
    )
    return NativeLine(spans=[span], block_id=0, line_id=line_id, bbox=bbox)


def _snapshot_anchor() -> Anchor:
    return Anchor(
        anchor_id=1,
        bbox=BBox(10.0, 12.0, 190.0, 115.0),
        pp_label="image",
        pp_order=1,
        kind="VISUAL",
    )


def test_visual_detects_native_code_and_preserves_indentation() -> None:
    source_lines = [
        "int main(void) {",
        "    int value = 0;",
        "    if (value > 0) {",
        "        value--;",
        "    }",
        "}",
    ]
    native = NativePage(
        page_number=1,
        width=200.0,
        height=160.0,
        lines=[_native_line(text, index) for index, text in enumerate(source_lines)],
    )
    anchor = _snapshot_anchor()

    refine_snapshot_anchors(native, [anchor])

    assert anchor.figure_type == "code"
    assert anchor.code_text == "\n".join(source_lines)
    assert anchor.extraction_method == "native"
    assert anchor.extraction_confidence == 1.0

    snapshot = SnapshotBlock(hint="image", bbox=anchor.bbox)
    annotate_code_figure(native, anchor, snapshot)
    assert snapshot.figure_type == "code"
    assert snapshot.code_text == "\n".join(source_lines)


def test_visual_does_not_classify_regular_diagram_labels_as_code() -> None:
    labels = ["程序片段示例", "OSC", "CPU", "RAM", "GPIO"]
    native = NativePage(
        page_number=1,
        width=200.0,
        height=160.0,
        lines=[_native_line(text, index) for index, text in enumerate(labels)],
    )
    anchor = _snapshot_anchor()

    refine_snapshot_anchors(native, [anchor])

    assert anchor.figure_type is None
    assert anchor.code_text is None
    assert anchor.extraction_method is None
    assert anchor.extraction_confidence is None


def test_visual_does_not_treat_multi_column_table_cells_as_indentation() -> None:
    cells = [
        _native_line("[7:0]", 0, x=20.0, y=20.0),
        _native_line("VALUE=0;", 1, x=105.0, y=20.0),
        _native_line("[15:8]", 2, x=20.0, y=40.0),
        _native_line("VALUE=1;", 3, x=105.0, y=40.0),
        _native_line("[23:16]", 4, x=20.0, y=60.0),
        _native_line("VALUE=2;", 5, x=105.0, y=60.0),
    ]
    native = NativePage(page_number=1, width=200.0, height=160.0, lines=cells)
    anchor = _snapshot_anchor()

    refine_snapshot_anchors(native, [anchor])

    assert anchor.figure_type is None
    assert anchor.code_text is None


def test_preclassified_raster_code_is_marked_for_future_transcription() -> None:
    native = NativePage(page_number=1, width=200.0, height=160.0)
    anchor = _snapshot_anchor()
    anchor.figure_type = "code"
    snapshot = SnapshotBlock(hint="image", bbox=anchor.bbox)

    annotate_code_figure(native, anchor, snapshot)

    assert snapshot.figure_type == "code"
    assert snapshot.code_text is None
    assert snapshot.extraction_method == "none"
    assert snapshot.extraction_confidence is None


def _document(snapshot: dict) -> ParsedDocument:
    path = ["5 模数转换器", "5.2 初始化"]
    heading = {
        "type": "text",
        "kind": "heading",
        "text": "5.2 ADC 初始化",
        "section_id": "5.2",
        "section_path": path,
    }
    snapshot = {
        "section_id": "5.2",
        "section_path": path,
        "block_index": 1,
        **snapshot,
    }
    return ParsedDocument(
        doc_id="demo_spec_v1.0",
        product="DEMO",
        version="1.0",
        pdf_name="DEMO_SPEC_V1.0.pdf",
        pages=[
            {
                "page": 12,
                "width": 595.0,
                "height": 842.0,
                "blocks": [heading, snapshot],
            }
        ],
    )


def test_recovered_code_stays_between_its_original_prose() -> None:
    code_text = "#include <stdint.h>\nvoid adc_init(void) {\n    ADC_CTRL = 1;\n}"
    image_ref = "snapshots/p012_s1_code.png"
    doc = _document(
        {
            "type": "snapshot",
            "hint": "C语言程序片段",
            "caption": "ADC 初始化示例",
            "annotations": "配置 ADC 控制寄存器的示例代码。",
            "bbox": [80.0, 180.0, 510.0, 360.0],
            "image_ref": image_ref,
            "figure_type": "code",
            "code_text": code_text,
            "extraction_method": "native",
            "extraction_confidence": 1.0,
        }
    )

    blocks = doc.pages[0]["blocks"]
    common = {"type": "text", "kind": "paragraph", "section_id": "5.2",
              "section_path": ["5 模数转换器", "5.2 初始化"]}
    blocks.insert(1, {**common, "text": "配置 ADC 控制寄存器的示例代码。"})
    blocks.append({**common, "text": "配置完成后开始转换。"})
    units = build_semantic_units(doc)
    assert len(units) == 1 and units[0].unit_type == "text"
    assert [b.kind for b in units[0].blocks] == ["paragraph", "code", "paragraph"]

    chunks = build_chunks(doc, units)
    code_chunks = [chunk for chunk in chunks if chunk.metadata.code_refs]
    assert len(code_chunks) == 1
    chunk = code_chunks[0]
    assert chunk.kind == "TEXT"
    assert chunk.metadata.semantic_type == "text"
    assert "```c\n" + code_text in chunk.content
    assert code_text in chunk.content
    assert code_text in chunk.embedding_text
    for text in (chunk.content, chunk.embedding_text):
        assert text.index("配置 ADC") < text.index(code_text) < text.index("配置完成")
        assert text.count("配置 ADC 控制寄存器") == 1
    ref = chunk.metadata.code_refs[0]
    assert ref["image_ref"] == image_ref and ref["page"] == 12
    assert ref["language"] == "c"
    assert ref["bbox"] == [80.0, 180.0, 510.0, 360.0]
    assert chunk.metadata.atomic is False
    assert chunk.metadata.structured is True
    assert not any(c.metadata.semantic_type == "code" for c in chunks)


def test_code_retrieval_preserves_leading_indent_and_trailing_newline() -> None:
    code_text = "    MOVIA 0x55\n    MOVAR ADCON0\n"
    doc = _document({
        "type": "snapshot", "figure_type": "code", "caption": "写寄存器",
        "description": "按图展示寄存器初始化的详细说明。",
        "code_text": code_text, "code_language": "asm",
        "image_ref": "snapshots/code.png",
    })
    chunks = build_chunks(doc, build_semantic_units(doc))
    code = next(c for c in chunks if c.metadata.code_refs)
    assert code.embedding_text.endswith(code_text)
    assert code_text in code.content
    assert "详细说明" not in code.embedding_text


def test_long_inline_example_is_not_split_or_repeated_as_overlap() -> None:
    code_text = "MOVIA 0x55 ; 保留完整示例\n" * 180
    doc = _document({
        "type": "snapshot", "figure_type": "code", "code_text": code_text,
        "code_language": "asm", "image_ref": "snapshots/long.png",
    })
    common = {"type": "text", "kind": "paragraph", "section_id": "5.2",
              "section_path": ["5 模数转换器", "5.2 初始化"]}
    doc.pages[0]["blocks"].insert(1, {**common, "text": "下面给出完整示例。"})
    doc.pages[0]["blocks"].append({**common, "text": "示例执行完毕。"})
    chunks = build_chunks(doc, build_semantic_units(doc))
    owners = [c for c in chunks if c.metadata.code_refs]
    assert len(owners) == 1 and owners[0].metadata.atomic is True
    assert code_text in owners[0].content and code_text in owners[0].embedding_text
    assert sum(c.embedding_text.count("MOVIA 0x55") for c in chunks) == 180
    assert owners[0].metadata.semantic_type == "text"


def test_untranscribed_code_snapshot_keeps_figure_and_original_image() -> None:
    image_ref = "snapshots/p012_s1_raster.png"
    doc = _document(
        {
            "type": "snapshot",
            "hint": "image",
            "annotations": "程序片段示例",
            "bbox": [80.0, 180.0, 510.0, 360.0],
            "image_ref": image_ref,
            "figure_type": "code",
            "extraction_method": "none",
        }
    )

    units = build_semantic_units(doc)
    assert not any(unit.unit_type == "code" for unit in units)
    figures = [unit for unit in units if unit.unit_type == "figure"]
    assert len(figures) == 1
    assert figures[0].figure_type == "code"

    chunks = build_chunks(doc, units)
    figure = next(chunk for chunk in chunks if chunk.metadata.semantic_type == "figure")
    assert figure.metadata.image_ref == image_ref
    assert figure.metadata.figure_type == "code"
    assert figure.metadata.extraction_method == "none"
    assert figure.metadata.atomic is True
    assert figure.metadata.structured is False
    assert "待转录代码图" in figure.embedding_text
    assert f"![图]({image_ref})" in figure.content
