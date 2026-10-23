"""Code sidecar -> chunker alignment tests (synthetic documents, no PDFs)."""

from __future__ import annotations

from app.chunking import build_chunks, build_semantic_units
from app.chunking.code_sidecar import CodeRecord, align_pages, load_code_sidecar
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


def _figure(image_ref: str, section_id: str, path: list[str], annotations: str = "") -> dict:
    return {
        "type": "snapshot",
        "hint": "table",
        "image_ref": image_ref,
        "annotations": annotations,
        "bbox": [60.0, 100.0, 520.0, 300.0],
        "section_id": section_id,
        "section_path": path,
        "block_index": 1,
    }


def _record(
    *,
    image_ref: str,
    status: str = "complete",
    effective: str | None = None,
    code_text: str = "MOVIA  0x55\nMOVAR  ADCON0",
    duplicate_of: str | None = None,
    review_status: str = "",
) -> CodeRecord:
    return CodeRecord(
        figure_id=f"{DOC_ID}:p001:s1",
        doc_id=DOC_ID,
        page=1,
        image_ref=image_ref,
        status=status,
        effective_status=effective or status,
        review_status=review_status,
        duplicate_of=duplicate_of,
        code_text=code_text,
        source_bbox=(60.0, 100.0, 520.0, 300.0),
        issues=(),
        lines=(),
    )


def _sections() -> list[dict]:
    return [
        _page(
            1,
            [
                _heading("2.1 示例", "2.1", ["2 章节", "2.1 示例"]),
                _figure(
                    "snapshots/p001_s1_table.png",
                    "2.1",
                    ["2 章节", "2.1 示例"],
                    "MOVIA  0x55\nMOVAR  ADCON0",
                ),
            ],
        )
    ]


def test_complete_record_becomes_inline_code_without_path_in_embedding():
    doc = _doc(_sections())
    sidecar = {DOC_ID: [_record(image_ref="snapshots/p001_s1_table.png")]}
    units = build_semantic_units(doc, code_sidecar=sidecar)
    assert any(block.kind == "code" for unit in units for block in unit.blocks)
    chunks = build_chunks(doc, units)
    code = [chunk for chunk in chunks if chunk.metadata.code_refs]
    assert len(code) == 1
    chunk = code[0]
    assert chunk.metadata.semantic_type == "text"
    assert not any(c.metadata.semantic_type == "code" for c in chunks)
    assert "MOVAR  ADCON0" in chunk.content
    assert "snapshots/" not in chunk.embedding_text
    assert ".png" not in chunk.embedding_text


def test_partial_record_stays_figure_and_is_not_a_code_chunk():
    doc = _doc(_sections())
    sidecar = {DOC_ID: [_record(image_ref="snapshots/p001_s1_table.png", status="partial")]}
    units = build_semantic_units(doc, code_sidecar=sidecar)
    assert not any(unit.unit_type == "code" for unit in units)
    chunks = build_chunks(doc, units)
    assert not [chunk for chunk in chunks if chunk.metadata.semantic_type == "code"]
    assert [chunk for chunk in chunks if chunk.metadata.semantic_type == "figure"]


def test_approved_partial_record_is_used_as_code():
    doc = _doc(_sections())
    sidecar = {
        DOC_ID: [
            _record(
                image_ref="snapshots/p001_s1_table.png",
                status="partial",
                effective="complete",
                review_status="approved",
                code_text="例:定义复位向量\nMOVIA  0x55",
            )
        ]
    }
    units = build_semantic_units(doc, code_sidecar=sidecar)
    chunks = build_chunks(doc, units)
    code = [chunk for chunk in chunks if chunk.metadata.code_refs]
    assert len(code) == 1
    assert code[0].content.splitlines()[1].startswith("例:定义复位向量")


def test_duplicate_record_drops_block_so_no_second_chunk():
    pages = [_page(1, [
        _heading("2.1 示例", "2.1", ["2 章节", "2.1 示例"]),
        _figure("snapshots/p001_s1_table.png", "2.1", ["2 章节", "2.1 示例"], "MOVIA  0x55"),
        _figure("snapshots/p001_s2_table.png", "2.1", ["2 章节", "2.1 示例"], ""),
    ])]
    doc = _doc(pages)
    records = [
        _record(image_ref="snapshots/p001_s1_table.png"),
        _record(
            image_ref="snapshots/p001_s2_table.png",
            status="no-fit",
            duplicate_of=f"{DOC_ID}:p001:s1",
        ),
    ]
    aligned, report = align_pages(pages, records)
    assert report["dropped_duplicates"] == 1
    blocks = aligned[0]["blocks"]
    assert len([block for block in blocks if block["type"] == "snapshot"]) == 1

    chunks = build_chunks(doc, build_semantic_units(doc, code_sidecar={DOC_ID: records}))
    figures = [chunk for chunk in chunks if chunk.metadata.semantic_type == "figure"]
    assert figures == []


def test_without_sidecar_behaviour_is_unchanged():
    doc = _doc(_sections())
    chunks = build_chunks(doc, build_semantic_units(doc))
    assert [chunk for chunk in chunks if chunk.metadata.semantic_type == "figure"]


def test_load_code_sidecar_reads_run_directory(tmp_path):
    doc_dir = tmp_path / DOC_ID
    doc_dir.mkdir()
    record = _record(image_ref="snapshots/p001_s1_table.png")
    (doc_dir / "code.jsonl").write_text(
        __import__("json").dumps(
            {
                "figure_id": record.figure_id,
                "doc_id": record.doc_id,
                "page": record.page,
                "image_ref": record.image_ref,
                "status": record.status,
                "effective_status": record.effective_status,
                "review_status": "",
                "duplicate_of": None,
                "code_text": record.code_text,
                "source_bbox": list(record.source_bbox),
                "issues": [],
                "code_lines": [
                    {
                        "line_index": 0,
                        "text": "MOVIA  0x55",
                        "segments": [
                            {
                                "id": "seg001",
                                "text": "MOVIA",
                                "bbox": [70.0, 110.0, 120.0, 122.0],
                                "overflow_pt": {
                                    "left": 0.0,
                                    "top": 0.0,
                                    "right": 0.0,
                                    "bottom": 0.0,
                                },
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = load_code_sidecar(tmp_path)
    assert loaded[DOC_ID][0].image_ref == "snapshots/p001_s1_table.png"
    assert loaded[DOC_ID][0].is_code
    # the real sidecar writes `code_lines`; provenance must survive loading
    assert len(loaded[DOC_ID][0].lines) == 1
    assert loaded[DOC_ID][0].lines[0]["text"] == "MOVIA  0x55"
    assert loaded[DOC_ID][0].lines[0]["segments"][0]["text"] == "MOVIA"
