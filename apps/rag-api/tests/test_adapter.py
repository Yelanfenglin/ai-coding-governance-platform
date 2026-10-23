from __future__ import annotations

import json
from copy import deepcopy
from uuid import UUID, uuid4

import pytest

from app.ingest.adapter import (
    MAX_TEXT_BYTES,
    AuthoritativeDocumentIdentity,
    ChunkKind,
    ChunkMappingError,
    map_jsonl,
    map_records,
)
from app.storage import StorageManager


def _real_format_row(
    product: str,
    semantic_type: str,
    ordinal: int = 1,
    *,
    excluded: bool = False,
) -> dict:
    """A compact fixture with every real field listed by plan section 1.3."""

    version = {"M8P5170": "1.06", "M8P6272": "1.10", "M9F6820": "1.07"}[product]
    content = f"{product} {semantic_type} faithful content"
    embedding_text = f"{product} Version {version} > 2 CPU\n{content}"
    return {
        "doc_id": f"{product.lower()}_spec_v{version}",
        "product": product,
        "version": version,
        "pdf_name": f"{product}_SPEC_V{version}CN.pdf",
        "chunk_id": f"{product.lower()}_spec_v{version}:2:{semantic_type}:{ordinal:03d}",
        "parent_id": f"{product.lower()}_spec_v{version}:2:parent:001",
        "prev_id": "",
        "next_id": "",
        "object_key": "",
        "section_id": "2",
        "section_path": ["2 CPU", "2.1 程序存储器"],
        "heading_level": 2,
        "semantic_type": semantic_type,
        "block_types": ["paragraph"],
        "register": "PCL" if semantic_type == "register" else "",
        "table_id": "table-1" if semantic_type == "table" else "",
        "row_group": ["rows:1-2"] if semantic_type == "table" else [],
        "instruction_category": "",
        "content": content,
        "embedding_text": embedding_text,
        "title": "2.1 程序存储器",
        "image_ref": "",
        "image_description": None,
        "page_start": 8,
        "page_end": 9,
        "source_pages": [8, 9],
        "structured": True,
        "atomic": semantic_type == "register",
        "exclude_from_embedding": excluded,
        "chars": len(content),
        "tokens": 12,
        "parser_version": "0.2.0",
        "chunker_version": "0.1.0",
    }


@pytest.mark.parametrize(
    ("source_product", "semantic_type", "expected_kind"),
    [
        ("M8P5170", "text", ChunkKind.TEXT),
        ("M8P6272", "table", ChunkKind.TABLE),
        ("M9F6820", "register", ChunkKind.MIXED),
    ],
)
def test_real_three_document_shapes_map_with_authoritative_identity(
    source_product, semantic_type, expected_kind
):
    row = _real_format_row(source_product, semantic_type)
    if semantic_type == "table":
        row["table_repr"] = {"headers": ["名称", "说明"], "rows": [["VDD", "电源"]]}
    document_id = uuid4()

    first = map_records(
        document_id,
        [row],
        product="PG-PRODUCT",
        version="9.9",
    )[0]
    replay = map_records(
        document_id,
        [row],
        product="PG-PRODUCT",
        version="9.9",
    )[0]

    assert first.id == replay.id
    assert first.chunk_index == 0
    assert first.embedding_id == f"{document_id}:0"
    assert first.kind == expected_kind
    assert first.section_path == "2 CPU > 2.1 程序存储器"
    assert first.product == "PG-PRODUCT"
    assert first.version == "9.9"
    assert first.chunk_metadata["doc_id"] == str(document_id)
    assert first.chunk_metadata["product"] == "PG-PRODUCT"
    assert first.chunk_metadata["version"] == "9.9"
    assert first.chunk_metadata["source_identity"]["product"] == source_product
    assert first.table_repr == row.get("table_repr")


def test_toc_parent_and_other_excluded_rows_do_not_consume_chunk_indices():
    toc = _real_format_row("M8P5170", "toc", excluded=True)
    toc["embedding_text"] = ""
    first = _real_format_row("M8P5170", "text", 1)
    parent = _real_format_row("M8P5170", "parent", excluded=True)
    parent["embedding_text"] = ""
    explicitly_excluded = _real_format_row("M8P5170", "text", 2, excluded=True)
    second = _real_format_row("M8P5170", "register", 3)
    document_id = uuid4()

    records = map_records(
        document_id,
        [toc, first, parent, explicitly_excluded, second],
        product="M8P5170",
        version="1.06",
    )

    assert [record.chunk_index for record in records] == [0, 1]
    assert [record.embedding_id for record in records] == [
        f"{document_id}:0",
        f"{document_id}:1",
    ]


def test_figure_and_code_images_are_copied_to_canonical_refs(tmp_path):
    storage = StorageManager(tmp_path / "storage")
    document_id = uuid4()
    parsed = storage.parsed_dir(document_id)
    snapshot = parsed / "snapshots" / "p009_s1_image.png"
    snapshot.parent.mkdir()
    snapshot.write_bytes(b"\x89PNG\r\n\x1a\nfixture")

    figure = _real_format_row("M9F6820", "figure", 1)
    figure["image_ref"] = "snapshots/p009_s1_image.png"
    code = _real_format_row("M9F6820", "code", 2)
    code["image_ref"] = "snapshots/p009_s1_image.png"
    jsonl = storage.chunk_jsonl(document_id)
    jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in (figure, code)),
        encoding="utf-8",
    )

    records = map_jsonl(
        jsonl,
        AuthoritativeDocumentIdentity(document_id, "M9F6820", "1.07"),
        storage=storage,
        image_base_dir=parsed,
    )

    assert [record.kind for record in records] == [ChunkKind.VISUAL, ChunkKind.CODE]
    for record in records:
        assert record.image_ref == f"images/{document_id}/{record.id}.png"
        assert storage.resolve_image_ref(record.image_ref).read_bytes() == snapshot.read_bytes()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.update(content=" "), "content must not be empty"),
        (lambda row: row.update(page_start=0), "page_start must be a positive integer"),
        (lambda row: row.update(page_end=7), "page_end must be"),
        (lambda row: row.update(content="界" * (MAX_TEXT_BYTES // 3 + 1)), "UTF-8 bytes"),
    ],
)
def test_adapter_rejects_empty_invalid_pages_and_oversize(mutate, message):
    row = deepcopy(_real_format_row("M8P6272", "text"))
    mutate(row)

    with pytest.raises(ChunkMappingError, match=message):
        map_records(uuid4(), [row], product="M8P6272", version="1.10")


def test_image_ref_traversal_is_rejected(tmp_path):
    storage = StorageManager(tmp_path / "storage")
    document_id = uuid4()
    parsed = storage.parsed_dir(document_id)
    row = _real_format_row("M9F6820", "figure")
    row["image_ref"] = "../outside.png"

    with pytest.raises(ChunkMappingError, match="path traversal"):
        map_records(
            document_id,
            [row],
            product="M9F6820",
            version="1.07",
            storage=storage,
            image_base_dir=parsed,
        )


def test_generated_chunk_id_is_a_uuid():
    record = map_records(
        uuid4(),
        [_real_format_row("M8P5170", "text")],
        product="M8P5170",
        version="1.06",
    )[0]
    assert UUID(str(record.id)) == record.id
