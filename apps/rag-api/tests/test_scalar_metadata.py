"""Behavioural tests for the v2 scalar metadata contract.

These cover the two gaps found in the `rag_specs_v2` ten-sample acceptance:
multi-value package codes were truncated to the first hit, and live ingestion
left all three additive scalar fields empty.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from app.ingest.adapter import ChunkKind, PersistenceChunk
from app.ingest.pipeline import IngestPipeline
from app.scalar_metadata import (
    PACKAGE_FIELD_MAX_COUNT,
    ScalarMetadata,
    extract_packages,
    normalize_register,
    normalize_section_id,
    package_matches,
    package_mentioned_in,
    packages_field_value,
    parse_package_field,
    scalar_metadata,
)
from app.vectordb.milvus_store import _require_packages


def test_extracts_every_package_in_first_appearance_order() -> None:
    packages = extract_packages("可选封装：SOP16、SOP20、TSSOP20，另见 SOP16 尺寸图")
    assert packages == ("SOP16", "SOP20", "TSSOP20")


def test_canonicalises_suffix_and_case_variants() -> None:
    assert extract_packages("SOP-8 与 sop 8") == ("SOP8",)
    assert extract_packages("SOT23-6") == ("SOT236",)
    assert packages_field_value(("sop-16", " TSSOP 20 ")) == "SOP16,TSSOP20"


def test_does_not_match_a_longer_package_code() -> None:
    # "SSOP28" must never yield "SOP28": that was the original substring bug.
    assert extract_packages("SSOP28 外形图") == ("SSOP28",)
    assert package_matches("SSOP28", "SOP28") is False
    assert package_mentioned_in("SSOP28 外形图", "SOP28") is False
    assert package_mentioned_in("SOP16 外形图", "SOP16") is True


def test_does_not_absorb_a_trailing_measurement() -> None:
    # "SOP8 1.27mm" was previously extracted as "SOP81"; only a hyphen may join
    # a second numeric group.
    assert extract_packages("SOP8 1.27mm 焊盘间距") == ("SOP8",)
    assert extract_packages("DFN8 1.27 尺寸") == ("DFN8",)


def test_ignores_unrelated_part_and_chunk_identifiers() -> None:
    assert extract_packages("M8P515 第 5.3 节，SOT23 焊盘") == ("SOT23",)


def test_membership_is_a_set_not_a_prefix() -> None:
    stored = extract_packages("SOP16 / SOP20 / TSSOP20")
    assert stored == ("SOP16", "SOP20", "TSSOP20")
    # Milvus reads packages back as an ARRAY list; both forms must match.
    assert package_matches(list(stored), "SOP20") is True
    assert package_matches(packages_field_value(stored), "TSSOP20") is True
    assert package_matches(stored, "SOP2") is False
    assert parse_package_field("SOP16,SOP20,TSSOP20") == ("SOP16", "SOP20", "TSSOP20")


def test_field_capacity_drops_whole_tokens_only() -> None:
    codes = " ".join(f"SOP{index}" for index in range(1, 41))
    tokens = extract_packages(codes)
    assert len(tokens) == PACKAGE_FIELD_MAX_COUNT
    assert tokens[:3] == ("SOP1", "SOP2", "SOP3")
    # Matching is unbounded: the capacity limit must not hide a real code.
    assert "SOP40" in extract_packages(codes, max_tokens=None)


def test_overlong_code_is_dropped_not_truncated() -> None:
    oversized = "SOP" + "1" * 40
    assert extract_packages(oversized) == ()


def test_register_and_section_id_normalisation() -> None:
    assert normalize_register(" pwm1cr ") == "PWM1CR"
    assert normalize_register(None) == ""
    assert normalize_section_id("8.7", section_path="8 定时器 > 8.7 PWM1CR") == "8.7"
    assert normalize_section_id("", section_path="5 端口 > 5.3 PUA") == "5.3"
    assert normalize_section_id(None, section_path="封装") == ""


def test_scalar_metadata_uses_chunk_metadata_then_text_evidence() -> None:
    metadata = scalar_metadata(
        chunk_metadata={"section_id": "8.7", "register": "pwm1cr"},
        text="PWM1CR 控制寄存器，封装 SOP16 / SOP20",
        section_path="8 定时器 > 8.7 PWM1CR",
    )
    assert metadata == ScalarMetadata(
        section_id="8.7", register="PWM1CR", packages=("SOP16", "SOP20")
    )
    assert metadata.as_fields() == {
        "section_id": "8.7",
        "register": "PWM1CR",
        "packages": ["SOP16", "SOP20"],
    }


def test_store_serialises_the_whole_package_set_as_a_list() -> None:
    assert _require_packages(("SOP16", "SOP20", "TSSOP20")) == [
        "SOP16",
        "SOP20",
        "TSSOP20",
    ]
    assert _require_packages(()) == []
    with pytest.raises(TypeError):
        _require_packages(16)
    with pytest.raises(TypeError):
        # The delimited legacy form must be parsed before reaching the store.
        _require_packages("SOP16,SOP20")
    with pytest.raises(ValueError):
        _require_packages(tuple(f"SOP{index}" for index in range(1, 10)))


class _FakeProvider:
    dimension = 1024
    model_id = "text-embedding-v3"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.01] * 1024 for _ in texts]


def _chunk(**overrides: object) -> PersistenceChunk:
    document_id = uuid4()
    values: dict[str, object] = {
        "id": uuid4(),
        "document_id": document_id,
        "chunk_index": 0,
        "embedding_id": f"{document_id}:0",
        "logical_chunk_id": "logical-0",
        "product": "M9F6820",
        "version": "1.00",
        "kind": ChunkKind.MIXED,
        "semantic_type": "text",
        "section_path": "19 封装 > 19.1 封装尺寸",
        "section_title": "封装尺寸",
        "page_start": 71,
        "page_end": 71,
        "text": "本芯片提供 SOP16、SOP20、TSSOP20 三种封装。",
        "embedding_text": "19.1 封装尺寸 本芯片提供 SOP16、SOP20、TSSOP20 三种封装。",
        "table_repr": None,
        "image_ref": None,
        "chunk_metadata": {"section_id": "19.1", "register": ""},
    }
    values.update(overrides)
    return PersistenceChunk(**values)  # type: ignore[arg-type]


def test_embed_fills_all_three_scalar_fields() -> None:
    pipeline = IngestPipeline(
        _FakeProvider(),
        store=object(),
        repo=object(),
        storage=object(),
    )

    async def run():
        try:
            return await pipeline.embed([_chunk()])
        finally:
            await pipeline.aclose()

    records = asyncio.run(run())

    assert len(records) == 1
    record = records[0]
    assert record.section_id == "19.1"
    assert record.register == ""
    assert record.packages == ("SOP16", "SOP20", "TSSOP20")
    assert record.embedding_model_id == "text-embedding-v3:1024"


def test_embed_keeps_register_and_falls_back_to_section_path() -> None:
    pipeline = IngestPipeline(
        _FakeProvider(),
        store=object(),
        repo=object(),
        storage=object(),
    )
    chunk = _chunk(
        text="PWM1CR 位定义",
        embedding_text="8.7 PWM1CR 位定义",
        section_path="8 定时器 > 8.7 PWM1CR",
        chunk_metadata={"section_id": "", "register": " pwm1cr "},
    )

    async def run():
        try:
            return await pipeline.embed([chunk])
        finally:
            await pipeline.aclose()

    record = asyncio.run(run())[0]

    assert record.section_id == "8.7"
    assert record.register == "PWM1CR"
    assert record.packages == ()
