from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from app.ingest.pipeline import (
    DocumentStatus,
    IngestCommand,
    IngestInvariantError,
    IngestPipeline,
)
from app.storage import StorageManager


@dataclass
class FakeDocument:
    id: UUID
    product: str
    version: str
    status: DocumentStatus
    current: bool = False
    parse_error: str | None = None


class FakeParser:
    def __init__(self, chunk_count: int = 3, *, missing_snapshot: bool = False):
        self.chunk_count = chunk_count
        self.missing_snapshot = missing_snapshot
        self.calls = 0

    def __call__(self, pdf_path: Path, layout_path: Path, out_dir: Path) -> dict:
        self.calls += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        blocks = []
        for index in range(1, self.chunk_count + 1):
            section = str(index)
            section_path = [f"{section} Section {section}"]
            blocks.extend(
                [
                    {
                        "type": "text",
                        "kind": "heading",
                        "text": section_path[0],
                        "section_id": section,
                        "section_path": section_path,
                    },
                    {
                        "type": "text",
                        "kind": "paragraph",
                        "text": f"canonical paragraph {index}",
                        "section_id": section,
                        "section_path": section_path,
                    },
                ]
            )
        pages = [{"page": 1, "blocks": blocks}]
        summary = {
            "pdf": Path(pdf_path).name,
            "layout": Path(layout_path).name,
            "pages": 1,
            "quality": {
                "orphan_text_lines": 0,
                "duplicate_owned_lines": 0,
                "missing_snapshot_files": ["missing.png"] if self.missing_snapshot else [],
            },
        }
        (out_dir / "pages.json").write_text(
            json.dumps(pages, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False), encoding="utf-8"
        )
        return summary


class FakeProvider:
    dimension = 1024
    model_id = "fake-embedding"
    fingerprint = "fake-embedding:1024"

    def __init__(self, *, non_finite: bool = False):
        self.calls: list[list[str]] = []
        self.non_finite = non_finite

    async def embed(self, texts):
        self.calls.append(list(texts))
        vectors = []
        for _ in texts:
            vector = [0.0] * self.dimension
            vector[0] = float("nan") if self.non_finite else 1.0
            vectors.append(vector)
        return vectors


class FakeStore:
    def __init__(self, *, reported_delta: int = 0):
        self.records: dict[str, object] = {}
        self.reported_delta = reported_delta
        self.upsert_batches: list[list[str]] = []
        self.delete_calls: list[UUID] = []

    async def upsert(self, records):
        batch = list(records)
        self.upsert_batches.append([record.id for record in batch])
        for record in batch:
            self.records[record.id] = record
        return len(batch)

    async def count_document(self, document_id):
        actual = sum(
            1 for record in self.records.values() if record.document_id == document_id
        )
        return actual + self.reported_delta

    async def delete_document(self, document_id):
        self.delete_calls.append(document_id)
        self.records = {
            key: record
            for key, record in self.records.items()
            if record.document_id != document_id
        }


class FakeTransaction:
    def __init__(self, repo: FakeRepo):
        self.repo = repo
        self.snapshot = None

    async def __aenter__(self):
        self.snapshot = (deepcopy(self.repo.documents), deepcopy(self.repo.chunks))
        self.repo.activation_in_transaction = False
        self.repo.events.append("transaction:enter")
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        commit_failure = self.repo.fail_commit and self.repo.activation_in_transaction
        should_rollback = exc_type is not None or commit_failure
        if should_rollback:
            self.repo.documents, self.repo.chunks = self.snapshot
            self.repo.events.append("transaction:rollback")
        else:
            self.repo.events.append("transaction:commit")
        if exc_type is None and commit_failure:
            raise RuntimeError("simulated commit failure")
        return False


class FakeRepo:
    def __init__(self, documents, *, fail_commit: bool = False):
        self.documents = {document.id: document for document in documents}
        self.chunks: dict[UUID, list[object]] = {}
        self.fail_commit = fail_commit
        self.activation_in_transaction = False
        self.events: list[str] = []

    async def get_document(self, document_id, *, for_update=False):
        del for_update
        return self.documents.get(document_id)

    async def set_status(self, document_id, status, *, error=None):
        document = self.documents[document_id]
        document.status = status
        document.parse_error = error
        self.events.append(f"status:{status.value}")

    def transaction(self):
        return FakeTransaction(self)

    async def lock_product(self, product):
        self.events.append(f"lock:{product}")

    async def replace_chunks(self, document_id, records):
        self.chunks[document_id] = list(records)
        self.events.append("replace_chunks")

    async def activate_candidate(self, document_id, *, expected_vector_count):
        candidate = self.documents[document_id]
        if len(self.chunks.get(document_id, [])) != expected_vector_count:
            raise AssertionError("fake repository count mismatch")
        for other in self.documents.values():
            if other.id != document_id and other.product == candidate.product and other.current:
                other.current = False
                other.status = DocumentStatus.SUPERSEDED
        candidate.current = True
        candidate.status = DocumentStatus.PARSED
        self.activation_in_transaction = True
        self.events.append("activate")
        return []

    async def count_chunks(self, document_id):
        return len(self.chunks.get(document_id, []))


def _command(tmp_path: Path, document_id: UUID) -> IngestCommand:
    pdf = tmp_path / f"{document_id}.pdf"
    layout = tmp_path / f"{document_id}.layout.json"
    pdf.write_bytes(b"%PDF-1.7\npipeline fixture")
    layout.write_text("{}", encoding="utf-8")
    return IngestCommand(
        document_id=document_id,
        pdf_path=pdf,
        layout_path=layout,
        product="FILENAME-GUESS",
        version="0.0",
        language="zh-CN",
        title="Offline fixture",
    )


def _run_and_close(pipeline: IngestPipeline, command: IngestCommand):
    async def execute():
        try:
            return await pipeline.run(command)
        finally:
            await pipeline.aclose()

    return asyncio.run(execute())


def test_pipeline_batches_and_uses_pg_identity_for_audit_and_persistence(tmp_path):
    document_id = uuid4()
    old_document_id = uuid4()
    candidate = FakeDocument(
        document_id, "PG-M8P6272", "1.10", DocumentStatus.PENDING, current=False
    )
    old = FakeDocument(
        old_document_id, "PG-M8P6272", "1.09", DocumentStatus.PARSED, current=True
    )
    repo = FakeRepo([candidate, old])
    provider = FakeProvider()
    store = FakeStore()
    store.records["old:0"] = SimpleNamespace(document_id=old_document_id)
    parser = FakeParser(chunk_count=3)
    storage = StorageManager(tmp_path / "storage")
    pipeline = IngestPipeline(
        provider,
        store,
        repo,
        storage,
        embedding_batch_size=2,
        upsert_batch_size=2,
        parse_document_fn=parser,
    )

    result = _run_and_close(pipeline, _command(tmp_path, document_id))

    assert result.document_id == document_id
    assert result.chunk_count == result.vector_count == 3
    assert result.status == DocumentStatus.PARSED
    assert [len(batch) for batch in provider.calls] == [2, 1]
    assert [len(batch) for batch in store.upsert_batches] == [2, 1]
    assert candidate.status == DocumentStatus.PARSED
    assert candidate.current is True
    assert old.status == DocumentStatus.SUPERSEDED
    assert old.current is False
    assert "old:0" in store.records

    records = repo.chunks[document_id]
    assert all(record.product == "PG-M8P6272" for record in records)
    assert all(record.version == "1.10" for record in records)
    assert all(record.chunk_metadata["doc_id"] == str(document_id) for record in records)
    assert repo.events.index("lock:PG-M8P6272") < repo.events.index("replace_chunks")
    assert repo.events.index("replace_chunks") < repo.events.index("activate")

    audit_rows = [
        json.loads(line)
        for line in storage.chunk_jsonl(document_id).read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["doc_id"] == str(document_id) for row in audit_rows)
    assert all(row["product"] == "PG-M8P6272" for row in audit_rows)
    assert all(row["version"] == "1.10" for row in audit_rows)


def test_rerun_of_parsed_document_is_idempotent(tmp_path):
    document_id = uuid4()
    candidate = FakeDocument(
        document_id, "M8P5170", "1.06", DocumentStatus.PENDING, current=False
    )
    repo = FakeRepo([candidate])
    provider = FakeProvider()
    store = FakeStore()
    parser = FakeParser(chunk_count=2)
    pipeline = IngestPipeline(
        provider,
        store,
        repo,
        StorageManager(tmp_path / "storage"),
        parse_document_fn=parser,
    )
    command = _command(tmp_path, document_id)

    async def execute_twice():
        try:
            return await pipeline.run(command), await pipeline.run(command)
        finally:
            await pipeline.aclose()

    first, second = asyncio.run(execute_twice())

    assert first == second
    assert parser.calls == 1
    assert len(provider.calls) == 1
    assert len(store.upsert_batches) == 1
    assert len(repo.chunks[document_id]) == 2
    assert len({record.id for record in repo.chunks[document_id]}) == 2


def test_success_cleans_the_flat_staging_upload(tmp_path):
    document_id = uuid4()
    staging_id = uuid4()
    candidate = FakeDocument(
        document_id, "M8P5170", "1.06", DocumentStatus.PENDING, current=False
    )
    storage = StorageManager(tmp_path / "storage")
    staged_pdf = storage.root / "staging" / f"{staging_id}.pdf"
    staged_pdf.parent.mkdir()
    staged_pdf.write_bytes(b"%PDF-1.7\nstaged pipeline fixture")
    command = replace(_command(tmp_path, document_id), pdf_path=staged_pdf)
    pipeline = IngestPipeline(
        FakeProvider(),
        FakeStore(),
        FakeRepo([candidate]),
        storage,
        parse_document_fn=FakeParser(chunk_count=1),
    )

    result = _run_and_close(pipeline, command)

    assert result.status == DocumentStatus.PARSED
    assert not staged_pdf.exists()


def test_count_gate_failure_compensates_and_never_enters_activation(tmp_path):
    document_id = uuid4()
    candidate = FakeDocument(
        document_id, "M8P6272", "1.10", DocumentStatus.PENDING, current=False
    )
    repo = FakeRepo([candidate])
    store = FakeStore(reported_delta=1)
    storage = StorageManager(tmp_path / "storage")
    pipeline = IngestPipeline(
        FakeProvider(),
        store,
        repo,
        storage,
        parse_document_fn=FakeParser(chunk_count=2),
    )

    with pytest.raises(IngestInvariantError, match="count gate"):
        _run_and_close(pipeline, _command(tmp_path, document_id))

    assert candidate.status == DocumentStatus.FAILED
    assert candidate.current is False
    assert "replace_chunks" not in repo.events
    assert not any(
        record.document_id == document_id for record in store.records.values()
    )
    assert store.delete_calls.count(document_id) == 2
    assert not (storage.root / "parsed" / str(document_id)).exists()
    assert not (storage.root / "chunks" / str(document_id)).exists()


def test_activation_commit_failure_rolls_back_old_current_and_compensates(tmp_path):
    document_id = uuid4()
    old_document_id = uuid4()
    candidate = FakeDocument(
        document_id, "M9F6820", "1.07", DocumentStatus.PENDING, current=False
    )
    old = FakeDocument(
        old_document_id, "M9F6820", "1.06", DocumentStatus.PARSED, current=True
    )
    repo = FakeRepo([candidate, old], fail_commit=True)
    store = FakeStore()
    pipeline = IngestPipeline(
        FakeProvider(),
        store,
        repo,
        StorageManager(tmp_path / "storage"),
        parse_document_fn=FakeParser(chunk_count=1),
    )

    with pytest.raises(RuntimeError, match="commit failure"):
        _run_and_close(pipeline, _command(tmp_path, document_id))

    rolled_back_old = repo.documents[old_document_id]
    failed_candidate = repo.documents[document_id]
    assert rolled_back_old.status == DocumentStatus.PARSED
    assert rolled_back_old.current is True
    assert failed_candidate.status == DocumentStatus.FAILED
    assert failed_candidate.current is False
    assert document_id not in repo.chunks
    assert not any(
        record.document_id == document_id for record in store.records.values()
    )
    assert "transaction:rollback" in repo.events
    assert failed_candidate.parse_error == (
        "ingest failed during Milvus stage / PostgreSQL activation: RuntimeError"
    )


def test_summary_and_embedding_invariants_fail_before_activation(tmp_path):
    for parser, provider, expected in (
        (FakeParser(missing_snapshot=True), FakeProvider(), "missing snapshot"),
        (FakeParser(), FakeProvider(non_finite=True), "non-finite"),
    ):
        document_id = uuid4()
        candidate = FakeDocument(
            document_id, "M8P5170", "1.06", DocumentStatus.PENDING, current=False
        )
        repo = FakeRepo([candidate])
        pipeline = IngestPipeline(
            provider,
            FakeStore(),
            repo,
            StorageManager(tmp_path / str(document_id)),
            parse_document_fn=parser,
        )

        with pytest.raises(IngestInvariantError, match=expected):
            _run_and_close(pipeline, _command(tmp_path, document_id))

        assert repo.documents[document_id].status == DocumentStatus.FAILED
        assert "replace_chunks" not in repo.events
