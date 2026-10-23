from __future__ import annotations

import asyncio
import hashlib
from uuid import uuid4

import pytest

from app.storage import InvalidUploadError, StorageManager, StorageSecurityError


class FakeUpload:
    def __init__(self, data: bytes, filename: str = "manual.pdf", *, read_size: int = 3):
        self.filename = filename
        self._data = data
        self._offset = 0
        self._read_size = read_size

    async def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._data):
            return b""
        take = self._read_size if size < 0 else min(size, self._read_size)
        chunk = self._data[self._offset : self._offset + take]
        self._offset += len(chunk)
        return chunk


def test_save_upload_streams_hashes_and_uses_random_name(tmp_path):
    manager = StorageManager(tmp_path / "rag-storage", upload_max_bytes=1024, io_chunk_bytes=4)
    payload = b"%PDF-1.7\nsmall offline fixture"
    staging_id = uuid4()

    stored = asyncio.run(
        manager.save_upload(FakeUpload(payload, r"C:\client\secret-name.pdf"), staging_id)
    )

    assert stored.path.read_bytes() == payload
    assert stored.sha256 == hashlib.sha256(payload).hexdigest()
    assert stored.size_bytes == len(payload)
    assert stored.original_filename == "secret-name.pdf"
    assert stored.path.name != stored.original_filename
    assert stored.path == manager.root / "staging" / f"{staging_id}.pdf"
    assert stored.path.is_relative_to(manager.root)
    assert not (stored.path.parent / f".{staging_id}.part").exists()

    manager.cleanup_staging(staging_id)
    assert not stored.path.exists()


@pytest.mark.parametrize(
    ("payload", "limit"),
    [
        (b"not a pdf", 1024),
        (b"%PDF-" + b"x" * 20, 10),
    ],
)
def test_save_upload_rejects_bad_magic_or_size_without_publishing(tmp_path, payload, limit):
    manager = StorageManager(tmp_path / "storage", upload_max_bytes=limit, io_chunk_bytes=2)
    staging_id = uuid4()

    with pytest.raises(InvalidUploadError):
        asyncio.run(manager.save_upload(FakeUpload(payload, "../../untrusted.pdf"), staging_id))

    assert not (manager.root / "staging" / str(staging_id)).exists()


def test_paths_and_image_publication_stay_inside_storage(tmp_path):
    manager = StorageManager(tmp_path / "storage")
    document_id = uuid4()
    chunk_id = uuid4()
    parsed_dir = manager.parsed_dir(document_id)
    source = parsed_dir / "snapshots" / "figure.png"
    source.parent.mkdir()
    source.write_bytes(b"\x89PNG\r\n\x1a\nfixture")

    image_ref = manager.publish_image(document_id, chunk_id, source)
    resolved = manager.resolve_image_ref(image_ref)

    assert image_ref == f"images/{document_id}/{chunk_id}.png"
    assert resolved.read_bytes() == source.read_bytes()
    assert resolved.is_relative_to(manager.root)
    assert manager.chunk_jsonl(document_id).is_relative_to(manager.root)

    for unsafe in ("../outside.png", "../../outside.png", "C:/outside.png", "."):
        with pytest.raises(StorageSecurityError):
            manager.resolve_image_ref(unsafe)


def test_delete_document_assets_only_accepts_uuid_scoped_directories(tmp_path):
    manager = StorageManager(tmp_path / "storage")
    document_id = uuid4()
    other_id = uuid4()

    manager.parsed_dir(document_id)
    manager.chunk_jsonl(document_id).write_text("{}\n", encoding="utf-8")
    manager.parsed_dir(other_id)
    for category in ("uploads", "images"):
        target = manager.root / category / str(document_id)
        target.mkdir(parents=True)
        (target / "artifact").write_bytes(b"x")

    manager.delete_document_assets(document_id)

    assert not (manager.root / "parsed" / str(document_id)).exists()
    assert not (manager.root / "chunks" / str(document_id)).exists()
    assert not (manager.root / "uploads" / str(document_id)).exists()
    assert not (manager.root / "images" / str(document_id)).exists()
    assert (manager.root / "parsed" / str(other_id)).is_dir()

    with pytest.raises(StorageSecurityError):
        manager.delete_document_assets("../../escape")  # type: ignore[arg-type]
