"""Safe local artifact storage for the RAG ingestion service."""

from __future__ import annotations

import hashlib
import inspect
import os
import shutil
from collections.abc import Awaitable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Protocol
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from fastapi import UploadFile
else:

    class UploadFile(Protocol):
        """Runtime-light subset of FastAPI's UploadFile used by this module."""

        filename: str | None

        def read(self, size: int = -1) -> Awaitable[bytes] | bytes: ...


DEFAULT_UPLOAD_MAX_BYTES = 100 * 1024 * 1024
DEFAULT_IO_CHUNK_BYTES = 1024 * 1024
PDF_MAGIC = b"%PDF-"


class StorageError(RuntimeError):
    """Base error for storage operations."""


class InvalidUploadError(StorageError, ValueError):
    """The uploaded stream is not an accepted PDF."""


class StorageSecurityError(StorageError, ValueError):
    """A requested path or deletion target is outside the storage contract."""


@dataclass(frozen=True, slots=True)
class StoredUpload:
    path: Path
    original_filename: str
    sha256: str
    size_bytes: int


class StorageManager:
    """Own all filesystem paths and mutations used by the RAG service.

    Paths returned by this class are absolute, resolved paths below ``root``.
    The caller-provided staging UUID is the random filename, so cleanup remains
    deterministic without ever trusting the client-supplied filename.
    """

    def __init__(
        self,
        storage_dir: str | Path,
        *,
        upload_max_bytes: int = DEFAULT_UPLOAD_MAX_BYTES,
        io_chunk_bytes: int = DEFAULT_IO_CHUNK_BYTES,
    ) -> None:
        if upload_max_bytes <= 0:
            raise ValueError("upload_max_bytes must be positive")
        if io_chunk_bytes <= 0:
            raise ValueError("io_chunk_bytes must be positive")

        root = Path(storage_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir():
            raise StorageError("storage_dir is not a directory")

        self._root = root
        self._upload_max_bytes = upload_max_bytes
        self._io_chunk_bytes = io_chunk_bytes

    @property
    def root(self) -> Path:
        return self._root

    @property
    def storage_dir(self) -> Path:
        """Compatibility alias for callers using the settings field name."""

        return self._root

    async def save_upload(self, upload: UploadFile, staging_id: UUID) -> StoredUpload:
        """Stream a PDF to an atomic, randomly named staging file.

        The SHA-256 digest and byte limit are evaluated while streaming.  A
        failed validation never publishes the ``.pdf`` destination.
        """

        validated_id = self._validated_uuid(staging_id, label="staging_id")
        staging_dir = self._safe_join("staging")
        staging_dir.mkdir(parents=True, exist_ok=True)

        random_name = f"{validated_id}.pdf"
        destination = self._safe_join("staging", random_name)
        temporary = self._safe_join("staging", f".{validated_id}.part")
        original_filename = self._safe_original_filename(getattr(upload, "filename", None))
        digest = hashlib.sha256()
        size_bytes = 0
        prefix = bytearray()
        was_published = False

        try:
            with temporary.open("xb") as output:
                while True:
                    value = upload.read(self._io_chunk_bytes)
                    chunk = await value if inspect.isawaitable(value) else value
                    if not isinstance(chunk, bytes):
                        raise InvalidUploadError("upload.read() must return bytes")
                    if not chunk:
                        break

                    size_bytes += len(chunk)
                    if size_bytes > self._upload_max_bytes:
                        raise InvalidUploadError(
                            f"PDF exceeds the {self._upload_max_bytes}-byte upload limit"
                        )
                    if len(prefix) < len(PDF_MAGIC):
                        needed = len(PDF_MAGIC) - len(prefix)
                        prefix.extend(chunk[:needed])
                    digest.update(chunk)
                    output.write(chunk)

                if bytes(prefix) != PDF_MAGIC:
                    raise InvalidUploadError("uploaded file does not start with %PDF- magic")
                output.flush()
                os.fsync(output.fileno())

            os.replace(temporary, destination)
            was_published = True
            published_path = destination.resolve(strict=True)
            self._assert_contained(published_path)
            return StoredUpload(
                path=published_path,
                original_filename=original_filename,
                sha256=digest.hexdigest(),
                size_bytes=size_bytes,
            )
        except BaseException:
            temporary.unlink(missing_ok=True)
            if was_published:
                destination.unlink(missing_ok=True)
            raise

    def parsed_dir(self, document_id: UUID) -> Path:
        document_dir = self._uuid_dir(
            "parsed", self._validated_uuid(document_id, label="document_id")
        )
        document_dir.mkdir(parents=True, exist_ok=True)
        return document_dir.resolve(strict=True)

    def chunk_jsonl(self, document_id: UUID) -> Path:
        document_dir = self._uuid_dir(
            "chunks", self._validated_uuid(document_id, label="document_id")
        )
        document_dir.mkdir(parents=True, exist_ok=True)
        return self._safe_join("chunks", document_dir.name, "chunks.jsonl")

    def publish_image(self, document_id: UUID, chunk_id: UUID, source: Path) -> str:
        """Atomically copy one parsed image into its canonical public location."""

        valid_document_id = self._validated_uuid(document_id, label="document_id")
        valid_chunk_id = self._validated_uuid(chunk_id, label="chunk_id")
        try:
            source_path = Path(source).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise StorageError("image source does not exist") from exc
        self._assert_contained(source_path)
        if not source_path.is_file():
            raise StorageError("image source is not a regular file")

        image_dir = self._uuid_dir("images", valid_document_id)
        image_dir.mkdir(parents=True, exist_ok=True)
        destination = self._safe_join("images", str(valid_document_id), f"{valid_chunk_id}.png")
        temporary = self._safe_join(
            "images", str(valid_document_id), f".{valid_chunk_id}.{uuid4().hex}.tmp"
        )

        try:
            with source_path.open("rb") as source_file, temporary.open("xb") as output:
                shutil.copyfileobj(source_file, output, length=self._io_chunk_bytes)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

        published = destination.resolve(strict=True)
        self._assert_contained(published)
        return published.relative_to(self._root).as_posix()

    def resolve_image_ref(self, image_ref: str) -> Path:
        """Resolve a storage-relative image reference without allowing traversal."""

        if not isinstance(image_ref, str) or not image_ref.strip():
            raise StorageSecurityError("image_ref must be a non-empty relative path")
        normalized = image_ref.strip().replace("\\", "/")
        pure = PurePosixPath(normalized)
        if (
            not pure.parts
            or normalized in {".", ".."}
            or pure.is_absolute()
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise StorageSecurityError("image_ref must not be absolute or contain traversal")
        # Drive-qualified paths are not absolute according to PurePosixPath.
        if pure.parts and ":" in pure.parts[0]:
            raise StorageSecurityError("image_ref must not contain a drive prefix")
        return self._safe_join(*pure.parts)

    def cleanup_staging(self, staging_id: UUID) -> None:
        validated_id = self._validated_uuid(staging_id, label="staging_id")
        self._safe_join("staging", f"{validated_id}.pdf").unlink(missing_ok=True)
        self._safe_join("staging", f".{validated_id}.part").unlink(missing_ok=True)

    def delete_document_assets(self, document_id: UUID) -> None:
        """Delete only the four validated UUID-scoped document directories."""

        valid_document_id = self._validated_uuid(document_id, label="document_id")
        for category in ("uploads", "parsed", "chunks", "images"):
            self._remove_uuid_directory(self._uuid_dir(category, valid_document_id))

    def _safe_join(self, *parts: str) -> Path:
        candidate = self._root.joinpath(*parts).resolve(strict=False)
        self._assert_contained(candidate)
        return candidate

    def _uuid_dir(self, category: str, value: UUID) -> Path:
        base = self._safe_join(category)
        candidate = base / str(value)
        if candidate.is_symlink():
            raise StorageSecurityError("refusing a symlink UUID directory")
        resolved = candidate.resolve(strict=False)
        self._assert_contained(resolved)
        if resolved.parent != base:
            raise StorageSecurityError("UUID directory is not a direct storage child")
        return resolved

    def _assert_contained(self, path: Path) -> None:
        try:
            path.relative_to(self._root)
        except ValueError as exc:
            raise StorageSecurityError("resolved path escapes RAG_STORAGE_DIR") from exc

    @staticmethod
    def _validated_uuid(value: UUID, *, label: str) -> UUID:
        if isinstance(value, UUID):
            return value
        if isinstance(value, str):
            try:
                return UUID(value)
            except ValueError as exc:
                raise StorageSecurityError(f"{label} must be a valid UUID") from exc
        raise StorageSecurityError(f"{label} must be a valid UUID")

    @staticmethod
    def _safe_original_filename(filename: str | None) -> str:
        if not filename:
            return "upload.pdf"
        name = PurePosixPath(filename.replace("\\", "/")).name.strip()
        return name if name not in {"", ".", ".."} else "upload.pdf"

    @staticmethod
    def _remove_uuid_directory(path: Path) -> None:
        if not path.exists():
            return
        if path.is_symlink() or not path.is_dir():
            raise StorageSecurityError("recursive deletion requires a real UUID directory")
        shutil.rmtree(path)


__all__ = [
    "DEFAULT_UPLOAD_MAX_BYTES",
    "InvalidUploadError",
    "StorageError",
    "StorageManager",
    "StorageSecurityError",
    "StoredUpload",
]
