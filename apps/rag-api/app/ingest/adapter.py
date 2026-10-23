"""Map the audited ``chunks.jsonl`` format to persistence records."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID, uuid5

from app.storage import StorageManager, StorageSecurityError

MAX_PRODUCT_BYTES = 64
MAX_VERSION_BYTES = 64
MAX_LOGICAL_ID_BYTES = 512
MAX_SECTION_PATH_BYTES = 1024
MAX_SECTION_TITLE_BYTES = 512
MAX_IMAGE_REF_BYTES = 2048
MAX_TEXT_BYTES = 16_384
MAX_EMBEDDING_TEXT_BYTES = 16_384


class ChunkMappingError(ValueError):
    """A JSONL row cannot be represented by the frozen persistence contract."""


class ChunkKind(StrEnum):
    TEXT = "TEXT"
    TABLE = "TABLE"
    VISUAL = "VISUAL"
    CODE = "CODE"
    MIXED = "MIXED"


_KIND_BY_SEMANTIC_TYPE = {
    "text": ChunkKind.TEXT,
    "table": ChunkKind.TABLE,
    "figure": ChunkKind.VISUAL,
    "code": ChunkKind.CODE,
    "register": ChunkKind.MIXED,
}
_NON_PERSISTED_TYPES = frozenset({"toc", "parent"})


@dataclass(frozen=True, slots=True)
class AuthoritativeDocumentIdentity:
    """Identity loaded from PostgreSQL (never inferred from the filename)."""

    document_id: UUID
    product: str
    version: str

    def __post_init__(self) -> None:
        try:
            document_id = (
                self.document_id
                if isinstance(self.document_id, UUID)
                else UUID(str(self.document_id))
            )
        except (TypeError, ValueError, AttributeError) as exc:
            raise ChunkMappingError("document_id must be a valid UUID") from exc
        object.__setattr__(self, "document_id", document_id)
        _validate_nonempty_string("product", self.product, MAX_PRODUCT_BYTES)
        _validate_nonempty_string("version", self.version, MAX_VERSION_BYTES)


@dataclass(frozen=True, slots=True)
class PersistenceChunk:
    """Canonical one-to-one PostgreSQL/Milvus chunk input."""

    id: UUID
    document_id: UUID
    chunk_index: int
    embedding_id: str
    logical_chunk_id: str
    product: str
    version: str
    kind: ChunkKind
    semantic_type: str
    section_path: str
    section_title: str
    page_start: int
    page_end: int
    text: str
    embedding_text: str
    table_repr: Any | None
    image_ref: str | None
    chunk_metadata: dict[str, Any]

    @property
    def chunk_id(self) -> UUID:
        """Alias used by some repository DTOs."""

        return self.id

    @property
    def content(self) -> str:
        return self.text

    @property
    def metadata(self) -> dict[str, Any]:
        """Read-only-name alias; ORM models should use ``chunk_metadata``."""

        return self.chunk_metadata

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL strictly enough to report a stable, actionable line number."""

    source = Path(path)
    rows: list[dict[str, Any]] = []
    try:
        with source.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line, parse_constant=_reject_json_constant)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ChunkMappingError(
                        f"invalid JSON object at {source.name}:{line_number}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ChunkMappingError(
                        f"JSONL row at {source.name}:{line_number} must be an object"
                    )
                rows.append(value)
    except UnicodeDecodeError as exc:
        raise ChunkMappingError(f"{source.name} is not valid UTF-8") from exc
    return rows


def map_jsonl(
    path: str | Path,
    identity: AuthoritativeDocumentIdentity,
    *,
    storage: StorageManager | None = None,
    image_base_dir: str | Path | None = None,
) -> list[PersistenceChunk]:
    return map_chunk_records(
        load_jsonl(path),
        identity,
        storage=storage,
        image_base_dir=image_base_dir,
    )


def map_records(
    document_id: UUID,
    chunks: Iterable[Mapping[str, Any] | Any],
    *,
    product: str,
    version: str,
    storage: StorageManager | None = None,
    image_base_dir: str | Path | None = None,
) -> list[PersistenceChunk]:
    """Plan-compatible adapter entry using explicit authoritative fields."""

    return map_chunk_records(
        chunks,
        AuthoritativeDocumentIdentity(document_id, product, version),
        storage=storage,
        image_base_dir=image_base_dir,
    )


def map_chunk_records(
    chunks: Iterable[Mapping[str, Any] | Any],
    identity: AuthoritativeDocumentIdentity,
    *,
    storage: StorageManager | None = None,
    image_base_dir: str | Path | None = None,
) -> list[PersistenceChunk]:
    """Filter non-indexed rows and deterministically enumerate Level-1 rows."""

    records: list[PersistenceChunk] = []
    logical_ids: set[str] = set()
    embedding_ids: set[str] = set()
    persistence_ids: set[UUID] = set()

    for source_ordinal, item in enumerate(chunks):
        row = _as_mapping(item, source_ordinal)
        semantic_type = _required_string(row, "semantic_type").strip().lower()
        excluded = _required_bool(row, "exclude_from_embedding")

        # TOC/parent are Level-2/audit artifacts.  Exclude flags on any future
        # semantic type remain authoritative as well.
        if semantic_type in _NON_PERSISTED_TYPES or excluded:
            continue
        try:
            kind = _KIND_BY_SEMANTIC_TYPE[semantic_type]
        except KeyError as exc:
            raise ChunkMappingError(f"unsupported semantic_type: {semantic_type!r}") from exc

        chunk_index = len(records)
        logical_chunk_id = _required_string(row, "chunk_id")
        _validate_nonempty_string(
            "chunk_id", logical_chunk_id, MAX_LOGICAL_ID_BYTES, row_index=source_ordinal
        )
        if logical_chunk_id in logical_ids:
            raise ChunkMappingError(f"duplicate chunk_id: {logical_chunk_id}")

        text = _required_string(row, "content")
        embedding_text = _required_string(row, "embedding_text")
        _validate_nonempty_string(
            "content", text, MAX_TEXT_BYTES, row_index=source_ordinal
        )
        _validate_nonempty_string(
            "embedding_text",
            embedding_text,
            MAX_EMBEDDING_TEXT_BYTES,
            row_index=source_ordinal,
        )

        page_start = _required_positive_int(row, "page_start", source_ordinal)
        page_end = _required_positive_int(row, "page_end", source_ordinal)
        if page_end < page_start:
            raise ChunkMappingError(
                f"row {source_ordinal}: page_end must be greater than or equal to page_start"
            )
        source_pages = _validate_source_pages(
            row.get("source_pages"), page_start, page_end, source_ordinal
        )

        section_parts = _section_parts(row.get("section_path"), source_ordinal)
        section_path = " > ".join(section_parts)
        _validate_string_limit("section_path", section_path, MAX_SECTION_PATH_BYTES)
        section_title = _optional_string(row, "title")
        _validate_string_limit("title", section_title, MAX_SECTION_TITLE_BYTES)

        persistence_id = uuid5(identity.document_id, f"chunk:{chunk_index}")
        embedding_id = f"{identity.document_id}:{chunk_index}"
        if persistence_id in persistence_ids or embedding_id in embedding_ids:
            raise ChunkMappingError("generated persistence IDs are not unique")

        image_ref = _map_image_ref(
            row.get("image_ref"),
            identity=identity,
            persistence_id=persistence_id,
            storage=storage,
            image_base_dir=image_base_dir,
            row_index=source_ordinal,
        )
        table_repr = row.get("table_repr")
        _validate_table_repr(table_repr, source_ordinal)

        metadata = dict(row)
        metadata["source_identity"] = {
            "doc_id": row.get("doc_id"),
            "product": row.get("product"),
            "version": row.get("version"),
        }
        # The canonical fields always win over filename-derived JSONL values.
        metadata.update(
            {
                "doc_id": str(identity.document_id),
                "product": identity.product,
                "version": identity.version,
                "chunk_index": chunk_index,
                "persistence_chunk_id": str(persistence_id),
                "embedding_id": embedding_id,
                "kind": kind.value,
                "semantic_type": semantic_type,
                "section_path": section_parts,
                "source_pages": source_pages,
                "image_ref": image_ref or "",
                "table_repr": table_repr,
            }
        )
        _validate_json_value(metadata, source_ordinal, "chunk_metadata")

        records.append(
            PersistenceChunk(
                id=persistence_id,
                document_id=identity.document_id,
                chunk_index=chunk_index,
                embedding_id=embedding_id,
                logical_chunk_id=logical_chunk_id,
                product=identity.product,
                version=identity.version,
                kind=kind,
                semantic_type=semantic_type,
                section_path=section_path,
                section_title=section_title,
                page_start=page_start,
                page_end=page_end,
                text=text,
                embedding_text=embedding_text,
                table_repr=table_repr,
                image_ref=image_ref,
                chunk_metadata=metadata,
            )
        )
        logical_ids.add(logical_chunk_id)
        persistence_ids.add(persistence_id)
        embedding_ids.add(embedding_id)

    if not records:
        raise ChunkMappingError("document has no embedding-eligible Level-1 chunks")
    return records


def _as_mapping(item: Mapping[str, Any] | Any, row_index: int) -> dict[str, Any]:
    if isinstance(item, Mapping):
        return dict(item)
    to_dict = getattr(item, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return dict(value)
    raise ChunkMappingError(f"row {row_index}: chunk must be a mapping or expose to_dict()")


def _map_image_ref(
    value: Any,
    *,
    identity: AuthoritativeDocumentIdentity,
    persistence_id: UUID,
    storage: StorageManager | None,
    image_base_dir: str | Path | None,
    row_index: int,
) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ChunkMappingError(f"row {row_index}: image_ref must be a string")
    normalized = value.strip().replace("\\", "/")
    _validate_nonempty_string("image_ref", normalized, MAX_IMAGE_REF_BYTES, row_index=row_index)
    pure = PurePosixPath(normalized)
    if (
        not pure.parts
        or normalized in {".", ".."}
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ChunkMappingError(f"row {row_index}: image_ref contains path traversal")
    if pure.parts and ":" in pure.parts[0]:
        raise ChunkMappingError(f"row {row_index}: image_ref contains a drive prefix")
    if storage is None:
        raise ChunkMappingError(
            f"row {row_index}: image_ref requires a StorageManager for publication"
        )

    if pure.parts[0] == "images":
        expected_prefix = ("images", str(identity.document_id))
        if pure.parts[:2] != expected_prefix:
            raise ChunkMappingError(
                f"row {row_index}: canonical image_ref belongs to another document"
            )
        try:
            resolved = storage.resolve_image_ref(pure.as_posix())
        except StorageSecurityError as exc:
            raise ChunkMappingError(f"row {row_index}: unsafe image_ref") from exc
        if not resolved.is_file():
            raise ChunkMappingError(f"row {row_index}: canonical image_ref does not exist")
        return resolved.relative_to(storage.root).as_posix()

    if image_base_dir is None:
        raise ChunkMappingError(
            f"row {row_index}: relative image_ref requires image_base_dir"
        )
    try:
        base = Path(image_base_dir).resolve(strict=True)
        source = base.joinpath(*pure.parts).resolve(strict=True)
        source.relative_to(base)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ChunkMappingError(
            f"row {row_index}: image_ref is missing or escapes its artifact"
        ) from exc
    try:
        return storage.publish_image(identity.document_id, persistence_id, source)
    except (OSError, StorageSecurityError) as exc:
        raise ChunkMappingError(f"row {row_index}: image publication failed") from exc


def _required_string(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise ChunkMappingError(f"{field} must be a string")
    return value


def _optional_string(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ChunkMappingError(f"{field} must be a string")
    return value


def _required_bool(row: Mapping[str, Any], field: str) -> bool:
    value = row.get(field)
    if not isinstance(value, bool):
        raise ChunkMappingError(f"{field} must be a boolean")
    return value


def _required_positive_int(row: Mapping[str, Any], field: str, row_index: int) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ChunkMappingError(f"row {row_index}: {field} must be a positive integer")
    return value


def _section_parts(value: Any, row_index: int) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ChunkMappingError(f"row {row_index}: section_path must be a string array")
    parts: list[str] = []
    for part in value:
        if not isinstance(part, str):
            raise ChunkMappingError(f"row {row_index}: section_path must be a string array")
        stripped = part.strip()
        if stripped:
            parts.append(stripped)
    return parts


def _validate_source_pages(
    value: Any, page_start: int, page_end: int, row_index: int
) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ChunkMappingError(f"row {row_index}: source_pages must be an integer array")
    pages: list[int] = []
    for page in value:
        if isinstance(page, bool) or not isinstance(page, int) or page <= 0:
            raise ChunkMappingError(f"row {row_index}: source_pages contains an invalid page")
        if not page_start <= page <= page_end:
            raise ChunkMappingError(
                f"row {row_index}: source_pages must stay inside page_start/page_end"
            )
        pages.append(page)
    if not pages:
        raise ChunkMappingError(f"row {row_index}: source_pages must not be empty")
    if pages != sorted(set(pages)):
        raise ChunkMappingError(f"row {row_index}: source_pages must be sorted and unique")
    return pages


def _validate_table_repr(value: Any, row_index: int) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise ChunkMappingError(f"row {row_index}: table_repr must be an object or null")
    _validate_json_value(value, row_index, "table_repr")


def _validate_json_value(value: Any, row_index: int, field: str) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ChunkMappingError(f"row {row_index}: {field} is not valid JSON") from exc


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _validate_nonempty_string(
    field: str, value: str, max_bytes: int, *, row_index: int | None = None
) -> None:
    prefix = f"row {row_index}: " if row_index is not None else ""
    if not isinstance(value, str) or not value.strip():
        raise ChunkMappingError(f"{prefix}{field} must not be empty")
    _validate_string_limit(field, value, max_bytes, prefix=prefix)


def _validate_string_limit(
    field: str, value: str, max_bytes: int, *, prefix: str = ""
) -> None:
    length = len(value.encode("utf-8"))
    if length > max_bytes:
        raise ChunkMappingError(
            f"{prefix}{field} is {length} UTF-8 bytes; maximum is {max_bytes}"
        )


__all__ = [
    "AuthoritativeDocumentIdentity",
    "ChunkKind",
    "ChunkMappingError",
    "PersistenceChunk",
    "load_jsonl",
    "map_chunk_records",
    "map_jsonl",
    "map_records",
]
