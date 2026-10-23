"""Read the code-text sidecar and align it with parsed snapshot blocks.

The sidecar is produced by ``scripts/build_code_sidecar.py`` (plan §4.4).  Alignment is
purely in memory: parsed files are never rewritten.  Rules:

* a ``complete`` record turns its snapshot block into a code block (existing code branch).
* a ``partial`` record keeps the block as a figure and marks ``code_partial`` so the
  chunker never emits it as a complete code chunk.
* a ``no-fit`` record changes nothing.
* a block whose record carries ``duplicate_of`` is dropped, so duplicates cannot
  produce a second chunk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class CodeRecord:
    figure_id: str
    doc_id: str
    page: int
    image_ref: str
    status: str
    effective_status: str
    review_status: str
    duplicate_of: str | None
    code_text: str
    source_bbox: tuple[float, ...]
    issues: tuple[str, ...]
    lines: tuple[dict, ...]

    @property
    def is_code(self) -> bool:
        return self.effective_status == "complete" and bool(self.code_text.strip())


def load_code_sidecar(run_dir: str | Path) -> dict[str, list[CodeRecord]]:
    """Load ``<run_dir>/<doc_id>/code.jsonl`` into ``{doc_id: [records]}``."""
    root = Path(run_dir)
    out: dict[str, list[CodeRecord]] = {}
    if not root.is_dir():
        return out
    for doc_dir in sorted(root.iterdir()):
        path = doc_dir / "code.jsonl"
        if not doc_dir.is_dir() or not path.is_file():
            continue
        records: list[CodeRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            records.append(
                CodeRecord(
                    figure_id=raw["figure_id"],
                    doc_id=raw["doc_id"],
                    page=int(raw["page"]),
                    image_ref=raw.get("image_ref", ""),
                    status=raw.get("status", "no-fit"),
                    effective_status=raw.get("effective_status", raw.get("status", "no-fit")),
                    review_status=raw.get("review_status", ""),
                    duplicate_of=raw.get("duplicate_of"),
                    code_text=raw.get("code_text", ""),
                    source_bbox=tuple(raw.get("source_bbox") or ()),
                    issues=tuple(raw.get("issues") or ()),
                    lines=tuple(raw.get("code_lines") or raw.get("lines") or ()),
                )
            )
        out[doc_dir.name] = records
    return out


def align_pages(
    pages: list[dict], records: list[CodeRecord]
) -> tuple[list[dict], dict]:
    """Return (in-memory pages copy, alignment report)."""
    by_key = {(record.page, record.image_ref): record for record in records}
    aligned: list[dict] = []
    report = {
        "code_blocks": 0,
        "partial_blocks": 0,
        "dropped_duplicates": 0,
        "unmatched_records": len(records),
        "matched_records": 0,
    }
    for page in pages:
        page_copy = {**page, "blocks": []}
        blocks: list[dict] = page.get("blocks") or []
        for block in blocks:
            if block.get("type") != "snapshot":
                page_copy["blocks"].append(block)
                continue
            record = by_key.get((page.get("page"), block.get("image_ref", "")))
            if record is None:
                page_copy["blocks"].append(block)
                continue
            report["matched_records"] += 1
            if record.duplicate_of:
                report["dropped_duplicates"] += 1
                continue
            if record.is_code:
                page_copy["blocks"].append(
                    {
                        **block,
                        "figure_type": "code",
                        "code_text": record.code_text,
                        "code_figure_id": record.figure_id,
                        "code_review_status": record.review_status,
                    }
                )
                report["code_blocks"] += 1
            elif record.effective_status == "partial":
                page_copy["blocks"].append(
                    {
                        **block,
                        "code_partial": True,
                        "code_figure_id": record.figure_id,
                        "code_status": record.effective_status,
                    }
                )
                report["partial_blocks"] += 1
            else:
                page_copy["blocks"].append(block)
        aligned.append(page_copy)
    report["unmatched_records"] -= report["matched_records"]
    return aligned, report


def apply_code_sidecar(document: object, records_by_doc: dict[str, list[CodeRecord]]):
    """Return a document copy whose pages are aligned with the sidecar (no disk writes)."""
    records = records_by_doc.get(getattr(document, "doc_id", ""), [])
    if not records:
        return document, {"code_blocks": 0, "partial_blocks": 0, "dropped_duplicates": 0,
                          "unmatched_records": 0, "matched_records": 0}
    pages, report = align_pages(getattr(document, "pages", []), records)
    return replace(document, pages=pages), report
