"""Read figure descriptions and align them with parsed snapshot blocks (in memory).

The description sidecar is produced by the figure-description stage and keyed by the
stable ``figure_id`` (``<doc_id>:p<page>:f<ordinal>``).  This module only enriches the
in-memory page structure; parsed files are never rewritten.

Rules:
* a described snapshot gets ``figure_description`` (short/full/labels/status) and a
  ``description_status`` marker used by the chunker;
* a snapshot that repeats the same page + bbox as an earlier one is marked duplicate
  and skipped, so duplicated figures cannot produce a second chunk;
* undescribed snapshots stay unchanged (the chunker keeps its previous behaviour).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path


def figure_id_for(doc_id: str, page: int, ordinal: int) -> str:
    return f"{doc_id}:p{page:03d}:f{ordinal}"


def load_figure_descriptions(run_dir: str | Path) -> dict[str, dict]:
    """Load ``<run_dir>/<doc_id>/figures.jsonl`` into ``{figure_id: record}``."""
    root = Path(run_dir)
    out: dict[str, dict] = {}
    if not root.is_dir():
        return out
    for doc_dir in sorted(root.iterdir()):
        path = doc_dir / "figures.jsonl"
        if not doc_dir.is_dir() or not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            out[record["figure_id"]] = record
    return out


def apply_figure_descriptions(document: object, records: dict[str, dict]):
    """Return (document copy, report) with description fields and duplicate markers."""
    doc_id = getattr(document, "doc_id", "")
    pages = getattr(document, "pages", [])
    if not pages:
        return document, {"described": 0, "undescribed": 0, "duplicates": 0}

    described = undescribed = duplicates = 0
    new_pages: list[dict] = []
    for page in pages:
        page_copy = {**page, "blocks": []}
        seen: dict[tuple[float, ...], str] = {}
        ordinal = 0
        for block in page.get("blocks") or []:
            if block.get("type") != "snapshot":
                page_copy["blocks"].append(block)
                continue
            ordinal += 1
            figure_id = figure_id_for(doc_id, page.get("page", 0), ordinal)
            bbox = tuple(round(value, 1) for value in (block.get("bbox") or []))
            if bbox in seen:
                duplicates += 1
                page_copy["blocks"].append(
                    {**block, "figure_id": figure_id, "skip_block": True,
                     "duplicate_of": seen[bbox]}
                )
                continue
            seen[bbox] = figure_id
            record = records.get(figure_id)
            if record is None:
                undescribed += 1
                page_copy["blocks"].append(
                    {**block, "figure_id": figure_id, "description_status": "missing"}
                )
                continue
            generation_status = str(record.get("generation_status") or "")
            short = (record.get("description_short") or "").strip()
            full = (record.get("description_full") or "").strip()
            if record.get("excluded"):
                undescribed += 1
                page_copy["blocks"].append(
                    {
                        **block,
                        "figure_id": figure_id,
                        "description_status": "excluded",
                        "figure_excluded": True,
                        "exclusion_reason": record.get("exclusion_reason", ""),
                    }
                )
                continue
            if generation_status != "generated" or not short or not full:
                # Complete success requires BOTH a short and a full description; a
                # half-filled record must never count as described.
                undescribed += 1
                page_copy["blocks"].append(
                    {
                        **block,
                        "figure_id": figure_id,
                        "description_status": "failed",
                        "description_error": record.get("error")
                        or (
                            "incomplete_description"
                            if not (short and full)
                            else generation_status
                        ),
                    }
                )
                continue
            described += 1
            diagram_type = str(record.get("diagram_type") or "")
            page_copy["blocks"].append(
                {
                    **block,
                    "figure_id": figure_id,
                    # Diagram type is chunk metadata; it never overwrites a code block
                    # (the later code-sidecar pass promotes matching snapshots to code).
                    "figure_type": str(block.get("figure_type") or diagram_type),
                    "figure_description": {
                        "short": short,
                        "full": full,
                        "labels": record.get("visible_labels") or [],
                        "diagram_type": diagram_type,
                        "status": generation_status,
                        "uncertainties": record.get("uncertainties") or [],
                    },
                    "description_status": "described",
                }
            )
        new_pages.append(page_copy)
    return replace(document, pages=new_pages), {
        "described": described,
        "undescribed": undescribed,
        "duplicates": duplicates,
    }
