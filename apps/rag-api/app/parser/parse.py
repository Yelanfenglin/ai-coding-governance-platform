"""Deterministic PDF reconstruction pipeline and CLI."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import fitz

from .anchor import add_recovered_table_anchors, bind_native_lines, make_anchors
from .document import SectionTracker, link_table_continuations
from .layout import load_all_proposals
from .native_pdf import extract_native_page
from .native_tables import recover_missing_tables
from .page import (
    PageIR,
    SnapshotBlock,
    TableBlock,
    TextBlock,
    build_blocks,
    prepare_table_anchors,
)
from .postprocess import recover_document_headings
from .renderer import blocks_to_markdown, snapshot_filename
from .visual import add_recovered_visual_anchors, refine_snapshot_anchors


def _run(
    pdf_path: str | Path,
    layout_json: str | Path,
    out_dir: str | Path,
    pages: list[int] | None = None,
    dpi: int = 160,
    *,
    allow_missing_layout: bool = False,
    recover_native_tables: bool = True,
) -> dict:
    pdf_path = Path(pdf_path)
    layout_json = Path(layout_json)
    out = Path(out_dir)
    json_dir = out / "json"
    markdown_dir = out / "md"
    snapshot_dir = out / "snapshots"
    for directory in (json_dir, markdown_dir, snapshot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    all_proposals = load_all_proposals(layout_json)
    page_irs: list[PageIR] = []

    with fitz.open(str(pdf_path)) as document:
        page_range = pages if pages is not None else list(range(1, len(document) + 1))
        invalid = [page for page in page_range if page < 1 or page > len(document)]
        if invalid:
            raise ValueError(f"page out of range: {invalid}; PDF has {len(document)} pages")
        missing = [page for page in page_range if page not in all_proposals]
        if missing and not allow_missing_layout:
            raise ValueError(f"layout JSON misses pages: {missing}")

        for page_number in page_range:
            native = extract_native_page(document[page_number - 1])
            proposals = all_proposals.get(page_number, [])
            anchors = make_anchors(proposals)

            recovered = []
            if recover_native_tables:
                covered = [
                    proposal.bbox for proposal in proposals if proposal.hint in ("TABLE", "VISUAL")
                ]
                context = [
                    proposal.bbox for proposal in proposals if proposal.hint in ("TEXT", "TABLE")
                ]
                recovered = recover_missing_tables(
                    native,
                    covered,
                    native.width,
                    native.height,
                    context_bboxes=context,
                )
                add_recovered_table_anchors(anchors, recovered)

            prepare_table_anchors(native, anchors)
            refine_snapshot_anchors(native, anchors)
            add_recovered_visual_anchors(native, anchors)

            bind_native_lines(native, anchors)
            page_ir = build_blocks(page_number, native, anchors)
            page_ir.stats.update(_ownership_stats(native, anchors))
            page_irs.append(page_ir)

            _save_snapshots(document, page_number, page_ir, snapshot_dir, dpi)

    recover_document_headings(page_irs)
    section_tracker = SectionTracker()
    page_dicts = [section_tracker.annotate(_page_dict(page_ir)) for page_ir in page_irs]

    link_table_continuations(page_dicts)
    for page in page_dicts:
        (json_dir / f"p{page['page']:03d}.json").write_text(
            json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    markdown_parts = [f"# {pdf_path.stem}"]
    for page_ir in page_irs:
        markdown = blocks_to_markdown(page_ir, "../snapshots")
        markdown_parts.append(f"<!-- pdf-page: {page_ir.page_number} -->\n\n{markdown}")
    (markdown_dir / "full.md").write_text("\n\n".join(markdown_parts), encoding="utf-8")
    (out / "pages.json").write_text(
        json.dumps(page_dicts, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = _summary(pdf_path, layout_json, page_irs, page_dicts, snapshot_dir)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _ownership_stats(native, anchors) -> dict:
    native_ids = {(line.block_id, line.line_id) for line in native.lines if line.text.strip()}
    owners: Counter = Counter()
    for anchor in anchors:
        for line in anchor.lines:
            owners[(line.block_id, line.line_id)] += 1
    return {
        "native_text_lines": len(native_ids),
        "owned_text_lines": len(owners),
        "orphan_text_lines": len(native_ids - set(owners)),
        "duplicate_owned_lines": sum(1 for count in owners.values() if count > 1),
        "auto_text_lines": sum(len(anchor.lines) for anchor in anchors if anchor.source == "auto"),
        "noise_text_lines": sum(len(anchor.lines) for anchor in anchors if anchor.kind == "NOISE"),
    }


def _page_dict(page: PageIR) -> dict:
    blocks: list[dict] = []
    snapshot_ordinal = 0
    for block_index, block in enumerate(page.blocks):
        if isinstance(block, TextBlock):
            item = {
                "type": "text",
                "kind": block.kind,
                "text": block.text,
                "bbox": block.bbox.to_tuple(),
                "source": block.source,
                "pp_label": block.pp_label,
                "pp_order": block.pp_order,
                "font_size": block.font_size,
            }
        elif isinstance(block, TableBlock):
            cells = []
            for cell in block.table.origin_cells():
                cells.append(
                    {
                        "row": cell.row,
                        "col": cell.col,
                        "rowspan": cell.rowspan,
                        "colspan": cell.colspan,
                        "text": cell.text,
                        "bbox": cell.bbox.to_tuple() if cell.bbox else None,
                    }
                )
            item = {
                "type": "table",
                "bbox": block.bbox.to_tuple(),
                "source": block.source,
                "n_rows": block.table.n_rows,
                "n_cols": block.table.n_cols,
                "rows": block.table.text_rows(),
                "cells": cells,
            }
        elif isinstance(block, SnapshotBlock):
            snapshot_ordinal += 1
            item = {
                "type": "snapshot",
                "hint": block.hint,
                "bbox": block.bbox.to_tuple(),
                "annotations": block.annotations,
                "source": block.source,
                "image_ref": (
                    f"snapshots/{snapshot_filename(page.page_number, snapshot_ordinal, block.hint)}"
                ),
            }
        else:
            continue
        item["block_index"] = block_index
        blocks.append(item)
    return {
        "page": page.page_number,
        "width": page.width,
        "height": page.height,
        "stats": page.stats,
        "blocks": blocks,
    }


def _save_snapshots(
    document,
    page_number: int,
    page_ir: PageIR,
    snapshot_dir: Path,
    dpi: int,
) -> None:
    import numpy as np
    from PIL import Image

    snapshots = [block for block in page_ir.blocks if isinstance(block, SnapshotBlock)]
    if not snapshots:
        return
    page = document[page_number - 1]
    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )
    if pixmap.n == 4:
        image = image[:, :, :3]
    scale_x, scale_y = pixmap.width / page.rect.width, pixmap.height / page.rect.height
    for ordinal, snapshot in enumerate(snapshots, 1):
        bbox = snapshot.bbox
        x0 = max(0, int(bbox.x0 * scale_x))
        y0 = max(0, int(bbox.y0 * scale_y))
        x1 = min(pixmap.width, int(bbox.x1 * scale_x))
        y1 = min(pixmap.height, int(bbox.y1 * scale_y))
        if x1 <= x0 or y1 <= y0:
            continue
        filename = snapshot_filename(page_number, ordinal, snapshot.hint)
        Image.fromarray(image[y0:y1, x0:x1]).save(snapshot_dir / filename)


def _summary(
    pdf_path: Path,
    layout_json: Path,
    page_irs: list[PageIR],
    page_dicts: list[dict],
    snapshot_dir: Path,
) -> dict:
    stats = [page.stats for page in page_irs]
    expected_snapshots = [
        snapshot_dir / Path(block["image_ref"]).name
        for page in page_dicts
        for block in page["blocks"]
        if block["type"] == "snapshot"
    ]
    return {
        "pdf": pdf_path.name,
        "layout": layout_json.name,
        "pages": len(page_irs),
        "tables": sum(item.get("tables_rebuilt", 0) for item in stats),
        "tables_recovered": sum(item.get("tables_recovered", 0) for item in stats),
        "table_merge_repairs": sum(item.get("table_merge_repairs", 0) for item in stats),
        "table_topology_upgrades": sum(item.get("table_topology_upgrades", 0) for item in stats),
        "table_snapshot_candidates": sum(item.get("tables_snapshot", 0) for item in stats),
        # Backward-compatible artifact count.  This is a model label, not
        # visually confirmed table ground truth.
        "table_snapshots": sum(item.get("tables_snapshot", 0) for item in stats),
        "visual_snapshots": sum(item.get("visual_snapshots", 0) for item in stats),
        "quality": {
            "orphan_text_lines": sum(item.get("orphan_text_lines", 0) for item in stats),
            "duplicate_owned_lines": sum(item.get("duplicate_owned_lines", 0) for item in stats),
            "auto_text_lines": sum(item.get("auto_text_lines", 0) for item in stats),
            "missing_snapshot_files": [
                path.name for path in expected_snapshots if not path.is_file()
            ],
            "table_continuations": sum(
                1
                for page in page_dicts
                for block in page["blocks"]
                if block.get("continued_from_page")
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF + PP-DocLayoutV3 deterministic reconstructor")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pages", default="")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--allow-missing-layout", action="store_true")
    parser.add_argument("--no-native-table-recovery", action="store_true")
    args = parser.parse_args()
    pages = [int(value) for value in args.pages.split(",") if value.strip()] if args.pages else None
    summary = _run(
        args.pdf,
        args.layout,
        args.out,
        pages,
        args.dpi,
        allow_missing_layout=args.allow_missing_layout,
        recover_native_tables=not args.no_native_table_recovery,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
