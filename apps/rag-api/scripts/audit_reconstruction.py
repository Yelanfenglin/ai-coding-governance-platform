"""离线审计 parser 产物的结构不变量。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_PAGE_MARKER_RE = re.compile(r"<!-- pdf-page: (\d+) -->")
_HEADING_RE = re.compile(r"^(#{2,6})\s+(.+)$", re.MULTILINE)
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,5})(?:\s+|(?=[A-Za-z\u3400-\u9fff]))(.+)$")
_NEGATIVE_HEADING_RE = re.compile(r"^(?:注|说明|备注|注意|例)\s*[:：]")
_VALUE_HEADING_RE = re.compile(r"^\d+\s*(?:=|[:：])\s*\S")


def audit(output: Path) -> list[str]:
    errors: list[str] = []
    summary = _json(output / "summary.json")
    pages = _json(output / "pages.json")
    if not isinstance(pages, list):
        return [f"{output}: pages.json is not a list"]
    if summary.get("pages") != len(pages):
        errors.append(f"summary pages={summary.get('pages')} but pages.json has {len(pages)}")
    quality = summary.get("quality", {})
    for key in ("orphan_text_lines", "duplicate_owned_lines"):
        if quality.get(key) != 0:
            errors.append(f"quality.{key}={quality.get(key)}")
    if quality.get("missing_snapshot_files"):
        errors.append(f"quality.missing_snapshot_files={quality['missing_snapshot_files']}")

    expected_page = None
    total_implicit_groups = 0
    total_implicit_subrows = 0
    for page in pages:
        page_number = page.get("page")
        if expected_page is not None and page_number <= expected_page:
            errors.append(f"page order is not increasing at {page_number}")
        expected_page = page_number
        width, height = float(page.get("width", 0)), float(page.get("height", 0))
        seen_indices = set()
        page_implicit_groups = 0
        page_implicit_subrows = 0
        for block in page.get("blocks", []):
            index = block.get("block_index")
            if index in seen_indices:
                errors.append(f"p{page_number}: duplicate block_index {index}")
            seen_indices.add(index)
            _check_bbox(errors, page_number, block.get("bbox"), width, height)
            if block.get("type") == "snapshot":
                image = output / block.get("image_ref", "")
                if not image.is_file():
                    errors.append(f"p{page_number}: missing snapshot {image.name}")
            elif block.get("type") == "table":
                groups, subrows = _check_table(
                    errors,
                    page_number,
                    block,
                    width,
                    height,
                )
                page_implicit_groups += groups
                page_implicit_subrows += subrows
        stats = page.get("stats", {})
        if stats.get("implicit_subtable_groups", 0) != page_implicit_groups:
            errors.append(
                f"p{page_number}: implicit_subtable_groups="
                f"{stats.get('implicit_subtable_groups', 0)} but blocks have "
                f"{page_implicit_groups}"
            )
        if stats.get("implicit_subrows", 0) != page_implicit_subrows:
            errors.append(
                f"p{page_number}: implicit_subrows={stats.get('implicit_subrows', 0)} "
                f"but blocks have {page_implicit_subrows}"
            )
        total_implicit_groups += page_implicit_groups
        total_implicit_subrows += page_implicit_subrows

    if summary.get("implicit_subtable_groups", 0) != total_implicit_groups:
        errors.append(
            f"summary implicit_subtable_groups="
            f"{summary.get('implicit_subtable_groups', 0)} but pages have "
            f"{total_implicit_groups}"
        )
    if summary.get("implicit_subrows", 0) != total_implicit_subrows:
        errors.append(
            f"summary implicit_subrows={summary.get('implicit_subrows', 0)} "
            f"but pages have {total_implicit_subrows}"
        )

    markdown_path = output / "md" / "full.md"
    markdown = markdown_path.read_text(encoding="utf-8")
    # A page fully absorbed by a cross-page table stitch has no content of its
    # own and therefore no pdf-page marker, even though it is still a page entry
    # in pages.json.  Only require a marker for pages that have rendered blocks.
    expected_pages = [int(page["page"]) for page in pages if page.get("blocks")]
    markdown_pages = [int(value) for value in _PAGE_MARKER_RE.findall(markdown)]
    if markdown_pages != expected_pages:
        errors.append("Markdown pdf-page markers do not match pages.json")
    for reference in _IMAGE_RE.findall(markdown):
        if "://" in reference:
            continue
        resolved = (markdown_path.parent / reference).resolve()
        if not resolved.is_file():
            errors.append(f"Markdown image is missing: {reference}")
    _check_headings(errors, markdown)
    return errors


def _check_bbox(errors: list[str], page_number: int, bbox, width: float, height: float) -> None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        errors.append(f"p{page_number}: invalid bbox {bbox}")
        return
    x0, y0, x1, y1 = map(float, bbox)
    if x1 <= x0 or y1 <= y0:
        errors.append(f"p{page_number}: non-positive bbox {bbox}")
    if x0 < -2 or y0 < -2 or x1 > width + 2 or y1 > height + 2:
        errors.append(f"p{page_number}: bbox outside page {bbox}")


def _check_table(
    errors: list[str],
    page_number: int,
    table: dict,
    page_width: float,
    page_height: float,
) -> tuple[int, int]:
    rows = table.get("rows", [])
    n_rows, n_cols = table.get("n_rows"), table.get("n_cols")
    if len(rows) != n_rows or any(len(row) != n_cols for row in rows):
        errors.append(f"p{page_number}: table matrix does not match {n_rows}x{n_cols}")
    origins = set()
    occupied: dict[tuple[int, int], tuple[int, int]] = {}
    for cell in table.get("cells", []):
        row, col = cell.get("row"), cell.get("col")
        rowspan, colspan = cell.get("rowspan"), cell.get("colspan")
        values = (row, col, rowspan, colspan)
        if not all(isinstance(value, int) for value in values):
            errors.append(f"p{page_number}: non-integer table cell geometry {values}")
            continue
        if (row, col) in origins:
            errors.append(f"p{page_number}: duplicate table origin {(row, col)}")
        origins.add((row, col))
        if (
            row < 0
            or col < 0
            or rowspan < 1
            or colspan < 1
            or row + rowspan > n_rows
            or col + colspan > n_cols
        ):
            errors.append(f"p{page_number}: cell span outside table at {(row, col)}")
            continue
        for covered_row in range(row, row + rowspan):
            for covered_col in range(col, col + colspan):
                coordinate = (covered_row, covered_col)
                if coordinate in occupied:
                    errors.append(
                        f"p{page_number}: overlapping cells {occupied[coordinate]} "
                        f"and {(row, col)} at {coordinate}"
                    )
                occupied[coordinate] = (row, col)
    expected_cell_count = n_rows * n_cols
    if len(occupied) != expected_cell_count:
        errors.append(
            f"p{page_number}: table cells cover {len(occupied)}/{expected_cell_count} slots"
        )
    return _check_semantic_rows(
        errors,
        page_number,
        table,
        page_width,
        page_height,
    )


def _check_semantic_rows(
    errors: list[str],
    page_number: int,
    table: dict,
    page_width: float,
    page_height: float,
) -> tuple[int, int]:
    semantic_rows = table.get("semantic_rows")
    reconstruction = table.get("semantic_reconstruction")
    if semantic_rows is None and reconstruction is None:
        return 0, 0
    if not isinstance(semantic_rows, list) or not isinstance(reconstruction, dict):
        errors.append(f"p{page_number}: semantic table view is incomplete")
        return 0, 0

    n_rows, n_cols = table.get("n_rows"), table.get("n_cols")
    if any(not isinstance(row, list) or len(row) != n_cols for row in semantic_rows):
        errors.append(f"p{page_number}: semantic rows do not match {n_cols} columns")
    if reconstruction.get("kind") != "aligned_subrows" or not reconstruction.get("method"):
        errors.append(f"p{page_number}: invalid semantic reconstruction metadata")

    parent_rows = reconstruction.get("parent_rows")
    provenance = reconstruction.get("row_provenance")
    if not isinstance(parent_rows, list) or not isinstance(provenance, list):
        errors.append(f"p{page_number}: semantic reconstruction misses provenance")
        return 0, 0
    if parent_rows != sorted(set(parent_rows)) or any(
        not isinstance(row, int) or row <= 0 or row >= n_rows for row in parent_rows
    ):
        errors.append(f"p{page_number}: invalid semantic parent rows {parent_rows}")

    by_parent: dict[int, list[dict]] = {}
    semantic_indices: set[int] = set()
    for record in provenance:
        parent_row = record.get("parent_row")
        subrow = record.get("subrow")
        semantic_row = record.get("semantic_row")
        if not all(isinstance(value, int) for value in (parent_row, subrow, semantic_row)):
            errors.append(f"p{page_number}: non-integer semantic row provenance")
            continue
        by_parent.setdefault(parent_row, []).append(record)
        if semantic_row in semantic_indices:
            errors.append(f"p{page_number}: duplicate semantic row index {semantic_row}")
        semantic_indices.add(semantic_row)
        if semantic_row < 0 or semantic_row >= len(semantic_rows):
            errors.append(f"p{page_number}: semantic row index outside matrix {semantic_row}")
        cell_bboxes = record.get("cell_bboxes")
        source_lines = record.get("source_lines")
        if not isinstance(cell_bboxes, list) or len(cell_bboxes) != n_cols:
            errors.append(f"p{page_number}: semantic row {semantic_row} misses cell bboxes")
        else:
            for bbox in cell_bboxes:
                _check_bbox(errors, page_number, bbox, page_width, page_height)
        if not isinstance(source_lines, list) or len(source_lines) != n_cols:
            errors.append(f"p{page_number}: semantic row {semantic_row} misses source lines")
        delta = record.get("max_baseline_delta")
        if not isinstance(delta, (int, float)) or delta < 0:
            errors.append(f"p{page_number}: invalid baseline delta {delta}")

    if sorted(by_parent) != parent_rows:
        errors.append(
            f"p{page_number}: parent rows {parent_rows} do not match provenance {sorted(by_parent)}"
        )
    expected_semantic_rows = n_rows - len(parent_rows) + len(provenance)
    if len(semantic_rows) != expected_semantic_rows:
        errors.append(
            f"p{page_number}: semantic matrix has {len(semantic_rows)} rows; "
            f"expected {expected_semantic_rows}"
        )

    semantic_cursor = 0
    raw_rows = table.get("rows", [])
    for parent_row in range(n_rows):
        records = sorted(by_parent.get(parent_row, []), key=lambda item: item["subrow"])
        if not records:
            if (
                semantic_cursor < len(semantic_rows)
                and parent_row < len(raw_rows)
                and semantic_rows[semantic_cursor] != raw_rows[parent_row]
            ):
                errors.append(f"p{page_number}: raw row {parent_row} changed in semantic view")
            semantic_cursor += 1
            continue
        if [record["subrow"] for record in records] != list(range(len(records))):
            errors.append(f"p{page_number}: non-contiguous subrows at parent {parent_row}")
        for record in records:
            if record["semantic_row"] != semantic_cursor:
                errors.append(
                    f"p{page_number}: provenance maps parent {parent_row} to semantic row "
                    f"{record['semantic_row']}, expected {semantic_cursor}"
                )
            semantic_cursor += 1
    return len(parent_rows), len(provenance)


def _check_headings(errors: list[str], markdown: str) -> None:
    for hashes, title in _HEADING_RE.findall(markdown):
        if title.startswith("Page "):
            errors.append(f"page marker leaked into heading tree: {title}")
        if _NEGATIVE_HEADING_RE.match(title):
            errors.append(f"note box leaked into heading tree: {title}")
        if _VALUE_HEADING_RE.match(title):
            errors.append(f"enumerated value leaked into heading tree: {title}")
        numbered = _NUMBERED_HEADING_RE.match(title)
        if not numbered:
            continue
        expected = min(6, numbered.group(1).count(".") + 2)
        if len(hashes) != expected:
            errors.append(
                f"heading level mismatch: expected H{expected}, got H{len(hashes)}: {title}"
            )


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit parser outputs")
    parser.add_argument("outputs", nargs="+", type=Path)
    args = parser.parse_args()
    failed = False
    for output in args.outputs:
        errors = audit(output)
        if errors:
            failed = True
            print(f"[FAIL] {output}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"[OK] {output}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
