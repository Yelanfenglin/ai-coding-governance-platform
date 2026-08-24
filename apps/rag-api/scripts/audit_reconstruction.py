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
    for page in pages:
        page_number = page.get("page")
        if expected_page is not None and page_number <= expected_page:
            errors.append(f"page order is not increasing at {page_number}")
        expected_page = page_number
        width, height = float(page.get("width", 0)), float(page.get("height", 0))
        seen_indices = set()
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
                _check_table(errors, page_number, block)

    markdown_path = output / "md" / "full.md"
    markdown = markdown_path.read_text(encoding="utf-8")
    expected_pages = [int(page["page"]) for page in pages]
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


def _check_table(errors: list[str], page_number: int, table: dict) -> None:
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
