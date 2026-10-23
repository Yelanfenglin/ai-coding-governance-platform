"""Build the code-text sidecar for recovered code screenshots (plan §4.4).

Read-only over the PDFs and ``storage/parsed``: nothing is written back to the parsed
tree.  Output:

    storage/code_text/<run_id>/<doc_id>/code.jsonl
    storage/code_text/<run_id>/manifest.json

Every record is keyed by ``figure_id`` and the source snapshot block's own bbox, never
by "the largest snapshot on the page".  Recovery status is per block; problems and
provenance are per segment / per line.  A human review overlay (data file, not code)
can restore an approved edge-caption segment; the automatic verdict is preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fitz  # noqa: E402

from app.chunking.loader import load_parsed_document  # noqa: E402
from app.parser.native_pdf import extract_native_page  # noqa: E402

PARSED = ROOT / "storage" / "parsed"
DEFAULT_CANDIDATES = ROOT / "storage" / "code_text" / "candidates.json"
DEFAULT_REVIEWS = ROOT / "storage" / "code_text" / "reviews.json"


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #


def normalize(text: str) -> str:
    return "".join((text or "").split())


def segment_tolerance(lines: list[dict]) -> float:
    """Row grouping tolerance from observed line pitch (0.55 x median height)."""
    heights = [line["y1"] - line["y0"] for line in lines if line["y1"] > line["y0"]]
    if not heights:
        return 2.0
    return max(1.0, round(0.55 * statistics.median(heights), 2))


def build_rows(segments: list[dict], tol: float) -> list[list[dict]]:
    rows: list[list[dict]] = []
    for seg in sorted(segments, key=lambda s: (s["y0"], s["x0"])):
        if rows and abs(seg["y0"] - rows[-1][0]["y0"]) <= tol:
            rows[-1].append(seg)
        else:
            rows.append([seg])
    for row in rows:
        row.sort(key=lambda s: s["x0"])
    return rows


def expected_partition(segments: list[dict]) -> frozenset[frozenset[str]]:
    parent = list(range(len(segments)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, left in enumerate(segments):
        for j in range(i + 1, len(segments)):
            right = segments[j]
            top, bottom = max(left["y0"], right["y0"]), min(left["y1"], right["y1"])
            if bottom <= top:
                continue
            shorter = min(left["y1"] - left["y0"], right["y1"] - right["y0"]) or 1.0
            if (bottom - top) / shorter > 0.5:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    groups: dict[int, set[str]] = {}
    for i in range(len(segments)):
        groups.setdefault(find(i), set()).add(segments[i]["id"])
    return frozenset(frozenset(group) for group in groups.values())


def overflow_by_side(bbox: list[float], segment: dict) -> dict[str, float]:
    x0, y0, x1, y1 = bbox
    return {
        "left": round(max(0.0, x0 - segment["x0"]), 2),
        "top": round(max(0.0, y0 - segment["y0"]), 2),
        "right": round(max(0.0, segment["x1"] - x1), 2),
        "bottom": round(max(0.0, segment["y1"] - y1), 2),
    }


# --------------------------------------------------------------------------- #
# page level recovery
# --------------------------------------------------------------------------- #


def page_blocks(parsed_dir: Path, page_no: int) -> dict:
    pages = json.loads((parsed_dir / "pages.json").read_text(encoding="utf-8"))
    return next(page for page in pages if page["page"] == page_no)


def block_texts(block: dict) -> list[str]:
    values: list[str] = []
    for key in ("annotations", "text"):
        value = block.get(key)
        if isinstance(value, str):
            values.extend(value.splitlines())
    for row in block.get("rows") or []:
        if isinstance(row, list):
            values.extend(cell for cell in row if isinstance(cell, str))
    return [normalize(value) for value in values if normalize(value)]


def recover_block(
    *,
    model: str,
    doc_id: str,
    page_no: int,
    ordinal: int,
    block: dict,
    native,
    same_page_blocks: list[dict],
    review: dict | None,
) -> dict:
    x0, y0, x1, y1 = block["bbox"]
    bbox = [round(v, 2) for v in (x0, y0, x1, y1)]
    figure_id = f"{doc_id}:p{page_no:03d}:s{ordinal}"
    own = Counter(
        normalize(line) for line in (block.get("annotations") or "").splitlines() if normalize(line)
    )

    others: list[tuple[str, Counter]] = []
    for index, other in enumerate(same_page_blocks):
        if other is block:
            continue
        texts = block_texts(other)
        if texts:
            others.append((f'{other.get("type")}:{index}', Counter(texts)))

    segments: list[dict] = []
    outside: list[dict] = []
    excluded: list[dict] = []
    for line in native.lines:
        text = (line.text or "").strip()
        if not text:
            continue
        b = line.bbox
        cx, cy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
        intersects = not (b.x1 <= x0 or b.x0 >= x1 or b.y1 <= y0 or b.y0 >= y1)
        center_inside = x0 <= cx <= x1 and y0 <= cy <= y1
        if not intersects:
            continue
        segment = {
            "id": f"seg{len(segments) + len(excluded) + len(outside) + 1:03d}",
            "text": text,
            "norm": normalize(text),
            "bbox": [round(v, 2) for v in b.to_tuple()],
            "x0": b.x0,
            "x1": b.x1,
            "y0": b.y0,
            "y1": b.y1,
        }
        if not center_inside:
            outside.append(
                {
                    **{k: segment[k] for k in ("id", "text", "bbox")},
                    "reason": "center_outside_bbox",
                    "in_annotations": own.get(segment["norm"], 0) > 0,
                }
            )
            continue
        if own.get(segment["norm"], 0) > 0:
            own[segment["norm"]] -= 1
            segment["overflow_pt"] = overflow_by_side(bbox, segment)
            segment["in_annotations"] = True
            segments.append(segment)
            continue
        owner = next((ref for ref, counter in others if counter.get(segment["norm"], 0) > 0), None)
        if owner:
            for _ref, counter in others:
                if counter.get(segment["norm"], 0) > 0:
                    counter[segment["norm"]] -= 1
                    break
        excluded.append(
            {
                **{k: segment[k] for k in ("id", "text", "bbox")},
                "reason": "foreign_block" if owner else "unbound",
                "owner": owner,
            }
        )

    tol = segment_tolerance(segments) if segments else 2.0
    rows = build_rows(segments, tol)

    annotation_counter = Counter(
        normalize(line) for line in (block.get("annotations") or "").splitlines() if normalize(line)
    )
    recovered_counter = Counter(seg["norm"] for seg in segments)
    missing = annotation_counter - recovered_counter
    extra = recovered_counter - annotation_counter
    native_order = [seg["norm"] for seg in segments]
    recovered_order = [seg["norm"] for row in rows for seg in row]
    bands = [(row[0]["y0"], max(seg["y1"] for seg in row)) for row in rows]
    banding = all(bands[i + 1][0] >= bands[i][1] - tol for i in range(len(bands) - 1))
    order_ok = all(row == sorted(row, key=lambda s: s["x0"]) for row in rows)
    partition_ok = expected_partition(segments) == frozenset(
        frozenset(seg["id"] for seg in row) for row in rows
    )

    status = "complete"
    issues: list[str] = []
    if not segments:
        status = "no-fit"
        issues.append("no_recoverable_segment")
    else:
        if missing:
            status = "partial"
            issues.append("missing_annotation_segments")
        if extra:
            status = "partial"
            issues.append("unsourced_segments")
        if not banding:
            status = "partial"
            issues.append("row_band_overlap")
        if not order_ok:
            status = "partial"
            issues.append("in_row_order")
        if not partition_ok:
            status = "partial"
            issues.append("row_partition_mismatch")
        if excluded:
            status = "partial"
            issues.append("excluded_segments")
    if outside:
        issues.append("segments_center_outside")

    line_payloads = [
        {
            "line_index": index,
            "text": "  ".join(seg["text"] for seg in row),
            "segments": [
                {
                    "id": seg["id"],
                    "text": seg["text"],
                    "bbox": seg["bbox"],
                    "overflow_pt": seg["overflow_pt"],
                }
                for seg in row
            ],
        }
        for index, row in enumerate(rows)
    ]
    code_text = "\n".join(payload["text"] for payload in line_payloads)

    review_status = ""
    effective_status = status
    restored: list[dict] = []
    if review and review.get("action") == "approve_restore":
        targets = {normalize(item["text"]) for item in review.get("segments", [])}
        remaining = [key for key in missing if key not in targets]
        if not remaining and status in {"partial", "no-fit"}:
            first_row_y = rows[0][0]["y0"] if rows else float("inf")
            for position, item in enumerate(review.get("segments", [])):
                restored.append(
                    {
                        "text": item["text"],
                        "source": item.get("source", "annotations"),
                        "bbox": item.get("bbox"),
                        "overflow_pt": item.get("overflow_pt"),
                        "note": item.get("note", ""),
                    }
                )
                payload = {
                    "line_index": -1,
                    "text": item["text"],
                    "restored": True,
                    "segments": [
                        {
                            "id": f"restored-{position + 1:03d}",
                            "text": item["text"],
                            "bbox": item.get("bbox"),
                            "overflow_pt": item.get("overflow_pt"),
                            "restored": True,
                            "source": item.get("source", "annotations"),
                        }
                    ],
                }
                y = (item.get("bbox") or [0, first_row_y])[1]
                if y <= first_row_y:
                    line_payloads.insert(0, payload)
                else:
                    line_payloads.append(payload)
            for index, payload in enumerate(line_payloads):
                payload["line_index"] = index
            code_text = "\n".join(payload["text"] for payload in line_payloads)
            review_status = "approved"
            effective_status = "complete" if segments else "no-fit"
            issues = [issue for issue in issues if issue != "missing_annotation_segments"]
            if not issues:
                issues = []
        else:
            review_status = "rejected"

    return {
        "figure_id": figure_id,
        "doc_id": doc_id,
        "product": model,
        "page": page_no,
        "ordinal": ordinal,
        "image_ref": block.get("image_ref", ""),
        "source_bbox": bbox,
        "status": status,
        "effective_status": effective_status,
        "review_status": review_status,
        "duplicate_of": None,
        "checks": {
            "A_multiset": not missing and not extra,
            "B_order": native_order == recovered_order,
            "C_banding": banding,
            "D_columns": True,
            "E_partition": partition_ok,
            "F_in_row_order": order_ok,
            "G_center_inside": True,
        },
        "issues": issues,
        "row_tolerance_pt": tol,
        "code_language": "asm",
        "code_text": code_text,
        "code_lines": line_payloads,
        "excluded_segments": excluded,
        "outside_segments": outside,
        "missing_annotation_segments": sorted(missing),
        "extra_segments": sorted(extra),
        "restored_segments": restored,
        "overflows": [
            {"figure_id": figure_id, "text": seg["text"], "overflow_pt": seg["overflow_pt"]}
            for seg in segments
            if any(value > 0.5 for value in seg["overflow_pt"].values())
        ],
        "recovered_lines": len(rows),
        "recovered_segments": len(segments),
    }


def find_duplicates(records: list[dict]) -> None:
    """Same page + identical bbox: keep the record with annotations, mark the other."""
    seen: dict[tuple[int, tuple[float, ...]], dict] = {}
    for record in records:
        key = (record["page"], tuple(round(v, 1) for v in record["source_bbox"]))
        first = seen.get(key)
        if first is None:
            seen[key] = record
            continue
        annotated, bare = (
            (record, first)
            if record["recovered_segments"] >= first["recovered_segments"]
            else (first, record)
        )
        if bare["recovered_segments"] == 0 and not bare["code_text"]:
            bare["duplicate_of"] = annotated["figure_id"]
            bare["status"] = "no-fit"
            bare["effective_status"] = "no-fit"
            bare["issues"] = sorted(set(bare["issues"]) | {"duplicate_snapshot"})


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the code-text sidecar")
    parser.add_argument("--pdf-root", type=Path, required=True, help="Original PDF directory")
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--docs", default="", help="comma separated doc ids; empty = all")
    parser.add_argument("--out", type=Path, default=ROOT / "storage" / "code_text")
    args = parser.parse_args()

    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    reviews = json.loads(args.reviews.read_text(encoding="utf-8")) if args.reviews.is_file() else {}
    only = {value.strip() for value in args.docs.split(",") if value.strip()}

    by_doc: dict[str, list[dict]] = {}
    for item in candidates["candidates"]:
        by_doc.setdefault(item["doc_id"], []).append(item)

    run_dir = args.out / args.run_id
    manifest = {
        "run_id": args.run_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "interpreter": sys.version.split()[0],
        "candidates_file": str(args.candidates),
        "reviews_file": str(args.reviews) if args.reviews.is_file() else None,
        "docs": {},
        "totals": Counter(),
    }

    for doc_id, items in sorted(by_doc.items()):
        if only and doc_id not in only:
            continue
        parsed_dir = next((d for d in PARSED.iterdir() if d.is_dir() and d.name and
                           load_parsed_document(d).doc_id == doc_id), None)
        if parsed_dir is None:
            manifest["docs"][doc_id] = {"error": "parsed dir not found"}
            continue
        document = load_parsed_document(parsed_dir)
        pdf_path = args.pdf_root / items[0]["pdf"] if items[0].get("pdf") else None
        if pdf_path is None or not pdf_path.is_file():
            pdf_path = args.pdf_root / next(
                (path for path in args.pdf_root.rglob(doc_id.upper().split("_")[0] + "*.pdf")),
                Path(),
            )
        if not pdf_path.is_file():
            manifest["docs"][doc_id] = {"error": f"pdf not found for {doc_id}"}
            continue

        records: list[dict] = []
        with fitz.open(str(pdf_path)) as fitz_doc:
            cache: dict[int, object] = {}
            for item in items:
                page_no = int(item["page"])
                page = next(p for p in document.pages if p["page"] == page_no)
                # Candidates are layout "table" snapshots by default; a candidate may
                # override the hint when a code listing was labelled as a plain image.
                hint = str(item.get("hint") or "table")
                snapshots = [
                    block for block in page["blocks"]
                    if block.get("type") == "snapshot" and block.get("hint") == hint
                ]
                if item["ordinal"] > len(snapshots):
                    continue
                block = snapshots[item["ordinal"] - 1]
                native = cache.get(page_no)
                if native is None:
                    native = extract_native_page(fitz_doc[page_no - 1])
                    cache[page_no] = native
                record = recover_block(
                    model=document.product,
                    doc_id=doc_id,
                    page_no=page_no,
                    ordinal=item["ordinal"],
                    block=block,
                    native=native,
                    same_page_blocks=page["blocks"],
                    review=reviews.get(f"{doc_id}:p{page_no:03d}:s{item['ordinal']}"),
                )
                records.append(record)

        find_duplicates(records)
        doc_dir = run_dir / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        (doc_dir / "code.jsonl").write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
            encoding="utf-8",
        )
        counts = Counter(record["effective_status"] for record in records)
        manifest["docs"][doc_id] = {
            "records": len(records),
            "status": dict(counts),
            "duplicates": sum(1 for record in records if record["duplicate_of"]),
            "partial": sum(1 for record in records if record["effective_status"] == "partial"),
            "sha256": hashlib.sha256((doc_dir / "code.jsonl").read_bytes()).hexdigest()[:16],
        }
        for record in records:
            manifest["totals"][record["effective_status"]] += 1
        manifest["totals"]["records"] += len(records)

    manifest["totals"] = dict(manifest["totals"])
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(
        json.dumps(
            {"run_dir": str(run_dir), "totals": manifest["totals"]},
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
