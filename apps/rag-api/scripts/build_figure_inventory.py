"""Build the full figure inventory from the parsed tree (plan §2 of the next batch).

Every snapshot in every ``pages.json`` is recorded with a stable ``figure_id`` and a
destination class.  The inventory does not start from existing figure chunks, and it
does not treat the earlier 65 non-code candidates as the whole input.

Classes:
    code_covered        already promoted to code by the code sidecar (effective complete)
    code_unresolved     sidecar record exists but stays partial / no-fit
    code_candidate      recovery passes and the figure is human-confirmed as code
    duplicate           same page + same bbox as an earlier snapshot
    missing_image       image_ref file is absent on disk
    to_describe         everything else: needs an image description
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.chunking.code_sidecar import load_code_sidecar  # noqa: E402
from app.chunking.loader import load_parsed_document  # noqa: E402

PARSED = ROOT / "storage" / "parsed"


def image_present(doc_dir: Path, image_ref: str) -> bool:
    return bool(image_ref) and (doc_dir / image_ref).is_file()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True, help="candidate chunk run dir")
    parser.add_argument(
        "--confirmed-code",
        type=Path,
        default=ROOT / "storage" / "code_text" / "confirmed_code.json",
        help="data file listing human-confirmed incremental code figures",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "storage" / "figure_inventory")
    args = parser.parse_args()

    confirmed = {}
    if args.confirmed_code.is_file():
        confirmed = json.loads(args.confirmed_code.read_text(encoding="utf-8"))

    sidecar = load_code_sidecar(args.sidecar)
    records = []
    totals: Counter = Counter()

    for doc_dir in sorted(PARSED.iterdir()):
        if not doc_dir.is_dir() or not (doc_dir / "pages.json").is_file():
            continue
        document = load_parsed_document(doc_dir)
        chunks_path = args.chunks / doc_dir.name / "chunks.jsonl"
        chunk_refs: dict[str, list[dict]] = defaultdict(list)
        if chunks_path.is_file():
            for line in chunks_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                content = (chunk.get("content") or "").replace("(", " ").replace(")", " ")
                for token in content.split():
                    if "snapshots/" in token:
                        chunk_refs[token.rsplit("/", 1)[-1]].append(
                            {"chunk_id": chunk["chunk_id"], "semantic_type": chunk["semantic_type"]}
                        )
        by_key = {
            (record.page, record.image_ref.rsplit("/", 1)[-1]): record
            for record in sidecar.get(document.doc_id, [])
        }
        seen_boxes: dict[tuple[int, tuple[float, ...]], str] = {}
        per_doc: Counter = Counter()

        for page in document.pages:
            ordinal = 0
            for block_index, block in enumerate(page.get("blocks") or []):
                if block.get("type") != "snapshot":
                    continue
                ordinal += 1
                figure_id = f'{document.doc_id}:p{page["page"]:03d}:f{ordinal}'
                bbox = tuple(round(value, 1) for value in (block.get("bbox") or []))
                record = by_key.get((page["page"], block.get("image_ref", "").rsplit("/", 1)[-1]))
                chunk_hits = chunk_refs.get(block.get("image_ref", "").rsplit("/", 1)[-1], [])
                if not image_present(doc_dir, block.get("image_ref", "")):
                    klass = "missing_image"
                elif (page["page"], bbox) in seen_boxes:
                    klass = "duplicate"
                elif record is not None and record.effective_status == "complete":
                    klass = "code_covered"
                elif record is not None:
                    klass = "code_unresolved"
                elif f'{document.doc_id}:p{page["page"]:03d}:s{ordinal}' in confirmed:
                    klass = "code_candidate"
                else:
                    klass = "to_describe"
                if klass != "missing_image" and (page["page"], bbox) not in seen_boxes:
                    seen_boxes[(page["page"], bbox)] = figure_id
                entry = {
                    "figure_id": figure_id,
                    "code_figure_id": f'{document.doc_id}:p{page["page"]:03d}:s{ordinal}',
                    "doc_id": document.doc_id,
                    "product": document.product,
                    "version": document.version,
                    "page": page["page"],
                    "block_index": block_index,
                    "hint": block.get("hint"),
                    "image_ref": block.get("image_ref", ""),
                    "bbox": list(bbox),
                    "section_id": block.get("section_id", ""),
                    "section_path": block.get("section_path") or [],
                    "annotation_chars": len(block.get("annotations") or ""),
                    "class": klass,
                    "duplicate_of": (
                        seen_boxes.get((page["page"], bbox)) if klass == "duplicate" else None
                    ),
                    "in_chunk": bool(chunk_hits),
                    "chunk_ids": [hit["chunk_id"] for hit in chunk_hits],
                    "chunk_semantic_types": sorted({hit["semantic_type"] for hit in chunk_hits}),
                    "has_code_chunk": any(hit["semantic_type"] == "code" for hit in chunk_hits),
                    "sidecar_status": record.status if record else None,
                    "sidecar_effective_status": record.effective_status if record else None,
                }
                records.append(entry)
                per_doc[klass] += 1
                totals[klass] += 1

        totals["documents"] += 1
        doc_figures = len([r for r in records if r["doc_id"] == document.doc_id])
        manifest_doc = {"figures": doc_figures, "classes": dict(per_doc)}
        totals.setdefault("per_doc", {})[document.doc_id] = manifest_doc  # type: ignore[assignment]

    run_dir = args.out / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    (run_dir / "inventory.jsonl").write_text(payload, encoding="utf-8")
    manifest = {
        "run_id": args.run_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "interpreter": sys.version.split()[0],
        "sidecar": str(args.sidecar),
        "chunks": str(args.chunks),
        "confirmed_code": str(args.confirmed_code) if args.confirmed_code.is_file() else None,
        "totals": {key: value for key, value in totals.items() if key != "per_doc"},
        "per_doc": totals.get("per_doc", {}),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    unchunked = [record for record in records if not record["in_chunk"]]
    unchunked_by_class = Counter(record["class"] for record in unchunked)
    figure_only_code = [
        record
        for record in records
        if record["class"] == "code_covered"
        and record["in_chunk"]
        and not record["has_code_chunk"]
    ]
    manifest["totals"]["unchunked_total"] = len(unchunked)
    manifest["totals"]["unchunked_by_class"] = dict(unchunked_by_class)
    manifest["totals"]["figure_only_code"] = len(figure_only_code)
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "totals": manifest["totals"],
                "figure_only_code": [record["figure_id"] for record in figure_only_code],
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    print(f"unchunked records: {len(unchunked)} -> {dict(unchunked_by_class)}")
    for record in unchunked[:40]:
        print(
            f'  {record["figure_id"]:34s} {record["class"]:12s} {record["image_ref"]}'
        )


if __name__ == "__main__":
    main()
