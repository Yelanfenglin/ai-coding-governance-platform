"""Chunk statistics + invariants (Step 4 acceptance tool).

Usage:
  python scripts/audit_chunks.py storage/parsed/M8P5170 [storage/parsed/...]
      [--out storage/chunks]   # also write chunks.jsonl per doc
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chunking import build_chunks, build_semantic_units, load_parsed_document  # noqa: E402
from app.chunking.chunker import NORMAL_MAX, SOFT_MIN  # noqa: E402

_HEADING_LINE = __import__("re").compile(r"^\s*\d+(?:\.\d+){0,5}\s+\S")


def _percentiles(values: list[int]) -> dict:
    if not values:
        return {"p10": 0, "p50": 0, "p90": 0, "p95": 0, "max": 0}
    ordered = sorted(values)
    n = len(ordered)
    return {
        "p10": ordered[min(int(n * 0.10), n - 1)],
        "p50": ordered[min(int(n * 0.50), n - 1)],
        "p90": ordered[min(int(n * 0.90), n - 1)],
        "p95": ordered[min(int(n * 0.95), n - 1)],
        "max": ordered[-1],
    }


def audit(
    doc_dir: Path,
    code_sidecar: dict | None = None,
    figure_descriptions: dict | None = None,
) -> dict:
    doc = load_parsed_document(doc_dir)
    units = build_semantic_units(
        doc, code_sidecar=code_sidecar, figure_descriptions=figure_descriptions
    )
    chunks = build_chunks(doc, units)

    all_titles = {" ".join(u.title.split()) for u in units if u.title}

    types = Counter(c.metadata.semantic_type for c in chunks)
    embedded = [c for c in chunks if not c.exclude_from_embedding]

    errors: list[str] = []
    # invariants
    for chunk in chunks:
        if chunk.metadata.semantic_type == "toc" and not chunk.exclude_from_embedding:
            errors.append(f"{chunk.chunk_id}: toc not excluded from embedding")
        if chunk.metadata.semantic_type == "table":  # including attached conditions/notes
            if not chunk.metadata.table_id:
                errors.append(f"{chunk.chunk_id}: table chunk without table_id (orphan)")
            if not chunk.metadata.row_group:
                errors.append(f"{chunk.chunk_id}: table chunk without row_group (orphan)")
        if chunk.metadata.register and chunk.metadata.register.split()[0] not in chunk.content:
            errors.append(f"{chunk.chunk_id}: register chunk misses its register table")
        if chunk.metadata.semantic_type == "text":
            for line in chunk.content.splitlines():
                stripped = " ".join(line.split())
                if stripped in all_titles and stripped != " ".join(chunk.metadata.title.split()):
                    errors.append(
                        f"{chunk.chunk_id}: text chunk crosses a heading line: {stripped}"
                    )
        if chunk.exclude_from_embedding:
            continue
        if not chunk.metadata.source_pages:
            errors.append(f"{chunk.chunk_id}: empty source_pages")
        if not chunk.parent_id:
            errors.append(f"{chunk.chunk_id}: missing parent_id")

    oversize = [c for c in embedded if c.metadata.atomic and c.tokens > NORMAL_MAX]
    undersize = [c for c in embedded if c.tokens < SOFT_MIN]
    tok = [c.tokens for c in embedded]

    return {
        "doc": doc.product,
        "version": doc.version,
        "units": len(units),
        "chunks": len(chunks),
        "embedded": len(embedded),
        "types": dict(types),
        "tokens": _percentiles(tok),
        "oversize_atomic": len(oversize),
        "undersize_semantic": len(undersize),
        "errors": errors,
        "chunks_all": chunks,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="also write chunks.jsonl per doc")
    ap.add_argument(
        "--code-sidecar",
        type=Path,
        default=None,
        help="run directory of the code-text sidecar (storage/code_text/<run_id>)",
    )
    ap.add_argument(
        "--figure-descriptions",
        type=Path,
        default=None,
        help="run directory of figure descriptions (storage/figure_descriptions/<run_id>)",
    )
    args = ap.parse_args()

    code_sidecar = None
    if args.code_sidecar is not None:
        from app.chunking.code_sidecar import load_code_sidecar

        code_sidecar = load_code_sidecar(args.code_sidecar)

    figure_descriptions = None
    if args.figure_descriptions is not None:
        from app.chunking.figure_descriptions import load_figure_descriptions

        figure_descriptions = load_figure_descriptions(args.figure_descriptions)

    failed = False
    for doc_dir in args.outputs:
        report = audit(
            doc_dir,
            code_sidecar=code_sidecar,
            figure_descriptions=figure_descriptions,
        )
        t = report["tokens"]
        print(f"{report['doc']} {report['version']}")
        print("-" * 40)
        print(f"chunks                {report['chunks']}")
        for k, v in sorted(report["types"].items()):
            print(f"  {k:<20} {v}")
        print(f"tokens  p10={t['p10']} p50={t['p50']} p90={t['p90']} p95={t['p95']} max={t['max']}")
        print(f"oversize atomic       {report['oversize_atomic']}")
        print(f"undersize semantic    {report['undersize_semantic']}")
        if report["errors"]:
            failed = True
            print("INVARIANTS FAIL:")
            for e in report["errors"][:20]:
                print(f"  - {e}")
        else:
            print("invariants           OK")
        if args.out:
            out_dir = Path(args.out) / report["doc"]
            out_dir.mkdir(parents=True, exist_ok=True)
            with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
                for chunk in report["chunks_all"]:
                    f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            print(f"written              {out_dir / 'chunks.jsonl'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
