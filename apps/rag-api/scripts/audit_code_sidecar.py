"""Audit the code-text sidecar and prove the checker rejects known-bad results.

Two parts:

1. Structural audit of every sidecar record, recomputed from the stored data
   (annotations multiset, line order, row partition, in-row order, duplicate blocks).
2. Negative controls mutate segment occurrences, row assignments and rendered text;
   provenance-only validation must not accept altered text consumed by the chunker.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.chunking.loader import load_parsed_document  # noqa: E402

PARSED = ROOT / "storage" / "parsed"


def doc_dirs() -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for candidate in PARSED.iterdir():
        if candidate.is_dir() and (candidate / "pages.json").is_file():
            mapping[load_parsed_document(candidate).doc_id] = candidate
    return mapping


def normalize(text: str) -> str:
    return "".join((text or "").split())


def load_records(run_dir: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for doc_dir in sorted(run_dir.iterdir()):
        path = doc_dir / "code.jsonl"
        if doc_dir.is_dir() and path.is_file():
            lines = path.read_text(encoding="utf-8").splitlines()
            out[doc_dir.name] = [json.loads(line) for line in lines if line.strip()]
    return out


def annotations_for(doc_model_dir: Path, page_no: int, image_ref: str) -> Counter:
    pages = json.loads((doc_model_dir / "pages.json").read_text(encoding="utf-8"))
    page = next(p for p in pages if p["page"] == page_no)
    block = next(
        b for b in page["blocks"] if b.get("type") == "snapshot" and b.get("image_ref") == image_ref
    )
    return Counter(
        normalize(line) for line in (block.get("annotations") or "").splitlines() if normalize(line)
    )


def partition_of(lines: list[list[dict]]) -> frozenset[frozenset[str]]:
    return frozenset(frozenset(seg["id"] for seg in line) for line in lines)


def vertical_partition(lines: list[list[dict]]) -> frozenset[frozenset[str]]:
    segments = [(seg["id"], seg["bbox"]) for line in lines for seg in line]
    parent = {seg_id: seg_id for seg_id, _ in segments}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for i, (left_id, left) in enumerate(segments):
        for right_id, right in segments[i + 1 :]:
            top = max(left[1], right[1])
            bottom = min(left[3], right[3])
            if bottom <= top:
                continue
            shorter = min(left[3] - left[1], right[3] - right[1]) or 1.0
            if (bottom - top) / shorter > 0.5:
                ri, rj = find(left_id), find(right_id)
                if ri != rj:
                    parent[rj] = ri
    groups: dict[str, set[str]] = {}
    for seg_id, _ in segments:
        groups.setdefault(find(seg_id), set()).add(seg_id)
    return frozenset(frozenset(group) for group in groups.values())


def check_lines(record: dict, lines: list[list[dict]], annotations: Counter) -> list[str]:
    problems: list[str] = []
    segments = [seg for line in lines for seg in line]
    recovered = Counter(normalize(seg["text"]) for seg in segments)
    if annotations - recovered:
        problems.append("A_multiset_missing")
    if recovered - annotations:
        problems.append("A_multiset_extra")
    ordered = sorted(segments, key=lambda seg: (seg["bbox"][1], seg["bbox"][0]))
    if [normalize(seg["text"]) for seg in ordered] != [normalize(seg["text"]) for seg in segments]:
        problems.append("B_order")
    if partition_of(lines) != vertical_partition(lines):
        problems.append("E_partition")
    for line in lines:
        if [seg["bbox"][0] for seg in line] != sorted(seg["bbox"][0] for seg in line):
            problems.append("F_in_row_order")
            break
    if any(not line for line in lines):
        problems.append("empty_line")
    return problems


def check_rendered_text(record: dict) -> list[str]:
    """Validate the text consumed by the chunker against its stored provenance."""
    rows = record.get("code_lines") or []
    problems: list[str] = []
    if "\n".join(row["text"] for row in rows) != record.get("code_text", ""):
        problems.append("H_rendered_text")
    for row in rows:
        if normalize(row["text"]) != normalize("".join(s["text"] for s in row["segments"])):
            problems.append("H_line_text")
            break
    return problems


def mutations(lines: list[list[dict]]) -> list[tuple[str, list[list[dict]]]]:
    out: list[tuple[str, list[list[dict]]]] = []
    seen: Counter = Counter()
    for r_index, line in enumerate(lines):
        for s_index, seg in enumerate(line):
            seen[normalize(seg["text"])] += 1
            if seen[normalize(seg["text"])] == 2:
                mutated = copy.deepcopy(lines)
                del mutated[r_index][s_index]
                if not mutated[r_index]:
                    del mutated[r_index]
                out.append(("m1_drop_repeated", mutated))
                break
        if out:
            break
    if len(lines) >= 2:
        mutated = copy.deepcopy(lines)
        mutated[0] = sorted(mutated[0] + mutated[1], key=lambda seg: seg["bbox"][0])
        del mutated[1]
        out.append(("m2_merge_rows", mutated))

        mutated = copy.deepcopy(lines)
        if len(mutated[0]) >= 2:
            moved = mutated[0].pop()
            mutated[1] = sorted(mutated[1] + [moved], key=lambda seg: seg["bbox"][0])
            out.append(("m3_move_segment", mutated))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    records_by_doc = load_records(args.run)
    dirs = doc_dirs()
    problems: list[str] = []
    checked = 0
    duplicates = 0
    for doc_id, records in records_by_doc.items():
        parsed_dir = dirs.get(doc_id)
        if parsed_dir is None:
            problems.append(f"{doc_id}: parsed dir not found")
            continue
        seen_boxes: dict[tuple[int, tuple[float, ...]], str] = {}
        for record in records:
            checked += 1
            annotations = annotations_for(parsed_dir, record["page"], record["image_ref"])
            lines = [
                [
                    {"id": seg["id"], "text": seg["text"], "bbox": seg["bbox"]}
                    for seg in line["segments"]
                ]
                for line in record.get("code_lines", [])
            ]
            if lines:
                problems.extend(
                    f'{record["figure_id"]}: {problem}'
                    for problem in check_lines(record, lines, annotations)
                )
                problems.extend(
                    f'{record["figure_id"]}: {problem}' for problem in check_rendered_text(record)
                )
            if record.get("restored_segments"):
                for restored in record["restored_segments"]:
                    if normalize(restored["text"]) not in {
                        normalize(line.split("  ")[0]) for line in record["code_text"].splitlines()
                    } and normalize(restored["text"]) not in normalize(record["code_text"]):
                        problems.append(f'{record["figure_id"]}: restored_segment_missing')
            key = (record["page"], tuple(round(v, 1) for v in record["source_bbox"]))
            if key in seen_boxes:
                duplicates += 1
                if not record.get("duplicate_of"):
                    problems.append(f'{record["figure_id"]}: duplicate_without_marker')
            else:
                seen_boxes[key] = record["figure_id"]

    print(f"records checked: {checked}")
    print(f"duplicate bbox groups: {duplicates}")
    print(f"problems: {len(problems)}")
    for problem in problems[:20]:
        print("  -", problem)

    # negative controls on the first complete record that has a repeated segment
    rejected: dict[str, list[str]] = {}
    for doc_id, records in records_by_doc.items():
        parsed_dir = dirs.get(doc_id)
        if parsed_dir is None:
            continue
        for record in records:
            lines = [
                [
                    {"id": seg["id"], "text": seg["text"], "bbox": seg["bbox"]}
                    for seg in line["segments"]
                ]
                for line in record.get("code_lines", [])
            ]
            if not lines:
                continue
            annotations = annotations_for(parsed_dir, record["page"], record["image_ref"])
            baseline = check_lines(record, lines, annotations)
            if baseline:
                continue
            cases = mutations(lines)
            if len(cases) < 3:
                continue
            for name, mutated in cases:
                caught = check_lines(record, mutated, annotations)
                rejected[name] = caught
            altered_text = copy.deepcopy(record)
            altered_text["code_text"] += "\nMOVIA 0x00"
            rejected["m4_alter_rendered_text"] = check_rendered_text(altered_text)
            altered_line = copy.deepcopy(record)
            altered_line["code_lines"][0]["text"] += " BAD"
            altered_line["code_text"] = "\n".join(
                row["text"] for row in altered_line["code_lines"]
            )
            rejected["m5_alter_line_text"] = check_rendered_text(altered_line)
            print(f'negative controls on {record["figure_id"]}:')
            for name, caught in rejected.items():
                print(f'   {name:16s} -> rejected_by={caught or "NONE"}')
            break
        if rejected:
            break

    all_rejected = bool(rejected) and all(caught for caught in rejected.values())
    print(f"negative controls all rejected: {all_rejected}")
    report = {
        "records_checked": checked,
        "duplicate_groups": duplicates,
        "problems": problems,
        "negative_controls": rejected,
        "negative_controls_all_rejected": all_rejected,
    }
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report -> {args.report}")
    return 0 if not problems and all_rejected else 1


if __name__ == "__main__":
    raise SystemExit(main())
