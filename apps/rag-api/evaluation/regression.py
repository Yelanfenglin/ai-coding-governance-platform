"""Offline regression checklist for the three known failure shapes.

Run it by hand after any retrieval/selection change:

    python -m evaluation.runner regression

It is **not** CI: no daemon, no hooks, no blocking.  It reads the frozen bank plus
whatever run directories exist and prints one table with the verdict per case.

Cases
-----
``R050``  selection / multi-evidence coverage
    Four evidence groups (IOAICR4, INTCR0.IOAIE, OPTION.GIE, INTF0.IOAIF).  The
    question is whether the *selected contexts* cover all four.  At
    ``context_top_n=5`` the gold chunks sit at rank 6/7 and cannot be covered; at
    10 they are.  The case therefore carries an expectation per budget.

``R056``  source-table conflict
    One citation declares ``寄存器：PUA`` but also carries the ``PUB`` table whose
    Bit7 is ``PDA2``.  The deterministic detector must still report it.  If the
    finding disappears, either the citation set changed (re-baseline this case) or
    the detector regressed — the script says which, it does not guess.

``R055``  multi-source recall (substitute for holdout R093)
    Cross-product question with two evidence groups.  Same coverage logic as R050.
    R093 itself is holdout and deliberately not touched.

``baseline``  drift check
    Compares the recorded metrics of the 2026-10-20 baselines against frozen
    expected numbers, so a silent regression in dense/hybrid retrieval shows up
    without spending a single embedding call.

Exit code: 0 when every case matches its expectation, 1 otherwise.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "regression_set_20261020"
BANK = ROOT / "evaluation" / "retrieval"
sys.path.insert(0, str(ROOT))

from app.evidence_checks import Citation, detect_findings  # noqa: E402

# run label -> scoped results file
RUNS = {
    "v2_dense@5": ROOT
    / "output"
    / "retrieval_baseline_20261020"
    / "v2_dense"
    / "scoped_results.jsonl",
    "v2_dense@10": OUT.parent / "context_budget_20261020" / "ctx10_dense" / "scoped_results.jsonl",
}

# Frozen numbers, taken from an actual replay of output/retrieval_baseline_20261020 at
# 2026-10-20 11:2x, verified identical on both the pre-PR3 and post-PR3 code.
# v1_dense/unscoped was 68/65 in the 09:2x run; the replay (both code versions) gives
# 69/66, so the old pair was stale — re-baselined here instead of left failing.
BASELINE_EXPECTED = {
    "v1_dense": {"scoped_any": 82, "scoped_all": 82, "unscoped_any": 69, "unscoped_all": 66},
    "v2_dense": {"scoped_any": 82, "scoped_all": 82, "unscoped_any": 68, "unscoped_all": 65},
    "v1_bm25": {"scoped_any": 87, "scoped_all": 87, "unscoped_any": 77, "unscoped_all": 74},
    "v2_bm25": {"scoped_any": 87, "scoped_all": 87, "unscoped_any": 77, "unscoped_all": 74},
}
BASELINE_RUNS = {
    name: ROOT / "output" / "retrieval_baseline_20261020" / name / "rules_v2_2" / "metrics.json"
    for name in BASELINE_EXPECTED
}


@dataclass(frozen=True, slots=True)
class Case:
    question: str
    kind: str  # "coverage" | "conflict"
    expectation: dict[str, str]  # run label -> "pass" | "fail"
    note: str


CASES = (
    Case(
        question="R050",
        kind="coverage",
        expectation={"v2_dense@5": "fail", "v2_dense@10": "pass"},
        note="四层中断依据；rank 6/7，只有加预算才装得下",
    ),
    Case(
        question="R056",
        kind="conflict",
        expectation={"v2_dense@5": "pass", "v2_dense@10": "pass"},
        note="PUA 表头 + PUB 表（PDA2），检测器必须仍然报出",
    ),
    Case(
        question="R055",
        kind="coverage",
        expectation={"v2_dense@5": "fail", "v2_dense@10": "pass"},
        note="跨型号对比；替代 holdout R093",
    ),
)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def gold_groups() -> dict[str, dict[str, list[str]]]:
    groups: dict[str, dict[str, list[str]]] = {}
    for entry in rows(BANK / "current_chunk_mapping.jsonl"):
        groups.setdefault(entry["question_id"], {})[entry["evidence_id"]] = [
            candidate["chunk_id"] for candidate in entry["candidates"]
        ]
    return groups


def coverage_verdict(record: dict, groups: dict[str, list[str]]) -> tuple[bool, dict]:
    selected = set(record["selected_chunk_ids"])
    missing = [key for key, ids in groups.items() if not any(i in selected for i in ids)]
    order = [hit["chunk_id"] for hit in record["hits"]]
    ranks = {
        key: min((order.index(i) + 1 for i in ids if i in order), default=None)
        for key, ids in groups.items()
    }
    return (not missing), {"selected": len(selected), "missing": missing, "gold_ranks": ranks}


def conflict_verdict(record: dict) -> tuple[bool, dict]:
    by_chunk = {hit["chunk_id"]: hit["section_path"] for hit in record["hits"]}
    citations = [
        Citation(
            chunk_id=context["chunk_id"],
            text=context["text"],
            section_path=by_chunk.get(context["chunk_id"], ""),
        )
        for context in record["contexts"]
    ]
    findings = detect_findings(citations)
    kinds = sorted({finding.kind for finding in findings})
    return bool(findings), {"kinds": kinds, "details": [f.detail for f in findings][:3]}


def baseline_verdict() -> list[dict]:
    results: list[dict] = []
    for name, expected in BASELINE_EXPECTED.items():
        path = BASELINE_RUNS[name]
        if not path.is_file():
            results.append({"run": name, "status": "missing", "path": str(path)})
            continue
        aggregates = json.loads(path.read_text(encoding="utf-8"))["aggregates"]

        def at(track: str, aggregates: dict = aggregates) -> dict:
            return aggregates[track]["stages"]["candidate@30"]

        actual = {
            "scoped_any": at("scoped")["any_evidence_hit"],
            "scoped_all": at("scoped")["all_required_evidence"],
            "unscoped_any": at("unscoped")["any_evidence_hit"],
            "unscoped_all": at("unscoped")["all_required_evidence"],
        }
        diff = {
            key: {"expected": expected[key], "actual": actual[key]}
            for key in expected
            if expected[key] != actual[key]
        }
        results.append({"run": name, "status": "drift" if diff else "match", "diff": diff})
    return results


def main() -> int:
    groups = gold_groups()
    records = {
        label: {record["question_id"]: record for record in rows(path)}
        for label, path in RUNS.items()
        if path.is_file()
    }
    missing_runs = [label for label, path in RUNS.items() if not path.is_file()]

    report: dict[str, object] = {
        "runs_missing": missing_runs,
        "cases": [],
        "baseline": baseline_verdict(),
        "counts": {"ok": 0, "changed": 0, "skip": 0},
    }
    failures = 0
    counts = report["counts"]
    assert isinstance(counts, dict)

    print(f"{'case':6s} {'run':12s} {'expected':9s} {'actual':7s} verdict")
    for case in CASES:
        for label, expected in case.expectation.items():
            record = records.get(label, {}).get(case.question)
            if record is None:
                failures += 1
                counts["skip"] += 1
                print(f"{case.question:6s} {label:12s} {expected:9s} {'no-run':7s} SKIP")
                report["cases"].append(
                    {
                        "case": case.question,
                        "run": label,
                        "expected": expected,
                        "actual": "no-run",
                        "status": "skip",
                    }
                )
                continue
            if case.kind == "coverage":
                ok, detail = coverage_verdict(record, groups.get(case.question, {}))
            else:
                ok, detail = conflict_verdict(record)
            actual = "pass" if ok else "fail"
            status = "ok" if actual == expected else "changed"
            counts[status] += 1
            if status != "ok":
                failures += 1
            print(f"{case.question:6s} {label:12s} {expected:9s} {actual:7s} {status}")
            report["cases"].append(
                {
                    "case": case.question,
                    "note": case.note,
                    "run": label,
                    "expected": expected,
                    "actual": actual,
                    "status": status,
                    "detail": detail,
                }
            )

    print()
    for item in report["baseline"]:
        print(f"baseline {item['run']:9s} {item['status']}")
        if item["status"] == "missing":
            failures += 1
            counts["skip"] += 1
            print(f"          missing: {item['path']}")
        elif item["status"] == "drift":
            failures += 1
            counts["changed"] += 1
            print(f"          {json.dumps(item['diff'], ensure_ascii=False)}")

    report["failures"] = failures
    report["result"] = "OK" if failures == 0 else "FAIL"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print()
    print("RESULT:", "OK" if failures == 0 else f"{failures} check(s) failed — inspect result.json")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
