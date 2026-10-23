"""Score a frozen retrieval run with bank v2.1 rules.

Usage:
  python score_results.py --run-dir output/retrieval_top5_no_rerank_20261019 \
                          [--out-dir output/rules_v2_1_rescore_20261019/top5]

Reads the run's scoped_results.jsonl / unscoped_results.jsonl and writes
judgments.jsonl, failures.json, judgment_summary.json, metrics.json,
run_manifest.json. No model/API calls, no repository input changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

BANK = Path(__file__).resolve().parent
KS = (1, 3, 5, 8, 10, 30)


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def norm(text):
    return "".join(text.split())


def code_lines(text):
    return (
        "\n"
        + "\n".join(
            re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines() if line.strip()
        )
        + "\n"
    )


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    out = (args.out_dir or (run_dir / "rules_v2_1")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    questions = {q["id"]: q for q in rows(BANK / "questions.jsonl")}
    corpus = json.loads((BANK / "corpus.json").read_text(encoding="utf-8"))["sources"]
    sources = {s["source_id"]: (s["product"], s["api_version"]) for s in corpus}
    mapping = {row["evidence_id"]: row for row in rows(BANK / "current_chunk_mapping.jsonl")}
    rules = json.loads((BANK / "evidence_rules.json").read_text(encoding="utf-8"))
    expected = {e["evidence_id"] for q in questions.values() for e in q["evidence_requirements"]}
    code_ids = {
        e["evidence_id"]
        for q in questions.values()
        for e in q["evidence_requirements"]
        if "required_code_text" in e
    }
    assert expected == set(rules["facts"]) | code_ids

    def source_allowed(fragment, allowed):
        pairs = {sources[key] for key in allowed}
        return (fragment["product"], fragment["version"]) in pairs

    def evidence(e, fragments):
        allowed = {locator["source_id"] for locator in e["locators"]}
        usable = [f for f in fragments if source_allowed(f, allowed)]
        eid = e["evidence_id"]
        if "required_code_text" in e:
            required = code_lines(e["required_code_text"])
            matching = [
                f["chunk_id"] for f in usable if required in code_lines(f["text"])
            ]
            return dict(
                covered=bool(matching),
                supporting_hit_ids=matching,
                method="complete_contiguous_ordered_code_lines",
                required_lines=len(e["required_code_text"].splitlines()),
            )
        matches = []
        for alternatives in rules["facts"][eid]:
            proofs = [
                dict(chunk_id=f["chunk_id"], quote=quote)
                for f in usable
                for quote in alternatives
                if norm(quote) in norm(f["text"])
            ]
            matches.append(dict(alternatives=alternatives, proofs=proofs))
        covered = all(match["proofs"] for match in matches)
        supporting = sorted({proof["chunk_id"] for match in matches for proof in match["proofs"]})
        return dict(
            covered=covered,
            supporting_hit_ids=supporting,
            method="reviewed_full_fact_clauses",
            clause_matches=matches,
        )

    judgments = []
    aggregates = {}
    boundary = []
    failures = []
    for track in ("scoped", "unscoped"):
        run_rows = rows(run_dir / f"{track}_results.jsonl")
        per_stage = defaultdict(list)
        by_category = defaultdict(list)
        truncations = []
        for result in run_rows:
            q = questions[result["question_id"]]
            if not q["evidence_requirements"]:
                boundary.append(
                    dict(
                        question_id=q["id"],
                        track=track,
                        expected_behavior=q["expected_behavior"],
                        retrieved=len(result["hits"]),
                        strict_scope_pass=not result["hits"]
                        if q["category"] == "scope_boundary"
                        else None,
                        answer_behavior_pass=None,
                        generation_called=False,
                    )
                )
                continue
            hit_map = {hit["chunk_id"]: hit for hit in result["hits"]}
            context = [
                dict(hit_map[fragment["chunk_id"]], text=fragment["text"])
                for fragment in result["contexts"]
            ]
            truncated = [
                fragment["chunk_id"]
                for fragment in context
                if fragment["text"] != hit_map[fragment["chunk_id"]]["text"]
            ]
            if truncated:
                truncations.append(dict(question_id=q["id"], chunk_ids=truncated))
            stages = [(f"candidate@{k}", result["hits"][:k], k) for k in KS]
            stages.append(("context", context, None))
            for stage, fragments, k in stages:
                verdicts = {
                    e["evidence_id"]: evidence(e, fragments) for e in q["evidence_requirements"]
                }
                values = [verdict["covered"] for verdict in verdicts.values()]
                wrong = [
                    fragment["chunk_id"]
                    for fragment in fragments
                    if not source_allowed(fragment, q["allowed_sources"])
                ]
                required_image_ids = []
                if q["category"] == "figure_semantics":
                    required_image_ids = [
                        [
                            candidate["chunk_id"]
                            for candidate in mapping[e["evidence_id"]]["candidates"]
                        ]
                        for e in q["evidence_requirements"]
                    ]
                ids = {fragment["chunk_id"] for fragment in fragments}
                image_verdicts = [bool(ids.intersection(group)) for group in required_image_ids]
                judgment = dict(
                    question_id=q["id"],
                    track=track,
                    stage=stage,
                    k=k,
                    category=q["category"],
                    evidence_verdicts=verdicts,
                    any_complete_group=any(values),
                    all_required_evidence=all(values),
                    covered_groups=sum(values),
                    required_groups=len(values),
                    evidence_coverage=sum(values) / len(values),
                    wrong_source_hit_ids=wrong,
                    returned_fragments=len(fragments),
                    single_source_question=len(q["allowed_sources"]) == 1,
                    specified_image_locators=image_verdicts,
                    answer_behavior_pass=None,
                )
                judgments.append(judgment)
                per_stage[stage].append(judgment)
                by_category[(stage, q["category"])].append(judgment)
                if stage in {"candidate@30", "context"} and not all(values):
                    failures.append(
                        dict(
                            question_id=q["id"],
                            track=track,
                            stage=stage,
                            missing_evidence_ids=[
                                eid for eid, verdict in verdicts.items() if not verdict["covered"]
                            ],
                        )
                    )

        def summarize(group):
            singles = [row for row in group if row["single_source_question"]]
            image_rows = [row for row in group if row["specified_image_locators"]]
            codes = [row for row in group if row["category"] == "code_example"]
            return dict(
                questions=len(group),
                any_evidence_hit=sum(row["any_complete_group"] for row in group),
                all_required_evidence=sum(row["all_required_evidence"] for row in group),
                macro_evidence_coverage=sum(row["evidence_coverage"] for row in group) / len(group),
                covered_groups=sum(row["covered_groups"] for row in group),
                required_groups=sum(row["required_groups"] for row in group),
                single_source_wrong_hits=sum(len(row["wrong_source_hit_ids"]) for row in singles),
                single_source_returned_hits=sum(row["returned_fragments"] for row in singles),
                single_source_empty_results=sum(row["returned_fragments"] == 0 for row in singles),
                all_specified_images_located=sum(
                    all(row["specified_image_locators"]) for row in image_rows
                ),
                image_questions=len(image_rows),
                complete_code=sum(row["all_required_evidence"] for row in codes),
                code_questions=len(codes),
            )

        aggregates[track] = dict(
            stages={key: summarize(value) for key, value in per_stage.items()},
            categories={
                stage: {
                    category: summarize(group)
                    for (s, category), group in by_category.items()
                    if s == stage
                }
                for stage in per_stage
            },
            context_truncations=truncations,
        )
    with (out / "judgments.jsonl").open("w", encoding="utf-8") as stream:
        for judgment in judgments:
            stream.write(json.dumps(judgment, ensure_ascii=False) + "\n")
    paired_ids = {row["question_id"] for row in rows(run_dir / "unscoped_results.jsonl")}
    paired = {
        track: {
            stage: summarize(
                [
                    judgment
                    for judgment in judgments
                    if judgment["track"] == track
                    and judgment["stage"] == stage
                    and judgment["question_id"] in paired_ids
                ]
            )
            for stage in aggregates[track]["stages"]
        }
        for track in ("scoped", "unscoped")
    }
    dump(
        out / "metrics.json",
        dict(
            status="reviewed_development_baseline",
            independent_review="pending",
            bank_version=rules["bank_version"],
            rules_version=rules["schema_version"],
            method=rules["method"],
            no_rerank=json.loads((run_dir / "run.json").read_text(encoding="utf-8")).get(
                "reranker_enabled"
            )
            is False,
            generation_not_run=True,
            aggregates=aggregates,
            paired_single_model_questions=paired,
            boundary=boundary,
        ),
    )
    dump(out / "failures.json", failures)
    dump(
        out / "judgment_summary.json",
        {track: data["stages"] for track, data in aggregates.items()},
    )
    dump(
        out / "run_manifest.json",
        dict(
            scored_at=datetime.now(UTC).isoformat(),
            bank_version=rules["bank_version"],
            rules_sha256=sha256(BANK / "evidence_rules.json"),
            questions_sha256=sha256(BANK / "questions.jsonl"),
            run_dir=str(run_dir),
            scoped_results_sha256=sha256(run_dir / "scoped_results.jsonl"),
            unscoped_results_sha256=sha256(run_dir / "unscoped_results.jsonl"),
            judgments=len(judgments),
            failures=len(failures),
            model_api_calls=0,
            database_writes=0,
        ),
    )
    print(
        json.dumps(
            {
                "out": str(out),
                "scoped": aggregates["scoped"]["stages"]["context"],
                "unscoped": aggregates["unscoped"]["stages"]["context"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
