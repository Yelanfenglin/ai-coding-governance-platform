"""Exercise the real scoring CLI with gold fragments and deliberate corruptions.

Synthetic scorer checks are not runtime retrieval/answer evaluation.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

BANK = Path(__file__).resolve().parent
ROOT = BANK.parents[1]


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf8").splitlines() if x]


def dump_rows(path, data):
    path.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in data), encoding="utf8"
    )


@pytest.fixture
def bank_data():
    questions = rows(BANK / "questions.jsonl")
    corpus = json.loads((BANK / "corpus.json").read_text(encoding="utf8"))
    chunks = {}
    for source in corpus["sources"]:
        for c in rows(ROOT / source["chunk_file"]):
            text = c["embedding_text"] or c["content"]
            notes = [n for n in c.get("figure_uncertainties", []) if n and n not in text]
            if notes:
                text += "\n不确定项：" + "；".join(notes)
            chunks[c["chunk_id"]] = dict(
                product=source["product"], version=source["api_version"], text=text
            )
    mappings = {e["evidence_id"]: e for e in rows(BANK / "current_chunk_mapping.jsonl")}
    scoped = []
    for q in questions:
        hits = {}
        for e in q["evidence_requirements"]:
            for c in mappings[e["evidence_id"]]["candidates"]:
                hits[c["chunk_id"]] = dict(chunk_id=c["chunk_id"], **chunks[c["logical_chunk_id"]])
        scoped.append(dict(question_id=q["id"], hits=list(hits.values()), contexts=[]))
    return questions, scoped


def score(tmp_path, questions, results, split="all", allow_failure=False):
    selected = {q["id"] for q in questions if split == "all" or q["split"] == split}
    unscoped = {
        q["id"] for q in questions if q["evidence_requirements"] and len(q["allowed_sources"]) == 1
    }
    data = copy.deepcopy([r for r in results if r["question_id"] in selected])
    for result in data:
        result["contexts"] = [dict(chunk_id=h["chunk_id"], text=h["text"]) for h in result["hits"]]
    dump_rows(tmp_path / "scoped_results.jsonl", data)
    dump_rows(
        tmp_path / "unscoped_results.jsonl", [r for r in data if r["question_id"] in unscoped]
    )
    (tmp_path / "run.json").write_text(
        json.dumps(dict(run_type="synthetic_scorer_contract")), encoding="utf8"
    )
    command = [
        sys.executable,
        "-B",
        str(BANK / "score_results.py"),
        "--run-dir",
        str(tmp_path),
        "--split",
        split,
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf8",
        env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONUTF8": "1"},
        check=False,
    )
    if allow_failure:
        return result
    assert result.returncode == 0, result.stderr
    output = tmp_path / "rules_v2_2"
    return json.loads((output / "metrics.json").read_text(encoding="utf8")), rows(
        output / "judgments.jsonl"
    )


def test_all_gold_and_split_reports(tmp_path, bank_data):
    questions, results = bank_data
    metrics, _ = score(tmp_path, questions, results)
    assert metrics["status"] == "synthetic_scorer_contract"
    assert metrics["generation_not_run"] is True
    scoped = metrics["aggregates"]["scoped"]
    positive = [q for q in questions if q["evidence_requirements"]]
    assert scoped["stages"]["candidate@5"]["all_required_evidence"] == len(positive)
    assert scoped["stages"]["context"]["all_required_evidence"] == len(positive)
    for split in ("development", "holdout"):
        expected = sum(q["split"] == split for q in positive)
        assert scoped["splits"][split]["context"]["questions"] == expected
        assert scoped["splits"][split]["context"]["all_required_evidence"] == expected


@pytest.mark.parametrize(
    "qid,mutation",
    [
        ("R094", "wrong_product"),
        ("R089", "wrong_version"),
        ("R094", "duplicate_first_source"),
        ("R065", "missing_conflicting_source"),
        ("R091", "missing_code_page"),
        ("R089", "reorder_code"),
        ("R098", "missing_qualification"),
    ],
)
def test_corrupt_evidence_is_rejected(tmp_path, bank_data, qid, mutation):
    questions, results = bank_data
    result = next(r for r in results if r["question_id"] == qid)
    q = next(q for q in questions if q["id"] == qid)
    if mutation == "wrong_product":
        for h in result["hits"]:
            h["product"] = "M8P0000"
    elif mutation == "wrong_version":
        for h in result["hits"]:
            h["version"] = "V9.99"
    elif mutation == "duplicate_first_source":
        result["hits"] = [result["hits"][0], copy.deepcopy(result["hits"][0])]
        result["hits"][1]["chunk_id"] += "-duplicate"
    elif mutation == "missing_conflicting_source":
        result["hits"] = result["hits"][:1]
    elif mutation == "missing_code_page":
        required = q["evidence_requirements"][-1]["required_code_text"]
        for h in result["hits"]:
            h["text"] = h["text"].replace(required, "[续页缺失]")
    elif mutation == "reorder_code":
        required = q["evidence_requirements"][0]["required_code_text"]
        shuffled = "\n".join(reversed(required.splitlines()))
        for h in result["hits"]:
            h["text"] = h["text"].replace(required, shuffled)
    elif mutation == "missing_qualification":
        for h in result["hits"]:
            h["text"] = h["text"].replace("具体值不做设计保证", "[说明缺失]")
    _, judgments = score(tmp_path, questions, results)
    actual = next(
        j
        for j in judgments
        if j["question_id"] == qid and j["track"] == "scoped" and j["stage"] == "context"
    )
    assert actual["all_required_evidence"] is False
    if mutation == "duplicate_first_source":
        assert actual["covered_groups"] == 1
        assert actual["required_groups"] == 2


def test_code_spacing_does_not_change_score(tmp_path, bank_data):
    questions, results = bank_data
    target = next(r for r in results if r["question_id"] == "R090")
    for hit in target["hits"]:
        hit["text"] = "\n".join("  " + line.replace(" ", "\t") for line in hit["text"].splitlines())
    _, judgments = score(tmp_path, questions, results)
    assert all(j["all_required_evidence"] for j in judgments if j["question_id"] == "R090")


def test_holdout_only_run(tmp_path, bank_data):
    questions, results = bank_data
    metrics, _ = score(tmp_path, questions, results, split="holdout")
    assert metrics["selected_split"] == "holdout"
    assert set(metrics["aggregates"]["scoped"]["splits"]) == {"holdout"}


@pytest.mark.parametrize("mutation", ["omit", "duplicate"])
def test_partial_or_duplicate_runs_cannot_shrink_denominator(tmp_path, bank_data, mutation):
    questions, results = bank_data
    if mutation == "omit":
        results.pop()
    else:
        results.append(copy.deepcopy(results[-1]))
    result = score(tmp_path, questions, results, allow_failure=True)
    assert result.returncode != 0
    assert "incomplete or duplicate run" in result.stderr
