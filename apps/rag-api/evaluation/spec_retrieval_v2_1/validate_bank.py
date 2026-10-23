"""Validate frozen retrieval bank v2.1 (questions + optimized scoring rules) without calling a model.

Extends the v2.0 validator with rules-level checks:
  * every evidence id is exactly covered by `facts` or code requirements (no manual false hardcoding)
  * every fact clause is satisfiable by its mapped gold chunk's delivered text
    (embedding_text + figure_uncertainties appended the same way RagQueryService does)
  * clauses declared `pending_description_update` must still be missing today
  * code evidence must be contiguous in the delivered text
  * generic clauses (same-doc frequency >= 3) are reported as warnings
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

PENDING = "pending_description_update"


def load(path):
    return json.loads(path.read_text(encoding="utf8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line]


def norm(text):
    return "".join(str(text).split())


def delivered_text(chunk):
    text = chunk.get("embedding_text") or chunk.get("content") or ""
    uncertainties = chunk.get("figure_uncertainties") or []
    notes = [n for n in uncertainties if isinstance(n, str) and n.strip() and n not in text]
    if notes:
        text += "\n不确定项：" + "；".join(notes)
    return text


def check(pdf_root=None):
    manifest = load(HERE / "manifest.json")
    for name, expected in manifest.get("files", {}).items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected, name
    questions = rows(HERE / "questions.jsonl")
    queries = rows(HERE / "queries.jsonl")
    corpus = load(HERE / "corpus.json")
    rules = load(HERE / "evidence_rules.json")
    assert len(questions) == len({q["id"] for q in questions}) == len(queries) == 64
    assert queries == [dict(id=q["id"], input=q["input"]) for q in questions]
    assert all(set(q) == {"id", "input"} for q in queries)
    assert all(set(q["input"]) == {"question", "product", "version", "top_k"} for q in queries)
    assert all(q["input"]["top_k"] == 30 for q in questions)
    assert all(q["split"] == "development" for q in questions)
    positive = [q for q in questions if q["evidence_requirements"]]
    assert len(positive) == 56
    assert len({s for q in positive for s in q["allowed_sources"]}) == 16
    assert Counter(q["category"] for q in questions) == manifest["statistics"]["categories"]
    mapping = {r["evidence_id"]: r for r in rows(HERE / "current_chunk_mapping.jsonl")}
    by_source = {s["source_id"]: s for s in corpus["sources"]}
    chunks, pdfs = {}, {}
    anchors = images = code_segments = evidence_count = 0
    try:
        for source in corpus["sources"]:
            path = ROOT / source["chunk_file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source["chunk_file_sha256"]
            chunks.update({r["chunk_id"]: r for r in rows(path)})
            if pdf_root is not None:
                import fitz

                pdf = pdf_root / source["relative_pdf_path"]
                assert hashlib.sha256(pdf.read_bytes()).hexdigest() == source["file_sha256"]
                pdfs[source["source_id"]] = fitz.open(pdf)
        for q in positive:
            assert q["answer_checks"] and q["critical_errors"]
            for evidence in q["evidence_requirements"]:
                evidence_count += 1
                assert evidence["evidence_id"] in mapping
                candidates = [chunks[k] for k in evidence["current_chunk_candidates"]]
                assert all(not c["exclude_from_embedding"] for c in candidates)
                mapped = mapping[evidence["evidence_id"]]["candidates"]
                assert [r["logical_chunk_id"] for r in mapped] == (
                    evidence["current_chunk_candidates"])
                if "required_code_text" in evidence:
                    assert any(evidence["required_code_text"] in c["content"] for c in candidates)
                for loc in evidence["locators"]:
                    assert loc["source_id"] in q["allowed_sources"]
                    source = by_source[loc["source_id"]]
                    assert all(1 <= page <= source["pdf_pages"] for page in loc["pdf_pages"])
                    for proof in loc.get("pdf_text_anchor_checks", []):
                        anchors += 1
                        assert any(norm(proof["anchor"]) in norm(c["content"]) for c in candidates)
                        if pdf_root is not None:
                            assert any(
                                norm(proof["anchor"]) in norm(
                                    pdfs[loc["source_id"]][p - 1].get_text())
                                for p in proof["pdf_pages"])
                    if loc.get("image_ref"):
                        images += 1
                        image = ROOT / "storage/parsed" / source["product"] / loc["image_ref"]
                        assert hashlib.sha256(image.read_bytes()).hexdigest() == loc["image_sha256"]
                    if loc.get("source_lines"):
                        assert all(line["segments"] for line in loc["source_lines"])
                        for line in loc["source_lines"]:
                            for segment in line["segments"]:
                                code_segments += 1
                                if pdf_root is not None:
                                    assert norm(segment["text"]) in norm(
                                        pdfs[loc["source_id"]][loc["pdf_pages"][0] - 1].get_text())
        assert len(mapping) == evidence_count == 71
        unscoped = rows(HERE / "queries_unscoped.jsonl")
        expected = [dict(id=q["id"], input={**q["input"], "product": None, "version": None})
                    for q in positive if len(q["allowed_sources"]) == 1]
        assert unscoped == expected and len(unscoped) == 54
        judgments = rows(HERE / "judgment_template.jsonl")
        assert len(judgments) == 64
        assert all(value is None for r in judgments for value in r["evidence_verdicts"].values())

        # ---------- rules-level checks ----------
        code_ids = {
            e["evidence_id"]
            for q in positive
            for e in q["evidence_requirements"]
            if "required_code_text" in e
        }
        expected_ids = {e["evidence_id"] for q in positive for e in q["evidence_requirements"]}
        assert expected_ids == set(rules["facts"]) | code_ids, (
            expected_ids ^ (set(rules["facts"]) | code_ids))
        pending = rules.get(PENDING, {})
        metadata_dependent = rules.get("metadata_dependent", {})
        generic = []
        unsatisfiable = []
        pending_still_missing = []
        for q in positive:
            for evidence in q["evidence_requirements"]:
                eid = evidence["evidence_id"]
                gold = [chunks[c["logical_chunk_id"]] for c in mapping[eid]["candidates"]]
                delivered = [delivered_text(c) for c in gold]
                if "required_code_text" in evidence:
                    from_code = "\n" + "\n".join(
                        line.strip() for line in evidence["required_code_text"].splitlines()
                        if line.strip()) + "\n"
                    ok = any(from_code in "\n" + "\n".join(
                        line.strip() for line in text.splitlines() if line.strip()) + "\n"
                        for text in delivered)
                    assert ok, ("code not contiguous", eid)
                    continue
                clauses = rules["facts"][eid]
                found = []
                for clause in clauses:
                    hits = [alt for alt in clause if any(norm(alt) in norm(t) for t in delivered)]
                    found.append(bool(hits))
                if eid in pending:
                    if all(found):
                        pending_still_missing.append(eid)  # no longer pending: must be promoted
                    continue
                if not all(found):
                    unsatisfiable.append(eid)
                for clause in clauses:
                    for alt in clause:
                        n = sum(
                            1 for c in chunks.values()
                            if c.get("doc_id") == gold[0].get("doc_id")
                            and norm(alt) in norm(delivered_text(c))
                        )
                        if n >= 3:
                            generic.append(dict(evidence_id=eid, clause=alt[:60], chunks=n))
        assert not unsatisfiable, unsatisfiable
        assert not pending_still_missing, pending_still_missing
    finally:
        for pdf in pdfs.values():
            pdf.close()
    return dict(
        status="pass",
        bank_version=manifest["bank_version"],
        questions=64,
        positive_questions=56,
        evidence_groups=evidence_count,
        rule_groups=len(rules["facts"]),
        code_groups=len(code_ids),
        metadata_dependent=sorted(metadata_dependent),
        pending_description_update=sorted(pending),
        split_bundles=len(rules.get("changes_from_v2_0", [])),
        generic_warnings=len(generic),
        source_documents=16,
        image_locators=images,
        native_text_anchors=anchors,
        code_source_segments=code_segments,
        original_pdfs_checked=pdf_root is not None,
        retrieval_run=False,
        model_api_calls=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-root", type=Path)
    print(json.dumps(check(parser.parse_args().pdf_root), ensure_ascii=False))
