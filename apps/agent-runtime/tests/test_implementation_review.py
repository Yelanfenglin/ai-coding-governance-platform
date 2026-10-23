from __future__ import annotations

import copy
import json

import pytest

from app.contracts import ToolCall, ToolFinished, ToolResult, ToolStarted
from app.delivery import DeliveryRecorder, file_hashes, revision
from app.rag_tools import render_search_hit


def observe(recorder, call_id, name, content, *, error=False):
    call = ToolCall(call_id, name, {})
    recorder.observe(ToolStarted(1, 1, call))
    recorder.observe(ToolFinished(1, 1, call, ToolResult(content, is_error=error)))


@pytest.fixture
def review(tmp_path):
    (tmp_path / "module.c").write_text("int value = 2;\n", encoding="utf-8")
    recorder = DeliveryRecorder(tmp_path, tmp_path / ".agent-runtime" / "delivery",
                                state_dir=tmp_path / ".agent-runtime")
    observe(recorder, "source", "search_spec", render_search_hit({
        "document_id": "doc", "chunk_id": "chunk", "product": "M9F6820",
        "version": "V1.7", "page_start": 113, "page_end": 114,
        "section_path": "12.1 ADC", "text": "A retrieved statement.",
    }))
    observe(recorder, "check", "run_command", json.dumps({"exit_code": 0, "output": "ok"}))
    decision = {
        "id": "module-value", "title": "Module value", "requirement": "Use value 2",
        "rationale": "Model explanation, not a host correctness verdict.",
        "code_refs": [{"path": "module.c", "start_line": 1, "end_line": 1}],
        "evidence_refs": [{"document_id": "doc", "chunk_id": "chunk"}],
        "check_call_ids": ["check"], "limitations": ["Host checks only"],
    }
    return recorder, decision


def finish(recorder):
    files = recorder.snapshot()
    return recorder.implementation.finish(files, revision(file_hashes(files)))


def test_actual_sources_code_and_checks_are_bound_without_promoting_model_claims(review):
    recorder, decision = review
    assert not recorder.implementation.record({"decisions": [decision]}, None).is_error
    row = finish(recorder)[0]
    assert row["rationale"] == decision["rationale"]
    assert row["code_refs"][0]["excerpt"] == "int value = 2;"
    assert row["code_refs"][0]["status"] == "located"
    assert row["evidence_refs"][0] == {
        "document_id": "doc", "chunk_id": "chunk", "product": "M9F6820", "version": "V1.7",
        "pages": "113-114", "section": "12.1 ADC", "call_id": "source",
    }
    assert row["checks"][0]["status"] == "passed"
    assert "passed" not in row, "reference checks cannot certify the model's whole decision"


@pytest.mark.parametrize("bad", ["path", "lines", "source", "check"])
def test_unknown_or_invalid_references_return_correctable_tool_errors(review, bad):
    recorder, decision = review
    if bad == "path":
        decision["code_refs"][0]["path"] = "../private.c"
    elif bad == "lines":
        decision["code_refs"][0]["end_line"] = 20
    elif bad == "source":
        decision["evidence_refs"][0]["chunk_id"] = "never-returned"
    else:
        decision["check_call_ids"] = ["not-a-real-call"]
    result = recorder.implementation.record({"decisions": [decision]}, None)
    assert result.is_error and json.loads(result.content)["error"]
    assert finish(recorder) == []


@pytest.mark.parametrize("remove", [False, True])
def test_edits_after_recording_invalidate_location_and_old_checks(review, remove):
    recorder, decision = review
    recorder.implementation.record({"decisions": [decision]}, None)
    path = recorder.workspace / "module.c"
    if remove:
        path.unlink()
    else:
        path.write_text("int value = 99;\n", encoding="utf-8")
    row = finish(recorder)[0]
    assert row["code_refs"][0]["status"] == ("missing" if remove else "changed")
    assert row["code_refs"][0]["excerpt"] == "int value = 2;"
    assert row["checks"][0]["status"] == "outdated"


def test_missing_links_stay_unverified_even_when_other_checks_pass(review):
    recorder, decision = review
    decision.update(evidence_refs=[], check_call_ids=[])
    recorder.implementation.record({"decisions": [decision]}, None)
    row = finish(recorder)[0]
    assert row["checks"] == [] and row["evidence_refs"] == []


def test_failed_checks_and_host_checks_keep_their_actual_status(review):
    recorder, decision = review
    observe(recorder, "failed", "run_command", '{"exit_code":1,"output":"failed"}', error=True)
    observe(recorder, "host", "verify_project", '{"exit_code":0,"output":"host ok"}')
    decision["check_call_ids"] = ["failed", "host"]
    recorder.implementation.record({"decisions": [decision]}, None)
    assert [row["status"] for row in finish(recorder)[0]["checks"]] == ["failed", "passed"]


def test_updates_replace_by_id_and_invalid_batch_does_not_overwrite_record(review):
    recorder, decision = review
    recorder.implementation.record({"decisions": [decision]}, None)
    updated = {**decision, "rationale": "Corrected explanation"}
    invalid = copy.deepcopy(decision)
    invalid["id"] = "invalid"
    invalid["check_call_ids"] = ["unknown"]
    assert recorder.implementation.record({"decisions": [updated, invalid]}, None).is_error
    assert finish(recorder)[0]["rationale"] == decision["rationale"]
    recorder.implementation.record({"decisions": [updated]}, None)
    assert len(finish(recorder)) == 1
    assert finish(recorder)[0]["rationale"] == "Corrected explanation"
