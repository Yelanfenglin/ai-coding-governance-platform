from __future__ import annotations

import json
from pathlib import Path

from evaluation import runner


def test_run_dispatches_to_script_and_returns_its_exit_code(tmp_path: Path) -> None:
    script = tmp_path / "exit_script.py"
    script.write_text("raise SystemExit(3)\n", encoding="utf-8")
    assert runner._run(script, []) == 3


def test_run_missing_script_returns_tooling_error(tmp_path: Path) -> None:
    assert runner._run(tmp_path / "missing.py", []) == 2


def test_string_system_exit_is_failure_and_preserved(tmp_path: Path, capsys) -> None:
    script = tmp_path / "string_exit.py"
    script.write_text("raise SystemExit('validation failed')\n", encoding="utf-8")
    assert runner._run(script, []) == 1
    captured = capsys.readouterr()
    assert "validation failed" in captured.err


def test_system_exit_none_is_success(tmp_path: Path) -> None:
    script = tmp_path / "none_exit.py"
    script.write_text("raise SystemExit(None)\n", encoding="utf-8")
    assert runner._run(script, []) == 0


def test_main_dispatches_manual_commands(monkeypatch, tmp_path: Path) -> None:
    calls: list[object] = []

    monkeypatch.setattr(runner, "regression", lambda: calls.append("regression") or 0)
    monkeypatch.setattr(
        runner, "validate", lambda pdf_root: calls.append(("validate", pdf_root)) or 0
    )
    monkeypatch.setattr(
        runner,
        "score",
        lambda run_dir, split, out_dir=None: calls.append(("score", run_dir, split, out_dir)) or 0,
    )
    monkeypatch.setattr(
        runner, "status", lambda trace_path=None: calls.append(("status", trace_path)) or 0
    )

    assert runner.main(["regression"]) == 0
    assert runner.main(["validate"]) == 0
    assert runner.main(["score", "--run-dir", str(tmp_path), "--split", "development"]) == 0
    assert runner.main(["status"]) == 0
    assert calls[0] == "regression"
    assert calls[1] == ("validate", None)
    assert calls[2][:3] == ("score", tmp_path, "development")
    assert calls[3] == ("status", None)


def test_status_fails_on_missing_artifacts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        runner,
        "ARTIFACTS",
        {"retrieval": (tmp_path / "missing.json",)},
    )
    assert runner.status() == 1


def test_status_passes_when_artifacts_and_trace_exist(monkeypatch, tmp_path: Path) -> None:
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}", encoding="utf-8")
    trace = tmp_path / "trace.jsonl"
    trace.write_text('{"trace_id": "t"}\n', encoding="utf-8")
    monkeypatch.setattr(runner, "ARTIFACTS", {"retrieval": (artifact,)})
    assert runner.status(trace) == 0


def test_trace_path_writes_a_traceable_manifest(monkeypatch, tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text('{"trace_id": "t"}\n', encoding="utf-8")
    monkeypatch.setattr(runner, "EVAL_RUNS", tmp_path / "evaluation_runs")
    monkeypatch.setattr(runner, "status", lambda trace_path=None: 0)
    assert runner.main(["--trace-path", str(trace), "status"]) == 0
    manifests = list((tmp_path / "evaluation_runs").glob("*/manifest.json"))
    assert len(manifests) == 1
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert payload["trace_path"] == str(trace)
    assert payload["trace_exists"] is True


def test_main_all_stops_when_validation_fails(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(runner, "validate", lambda pdf_root: calls.append("validate") or 1)
    monkeypatch.setattr(runner, "regression", lambda: calls.append("regression") or 0)
    assert runner.main(["all"]) == 1
    assert calls == ["validate"]
