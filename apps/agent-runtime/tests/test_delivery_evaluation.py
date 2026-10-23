from __future__ import annotations

import json
import shutil
from pathlib import Path

from app import cli
from app.contracts import AssistantTurn, Message, ToolCall
from app.delivery import DeliveryRecorder
from app.loop import AgentRunResult
from app.providers.mock import MockProvider
from evaluation.runner import main, score


async def delivered_task(tmp_path, monkeypatch, capsys, *, edit=True, command_first=False):
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "config.json").write_text('{"duty":30}', encoding="utf-8")
    (seed / "business.py").write_text("unchanged = True\n", encoding="utf-8")
    root = tmp_path / "work"
    shutil.copytree(seed, root)
    calls = [ToolCall("verify", "run_command", {
        "command": ["python", "-c", "import json; print(json.load(open('config.json')))"]},
    )]
    if edit:
        change = ToolCall("write", "write_file", {"path": "config.json", "content": '{"duty":40}'})
        calls.insert(len(calls) if command_first else 0, change)
    provider = MockProvider([
        AssistantTurn(Message("assistant", tool_calls=calls), "tool_calls"),
        AssistantTurn(Message("assistant", "done"), "stop"),
    ])
    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **_: provider)
    delivery_dir = root / ".agent-runtime" / "deliveries" / "task"
    args = cli.build_parser().parse_args([
        "change duty to 40; preserve business", "--base-url", "http://unused.invalid",
        "--api-key", "DO_NOT_RECORD_SECRET", "--workspace", str(root),
        "--delivery-dir", str(delivery_dir), "--json", "--no-memory", "--no-compaction",
    ])
    assert await cli.run_once(args) == 0
    response = json.loads(capsys.readouterr().out)
    path = Path(response["delivery"])
    report = json.loads(path.read_text(encoding="utf-8"))
    case = tmp_path / "case.json"
    case.write_text(json.dumps({
        "id": "change-duty", "seed": str(seed), "checks": [
            {"id": "duty", "kind": "json_value", "path": "config.json",
             "pointer": "/duty", "expected": 40},
            {"id": "keep-business", "kind": "unchanged", "path": "business.py"},
            {"id": "independent-command", "kind": "command", "command": [
                "{python}", "-c", "import json; assert json.load(open('config.json'))['duty']==40",
            ]},
        ],
    }), encoding="utf-8")
    return root, path, report, case


async def test_cli_delivery_records_actual_changes_without_credentials(
    tmp_path, monkeypatch, capsys,
):
    root, path, report, case = await delivered_task(tmp_path, monkeypatch, capsys)
    assert report["changed_files"] == ["config.json"]
    assert report["commands"][0]["matches_delivery"]
    assert "'duty': 40" in report["commands"][0]["content"]
    assert report["task_acceptance"] == "unverified"
    assert "DO_NOT_RECORD_SECRET" not in path.read_text(encoding="utf-8")
    assert b'+{"duty":40}' in (path.parent / "changes.diff").read_bytes()
    judgment = score(case, path, root, tmp_path / "checks", human_interventions=0)
    assert judgment["passed"] and judgment["autonomous_pass"]
    assert all(row["passed"] for row in judgment["checks"])


async def test_later_source_changes_cannot_reuse_a_passed_delivery(tmp_path, monkeypatch, capsys):
    root, path, _, case = await delivered_task(tmp_path, monkeypatch, capsys)
    (root / "business.py").write_text("raise RuntimeError('regression')", encoding="utf-8")
    judgment = score(case, path, root, tmp_path / "checks")
    assert not judgment["passed"]
    binding = next(row for row in judgment["checks"] if row["id"] == "delivered_revision")
    assert not binding["passed"]


async def test_completed_answer_does_not_replace_task_checks(tmp_path, monkeypatch, capsys):
    root, path, _, case = await delivered_task(tmp_path, monkeypatch, capsys, edit=False)
    judgment = score(case, path, root, tmp_path / "checks")
    assert judgment["runtime_status"] == "completed"
    assert not judgment["passed"]
    assert not next(row for row in judgment["checks"] if row["id"] == "duty")["passed"]
    assert judgment["autonomous_pass"] is None


async def test_command_before_the_final_edit_is_marked_stale(tmp_path, monkeypatch, capsys):
    _, _, report, _ = await delivered_task(tmp_path, monkeypatch, capsys, command_first=True)
    assert not report["commands"][0]["matches_delivery"]
    assert "'duty': 30" in report["commands"][0]["content"]


async def test_development_check_returns_all_failures_without_a_delivery(
    tmp_path, monkeypatch, capsys,
):
    root, _, _, case = await delivered_task(tmp_path, monkeypatch, capsys)
    task = json.loads(case.read_text(encoding="utf-8"))
    task["checks"].append({"id": "manifest", "kind": "manifest"})
    case.write_text(json.dumps(task), encoding="utf-8")
    out = tmp_path / "live-check"
    argv = ["check", "--case", str(case), "--workspace", str(root), "--out", str(out)]
    assert main(argv) == 1
    summary = json.loads(capsys.readouterr().out)
    checks = {row["id"]: row for row in summary["results"][0]["checks"]}
    assert checks["independent-command"]["passed"]
    assert not checks["manifest"]["passed"]
    assert summary["model_calls"] == 0
    assert "runtime outcome not assessed" in summary["results"][0]["scope"]
    # All required checks, not one selected passing command, determine success.
    task["checks"].pop()
    case.write_text(json.dumps(task), encoding="utf-8")
    assert main(argv) == 0
    capsys.readouterr()


async def test_development_check_rejects_a_verifier_that_changes_source(
    tmp_path, monkeypatch, capsys,
):
    root, _, _, case = await delivered_task(tmp_path, monkeypatch, capsys)
    task = json.loads(case.read_text(encoding="utf-8"))
    task["checks"].append({"id": "mutating-check", "kind": "command", "command": [
        "{python}", "-c", "from pathlib import Path; Path('business.py').write_text('changed')",
    ]})
    case.write_text(json.dumps(task), encoding="utf-8")
    assert main(["check", "--case", str(case), "--workspace", str(root),
                 "--out", str(tmp_path / "check")]) == 1
    summary = json.loads(capsys.readouterr().out)
    checks = {row["id"]: row for row in summary["results"][0]["checks"]}
    assert checks["mutating-check"]["passed"]
    assert not checks["verifier_preserved_source"]["passed"]


async def test_batch_keeps_missing_runs_in_the_denominator(tmp_path, monkeypatch, capsys):
    root, path, _, case = await delivered_task(tmp_path, monkeypatch, capsys)
    runs = tmp_path / "runs.jsonl"
    runs.write_text("\n".join(json.dumps(row) for row in [
        {"case": str(case), "delivery": str(path), "workspace": str(root)},
        {"case": str(case), "delivery": "missing.json", "workspace": str(root)},
    ]), encoding="utf-8")
    out = tmp_path / "score"
    assert main(["batch", "--runs", str(runs), "--out", str(out)]) == 1
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert (summary["total"], summary["passed"]) == (2, 1)
    assert summary["model_calls"] == 0


async def test_relative_paths_are_resolved_before_changing_verifier_working_directory(
    tmp_path, monkeypatch, capsys,
):
    root, path, _, case = await delivered_task(tmp_path, monkeypatch, capsys)
    task = json.loads(case.read_text(encoding="utf-8"))
    task["checks"].append({"id": "workspace_argument", "kind": "command", "command": [
        "{python}", "-c",
        "from pathlib import Path; import sys; assert (Path(sys.argv[1])/'config.json').is_file()",
        "{workspace}",
    ]})
    case.write_text(json.dumps(task), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert score(case.relative_to(tmp_path), path.relative_to(tmp_path),
                 root.relative_to(tmp_path), Path("checks"))["passed"]


def test_existing_usart_host_oracle_and_counterexample(tmp_path, monkeypatch):
    """Exercises the migrated independent C harness, not an Agent success claim."""
    import os

    import pytest

    gcc = os.environ.get("CC") or shutil.which("gcc")
    if not gcc:
        pytest.skip("set CC to a host GCC executable")
    monkeypatch.setenv("CC", gcc)
    base = Path(__file__).resolve().parents[1] / "evaluation"
    root = tmp_path / "project"
    shutil.copytree(base / "fixtures/pwm_before_usart", root)
    record = DeliveryRecorder(root, tmp_path / "delivery", state_dir=tmp_path / "state")
    shutil.copytree(base / "fixtures/pwm_usart", root, dirs_exist_ok=True)

    def finish():
        result = AgentRunResult("completed", [], 1, 1)
        response = {"status": "completed", "answer": "fixture oracle", "steps": 1, "tool_calls": []}
        return record.finish(result, question="test fixture", settings={"test": "offline_oracle"},
                             response=response, exit_code=0)

    report = finish()
    judgment = score(base / "cases/usart_addition.json", report, root, tmp_path / "good")
    assert judgment["passed"], judgment
    # A plausible manifest cannot cover a real TX register regression.
    uart = root / "agent_demo/modules/demo_usart0.c"
    text = uart.read_text(encoding="utf-8")
    assert "MPTX0 = 0x00;" in text
    uart.write_text(text.replace("MPTX0 = 0x00;", "MPTX0 = 0x03;"), encoding="utf-8")
    report = finish()
    broken = score(base / "cases/usart_addition.json", report, root, tmp_path / "bad")
    assert not broken["passed"]
    failure = next(row for row in broken["checks"] if row["id"] == "host_behavior")
    assert not failure["passed"]
    assert "USART0.MPTX0: expected=0 actual=3" in failure["stdout"]
