"""Configured checks run real subprocesses and assess the last delivered revision."""
from __future__ import annotations

import json

import pytest

from app import cli
from app.contracts import AssistantTurn, Message, ToolCall
from app.providers.mock import MockProvider, stream_from_coroutine


def setup(tmp_path, *, script=None):
    project = tmp_path / "工程 with space"
    project.mkdir()
    (project / "config.json").write_text('{"duty":30}', encoding="utf-8")
    checker = tmp_path / "check.py"
    checker.write_text(script or (
        "import json,sys\n"
        "actual=json.load(open('config.json'))['duty']\n"
        "print(f'expected=40 actual={actual}')\n"
        "sys.exit(0 if actual==40 else 1)\n"
    ), encoding="utf-8")
    config = tmp_path / "verification.json"
    config.write_text(json.dumps({
        "command": ["{python}", "{config_dir}/check.py"],
        "cwd": "{workspace}", "timeout": 5,
    }), encoding="utf-8")
    args = cli.build_parser().parse_args([
        "repair", "--base-url", "http://unused.invalid", "--workspace", str(project),
        "--verification", str(config), "--json", "--no-memory", "--no-compaction",
    ])
    return project, config, args


def turn(*calls):
    return AssistantTurn(Message("assistant", tool_calls=list(calls)), "tool_calls")


def edit(value, cid="edit"):
    return ToolCall(cid, "write_file", {"path": "config.json", "content": f'{{"duty":{value}}}'})


DONE = AssistantTurn(Message("assistant", "done"), "stop")


async def test_failure_feedback_repair_final_check_and_delivery(tmp_path, monkeypatch, capsys):
    project, _, args = setup(tmp_path)

    class Provider:
        def generate_stream(self, request):
            return stream_from_coroutine(lambda: self.generate(request))

        async def generate(self, request):
            observations = [m for m in request.messages if m.role == "tool"]
            if not observations:
                return turn(ToolCall("check", "verify_project", {}))
            if len(observations) == 1:
                failure = json.loads(observations[-1].content)
                assert not failure["passed"] and "expected=40 actual=30" in failure["output"]
                return turn(edit(40))
            return DONE

    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **_: Provider())
    # Launching cwd is deliberately unrelated to the project and checker.
    monkeypatch.chdir(tmp_path)
    assert await cli.run_once(args) == 0
    response = json.loads(capsys.readouterr().out)
    verification = response["verification"]
    assert verification["status"] == "passed" and verification["matches_delivery"]
    assert [row["passed"] for row in verification["runs"]] == [False, True]
    from pathlib import Path
    delivery = json.loads(Path(response["delivery"]).read_text(encoding="utf-8"))
    assert delivery["task_acceptance"] == "passed"
    assert verification["runs"][-1]["output_revision"] == delivery["revision"]
    assert json.loads((project / "config.json").read_text())["duty"] == 40


@pytest.mark.parametrize("turns", [
    [DONE],
    [turn(edit(40), ToolCall("check", "verify_project", {})), turn(edit(30, "break")), DONE],
])
async def test_completion_or_stale_pass_does_not_certify_bad_source(
    tmp_path, monkeypatch, capsys, turns,
):
    _, _, args = setup(tmp_path)
    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **_: MockProvider(turns))
    assert await cli.run_once(args) == cli.EXIT_VERIFICATION
    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "completed"
    assert response["verification"]["status"] == "failed"
    assert "actual=30" in response["verification"]["runs"][-1]["output"]


@pytest.mark.parametrize("failure", ["missing", "timeout", "mutates"])
async def test_verifier_errors_and_mutating_checks_cannot_pass(
    tmp_path, monkeypatch, capsys, failure,
):
    _, config, args = setup(tmp_path)
    value = json.loads(config.read_text())
    if failure == "missing":
        value["command"] = [str(tmp_path / "nonexistent-program")]
    elif failure == "timeout":
        value.update(command=["{python}", "-c", "import time; time.sleep(10)"], timeout=0.1)
    else:
        value["command"] = ["{python}", "-c",
                            "from pathlib import Path; Path('config.json').write_text('{}')"]
    config.write_text(json.dumps(value))
    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **_: MockProvider([DONE]))
    assert await cli.run_once(args) == cli.EXIT_VERIFICATION
    row = json.loads(capsys.readouterr().out)["verification"]["runs"][-1]
    assert not row["passed"]
    assert ("source_error" if failure == "mutates" else "error") in row


async def test_budget_exhaustion_is_unverified_and_does_not_launch_final_check(
    tmp_path, monkeypatch, capsys,
):
    _, _, args = setup(tmp_path, script="raise AssertionError('must not be executed')")
    args.max_steps = 0
    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **_: MockProvider([]))
    assert await cli.run_once(args) == cli.EXIT_BUDGET
    response = json.loads(capsys.readouterr().out)
    assert response["verification"]["status"] == "unverified"
    assert response["verification"]["runs"] == []


async def test_resumed_turn_uses_current_host_configuration(tmp_path, monkeypatch, capsys):
    project, config, args = setup(tmp_path)
    args.session = str(tmp_path / "session.jsonl")
    (project / "config.json").write_text('{"duty":40}')
    for expected_code in (0, cli.EXIT_VERIFICATION):
        provider = MockProvider([turn(ToolCall("check", "verify_project", {})), DONE])
        monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda p=provider, **_: p)
        assert await cli.run_once(args) == expected_code
        response = json.loads(capsys.readouterr().out)
        if expected_code == 0:
            value = json.loads(config.read_text())
            value["command"] = ["{python}", "-c", "raise SystemExit(7)"]
            config.write_text(json.dumps(value))
        else:
            assert response["resumed_messages"] > 0
            assert response["verification"]["runs"][0]["exit_code"] == 7
