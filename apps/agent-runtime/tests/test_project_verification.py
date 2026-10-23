"""Host-supplied checks use the existing command -> observation -> repair loop."""

from __future__ import annotations

import json

from app import cli
from app.contracts import AssistantTurn, Message, ToolCall
from app.providers.mock import MockProvider, stream_from_coroutine
from app.session import JsonlSessionStore


def args(*extra, question="repair the project"):
    return cli.build_parser().parse_args([
        question, "--base-url", "http://unused.invalid", "--json",
        "--no-memory", "--no-compaction", *extra,
    ])


async def test_independent_failure_reaches_model_then_repair_and_recheck(
    tmp_path, monkeypatch, capsys,
):
    project = tmp_path / "工程 with space"
    project.mkdir()
    (project / "config.json").write_text('{"duty":30}', encoding="utf-8")
    checker = tmp_path / "independent_check.py"
    checker.write_text(
        "import json,sys\n"
        "actual=json.load(open('config.json'))['duty']\n"
        "print(f'expected duty=40; actual={actual}')\n"
        "sys.exit(0 if actual==40 else 1)\n", encoding="utf-8",
    )
    original_check = checker.read_bytes()
    command = ["python", str(checker)]
    wire_command = json.dumps(command, ensure_ascii=False)
    question = (
        "Repair the project. Run this independent check and recheck after changes: "
        f"{wire_command}"
    )

    class RepairingProvider:
        def __init__(self):
            self.requests = []

        def generate_stream(self, request):
            return stream_from_coroutine(lambda: self.generate(request))

        async def generate(self, request):
            self.requests.append(request)
            step = len(self.requests)
            if step == 1:
                assert request.messages[-1].content == question
                call = ToolCall("check1", "run_command", {"command": command})
            elif step == 2:
                observation = json.loads(request.messages[-1].content)
                assert observation["exit_code"] == 1
                assert "expected duty=40; actual=30" in observation["output"]
                call = ToolCall("fix", "edit_file", {
                    "path": "config.json", "old_text": "30", "new_text": "40",
                })
            elif step == 3:
                call = ToolCall("check2", "run_command", {"command": command})
            else:
                observation = json.loads(request.messages[-1].content)
                assert observation["exit_code"] == 0
                assert "actual=40" in observation["output"]
                return AssistantTurn(Message("assistant", "Repaired; independent check passed."),
                                     "stop")
            return AssistantTurn(Message("assistant", tool_calls=[call]), "tool_calls")

    provider = RepairingProvider()
    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **_: provider)
    code = await cli.run_once(args(
        "--workspace", str(project),
        "--session", str(tmp_path / "session.jsonl"),
        "--delivery-dir", str(tmp_path / "delivery"),
        question=question,
    ))
    output = json.loads(capsys.readouterr().out)
    assert code == 0 and output["status"] == "completed"
    assert output["tool_calls"] == ["run_command", "edit_file", "run_command"]
    assert json.loads((project / "config.json").read_text())["duty"] == 40
    assert checker.read_bytes() == original_check
    delivery = json.loads((tmp_path / "delivery/delivery.json").read_text(encoding="utf-8"))
    assert delivery["question"] == question
    assert delivery["commands"][0]["is_error"]
    assert not delivery["commands"][0]["matches_delivery"]
    assert not delivery["commands"][1]["is_error"]
    assert delivery["commands"][1]["matches_delivery"]


async def test_project_check_in_task_is_current_turn_input_when_resuming(
    tmp_path, monkeypatch, capsys,
):
    project = tmp_path / "project"
    project.mkdir()
    session = tmp_path / "session.jsonl"
    system_head = None
    for marker in ("check_old.py", "check_new.py"):
        command = json.dumps(["python", marker])
        question = f"Check the project with this command: {command}"
        provider = MockProvider([AssistantTurn(Message("assistant", "not verified"), "stop")])
        monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda p=provider, **_: p)
        await cli.run_once(args("--workspace", str(project), "--session", str(session),
                                question=question))
        capsys.readouterr()
        messages = provider.requests[0].messages
        assert messages[-1].content == question
        if system_head is None:
            system_head = messages[0].content
        assert messages[0].content == system_head
    replayed = await JsonlSessionStore(session).load_messages()
    assert replayed[-2].content == question
    assert "check_old.py" not in replayed[-2].content
