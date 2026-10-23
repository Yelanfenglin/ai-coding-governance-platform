from __future__ import annotations

import pytest

from app import cli
from app.contracts import AssistantTurn, Message, ModelRequest, ToolCall
from app.providers.mock import stream_from_coroutine
from app.session import JsonlSessionStore
from app.skills import load_skills, read_skill_tool, skills_prompt
from app.tools import ToolRegistry


async def test_skill_metadata_is_discovered_and_body_is_loaded_on_demand(tmp_path):
    folder = tmp_path / "another-task"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        '---\nname: another-task\ndescription: "A task: with punctuation"\n---\n'
        '# Another task\nBODY_NOT_IN_DISCOVERY\n', encoding="utf-8",
    )
    skills = load_skills(tmp_path)
    prompt = skills_prompt(skills)
    assert "another-task" in prompt and "A task: with punctuation" in prompt
    assert "BODY_NOT_IN_DISCOVERY" not in prompt
    registry = ToolRegistry().register(read_skill_tool(skills))
    loaded = await registry.execute(ToolCall("load", "read_skill", {"name": "another-task"}))
    assert not loaded.is_error and "BODY_NOT_IN_DISCOVERY" in loaded.content
    missing = await registry.execute(ToolCall("missing", "read_skill", {"name": "../secret"}))
    assert missing.is_error


@pytest.mark.parametrize("resume", [False, True])
async def test_workspace_cli_exposes_skills_in_new_and_existing_sessions(
    tmp_path, monkeypatch, resume,
):
    session_path = tmp_path / "session.jsonl"
    if resume:
        from app.contracts import MessageEntry
        await JsonlSessionStore(session_path).append(MessageEntry(
            turn=0, step=0, message=Message("system", "old head"), timestamp="2026-10-23T00:00:00Z",
        ))
    requests: list[ModelRequest] = []

    class Reader:
        def generate_stream(self, request):
            return stream_from_coroutine(lambda: self.generate(request))

        async def generate(self, request):
            requests.append(request)
            if len(requests) == 1:
                return AssistantTurn(Message("assistant", tool_calls=[
                    ToolCall("skill", "read_skill", {"name": "chip-migration"}),
                ]), "tool_calls")
            assert "保留行为的芯片改型迁移" in request.messages[-1].content
            return AssistantTurn(Message("assistant", "loaded"), "stop")

    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **kwargs: Reader())
    args = cli.build_parser().parse_args([
        "迁移已有模块", "--base-url", "http://unused.invalid", "--workspace", str(tmp_path),
        "--session", str(session_path), "--no-memory", "--json",
    ])
    assert await cli.run_once(args, console=False) == 0
    first = requests[0]
    assert "chip-migration" in "\n".join(m.content for m in first.messages)
    assert {"read_file", "write_file", "edit_file", "run_command", "read_skill"} <= {
        tool.name for tool in first.tools
    }
    if resume:
        assert first.messages[0].content.startswith("old head")
        assert sum(m.role == "system" for m in first.messages) == 1
