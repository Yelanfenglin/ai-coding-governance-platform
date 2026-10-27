"""CLI 的会话续接（S07）。

上游交互客户端从 Agent/session 取回上下文再继续对话；本地 `--session` 之前只追加
日志、仍以「新 system + 当前问题」启动，所以它不是「继续上次会话」。
这里用离线的替身 provider 验证：第二轮请求确实带上了第一轮的历史，且只有一条 system。
"""

from __future__ import annotations

import json
from pathlib import Path

from app import cli
from app.contracts import AssistantTurn, Message, MessageEntry, ModelRequest, ToolCall
from app.loop import system_message, user_message
from app.memory import JsonMemoryStore, MemoryMetadata
from app.providers.mock import stream_from_coroutine
from app.session import JsonlSessionStore


class RecordingProvider:
    """记录每次请求的 messages，回固定答案。"""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.requests: list[ModelRequest] = []

    def generate_stream(self, request: ModelRequest):
        return stream_from_coroutine(lambda: self.generate(request))

    async def generate(self, request: ModelRequest) -> AssistantTurn:
        self.requests.append(request)
        return AssistantTurn(Message(role="assistant", content=self.answer), "stop")


def _args(*extra: str) -> object:
    return cli.build_parser().parse_args(
        [
            "question",
            "--base-url",
            "http://unused.invalid/v1",
            "--api-key",
            "offline-test",
            "--rag-key",
            "offline-test",
            "--json",
            *extra,
        ]
    )


async def _run(provider: RecordingProvider, *extra: str) -> int:
    original = cli.OpenAiCompatibleProvider
    cli.OpenAiCompatibleProvider = lambda **kwargs: provider  # type: ignore[assignment]
    try:
        return await cli.run_once(_args(*extra))
    finally:
        cli.OpenAiCompatibleProvider = original  # type: ignore[assignment]


async def test_default_memory_and_history_belong_to_the_selected_workspace(
    tmp_path, monkeypatch, capsys,
) -> None:
    monkeypatch.delenv("AGENT_STATE_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    project_a, project_b = tmp_path / "a", tmp_path / "b"
    project_a.mkdir()
    project_b.mkdir()

    class RememberingProvider(RecordingProvider):
        async def generate(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return AssistantTurn(Message("assistant", tool_calls=[ToolCall(
                    "remember", "remember_fact", {
                        "key": "clock", "value": "PROJECT_A_CLOCK_16MHZ",
                        "source": "user confirmed",
                    },
                )]), "tool_calls")
            return AssistantTurn(Message("assistant", "PROJECT_A_HISTORY_MARKER"), "stop")

    await _run(
        RememberingProvider(""), "--workspace", str(project_a),
        "--product", "M9P6362", "--version", "V1.16",
        "--session", str(tmp_path / "a.jsonl"),
    )
    capsys.readouterr()
    other = RecordingProvider("project B")
    await _run(other, "--workspace", str(project_b), "--product", "M9P6362",
               "--version", "V1.16")
    capsys.readouterr()
    assert "PROJECT_A_CLOCK_16MHZ" not in str(other.requests[0].messages)

    resumed = RecordingProvider("project A again")
    await _run(resumed, "--workspace", str(project_a), "--product", "M9P6362",
               "--version", "V1.16")
    capsys.readouterr()
    assert "PROJECT_A_CLOCK_16MHZ" in str(resumed.requests[0].messages)
    from app.session_search import SessionSearch

    a_history = SessionSearch(project_a / ".agent-runtime" / "history.sqlite")
    b_history = SessionSearch(project_b / ".agent-runtime" / "history.sqlite")
    assert await a_history.search("PROJECT_A_HISTORY_MARKER")
    assert not await b_history.search("PROJECT_A_HISTORY_MARKER")
    assert not (tmp_path / ".agent-runtime").exists()


async def test_explicit_memory_directory_still_overrides_the_workspace(
    tmp_path, monkeypatch, capsys,
) -> None:
    monkeypatch.delenv("AGENT_STATE_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    facts = JsonMemoryStore(tmp_path / "explicit" / "facts.jsonl")
    await facts.remember("shared", "EXPLICIT_STORE_FACT")
    provider = RecordingProvider("ok")
    await _run(provider, "--workspace", str(tmp_path / "project"), "--state-dir", "explicit")
    capsys.readouterr()
    assert "EXPLICIT_STORE_FACT" in str(provider.requests[0].messages)
    assert not (tmp_path / "project" / ".agent-runtime").exists()


async def test_cli_resumes_the_session_history(tmp_path: Path, capsys) -> None:
    path = tmp_path / "session.jsonl"

    first = RecordingProvider("first answer")
    await _run(first, "--session", str(path))
    capsys.readouterr()
    assert [m.role for m in first.requests[0].messages] == ["system", "user"]

    second = RecordingProvider("second answer")
    await _run(second, "--session", str(path))
    printed = json.loads(capsys.readouterr().out)

    resumed = second.requests[0].messages
    roles = [m.role for m in resumed]
    assert roles == ["system", "user", "assistant", "user"], roles
    from app.goal_prompts import goal_guidance

    assert resumed[0] == system_message(cli.react_system_prompt() + "\n\n" + goal_guidance())
    assert resumed[1].content == "question", "第一轮的提问要一起带上"
    assert resumed[2].content == "first answer", "第一轮的回答要一起带上"
    assert printed["resumed_messages"] == 3, "system head 现在也持久化，可原样重放"
    assert printed["answer"] == "second answer", "只取本轮答案，不拿上一轮充数"


async def test_optional_graph_guidance_reaches_new_and_existing_sessions(tmp_path, capsys):
    from app.spec_evidence import SPEC_EVIDENCE_GUIDANCE

    graph = Path(__file__).resolve().parents[1] / "examples/spec-relations.example.json"
    session = tmp_path / "existing.jsonl"
    first = RecordingProvider("first")
    await _run(first, "--session", str(session), "--no-memory")
    capsys.readouterr()
    original_head = first.requests[0].messages[0].content
    second = RecordingProvider("graph enabled")
    await _run(second, "--session", str(session), "--no-memory", "--spec-relations", str(graph))
    capsys.readouterr()
    request = second.requests[0]
    assert request.messages[0].content == original_head
    assert SPEC_EVIDENCE_GUIDANCE in request.messages[-1].content
    assert "inspect_spec_evidence" in {tool.name for tool in request.tools}
    third = RecordingProvider("continued")
    await _run(third, "--session", str(session), "--no-memory", "--spec-relations", str(graph))
    capsys.readouterr()
    assert sum(SPEC_EVIDENCE_GUIDANCE in m.content for m in third.requests[0].messages) == 1
    fresh = RecordingProvider("fresh")
    await _run(fresh, "--session", str(tmp_path / "new.jsonl"), "--no-memory",
               "--spec-relations", str(graph))
    capsys.readouterr()
    assert SPEC_EVIDENCE_GUIDANCE in fresh.requests[0].messages[0].content


async def test_graph_source_inspection_uses_the_cli_selected_workspace(tmp_path, capsys):
    graph = Path(__file__).resolve().parents[1] / "examples/spec-relations.example.json"
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "src/init.c").write_text(
        "void init(void) {\n CTRL=0;\n DIR=4;\n}\n", encoding="utf-8",
    )

    class SourceProvider(RecordingProvider):
        async def generate(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return AssistantTurn(Message("assistant", tool_calls=[ToolCall(
                    "inspect", "inspect_spec_evidence", {
                        "product": "EXAMPLE_CHIP", "version": "V1.0", "entity": "pin_interrupt",
                        "facts": {"input_mode": "digital"},
                        "source": {"path": "src/init.c", "function": "init"},
                    },
                )]), "tool_calls")
            return AssistantTurn(Message("assistant", "reviewed"), "stop")

    provider = SourceProvider("")
    assert await _run(provider, "--workspace", str(project), "--no-memory",
                      "--spec-relations", str(graph)) == 0
    capsys.readouterr()
    message = next(m for m in provider.requests[-1].messages if m.role == "tool")
    report = json.loads(message.content)
    assert report["source_analysis"]["path"] == "src/init.c"
    assert report["source_analysis"]["facts"]["CTRL.EN"] == 0
    assert report["configuration_status"] == "unsatisfied"


async def test_cli_no_history_flag_pins_the_append_only_behaviour(tmp_path: Path, capsys) -> None:
    path = tmp_path / "session.jsonl"
    await _run(RecordingProvider("first answer"), "--session", str(path))
    capsys.readouterr()

    fresh = RecordingProvider("second answer")
    await _run(fresh, "--session", str(path), "--no-history")
    json.loads(capsys.readouterr().out)

    assert [m.role for m in fresh.requests[0].messages] == ["system", "user"]
    # 仍然追加到同一个日志：历史不丢，只是不加载。
    persisted = await JsonlSessionStore(path).load_raw_messages()
    assert [m.role for m in persisted] == ["system", "user", "assistant", "user", "assistant"]


async def test_cli_does_not_treat_a_previous_answer_as_this_runs_answer(
    tmp_path: Path, capsys
) -> None:
    """续接会话后本轮失败：不能把上一轮 assistant 消息当成本轮答案。"""

    path = tmp_path / "session.jsonl"
    await _run(RecordingProvider("first answer"), "--session", str(path))
    capsys.readouterr()

    class Failing:
        def generate_stream(self, request):
            return stream_from_coroutine(self.generate)

        async def generate(self, request):
            raise RuntimeError("provider down")

    original = cli.OpenAiCompatibleProvider
    cli.OpenAiCompatibleProvider = lambda **kwargs: Failing()  # type: ignore[assignment]
    try:
        exit_code = await cli.run_once(_args("--session", str(path)))
    finally:
        cli.OpenAiCompatibleProvider = original  # type: ignore[assignment]
    printed = json.loads(capsys.readouterr().out)

    assert exit_code != cli.EXIT_OK
    assert printed["answer"] == "", "上一轮答案不是本轮答案"


def test_continuation_repairs_unpaired_tool_calls() -> None:
    """RC01 的续接侧：中断留下的「只有调用没有结果」的历史不能原样送下一轮。

    反例：外层取消后会话里留着 assistant(tool_calls) 而没有对应 tool 消息，
    `_continuation` 直接追加 user 问题，多数 provider 会拒绝整个请求。
    """
    history = [
        system_message("sys"),
        user_message("run both"),
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(id="slow-id", name="slow", arguments={}),
                ToolCall(id="fast-id", name="fast", arguments={}),
            ],
        ),
    ]

    resumed = cli._continuation(history, "continue")

    assert [m.role for m in resumed] == ["system", "user", "assistant", "tool", "tool", "user"]
    results = {m.tool_call_id: m for m in resumed if m.role == "tool"}
    assert set(results) == {"slow-id", "fast-id"}
    assert results["slow-id"].content == cli.INTERRUPTED_TOOL_RESULT
    assert results["slow-id"].name == "slow"
    assert resumed[-1] == user_message("continue")


def test_continuation_drops_orphan_tool_results() -> None:
    """结果配不上任何调用时（例如调用被裁掉）丢弃，否则请求本身不合法。"""
    history = [
        system_message("sys"),
        Message(role="tool", content="orphan", tool_call_id="ghost"),
        user_message("q"),
    ]

    resumed = cli._continuation(history, "continue")

    assert [m.role for m in resumed] == ["system", "user", "user"]


def test_continuation_keeps_a_fully_paired_history_unchanged() -> None:
    """正常历史必须原样保留（补齐逻辑不能重排或复制结果）。"""
    history = [
        system_message("sys"),
        user_message("q"),
        Message(role="assistant", tool_calls=[ToolCall(id="c1", name="read", arguments={})]),
        Message(role="tool", content="ok", tool_call_id="c1", name="read"),
    ]

    resumed = cli._continuation(history, "next")

    assert resumed[:-1] == history


async def test_cli_wires_memory_correction_and_recalls_it_in_a_new_chat(tmp_path, capsys) -> None:
    state = tmp_path / "state"
    facts = JsonMemoryStore(state / "facts.jsonl")
    await facts.remember("m8.clock", "HIRC 16MHz", MemoryMetadata(
        source="用户2024约定", applies_to={"product": "M8P515", "version": "V2.3"},
    ))

    class CorrectingProvider(RecordingProvider):
        async def generate(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return AssistantTurn(Message("assistant", tool_calls=[ToolCall(
                    "fix", "replace_fact", {"key": "m8.clock", "value": "16MHz only", "source": ""},
                )]), "tool_calls")
            return AssistantTurn(Message("assistant", "corrected"), "stop")

    options = ("--state-dir", str(state), "--product", "M8P515", "--version", "V2.3")
    correction = CorrectingProvider("")
    await _run(correction, *options, "--session", str(tmp_path / "correct.jsonl"))
    result = json.loads(capsys.readouterr().out)
    assert result["answer"] == "corrected"
    tools = {tool.name: tool for tool in correction.requests[0].tools}
    assert {"remember_fact", "replace_fact", "forget_fact", "search_sessions"} <= tools.keys()
    assert "source" in tools["replace_fact"].input_schema["properties"]

    fresh = RecordingProvider("restored")
    new_session = tmp_path / "fresh.jsonl"
    await _run(fresh, *options, "--session", str(new_session))
    result = json.loads(capsys.readouterr().out)
    assert result["memory_recalled"] == 1
    snapshot = fresh.requests[0].messages[1]
    assert snapshot.role == "system" and "16MHz only" in snapshot.content
    assert "2024" not in snapshot.content and "HIRC" not in snapshot.content
    assert not any("Memory (durable" in m.content for m in await JsonlSessionStore(
        new_session
    ).load_raw_messages()), "记忆只进请求，不重复写进会话历史"


async def test_default_context_keeps_a_history_larger_than_32k(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("AGENT_CONTEXT_WINDOW", raising=False)
    session = JsonlSessionStore(tmp_path / "long.jsonl")
    text = "a" * 200_000
    for message in (system_message(cli.react_system_prompt()), user_message(text)):
        await session.append(MessageEntry(1, 1, message, "t"))
    provider = RecordingProvider("current answer")
    delivery = tmp_path / "delivery"
    assert await _run(provider, "--session", str(session.path), "--no-memory",
                      "--delivery-dir", str(delivery)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["compactions"] == []
    assert len(provider.requests) == 1
    assert any(message.content == text for message in provider.requests[0].messages)
    recorded = json.loads((delivery / "delivery.json").read_text(encoding="utf-8"))
    assert recorded["settings"]["context_window"] == 1_000_000


async def test_cli_automatically_compacts_and_keeps_the_current_answer(
    tmp_path, monkeypatch, capsys,
) -> None:
    monkeypatch.setenv("AGENT_CONTEXT_WINDOW", "2000")
    session = JsonlSessionStore(tmp_path / "long.jsonl")
    for message in (
        system_message(cli.react_system_prompt()), user_message("x" * 10_000),
        Message("assistant", "y" * 2_000),
    ):
        await session.append(MessageEntry(1, 1, message, "t"))

    class CompactingProvider(RecordingProvider):
        async def generate(self, request):
            self.requests.append(request)
            text = "checkpoint" if request.purpose == "compaction" else "current answer"
            return AssistantTurn(Message("assistant", text), "stop")

    provider = CompactingProvider("")
    code = await _run(provider, "--session", str(session.path))
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["answer"] == "current answer", "压缩改变消息数，不能沿用压缩前的索引取答案"
    assert payload["compactions"]
    assert provider.requests[0].purpose == "compaction"
    assert provider.requests[0].max_output_tokens == 8192
    assert provider.requests[-1].purpose is None
    replayed = await JsonlSessionStore(session.path).load_messages()
    from app.goal_prompts import goal_guidance

    projected = provider.requests[-1].messages
    assert replayed[1:-1] == projected[1:]
    assert projected[0].content == replayed[0].content + "\n\n" + goal_guidance()
    assert (await JsonlSessionStore(session.path).load_context_anchor()) is not None
