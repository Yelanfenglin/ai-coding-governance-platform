"""会话事件日志：写入顺序、重放重建、不完整 turn 识别。"""

from __future__ import annotations

from pathlib import Path

from app.contracts import (
    AssistantTurn,
    Message,
    MessageEntry,
    StepEndEntry,
    StepStartEntry,
    ToolCall,
    ToolResult,
    TurnStartEntry,
    Usage,
)
from app.loop import RunOptions, run_agent, user_message
from app.providers.mock import MockProvider
from app.session import JsonlSessionStore
from app.tools import FunctionTool, ToolRegistry


def echo_tool() -> FunctionTool:
    return FunctionTool(
        name="echo",
        description="Echo text back",
        handler=lambda args, context: ToolResult(content=str(args.get("text", ""))),
    )


async def test_session_records_turn_and_step_events(tmp_path: Path) -> None:
    store = JsonlSessionStore(tmp_path / "session.jsonl")
    provider = MockProvider(
        [
            AssistantTurn(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "hi"})],
                ),
                stop_reason="tool_calls",
            ),
            AssistantTurn(
                message=Message(role="assistant", content="done"), stop_reason="stop"
            ),
        ]
    )

    await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry().register(echo_tool()),
            messages=[user_message("go")],
            session=store,
        )
    )

    events = await store.load_events()
    assert [type(event).__name__ for event in events] == [
        "TurnStartEntry",
        "StepStartEntry",
        "MessageEntry",
        "MessageEntry",
        "StepEndEntry",
        "StepStartEntry",
        "MessageEntry",
        "StepEndEntry",
        "TurnEndEntry",
    ]


async def test_reload_rebuilds_messages(tmp_path: Path) -> None:
    store = JsonlSessionStore(tmp_path / "session.jsonl")
    provider = MockProvider(
        [
            AssistantTurn(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(id="call-1", name="echo", arguments={"text": "persisted"})
                    ],
                ),
                stop_reason="tool_calls",
            ),
            AssistantTurn(
                message=Message(role="assistant", content="done"), stop_reason="stop"
            ),
        ]
    )

    await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry().register(echo_tool()),
            messages=[user_message("go")],
            session=store,
        )
    )

    messages = await store.load_messages()
    assert [m.content for m in messages] == ["", "persisted", "done"]
    assert messages[1].tool_call_id == "call-1"
    assert messages[1].name == "echo"


async def test_timeline_reports_turns_steps_and_status(tmp_path: Path) -> None:
    store = JsonlSessionStore(tmp_path / "session.jsonl")
    provider = MockProvider(
        [
            AssistantTurn(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "hi"})],
                ),
                stop_reason="tool_calls",
            ),
            AssistantTurn(
                message=Message(role="assistant", content="done"), stop_reason="stop"
            ),
        ]
    )

    await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry().register(echo_tool()),
            messages=[user_message("go")],
            session=store,
        )
    )

    timeline = await store.load_timeline()
    assert (timeline.turns, timeline.steps) == (1, 2)
    assert timeline.status == "completed"
    assert timeline.last_turn_open is False


async def test_interrupted_turn_is_visible_as_open(tmp_path: Path) -> None:
    store = JsonlSessionStore(tmp_path / "session.jsonl")
    await store.append(TurnStartEntry(turn=1, timestamp="2026-01-01T00:00:00+00:00"))
    await store.append(StepStartEntry(turn=1, step=1, timestamp="2026-01-01T00:00:00+00:00"))
    await store.append(
        MessageEntry(
            turn=1,
            step=1,
            message=Message(role="assistant", content="half way"),
            timestamp="2026-01-01T00:00:00+00:00",
        )
    )

    timeline = await store.load_timeline()

    assert timeline.turns == 1
    assert timeline.steps == 1
    assert timeline.status is None
    assert timeline.last_turn_open is True


async def test_nodes_form_a_chain_and_the_leaf_is_tracked(tmp_path: Path) -> None:
    """Phase E：session tree —— 每个节点带 id / parent_id，append-only。"""
    store = JsonlSessionStore(tmp_path / "session.jsonl")
    provider = MockProvider(
        [
            AssistantTurn(
                message=Message(role="assistant", content="done"), stop_reason="stop"
            )
        ]
    )

    await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry(),
            messages=[user_message("go")],
            session=store,
        )
    )
    await store.append(MessageEntry(turn=1, step=1, message=user_message("go"), timestamp="t"))

    nodes = await store.load_nodes()
    assert len(nodes) >= 2
    assert nodes[0].parent_id is None
    for previous, current in zip(nodes, nodes[1:], strict=False):
        assert current.parent_id == previous.id
    assert store.leaf_id == nodes[-1].id


async def test_ancestors_walks_back_to_the_root(tmp_path: Path) -> None:
    store = JsonlSessionStore(tmp_path / "session.jsonl")
    entries = [
        MessageEntry(turn=1, step=1, message=user_message(text), timestamp="t")
        for text in ("a", "b", "c")
    ]
    ids = [await store.append(entry) for entry in entries]
    first, _, last = ids

    path = await store.ancestors(last)

    assert [node.id for node in path] == [first, path[1].id, last]
    assert len(path) == 3
    assert path[0].parent_id is None


async def test_step_end_records_the_step_usage(tmp_path: Path) -> None:
    store = JsonlSessionStore(tmp_path / "session.jsonl")
    provider = MockProvider(
        [
            AssistantTurn(
                message=Message(role="assistant", content="done"),
                stop_reason="stop",
                usage=Usage(input_tokens=3, output_tokens=2, total_tokens=5),
            )
        ]
    )

    await run_agent(
        RunOptions(
            provider=provider, tools=ToolRegistry(), messages=[user_message("go")], session=store
        )
    )

    steps = [event for event in await store.load_events() if isinstance(event, StepEndEntry)]
    assert len(steps) == 1
    assert steps[0].usage == Usage(input_tokens=3, output_tokens=2, total_tokens=5)


async def test_missing_file_loads_as_empty(tmp_path: Path) -> None:
    store = JsonlSessionStore(tmp_path / "nope.jsonl")

    assert await store.load_events() == []
    assert await store.load_messages() == []
