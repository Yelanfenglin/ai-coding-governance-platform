"""loop 行为：终止条件、多 tool calls、取消、错误、事件顺序。"""

from __future__ import annotations

import asyncio

from app.contracts import (
    AssistantTurn,
    Message,
    MessageEntry,
    ModelRequest,
    RunError,
    StepEndEntry,
    StepStartEntry,
    ToolCall,
    ToolContext,
    ToolResult,
    TurnEndEntry,
    Usage,
)
from app.loop import ABORTED_BEFORE_DISPATCH, RunOptions, run_agent, user_message
from app.providers.mock import MockProvider, stream_from_coroutine
from app.session import JsonlSessionStore
from app.tools import FunctionTool, ToolRegistry


def echo_tool() -> FunctionTool:
    return FunctionTool(
        name="echo",
        description="Echo text back",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        handler=lambda args, context: ToolResult(content=str(args["text"])),
    )


def call_turn(call_id: str, text: str) -> AssistantTurn:
    return AssistantTurn(
        message=Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id=call_id, name="echo", arguments={"text": text})],
        ),
        stop_reason="tool_calls",
    )


def final_turn(text: str) -> AssistantTurn:
    return AssistantTurn(message=Message(role="assistant", content=text), stop_reason="stop")


async def test_no_tool_call_completes_in_one_step() -> None:
    provider = MockProvider([final_turn("done")])
    events: list[object] = []

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry(),
            messages=[user_message("go")],
            on_event=events.append,
        )
    )

    assert result.status == "completed"
    assert (result.turns, result.steps) == (1, 1)
    assert len(provider.requests) == 1
    assert not any(isinstance(event, RunError) for event in events)


async def test_tool_call_then_final_answer_and_usage_is_aggregated() -> None:
    provider = MockProvider(
        [
            AssistantTurn(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(id="call-1", name="echo", arguments={"text": "hello"})
                    ],
                ),
                stop_reason="tool_calls",
                usage=Usage(input_tokens=3, output_tokens=2, total_tokens=5),
            ),
            AssistantTurn(
                message=Message(role="assistant", content="done"),
                stop_reason="stop",
                usage=Usage(input_tokens=4, output_tokens=1, total_tokens=5),
            ),
        ]
    )

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry().register(echo_tool()),
            messages=[user_message("go")],
        )
    )

    assert result.status == "completed"
    assert (result.turns, result.steps) == (1, 2)
    assert result.usage.total_tokens == 10
    assert result.messages[-1].content == "done"
    assert [m.content for m in result.messages if m.role == "tool"] == ["hello"]


async def test_multiple_tool_calls_run_in_model_order() -> None:
    provider = MockProvider(
        [
            AssistantTurn(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(id="call-1", name="echo", arguments={"text": "first"}),
                        ToolCall(id="call-2", name="echo", arguments={"text": "second"}),
                        ToolCall(id="call-3", name="echo", arguments={"text": "third"}),
                    ],
                ),
                stop_reason="tool_calls",
            ),
            final_turn("done"),
        ]
    )

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry().register(echo_tool()),
            messages=[user_message("go")],
        )
    )

    assert result.status == "completed"
    assert [m.content for m in result.messages if m.role == "tool"] == [
        "first",
        "second",
        "third",
    ]
    assert [m.tool_call_id for m in result.messages if m.role == "tool"] == [
        "call-1",
        "call-2",
        "call-3",
    ]


async def test_concludes_turn_ends_the_turn_without_another_model_call() -> None:
    submit = FunctionTool(
        name="submit",
        description="Submit the answer and end the turn",
        handler=lambda args, context: ToolResult(content="submitted", concludes_turn=True),
    )
    provider = MockProvider(
        [
            AssistantTurn(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="call-1", name="submit", arguments={})],
                ),
                stop_reason="tool_calls",
            )
        ]
    )

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry().register(submit),
            messages=[user_message("go")],
        )
    )

    assert result.status == "completed"
    assert len(provider.requests) == 1, "concludes_turn 之后不应再请求模型"
    assert result.steps == 1


async def test_concludes_turn_still_records_later_calls_in_the_same_group() -> None:
    submit = FunctionTool(
        name="submit",
        description="Submit the answer and end the turn",
        handler=lambda args, context: ToolResult(content="submitted", concludes_turn=True),
    )
    provider = MockProvider(
        [
            AssistantTurn(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(id="call-1", name="submit", arguments={}),
                        ToolCall(id="call-2", name="echo", arguments={"text": "after"}),
                    ],
                ),
                stop_reason="tool_calls",
            )
        ]
    )

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry().register(submit).register(echo_tool()),
            messages=[user_message("go")],
        )
    )

    assert result.status == "completed"
    assert len(provider.requests) == 1
    assert [m.content for m in result.messages if m.role == "tool"] == ["submitted", "after"]


async def test_length_stop_reason_becomes_max_tokens() -> None:
    provider = MockProvider(
        [
            AssistantTurn(
                message=Message(role="assistant", content="truncated"), stop_reason="length"
            )
        ]
    )

    result = await run_agent(
        RunOptions(
            provider=provider, tools=ToolRegistry(), messages=[user_message("go")]
        )
    )

    assert result.status == "max_tokens"
    assert result.steps == 1


async def test_provider_exception_becomes_error_result_and_event() -> None:
    class ExplodingProvider:
        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        def generate_stream(self, request):
            return stream_from_coroutine(lambda: self.generate(request))

        async def generate(self, request: ModelRequest) -> AssistantTurn:
            self.requests.append(request)
            raise RuntimeError("provider down")

    provider = ExplodingProvider()
    events: list[object] = []

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry(),
            messages=[user_message("go")],
            on_event=events.append,
        )
    )

    assert result.status == "error"
    assert result.error == "provider down"
    assert any(isinstance(event, RunError) for event in events)


async def test_error_stop_reason_is_reported() -> None:
    provider = MockProvider(
        [
            AssistantTurn(
                message=Message(role="assistant", content=""), stop_reason="error"
            )
        ]
    )

    result = await run_agent(
        RunOptions(provider=provider, tools=ToolRegistry(), messages=[user_message("go")])
    )

    assert result.status == "error"
    assert result.error == "model returned an error stop reason"


async def test_already_cancelled_run_never_calls_the_provider() -> None:
    provider = MockProvider([final_turn("never")])
    cancel_event = asyncio.Event()
    cancel_event.set()

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry(),
            messages=[user_message("go")],
            cancel_event=cancel_event,
        )
    )

    assert result.status == "aborted"
    assert provider.requests == []
    assert result.steps == 0


async def test_cancel_before_the_first_step_writes_no_step_events(tmp_path) -> None:
    """反例（F02 的另一半）：step 还没开始就取消，不能凭空补一条 `step/end`。

    收尾按「实际开始的 step」算 —— 一次都没开始就只关 turn，否则日志里会出现
    没有对应 `step/start` 的结束事件，等于伪造阶段。
    """
    session = JsonlSessionStore(tmp_path / "cancelled-before-step.jsonl")
    seed = [user_message("go")]
    for message in seed:
        await session.append(MessageEntry(0, 0, message, "t"))
    cancel_event = asyncio.Event()
    cancel_event.set()

    result = await run_agent(
        RunOptions(
            provider=MockProvider([final_turn("never")]),
            tools=ToolRegistry(),
            messages=seed,
            cancel_event=cancel_event,
            session=session,
        )
    )

    events = await session.load_events()
    assert result.status == "aborted"
    assert [e for e in events if isinstance(e, StepStartEntry)] == []
    assert [e for e in events if isinstance(e, StepEndEntry)] == []
    assert [e.status for e in events if isinstance(e, TurnEndEntry)] == ["aborted"]
    timeline = await session.load_timeline()
    assert timeline.last_turn_open is False


async def test_cancel_during_tools_writes_synthetic_results_for_the_rest() -> None:
    cancel_event = asyncio.Event()

    def slow_handler(args: dict[str, object], context: ToolContext) -> ToolResult:
        cancel_event.set()  # 第一个工具执行期间触发取消
        return ToolResult(content="first")

    registry = (
        ToolRegistry()
        .register(FunctionTool(name="first", description="slow", handler=slow_handler))
        .register(
            FunctionTool(
                name="echo",
                description="echo",
                handler=lambda a, c: ToolResult(content=str(a.get("text", ""))),
            )
        )
    )
    provider = MockProvider(
        [
            AssistantTurn(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(id="call-1", name="first", arguments={}),
                        ToolCall(id="call-2", name="echo", arguments={"text": "second"}),
                        ToolCall(id="call-3", name="echo", arguments={"text": "third"}),
                    ],
                ),
                stop_reason="tool_calls",
            )
        ]
    )

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=registry,
            messages=[user_message("go")],
            cancel_event=cancel_event,
        )
    )

    assert result.status == "aborted"
    assert [m.content for m in result.messages if m.role == "tool"] == [
        "first",
        ABORTED_BEFORE_DISPATCH,
        ABORTED_BEFORE_DISPATCH,
    ]
    assert len(provider.requests) == 1


async def test_failing_tool_becomes_an_observation_and_the_loop_continues() -> None:
    def explode(args: dict[str, object], context: ToolContext) -> ToolResult:
        raise RuntimeError("boom")

    registry = ToolRegistry().register(
        FunctionTool(name="explode", description="always fails", handler=explode)
    )
    provider = MockProvider(
        [
            AssistantTurn(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="call-1", name="explode", arguments={})],
                ),
                stop_reason="tool_calls",
            ),
            final_turn("recovered"),
        ]
    )

    result = await run_agent(
        RunOptions(provider=provider, tools=registry, messages=[user_message("go")])
    )

    assert result.status == "completed"
    tool_message = next(m for m in result.messages if m.role == "tool")
    assert "boom" in tool_message.content
    assert result.messages[-1].content == "recovered"


async def test_unknown_tool_is_reported_without_crashing() -> None:
    provider = MockProvider(
        [
            AssistantTurn(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="call-1", name="missing", arguments={})],
                ),
                stop_reason="tool_calls",
            ),
            final_turn("ok"),
        ]
    )

    result = await run_agent(
        RunOptions(provider=provider, tools=ToolRegistry(), messages=[user_message("go")])
    )

    assert result.status == "completed"
    assert any("unknown tool" in m.content for m in result.messages)


async def test_missing_required_argument_is_an_observation() -> None:
    provider = MockProvider(
        [
            AssistantTurn(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="call-1", name="echo", arguments={})],
                ),
                stop_reason="tool_calls",
            ),
            final_turn("ok"),
        ]
    )

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry().register(echo_tool()),
            messages=[user_message("go")],
        )
    )

    assert result.status == "completed"
    assert any("missing required argument" in m.content for m in result.messages)


async def test_turn_and_step_events_are_ordered() -> None:
    provider = MockProvider([call_turn("call-1", "hi"), final_turn("done")])
    events: list[object] = []

    await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry().register(echo_tool()),
            messages=[user_message("go")],
            on_event=events.append,
        )
    )

    # MockProvider 现在走流式：文本增量出现在 AssistantTurnEvent 之前。
    assert [type(event).__name__ for event in events] == [
        "TurnStarted",
        "StepStarted",
        "AssistantTurnEvent",  # 第一步只有 tool call，没有文本增量
        "ToolStarted",
        "ToolFinished",
        "StepFinished",
        "StepStarted",
        "AssistantTextDelta",
        "AssistantTurnEvent",
        "StepFinished",
        "TurnFinished",
    ]


async def test_provider_receives_tools_and_a_copy_of_the_messages() -> None:
    provider = MockProvider([final_turn("done")])
    registry = ToolRegistry().register(echo_tool())
    messages = [user_message("go")]

    result = await run_agent(
        RunOptions(provider=provider, tools=registry, messages=messages)
    )

    assert result.status == "completed"
    request = provider.requests[0]
    assert [tool.name for tool in request.tools] == ["echo"]
    assert [m.content for m in request.messages] == ["go"]
    assert len(result.messages) == 2, "不应修改调用方传入的 messages"
    assert len(messages) == 1
