"""验证插话进入 next-step、追问进入 next-turn，以及队列的认领与清理。"""

from __future__ import annotations

import asyncio

from app.contracts import (
    AssistantTurn,
    AssistantTurnEvent,
    InboxClaimed,
    Message,
    ToolCall,
    ToolResult,
)
from app.inbox import Inbox
from app.loop import RunOptions, run_agent, user_message
from app.providers.mock import MockProvider
from app.tools import FunctionTool, ToolRegistry


def final_turn(text: str) -> AssistantTurn:
    return AssistantTurn(message=Message(role="assistant", content=text), stop_reason="stop")


def tool_call_turn(name: str = "probe") -> AssistantTurn:
    return AssistantTurn(
        message=Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c1", name=name, arguments={})],
        ),
        stop_reason="tool_calls",
    )


def probe_tool(handler) -> FunctionTool:  # noqa: ANN001
    return FunctionTool(name="probe", handler=handler)


# --- Inbox 本身 ---------------------------------------------------------------


def test_claim_next_turn_takes_all_next_step_and_one_queued_turn() -> None:
    inbox = Inbox()
    inbox.steer(user_message("s1"))
    inbox.steer(user_message("s2"))
    inbox.followup(user_message("f1"))
    inbox.followup(user_message("f2"))

    claimed = inbox.claim("next-turn")

    assert [m.content for m in claimed] == ["s1", "s2", "f1"]
    assert [m.content for m in inbox.next_turn] == ["f2"]
    assert inbox.has_pending is True


def test_claim_next_step_only_takes_next_step() -> None:
    inbox = Inbox()
    inbox.steer(user_message("s1"))
    inbox.followup(user_message("f1"))

    claimed = inbox.claim("next-step")

    assert [m.content for m in claimed] == ["s1"]
    assert [m.content for m in inbox.next_turn] == ["f1"]


def test_clear_discards_everything() -> None:
    inbox = Inbox()
    inbox.steer(user_message("s1"))
    inbox.followup(user_message("f1"))

    discarded = inbox.clear()

    assert [m.content for m in discarded] == ["s1", "f1"]
    assert inbox.has_pending is False


def test_send_requires_an_explicit_target() -> None:
    inbox = Inbox()

    try:
        inbox.send(user_message("x"), "later")  # type: ignore[arg-type]
    except ValueError as error:
        assert "unknown inbox target" in str(error)
    else:
        raise AssertionError("未知目标必须报错，不允许猜")


def test_prepend_puts_the_message_first() -> None:
    inbox = Inbox()
    inbox.steer(user_message("s1"))
    inbox.prepend("next-step", user_message("urgent"))

    assert [m.content for m in inbox.next_step] == ["urgent", "s1"]


# --- 与 loop 的集成 -----------------------------------------------------------


async def test_steer_is_injected_at_the_next_step_without_a_new_turn() -> None:
    inbox = Inbox()

    async def handler(args: dict[str, object], context: object) -> ToolResult:
        inbox.steer(user_message("改成 B"))
        return ToolResult(content="probed")

    registry = ToolRegistry().register(probe_tool(handler))
    provider = MockProvider([tool_call_turn(), final_turn("A")])
    events: list[object] = []

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=registry,
            messages=[user_message("go")],
            inbox=inbox,
            on_event=events.append,
        )
    )

    assert result.status == "completed"
    assert result.turns == 1, "steering 不开新 turn"
    assert result.steps == 2
    assert "改成 B" in [m.content for m in provider.requests[1].messages]
    claimed = [event for event in events if isinstance(event, InboxClaimed)]
    assert len(claimed) == 1
    assert [m.content for m in claimed[0].messages] == ["改成 B"]


async def test_pending_steering_keeps_the_same_turn_after_completion() -> None:
    """回合完成后仍有插话输入时，在同一回合继续执行。"""
    inbox = Inbox()
    provider = MockProvider([tool_call_turn(), final_turn("A"), final_turn("B")])

    def on_event(event: object) -> None:
        if isinstance(event, AssistantTurnEvent) and event.message.content == "A":
            inbox.steer(user_message("还有一句"))

    async def handler(args: dict[str, object], context: object) -> ToolResult:
        return ToolResult(content="probed")

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry().register(probe_tool(handler)),
            messages=[user_message("go")],
            inbox=inbox,
            on_event=on_event,
        )
    )

    assert result.status == "completed"
    assert result.turns == 1
    assert result.steps == 3, "第二段 steering 在同一个 turn 里继续"
    assert "还有一句" in [m.content for m in provider.requests[2].messages]


async def test_followup_opens_a_new_turn() -> None:
    inbox = Inbox()

    async def handler(args: dict[str, object], context: object) -> ToolResult:
        inbox.followup(user_message("还有个问题"))
        return ToolResult(content="probed")

    provider = MockProvider([tool_call_turn(), final_turn("第一回合"), final_turn("第二回合")])

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry().register(probe_tool(handler)),
            messages=[user_message("go")],
            inbox=inbox,
        )
    )

    assert result.status == "completed"
    assert result.turns == 2, "followup 开新 turn"
    assert result.steps == 3
    assert "还有个问题" in [m.content for m in provider.requests[2].messages]
    assert result.messages[-1].content == "第二回合"


async def test_cancel_clears_the_inbox_by_default() -> None:
    inbox = Inbox()
    cancel_event = asyncio.Event()

    async def handler(args: dict[str, object], context: object) -> ToolResult:
        inbox.followup(user_message("待处理追问"))
        cancel_event.set()
        return ToolResult(content="probed")

    result = await run_agent(
        RunOptions(
            provider=MockProvider([tool_call_turn(), final_turn("never")]),
            tools=ToolRegistry().register(probe_tool(handler)),
            messages=[user_message("go")],
            inbox=inbox,
            cancel_event=cancel_event,
        )
    )

    assert result.status == "aborted"
    assert inbox.has_pending is False


async def test_cancel_can_keep_the_inbox() -> None:
    inbox = Inbox()
    cancel_event = asyncio.Event()

    async def handler(args: dict[str, object], context: object) -> ToolResult:
        inbox.followup(user_message("待处理追问"))
        cancel_event.set()
        return ToolResult(content="probed")

    result = await run_agent(
        RunOptions(
            provider=MockProvider([tool_call_turn(), final_turn("never")]),
            tools=ToolRegistry().register(probe_tool(handler)),
            messages=[user_message("go")],
            inbox=inbox,
            cancel_event=cancel_event,
            clear_inbox_on_cancel=False,
        )
    )

    assert result.status == "aborted"
    assert [m.content for m in inbox.next_turn] == ["待处理追问"]


async def test_without_an_inbox_the_behaviour_is_unchanged() -> None:
    provider = MockProvider([tool_call_turn(), final_turn("done")])

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry().register(
                probe_tool(lambda args, context: ToolResult(content="probed"))
            ),
            messages=[user_message("go")],
        )
    )

    assert result.status == "completed"
    assert (result.turns, result.steps) == (1, 2)
