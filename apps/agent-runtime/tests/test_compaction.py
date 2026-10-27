"""验证 token 计量、消息分组、摘要检查点与压缩失败回退。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.compaction import (
    COMPACTION_INSTRUCTION,
    SUMMARY_CLOSE_TAG,
    SUMMARY_OPEN_TAG,
    SUMMARY_SECTIONS,
    CompactionPolicy,
    ModelSummarizer,
    compact_context,
    estimate_context,
    is_compaction_summary,
    split_region,
    summarization_messages,
)
from app.contracts import (
    AssistantTurn,
    CompactionEntry,
    ContextCompacted,
    Message,
    MessageEntry,
    ToolCall,
    ToolResult,
    Usage,
)
from app.errors import ErrorCode, ProviderError
from app.loop import RunOptions, run_agent, system_message, user_message
from app.memory import JsonMemoryStore
from app.providers.mock import MockProvider
from app.session import JsonlSessionStore
from app.tools import FunctionTool, ToolRegistry


def echo_tool() -> FunctionTool:
    return FunctionTool(
        name="echo",
        description="Echo",
        handler=lambda args, context: ToolResult(content=str(args.get("text", ""))),
    )


async def fake_summarizer(transcript: list[Message]) -> str:
    # 摘要长度应小于被替换区域。
    return f"SUMMARY: condensed {len(transcript)} messages"


def big_usage() -> Usage:
    return Usage(input_tokens=900, output_tokens=10, total_tokens=910)


LONG = "x" * 2000  # 约 500 tokens（估算按字符数 / 4）


def region_seed() -> list[Message]:
    """system + 两段长消息：尾部只保留最后一组，前面那组进入压缩区域。

    `retain_tokens=160` 时正文若只有一段长消息，它会整个被保留、区域为空，压缩根本
    不会发生 —— 那样断言 `compact_context(...) is None` 的用例会「碰巧通过」，
    证明不了任何东西。
    """
    return [system_message("sys"), user_message(LONG), user_message(LONG)]


def script(context_tokens: int, text: str = LONG) -> list[AssistantTurn]:
    return [
        AssistantTurn(
            message=Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": text})],
            ),
            stop_reason="tool_calls",
            usage=Usage(
                input_tokens=context_tokens,
                output_tokens=10,
                total_tokens=context_tokens + 10,
            ),
        ),
        AssistantTurn(
            message=Message(role="assistant", content="done"), stop_reason="stop"
        ),
    ]


def test_policy_defaults_match_dsh() -> None:
    policy = CompactionPolicy(context_window=100_000)

    assert policy.threshold_ratio == 0.8, "DSH DEFAULT_THRESHOLD_RATIO"
    assert policy.retain_ratio == 0.16, "DSH DEFAULT_RETAIN_RATIO"
    assert policy.threshold_tokens == 80_000
    assert policy.retain_tokens == 16_000


def test_split_region_keeps_the_tail_as_whole_groups() -> None:
    """split_region 作用在「去掉固定 system prompt 之后的正文」上，按消息组切。"""
    body = [user_message("q1"), user_message("q2" * 200), user_message("q3" * 200)]

    region, kept = split_region(body, retain_tokens=200)

    assert [m.content for m in kept][-1:] == ["q3" * 200]
    assert estimate_context(kept) >= 200, "retain 是**最少**保留量（DSH region.ts:132-146）"
    assert len(region) + len(kept) == len(body)


def test_split_region_never_drops_the_last_group_to_fit_the_budget() -> None:
    """按完整消息组至少保留 retain_tokens，允许最后一组超出预算。"""
    body = [user_message("q1"), user_message("q2" * 400)]

    region, kept = split_region(body, retain_tokens=10)

    assert [m.content for m in kept] == ["q2" * 400]
    assert estimate_context(kept) > 10
    assert [m.content for m in region] == ["q1"]


async def test_compaction_never_summarizes_the_fixed_system_prompt() -> None:
    """system head 不进入被摘要区域；checkpoint 落在它之后，且是 user 角色。"""
    messages = [
        system_message("you are an agent"),
        user_message("q" * 400),
        user_message("tail"),
    ]
    outcome = await compact_context(
        messages, CompactionPolicy(100, retain_ratio=0.01, summarizer=fake_summarizer), 1000
    )

    assert outcome is not None
    assert outcome.messages[0] == system_message("you are an agent")
    checkpoint = outcome.messages[1]
    assert checkpoint.role == "user", "DSH region.ts:394 落成 user checkpoint，不是 system"
    assert SUMMARY_OPEN_TAG in checkpoint.content
    assert SUMMARY_CLOSE_TAG in checkpoint.content
    assert "SUMMARY:" in checkpoint.content
    assert "you are an agent" not in outcome.summary


def test_summarization_request_is_a_prefix_of_the_conversation() -> None:
    """摘要请求沿用原 system 前缀，并将摘要指令放在末尾。"""
    head = system_message("you are an agent")
    transcript = [user_message("q1"), user_message("q2")]

    messages = summarization_messages(transcript, prefix=[head])

    assert messages[0] == head
    assert messages[1:-1] == transcript, "区域消息原样保留，不展平、不重排"
    assert messages[-1].role == "user"
    assert COMPACTION_INSTRUCTION in messages[-1].content
    # 八段标题写死在这里：`SUMMARY_SECTIONS` 是从指令里派生的，用它断言会自我循环。
    for section in (
        "## Primary Request and Intent",
        "## Key Technical Concepts",
        "## Files and Code",
        "## Errors and Fixes",
        "## Pending Jobs",
        "## Current Work",
        "## Next Step",
        "## Critical Context",
    ):
        assert section in messages[-1].content, section
    assert len(SUMMARY_SECTIONS) == 8
    assert not any(message.role == "system" and message != head for message in messages)


def test_summarization_request_without_a_system_head_starts_at_the_region() -> None:
    transcript = [user_message("q1")]

    messages = summarization_messages(transcript)

    assert messages[:-1] == transcript
    assert messages[-1].role == "user"


def test_summarization_request_preserves_tool_calls() -> None:
    """摘要输入必须保留工具名称 / 参数 / call id（否则配置类事实永久丢失）。"""
    transcript = [
        user_message("Apply the requested PWM configuration."),
        Message(
            role="assistant",
            tool_calls=[
                ToolCall("config-1", "configure_pwm", {"frequency_hz": 20000, "duty_percent": 35})
            ],
        ),
        Message(role="tool", content="ok", tool_call_id="config-1", name="configure_pwm"),
    ]

    messages = summarization_messages(transcript)
    preserved = messages[:-1]

    assert preserved == transcript
    assert preserved[1].tool_calls[0].name == "configure_pwm"
    assert preserved[1].tool_calls[0].arguments == {"frequency_hz": 20000, "duty_percent": 35}
    assert preserved[2].tool_call_id == "config-1"


def test_summarization_request_can_flatten_for_text_only_providers() -> None:
    transcript = [
        Message(
            role="assistant",
            tool_calls=[ToolCall("c1", "configure_pwm", {"duty_percent": 35})],
        ),
        Message(role="tool", content="ok", tool_call_id="c1"),
    ]

    messages = summarization_messages(transcript, flatten=True)
    rendered = messages[-1].content

    for fragment in ("configure_pwm", "c1", "duty_percent", "35", "ok"):
        assert fragment in rendered, fragment


async def test_compaction_discards_a_blank_summary() -> None:
    """空白摘要保持原上下文。"""

    async def blank(transcript: list[Message]) -> str:
        return "   \n\t  "

    messages = region_seed()
    outcome = await compact_context(
        messages, CompactionPolicy(1000, summarizer=blank), 900
    )

    assert outcome is None, "空摘要会留下只有标题的 checkpoint，必须回退到原上下文"


async def test_compaction_discards_a_summary_that_is_not_shorter() -> None:
    """摘要检查点长度必须小于被替换区域。"""

    async def verbose(transcript: list[Message]) -> str:
        return "y" * 10_000

    messages = region_seed()
    outcome = await compact_context(
        messages, CompactionPolicy(1000, summarizer=verbose), 900
    )

    assert outcome is None, "更长的摘要会让上下文压力反增，必须拒绝"


async def test_compaction_returns_none_when_nothing_can_be_compacted() -> None:
    """没有可压缩消息区域时返回 None。"""
    messages = [system_message("sys"), user_message("q" * 400)]

    outcome = await compact_context(
        messages, CompactionPolicy(1000, summarizer=fake_summarizer), 900
    )

    assert outcome is None


async def test_compaction_triggers_at_the_threshold() -> None:
    policy = CompactionPolicy(context_window=1000, summarizer=fake_summarizer)
    provider = MockProvider(script(900))
    events: list[object] = []

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry().register(echo_tool()),
            messages=[system_message("sys"), user_message(LONG)],
            compaction=policy,
            on_event=events.append,
        )
    )

    assert result.status == "completed"
    compacted = [event for event in events if isinstance(event, ContextCompacted)]
    assert len(compacted) == 1
    # 为当前上下文和上次请求用量构造压力场景。
    assert compacted[0].context_tokens >= 900
    assert compacted[0].compacted_messages >= 1
    assert any(is_compaction_summary(m) for m in result.messages), "摘要进入上下文"


async def test_compaction_does_not_trigger_below_the_threshold() -> None:
    policy = CompactionPolicy(context_window=100_000, summarizer=fake_summarizer)
    provider = MockProvider(script(900))
    events: list[object] = []

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry().register(echo_tool()),
            messages=[system_message("sys"), user_message("go")],
            compaction=policy,
            on_event=events.append,
        )
    )

    assert result.status == "completed"
    assert not any(isinstance(event, ContextCompacted) for event in events)


async def test_compaction_failure_falls_back_to_the_original_context() -> None:
    async def broken(transcript: list[Message]) -> str:
        raise RuntimeError("summarizer down")

    policy = CompactionPolicy(context_window=1000, summarizer=broken)
    provider = MockProvider(script(900))
    events: list[object] = []

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry().register(echo_tool()),
            messages=region_seed(),
            compaction=policy,
            on_event=events.append,
        )
    )

    assert result.status == "completed", "压缩失败不能中断请求"
    assert not any(isinstance(event, ContextCompacted) for event in events)
    assert not any(is_compaction_summary(m) for m in result.messages)


async def test_compaction_is_recorded_without_deleting_original_events(
    tmp_path: Path,
) -> None:
    policy = CompactionPolicy(context_window=1000, summarizer=fake_summarizer)
    session = JsonlSessionStore(tmp_path / "session.jsonl")
    seed = [system_message("sys"), user_message("long-go" + LONG), user_message("tail")]
    # 初始上下文由调用方写入会话；重放需要它才能还原完整活动上下文。
    for message in seed:
        await session.append(MessageEntry(0, 0, message, "t"))

    await run_agent(
        RunOptions(
            provider=MockProvider(script(900)),
            tools=ToolRegistry().register(echo_tool()),
            messages=seed,
            compaction=policy,
            session=session,
        )
    )

    events = await session.load_events()
    assert sum(1 for event in events if isinstance(event, CompactionEntry)) == 1
    # 原始事件仍在：压缩不删历史（load_raw_messages 读未压缩的完整历史）
    persisted = await session.load_raw_messages()
    assert any(m.role == "tool" for m in persisted)
    assert any(m.role == "assistant" for m in persisted)
    # load_messages 是「活动上下文」：压缩后 = 原 system head + user checkpoint + 保留区
    live = await session.load_messages()
    assert live[0] == system_message("sys"), "重放不把 checkpoint 提成 system"
    assert is_compaction_summary(live[1])
    assert "SUMMARY:" in live[1].content


async def test_summary_usage_is_accounted() -> None:
    summarizer = ModelSummarizer(
        MockProvider(
            [
                AssistantTurn(
                    message=Message(role="assistant", content="S"),
                    stop_reason="stop",
                    usage=big_usage(),
                )
            ]
        )
    )
    policy = CompactionPolicy(context_window=1000, summarizer=summarizer)

    result = await run_agent(
        RunOptions(
            provider=MockProvider(script(900)),
            tools=ToolRegistry().register(echo_tool()),
            messages=region_seed(),
            compaction=policy,
        )
    )

    assert result.status == "completed"
    assert result.usage.total_tokens >= 1820, "压缩请求本身也计入 usage"


async def test_compact_context_returns_none_without_a_summarizer() -> None:
    policy = CompactionPolicy(context_window=100)

    assert await compact_context([user_message("q")], policy, 1000) is None


async def test_summary_abort_propagates_instead_of_falling_back() -> None:
    """RC02：摘要以 ABORTED 失败时必须向上传播，不能当成普通压缩失败吞掉。

    反例根因：`ProviderError.code` 是 `str`，用 `error.code is ErrorCode.ABORTED`
    比较永远为 False，于是取消被当成「摘要质量不好」回退，主模型请求继续跑。
    """

    async def abort(transcript: list[Message]) -> str:
        raise ProviderError("summary aborted", ErrorCode.ABORTED)

    messages = region_seed()

    with pytest.raises(ProviderError) as caught:
        await compact_context(
            messages, CompactionPolicy(1000, summarizer=abort), 900
        )
    assert caught.value.code == ErrorCode.ABORTED


async def test_summary_failure_by_code_only_aborted_propagates() -> None:
    """非 ABORTED 的 ProviderError 仍按可回退处理（两种 code 走不同分支）。"""

    async def fail(transcript: list[Message]) -> str:
        raise ProviderError("provider broke", ErrorCode.PROVIDER)

    messages = region_seed()
    outcome = await compact_context(
        messages, CompactionPolicy(1000, summarizer=fail), 900
    )

    assert outcome is None


async def test_rejected_summary_still_records_its_usage() -> None:
    """RC03：摘要被缩短/非空验收拒绝时，这次调用已经计费，不能整笔丢掉。

    反例：摘要报 40 tokens、主回答报 3 tokens，压缩结果被拒绝后 run 只报 3。
    """
    summary_provider = MockProvider(
        [AssistantTurn(Message(role="assistant", content="S" * 5000), "stop", usage=big_usage())]
    )
    policy = CompactionPolicy(1000, summarizer=ModelSummarizer(summary_provider))
    recorded: list[Usage] = []

    outcome = await compact_context(
        region_seed(),
        policy,
        900,
        on_usage=recorded.append,
    )

    assert outcome is None, "5000 字符的摘要比区域更长，应被拒绝"
    assert [usage.total_tokens for usage in recorded] == [910], "每次调用记一次，且只记一次"


async def test_rejected_summary_usage_reaches_the_run_total() -> None:
    """端到端：摘要被拒绝，run 的 usage 仍包含两次调用的和。"""
    summary_provider = MockProvider(
        [AssistantTurn(Message(role="assistant", content="S" * 5000), "stop", usage=big_usage())]
    )
    main_provider = MockProvider(
        [
            AssistantTurn(
                Message(role="assistant", content="done"),
                "stop",
                usage=Usage(input_tokens=2, output_tokens=1, total_tokens=3),
            )
        ]
    )

    result = await run_agent(
        RunOptions(
            provider=main_provider,
            tools=ToolRegistry(),
            messages=region_seed(),
            compaction=CompactionPolicy(1000, summarizer=ModelSummarizer(summary_provider)),
        )
    )

    assert len(summary_provider.requests) == 1, "摘要确实被调用过一次"
    assert result.usage.total_tokens == 913, "摘要 910 + 主调用 3"


async def test_summarization_request_reuses_the_memory_prefixed_request(tmp_path: Path) -> None:
    """RC05：开启记忆时，摘要请求的前缀必须等于主请求的前缀。

    反例：记忆块只加进主请求（system head + memory + ...），摘要请求却只有
    system head，前缀不一致 —— 此时说「摘要是真实请求前缀」不成立。
    """
    memory = JsonMemoryStore(tmp_path / "facts.jsonl")
    await memory.remember("clock", "20000 Hz")
    registry = ToolRegistry().register(
        FunctionTool(name="read", handler=lambda args, ctx: ToolResult("R" * 6000))
    )
    main_provider = MockProvider(
        [
            AssistantTurn(
                Message(role="assistant", tool_calls=[ToolCall("read-1", "read")]), "tool_calls"
            ),
            AssistantTurn(Message(role="assistant", content="done"), "stop"),
        ]
    )
    summary_provider = MockProvider(
        [AssistantTurn(Message(role="assistant", content="checkpoint"), "stop")]
    )

    result = await run_agent(
        RunOptions(
            provider=main_provider,
            tools=registry,
            messages=[system_message("original sys"), user_message("Q" * 2000)],
            memory=memory,
            compaction=CompactionPolicy(1000, summarizer=ModelSummarizer(summary_provider)),
        )
    )

    original = main_provider.requests[0].messages
    auxiliary = summary_provider.requests[0].messages
    assert [m.role for m in auxiliary] == ["system", "system", "user", "user"]
    assert auxiliary[:-1] == original, "去掉末尾指令后必须与主请求前缀逐条相同"
    assert any("clock: 20000 Hz" in m.content for m in auxiliary), "记忆块要一起进摘要前缀"
    # 记忆只参与请求组装，不写进上下文/会话历史（每个 turn 重新构建快照）。
    assert not any("clock: 20000 Hz" in m.content for m in result.messages)
