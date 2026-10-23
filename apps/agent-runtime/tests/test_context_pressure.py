"""按 DSH 固定实现验证 usage 锚点、有符号变化、完整请求与有限压缩复查。"""

from pathlib import Path

from app.compaction import (
    CompactionPolicy,
    ContextMeter,
    estimate_request,
    is_compaction_summary,
)
from app.contracts import (
    AssistantTurn,
    CompactionWarning,
    ContextCompacted,
    Message,
    MessageEntry,
    ModelRequest,
    StepEndEntry,
    ToolDefinition,
    Usage,
)
from app.loop import RunOptions, run_agent, system_message, user_message
from app.providers.mock import MockProvider
from app.session import JsonlSessionStore
from app.tools import ToolRegistry


def test_usage_anchor_counts_growth_and_shrink_instead_of_taking_max() -> None:
    provider = MockProvider([])
    request = ModelRequest([user_message("old" * 100)], [])
    answer = Message("assistant", "answer")
    meter = ContextMeter()
    meter.observe(request, provider, AssistantTurn(answer, "stop", Usage(1000, 10, 1010)))

    # 老算法 max(1000, 当前字符估算) 仍是 1000；增量测量必须高于 1310。
    expanded = ModelRequest([*request.messages, answer, user_message("x" * 1200)], [])
    assert meter.measure(expanded, provider) > 1310
    # 删除旧消息也必须计入负向变化，不能沿用 1000 的地板。
    assert meter.measure(ModelRequest([answer], []), provider) < 1000


def test_measure_includes_memory_tools_and_invalidates_changed_envelope() -> None:
    provider = MockProvider([])
    base = ModelRequest([system_message("sys"), user_message("q")], [])
    memory = system_message("stable fact " * 100)
    tool = ToolDefinition("large_schema", "documentation " * 100, {"type": "object"})
    full = ModelRequest([base.messages[0], memory, base.messages[1]], [tool])
    meter = ContextMeter()
    assert meter.measure(full, provider) > meter.measure(base, provider) + 500

    meter.observe(base, provider, AssistantTurn(Message("assistant", "ok"), "stop", Usage(9000)))
    # 工具配置改变不能把旧调用的 9000 token 用量套在新请求上。
    assert meter.measure(full, provider) == estimate_request(full) < 9000


async def test_anchor_survives_session_reopen_and_still_measures_user_growth(
    tmp_path: Path,
) -> None:
    provider = MockProvider([])
    request = ModelRequest([system_message("sys"), user_message("q")], [])
    answer = Message("assistant", "ok")
    meter = ContextMeter()
    meter.observe(request, provider, AssistantTurn(answer, "stop", Usage(5000, 10, 5010)))
    path = tmp_path / "session.jsonl"
    store = JsonlSessionStore(path)
    for message in [*request.messages, answer]:
        await store.append(MessageEntry(1, 1, message, "t"))
    await store.append(StepEndEntry(1, 1, "t", Usage(5000, 10, 5010), meter.anchor))

    reopened = JsonlSessionStore(path)
    restored = ContextMeter(await reopened.load_context_anchor())
    assert restored.anchor == meter.anchor
    next_request = ModelRequest([*await reopened.load_messages(), user_message("x" * 1000)], [])
    assert restored.measure(next_request, provider) > 5260


async def test_compaction_rechecks_pressure_and_compresses_again() -> None:
    calls = []

    async def summarize(messages):
        calls.append(messages)
        return "q" * 2000 if len(calls) == 1 else "short"

    events = []
    result = await run_agent(RunOptions(
        provider=MockProvider([AssistantTurn(Message("assistant", "done"), "stop")]),
        tools=ToolRegistry(),
        messages=[system_message("sys"), user_message("x" * 6000),
                  Message("assistant", "y" * 2000), user_message("next")],
        compaction=CompactionPolicy(1000, summarizer=summarize), on_event=events.append,
    ))
    compactions = [event for event in events if isinstance(event, ContextCompacted)]
    assert result.status == "completed"
    assert len(calls) == len(compactions) == 2
    assert compactions[0].context_tokens_after >= 800
    assert compactions[1].context_tokens_after < 800
    assert sum(is_compaction_summary(message) for message in result.messages) == 1
    assert not any(isinstance(event, CompactionWarning) for event in events)


async def test_unresolved_pressure_is_bounded_and_reported_without_failing_turn() -> None:
    calls = []

    async def summarize(messages):
        calls.append(messages)
        return "z" * (5000 if len(calls) == 1 else 4500)

    events = []
    result = await run_agent(RunOptions(
        provider=MockProvider([AssistantTurn(Message("assistant", "done"), "stop")]),
        tools=ToolRegistry(),
        messages=[system_message("sys"), user_message("x" * 6000),
                  Message("assistant", "y" * 2000), user_message("next")],
        compaction=CompactionPolicy(1000, summarizer=summarize), on_event=events.append,
    ))
    assert result.status == "completed", "DSH pre-step 警告后继续 turn"
    assert len(calls) == 2, "不得无限付费压缩"
    warning, = [event for event in events if isinstance(event, CompactionWarning)]
    assert warning.attempts == 2
    assert warning.context_tokens >= warning.threshold_tokens == 800


async def test_rejected_summary_reports_failure_and_preserves_the_turn() -> None:
    async def summarize(messages):
        return "   "

    events = []
    seed = [system_message("sys"), user_message("x" * 6000), user_message("y" * 2000)]
    result = await run_agent(RunOptions(
        provider=MockProvider([AssistantTurn(Message("assistant", "done"), "stop")]),
        tools=ToolRegistry(), messages=seed,
        compaction=CompactionPolicy(1000, summarizer=summarize), on_event=events.append,
    ))
    assert result.status == "completed" and result.messages[:-1] == seed
    warning, = [event for event in events if isinstance(event, CompactionWarning)]
    assert warning.attempts == 1 and "no text summary" in warning.message
