"""验证记忆容量、增删改、适用范围过滤、画像分库、历史检索与快照注入。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.contracts import (
    AssistantTurn,
    MemoryRecalled,
    Message,
    MessageEntry,
    ToolCall,
    ToolResult,
)
from app.inbox import Inbox
from app.loop import RunOptions, run_agent, system_message, user_message
from app.memory import (
    DEFAULT_MEMORY_CHARS,
    DEFAULT_USER_CHARS,
    JsonMemoryStore,
    MemoryFullError,
    MemoryMetadata,
)
from app.memory_tools import memory_tools, search_sessions_tool
from app.providers.mock import MockProvider
from app.session import JsonlSessionStore
from app.session_search import SessionSearch
from app.tools import FunctionTool, ToolRegistry


def final_turn(text: str) -> AssistantTurn:
    return AssistantTurn(message=Message(role="assistant", content=text), stop_reason="stop")


def tool_turn(name: str, key: str = "k", value: str = "v") -> AssistantTurn:
    return AssistantTurn(
        message=Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="c1", name=name, arguments={"key": key, "value": value})
            ],
        ),
        stop_reason="tool_calls",
    )


# --- 有界写入 ---------------------------------------------------------------


async def test_remember_then_recall_and_persist(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memory.json")

    await store.remember(
        "chip.target", "PIC16F877A", MemoryMetadata(source="session-1", scope="project")
    )
    await store.remember("answer.language", "中文", MemoryMetadata(source="session-1"))

    reloaded = JsonMemoryStore(tmp_path / "memory.json")
    items = await reloaded.recall()

    assert {item.key for item in items} == {"chip.target", "answer.language"}
    assert next(item for item in items if item.key == "chip.target").source == "session-1"


async def test_budget_is_bounded_and_fails_loudly(tmp_path: Path) -> None:
    """容量不足时返回错误并保留已有条目。"""
    store = JsonMemoryStore(tmp_path / "memory.json", max_chars=80)

    await store.remember("a", "x" * 10)
    try:
        await store.remember("b", "y" * 200)
    except MemoryFullError as error:
        assert "budget exceeded" in str(error)
    else:
        raise AssertionError("超预算必须报错")

    items = await store.all()
    assert [item.key for item in items] == ["a"], "旧条目不能被静默淘汰"


async def test_default_budgets_are_configurable(tmp_path: Path) -> None:
    assert DEFAULT_MEMORY_CHARS == 6000
    assert DEFAULT_USER_CHARS == 3000
    assert JsonMemoryStore(tmp_path / "m.json", max_chars=123).max_chars == 123


# --- 更新 / 删除 / 失效 -----------------------------------------------------


async def test_remember_same_key_updates_in_place(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memory.json")

    await store.remember("chip.target", "PIC16F877A")
    await store.remember("chip.target", "PIC18F4550")

    items = await store.all()
    assert len(items) == 1
    assert items[0].value == "PIC18F4550"


async def test_replace_remove_and_invalidate(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memory.json")
    await store.remember("k", "old")

    assert (await store.replace("k", "new")) is not None
    assert (await store.all())[0].value == "new"
    assert await store.replace("missing", "x") is None

    assert await store.invalidate("k") is True
    assert await store.recall() == [], "软失效后不再召回"
    assert (await store.all())[0].invalidated_at is not None, "记录保留，可审计"

    assert await store.remove("k") is True
    assert await store.remove("k") is False


async def test_correction_can_update_or_clear_provenance_without_losing_scope(
    tmp_path: Path,
) -> None:
    store = JsonMemoryStore(tmp_path / "facts.jsonl")
    await store.remember("m8.clock", "HIRC 16MHz", MemoryMetadata(
        source="用户2024约定", applies_to={"chip": "M8P515", "version": "V2.3"},
    ))
    registry = ToolRegistry()
    for tool in memory_tools(store):
        registry.register(tool)
    result = await registry.execute(ToolCall("fix", "replace_fact", {
        "key": "m8.clock", "value": "16MHz; source unconfirmed", "source": "",
    }))
    assert not result.is_error
    saved = json.loads(result.content)
    assert saved["source"] == ""
    assert saved["applies_to"] == {"chip": "M8P515", "version": "V2.3"}
    reloaded = JsonMemoryStore(store.path)
    assert "2024" not in await reloaded.render()
    assert "HIRC" not in await reloaded.render()
    # product/chip 两种调用约定必须指同一个条件，不能仅靠 version 不同才隔离。
    assert await reloaded.recall(applies_to={"product": "M9P6362", "version": "V2.3"}) == []


async def test_memory_tools_route_profile_and_use_default_model_scope(tmp_path: Path) -> None:
    facts = JsonMemoryStore(tmp_path / "facts.jsonl")
    profile = JsonMemoryStore(tmp_path / "profile.jsonl", default_scope="user")
    registry = ToolRegistry()
    for tool in memory_tools(
        facts, user_profile=profile, applies_to={"product": "M8P515", "version": "V2.3"},
        default_source="session:confirmed",
    ):
        registry.register(tool)
    for args in (
        {"key": "m8.clock", "value": "16MHz"},
        {"key": "language", "value": "中文", "target": "user"},
    ):
        assert not (await registry.execute(ToolCall("c", "remember_fact", args))).is_error
    fact, = await facts.all()
    preference, = await profile.all()
    assert fact.source == "session:confirmed"
    assert fact.applies_to == {"product": "M8P515", "version": "V2.3"}
    assert preference.scope == "user" and preference.applies_to == {}


async def test_applies_to_filters_out_conflicting_facts(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memory.json")
    await store.remember(
        "pwm.period", "PR2=0xFF", MemoryMetadata(applies_to={"chip": "PIC16F877A"})
    )
    await store.remember("answer.language", "中文")

    current = await store.recall(applies_to={"chip": "PIC18F4550"})
    matching = await store.recall(applies_to={"chip": "PIC16F877A"})

    assert [item.key for item in current] == ["answer.language"], "换型号后旧事实不召回"
    assert {item.key for item in matching} == {"pwm.period", "answer.language"}


async def test_render_shows_usage_and_skips_empty_stores(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memory.json", max_chars=6000)
    assert await store.render() == "", "空库不注入"

    await store.remember("chip.target", "PIC16F877A")
    block = await store.render()

    assert "## Memory (durable facts)" in block
    assert "/6000 chars]" in block
    assert "chip.target: PIC16F877A" in block


# --- 用户画像分库 -----------------------------------------------------------


async def test_user_profile_store_uses_its_own_scope_and_budget(tmp_path: Path) -> None:
    profile = JsonMemoryStore(
        tmp_path / "user.json", max_chars=DEFAULT_USER_CHARS, default_scope="user"
    )
    await profile.remember("style", "简洁、带 citation")

    item = (await profile.all())[0]
    assert item.scope == "user"
    assert "User profile" in await profile.render()


# --- Agent 工具 -------------------------------------------------------------


async def test_memory_tools_let_the_agent_write(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memory.json")
    registry = ToolRegistry()
    for tool in memory_tools(store):
        registry.register(tool)

    result = await registry.execute(
        ToolCall(
            id="c1",
            name="remember_fact",
            arguments={"key": "chip.target", "value": "PIC16F877A", "source": "chunk-42"},
        )
    )

    assert result.is_error is False
    items = await store.all()
    assert items[0].key == "chip.target"
    assert items[0].source == "chunk-42"
    assert registry.execution_mode("remember_fact") == "exclusive", "写入是屏障"


async def test_memory_full_is_an_observation_not_a_crash(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memory.json", max_chars=40)
    registry = ToolRegistry()
    for tool in memory_tools(store):
        registry.register(tool)

    result = await registry.execute(
        ToolCall(id="c1", name="remember_fact", arguments={"key": "k", "value": "x" * 500})
    )

    assert result.is_error is True
    assert "memory full" in result.content


# --- 会话历史检索 --------------------------------------------------------


async def test_session_search_finds_past_messages(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions"
    session = JsonlSessionStore(session_dir / "s1.jsonl")
    await session.append(
        MessageEntry(
            turn=1,
            step=1,
            message=Message(role="tool", content="ADCON0 低两位 是通道选择"),
            timestamp="t",
        )
    )

    search = SessionSearch(tmp_path / "search.db")
    assert await search.index_directory(session_dir) == 1

    hits = await search.search("ADCON0")
    assert len(hits) == 1
    assert "通道选择" in hits[0].content
    assert hits[0].session == "s1"

    assert await search.search("完全没有的东西") == []
    assert await search.search("") == []


async def test_session_search_falls_back_to_like_for_substrings(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions"
    session = JsonlSessionStore(session_dir / "s1.jsonl")
    await session.append(
        MessageEntry(
            turn=1,
            step=1,
            message=Message(role="tool", content="通道选择位由 CHS 决定"),
            timestamp="t",
        )
    )

    search = SessionSearch(tmp_path / "search.db")
    await search.index_directory(session_dir)

    hits = await search.search("通道选择位")
    assert [hit.content for hit in hits] == ["通道选择位由 CHS 决定"]


async def test_session_search_indexes_tool_metadata(tmp_path: Path) -> None:
    """H02：只索引 content 会丢掉工具名 / 参数 / call id 里的历史事实。

    反例：一次「只有工具调用、没有正文」的 assistant 消息，在旧实现里被当成空消息
    跳过；配置类事实（PWM 频率、通道号）只存在于参数里，于是永远查不回来。
    """
    session_dir = tmp_path / "sessions"
    session = JsonlSessionStore(session_dir / "s1.jsonl")
    await session.append(
        MessageEntry(
            turn=1,
            step=1,
            message=Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall("call-42", "configure_pwm", {"frequency_hz": 20000})
                ],
            ),
            timestamp="t",
        )
    )
    await session.append(
        MessageEntry(
            turn=1,
            step=1,
            message=Message(
                role="tool", content="ok", tool_call_id="call-42", name="configure_pwm"
            ),
            timestamp="t",
        )
    )

    search = SessionSearch(tmp_path / "search.db")
    assert await search.index_directory(session_dir) == 2, "无正文的工具调用不能被跳过"

    # 工具名、参数值、call id 都能作为检索键命中同一批历史消息。
    for query in ("configure_pwm", "20000", "call-42"):
        hits = await search.search(query)
        assert hits, query
        assert any(hit.tool_name == "configure_pwm" for hit in hits), query

    call_hit = next(hit for hit in await search.search("20000") if hit.tool_calls)
    assert "frequency_hz" in call_hit.tool_calls
    assert "call-42" in call_hit.render(), "返回结果要保留工具关联信息"

    result_hit = next(
        hit for hit in await search.search("call-42") if hit.role == "tool"
    )
    assert result_hit.tool_call_id == "call-42"
    assert result_hit.tool_name == "configure_pwm"


async def test_session_search_like_fallback_also_covers_tool_fields(tmp_path: Path) -> None:
    """兜底路径不能只 LIKE content，否则中文子串命中时同样丢掉工具参数。"""
    session_dir = tmp_path / "sessions"
    session = JsonlSessionStore(session_dir / "s1.jsonl")
    await session.append(
        MessageEntry(
            turn=1,
            step=1,
            message=Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall("c-1", "set_channel", {"通道选择位": "ADCON0"})],
            ),
            timestamp="t",
        )
    )

    search = SessionSearch(tmp_path / "search.db")
    await search.index_directory(session_dir)

    hits = await search.search("通道选择位")
    assert hits, "LIKE 兜底也必须覆盖 tool_calls"
    assert hits[0].tool_name == "set_channel"


async def test_search_sessions_tool(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions"
    session = JsonlSessionStore(session_dir / "s1.jsonl")
    await session.append(
        MessageEntry(
        turn=2, step=1, message=Message(role="tool", content="PWM 周期 由 PR2 决定"), timestamp="t"
    )
    )
    search = SessionSearch(tmp_path / "search.db")
    await search.index_directory(session_dir)

    registry = ToolRegistry().register(search_sessions_tool(search))
    result = await registry.execute(
        ToolCall(id="c1", name="search_sessions", arguments={"query": "PWM"})
    )

    assert "PR2" in result.content
    assert registry.execution_mode("search_sessions") == "parallel"


# --- 注入 loop --------------------------------------------------------------


async def test_memory_is_injected_after_the_system_prompt(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "memory.json")
    await store.remember("chip.target", "PIC16F877A")
    provider = MockProvider([final_turn("done")])
    events: list[object] = []

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry(),
            messages=[system_message("sys"), user_message("go")],
            memory=store,
            on_event=events.append,
        )
    )

    assert result.status == "completed"
    roles = [m.role for m in provider.requests[0].messages]
    content = provider.requests[0].messages[1].content
    assert roles[:2] == ["system", "system"], "记忆块紧跟 system prompt"
    assert "chip.target: PIC16F877A" in content
    assert user_message("go").content in provider.requests[0].messages[-1].content

    recalled = [event for event in events if isinstance(event, MemoryRecalled)]
    assert len(recalled) == 1
    assert [item.key for item in recalled[0].items] == ["chip.target"]


async def test_memory_snapshot_is_replaced_not_stacked(tmp_path: Path) -> None:
    """每个 turn 刷新一次：不能越积越多。"""
    store = JsonMemoryStore(tmp_path / "memory.json")
    await store.remember("chip.target", "PIC16F877A")
    inbox = Inbox()

    async def handler(args: dict[str, object], context: object) -> ToolResult:
        inbox.followup(user_message("还有个问题"))
        return ToolResult(content="probed")

    registry = ToolRegistry().register(FunctionTool(name="probe", handler=handler))
    provider = MockProvider([tool_turn("probe"), final_turn("A"), final_turn("B")])

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=registry,
            messages=[system_message("sys"), user_message("go")],
            memory=store,
            inbox=inbox,
        )
    )

    assert result.turns == 2
    blocks = [m for m in provider.requests[-1].messages if "chip.target" in m.content]
    assert len(blocks) == 1, "同一份记忆只出现一次"


async def test_no_memory_store_injects_nothing() -> None:
    provider = MockProvider([final_turn("done")])

    await run_agent(
        RunOptions(
            provider=provider, tools=ToolRegistry(), messages=[user_message("go")]
        )
    )

    assert not any("Memory (durable facts)" in m.content for m in provider.requests[0].messages)


async def test_old_index_upgrade_keeps_existing_rows(tmp_path: Path) -> None:
    """RC04：第一次查询旧索引不能把历史清空。

    反例：`_ensure_schema` 发现旧表缺工具列就 DROP 重建，`search` 也会走这段，
    于是「查询一次 → 全部历史索引消失 → 返回正常的零命中」，调用方毫无察觉。
    """
    db = tmp_path / "old-index.db"
    with sqlite3.connect(db) as connection:
        connection.execute(
            "CREATE VIRTUAL TABLE messages USING fts5("
            "session, turn UNINDEXED, step UNINDEXED, role UNINDEXED, content,"
            " tokenize='unicode61')"
        )
        connection.execute(
            "INSERT INTO messages VALUES ('old-session', 1, 1, 'user', 'existingfact')"
        )

    hits = await SessionSearch(db).search("existingfact")

    assert [hit.content for hit in hits] == ["existingfact"], "升级不能丢掉可迁移的旧行"
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1


async def test_upgraded_index_still_accepts_new_tool_fields(tmp_path: Path) -> None:
    """升级后的表能继续写入新列（旧行留空，重新索引的会话补全）。"""
    db = tmp_path / "old-index.db"
    with sqlite3.connect(db) as connection:
        connection.execute(
            "CREATE VIRTUAL TABLE messages USING fts5("
            "session, turn UNINDEXED, step UNINDEXED, role UNINDEXED, content,"
            " tokenize='unicode61')"
        )
        connection.execute("INSERT INTO messages VALUES ('old-session', 1, 1, 'user', 'kept')")

    session = JsonlSessionStore(tmp_path / "sessions" / "s2.jsonl")
    await session.append(
        MessageEntry(
            turn=1,
            step=1,
            message=Message(
                role="assistant", content="", tool_calls=[ToolCall("c-9", "configure_pwm", {})]
            ),
            timestamp="t",
        )
    )

    search = SessionSearch(db)
    assert await search.index_file(session.path) == 1
    assert [hit.content for hit in await search.search("kept")] == ["kept"], "旧行仍在"
    hits = await search.search("configure_pwm")
    assert hits and hits[0].tool_name == "configure_pwm"
