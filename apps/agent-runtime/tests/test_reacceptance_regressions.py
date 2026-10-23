"""验收复验反例回归（2026-10-20 第二轮，已修复并迁入正式测试）。

来源：`output/reacceptance_20261020/report.md` 与 `followup_probes.py`。
覆盖：SSE 跨块拼接、并发写入不超预算、旧 JSON 记忆格式迁移、失效时间持久化、
外层 `task.cancel()` 后 provider 子任务必须收掉。

不调用真实模型或外部网络。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import asdict
from pathlib import Path

from app.contracts import AssistantTurn, MemoryItem, Message, ModelRequest
from app.loop import RunOptions, run_agent, user_message
from app.memory import JsonMemoryStore, MemoryFullError
from app.providers.mock import chunks_for_turn
from app.providers.openai_compatible import OpenAiCompatibleProvider
from app.tools import ToolRegistry


async def test_fragmented_sse_payload_is_reassembled() -> None:
    text = (
        'data: {"choices":[{"delta":{"content":"OK"},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        'data: [DONE]\n\n'
    )

    async def chunks(request):
        yield text[:13]  # A legal text chunk, ending inside the JSON object.
        yield text[13:]

    provider = OpenAiCompatibleProvider(
        "http://unused.invalid/v1", "test", stream_transport=chunks
    )
    result = await run_agent(RunOptions(
        provider=provider, tools=ToolRegistry(), messages=[user_message("go")]
    ))
    assert result.status == "completed", result.error
    assert result.messages[-1].content == "OK"


async def test_concurrent_public_writes_cannot_exceed_budget(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "facts.json", max_chars=100)
    results = await asyncio.gather(*(
        store.remember(f"key-{index}", "x" * 20) for index in range(24)
    ), return_exceptions=True)
    unexpected = [
        error for error in results
        if isinstance(error, BaseException) and not isinstance(error, MemoryFullError)
    ]
    assert not unexpected, unexpected
    used, budget = store.usage(await store.all())
    assert used <= budget, {
        "used": used, "budget": budget,
        "successful_writes": sum(not isinstance(result, BaseException) for result in results),
    }


async def test_existing_json_memory_is_not_silently_lost(tmp_path: Path) -> None:
    path = tmp_path / "facts.json"
    item = MemoryItem(key="clock", value="20MHz", source="chunk-42")
    path.write_text(
        json.dumps({"items": [asdict(item)]}, indent=2), encoding="utf-8"
    )
    store = JsonMemoryStore(path)
    recalled = await store.recall()
    assert [entry.key for entry in recalled] == ["clock"], (
        "the old JSON format is interpreted as an empty journal"
    )


async def test_invalidation_time_is_persisted_not_regenerated(tmp_path: Path) -> None:
    store = JsonMemoryStore(tmp_path / "facts.json")
    await store.remember("clock", "20MHz")
    await store.invalidate("clock")
    first = (await store.all())[0].invalidated_at
    second = (await JsonMemoryStore(store.path).all())[0].invalidated_at
    assert first == second, (first, second)


async def test_parent_task_cancellation_closes_pending_provider() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()

    class Provider:
        def generate_stream(self, request: ModelRequest):
            return self.stream()

        async def stream(self):
            try:
                started.set()
                await release.wait()
                for chunk in chunks_for_turn(
                    AssistantTurn(Message("assistant", "done"), "stop")
                ):
                    yield chunk
            finally:
                closed.set()

    task = asyncio.create_task(run_agent(RunOptions(
        provider=Provider(), tools=ToolRegistry(), messages=[user_message("go")],
        cancel_event=asyncio.Event(),
    )))
    await started.wait()
    task.cancel()  # Also how asyncio.wait_for() implements an outer request timeout.
    with contextlib.suppress(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    closed_on_cancel = closed.is_set()
    # Release and collect outstanding tasks even when the acceptance assertion will fail.
    release.set()
    for _ in range(3):
        await asyncio.sleep(0)
    assert closed_on_cancel, "pending __anext__ survives cancellation of run_agent"
