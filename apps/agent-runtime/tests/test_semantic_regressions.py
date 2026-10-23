"""语义对齐回归：上游保证 → 本地保留/改变 → 反例。

S1 压缩按**当前**上下文触发（DSH compaction-basic/index.ts 重新测量 session）
S2 摘要请求接入取消链路（DSH summarizer.ts 传 agent signal）
S3 父运行取消时收拢已启动的工具任务（DSH tool-calls.ts 的 settle 保证）
S4 同一持久化路径的提交共享一把锁（Hermes 基于路径的文件锁）
S5 空 completion 识别为 EMPTY_RESPONSE（DSH translate.ts）
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

from app.compaction import CompactionPolicy, ModelSummarizer, is_compaction_summary
from app.contracts import (
    AssistantTurn,
    Message,
    ToolCall,
    ToolResult,
    Usage,
)
from app.errors import ErrorCode, ProviderError
from app.loop import RunOptions, run_agent, user_message
from app.memory import JsonMemoryStore, MemoryFullError, _file_lock, _replay
from app.providers.mock import MockProvider, stream_from_coroutine
from app.providers.openai_compatible import chunks_from_sse
from app.retry import RetryPolicy
from app.tools import FunctionTool, ToolRegistry


def final(text: str = "done") -> AssistantTurn:
    return AssistantTurn(Message(role="assistant", content=text), "stop")


def call(name: str, usage: Usage | None = None) -> AssistantTurn:
    return AssistantTurn(
        Message(role="assistant", tool_calls=[ToolCall("call-1", name)]),
        "tool_calls",
        usage=usage,
    )


# --- S1：压缩必须依据当前上下文（含新增工具结果）-------------------------------


async def test_compaction_triggers_after_a_large_tool_result() -> None:
    summaries: list[int] = []

    async def summarize(messages: list[Message]) -> str:
        summaries.append(len(messages))
        return "checkpoint"

    provider = MockProvider([call("read", Usage(input_tokens=10)), final()])
    registry = ToolRegistry().register(
        FunctionTool(name="read", handler=lambda args, ctx: ToolResult("X" * 5000))
    )
    policy = CompactionPolicy(context_window=128, summarizer=summarize)

    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=registry,
            # 首条消息本身要足够大，否则「被替换区域」小于 framed checkpoint，
            # 会先被 DSH 的缩短验收（region.ts:398-406）挡掉 —— 那是另一条规则。
            messages=[user_message("read " * 400)],
            compaction=policy,
        )
    )

    assert result.status == "completed"
    # 上一次 input usage 只有 10 token，但下一个请求已经远超窗口：必须压缩。
    assert summaries, "新增的工具结果没有进入压力判断，压缩未触发"
    checkpoints = [m for m in result.messages if is_compaction_summary(m)]
    assert checkpoints, "压缩结果要落成 user checkpoint（DSH summarizer.ts::frameSummary）"
    assert "checkpoint" in checkpoints[0].content


# --- S2：摘要请求也要能取消 ----------------------------------------------------


async def test_summarization_is_cancellable() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    cancel = asyncio.Event()

    class Provider:
        request = None

        def generate_stream(self, request):
            return stream_from_coroutine(lambda: self.generate(request))

        async def generate(self, request):
            self.request = request
            entered.set()
            await release.wait()
            return final("summary")

    provider = Provider()
    task = asyncio.create_task(
        run_agent(
            RunOptions(
                provider=provider,
                tools=ToolRegistry(),
                messages=[user_message("X" * 2000)],
                cancel_event=cancel,
                compaction=CompactionPolicy(128, summarizer=ModelSummarizer(provider)),
            )
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert provider.request is not None
        assert provider.request.cancelled is not None, "摘要请求没有取消探针"
        cancel.set()
        done, _ = await asyncio.wait({task}, timeout=0.5)
        assert done, "摘要期间设置取消后，run 仍在挂起"
        result = task.result()
        assert result.status == "aborted"
    finally:
        release.set()
        if not task.done():
            task.cancel()
            with __import__("contextlib").suppress(asyncio.CancelledError):
                await task


# --- S3：父运行取消后，工具子任务必须被收拢 -------------------------------------


async def test_tool_tasks_do_not_survive_parent_cancellation() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    effects: list[str] = []

    async def work(args: dict[str, object], context: object) -> ToolResult:
        entered.set()
        await release.wait()
        effects.append("work completed after parent cancellation")
        return ToolResult("ok")

    registry = ToolRegistry().register(FunctionTool(name="work", handler=work))
    parent = asyncio.create_task(
        run_agent(
            RunOptions(
                provider=MockProvider([call("work"), final()]),
                tools=registry,
                messages=[user_message("work")],
            )
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        parent.cancel()
        with __import__("contextlib").suppress(asyncio.CancelledError):
            await parent
        release.set()
        for _ in range(3):
            await asyncio.sleep(0)
        assert effects == [], effects
    finally:
        release.set()
        if not parent.done():
            parent.cancel()
            with __import__("contextlib").suppress(asyncio.CancelledError):
                await parent


# --- S4：同一路径的提交共享锁 --------------------------------------------------


async def test_memory_stores_on_the_same_path_share_the_commit_lock(tmp_path: Path) -> None:
    path = tmp_path / "memory.jsonl"
    first = JsonMemoryStore(path, max_chars=120)
    second = JsonMemoryStore(path, max_chars=120)

    assert first._write_lock is second._write_lock, "锁只属于单个 store 对象"

    writes = [
        first.remember("a", "X" * 60),
        second.remember("b", "Y" * 60),
        first.remember("c", "Z" * 60),
    ]
    results = await asyncio.gather(*writes, return_exceptions=True)
    unexpected = [
        error
        for error in results
        if isinstance(error, BaseException) and not isinstance(error, MemoryFullError)
    ]
    assert not unexpected, unexpected

    used, budget = first.usage(await first.all())
    assert used <= budget, {"used": used, "budget": budget}


def test_memory_commit_blocks_another_process(tmp_path: Path) -> None:
    """M06：进程内锁管不住第二个进程，提交必须落在 OS 文件锁上。

    反例：另一个进程在锁被持有时提交，如果只是 threading.Lock，它会立刻写成功 ——
    两个进程各自依据旧状态通过预算检查，最后顶穿预算或互相覆盖。
    """
    path = tmp_path / "memory.jsonl"
    script = (
        "import asyncio, sys\n"
        "from pathlib import Path\n"
        "from app.memory import JsonMemoryStore\n"
        # 先报到：让父进程等到「导入完成、马上要提交」这一刻再计时，
        # 否则量到的是解释器启动耗时，证明不了锁的作用。
        "print('READY', flush=True)\n"
        "asyncio.run(JsonMemoryStore(Path(sys.argv[1])).remember('k', 'v'))\n"
        "print('DONE', flush=True)\n"
    )
    root = Path(__file__).resolve().parents[1]

    with _file_lock(path):
        child = subprocess.Popen(
            [sys.executable, "-c", script, str(path)],
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "READY"
        time.sleep(0.4)
        blocked = child.poll() is None

    assert blocked, "锁被持有时另一个进程不应写成功（旧实现只有进程内锁）"
    assert child.wait(timeout=30) == 0, child.stderr.read() if child.stderr else ""
    rest = child.stdout.read() if child.stdout else ""
    assert "DONE" in rest

    items = _replay(path.read_text(encoding="utf-8"), "project")
    assert [(item.key, item.value) for item in items] == [("k", "v")]


# --- S5：空 completion 是 EMPTY_RESPONSE，不是成功 ------------------------------


async def test_empty_completion_is_retried_not_completed() -> None:
    class Provider:
        calls = 0

        def generate_stream(self, request):
            self.calls += 1
            text = "" if self.calls == 1 else "recovered"

            async def chunks():
                payload = {
                    "choices": [{"delta": {"content": text}, "finish_reason": "stop"}]
                }
                yield "data: " + json.dumps(payload) + "\n\n"
                yield "data: [DONE]\n\n"

            return chunks_from_sse(chunks())

    provider = Provider()
    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=ToolRegistry(),
            messages=[user_message("answer")],
            retry_policy=RetryPolicy(max_retries=1, base_delay=0.001, jitter=False),
        )
    )

    assert provider.calls == 2, "空响应没有被识别为可重试错误"
    assert result.status == "completed"
    assert result.messages[-1].content == "recovered"


async def test_empty_completion_is_an_error_when_retries_are_exhausted() -> None:
    class Provider:
        def generate_stream(self, request):
            async def chunks():
                payload = {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]}
                yield "data: " + json.dumps(payload) + "\n\n"
                yield "data: [DONE]\n\n"

            return chunks_from_sse(chunks())

    result = await run_agent(
        RunOptions(provider=Provider(), tools=ToolRegistry(), messages=[user_message("hi")])
    )

    assert result.status == "error"
    assert "completed response with no content" in (result.error or "")


async def test_reasoning_only_completion_is_not_empty() -> None:
    """只有推理内容、正文为空是合法结束，不能当成空响应。"""

    class Provider:
        def generate_stream(self, request):
            async def chunks():
                payload = {
                    "choices": [
                        {
                            "delta": {"reasoning_content": "thinking", "content": ""},
                            "finish_reason": "stop",
                        }
                    ]
                }
                yield "data: " + json.dumps(payload) + "\n\n"
                yield "data: [DONE]\n\n"

            return chunks_from_sse(chunks())

    result = await run_agent(
        RunOptions(provider=Provider(), tools=ToolRegistry(), messages=[user_message("hi")])
    )

    assert result.status == "completed"


async def test_empty_response_error_becomes_error_result() -> None:
    """适配器抛出的 EMPTY_RESPONSE 仍然以错误码形式到达调用方，不被吞成空回答。"""

    class Provider:
        def generate_stream(self, request):
            return stream_from_coroutine(lambda: self.generate(request))

        async def generate(self, request):
            raise ProviderError("provider returned an empty completion", ErrorCode.EMPTY_RESPONSE)

    result = await run_agent(
        RunOptions(provider=Provider(), tools=ToolRegistry(), messages=[user_message("hi")])
    )

    assert result.status == "error"
    assert "empty completion" in (result.error or "").lower()
