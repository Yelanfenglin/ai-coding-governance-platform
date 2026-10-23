from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app import cli
from app.contracts import (
    AssistantTurn,
    BlockStart,
    Budget,
    CompactionEntry,
    Message,
    MessageEntry,
    RunControl,
    TextDelta,
    ToolCall,
    ToolContext,
    ToolResult,
    TurnEndEntry,
    Usage,
)
from app.inbox import Inbox
from app.loop import RunOptions, run_agent, system_message, user_message
from app.providers.mock import MockProvider, chunks_for_turn
from app.session import JsonlSessionStore
from app.subagents import SubagentRunner
from app.tools import FunctionTool, ToolRegistry


def final(text, tokens=1):
    return AssistantTurn(Message("assistant", text), "stop", Usage(total_tokens=tokens))


def call(name, arguments, tokens=1):
    return AssistantTurn(Message("assistant", tool_calls=[ToolCall("call", name, arguments)]),
                         "tool_calls", Usage(total_tokens=tokens))


def options(provider):
    return RunOptions(provider, ToolRegistry().register(FunctionTool(
        "lookup", handler=lambda a, c: ToolResult("evidence"), execution_mode="parallel",
    )), [system_message("child host")])


async def append_messages(session, *messages):
    for message in messages:
        await session.append(MessageEntry(0, 0, message, "now"))


async def test_fork_copies_only_completed_context_and_keeps_tool_pairs(tmp_path):
    session = JsonlSessionStore(tmp_path / "parent.jsonl")
    await append_messages(session, system_message("parent host"), user_message("old question"),
                          call("lookup", {}).message,
                          Message("tool", "old evidence", tool_call_id="call", name="lookup"),
                          Message("assistant", "old conclusion"))
    await session.append(TurnEndEntry(1, "completed", "now"))
    await append_messages(session, user_message("IN_FLIGHT_USER"),
                          Message("assistant", tool_calls=[ToolCall("open", "subagent", {})]))
    providers = []

    def factory():
        provider = MockProvider([final("checked")])
        providers.append(provider)
        return options(provider)

    runner = SubagentRunner(factory, tmp_path / "children", parent_session=session)
    await runner.execute({"prompt": "current task", "context_mode": "fork"}, ToolContext())
    await runner.execute({"prompt": "independent", "context_mode": "spawn"}, ToolContext())
    fork = providers[0].requests[0].messages
    assert fork[0].content == "child host" and fork[-1].content == "current task"
    assert any(m.content == "old evidence" and m.tool_call_id == "call" for m in fork)
    assert not any("IN_FLIGHT" in m.content or any(c.id == "open" for c in m.tool_calls)
                   for m in fork)
    assert not any("old" in m.content for m in providers[1].requests[0].messages)
    record = runner.records[0]
    replay = await JsonlSessionStore(record["session"]).load_messages()
    assert any(m.content == "old conclusion" for m in replay)


async def test_fork_before_first_completed_turn_starts_fresh(tmp_path):
    session = JsonlSessionStore(tmp_path / "parent.jsonl")
    await append_messages(session, system_message("parent"), user_message("not yet completed"))
    child = MockProvider([final("done")])
    runner = SubagentRunner(lambda: options(child), tmp_path / "children", parent_session=session)
    await runner.execute({"prompt": "new", "context_mode": "fork"}, ToolContext())
    assert [m.content for m in child.requests[0].messages] == ["child host", "new"]


async def test_fork_replays_compaction_instead_of_reinjecting_raw_history(tmp_path):
    session = JsonlSessionStore(tmp_path / "parent.jsonl")
    await append_messages(session, system_message("parent"), user_message("RAW_OLD"),
                          Message("assistant", "kept evidence"))
    await session.append(CompactionEntry(1, 1, "verified constraints", None, "now", 1))
    await session.append(TurnEndEntry(1, "completed", "now"))
    child = MockProvider([final("done")])
    runner = SubagentRunner(lambda: options(child), tmp_path / "children", parent_session=session)
    await runner.execute({"prompt": "review", "context_mode": "fork"}, ToolContext())
    messages = child.requests[0].messages
    assert "RAW_OLD" not in str(messages)
    assert any("compacted-summary" in m.content and m.role == "user" for m in messages)
    assert any(m.content == "kept evidence" for m in messages)


async def test_background_returns_before_child_finishes_and_wait_timeout_keeps_it_alive(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()

    class Gated:
        async def generate_stream(self, request):
            entered.set()
            await release.wait()
            for chunk in chunks_for_turn(final("finding", 7)):
                yield chunk

    inbox = Inbox()
    runner = SubagentRunner(lambda: options(Gated()), tmp_path / "children", parent_inbox=inbox)
    control = RunControl(budget=Budget(max_steps=5))
    context = ToolContext(run_control=control)
    response = json.loads((await runner.execute({
        "prompt": "research", "run_in_background": True,
    }, context)).content)
    assert response["status"] == "running"
    await asyncio.wait_for(entered.wait(), 2)
    waiting = await runner.wait({"agent_id": response["agent_id"], "timeout_seconds": 0}, context)
    assert json.loads(waiting.content)["status"] == "running"
    release.set()
    finished = await runner.wait({"agent_id": response["agent_id"]}, context)
    assert json.loads(finished.content)["answer"] == "finding"
    assert control.usage.total_tokens == 7
    assert inbox.next_step[0].source["kind"] == "subagent-settled"
    assert "finding" in inbox.next_step[0].content
    assert "prompt" not in json.loads(finished.content)  # full audit stays on disk
    await runner.close()


async def test_idle_followup_reuses_session_and_accounts_only_new_work(tmp_path):
    child = MockProvider([final("first finding", 10), final("second finding", 20)])
    control = RunControl(budget=Budget(max_steps=8))
    context = ToolContext(run_control=control)
    runner = SubagentRunner(lambda: options(child), tmp_path / "children")
    first = json.loads((await runner.execute({"prompt": "first", "continuable": True},
                                             context)).content)
    registry = ToolRegistry()
    for tool in runner.tools():
        registry.register(tool)
    sent = await registry.execute(ToolCall("send", "send_message", {
        "agent_id": first["agent_id"], "message": "second",
    }), context)
    assert json.loads(sent.content)["delivered"] and "answer" not in sent.content
    last = json.loads((await runner.wait({"agent_id": first["agent_id"]}, context)).content)
    assert last["answer"] == "second finding" and last["usage"]["total_tokens"] == 30
    assert control.usage.total_tokens == 30 and len(child.requests) == 2
    assert any(m.content == "first finding" for m in child.requests[1].messages)
    assert last["session"] == first["session"]
    await runner.close()


async def test_continuable_child_reopens_by_id_without_repeating_previous_work(tmp_path):
    parent = JsonlSessionStore(tmp_path / "parent.jsonl")
    runner = SubagentRunner(lambda: options(MockProvider([final("old", 11)])),
                            tmp_path / "children", parent_session=parent)
    first = json.loads((await runner.execute({"prompt": "first", "continuable": True},
                                             ToolContext())).content)
    await runner.close()
    child = MockProvider([final("continued", 13)])
    resumed = SubagentRunner(lambda: options(child), tmp_path / "children", parent_session=parent)
    control = RunControl(budget=Budget(max_steps=5))
    registry = ToolRegistry()
    for tool in resumed.tools():
        registry.register(tool)
    await registry.execute(ToolCall("send", "send_message", {
        "agent_id": first["agent_id"], "message": "continue",
    }), ToolContext(run_control=control))
    last = json.loads((await resumed.wait({"agent_id": first["agent_id"]}, ToolContext())).content)
    assert any(m.content == "old" for m in child.requests[0].messages)
    assert len(child.requests) == 1 and control.usage.total_tokens == 13
    assert last["usage"]["total_tokens"] == 24
    other = SubagentRunner(lambda: options(child), tmp_path / "children",
                           parent_session=JsonlSessionStore(tmp_path / "other.jsonl"))
    with pytest.raises(ValueError, match="not a direct child"):
        await other.wait({"agent_id": first["agent_id"]}, ToolContext())
    await resumed.close()


async def test_one_shot_followup_is_rejected_without_a_new_model_call(tmp_path):
    child = MockProvider([final("done")])
    runner = SubagentRunner(lambda: options(child), tmp_path / "children")
    first = json.loads((await runner.execute({"prompt": "first"}, ToolContext())).content)
    registry = ToolRegistry()
    for tool in runner.tools():
        registry.register(tool)
    response = await registry.execute(ToolCall("send", "send_message", {
        "agent_id": first["agent_id"], "message": "again",
    }))
    assert response.is_error and "one-shot" in response.content and len(child.requests) == 1


async def test_running_child_receives_correction_at_next_step(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()

    class Gated(MockProvider):
        async def generate_stream(self, request):
            turn = self._take(request)
            if len(self.requests) == 1:
                entered.set()
                await release.wait()
            for chunk in chunks_for_turn(turn):
                yield chunk

    child = Gated([call("lookup", {}), final("corrected")])
    runner = SubagentRunner(lambda: options(child), tmp_path / "children")
    first = json.loads((await runner.execute({
        "prompt": "research", "continuable": True, "run_in_background": True,
    }, ToolContext())).content)
    await asyncio.wait_for(entered.wait(), 2)
    registry = ToolRegistry()
    for tool in runner.tools():
        registry.register(tool)
    rejected = await registry.execute(ToolCall("resize", "send_message", {
        "agent_id": first["agent_id"], "message": "resize", "max_steps": 2,
    }))
    assert rejected.is_error and "idle child" in rejected.content
    await registry.execute(ToolCall("send", "send_message", {
        "agent_id": first["agent_id"], "message": "use 16MHz instead",
    }))
    release.set()
    await runner.wait({"agent_id": first["agent_id"]}, ToolContext())
    correction = next(m for m in child.requests[1].messages if m.content == "use 16MHz instead")
    assert correction.source["kind"] == "agent-message"
    await runner.close()


async def test_child_messages_parent_with_agent_attribution(tmp_path):
    inbox = Inbox()
    child = None
    runner = SubagentRunner(lambda: options(child), tmp_path / "children", parent_inbox=inbox)
    child = MockProvider([call("send_message", {
        "agent_id": runner.parent_id, "message": "pin conflict found",
    }), final("report")])
    response = await runner.execute({"prompt": "check", "continuable": True}, ToolContext())
    record = json.loads(response.content)
    assert not response.is_error
    assert inbox.next_step[0].source == {
        "kind": "agent-message", "form": "relay", "senderSessionId": record["agent_id"],
    }
    assert "pin conflict" in inbox.next_step[0].content


async def test_interrupt_cancels_only_child_and_host_close_collects_tasks(tmp_path):
    entered, closed, parent_cancel = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class Hanging:
        async def generate_stream(self, request):
            try:
                yield BlockStart(0, "text")
                yield TextDelta(0, "partial")
                entered.set()
                await asyncio.Event().wait()
            finally:
                closed.set()

    runner = SubagentRunner(lambda: options(Hanging()), tmp_path / "children")
    context = ToolContext(cancel_event=parent_cancel)
    first = json.loads((await runner.execute({
        "prompt": "check", "continuable": True, "run_in_background": True,
    }, context)).content)
    await asyncio.wait_for(entered.wait(), 2)
    await runner.interrupt({"agent_id": first["agent_id"]}, context)
    result = await asyncio.wait_for(runner.wait({"agent_id": first["agent_id"]}, context), 2)
    assert result.is_error and json.loads(result.content)["status"] == "aborted"
    assert closed.is_set() and not parent_cancel.is_set()
    await runner.close()


async def test_model_child_allowance_stops_work_without_spending_parent_budget(tmp_path):
    child = MockProvider([call("lookup", {}, 4), final("must not run", 50)])
    runner = SubagentRunner(lambda: options(child), tmp_path / "children")
    parent = MockProvider([call("subagent", {"prompt": "research", "max_steps": 1}, 3),
                           final("unfinished: more investigation needed", 2)])
    result = await run_agent(RunOptions(
        parent, ToolRegistry().register(runner.tool()), [user_message("task")],
        budget=Budget(max_steps=10),
    ))
    assert result.status == "completed" and result.total_steps == 3
    assert len(child.requests) == 1 and result.usage.total_tokens == 9
    assert runner.records[0]["status"] == "budget_exceeded"
    assert "allowance exhausted" in runner.records[0]["error"]


async def test_cold_running_record_is_not_reported_as_a_live_background_process(tmp_path):
    parent = JsonlSessionStore(tmp_path / "parent.jsonl")
    runner = SubagentRunner(lambda: options(MockProvider([final("old")])),
                            tmp_path / "children", parent_session=parent)
    first = json.loads((await runner.execute({"prompt": "task", "continuable": True},
                                             ToolContext())).content)
    path = Path(first["session"]).parent / "result.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["status"] = "running"
    path.write_text(json.dumps(record), encoding="utf-8")
    reopened = SubagentRunner(lambda: options(MockProvider([])), tmp_path / "children",
                              parent_session=parent)
    response = await reopened.wait({"agent_id": first["agent_id"]}, ToolContext())
    assert response.is_error and json.loads(response.content)["status"] == "aborted"


async def test_queued_messages_keep_order_across_interrupt_and_resume(tmp_path):
    entered = asyncio.Event()

    class Interrupted:
        def __init__(self):
            self.requests = []

        async def generate_stream(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                entered.set()
                yield BlockStart(0, "text")
                yield TextDelta(0, "partial")
                await asyncio.Event().wait()
            else:
                for chunk in chunks_for_turn(final("updated")):
                    yield chunk

    child = Interrupted()
    runner = SubagentRunner(lambda: options(child), tmp_path / "children")
    first = json.loads((await runner.execute({
        "prompt": "initial", "run_in_background": True, "continuable": True,
    }, ToolContext())).content)
    await asyncio.wait_for(entered.wait(), 2)
    registry = ToolRegistry()
    for tool in runner.tools():
        registry.register(tool)
    await registry.execute(ToolCall("a", "send_message", {
        "agent_id": first["agent_id"], "message": "older correction",
    }))
    await runner.interrupt({"agent_id": first["agent_id"]}, ToolContext())
    await runner.wait({"agent_id": first["agent_id"]}, ToolContext())
    await registry.execute(ToolCall("b", "send_message", {
        "agent_id": first["agent_id"], "message": "newer correction",
    }))
    await runner.wait({"agent_id": first["agent_id"]}, ToolContext())
    texts = [m.content for m in child.requests[1].messages]
    assert texts.index("older correction") < texts.index("newer correction")
    await runner.close()


async def test_cold_resume_repairs_interrupted_tool_pair_without_reexecuting_it(tmp_path):
    parent = JsonlSessionStore(tmp_path / "parent.jsonl")
    runner = SubagentRunner(lambda: options(MockProvider([final("old")])),
                            tmp_path / "children", parent_session=parent)
    first = json.loads((await runner.execute({"prompt": "first", "continuable": True},
                                             ToolContext())).content)
    await append_messages(JsonlSessionStore(first["session"]), call("lookup", {}).message)
    await runner.close()
    child = MockProvider([final("resumed")])
    resumed = SubagentRunner(lambda: options(child), tmp_path / "children", parent_session=parent)
    registry = ToolRegistry()
    for tool in resumed.tools():
        registry.register(tool)
    await registry.execute(ToolCall("send", "send_message", {
        "agent_id": first["agent_id"], "message": "continue",
    }))
    await resumed.wait({"agent_id": first["agent_id"]}, ToolContext())
    messages = child.requests[0].messages
    index = next(i for i, m in enumerate(messages) if m.tool_calls)
    assert messages[index + 1].role == "tool" and messages[index + 1].tool_call_id == "call"
    assert "interrupted" in messages[index + 1].content
    assert resumed.records[0]["tool_calls"] == []  # no lookup rerun
    await resumed.close()


async def test_cli_collects_background_children_before_returning(tmp_path, monkeypatch, capsys):
    entered, closed = asyncio.Event(), asyncio.Event()

    class Hanging:
        async def generate_stream(self, request):
            try:
                entered.set()
                yield BlockStart(0, "text")
                yield TextDelta(0, "partial")
                await asyncio.Event().wait()
            finally:
                closed.set()

    class Parent(MockProvider):
        async def generate_stream(self, request):
            if self.requests:
                await entered.wait()
            for chunk in chunks_for_turn(self._take(request)):
                yield chunk

    parent = Parent([call("subagent", {"prompt": "research", "run_in_background": True}),
                     final("ending now")])
    providers = iter([parent, Hanging()])
    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **kw: next(providers))
    code = await asyncio.wait_for(cli.run_once(cli.build_parser().parse_args([
        "Start research, then stop without waiting", "--base-url", "http://unused.invalid",
        "--subagents", "--workspace", str(tmp_path), "--json", "--no-compaction",
        "--max-steps", "4",
    ])), 2)
    response = json.loads(capsys.readouterr().out)
    assert code == 0 and closed.is_set()
    assert response["subagents"][0]["status"] == "aborted"
    assert response["answer"] == "ending now"  # child's stream does not become parent answer


def test_explicit_parent_session_continues_child_in_a_new_python_process(tmp_path):
    harness = tmp_path / "driver.py"
    harness.write_text('''
import asyncio, json, sys
from pathlib import Path
from app import cli
from app.contracts import AssistantTurn, Message, ToolCall, Usage
from app.providers.mock import MockProvider
root, phase = Path(sys.argv[1]), sys.argv[2]
root.mkdir(exist_ok=True)
def final(text):
    return AssistantTurn(Message("assistant", text), "stop", Usage(total_tokens=10))
def call(name, args, cid):
    return AssistantTurn(Message("assistant", tool_calls=[ToolCall(cid, name, args)]),
                         "tool_calls", Usage(total_tokens=10))
if phase == "first":
    parent = MockProvider([call("subagent", {"prompt":"research", "continuable":True}, "new"),
                           final("recorded")])
    child = MockProvider([final("verified clock 16MHz")])
else:
    previous = json.loads((root.parent / "first.json").read_text(encoding="utf-8"))
    aid = previous["subagents"][0]["agent_id"]
    parent = MockProvider([call("send_message", {"agent_id":aid,"message":"continue"}, "send"),
                           call("wait_agent", {"agent_id":aid}, "wait"), final("continued")])
    child = MockProvider([final("still 16MHz")])
providers = iter([parent, child])
cli.OpenAiCompatibleProvider = lambda **kw: next(providers)
code = asyncio.run(cli.run_once(cli.build_parser().parse_args([
    "Research" if phase == "first" else "Continue previous research", "--workspace", str(root),
    "--session", str(root.parent / "parent.jsonl"), "--subagents", "--no-compaction", "--json",
    "--base-url", "http://unused.invalid", "--model", "deepseek-flash", "--max-steps", "6",
])))
if phase == "second":
    assert any(m.content == "verified clock 16MHz" for m in child.requests[0].messages)
    assert len(child.requests) == 1
raise SystemExit(code)
''', encoding="utf-8")
    results = []
    root = Path(__file__).resolve().parents[1]
    for phase in ("first", "second"):
        completed = subprocess.run(
            [sys.executable, "-B", str(harness), str(tmp_path / "work"), phase],
            cwd=root, env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONIOENCODING": "utf-8",
                           "PYTHONPATH": str(root)},
            capture_output=True, text=True, encoding="utf-8", timeout=20,
        )
        assert completed.returncode == 0, completed.stderr
        (tmp_path / f"{phase}.json").write_text(completed.stdout, encoding="utf-8")
        results.append(json.loads(completed.stdout))
    assert results[0]["subagents"][0]["agent_id"] == results[1]["subagents"][0]["agent_id"]
    assert results[0]["usage"]["total_tokens"] == 30
    assert results[1]["usage"]["total_tokens"] == 40  # old child work not billed again
