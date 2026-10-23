from __future__ import annotations

import asyncio
import json

import pytest

from app import cli
from app.contracts import AssistantTurn, Budget, Message, ToolCall, ToolContext, ToolResult, Usage
from app.loop import RunOptions, run_agent, system_message, user_message
from app.providers.mock import MockProvider, stream_from_coroutine
from app.session import JsonlSessionStore
from app.subagents import SubagentRunner
from app.tools import FunctionTool, ToolRegistry


def final(text="done", tokens=1):
    return AssistantTurn(Message("assistant", text), "stop", Usage(total_tokens=tokens))


def call(name, arguments, *, cid="call", tokens=1):
    return AssistantTurn(Message("assistant", tool_calls=[ToolCall(cid, name, arguments)]),
                         "tool_calls", Usage(total_tokens=tokens))


def echo_tools():
    return ToolRegistry().register(FunctionTool(
        "lookup", handler=lambda a, c: ToolResult("document=doc chunk=chunk-1 page=12 fact=16MHz"),
        execution_mode="parallel",
    ))


async def test_fresh_child_context_results_usage_and_shared_remaining_budget(tmp_path):
    child = MockProvider([call("lookup", {}, tokens=30), final("16MHz [doc/chunk-1 p12]", 40)])
    runner = SubagentRunner(lambda: RunOptions(
        child, echo_tools(), [system_message("child host prefix"), user_message("NOT_INHERITED")],
    ), tmp_path / "children")
    parent = MockProvider([
        call("subagent", {"prompt": "research clock"}, cid="parent-call", tokens=10),
        final("confirmed clock", 20),
    ])
    result = await run_agent(RunOptions(
        parent, ToolRegistry().register(runner.tool()),
        [system_message("parent"), user_message("PARENT_PRIVATE_HISTORY")],
        budget=Budget(max_steps=4),
    ))
    assert result.status == "completed" and result.steps == 2 and result.total_steps == 4
    assert result.usage.total_tokens == 100
    assert "NOT_INHERITED" not in str(child.requests)
    assert "PARENT_PRIVATE_HISTORY" not in str(child.requests)
    assert child.requests[0].messages[0].content == "child host prefix"
    assert "Steps: limit=4; used=1; remaining=3" in child.requests[0].messages[-1].content
    assert "Steps: limit=4; used=3; remaining=1" in parent.requests[-1].messages[-1].content
    response = json.loads(next(m.content for m in result.messages if m.role == "tool"))
    assert response["parent_call_id"] == "parent-call"
    assert response["usage"]["total_tokens"] == 70
    assert response["answer"] == "16MHz [doc/chunk-1 p12]"
    history = await JsonlSessionStore(response["session"]).load_messages()
    assert "PARENT_PRIVATE_HISTORY" not in str(history)
    assert any(m.tool_call_id == "call" for m in history if m.role == "tool")


async def test_child_cannot_spend_the_parent_final_response_allowance(tmp_path):
    child = MockProvider([call("lookup", {}, tokens=4), final("must not run", 50)])
    runner = SubagentRunner(lambda: RunOptions(child, echo_tools(), [system_message("child")]),
                            tmp_path / "children")
    errors = []
    from app.contracts import ToolFinished

    def observe(event):
        if isinstance(event, ToolFinished):
            errors.append(event.result.is_error)

    parent = MockProvider([call("subagent", {"prompt": "research"}, tokens=3), final("no")])
    result = await run_agent(RunOptions(
        parent, ToolRegistry().register(runner.tool()), [user_message("task")],
        budget=Budget(max_steps=3), on_event=observe,
    ))
    assert result.status == "completed" and result.total_steps == 3
    assert result.messages[-1].content == "no"
    assert len(parent.requests) == 2 and len(child.requests) == 1
    assert result.usage.total_tokens == 8
    assert errors == [True]
    assert runner.records[0]["status"] == "budget_exceeded"


async def test_truncated_child_keeps_partial_output_but_is_not_success(tmp_path):
    child = MockProvider([
        AssistantTurn(Message("assistant", "partial finding", [ToolCall("lookup", "lookup", {})]),
                      "tool_calls"),
        AssistantTurn(Message("assistant", ""), "length"),
    ])
    runner = SubagentRunner(lambda: RunOptions(child, echo_tools(), []), tmp_path / "children")
    result = await runner.execute({"prompt": "research"}, ToolContext(tool_call_id="parent"))
    record = json.loads(result.content)
    assert result.is_error and record["status"] == "max_tokens"
    assert record["answer"] == "partial finding" and record["parent_call_id"] == "parent"


async def test_two_children_really_run_in_parallel_with_a_single_budget(tmp_path):
    entered = 0
    both_entered = asyncio.Event()
    children = []

    class Child:
        def generate_stream(self, request):
            return stream_from_coroutine(lambda: self.generate(request))

        async def generate(self, request):
            nonlocal entered
            entered += 1
            if entered == 2:
                both_entered.set()
            await both_entered.wait()
            return final("independent result", 5)

    def options():
        child = Child()
        children.append(child)
        return RunOptions(child, echo_tools(), [system_message("child")])

    runner = SubagentRunner(options, tmp_path / "children")
    parent = MockProvider([AssistantTurn(Message("assistant", tool_calls=[
        ToolCall("one", "subagent", {"prompt": "first"}),
        ToolCall("two", "subagent", {"prompt": "second"}),
    ]), "tool_calls", Usage(total_tokens=2)), final("combined", 3)])
    result = await asyncio.wait_for(run_agent(RunOptions(
        parent, ToolRegistry().register(runner.tool()), [user_message("task")],
        budget=Budget(max_steps=4),
    )), timeout=2)
    assert result.status == "completed" and result.total_steps == 4
    assert result.usage.total_tokens == 15 and len(children) == 2
    assert len({row["session"] for row in runner.records}) == 2
    assert [m.tool_call_id for m in result.messages if m.role == "tool"] == ["one", "two"]


@pytest.mark.parametrize("outer_cancel", [False, True])
async def test_parent_cancellation_closes_child_stream(tmp_path, outer_cancel):
    from app.contracts import BlockStart, TextDelta

    entered, closed, cancel = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class Hanging:
        async def generate_stream(self, request):
            try:
                yield BlockStart(0, "text")
                yield TextDelta(0, "partial evidence")
                entered.set()
                await asyncio.Event().wait()
            finally:
                closed.set()

    runner = SubagentRunner(lambda: RunOptions(Hanging(), echo_tools(), []), tmp_path / "children")
    parent = MockProvider([call("subagent", {"prompt": "research"})])
    task = asyncio.create_task(run_agent(RunOptions(
        parent, ToolRegistry().register(runner.tool()), [user_message("task")], cancel_event=cancel,
    )))
    await asyncio.wait_for(entered.wait(), 2)
    if outer_cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
    else:
        cancel.set()
        assert (await asyncio.wait_for(task, 2)).status == "aborted"
    assert closed.is_set()
    assert runner.records[0]["status"] == "aborted"
    assert runner.records[0]["answer"] == "partial evidence"
    assert json.loads((next((tmp_path / "children").iterdir()) / "result.json").read_text(
        encoding="utf-8",
    ))["status"] == "aborted"


async def test_cli_subagents_inherit_model_configuration_and_have_read_only_tools(
    tmp_path, monkeypatch, capsys,
):
    child = MockProvider([final("reviewed")])
    parent = MockProvider([call("subagent", {"prompt": "review files"}), final("delivered")])
    configurations = []

    def factory(**kwargs):
        configurations.append(kwargs)
        return parent if len(configurations) == 1 else child

    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", factory)
    code = await cli.run_once(cli.build_parser().parse_args([
        "task", "--base-url", "http://unused.invalid", "--workspace", str(tmp_path),
        "--subagents", "--json", "--max-steps", "3", "--no-compaction",
    ]))
    payload = json.loads(capsys.readouterr().out)
    assert code == 0 and payload["total_steps"] == 3
    assert payload["answer"] == "delivered" and payload["subagents"][0]["answer"] == "reviewed"
    assert configurations[0] == configurations[1]
    assert configurations[1]["model"] == "deepseek-flash"
    assert configurations[1]["thinking"] == "enabled"
    assert configurations[1]["reasoning_effort"] == "high"
    names = {t.name for t in child.requests[0].tools}
    assert {"read_file", "search_spec", "read_context", "search_sessions"} <= names
    assert not names.intersection({"write_file", "edit_file", "run_command", "remember_fact",
                                   "replace_fact", "forget_fact", "subagent"})
