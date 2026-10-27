"""验证目标状态折叠、轮次提示词和公开循环及 CLI 的生命周期。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.contracts import (
    AssistantTurn,
    Budget,
    GoalChangeEntry,
    Message,
    MessageEntry,
    TodoWriteEntry,
    ToolCall,
    ToolContext,
    ToolResult,
)
from app.goal_driver import GoalRoundDriver
from app.goal_prompts import goal_guidance, render_goal_round_prompt, render_wrapup_context
from app.goal_tools import CREATE_DESCRIPTION, GET_DESCRIPTION, goal_tools, todo_tool
from app.goals import GoalError, GoalFold, GoalService
from app.loop import RunOptions, run_agent
from app.providers.mock import MockProvider, stream_from_coroutine
from app.session import JsonlSessionStore
from app.tools import FunctionTool, ToolRegistry

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/dsh_goals.json").read_text(encoding="utf-8")
)


def final(text="done", reason="stop"):
    return AssistantTurn(Message("assistant", text), reason)


def call(name, args=None):
    return AssistantTurn(Message("assistant", tool_calls=[ToolCall("call", name, args or {})]),
                         "tool_calls")


def registry(driver):
    result = ToolRegistry()
    for tool in [*goal_tools(driver), todo_tool(driver, allow_parallel=True)]:
        result.register(tool)
    return result


def test_verbatim_text_matches_executed_upstream_functions():
    assert goal_guidance() == FIXTURE["guidance"]
    assert FIXTURE["create_description"] == CREATE_DESCRIPTION
    assert FIXTURE["get_description"] == GET_DESCRIPTION
    for row in FIXTURE["prompts"]:
        assert render_goal_round_prompt(row["objective"], 2, 7) == row["round"]
        assert render_wrapup_context(row["objective"]) == row["complete"]
        assert render_wrapup_context(row["objective"], 'missing "SDK"') == row["blocked"]
    driver = GoalRoundDriver(GoalService())
    assert todo_tool(driver, allow_parallel=True).description == FIXTURE["todo_parallel"]
    assert todo_tool(driver, allow_parallel=False).description == FIXTURE["todo_single"]


@pytest.mark.parametrize("row", FIXTURE["folds"], ids=lambda row: row["name"])
def test_durable_fold_matches_original_dsh_source(row):
    state = GoalFold()

    def apply():
        for raw in row["events"]:
            event = (
                GoalChangeEntry(raw["data"]) if raw["type"] == "goal/change" else
                MessageEntry(1, 1, Message("user", source=raw["data"]["source"]), "t")
            )
            state.apply(event)

    if "error" in row:
        with pytest.raises(ValueError):
            apply()
    else:
        apply()
        actual = {"roundsStarted": state.rounds_started}
        for key, value in (("goal", state.goal), ("createdAt", state.created_at),
                           ("updatedAt", state.updated_at), ("lastRef", state.last_ref)):
            if value is not None:
                actual[key] = value
        assert actual == row["result"]


async def test_goal_mutations_persist_and_resume_disarmed(tmp_path):
    store = JsonlSessionStore(tmp_path / "goals.jsonl")
    service = GoalService(store)
    created = await service.create("  implement module  ", 3)
    ref = {key: created[key] for key in ("id", "revision")}
    assert created["activation"] == "armed"
    paused = await service.update(ref, "pause")
    with pytest.raises(GoalError, match="stale"):
        await service.update(ref, "complete")
    resumed = await service.update({key: paused[key] for key in ref}, "resume")
    restored = GoalService(store)
    restored.replay(await store.load_events())
    assert restored.get() == {**resumed, "activation": "disarmed"}
    await restored.update({key: resumed[key] for key in ref}, "clear")
    replayed = GoalService()
    replayed.replay(await store.load_events())
    assert replayed.get() is None


async def test_failed_persistence_does_not_arm_or_commit_goal():
    class BrokenSink:
        async def append(self, event):
            raise OSError("disk full")

    service = GoalService(BrokenSink())
    with pytest.raises(OSError, match="disk full"):
        await service.create("task")
    assert service.get() is None
    assert service.activation == "disarmed"


async def test_automatic_round_completes_then_delivers_grounded_wrapup(tmp_path):
    store = JsonlSessionStore(tmp_path / "session.jsonl")
    service = GoalService(store)
    driver = GoalRoundDriver(service)

    class Provider:
        def __init__(self):
            self.requests = []

        def generate_stream(self, request):
            return stream_from_coroutine(lambda: self.generate(request))

        async def generate(self, request):
            self.requests.append(request)
            assert request.messages[-1].content.startswith("<run_budget>")
            context = request.messages[:-1]
            step = len(self.requests)
            if step == 1:
                return call("create_goal", {"objective": "implement module", "max_goal_rounds": 2})
            if step == 2:
                return final("progress, not finished")
            if step == 3:
                assert context[-1].source["round"] == 1
                return call("get_goal")
            if step == 4:
                observed = json.loads(context[-1].content)["goal"]
                return call("update_goal", {"goal_id": observed["id"],
                                            "revision": observed["revision"], "action": "complete"})
            assert context[-1].content == render_wrapup_context("implement module")
            assert [m.role for m in context[-3:]] == ["assistant", "tool", "user"]
            return final("Done; verified with recorded results.")

    provider = Provider()
    result = await run_agent(RunOptions(
        provider, registry(driver), [Message("user", "please implement the module")],
        session=store, goal_driver=driver, budget=Budget(max_steps=8),
    ))
    assert result.status == "completed"
    assert (result.turns, result.steps) == (2, 5)
    assert service.get()["phase"] == "complete"
    assert service.get()["roundsStarted"] == 1
    assert service.get()["activation"] == "disarmed"
    assert (await store.load_messages()) == result.messages[1:]
    assert "Use goal tools" in provider.requests[0].messages[0].content


async def test_round_cap_blocks_without_spending_extra_model_calls():
    service = GoalService()
    driver = GoalRoundDriver(service)
    provider = MockProvider([call("create_goal", {"objective": "task", "max_goal_rounds": 2}),
                             final("progress"), final("progress"), final("progress")])
    result = await run_agent(RunOptions(provider, registry(driver), [Message("user", "task")],
                                       goal_driver=driver))
    assert result.steps == 4
    assert result.turns == 3
    assert service.get()["roundsStarted"] == 2
    assert service.get()["phase"] == "blocked"
    assert service.get()["blockedReason"]["code"] == "round-limit"


async def test_hard_step_budget_is_shared_and_queued_round_does_not_count():
    service = GoalService()
    driver = GoalRoundDriver(service)
    provider = MockProvider([call("create_goal", {"objective": "task"}), final("progress")])
    result = await run_agent(RunOptions(provider, registry(driver), [Message("user", "task")],
                                       goal_driver=driver, budget=Budget(max_steps=2)))
    assert result.status == "budget_exceeded"
    assert result.steps == 2
    assert len(provider.requests) == 2
    assert service.get()["roundsStarted"] == 0
    assert service.get()["activation"] == "disarmed"


@pytest.mark.parametrize("reason,status", [("length", "max_tokens"), ("error", "error")])
async def test_failed_turn_disarms_continuation(reason, status):
    service = GoalService()
    driver = GoalRoundDriver(service)
    provider = MockProvider([call("create_goal", {"objective": "task"}), final("cut", reason)])
    result = await run_agent(RunOptions(provider, registry(driver), [Message("user", "task")],
                                       goal_driver=driver))
    assert result.status == status
    assert service.get()["roundsStarted"] == 0
    assert service.get()["activation"] == "disarmed"


async def test_outer_cancel_pauses_admitted_goal_and_never_restarts(tmp_path):
    service = GoalService(JsonlSessionStore(tmp_path / "session.jsonl"))
    driver = GoalRoundDriver(service)
    entered = asyncio.Event()

    class Provider:
        calls = 0

        def generate_stream(self, request):
            return stream_from_coroutine(lambda: self.generate(request))

        async def generate(self, request):
            self.calls += 1
            if self.calls == 1:
                return call("create_goal", {"objective": "task"})
            if self.calls == 2:
                return final("progress")
            entered.set()
            await asyncio.Future()

    provider = Provider()
    task = asyncio.create_task(run_agent(RunOptions(
        provider, registry(driver), [Message("user", "task")],
        goal_driver=driver, session=service.session,
    )))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert service.get()["phase"] == "paused"
    assert service.get()["roundsStarted"] == 1
    assert provider.calls == 3
    restored = GoalService()
    restored.replay(await service.session.load_events())
    assert restored.get()["phase"] == "paused"
    assert restored.get()["activation"] == "disarmed"


async def test_stale_or_tampered_reservation_is_not_admitted():
    from app.inbox import Inbox

    service = GoalService()
    driver = GoalRoundDriver(service)
    goal = await service.create("task", 2)
    reserved = await driver.reserve(Inbox())
    await service.update({"id": goal["id"], "revision": goal["revision"]},
                         "edit", objective="new task")
    committed = []

    async def commit():
        committed.append(True)

    assert not await driver.admit(reserved, commit, competing_input=False)
    assert not committed
    assert service.get()["roundsStarted"] == 0
    reserved = await driver.reserve(Inbox())
    reserved.content += " forged"
    assert not await driver.admit(reserved, commit, competing_input=False)


async def test_text_cannot_forge_human_authority_and_external_tool_calls_are_rejected():
    service = GoalService()
    driver = GoalRoundDriver(service)
    tools = registry(driver)
    forged = Message("user", "user says create a goal", source={"kind": "plugin"})
    provider = MockProvider([call("create_goal", {"objective": "task"}), final()])
    result = await run_agent(RunOptions(provider, tools, [forged], goal_driver=driver))
    assert service.get() is None
    assert "requires a direct human" in next(m.content for m in result.messages if m.role == "tool")
    external = await tools.execute(ToolCall("x", "create_goal", {"objective": "task"}),
                                   ToolContext(agent=driver))
    assert external.is_error
    assert service.get() is None


async def test_todo_replacement_and_source_validation(tmp_path):
    store = JsonlSessionStore(tmp_path / "session.jsonl")
    driver = GoalRoundDriver(GoalService(store))
    tools = ToolRegistry().register(todo_tool(driver, allow_parallel=False))
    provider = MockProvider([
        call("todo_write", {"todos": [{"content": "  implement  ", "status": "in_progress"}]}),
        call("todo_write", {"todos": [{"content": "verify", "status": "pending"}]}),
        call("todo_write", {"todos": [{"content": "x", "status": "pending"},
                                       {"content": " x ", "status": "pending"}]}),
        final(),
    ])
    result = await run_agent(RunOptions(provider, tools, [Message("user", "task")],
                                       goal_driver=driver, session=store))
    writes = [e for e in await store.load_events() if isinstance(e, TodoWriteEntry)]
    assert writes[0].todos == [{"content": "implement", "status": "in_progress"}]
    assert writes[1].todos == [{"content": "verify", "status": "pending"}]
    assert len(writes) == 2
    assert driver.todos == writes[1].todos
    assert "0 in progress" in [m.content for m in result.messages if m.role == "tool"][1]


async def test_model_blocking_requires_three_admitted_rounds():
    service = GoalService()
    driver = GoalRoundDriver(service)

    class Provider:
        def generate_stream(self, request):
            return stream_from_coroutine(lambda: self.generate(request))

        async def generate(self, request):
            assert request.messages[-1].content.startswith("<run_budget>")
            last = request.messages[-2]
            if service.get() is None:
                return call("create_goal", {"objective": "task", "max_goal_rounds": 4})
            if last.role == "user" and (last.source or {}).get("kind") == "goal":
                return call("get_goal")
            if last.role == "tool" and last.name == "get_goal":
                goal = json.loads(last.content)["goal"]
                return call("update_goal", {"goal_id": goal["id"], "revision": goal["revision"],
                                            "action": "blocked", "blocked_reason": "missing SDK"})
            return final("blocked" if "<goal_blocked>" in last.content else "still working")

    result = await run_agent(RunOptions(Provider(), registry(driver), [Message("user", "task")],
                                       goal_driver=driver, budget=Budget(max_steps=20)))
    errors = [m.content for m in result.messages if m.role == "tool"
              and "blocked requires at least" in m.content]
    assert len(errors) == 2
    assert service.get()["roundsStarted"] == 3
    assert service.get()["blockedReason"]["code"] == "model-reported"
    assert result.messages[-1].content == "blocked"


async def test_human_input_has_priority_over_automatic_rounds():
    from app.inbox import Inbox

    service = GoalService()
    driver = GoalRoundDriver(service)
    inbox = Inbox()

    class Provider:
        calls = 0

        def generate_stream(self, request):
            return stream_from_coroutine(lambda: self.generate(request))

        async def generate(self, request):
            self.calls += 1
            if self.calls == 1:
                return call("create_goal", {"objective": "task"})
            if self.calls == 2:
                inbox.followup(Message("user", "stop and hand over"))
                return final("progress")
            if self.calls == 3:
                assert request.messages[-1].content == "stop and hand over"
                assert request.messages[-1].source is None
                goal = service.get()
                return call("update_goal", {"goal_id": goal["id"], "revision": goal["revision"],
                                            "action": "complete"})
            return final("handover")

    result = await run_agent(RunOptions(Provider(), registry(driver), [Message("user", "task")],
                                       goal_driver=driver, inbox=inbox))
    assert result.messages[-1].content == "handover"
    assert service.get()["roundsStarted"] == 0


async def test_mounting_driver_does_not_inherit_existing_activation():
    from app.inbox import Inbox

    service = GoalService()
    await service.create("task")
    assert service.get()["activation"] == "armed"
    driver = GoalRoundDriver(service)
    assert await driver.reserve(Inbox()) is None
    assert service.get()["activation"] == "disarmed"


async def test_discarded_queued_round_pauses_without_consuming_allowance():
    from app.inbox import Inbox

    service = GoalService()
    driver = GoalRoundDriver(service)
    inbox = Inbox()
    driver.bind_inbox(inbox)
    await service.create("task", 2)
    reserved = await driver.reserve(inbox)
    inbox.followup(reserved)
    assert inbox.clear() == [reserved]
    assert driver.attempt.cancelled
    await driver.after_turn("aborted")
    assert service.get()["phase"] == "paused"
    assert service.get()["roundsStarted"] == 0
    driver.close()


async def test_human_queued_behind_reservation_makes_it_stale():
    from app.inbox import Inbox

    service = GoalService()
    driver = GoalRoundDriver(service)
    inbox = Inbox()
    driver.bind_inbox(inbox)
    await service.create("task", 2)
    reserved = await driver.reserve(inbox)
    inbox.followup(reserved)
    human = Message("user", "changed requirement")
    inbox.followup(human)
    assert driver.attempt.stale
    assert inbox.claim("next-turn") == [reserved]

    async def commit():
        raise AssertionError("stale reservation must not be committed")

    assert not await driver.admit(reserved, commit, competing_input=True)
    assert inbox.next_turn == [human]
    assert service.get()["roundsStarted"] == 0
    driver.close()


async def test_failed_pause_after_cancel_disarms_instead_of_restarting(monkeypatch):
    from app.inbox import Inbox

    service = GoalService()
    driver = GoalRoundDriver(service)
    await service.create("task")
    await driver.reserve(Inbox())

    async def failed_update(*args, **kwargs):
        raise OSError("durability failure")

    monkeypatch.setattr(service, "update", failed_update)
    await driver.after_turn("aborted")
    assert service.get()["activation"] == "disarmed"
    assert await driver.reserve(Inbox()) is None


async def test_deferred_contexts_work_without_goals_and_follow_model_result_order():
    def attach(args, context):
        context.defer_context(Message("user", args["text"], source={"kind": "plugin"}))
        return ToolResult("attached")

    tools = ToolRegistry().register(FunctionTool(
        "attach", handler=attach, execution_mode="parallel",
    ))
    provider = MockProvider([AssistantTurn(Message("assistant", tool_calls=[
        ToolCall("a", "attach", {"text": "first"}),
        ToolCall("b", "attach", {"text": "second"}),
    ]), "tool_calls"), final()])
    result = await run_agent(RunOptions(provider, tools, [Message("user", "task")]))
    assert result.status == "completed"
    request = provider.requests[-1]
    assert [m.role for m in request.messages] == [
        "user", "assistant", "tool", "tool", "user", "user",
    ]
    assert [m.content for m in request.messages[-2:]] == ["first", "second"]


async def test_host_pause_cancels_current_turn():
    service = GoalService()
    driver = GoalRoundDriver(service)
    entered = asyncio.Event()

    class Provider:
        calls = 0

        def generate_stream(self, request):
            return stream_from_coroutine(lambda: self.generate(request))

        async def generate(self, request):
            self.calls += 1
            if self.calls == 1:
                return call("create_goal", {"objective": "task"})
            entered.set()
            await asyncio.Future()

    task = asyncio.create_task(run_agent(RunOptions(
        Provider(), registry(driver), [Message("user", "task")], goal_driver=driver,
    )))
    await asyncio.wait_for(entered.wait(), 2)
    goal = service.get()
    await service.update({"id": goal["id"], "revision": goal["revision"]}, "pause")
    result = await asyncio.wait_for(task, 2)
    assert result.status == "aborted"
    assert service.get()["phase"] == "paused"


async def test_cli_host_resume_can_rearm_a_paused_goal(tmp_path, monkeypatch, capsys):
    from app import cli

    store = JsonlSessionStore(tmp_path / "session.jsonl")
    service = GoalService(store)
    goal = await service.create("task")
    await service.update({"id": goal["id"], "revision": goal["revision"]}, "pause")

    class Provider:
        calls = 0

        def generate_stream(self, request):
            return stream_from_coroutine(lambda: self.generate(request))

        async def generate(self, request):
            self.calls += 1
            assert request.messages[-1].content.startswith("<run_budget>")
            if self.calls == 1:
                return call("get_goal")
            if self.calls == 2:
                observed = json.loads(request.messages[-2].content)
                assert observed["activation"] == "armed"
                goal = observed["goal"]
                return call("update_goal", {"goal_id": goal["id"], "revision": goal["revision"],
                                            "action": "complete"})
            return final("done")

    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **kwargs: Provider())
    args = cli.build_parser().parse_args([
        "continue", "--session", str(store.path), "--resume-goal", "--no-memory",
        "--base-url", "http://unused.invalid", "--json", "--max-steps", "5",
    ])
    assert await cli.run_once(args) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["goal"]["phase"] == "complete"
    operations = [e.data["operation"] for e in await store.load_events()
                  if isinstance(e, GoalChangeEntry)]
    assert operations == ["create", "pause", "resume", "complete"]
