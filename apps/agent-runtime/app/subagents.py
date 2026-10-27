"""管理子任务创建、会话分叉、结果等待和后续续跑。

复用运行循环、输入队列和会话日志，并为父任务预留步骤预算。
宿主在事件循环退出前关闭管理器并回收子任务。
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from .compaction import leading_system_count
from .contracts import (
    AssistantTextDelta,
    AssistantTurnEvent,
    CompactionEntry,
    Message,
    MessageEntry,
    ToolContext,
    ToolRegistryLike,
    ToolResult,
    TurnEndEntry,
    Usage,
)
from .inbox import Inbox
from .loop import RunOptions, run_agent
from .session import JsonlSessionStore, apply_compaction, repair_tool_pairing
from .tools import FunctionTool, ToolRegistry


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class _Child:
    options: RunOptions
    record: dict
    session: JsonlSessionStore
    inbox: Inbox
    cancel: asyncio.Event
    task: asyncio.Task | None = None


@dataclass
class _MessageTools:
    """Overlay control tools without assuming the host registry has a get() API."""

    base: ToolRegistryLike
    controls: ToolRegistry

    def list(self):
        names = {tool.name for tool in self.controls.list()}
        return [tool for tool in self.base.list() if tool.name not in names] + self.controls.list()

    def execution_mode(self, name):
        registry = self.controls if self.controls.get(name) else self.base
        return registry.execution_mode(name)

    async def execute(self, call, context=None):
        registry = self.controls if self.controls.get(call.name) else self.base
        return await registry.execute(call, context)


class SubagentRunner:
    def __init__(self, options_factory: Callable[[], RunOptions], directory: Path, *,
                 parent_session: JsonlSessionStore | None = None,
                 parent_inbox: Inbox | None = None, parent_reserve_steps: int = 1) -> None:
        self.options_factory = options_factory
        self.directory = directory
        self.parent_session = parent_session
        self.parent_inbox = parent_inbox
        self.parent_id = (str(parent_session.path.resolve()) if parent_session
                          else uuid.uuid4().hex)
        self.parent_reserve_steps = parent_reserve_steps
        self.records: list[dict] = []
        self._children: dict[str, _Child] = {}
        self._closed = False

    def tool(self) -> FunctionTool:
        modes = ["spawn", "fork"] if self.parent_session else ["spawn"]
        return FunctionTool(
            "subagent",
            "Delegate focused work when it benefits from a separate context. Choose spawn "
            "for a self-contained task; fork inherits only completed parent turns, excluding "
            "the current turn. Provide current requirements/file references in the prompt. "
            "Use run_in_background to continue independent work; collect with wait_agent. "
            "continuable keeps the child available for send_message follow-ups. Steps and "
            "time are shared; avoid duplicating research and return concise evidence. "
            "Do small tasks yourself. Collect needed results before your final answer.",
            {"type": "object", "properties": {
                "prompt": {"type": "string", "minLength": 1},
                "context_mode": {"type": "string", "enum": modes, "default": "spawn"},
                "run_in_background": {"type": "boolean", "default": False},
                "continuable": {"type": "boolean", "default": False},
                "max_steps": {"type": "integer", "minimum": 1,
                              "description": "Optional model-step allowance for this activation."},
            }, "required": ["prompt"], "additionalProperties": False},
            self.execute, execution_mode="parallel",
        )

    def tools(self) -> list[FunctionTool]:
        return [self.tool(), self._message_tool(), FunctionTool(
            "wait_agent", "Wait for a child result without cancelling it on wait timeout. "
            "A running response is not task success. Full history remains in its session.",
            {"type": "object", "properties": {
                "agent_id": {"type": "string", "minLength": 1},
                "timeout_seconds": {"type": "number", "minimum": 0, "maximum": 60,
                                    "default": 30},
            }, "required": ["agent_id"], "additionalProperties": False},
            self.wait, execution_mode="parallel",
        ), FunctionTool(
            "interrupt_agent", "Request cancellation of a child's current activation. "
            "A continuable child stays available; no tool is rerun to repair history.",
            {"type": "object", "properties": {"agent_id": {"type": "string"}},
             "required": ["agent_id"], "additionalProperties": False},
            self.interrupt, execution_mode="parallel",
        )]

    async def execute(self, args: dict, context: ToolContext) -> ToolResult:
        if self._closed:
            raise ValueError("subagent manager is closed")
        original = self.options_factory()
        messages = list(original.messages[:leading_system_count(original.messages)])
        mode = args.get("context_mode", "spawn")
        if mode == "fork":
            history = await self._completed_messages()
            messages.extend(history[leading_system_count(history):])
        elif mode != "spawn":
            raise ValueError(f"unknown context mode: {mode}")
        agent_id = uuid.uuid4().hex
        session = JsonlSessionStore(self.directory / agent_id / "session.jsonl")
        record = {
            "agent_id": agent_id, "parent_id": self.parent_id,
            "parent_call_id": context.tool_call_id, "context_mode": mode,
            "continuable": args.get("continuable", False), "prompt": args["prompt"],
            "status": "running", "answer": "", "started_at": _now(),
            "session": str(session.path.resolve()), "tool_calls": [],
            "usage": asdict(Usage()), "steps": 0,
        }
        child = _Child(original, record, session, Inbox(), asyncio.Event())
        self._children[agent_id] = child
        self.records.append(record)
        for message in copy.deepcopy(messages):
            await session.append(MessageEntry(0, 0, message, _now()))
        background = args.get("run_in_background", False)
        self._start(child, args["prompt"], context, background=background,
                    max_steps=args.get("max_steps"))
        if not background:
            await child.task
        return self._result(child)

    async def _completed_messages(self) -> list[Message]:
        if self.parent_session is None:
            raise ValueError("fork requires a parent session")
        events = await self.parent_session.load_events()
        boundary = next((i + 1 for i in range(len(events) - 1, -1, -1)
                         if isinstance(events[i], TurnEndEntry)), 0)
        messages: list[Message] = []
        for event in events[:boundary]:
            if isinstance(event, MessageEntry):
                messages.append(event.message)
            elif isinstance(event, CompactionEntry):
                messages = apply_compaction(messages, event.summary, event.kept_messages)
        return messages

    def _start(self, child: _Child, prompt: str, context: ToolContext, *,
               background: bool, max_steps: int | None = None) -> None:
        if self._closed:
            raise ValueError("subagent manager is closed")
        child.cancel = asyncio.Event()
        child.record.update(status="running", answer="", error=None)
        child.inbox.steer(Message("user", prompt, source={
            "kind": "agent-message", "form": "relay", "senderSessionId": self.parent_id,
        }))
        self._save(child)
        child.task = asyncio.create_task(self._drive(
            child, context, background=background, max_steps=max_steps,
        ))

    async def _drive(self, child: _Child, context: ToolContext, *,
                     background: bool, max_steps: int | None) -> None:
        record = child.record
        streamed: list[str] = []

        def observe(event):
            if child.options.on_event is not None:
                child.options.on_event(event)
            if isinstance(event, AssistantTextDelta):
                streamed.append(event.text)
            elif isinstance(event, AssistantTurnEvent):
                if event.message.content:
                    record["answer"] = event.message.content
                record["tool_calls"].extend(call.name for call in event.message.tool_calls)

        async def link_cancel():
            await context.cancel_event.wait()
            child.cancel.set()

        watcher = (asyncio.create_task(link_cancel()) if context.cancel_event else None)
        try:
            messages = repair_tool_pairing(await child.session.load_messages())
            tools = child.options.tools
            if record["continuable"]:
                tools = _MessageTools(tools, ToolRegistry().register(
                    self._message_tool(sender=record["agent_id"]),
                ))
            result = await run_agent(replace(
                child.options, messages=messages, tools=tools, session=child.session,
                inbox=child.inbox, cancel_event=child.cancel,
                clear_inbox_on_cancel=False, run_control=context.run_control,
                local_max_steps=int(max_steps) if max_steps is not None else None,
                reserved_parent_steps=self.parent_reserve_steps,
                context_anchor=await child.session.load_context_anchor(), on_event=observe,
            ))
            cumulative = Usage(**record["usage"])
            cumulative.add(result.usage)
            record.update(status=result.status, usage=asdict(cumulative),
                          steps=record["steps"] + result.steps, error=result.error)
        except asyncio.CancelledError:
            record["status"] = "aborted"
            raise
        except Exception as exc:
            record.update(status="error", error=str(exc))
        finally:
            if watcher:
                watcher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watcher
            record["answer"] = record["answer"] or "".join(streamed)
            record["ended_at"] = _now()
            self._save(child)
            if background and self.parent_inbox is not None and not self._closed:
                self.parent_inbox.steer(Message(
                    "user", f"Background subagent {record['agent_id']} ended: "
                    f"{record['status']}.\n{record['answer']}", source={
                        "kind": "subagent-settled", "form": "notice",
                        "senderSessionId": record["agent_id"],
                    },
                ))

    def _save(self, child: _Child) -> None:
        path = child.session.path.parent / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(child.record, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    def _result(self, child: _Child) -> ToolResult:
        # 模型接收结果正文和子任务标识，完整审计记录另行保存。
        keys = ("agent_id", "parent_call_id", "status", "answer", "usage", "session",
                "continuable", "error")
        result = {key: child.record.get(key) for key in keys}
        return ToolResult(json.dumps(result, ensure_ascii=False),
                          is_error=child.record["status"] not in {"completed", "running"})

    async def _lookup(self, agent_id: str) -> _Child:
        if agent_id in self._children:
            return self._children[agent_id]
        if len(agent_id) != 32 or any(c not in "0123456789abcdef" for c in agent_id):
            raise ValueError("invalid child agent id")
        directory = self.directory / agent_id
        record = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        if record["parent_id"] != self.parent_id:
            raise ValueError("agent is not a direct child of this parent session")
        child = _Child(self.options_factory(), record,
                       JsonlSessionStore(directory / "session.jsonl"), Inbox(), asyncio.Event())
        if record["status"] == "running":
            record.update(status="aborted", error="previous host ended before child settlement")
            self._save(child)
        self._children[agent_id] = child
        self.records.append(record)
        return child

    async def wait(self, args: dict, context: ToolContext) -> ToolResult:
        child = await self._lookup(args["agent_id"])
        if child.task is not None and not child.task.done():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.shield(child.task), args.get("timeout_seconds", 30))
        return self._result(child)

    def _message_tool(self, sender: str | None = None) -> FunctionTool:
        async def send(args: dict, context: ToolContext) -> ToolResult:
            if self._closed:
                raise ValueError("subagent manager is closed")
            message_id = uuid.uuid4().hex
            if sender is not None:
                if args["agent_id"] != self.parent_id or self.parent_inbox is None:
                    raise ValueError("a child may message only its live direct parent")
                self.parent_inbox.steer(Message(
                    "user", f"Agent {sender} sent a message: {args['message']}", source={
                        "kind": "agent-message", "form": "relay", "senderSessionId": sender,
                    },
                ))
            else:
                child = await self._lookup(args["agent_id"])
                if not child.record["continuable"]:
                    raise ValueError("one-shot child cannot receive follow-up messages")
                if child.task is not None and not child.task.done():
                    if "max_steps" in args:
                        raise ValueError(
                            "max_steps can be assigned only to an idle child activation",
                        )
                    child.inbox.steer(Message("user", args["message"], source={
                        "kind": "agent-message", "form": "relay",
                        "senderSessionId": self.parent_id,
                    }))
                else:
                    self._start(child, args["message"], context, background=True,
                                max_steps=args.get("max_steps"))
            return ToolResult(json.dumps({"message_id": message_id, "delivered": True}))

        return FunctionTool(
            "send_message", (f"Send a message only to your parent {self.parent_id}. " if sender
                             else "Send to a direct continuable child. Running: steer its next "
                             "step. Idle: start a new activation with its existing history. ")
            + "Returns delivery confirmation, not the recipient's answer.",
            {"type": "object", "properties": {
                "agent_id": {"type": "string", "minLength": 1},
                "message": {"type": "string", "minLength": 1},
                "max_steps": {"type": "integer", "minimum": 1,
                              "description": "New activation allowance; only for an idle child."},
            }, "required": ["agent_id", "message"], "additionalProperties": False},
            send, execution_mode="parallel",
        )

    async def interrupt(self, args: dict, context: ToolContext) -> ToolResult:
        child = await self._lookup(args["agent_id"])
        child.cancel.set()
        return ToolResult(json.dumps({"accepted": True}))

    async def close(self) -> None:
        self._closed = True
        tasks = []
        for child in self._children.values():
            if child.task is not None and not child.task.done():
                child.cancel.set()
                tasks.append(child.task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
