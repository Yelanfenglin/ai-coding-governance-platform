"""append-only 的 JSONL 事件日志。

存的是事件（turn/start、step/start、message、step/end、turn/end），不是裸消息列表，
这样重新加载时既能重建 messages，也能看出 turn/step 边界和最后一个 turn 是否中断。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .compaction import frame_checkpoint, leading_system_count
from .contracts import (
    AgentStatus,
    CompactionEntry,
    ContextUsageAnchor,
    GoalChangeEntry,
    Message,
    MessageEntry,
    SessionEvent,
    StepEndEntry,
    StepStartEntry,
    TodoWriteEntry,
    ToolCall,
    TurnEndEntry,
    TurnStartEntry,
    Usage,
)


@dataclass(slots=True)
class SessionTimeline:
    turns: int
    steps: int
    messages: list[Message]
    status: AgentStatus | None
    last_turn_open: bool


@dataclass(slots=True, frozen=True)
class SessionNode:
    """带树形结构的会话节点（Phase E：session tree）。

    append-only，不删除历史节点；`parent_id` 指向上一个节点，当前 leaf 是活动分支。
    """

    id: str
    parent_id: str | None
    event: SessionEvent


class JsonlSessionStore:
    """SessionSink 实现：每行一个节点（`id` / `parent_id` / `event`）。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._leaf: str | None = None

    @property
    def leaf_id(self) -> str | None:
        return self._leaf

    async def append(self, event: SessionEvent) -> str:
        """追加一个节点并返回它的 id。

        重新打开已有文件时先恢复末端节点，避免新 run 另起一个根（树断链）。
        """
        if self._leaf is None:
            await self._restore_leaf()
        node = SessionNode(id=uuid.uuid4().hex, parent_id=self._leaf, event=event)
        line = json.dumps(node_to_dict(node), ensure_ascii=False)
        await asyncio.to_thread(self._write_line, line)
        self._leaf = node.id
        return node.id

    async def load_nodes(self) -> list[SessionNode]:
        text = await asyncio.to_thread(self._read_text)
        nodes = [node_from_dict(json.loads(line)) for line in text.splitlines() if line.strip()]
        if nodes:
            self._leaf = nodes[-1].id
        return nodes

    async def _restore_leaf(self) -> None:
        """把 `_leaf` 恢复成文件里最后一个节点的 id。"""
        text = await asyncio.to_thread(self._read_text)
        for line in reversed(text.splitlines()):
            if not line.strip():
                continue
            try:
                node = json.loads(line)
            except ValueError:
                break
            if isinstance(node, dict) and isinstance(node.get("id"), str):
                self._leaf = node["id"]
            break

    async def ancestors(self, node_id: str) -> list[SessionNode]:
        """从根到该节点的路径；用于回溯/回放。"""
        nodes = await self.load_nodes()
        by_id = {node.id: node for node in nodes}
        chain: list[SessionNode] = []
        current = by_id.get(node_id)
        while current is not None:
            chain.append(current)
            current = by_id.get(current.parent_id) if current.parent_id is not None else None
        chain.reverse()
        return chain

    async def load_events(self) -> list[SessionEvent]:
        return [node.event for node in await self.load_nodes()]

    async def load_messages(self) -> list[Message]:
        """重放出**当前活动上下文**：压缩之后 = 摘要 + 保留区原文。

        原始事件不删除；要读未经压缩的完整历史用 `load_raw_messages()`。
        """
        messages: list[Message] = []
        for event in await self.load_events():
            if isinstance(event, MessageEntry):
                messages.append(event.message)
            elif isinstance(event, CompactionEntry):
                messages = apply_compaction(messages, event.summary, event.kept_messages)
        return messages

    async def load_raw_messages(self) -> list[Message]:
        """未经压缩处理的全部历史消息（审计 / 调试用）。"""
        return [
            event.message for event in await self.load_events() if isinstance(event, MessageEntry)
        ]

    async def load_context_anchor(self) -> ContextUsageAnchor | None:
        """续接采用最近成功请求的计量基准；旧日志没有它则回到完整请求估算。"""
        for event in reversed(await self.load_events()):
            if isinstance(event, StepEndEntry) and event.context_anchor is not None:
                return event.context_anchor
        return None

    async def load_timeline(self) -> SessionTimeline:
        events = await self.load_events()
        turns = sum(1 for event in events if isinstance(event, TurnStartEntry))
        steps = sum(1 for event in events if isinstance(event, StepStartEntry))
        messages = await self.load_messages()
        status = next(
            (
                event.status
                for event in reversed(events)
                if isinstance(event, TurnEndEntry)
            ),
            None,
        )
        ends = sum(1 for event in events if isinstance(event, TurnEndEntry))
        return SessionTimeline(
            turns=turns,
            steps=steps,
            messages=messages,
            status=status,
            last_turn_open=turns > ends,
        )

    def _write_line(self, line: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")

    def _read_text(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""


def apply_compaction(
    messages: list[Message], summary: str, kept_messages: int
) -> list[Message]:
    """按压缩记录重建上下文：原 system 前缀、user 摘要检查点和保留的原文。

    保留区外的旧检查点随区域滚动合并，保留区内的检查点原样留存。
    """
    prefix_end = leading_system_count(messages)
    fixed = messages[:prefix_end]
    body = messages[prefix_end:]
    kept = body[len(body) - kept_messages :] if kept_messages > 0 else []
    return [
        *fixed,
        Message(role="user", content=frame_checkpoint(summary)),
        *kept,
    ]


# --- 序列化 -------------------------------------------------------------------


def event_to_dict(event: SessionEvent) -> dict[str, Any]:
    if isinstance(event, GoalChangeEntry):
        return {"type": "goal/change", "data": event.data, "timestamp": event.timestamp}
    if isinstance(event, TodoWriteEntry):
        return {"type": "todo/write", "todos": event.todos, "timestamp": event.timestamp}
    if isinstance(event, TurnStartEntry):
        return {"type": "turn/start", "turn": event.turn, "timestamp": event.timestamp}
    if isinstance(event, StepStartEntry):
        return {
            "type": "step/start",
            "turn": event.turn,
            "step": event.step,
            "timestamp": event.timestamp,
        }
    if isinstance(event, MessageEntry):
        return {
            "type": "message",
            "turn": event.turn,
            "step": event.step,
            "message": message_to_dict(event.message),
            "timestamp": event.timestamp,
        }
    if isinstance(event, StepEndEntry):
        payload = {
            "type": "step/end",
            "turn": event.turn,
            "step": event.step,
            "timestamp": event.timestamp,
        }
        if event.usage is not None:
            payload["usage"] = usage_to_dict(event.usage)
        if event.context_anchor is not None:
            payload["context_anchor"] = asdict(event.context_anchor)
        return payload
    if isinstance(event, CompactionEntry):
        return {
            "type": "compaction",
            "turn": event.turn,
            "step": event.step,
            "summary": event.summary,
            "first_kept_entry_id": event.first_kept_entry_id,
            "timestamp": event.timestamp,
            "kept_messages": event.kept_messages,
        }
    if isinstance(event, TurnEndEntry):
        return {
            "type": "turn/end",
            "turn": event.turn,
            "status": event.status,
            "timestamp": event.timestamp,
        }
    raise TypeError(f"unsupported session event: {event!r}")


def event_from_dict(raw: dict[str, Any]) -> SessionEvent:
    event_type = raw.get("type")
    timestamp = str(raw.get("timestamp", ""))
    if event_type == "goal/change":
        return GoalChangeEntry(raw["data"], timestamp)
    if event_type == "todo/write":
        return TodoWriteEntry(raw["todos"], timestamp)
    if event_type == "turn/start":
        return TurnStartEntry(turn=int(raw["turn"]), timestamp=timestamp)
    if event_type == "step/start":
        return StepStartEntry(turn=int(raw["turn"]), step=int(raw["step"]), timestamp=timestamp)
    if event_type == "message":
        return MessageEntry(
            turn=int(raw["turn"]),
            step=int(raw["step"]),
            message=message_from_dict(raw["message"]),
            timestamp=timestamp,
        )
    if event_type == "step/end":
        return StepEndEntry(
            turn=int(raw["turn"]),
            step=int(raw["step"]),
            timestamp=timestamp,
            usage=usage_from_dict(raw["usage"]) if raw.get("usage") is not None else None,
            context_anchor=(
                ContextUsageAnchor(**raw["context_anchor"])
                if raw.get("context_anchor") is not None else None
            ),
        )
    if event_type == "compaction":
        return CompactionEntry(
            turn=int(raw["turn"]),
            step=int(raw["step"]),
            summary=raw["summary"],
            first_kept_entry_id=raw.get("first_kept_entry_id"),
            timestamp=timestamp,
            kept_messages=int(raw.get("kept_messages", 0)),
        )
    if event_type == "turn/end":
        return TurnEndEntry(
            turn=int(raw["turn"]), status=raw["status"], timestamp=timestamp
        )
    raise ValueError(f"unknown session event type: {event_type!r}")


def message_to_dict(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.reasoning_content:
        payload["reasoning_content"] = message.reasoning_content
    if message.tool_calls:
        payload["tool_calls"] = [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in message.tool_calls
        ]
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.name is not None:
        payload["name"] = message.name
    if message.interrupted:
        payload["interrupted"] = True
    if message.source is not None:
        payload["source"] = message.source
    return payload


def node_to_dict(node: SessionNode) -> dict[str, Any]:
    return {"id": node.id, "parent_id": node.parent_id, "event": event_to_dict(node.event)}


def node_from_dict(raw: dict[str, Any]) -> SessionNode:
    return SessionNode(
        id=raw["id"], parent_id=raw.get("parent_id"), event=event_from_dict(raw["event"])
    )


def repair_tool_pairing(
    messages: list[Message],
    interrupted_result: str = "Error: tool call was interrupted before it returned a result",
) -> list[Message]:
    """让历史满足「每个 tool call 都有 result、每个 result 都有 call」。

    中断（外层取消、进程退出、旧版本留下的会话）会留下只有调用的 assistant 消息。
    这种历史直接送下一轮，多数 provider 会拒绝整个请求；续接前先补齐：
    缺的结果补一条中断说明，配不上任何调用的结果丢弃（它无法被解释）。
    只修上下文，不重跑工具。
    """
    repaired: list[Message] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        index += 1
        if message.role == "tool":
            # 前面没有带它的 assistant：孤立结果，丢掉。
            continue
        repaired.append(message)
        if message.role != "assistant" or not message.tool_calls:
            continue
        expected = [call.id for call in message.tool_calls]
        produced: set[str] = set()
        while index < len(messages) and messages[index].role == "tool":
            result = messages[index]
            index += 1
            if result.tool_call_id in expected:
                produced.add(result.tool_call_id)
                repaired.append(result)
        repaired.extend(
            Message(
                role="tool",
                content=interrupted_result,
                tool_call_id=call.id,
                name=call.name,
            )
            for call in message.tool_calls
            if call.id not in produced
        )
    return repaired


def usage_to_dict(usage: Usage) -> dict[str, Any]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }


def usage_from_dict(raw: dict[str, Any]) -> Usage:
    return Usage(
        input_tokens=int(raw.get("input_tokens", 0)),
        output_tokens=int(raw.get("output_tokens", 0)),
        total_tokens=int(raw.get("total_tokens", 0)),
    )


def message_from_dict(raw: dict[str, Any]) -> Message:
    return Message(
        role=raw["role"],
        content=raw.get("content", ""),
        tool_calls=[
            ToolCall(
                id=item["id"], name=item["name"], arguments=item.get("arguments", {})
            )
            for item in raw.get("tool_calls", [])
        ],
        tool_call_id=raw.get("tool_call_id"),
        name=raw.get("name"),
        interrupted=bool(raw.get("interrupted", False)),
        source=raw.get("source"),
        reasoning_content=raw.get("reasoning_content", ""),
    )
