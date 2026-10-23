"""Inbox：运行中插话（steer）与回合结束后追问（followup）的待处理输入队列。

对齐 DSH `packages/core/agent-loop/src/inbox.ts` 与 `agent.ts`：

- `steer(message)` → `next-step`：当前 assistant turn 与其 tool calls 完成后注入，
  **不开新 turn**；
- `followup(message)` → `next-turn`：当前 turn 完全结束后开一个新 turn；
- `send` 必须显式选择目标，不猜；
- `claim(target)`：next-step 全部取走；target 为 `next-turn` 时再取一条 next-turn。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from .contracts import Message

InboxTarget = Literal["next-step", "next-turn"]

TARGETS: tuple[InboxTarget, InboxTarget] = ("next-step", "next-turn")


class Inbox:
    """进程内的待处理输入队列。由调用方在运行中写入，loop 在边界上 claim。"""

    def __init__(self) -> None:
        self._queues: dict[InboxTarget, list[Message]] = {"next-step": [], "next-turn": []}
        self._listeners: list[Callable[[str, InboxTarget, Message], None]] = []

    def subscribe(self, listener: Callable[[str, InboxTarget, Message], None],
                  ) -> Callable[[], None]:
        """DSH inbox inserted/claimed/discarded lifecycle, owned by the live driver."""
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    def _notify(self, event: str, target: InboxTarget, message: Message) -> None:
        for listener in tuple(self._listeners):
            listener(event, target, message)

    @property
    def next_step(self) -> list[Message]:
        """等待下一个 step 边界的输入（steering）。"""
        return list(self._queues["next-step"])

    @property
    def next_turn(self) -> list[Message]:
        """等待开新 turn 的输入（followup）。"""
        return list(self._queues["next-turn"])

    @property
    def has_pending(self) -> bool:
        return bool(self._queues["next-step"]) or bool(self._queues["next-turn"])

    def send(self, message: Message, target: InboxTarget) -> None:
        """写入队列；目标必须显式指定。"""
        if target not in TARGETS:
            raise ValueError(f"unknown inbox target: {target!r}")
        self._queues[target].append(message)
        self._notify("inserted", target, message)

    def steer(self, message: Message) -> None:
        """插话：在本 turn 的下一个 step 边界注入。"""
        self.send(message, "next-step")

    def followup(self, message: Message) -> None:
        """追问：当前 turn 结束后开一个新 turn。"""
        self.send(message, "next-turn")

    def prepend(self, target: InboxTarget, message: Message) -> None:
        if target not in TARGETS:
            raise ValueError(f"unknown inbox target: {target!r}")
        self._queues[target].insert(0, message)
        self._notify("inserted", target, message)

    def claim(self, target: InboxTarget) -> list[Message]:
        """取出一个 step 的输入：next-step 全部；`next-turn` 时再取一条 next-turn。"""
        if target not in TARGETS:
            raise ValueError(f"unknown inbox target: {target!r}")
        claimed = self._take("next-step")
        for message in claimed:
            self._notify("claimed", "next-step", message)
        if target == "next-turn" and self._queues["next-turn"]:
            message = self._queues["next-turn"].pop(0)
            claimed.append(message)
            self._notify("claimed", "next-turn", message)
        return claimed

    def clear(self) -> list[Message]:
        """丢弃全部待处理输入（DSH `Inbox.clear`：先 next-step 再 next-turn）。"""
        cleared: list[Message] = []
        for target in TARGETS:
            taken = self._take(target)
            cleared.extend(taken)
            for message in taken:
                self._notify("discarded", target, message)
        return cleared

    def _take(self, target: InboxTarget) -> list[Message]:
        taken = self._queues[target]
        self._queues[target] = []
        return taken
