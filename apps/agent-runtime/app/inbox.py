"""管理运行中插话与回合结束后的追问队列。

steer 写入 next-step，在当前模型响应及工具调用完成后注入；
followup 写入 next-turn，在当前回合结束后启动新回合。
认领输入时取出全部 next-step 消息，并按目标决定是否追加一条 next-turn 消息。
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
        """订阅由运行驱动器发出的输入插入、认领和丢弃事件。"""
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
        """按 next-step、next-turn 的顺序清空待处理输入。"""
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
