"""Trajectory：把一次 run 的事件流压成可断言、可复跑的轨迹摘要（Phase G）。

评测不只看最终答案：还看工具调用次数与顺序、并行度、retry 次数、压缩次数、
取消与终止原因、usage 累加。这里只做**记录与汇总**，不做评分。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .contracts import (
    AgentEvent,
    AgentStatus,
    AssistantTurnEvent,
    ContextCompacted,
    MemoryRecalled,
    ProviderRetryScheduled,
    StepStarted,
    ToolFinished,
    ToolStarted,
    TurnFinished,
    TurnStarted,
    Usage,
)
from .loop import AgentRunResult


@dataclass(slots=True)
class ToolCallRecord:
    name: str
    call_id: str
    is_error: bool = False
    concludes_turn: bool = False
    aborted_before_dispatch: bool = False


@dataclass(slots=True)
class Trajectory:
    status: AgentStatus | None = None
    turns: int = 0
    steps: int = 0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    retries: int = 0
    compactions: int = 0
    memory_recalls: int = 0
    interrupted_assistant_messages: int = 0
    errors: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    event_types: list[str] = field(default_factory=list)

    @property
    def tool_names(self) -> list[str]:
        return [record.name for record in self.tool_calls]

    def finish(self, result: AgentRunResult) -> Trajectory:
        """用 `AgentRunResult` 补齐 usage / 步数 / 终止原因，便于一次性断言。"""
        self.status = result.status
        self.usage = result.usage
        self.turns = result.turns
        self.steps = result.steps
        if result.error:
            self.errors.append(result.error)
        return self

    def summary(self) -> dict[str, object]:
        """可打印、可写入报告的一行摘要。"""
        return {
            "status": self.status,
            "turns": self.turns,
            "steps": self.steps,
            "tool_calls": self.tool_names,
            "retries": self.retries,
            "compactions": self.compactions,
            "memory_recalls": self.memory_recalls,
            "errors": self.errors,
            "total_tokens": self.usage.total_tokens,
        }


def trajectory_recorder() -> tuple[Trajectory, Callable[[AgentEvent], None]]:
    """返回 `(Trajectory, on_event)`：把回调接到 `RunOptions.on_event` 即可。"""
    trajectory = Trajectory()

    def on_event(event: AgentEvent) -> None:
        trajectory.event_types.append(type(event).__name__)
        if isinstance(event, TurnStarted):
            trajectory.turns = max(trajectory.turns, event.turn)
        elif isinstance(event, StepStarted):
            trajectory.steps = max(trajectory.steps, event.step)
        elif isinstance(event, TurnFinished):
            trajectory.status = event.status
        elif isinstance(event, ToolStarted):
            trajectory.tool_calls.append(
                ToolCallRecord(name=event.call.name, call_id=event.call.id)
            )
        elif isinstance(event, ToolFinished):
            record = _last_record(trajectory, event.call.id)
            if record is not None:
                record.is_error = event.result.is_error
                record.concludes_turn = event.result.concludes_turn
                record.aborted_before_dispatch = event.result.content.startswith(
                    "Error: tool call aborted before dispatch"
                )
        elif isinstance(event, ProviderRetryScheduled):
            trajectory.retries += 1
        elif isinstance(event, ContextCompacted):
            trajectory.compactions += 1
        elif isinstance(event, MemoryRecalled):
            trajectory.memory_recalls += 1
        elif isinstance(event, AssistantTurnEvent) and event.message.interrupted:
            trajectory.interrupted_assistant_messages += 1

    return trajectory, on_event


def _last_record(trajectory: Trajectory, call_id: str) -> ToolCallRecord | None:
    for record in reversed(trajectory.tool_calls):
        if record.call_id == call_id:
            return record
    return None


#: 评测脚本可以用 `TrajectoryRecorder()` 这个更直白的名字。
TrajectoryRecorder = trajectory_recorder
