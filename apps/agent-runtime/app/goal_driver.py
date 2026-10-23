"""DSH goal-round-driver scheduling adapted to the single live Python run.

Copyright (c) 2026 DeepSeek. MIT; see THIRD_PARTY_NOTICES.md.
Source commit aa8262ec091698bae9a6b04773a6b5b06ad4aef2.
Round reservation/admission, revision fences, idle-only continuation and stop
semantics are retained. CLI has no Cordis plugin registry or UI/RPC lifecycle.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from .contracts import (
    AgentStatus,
    Message,
    MessageEntry,
    SessionEvent,
    TodoWriteEntry,
    ToolContext,
    TurnEndEntry,
    TurnStartEntry,
)
from .goal_prompts import render_goal_round_prompt
from .goals import GoalError, GoalService, positive_integer
from .inbox import Inbox

TOOL_INITIATOR: ContextVar[object | None] = ContextVar("goal_tool_initiator", default=None)


@dataclass
class RoundAttempt:
    message: Message
    goal_id: str
    revision: int
    round_number: int
    admitted: bool = False
    stale: bool = False
    claimed: bool = False
    cancelled: bool = False


class GoalRoundDriver:
    def __init__(self, goals: GoalService, *, blocked_after: int = 3):
        self.goals = goals
        goals.disarm()
        self.blocked_after = positive_integer(blocked_after, "blockedAfterConsecutiveRounds")
        self.attempt: RoundAttempt | None = None
        self.running = False
        self.turn_open = False
        self.turn_inputs: list[dict[str, Any]] = []
        self.todos: list[dict[str, str]] | None = None
        self.cancel: Callable[[], None] | None = None
        self.unsubscribe: Callable[[], None] | None = None
        goals.on_changed = self._changed

    def bind_inbox(self, inbox: Inbox) -> None:
        self.unsubscribe = inbox.subscribe(self._inbox_event)

    def _inbox_event(self, event: str, target: str, message: Message) -> None:
        attempt = self.attempt
        if attempt is None:
            return
        if message is attempt.message:
            if event == "claimed":
                attempt.claimed = True
            elif event == "discarded":
                attempt.cancelled = True
        elif event == "inserted" and target == "next-turn" and not attempt.admitted:
            attempt.stale = True

    def _changed(self, operation: str) -> None:
        goal = self.goals.get()
        attempt = self.attempt
        if attempt is not None and (goal is None or goal["id"] != attempt.goal_id
                                    or goal["revision"] != attempt.revision):
            attempt.stale = True
        # A host pause cancels active work. A pause from this model turn does not.
        if (operation == "pause" and self.running and TOOL_INITIATOR.get() is not self
                and self.cancel is not None):
            self.cancel()

    def observe(self, event: SessionEvent) -> None:
        if isinstance(event, TurnStartEntry):
            self.turn_open, self.turn_inputs, self.todos = True, [], None
        elif isinstance(event, TurnEndEntry):
            self.turn_open = False
        elif isinstance(event, TodoWriteEntry):
            self.todos = copy.deepcopy(event.todos)
        elif isinstance(event, MessageEntry) and event.message.role == "user":
            self.goals.state.apply(event)
            self.note_input(event.message)

    def note_input(self, message: Message) -> None:
        if message.role == "user":
            self.turn_inputs.append(copy.deepcopy(message.source or {"kind": "user"}))

    async def record(self, event: SessionEvent) -> None:
        if self.goals.session is not None:
            try:
                await self.goals.session.append(event)
            except BaseException:
                self.goals.disarm()
                raise
        self.observe(event)

    def require_execution(self, context: ToolContext) -> None:
        if (context.agent is not self or TOOL_INITIATOR.get() is not self
                or not self.running or not self.turn_open):
            raise GoalError(
                "goal tools require the exact live calling agent inside its active driver",
                "GOAL_TOOL_DRIVER_REQUIRED",
            )

    def require_direct_human(self) -> None:
        if not any(source.get("kind") == "user" for source in self.turn_inputs):
            raise GoalError("this goal operation requires a direct human turn on a top-level agent",
                            "GOAL_TOOL_AUTHORITY_REQUIRED")

    def completion_authority(self) -> str:
        if any(source.get("kind") == "user" for source in self.turn_inputs):
            return "direct-human"
        goal = self.goals.get()
        if goal is not None and any(
            source.get("kind") == "goal" and source.get("goalId") == goal["id"]
            and source.get("revision") == goal["revision"]
            and source.get("round") == goal["roundsStarted"] for source in self.turn_inputs
        ):
            return "goal-round"
        raise GoalError(
            "complete and blocked require a direct human turn or the current goal round",
            "GOAL_TOOL_AUTHORITY_REQUIRED",
        )

    async def reserve(self, inbox: Inbox) -> Message | None:
        """Whole-agent idle only. Ordinary queued input wins over automatic work."""
        try:
            return await self._reserve(inbox)
        except Exception as error:
            logging.getLogger(__name__).warning(
                "goal-round-driver failed; disarming: %s", error,
            )
            self.goals.disarm()
            return None

    async def _reserve(self, inbox: Inbox) -> Message | None:
        if inbox.has_pending:
            return None
        goal = self.goals.get()
        if goal is None or goal["phase"] != "active" or goal["activation"] != "armed":
            return None
        if goal["roundsStarted"] >= goal["maxGoalRounds"]:
            await self.goals.update(
                {"id": goal["id"], "revision": goal["revision"]}, "block",
                blocked_reason={"code": "round-limit", "message": (
                    f"Goal reached its configured limit of {goal['maxGoalRounds']} rounds."
                )},
            )
            return None
        # All session append calls have settled at idle. Recheck after that
        # durability boundary; reserve does not spend the allowance.
        round_number = goal["roundsStarted"] + 1
        message = Message(
            role="user", content=render_goal_round_prompt(
                goal["objective"], round_number, goal["maxGoalRounds"],
            ), source={"kind": "goal", "goalId": goal["id"],
                       "revision": goal["revision"], "round": round_number},
        )
        self.attempt = RoundAttempt(message, goal["id"], goal["revision"], round_number)
        return message

    async def admit(self, message: Message, commit: Callable[[], Awaitable[None]],
                    *, competing_input: bool) -> bool:
        """Count only the exact reserved prompt that enters history, under CAS lock."""
        async with self.goals.lock:
            attempt = self.attempt
            goal = self.goals.get()
            valid = (
                not competing_input and attempt is not None and not attempt.stale
                and not attempt.admitted and attempt.claimed and not attempt.cancelled
                and message is attempt.message
                and goal is not None and goal["phase"] == "active"
                and goal["activation"] == "armed" and goal["id"] == attempt.goal_id
                and goal["revision"] == attempt.revision
                and attempt.round_number == goal["roundsStarted"] + 1
                and message.source == {"kind": "goal", "goalId": attempt.goal_id,
                                       "revision": attempt.revision, "round": attempt.round_number}
                and message.content == render_goal_round_prompt(
                    goal["objective"], attempt.round_number, goal["maxGoalRounds"],
                )
            )
            if not valid:
                self.attempt = None
                return False
            await commit()
            attempt.admitted = True
            return True

    async def after_turn(self, status: AgentStatus) -> None:
        if status == "aborted":
            goal = self.goals.get()
            attempt = self.attempt
            if (attempt is not None and goal is not None and goal["phase"] == "active"
                    and goal["activation"] == "armed" and goal["id"] == attempt.goal_id
                    and goal["revision"] == attempt.revision):
                try:
                    await self.goals.update(
                        {"id": goal["id"], "revision": goal["revision"]}, "pause",
                    )
                except Exception as error:
                    logging.getLogger(__name__).warning(
                        "goal-round-driver could not pause cancelled goal; disarming: %s", error,
                    )
                    self.goals.disarm()
            else:
                self.goals.disarm()
        elif status != "completed":
            # The existing local step/time hard budget also stops continuation.
            self.goals.disarm()
        self.attempt = None

    def close(self) -> None:
        self.running, self.turn_open = False, False
        self.goals.disarm()
        if self.unsubscribe is not None:
            self.unsubscribe()
            self.unsubscribe = None
