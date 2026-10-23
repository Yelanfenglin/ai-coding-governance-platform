"""DSH goal/src/{fold,index}.ts, aa8262ec: same-session goal state.

Copyright (c) 2026 DeepSeek. MIT; see THIRD_PARTY_NOTICES.md.
The Python host awaits its session sink; state changes commit only after append.
Cordis/RPC/UI integration is not part of this single-agent CLI adapter.
"""

from __future__ import annotations

import asyncio
import copy
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .contracts import GoalChangeEntry, MessageEntry, SessionEvent, SessionSink


class GoalError(ValueError):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


def positive_integer(value: Any, field_name: str) -> int:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not float(value).is_integer() or not 1 <= value <= 2**53 - 1):
        raise ValueError(f"{field_name} must be a positive safe integer")
    return int(value)


def _integer(value: Any, name: str) -> int:
    if value == 0 and not isinstance(value, bool):
        return 0
    return positive_integer(value, name)


def _keys(value: Any, names: str) -> None:
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise ValueError(f"goal change must have exactly {names} fields")


def _objective(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GoalError("goal objective must be a non-empty string", "GOAL_INVALID_OBJECTIVE")
    return value.strip()


def _max_rounds(value: Any) -> int:
    try:
        return positive_integer(value, "maxGoalRounds")
    except ValueError as exc:
        raise GoalError(str(exc), "GOAL_INVALID_MAX_ROUNDS") from exc


def _block_reason(value: Any) -> dict[str, str]:
    _keys(value, "code message")
    if (not isinstance(value["code"], str)
            or re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", value["code"]) is None
            or not isinstance(value["message"], str) or not value["message"].strip()):
        raise GoalError("invalid goal block reason", "GOAL_INVALID_BLOCK_REASON")
    return {"code": value["code"], "message": value["message"].strip()}


@dataclass
class GoalFold:
    """Strict durable fold. Activation deliberately does not survive replay."""

    goal: dict[str, Any] | None = None
    rounds_started: int = 0
    created_at: int | None = None
    updated_at: int | None = None
    last_ref: dict[str, Any] | None = None
    seen_goal_ids: set[str] = field(default_factory=set)

    def apply(self, event: SessionEvent) -> None:
        if isinstance(event, MessageEntry):
            source = event.message.source or {}
            if event.message.role != "user" or source.get("kind") != "goal":
                return
            goal = self.goal
            round_number = positive_integer(source.get("round"), "round")
            positive_integer(source.get("revision"), "revision")
            if (goal is None or goal["phase"] != "active"
                    or source.get("goalId") != goal["id"]
                    or source.get("revision") != goal["revision"]
                    or round_number != self.rounds_started + 1
                    or round_number > goal["maxGoalRounds"]):
                raise ValueError("goal message is not the next admitted round of the active goal")
            self.rounds_started = round_number
            return
        if not isinstance(event, GoalChangeEntry):
            return
        change = event.data
        if (change.get("kind") != "goal/change" or isinstance(change.get("version"), bool)
                or change.get("version") != 1):
            raise ValueError("unsupported goal change kind/version")
        operation = change.get("operation")
        current = self.goal
        if operation == "clear":
            _keys(change, "kind version operation cleared clearedAt")
            _keys(change["cleared"], "id revision")
            ref = change["cleared"]
            positive_integer(ref["revision"], "revision")
            if (current is None or ref["id"] != current["id"]
                    or ref["revision"] != current["revision"] + 1
                    or _integer(change["clearedAt"], "clearedAt") < self.updated_at):
                raise ValueError("invalid goal clear transition")
            self.goal, self.rounds_started = None, 0
            self.created_at = self.updated_at = None
            self.last_ref = copy.deepcopy(ref)
            return
        _keys(change, "kind version operation goal roundsStarted createdAt updatedAt")
        if operation not in {"create", "edit", "pause", "resume", "complete", "block"}:
            raise ValueError("goal change operation is invalid")
        goal = change["goal"]
        if not isinstance(goal, dict):
            raise ValueError("goal snapshot must be a record")
        phase = goal.get("phase")
        _keys(goal, "id revision objective phase maxGoalRounds"
              + (" blockedReason" if phase == "blocked" else ""))
        if phase not in {"active", "paused", "blocked", "complete"}:
            raise ValueError("invalid goal phase")
        if not isinstance(goal["id"], str) or not goal["id"]:
            raise ValueError("goal id must be non-empty")
        positive_integer(goal["revision"], "revision")
        positive_integer(goal["maxGoalRounds"], "maxGoalRounds")
        if _objective(goal["objective"]) != goal["objective"]:
            raise ValueError("goal objective must be normalized")
        if phase == "blocked" and _block_reason(goal["blockedReason"]) != goal["blockedReason"]:
            raise ValueError("goal block reason must be normalized")
        rounds = _integer(change["roundsStarted"], "roundsStarted")
        created = _integer(change["createdAt"], "createdAt")
        updated = _integer(change["updatedAt"], "updatedAt")
        if updated < created:
            raise ValueError("goal update cannot precede creation")
        if operation == "create":
            if (goal["revision"] != 1 or phase != "active" or rounds != 0
                    or (current is not None and current["phase"] != "complete")
                    or goal["id"] in self.seen_goal_ids):
                raise ValueError("goal create requires a fresh active revision-one goal")
            self.seen_goal_ids.add(goal["id"])
        else:
            if (current is None or goal["id"] != current["id"]
                    or goal["revision"] != current["revision"] + 1):
                raise ValueError("goal mutation must advance current revision by one")
            if (created != self.created_at or updated < self.updated_at
                    or rounds != self.rounds_started):
                raise ValueError("goal mutation must preserve counters and timestamps")
            if operation == "edit":
                if phase != current["phase"] or goal.get("blockedReason") != current.get(
                    "blockedReason"
                ):
                    raise ValueError("goal edit cannot change phase or blocked reason")
            else:
                if any(goal[k] != current[k] for k in ("objective", "maxGoalRounds")):
                    raise ValueError("goal transition cannot change definition")
                valid = {
                    "pause": current["phase"] == "active" and phase == "paused",
                    "resume": current["phase"] in {"active", "paused", "blocked"}
                    and phase == "active" and rounds < goal["maxGoalRounds"],
                    "complete": current["phase"] != "complete" and phase == "complete",
                    "block": current["phase"] == "active" and phase == "blocked",
                }
                if not valid[operation]:
                    raise ValueError(f"goal {operation} has an invalid transition")
        self.goal = copy.deepcopy(goal)
        self.rounds_started, self.created_at, self.updated_at = rounds, created, updated
        self.last_ref = {"id": goal["id"], "revision": goal["revision"]}


class GoalService:
    """DSH GoalService lifecycle and CAS; JSONL is the owning durable source."""

    def __init__(self, session: SessionSink | None = None, *, default_max_goal_rounds: int = 256):
        self.session = session
        self.default_max_goal_rounds = _max_rounds(default_max_goal_rounds)
        self.state = GoalFold()
        self.activation = "disarmed"
        self.lock = asyncio.Lock()
        self.on_changed: Callable[[str], None] | None = None

    def replay(self, events: list[SessionEvent]) -> None:
        state = GoalFold()
        for event in events:
            state.apply(event)
        self.state, self.activation = state, "disarmed"

    def get(self) -> dict[str, Any] | None:
        if self.state.goal is None:
            return None
        return {**copy.deepcopy(self.state.goal), "roundsStarted": self.state.rounds_started,
                "createdAt": self.state.created_at, "updatedAt": self.state.updated_at,
                "activation": self.activation}

    def disarm(self) -> None:
        self.activation = "disarmed"

    def _current(self, ref: dict[str, Any]) -> dict[str, Any]:
        goal = self.state.goal
        if goal is None:
            raise GoalError("no current goal", "GOAL_NOT_FOUND")
        if ref != {"id": goal["id"], "revision": goal["revision"]}:
            raise GoalError("stale goal ref", "GOAL_STALE_REVISION")
        return goal

    async def create(self, objective: str, max_goal_rounds: int | None = None) -> dict[str, Any]:
        objective = _objective(objective)
        cap = _max_rounds(
            self.default_max_goal_rounds if max_goal_rounds is None else max_goal_rounds
        )
        async with self.lock:
            if self.state.goal is not None and self.state.goal["phase"] != "complete":
                raise GoalError("goal already exists", "GOAL_ALREADY_EXISTS")
            return await self._commit("create", {
                "id": f"goal-{uuid.uuid4()}", "revision": 1, "objective": objective,
                "phase": "active", "maxGoalRounds": cap,
            }, "armed")

    async def update(self, ref: dict[str, Any], action: str, *, objective: str | None = None,
                     max_goal_rounds: int | None = None,
                     blocked_reason: dict[str, str] | None = None) -> dict[str, Any] | None:
        async with self.lock:
            current = self._current(ref)
            goal = {**current, "revision": current["revision"] + 1}
            activation = self.activation
            if action == "edit":
                if objective is None and max_goal_rounds is None:
                    raise GoalError("goal edit requires a replacement", "GOAL_INVALID_EDIT")
                if objective is not None:
                    goal["objective"] = _objective(objective)
                if max_goal_rounds is not None:
                    goal["maxGoalRounds"] = _max_rounds(max_goal_rounds)
            elif action == "resume":
                if (current["phase"] not in {"active", "paused", "blocked"}
                        or (current["phase"] == "active" and activation == "armed")
                        or self.state.rounds_started >= current["maxGoalRounds"]):
                    raise GoalError("goal cannot be resumed", "GOAL_INVALID_TRANSITION")
                goal["phase"], activation = "active", "armed"
                goal.pop("blockedReason", None)
            elif action in {"pause", "complete", "block"}:
                allowed = {"pause": {"active"}, "block": {"active"},
                           "complete": {"active", "paused", "blocked"}}
                if current["phase"] not in allowed[action]:
                    raise GoalError(f"cannot {action} goal", "GOAL_INVALID_TRANSITION")
                goal["phase"] = {"pause": "paused", "complete": "complete", "block": "blocked"}[
                    action
                ]
                goal.pop("blockedReason", None)
                if action == "block":
                    goal["blockedReason"] = _block_reason(blocked_reason)
                activation = "disarmed"
            elif action == "clear":
                change = {"kind": "goal/change", "version": 1, "operation": "clear",
                          "cleared": {"id": goal["id"], "revision": goal["revision"]},
                          "clearedAt": self._now()}
                await self._write(change, "disarmed")
                return None
            else:
                raise GoalError("invalid goal update", "GOAL_INVALID_TRANSITION")
            return await self._commit(action, goal, activation)

    def _now(self) -> int:
        return max(int(time.time() * 1000), self.state.updated_at or 0)

    async def _commit(self, operation: str, goal: dict[str, Any],
                      activation: str) -> dict[str, Any]:
        now = self._now()
        create = operation == "create"
        change = {"kind": "goal/change", "version": 1, "operation": operation, "goal": goal,
                  "roundsStarted": 0 if create else self.state.rounds_started,
                  "createdAt": now if create else self.state.created_at, "updatedAt": now}
        await self._write(change, activation)
        return self.get()  # type: ignore[return-value]

    async def _write(self, change: dict[str, Any], activation: str) -> None:
        event = GoalChangeEntry(copy.deepcopy(change))
        candidate = copy.deepcopy(self.state)
        candidate.apply(event)
        if self.session is not None:
            try:
                await self.session.append(event)
            except BaseException:
                self.disarm()
                raise
        self.state, self.activation = candidate, activation
        if self.on_changed is not None:
            self.on_changed(change["operation"])
