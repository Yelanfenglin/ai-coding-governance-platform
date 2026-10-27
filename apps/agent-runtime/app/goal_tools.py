"""通过 FunctionTool 提供目标与待办事项的模型调用接口。"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from .contracts import Message, TodoWriteEntry, ToolContext, ToolResult
from .goal_prompts import render_wrapup_context
from .goals import GoalError, positive_integer
from .tools import FunctionTool

if TYPE_CHECKING:
    from .goal_driver import GoalRoundDriver

CREATE_DESCRIPTION = (
    "Create one persisted same-session completion goal when the current direct human request "
    "is a long-running objective that should continue across autonomous goal rounds. You may "
    'infer that intent without requiring the user to say "create a goal". Do not use this for '
    "trivial single-turn work. Execution rejects non-human and subagent authority."
)
GET_DESCRIPTION = (
    "Read the current same-session goal, including its exact id/revision, objective, phase, "
    "completed continuation rounds, round limit, blocker reason when present, and whether another "
    "continuation "
    "is armed. Call this before updating a goal."
)
UPDATE_DESCRIPTION = (
    "Update the exact current goal revision. edit, pause, and resume require a direct "
    "top-level human request. During an automatic continuation of the current goal, complete "
    "and blocked are also allowed. blocked is rejected before the configured minimum round count; "
    "the model remains responsible for judging that the same condition persisted across those "
    "rounds and must explain it in blocked_reason."
)


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": required}


def goal_value(goal: dict[str, Any] | None) -> ToolResult:
    if goal is None:
        value = {"goal": None}
    else:
        fields = ("id", "revision", "objective", "phase", "roundsStarted", "maxGoalRounds")
        value = {"goal": {key: goal[key] for key in fields}, "activation": goal["activation"]}
        if "blockedReason" in goal:
            value["goal"]["blockedReason"] = goal["blockedReason"]
    return ToolResult(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def goal_tools(driver: GoalRoundDriver) -> list[FunctionTool]:
    async def get_goal(args: dict[str, Any], context: ToolContext) -> ToolResult:
        driver.require_execution(context)
        return goal_value(driver.goals.get())

    async def create_goal(args: dict[str, Any], context: ToolContext) -> ToolResult:
        driver.require_execution(context)
        driver.require_direct_human()
        return goal_value(await driver.goals.create(args["objective"], args.get("max_goal_rounds")))

    async def update_goal(args: dict[str, Any], context: ToolContext) -> ToolResult:
        driver.require_execution(context)
        goal_id = args["goal_id"]
        if not goal_id or goal_id != goal_id.strip():
            raise GoalError("goal_id must be non-empty and normalized", "GOAL_TOOL_INVALID_UPDATE")
        ref = {"id": goal_id, "revision": positive_integer(args["revision"], "revision")}
        action = args["action"]
        objective = args.get("objective") or None
        cap = args.get("max_goal_rounds") or None
        reason = args.get("blocked_reason") or None
        if action in {"edit", "pause", "resume"}:
            driver.require_direct_human()
            if (reason is not None
                    or (action != "edit" and (objective is not None or cap is not None))):
                raise GoalError("replacement fields require edit; blocked_reason requires blocked",
                                "GOAL_TOOL_INVALID_UPDATE")
            current = driver.goals.get()
            if (action == "resume" and current is not None and current["id"] == ref["id"]
                    and current["revision"] == ref["revision"] and current["phase"] == "paused"):
                raise GoalError("the model cannot resume a paused goal; the user must resume it",
                                "GOAL_TOOL_RESUME_PAUSED")
        else:
            authority = driver.completion_authority()
            if objective is not None or cap is not None or (action == "complete" and reason):
                raise GoalError("invalid goal update fields", "GOAL_TOOL_INVALID_UPDATE")
            if action == "blocked":
                if reason is None or not reason.strip():
                    raise GoalError("blocked_reason is required", "GOAL_TOOL_INVALID_UPDATE")
                current = driver.goals.get()
                if (authority == "goal-round" and current is not None
                        and current["roundsStarted"] < driver.blocked_after):
                    raise GoalError(
                        f"blocked requires at least {driver.blocked_after} consecutive goal rounds;"
                        " "
                        f"current round is {current['roundsStarted']}", "GOAL_TOOL_BLOCK_THRESHOLD",
                    )
        goal = await driver.goals.update(
            ref, "block" if action == "blocked" else action,
            objective=objective, max_goal_rounds=cap,
            blocked_reason={"code": "model-reported", "message": reason} if reason else None,
        )
        if action in {"complete", "blocked"} and authority == "goal-round":
            summary = f"{action}: {goal['objective']}"
            if len(summary) > 120:
                summary = summary[:119] + "…"
            context.defer_context(Message(  # type: ignore[misc]
                role="user", content=render_wrapup_context(goal["objective"], reason),
                source={"kind": "plugin", "plugin": "tool-goal", "form": "notice",
                        "summary": summary},
            ))
        return goal_value(goal)

    text = {"type": "string"}
    number = {"type": "number"}
    return [
        FunctionTool("get_goal", GET_DESCRIPTION, _schema({}, []), get_goal),
        FunctionTool("create_goal", CREATE_DESCRIPTION, _schema(
            {"objective": {**text, "description": (
                "The concrete completion objective inferred from the direct human request."
            )}, "max_goal_rounds": {**number, "description": (
                "Optional positive safe-integer limit on automatic continuation rounds."
            )}},
            ["objective"]), create_goal),
        FunctionTool("update_goal", UPDATE_DESCRIPTION, _schema({
            "goal_id": {**text, "description": "Exact id returned by get_goal."},
            "revision": {**number, "description": "Exact positive revision returned by get_goal."},
            "action": {"type": "string", "enum": [
                "edit", "pause", "resume", "complete", "blocked",
            ], "description": "edit | pause | resume | complete | blocked"},
            "objective": {**text, "description": (
                "Replacement objective; valid only with action edit."
            )},
            "max_goal_rounds": {**number, "description": (
                "Replacement cap; valid only with action edit."
            )},
            "blocked_reason": {**text, "description":
                               "Concrete blocking condition; required only with action blocked."},
        }, ["goal_id", "revision", "action"]), update_goal),
    ]


def todo_tool(driver: GoalRoundDriver, *, allow_parallel: bool) -> FunctionTool:
    description = (
        "Record and update a structured task list for the current work. Send the ENTIRE "
        "list every call — it REPLACES the previous list (there are no partial updates, "
        "no per-item edits). Use it to plan multi-step work and show progress: add one "
        "todo per concrete step before you start. "
    ) + (
        "Mark every todo being actively worked on `in_progress` — several at once when work "
        "genuinely runs in parallel (e.g. concurrent subagents or background commands), one for "
        "sequential work; while work remains, at least one task should be `in_progress`. "
        if allow_parallel else
        "Keep AT MOST ONE todo `in_progress` at a time; while work remains, exactly one active "
        "task should be `in_progress`. "
    ) + (
        "Mark a todo `completed` the moment it is done (do not batch completions), and allow no "
        "`in_progress` item only once all work is complete. Skip the list for trivial "
        "single-step tasks. Statuses: `pending` (not started), `in_progress` (being "
        "worked on now), `completed` (finished)."
    )

    async def write(args: dict[str, Any], context: ToolContext) -> ToolResult:
        driver.require_execution(context)
        todos, seen = [], set()
        for item in args["todos"]:
            content = item["content"].strip()
            if not content or content in seen:
                raise ValueError("todo content must be non-empty and unique")
            seen.add(content)
            todos.append({"content": content, "status": item["status"]})
        active = sum(item["status"] == "in_progress" for item in todos)
        if not allow_parallel and active > 1:
            raise ValueError(f"invalid todos: at most one task may be in_progress (got {active})")
        await driver.record(TodoWriteEntry(todos))
        pending = sum(item["status"] == "pending" for item in todos)
        completed = sum(item["status"] == "completed" for item in todos)
        return ToolResult(
            f"Updated todo list: {pending} pending, {active} in progress, {completed} completed."
        )

    schema = _schema({"todos": {"type": "array",
                                "description": (
                                    "The COMPLETE task list, replacing any previous list."
                                ),
                                "items": _schema({
        "content": {"type": "string", "description": "What the task is — a short imperative line."},
        "status": {"type": "string", "enum": [
            "pending", "in_progress", "completed",
        ], "description": "pending (not started) | in_progress (now) | completed (done)."},
    }, ["content", "status"])}}, ["todos"])
    return FunctionTool("todo_write", description, schema, write)
