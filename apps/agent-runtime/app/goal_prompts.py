"""Verbatim DSH goal-round-driver/prompt.ts and tool-goal/wrapup.ts text.

Copyright (c) 2026 DeepSeek. MIT; see THIRD_PARTY_NOTICES.md.
Source commit aa8262ec091698bae9a6b04773a6b5b06ad4aef2.
"""

import json

GROUNDING = (
    "Report only what earlier rounds and tool results in this session actually establish; "
    "when a detail is not in the session, say so instead of inventing it. "
)


def quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def render_goal_round_prompt(objective: str, round_number: int, max_goal_rounds: int) -> str:
    return (
        "<goal_round>\n"
        f"Objective: {quoted(objective)}\n"
        f"Round: {round_number}/{max_goal_rounds}\n\n"
        "Continue working toward the objective in this same session. Treat the current workspace, "
        "tool results, and durable session state as authoritative; inspect them instead of "
        "assuming "
        "earlier narration is still current. Make concrete progress and verify the result. Before "
        "claiming completion, gather evidence that the whole objective is achieved, read the "
        "current "
        "goal, and mark it complete. If work remains, leave the goal active for the next round. "
        "Follow "
        "the configured goal-tool policy before reporting a blocker.\n"
        "</goal_round>"
    )


def render_wrapup_context(objective: str, blocked_reason: str | None = None) -> str:
    heading = f"Objective: {quoted(objective)}\n"
    if blocked_reason is None:
        return (
            "<goal_complete>\n" + heading
            + "The goal is marked complete and this autonomous run is ending. Write the closing "
            "message to the user now: state the outcome, summarize what was done and how it was "
            "verified, and point to the concrete results (files, commits, or other artifacts). "
            + GROUNDING
            + "Note anything the user should review or do next. Address the user directly. Do not "
            "call any more tools in this run; further work waits for the user's next instruction.\n"
            "</goal_complete>"
        )
    return (
        "<goal_blocked>\n" + heading + f"Blocked: {quoted(blocked_reason)}\n"
        + "The goal is marked blocked and this autonomous run is ending. Write the closing "
        "message to the user now: state what has been completed so far, describe the concrete "
        "blocking condition and what you tried, and say exactly what you need from the user to "
        "continue. " + GROUNDING
        + "Address the user directly. Do not call any more tools in this run; further work "
        "waits for the user's next instruction.\n</goal_blocked>"
    )


def goal_guidance(blocked_after: int = 3) -> str:
    """DSH tool-goal/index.ts::guidance, including its original policy."""
    return (
        "Use goal tools for one long-running completion objective in the current session. "
        "create_goal may infer goal intent from a direct human request in any language; do not "
        "create a goal for routine single-turn work. Call get_goal before update_goal and copy its "
        "exact goal_id and revision. After session resume or fork, an active goal is disarmed: "
        "when "
        "a human asks to continue or resume in any wording or language, use update_goal action "
        "resume to rearm it. Mark complete only when the objective is actually achieved. Mark "
        f"blocked only after the same blocking condition persists for at least {blocked_after} "
        "consecutive goal rounds, and report that concrete condition in blocked_reason; "
        "difficulty, "
        "uncertainty, or useful remaining work is not blocked."
    )
