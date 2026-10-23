"""Assess the declared fresh-session, no-tool memory recall task from saved output."""
from __future__ import annotations

import json


def check_recall(response: dict, expected: dict) -> dict:
    answer = response.get("answer", "")
    try:
        start = answer.index("{")
        actual, _ = json.JSONDecoder().raw_decode(answer[start:])
    except (ValueError, TypeError):
        actual = None
    checks = {
        "fresh_session": response.get("resumed_messages") == 0,
        "memory_recalled": response.get("memory_recalled", 0) > 0,
        "no_tools": response.get("tool_calls") == [],
        "facts": (isinstance(actual, dict) and all(
            key in actual and actual[key] == value
            and isinstance(actual[key], bool) == isinstance(value, bool)
            for key, value in expected.items()
        )),
    }
    return {"passed": all(checks.values()), "detail": checks,
            "actual": actual, "expected": expected}
