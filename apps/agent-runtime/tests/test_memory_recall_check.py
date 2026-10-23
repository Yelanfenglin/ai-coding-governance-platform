"""The memory_recall check scores a saved answer and never touches the project."""
from __future__ import annotations

import pytest

from evaluation.runner import check_requirement

EXPECTED = {"frequency_hz": 20000, "duty_percent": 25, "baudrate_target": 115200}


def saved(answer, *, tool_calls=(), resumed=0, memory_recalled=1):
    return {"response": {
        "answer": answer, "tool_calls": list(tool_calls),
        "resumed_messages": resumed, "memory_recalled": memory_recalled,
    }}


def run(payload, tmp_path, expected=EXPECTED):
    check = {"id": "recalled_facts", "kind": "memory_recall", "expected": expected}
    return check_requirement(check, payload, tmp_path, tmp_path / "out")


def test_answer_quoting_all_facts_passes(tmp_path):
    answer = (
        "```json\n"
        '{"frequency_hz": 20000, "duty_percent": 25, "baudrate_target": 115200}\n'
        "```\n来源：项目记忆中的最后一次约定。"
    )
    verdict = run(saved(answer), tmp_path)
    assert verdict["passed"] is True
    assert verdict["actual"] == EXPECTED
    assert verdict["detail"] == {
        "fresh_session": True, "memory_recalled": True, "no_tools": True, "facts": True,
    }


def test_wrong_fact_fails_even_when_the_answer_is_well_formed(tmp_path):
    answer = '{"frequency_hz": 20000, "duty_percent": 30, "baudrate_target": 115200}'
    verdict = run(saved(answer), tmp_path)
    assert verdict["passed"] is False
    assert verdict["detail"]["facts"] is False
    assert verdict["detail"]["fresh_session"] is True


def test_missing_fact_is_not_treated_as_a_guess(tmp_path):
    answer = '{"frequency_hz": 20000, "duty_percent": 25}'
    verdict = run(saved(answer), tmp_path)
    assert verdict["passed"] is False and verdict["detail"]["facts"] is False


@pytest.mark.parametrize("payload", [
    saved('{"frequency_hz": 20000, "duty_percent": 25, "baudrate_target": 115200}',
          tool_calls=[{"name": "read_file"}]),
    saved('{"frequency_hz": 20000, "duty_percent": 25, "baudrate_target": 115200}', resumed=3),
    saved('{"frequency_hz": 20000, "duty_percent": 25, "baudrate_target": 115200}',
          memory_recalled=0),
])
def test_fresh_session_without_tools_or_memory_fails(tmp_path, payload):
    verdict = run(payload, tmp_path)
    assert verdict["passed"] is False
    assert verdict["detail"]["facts"] is True


def test_prose_answer_without_json_counts_as_no_recall(tmp_path):
    verdict = run(saved("我记得大概是 20kHz、25%，但波特率不确定。"), tmp_path)
    assert verdict["passed"] is False
    assert verdict["actual"] is None


def test_boolean_fact_must_not_pass_as_an_integer(tmp_path):
    payload = saved('{"hardware_validated": 1}')
    verdict = run(payload, tmp_path, expected={"hardware_validated": True})
    assert verdict["passed"] is False and verdict["detail"]["facts"] is False


def test_missing_saved_response_is_an_explicit_error_not_a_pass(tmp_path):
    with pytest.raises(ValueError, match="memory_recall requires"):
        run({"before": {}}, tmp_path)
