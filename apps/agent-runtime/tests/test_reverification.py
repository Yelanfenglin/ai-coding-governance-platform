"""A host recheck adds evidence without rewriting the original task result."""
from __future__ import annotations

import json

import pytest

from app import cli
from app.contracts import AssistantTurn, Message
from app.providers.mock import MockProvider
from app.reverification import recheck_delivery, recheck_summary


@pytest.fixture
async def failed_delivery(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "module.c").write_text("int value = 1;\n", encoding="utf-8")
    health = tmp_path / "health.txt"
    health.write_text("down", encoding="utf-8")
    checker = tmp_path / "check.py"
    checker.write_text(
        "from pathlib import Path\n"
        f"state = Path({str(health)!r}).read_text()\n"
        "if state == 'mutate': Path('module.c').write_text('changed')\n"
        "print('host dependency: ' + state)\n"
        "raise SystemExit(7 if state == 'down' else 0)\n", encoding="utf-8",
    )
    config = tmp_path / "verification.json"
    config.write_text(json.dumps({"command": ["{python}", str(checker)]}), encoding="utf-8")
    directory = tmp_path / "delivery"
    args = cli.build_parser().parse_args([
        "check", "--workspace", str(project), "--base-url", "http://unused.invalid",
        "--verification", str(config), "--delivery-dir", str(directory), "--no-memory",
    ])
    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **_: MockProvider([
        AssistantTurn(Message("assistant", "done"), "stop"),
    ]))
    assert await cli.run_once(args, console=False) == cli.EXIT_VERIFICATION

    def no_model(**_):
        raise AssertionError("rechecks must not construct a model provider")

    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", no_model)
    return project, config, directory / "delivery.json", health


async def test_rechecks_preserve_failure_and_append_success_and_failure(failed_delivery):
    project, config, path, health = failed_delivery
    original = path.read_bytes()
    health.write_text("ready", encoding="utf-8")
    passed = await recheck_delivery(path, config, project)
    assert passed["passed"] and passed["matches_delivery"]
    assert passed["input_revision"] == passed["output_revision"]
    health.write_text("down", encoding="utf-8")
    failed = await recheck_delivery(path, config, project)
    assert not failed["passed"] and failed["exit_code"] == 7
    assert failed["id"] != passed["id"]
    summary = recheck_summary(path, config, project)
    assert summary["available"] and [r["passed"] for r in summary["runs"]] == [True, False]
    assert path.read_bytes() == original
    assert json.loads(original)["task_acceptance"] == "failed"


@pytest.mark.parametrize("change", ["source", "command", "timeout"])
async def test_recheck_rejects_changed_source_or_host_configuration(failed_delivery, change):
    project, config, path, health = failed_delivery
    health.write_text("ready", encoding="utf-8")
    if change == "source":
        (project / "module.c").write_text("int value = 2;\n", encoding="utf-8")
    else:
        value = json.loads(config.read_text())
        value.update({"command": ["{python}", "-c", "pass"]} if change == "command"
                     else {"timeout": 120})
        config.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="已改变|已变化"):
        await recheck_delivery(path, config, project)
    summary = recheck_summary(path, config, project)
    assert not summary["available"] and summary["runs"] == []


async def test_recheck_that_mutates_project_fails_and_previous_results_become_stale(
    failed_delivery,
):
    project, config, path, health = failed_delivery
    health.write_text("ready", encoding="utf-8")
    assert (await recheck_delivery(path, config, project))["passed"]
    health.write_text("mutate", encoding="utf-8")
    failed = await recheck_delivery(path, config, project)
    assert not failed["passed"] and not failed["matches_delivery"]
    assert "source_error" in failed
    summary = recheck_summary(path, config, project)
    assert not summary["available"]
    assert all(not r["matches_current"] for r in summary["runs"])
