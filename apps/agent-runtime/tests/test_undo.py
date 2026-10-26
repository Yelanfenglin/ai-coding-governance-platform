"""Undo restores exact file contents and never overwrites subsequent user edits."""
from __future__ import annotations

import json

import pytest

from app import undo
from app.delivery import DeliveryRecorder
from app.loop import AgentRunResult


@pytest.fixture
def delivered(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    # Includes local edits made before the Agent, a non-UTF8 file and an empty file.
    original = {"module.c": "// 用户已有改动\r\nint value = 1;\r\n".encode("gbk"),
                "deleted.h": b"", "other.c": b"int other = 1;\n"}
    for name, raw in original.items():
        (root / name).write_bytes(raw)
    recorder = DeliveryRecorder(root, root / ".agent-runtime" / "delivery",
                                state_dir=root / ".agent-runtime")
    (root / "module.c").write_bytes(b"int value = 2;\n")
    (root / "deleted.h").unlink()
    (root / "generated").mkdir()
    (root / "generated/new.c").write_bytes(b"int new_value = 3;")
    path = recorder.finish(AgentRunResult("completed", [], 1, 1), question="change files",
                           settings={}, response={"answer": "done"}, exit_code=0)
    return root, path, original


def test_undo_restores_modified_deleted_and_new_files_and_keeps_other_edits(delivered):
    root, path, original = delivered
    artifacts = {p: p.read_bytes() for p in path.parent.glob("*.*")}
    (root / "other.c").write_bytes(b"user edited another file\n")
    (root / "user-added.txt").write_bytes(b"keep me")
    assert undo.undo_summary(path, root)["available"]
    result = undo.undo_delivery(path, root)
    assert result["status"] == "completed"
    assert set(result["files"]) == {"module.c", "deleted.h", "generated/new.c"}
    assert (root / "module.c").read_bytes() == original["module.c"]
    assert (root / "deleted.h").read_bytes() == b""
    assert not (root / "generated/new.c").exists()
    assert (root / "other.c").read_bytes() == b"user edited another file\n"
    assert (root / "user-added.txt").read_bytes() == b"keep me"
    assert all(p.read_bytes() == raw for p, raw in artifacts.items())
    summary = undo.undo_summary(path, root)
    assert not summary["available"] and summary["runs"] == [result]
    # A duplicate POST/CLI retry must not remove or overwrite later user work.
    (root / "module.c").write_bytes(b"edited after undo")
    assert undo.undo_delivery(path, root) == result
    assert (root / "module.c").read_bytes() == b"edited after undo"


@pytest.mark.parametrize("change", ["modified", "removed", "new-edited", "deleted-recreated",
                                         "directory", "parent-file"])
def test_conflict_is_checked_for_all_files_before_any_write(delivered, change):
    root, path, _ = delivered
    if change == "modified":
        (root / "module.c").write_bytes(b"user edit")
    elif change == "removed":
        (root / "module.c").unlink()
    elif change == "new-edited":
        (root / "generated/new.c").write_bytes(b"user edit")
    elif change == "deleted-recreated":
        (root / "deleted.h").write_bytes(b"user created this")
    elif change == "directory":
        (root / "module.c").unlink()
        (root / "module.c").mkdir()
    else:
        (root / "generated/new.c").unlink()
        (root / "generated").rmdir()
        (root / "generated").write_bytes(b"user file")
    current = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        undo.undo_delivery(path, root)
    assert all(p.read_bytes() == raw for p, raw in current.items())
    assert not undo.undo_summary(path, root)["available"]
    assert not (root / "deleted.h").exists() or change == "deleted-recreated"


@pytest.mark.parametrize("invalid", ["backup-missing", "backup-changed", "legacy", "wrong-root",
                                          "unsafe-path", "incomplete"])
def test_invalid_records_cannot_change_files(delivered, invalid, tmp_path):
    root, path, _ = delivered
    report = json.loads(path.read_text(encoding="utf-8"))
    if invalid == "backup-missing":
        (path.parent / "undo/before/module.c").unlink()
    elif invalid == "backup-changed":
        (path.parent / "undo/before/module.c").write_bytes(b"broken")
    elif invalid == "legacy":
        report.pop("undo")
    elif invalid == "wrong-root":
        report["workspace"] = str(tmp_path / "another-project")
    elif invalid == "unsafe-path":
        report["changed_files"].append("../outside.c")
        report["after"]["../outside.c"] = "invalid"
    else:
        directory = path.parent / "undo/runs"
        directory.mkdir()
        (directory / "interrupted.json").write_text(json.dumps({
            "id": "interrupted", "status": "incomplete", "started_at": "2026-01-01",
            "delivery_revision": report["revision"], "delivery_started_at": report["started_at"],
        }))
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        undo.undo_delivery(path, root)
    assert (root / "module.c").read_bytes() == b"int value = 2;\n"
    assert (root / "generated/new.c").exists()
    assert not (root / "deleted.h").exists()


def test_linked_file_cannot_redirect_undo_outside_project(delivered, tmp_path):
    root, path, _ = delivered
    outside = tmp_path / "outside.c"
    outside.write_bytes((root / "module.c").read_bytes())
    (root / "module.c").unlink()
    try:
        (root / "module.c").symlink_to(outside)
    except OSError:
        pytest.skip("creating a symlink requires local permission")
    with pytest.raises(ValueError, match="链接"):
        undo.undo_delivery(path, root)
    assert outside.read_bytes() == b"int value = 2;\n"
    assert not (root / "deleted.h").exists()


def test_io_failure_recovers_already_restored_files_and_records_failure(delivered, monkeypatch):
    root, path, _ = delivered
    replace = undo._replace
    count = 0

    def fail_once(target, raw):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("disk write failed")
        replace(target, raw)

    monkeypatch.setattr(undo, "_replace", fail_once)
    with pytest.raises(OSError, match="disk write failed"):
        undo.undo_delivery(path, root)
    assert (root / "module.c").read_bytes() == b"int value = 2;\n"
    assert (root / "generated/new.c").read_bytes() == b"int new_value = 3;"
    assert not (root / "deleted.h").exists()
    summary = undo.undo_summary(path, root)
    assert summary["available"]
    assert summary["runs"][0]["status"] == "failed"
    assert summary["runs"][0]["recovered"]
    assert undo.undo_delivery(path, root)["status"] == "completed"


def test_cli_undo_uses_saved_delivery_without_a_model(delivered, capsys):
    root, path, original = delivered
    assert undo.main(["--workspace", str(root), "--delivery", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    assert (root / "module.c").read_bytes() == original["module.c"]


def test_failure_to_recover_is_visible_and_blocks_another_undo(delivered, monkeypatch):
    root, path, _ = delivered
    replace = undo._replace
    count = 0

    def fail_during_write_and_recovery(target, raw):
        nonlocal count
        count += 1
        if count in {2, 3}:
            raise OSError("disk unavailable")
        replace(target, raw)

    monkeypatch.setattr(undo, "_replace", fail_during_write_and_recovery)
    with pytest.raises(OSError):
        undo.undo_delivery(path, root)
    summary = undo.undo_summary(path, root)
    assert not summary["available"]
    assert not summary["runs"][0]["recovered"]
    assert summary["runs"][0]["recovery_errors"]
    with pytest.raises(ValueError, match="未完整结束"):
        undo.undo_delivery(path, root)


def test_cli_reports_a_conflict_and_returns_failure(delivered, capsys):
    root, path, _ = delivered
    (root / "module.c").write_bytes(b"keep user edit")
    assert undo.main(["--workspace", str(root), "--delivery", str(path)]) == 1
    assert "module.c" in json.loads(capsys.readouterr().out)["error"]
    assert (root / "module.c").read_bytes() == b"keep user edit"


def test_reused_delivery_directory_cannot_return_an_old_undo_success(delivered):
    root, path, _ = delivered
    undo.undo_delivery(path, root)
    report = json.loads(path.read_text(encoding="utf-8"))
    report["started_at"] = "another delivery"
    path.write_text(json.dumps(report), encoding="utf-8")
    (root / "module.c").write_bytes(b"new task contents")
    with pytest.raises(ValueError, match="目录已被复用"):
        undo.undo_delivery(path, root)
    assert (root / "module.c").read_bytes() == b"new task contents"
