"""Undo recorded delivery file changes after checking for subsequent edits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def _save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _records(delivery_path: Path, report: dict | None = None) -> list[dict]:
    rows = [json.loads(path.read_text(encoding="utf-8"))
            for path in (delivery_path.parent / "undo" / "runs").glob("*.json")]
    if report is not None and any(
        row.get("delivery_revision") != report["revision"]
        or row.get("delivery_started_at") != report["started_at"] for row in rows
    ):
        raise ValueError("交付目录已被复用，撤回记录与当前交付不一致")
    return sorted(rows, key=lambda row: row["started_at"])


def _file(root: Path, name: str) -> Path:
    relative = Path(name)
    if (not name or relative.is_absolute() or ".." in relative.parts
            or relative.as_posix() != name or ":" in name or name == "."):
        raise ValueError(f"无效的交付文件路径：{name}")
    target = root / relative
    for part in (target, *target.parents):
        if part == root:
            break
        if part.is_symlink() or part.is_junction():
            raise ValueError(f"文件路径含链接，不能撤回：{name}")
        if part != target and part.exists() and not part.is_dir():
            raise ValueError(f"文件父路径已改变：{name}")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"文件路径超出工程：{name}")
    return target


def _read(path: Path) -> bytes | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"交付文件已变为目录：{path.name}")
    return path.read_bytes()


def _hash(raw: bytes | None) -> str | None:
    return hashlib.sha256(raw).hexdigest() if raw is not None else None


def _prepare(delivery_path: Path, workspace: Path) -> tuple[dict, list[tuple]]:
    report = json.loads(delivery_path.read_text(encoding="utf-8"))
    if not report.get("workspace") or Path(report["workspace"]).resolve() != workspace.resolve():
        raise ValueError("交付记录不属于当前工程")
    if report.get("undo") != {"schema_version": 1}:
        raise ValueError("这份历史交付未保存撤回所需的原始文件")
    rows = _records(delivery_path, report)
    if any(row["status"] == "completed" for row in rows):
        raise ValueError("本轮文件改动已撤回")
    if any(row["status"] == "incomplete" or (
        row["status"] == "failed" and not row.get("recovered")
    ) for row in rows):
        raise ValueError("上次撤回未完整结束，请先核对工程和撤回记录")
    names = report["changed_files"]
    if not names:
        raise ValueError("本轮没有可撤回的文件改动")
    expected = {name for name in report["before"].keys() | report["after"].keys()
                if report["before"].get(name) != report["after"].get(name)}
    if set(names) != expected or len(names) != len(set(map(Path, names))):
        raise ValueError("交付文件清单与版本记录不一致")
    plan, conflicts = [], []
    root = workspace.resolve()
    backups = (delivery_path.parent / "undo" / "before").resolve()
    for name in names:
        target = _file(root, name)
        current = _read(target)
        if _hash(current) != report["after"].get(name):
            conflicts.append(name)
        original = None
        if name in report["before"]:
            original = _read(_file(backups, name))
            if _hash(original) != report["before"][name]:
                raise ValueError(f"撤回备份缺失或已改变：{name}")
        plan.append((name, target, original, current))
    if conflicts:
        raise ValueError("以下文件在交付后已改变，未撤回任何文件：" + "、".join(conflicts))
    return report, plan


def _replace(path: Path, raw: bytes | None) -> None:
    if raw is None:
        path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    descriptor, temporary = tempfile.mkstemp(prefix=".agent-undo-", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
        if mode is not None:
            temporary.chmod(mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def undo_delivery(delivery_path: Path, workspace: Path) -> dict:
    # A repeated request returns the first success without touching files again.
    report = json.loads(delivery_path.read_text(encoding="utf-8"))
    if not report.get("workspace") or Path(report["workspace"]).resolve() != workspace.resolve():
        raise ValueError("交付记录不属于当前工程")
    previous = next((row for row in _records(delivery_path, report)
                     if row["status"] == "completed"), None)
    if previous is not None:
        return previous
    report, plan = _prepare(delivery_path, workspace)
    identifier = uuid4().hex
    record = delivery_path.parent / "undo" / "runs" / f"{identifier}.json"
    row = {"id": identifier, "started_at": datetime.now(UTC).isoformat(),
           "delivery_revision": report["revision"], "delivery_started_at": report["started_at"],
           "status": "incomplete",
           "files": [name for name, *_ in plan], "applied_files": []}
    _save(record, row)
    applied = []
    try:
        for name, target, original, current in plan:
            _file(workspace.resolve(), name)
            if _read(target) != current:
                raise ValueError(f"撤回期间文件已改变：{name}")
            _replace(target, original)
            applied.append((name, target, original, current))
            row["applied_files"].append(name)
            _save(record, row)
        row["status"] = "completed"
    except (OSError, ValueError) as exc:
        recovery_errors = []
        for name, target, original, current in reversed(applied):
            try:
                _file(workspace.resolve(), name)
                if _read(target) != original:
                    raise ValueError(f"撤回期间文件再次改变：{name}")
                _replace(target, current)
            except (OSError, ValueError) as recovery:
                recovery_errors.append(f"{name}: {recovery}")
        row.update(status="failed", error=str(exc), recovered=not recovery_errors,
                   recovery_errors=recovery_errors)
        raise
    finally:
        row["finished_at"] = datetime.now(UTC).isoformat()
        _save(record, row)
    return row


def undo_summary(delivery_path: Path, workspace: Path) -> dict:
    try:
        _prepare(delivery_path, workspace)
        reason = None
    except (OSError, ValueError, KeyError) as exc:
        reason = str(exc)
    return {"available": reason is None, "reason": reason, "runs": _records(delivery_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        row = undo_delivery(args.delivery, args.workspace)
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(row, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
