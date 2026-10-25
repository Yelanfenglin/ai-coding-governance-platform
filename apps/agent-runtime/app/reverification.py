"""Explicit host rechecks of an unchanged delivery, without invoking a model."""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .project_verification import ProjectVerification


def prepare_recheck(delivery_path: Path, config: Path | None,
                    workspace: Path) -> tuple[dict, ProjectVerification]:
    if config is None:
        raise ValueError("当前服务未配置宿主验收")
    delivery = json.loads(delivery_path.read_text(encoding="utf-8"))
    if Path(delivery["workspace"]).resolve() != workspace.resolve():
        raise ValueError("交付记录不属于当前工程")
    previous = next((row for row in reversed((delivery.get("verification") or {}).get("runs", []))
                     if row["phase"] == "delivery"), None)
    if previous is None:
        raise ValueError("本次交付没有可复验的宿主最终检查")
    checker = ProjectVerification(config, workspace, exclude=tuple(
        Path(path) for path in delivery["snapshot_excludes"]
    ))
    if (checker.command != previous["command"] or str(checker.cwd) != previous["cwd"]
            or checker.timeout != previous["timeout"]):
        raise ValueError("宿主验收命令或参数已改变，不能作为原交付的复验")
    if checker.current_revision() != delivery["revision"]:
        raise ValueError("工程文件已变化，不能复验旧交付；请对当前版本重新执行任务")
    return delivery, checker


def _save(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


async def recheck_delivery(delivery_path: Path, config: Path | None, workspace: Path) -> dict:
    delivery, checker = prepare_recheck(delivery_path, config, workspace)
    identifier = uuid4().hex
    path = delivery_path.parent / "rechecks" / f"{identifier}.json"
    row = {"id": identifier, "started_at": datetime.now(UTC).isoformat(),
           "delivery_revision": delivery["revision"], "status": "incomplete", "passed": False}
    _save(path, row)
    try:
        outcome = await checker.run(phase="recheck")
        matches = (outcome["input_revision"] == outcome["output_revision"]
                   == delivery["revision"])
        row.update(outcome, matches_delivery=matches)
        row["passed"] = outcome["passed"] and matches
        row["status"] = "passed" if row["passed"] else "failed"
    except asyncio.CancelledError:
        row.update(status="cancelled", error="复验已中断")
        raise
    finally:
        row["finished_at"] = datetime.now(UTC).isoformat()
        _save(path, row)
    return row


def recheck_summary(delivery_path: Path, config: Path | None, workspace: Path) -> dict:
    try:
        _, checker = prepare_recheck(delivery_path, config, workspace)
        revision = checker.current_revision()
        reason = None
    except (OSError, ValueError, KeyError) as exc:
        revision, reason = None, str(exc)
    records = [json.loads(path.read_text(encoding="utf-8"))
               for path in (delivery_path.parent / "rechecks").glob("*.json")]
    records.sort(key=lambda row: row["started_at"])
    for row in records:
        row["matches_current"] = bool(revision and row.get("matches_delivery")
                                      and row.get("output_revision") == revision)
    return {"available": reason is None, "reason": reason, "runs": records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    try:
        row = asyncio.run(recheck_delivery(args.delivery, args.verification, args.workspace))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(row, ensure_ascii=False, indent=2))
    return 0 if row["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
