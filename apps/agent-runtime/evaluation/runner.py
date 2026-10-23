"""Score a delivery against independent task checks.

Follows rag-api/evaluation/score_results.py: frozen input, explicit denominator,
per-case judgments and an output summary. Checks are local domain extensions,
not an upstream Agent runtime feature. This entry never reruns the Agent/model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from app.delivery import file_hashes, project_files
from app.demos.manifest_check import check_demo_manifest
from evaluation.checks.memory_recall import check_recall


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_requirement(check: dict, delivery: dict, workspace: Path, out: Path) -> dict:
    kind = check["kind"]
    if kind == "memory_recall":
        if "response" not in delivery:
            raise ValueError("memory_recall requires score with a saved delivery response")
        return check_recall(delivery["response"], check["expected"])
    if kind == "manifest":
        actual = check_demo_manifest(workspace)
        return {"passed": actual["ok"], "detail": actual["failed"]}
    if kind == "unchanged":
        name = check["path"]
        old = delivery["before"][name]
        actual = hashlib.sha256((workspace / name).read_bytes()).hexdigest()
        return {"passed": old == actual, "detail": name}
    if kind == "json_value":
        value = read_json(workspace / check["path"])
        for part in check["pointer"].strip("/").split("/"):
            key = part.replace("~1", "/").replace("~0", "~")
            value = value[int(key)] if isinstance(value, list) else value[key]
        expected = check["expected"]
        if isinstance(expected, (dict, list)):
            raise ValueError("json_value checks require a scalar expectation")
        passed = value == expected and isinstance(value, bool) == isinstance(expected, bool)
        return {"passed": passed,
                "actual": value, "expected": expected}
    if kind == "command":
        # Case files are trusted verifier configuration, never model-produced commands.
        command = [part.replace("{python}", sys.executable)
                   .replace("{workspace}", str(workspace)).replace("{out}", str(out))
                   .replace("{evaluation}", str(Path(__file__).resolve().parent))
                   for part in check["command"]]
        run = subprocess.run(
            command, cwd=workspace, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=check.get("timeout", 60),
            env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONIOENCODING": "utf-8"},
        )
        return {"passed": run.returncode == check.get("exit_code", 0), "command": command,
                "exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
    raise ValueError(f"unsupported check: {kind}")


def score(case_path: Path, delivery_path: Path, workspace: Path, out: Path,
          human_interventions: int | None = None) -> dict:
    case_path, delivery_path, workspace, out = (
        path.resolve() for path in (case_path, delivery_path, workspace, out)
    )
    case, delivery = read_json(case_path), read_json(delivery_path)
    if not case["checks"]:
        raise ValueError("task has no acceptance checks")
    out.mkdir(parents=True, exist_ok=True)
    excludes = tuple(Path(path) for path in delivery["snapshot_excludes"])
    def current():
        return file_hashes(project_files(workspace, exclude=excludes))

    seed = (case_path.resolve().parent / case["seed"]).resolve()
    if not seed.is_dir():
        raise ValueError(f"missing task seed: {seed}")
    checks = [
        {"id": "frozen_seed", "passed": file_hashes(project_files(seed)) == delivery["before"]},
        {"id": "delivered_revision", "passed": current() == delivery["after"]},
    ]
    expected_status = case.get("expected_status", "completed")
    response = delivery["response"]
    checks.append({
        "id": "runtime_outcome",
        "passed": (response["status"] == expected_status
                   and delivery["exit_code"] == case.get("expected_exit_code", 0)
                   and (bool(response["answer"].strip())
                        or not case.get("require_answer", expected_status == "completed"))),
    })
    # Do not run checks against some later, unrelated revision.
    if all(item["passed"] for item in checks[:2]):
        for check in case["checks"]:
            try:
                verdict = check_requirement(check, delivery, workspace, out)
            except (OSError, ValueError, KeyError, IndexError, subprocess.TimeoutExpired) as exc:
                verdict = {"passed": False, "error": str(exc)}
            checks.append({"id": check["id"], **verdict})
        checks.append({"id": "verifier_preserved_source", "passed": current() == delivery["after"]})
    passed = all(item["passed"] for item in checks)
    return {
        "id": case["id"], "passed": passed, "checks": checks,
        "delivery": str(delivery_path.resolve()), "case": str(case_path.resolve()),
        "revision": delivery["revision"], "settings": delivery["settings"],
        "runtime_status": response["status"], "steps": response["steps"],
        "tool_calls": len(response["tool_calls"]), "usage": delivery["usage"],
        "elapsed_seconds": delivery["elapsed_seconds"],
        "human_interventions": human_interventions,
        "autonomous_pass": (passed and human_interventions == 0
                            if human_interventions is not None else None),
        "validation_layer": case.get("validation_layer", "declared_task_checks"),
        "vendor_compile": delivery["vendor_compile"], "hardware": delivery["hardware"],
    }


def check_project(case_path: Path, workspace: Path, out: Path) -> dict:
    """Run the same task checks during development, without requiring a delivery."""
    case_path, workspace, out = (path.resolve() for path in (case_path, workspace, out))
    case = read_json(case_path)
    if not case["checks"]:
        raise ValueError("task has no acceptance checks")
    seed = (case_path.parent / case["seed"]).resolve()
    if not seed.is_dir():
        raise ValueError(f"missing task seed: {seed}")
    out.mkdir(parents=True, exist_ok=True)
    baseline = {"before": file_hashes(project_files(seed))}
    before = file_hashes(project_files(workspace, exclude=(out,)))
    checks = []
    for item in case["checks"]:
        try:
            verdict = check_requirement(item, baseline, workspace, out)
        except (OSError, ValueError, KeyError, IndexError, subprocess.TimeoutExpired) as exc:
            verdict = {"passed": False, "error": str(exc)}
        checks.append({"id": item["id"], **verdict})
    checks.append({"id": "verifier_preserved_source", "passed": (
        file_hashes(project_files(workspace, exclude=(out,))) == before
    )})
    return {"id": case["id"], "passed": all(row["passed"] for row in checks), "checks": checks,
            "scope": "requirements_only; delivery and runtime outcome not assessed"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("score", help="score existing delivery; no model calls")
    one.add_argument("--case", type=Path, required=True)
    one.add_argument("--delivery", type=Path, required=True)
    one.add_argument("--workspace", type=Path, required=True)
    one.add_argument("--human-interventions", type=int)
    one.add_argument("--out", type=Path, required=True)
    check = commands.add_parser("check", help="run task checks during development; no model calls")
    check.add_argument("--case", type=Path, required=True)
    check.add_argument("--workspace", type=Path, required=True)
    check.add_argument("--out", type=Path, required=True)
    batch = commands.add_parser("batch", help="score every declared run, including missing inputs")
    batch.add_argument("--runs", type=Path, required=True)
    batch.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "check":
        entries = [{"case": str(args.case.resolve()), "workspace": str(args.workspace.resolve())}]
        base = Path.cwd()
    elif args.command == "score":
        entries = [{"case": str(args.case.resolve()), "delivery": str(args.delivery.resolve()),
                    "workspace": str(args.workspace.resolve()),
                    "human_interventions": args.human_interventions}]
        base = Path.cwd()
    else:
        base = args.runs.resolve().parent
        entries = [json.loads(line) for line in args.runs.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    if not entries:
        parser.error("no runs declared")
    results = []
    for index, entry in enumerate(entries):
        try:
            if args.command == "check":
                result = check_project(base / entry["case"], base / entry["workspace"],
                                       args.out / f"check-{index}")
            else:
                result = score(
                    base / entry["case"], base / entry["delivery"], base / entry["workspace"],
                    args.out / f"check-{index}", entry.get("human_interventions"),
                )
        except (OSError, ValueError, KeyError, IndexError) as exc:
            result = {"id": entry.get("case"), "passed": False, "error": str(exc)}
        results.append(result)
    summary = {"total": len(results), "passed": sum(row["passed"] for row in results),
               "results": results, "model_calls": 0}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "judgments.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8",
    )
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
