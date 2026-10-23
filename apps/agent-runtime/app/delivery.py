"""Task delivery evidence collected by the host, separate from the model's answer.

Uses the existing workspace snapshot and Trajectory recorder. File revisions bind
command output to the delivered source; they do not certify that a command tests
the requirements. Task scoring remains an independent, manually invoked step.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .contracts import ToolFinished, ToolStarted
from .demos.manifest_check import IGNORED_DIRS
from .implementation import ImplementationReview
from .trajectory import trajectory_recorder


def project_files(target: Path, *, exclude: tuple[Path, ...] = ()) -> dict[Path, bytes]:
    """The same source/config/document scope used by the CLI manifest gate."""
    suffixes = {".c", ".h", ".cpp", ".hpp", ".s", ".asm", ".inc", ".py", ".md",
                ".txt", ".json", ".prj", ".toml", ".yaml", ".yml"}
    ignored = IGNORED_DIRS | {".git", ".agent-runtime"}
    return {
        path.relative_to(target): path.read_bytes()
        for path in target.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
        and not ignored.intersection(path.relative_to(target).parts)
        and not any(path.resolve().is_relative_to(item.resolve()) for item in exclude)
    }


def file_hashes(files: dict[Path, bytes]) -> dict[str, str]:
    return {path.as_posix(): hashlib.sha256(raw).hexdigest() for path, raw in sorted(files.items())}


def revision(hashes: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


class DeliveryRecorder:
    def __init__(self, workspace: Path | None, directory: Path, *, state_dir: Path) -> None:
        self.workspace = workspace
        self.directory = directory.resolve()
        self.excluded = (self.directory, state_dir.resolve())
        self.before = self.snapshot()
        self.started_at = datetime.now(UTC).isoformat()
        self.clock = time.monotonic()
        self.trajectory, self.record_event = trajectory_recorder()
        self.tools: list[dict] = []
        self.implementation = (ImplementationReview(workspace, self.snapshot, self.tools)
                               if workspace else None)

    def snapshot(self) -> dict[Path, bytes]:
        return project_files(self.workspace, exclude=self.excluded) if self.workspace else {}

    def observe(self, event: object) -> None:
        self.record_event(event)
        if isinstance(event, ToolStarted):
            self.tools.append({
                "call_id": event.call.id, "name": event.call.name,
                "arguments": asdict(event.call)["arguments"],
                "input_revision": (
                    revision(file_hashes(self.snapshot())) if event.call.name in {
                        "run_command", "verify_project",
                    }
                    else None
                ),
            })
        elif isinstance(event, ToolFinished):
            row = next((item for item in reversed(self.tools)
                        if item["call_id"] == event.call.id), None)
            if row is not None:
                row.update(content=event.result.content, is_error=event.result.is_error)
                if row["name"] in {"run_command", "verify_project"}:
                    row["output_revision"] = revision(file_hashes(self.snapshot()))

    def finish(self, result, *, question: str, settings: dict, response: dict,
               exit_code: int, session: Path | None = None) -> Path:
        after = self.snapshot()
        before_hashes, after_hashes = file_hashes(self.before), file_hashes(after)
        final_revision = revision(after_hashes)
        commands = [dict(row, matches_delivery=(
            row["input_revision"] == row.get("output_revision") == final_revision
        )) for row in self.tools if row["name"] == "run_command"]
        evidence_tools = {"search_spec", "read_context", "read_official_example"}
        changes = sorted(path for path in self.before.keys() | after.keys()
                         if self.before.get(path) != after.get(path))
        patch = b"".join(
            b"".join(difflib.diff_bytes(
                difflib.unified_diff, self.before.get(path, b"").splitlines(keepends=True),
                after.get(path, b"").splitlines(keepends=True),
                fromfile=f"before/{path.as_posix()}".encode(),
                tofile=f"after/{path.as_posix()}".encode(),
            )) for path in changes
        )
        report = {
            "schema_version": 1, "question": question,
            "started_at": self.started_at, "elapsed_seconds": time.monotonic() - self.clock,
            "settings": settings, "response": response, "exit_code": exit_code,
            "task_acceptance": response.get("verification", {}).get("status", "unverified"),
            "verification": response.get("verification"),
            "decisions": self.implementation.finish(after, final_revision)
                         if self.implementation else [],
            "session": str(session.resolve()) if session else None,
            "workspace": str(self.workspace) if self.workspace else None,
            "snapshot_excludes": [str(path) for path in self.excluded],
            "before": before_hashes, "after": after_hashes, "revision": final_revision,
            "changed_files": [path.as_posix() for path in changes],
            "commands": commands,
            "evidence": [row for row in self.tools if row["name"] in evidence_tools],
            "tools": self.tools, "trajectory": self.trajectory.finish(result).summary(),
            "usage": asdict(result.usage),
            "runtime_source": file_hashes(project_files(Path(__file__).resolve().parent)),
            "vendor_compile": "not_assessed", "hardware": "not_assessed",
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "delivery.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (self.directory / "changes.diff").write_bytes(patch)
        (self.directory / "report.md").write_text(
            f"# Task delivery\n\n{question}\n\n"
            f"Runtime: {result.status}; exit code: {exit_code}. "
            f"Task acceptance: {report['task_acceptance']} (configured checks only).\n\n"
            f"Revision: `{final_revision}`\n\n"
            "Changed files:\n\n" + "\n".join(f"- `{name}`" for name in report["changed_files"])
            + "\n\nCommands:\n\n" + "\n".join(
                f"- `{row['arguments']}`; error={row.get('is_error')}; "
                f"matches delivered revision={row['matches_delivery']}" for row in commands
            ) + "\n\nVerification:\n\n```json\n"
            + json.dumps(report["verification"], ensure_ascii=False, indent=2)
            + "\n```\n\n" + response["answer"]
            + "\n\nCommand/version matches do not establish requirement correctness. "
            "Vendor compilation and hardware validation require separate evidence.\n",
            encoding="utf-8",
        )
        return path
