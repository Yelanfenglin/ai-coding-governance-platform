"""Explicit host checks using the existing command executor and tool feedback.

This is a local delivery extension, not a DSH feature or a sandbox. Configuration
is loaded by the host once; the model cannot replace its argv through tool args.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .delivery import file_hashes, project_files, revision
from .observability import failure, observe
from .tools import FunctionTool, validate_arguments
from .workspace_tools import execute_command, result, schema


class ProjectVerification:
    def __init__(self, config: Path, workspace: Path, *, exclude: tuple[Path, ...] = ()):
        config = config.resolve()
        value = json.loads(config.read_text(encoding="utf-8"))
        invalid = validate_arguments(schema({
            "command": {"type": "array", "minItems": 1, "items": {"type": "string"}},
            "cwd": {"type": "string"},
            "timeout": {"type": "number", "exclusiveMinimum": 0},
        }, ["command"]), value)
        if invalid:
            raise ValueError(f"verification config: {invalid}")

        def expand(text: str) -> str:
            return (text.replace("{python}", sys.executable)
                    .replace("{workspace}", str(workspace))
                    .replace("{config_dir}", str(config.parent)))

        self.workspace = workspace
        self.exclude = exclude
        self.config = str(config)
        self.command = [expand(part) for part in value["command"]]
        self.cwd = (workspace / expand(value.get("cwd", "."))).resolve()
        if not self.cwd.is_dir():
            raise ValueError(f"verification cwd is not a directory: {self.cwd}")
        self.timeout = value.get("timeout", 60)
        self.runs: list[dict] = []

    def current_revision(self) -> str:
        return revision(file_hashes(project_files(self.workspace, exclude=self.exclude)))

    async def run(self, *, phase: str) -> dict:
        with observe("project.verify", **{"verification.phase": phase}) as span:
            row = await self._run(phase=phase)
            span.set_attribute("verification.passed", row["passed"])
            if not row["passed"]:
                failure(span, "verification_failed")
            return row

    async def _run(self, *, phase: str) -> dict:
        before = self.current_revision()
        try:
            outcome = await execute_command(self.cwd, {
                "command": self.command, "timeout": self.timeout,
            })
            payload = json.loads(outcome.content)
        except OSError as exc:
            payload = {"error": str(exc)}
        after = self.current_revision()
        row = {
            "phase": phase, "command": self.command, "cwd": str(self.cwd),
            "timeout": self.timeout, "input_revision": before, "output_revision": after,
            **payload,
            "passed": payload.get("exit_code") == 0 and before == after,
        }
        if before != after:
            row["source_error"] = "verification changed project files; recheck the final revision"
        self.runs.append(row)
        return row

    def tool(self) -> FunctionTool:
        async def check(args, context):
            row = await self.run(phase="development")
            return result({**row, "check_call_id": context.tool_call_id}, error=not row["passed"])

        return FunctionTool(
            "verify_project",
            "运行宿主指定的工程验收。修改后调用；失败时按输出修复再调用。"
            "命令与工作目录由宿主固定，不能通过参数修改。"
            + json.dumps({"command": self.command, "cwd": str(self.cwd)}, ensure_ascii=False),
            schema({}), check,
        )

    def report(self) -> dict:
        final = next((row for row in reversed(self.runs) if row["phase"] == "delivery"), None)
        matches = bool(final and final["input_revision"] == final["output_revision"]
                       == self.current_revision())
        return {
            "status": ("passed" if final["passed"] and matches else "failed")
                      if final else "unverified",
            "scope": "host_configured_command", "config": self.config,
            "matches_delivery": matches, "runs": self.runs,
        }
