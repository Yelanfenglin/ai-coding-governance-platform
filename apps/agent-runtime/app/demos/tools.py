"""Agent 在指定本地工程中生成模块；不经过上传、下载或独立 Web 服务。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from ..contracts import ToolResult
from ..tools import FunctionTool
from .catalog import MODULES, DemoError
from .firmware import FirmwareProject
from .project import DemoProject


def skill_prompt() -> str:
    return Path(__file__).with_name("SKILL.md").read_text(encoding="utf-8")


def demo_tools(project: DemoProject) -> list[FunctionTool]:
    async def list_modules(args, context):
        return ToolResult(json.dumps(project.catalog(), ensure_ascii=False))

    async def inspect(args, context):
        state = await asyncio.to_thread(project.read)
        return ToolResult(json.dumps(state, ensure_ascii=False))

    async def generate(args, context):
        try:
            state = await asyncio.to_thread(
                project.generate,
                args["module_id"],
                args.get("parameters", {}),
                reserved_resources=args.get("reserved_resources"),
            )
        except DemoError as error:
            return ToolResult(str(error), is_error=True)
        return ToolResult(json.dumps(state, ensure_ascii=False))

    async def read_file(args, context):
        text = await asyncio.to_thread(project.file, args["path"])
        start = args.get("start_line", 1)
        limit = args.get("max_lines", 160)
        lines = text.splitlines()
        return ToolResult(json.dumps({
            "path": args["path"], "total_lines": len(lines),
            "content": "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines)
                                   if start - 1 <= i < start - 1 + limit),
        }, ensure_ascii=False))

    async def search(args, context):
        state = await asyncio.to_thread(project.read)
        matches = []
        for item in state["files"]:
            text = await asyncio.to_thread(project.file, item["path"])
            for index, line in enumerate(text.splitlines(), 1):
                if args["text"] in line:
                    matches.append({"path": item["path"], "line": index, "text": line})
                    if len(matches) >= 30:
                        return ToolResult(json.dumps({"matches": matches, "truncated": True},
                                                     ensure_ascii=False))
        return ToolResult(json.dumps({"matches": matches, "truncated": False}, ensure_ascii=False))

    empty = {"type": "object", "properties": {}, "additionalProperties": False}
    tools = [
        FunctionTool(
            "list_demo_modules",
            "列出可实际生成的模块、参数默认值与官方源码来源。",
            empty,
            list_modules,
            execution_mode="parallel",
        ),
        FunctionTool(
            "inspect_demo_project",
            "读取当前工程模块、资源和文件，追加前先检查。",
            empty,
            inspect,
            execution_mode="parallel",
        ),
        FunctionTool(
            "generate_demo_module",
            "在当前本地工程实际生成/更新一个模块，合并共享初始化与中断入口。"
            "参数更新只需传变更字段，未提及的配置和其它模块保留。"
            "支持参数由list_demo_modules给出。reserved_resources省略时保留已有清单；"
            "只有用户明确调整外部资源时才修改该字段。冲突时不写文件。",
            {
                "type": "object",
                "properties": {
                    "module_id": {"type": "string", "enum": list(MODULES)},
                    "parameters": {"type": "object"},
                    "reserved_resources": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["module_id"],
                "additionalProperties": False,
            },
            generate,
        ),
        FunctionTool(
            "read_project_file", "读取所选工程中的文件，带行号；先看inspect中的文件列表。",
            {"type": "object", "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "max_lines": {"type": "integer", "minimum": 1, "maximum": 400},
            }, "required": ["path"], "additionalProperties": False},
            read_file, execution_mode="parallel",
        ),
        FunctionTool(
            "search_project", "在工程清单文件中按原文搜索，最多返回30处及行号。",
            {"type": "object", "properties": {"text": {"type": "string", "minLength": 1}},
             "required": ["text"], "additionalProperties": False},
            search, execution_mode="parallel",
        ),
    ]
    if isinstance(project, FirmwareProject):
        async def preview(args, context):
            try:
                result = await asyncio.to_thread(
                    project.generate, args["module_id"], args.get("parameters", {}),
                    reserved_resources=args.get("reserved_resources"), preview=True,
                )
            except DemoError as error:
                return ToolResult(str(error), is_error=True)
            return ToolResult(json.dumps(result, ensure_ascii=False))

        tools.append(FunctionTool(
            "preview_demo_integration", "预览接入现有工程的差异和资源检查，不写文件。",
            tools[2].input_schema, preview, execution_mode="parallel",
        ))
    return tools
