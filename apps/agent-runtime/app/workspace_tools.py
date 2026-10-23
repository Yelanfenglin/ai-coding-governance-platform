"""本地编程工具；模型决定步骤，工具执行文件操作与命令。

工具分工参考 Pi 的 read/edit/write/bash，具体 Python 适配见验收报告。
文件路径相对所选工程；命令是本机进程，不提供操作系统沙箱。
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import signal
import subprocess
import sys
from contextlib import suppress
from itertools import islice
from pathlib import Path
from zipfile import ZipFile

from .contracts import ToolResult
from .demos.firmware import SourceFile
from .demos.manifest_check import STATE_FILE as MANIFEST_STATE_FILE
from .demos.manifest_check import check_demo_manifest
from .tools import FunctionTool


def local_path(root: Path, name: str) -> Path:
    target = (root / name.replace("\\", "/")).resolve()
    if not target.is_relative_to(root):
        raise ValueError("文件路径应位于所选目录内")
    return target


def result(value: dict, *, error: bool = False) -> ToolResult:
    return ToolResult(json.dumps(value, ensure_ascii=False), is_error=error)


def line_view(text: str, args: dict) -> dict:
    lines = text.splitlines()
    start, limit = args.get("start_line", 1), args.get("max_lines", 160)
    selected = lines[start - 1 : start - 1 + limit]
    return {
        "total_lines": len(lines),
        "content": "\n".join(f"{i}: {line}" for i, line in enumerate(selected, start)),
        "has_more": start - 1 + len(selected) < len(lines),
    }


def schema(properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties,
            "required": required or [], "additionalProperties": False}


PAGING = {
    "start_line": {"type": "integer", "minimum": 1},
    "max_lines": {"type": "integer", "minimum": 1, "maximum": 400},
}
PATH = {"type": "string", "minLength": 1}


async def kill_command(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    if os.name == "nt":
        killer = await asyncio.create_subprocess_exec(
            "taskkill", "/PID", str(process.pid), "/T", "/F",
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        await killer.wait()
    else:
        with suppress(ProcessLookupError):  # 自然退出可与超时同时发生。
            os.killpg(process.pid, signal.SIGKILL)
    await process.wait()


async def execute_command(root: Path, args: dict) -> ToolResult:
    command = list(args["command"])
    if command[0] == "python":
        command[0] = sys.executable
    root.mkdir(parents=True, exist_ok=True)
    options = (
        {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt" else {"start_new_session": True}
    )
    process = await asyncio.create_subprocess_exec(
        *command, cwd=root, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONIOENCODING": "utf-8"},
        **options,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), args.get("timeout", 60))
    except TimeoutError:
        await kill_command(process)
        return result({"error": "command timed out", "timeout": args.get("timeout", 60)},
                      error=True)
    except asyncio.CancelledError:
        await kill_command(process)
        raise
    output = stdout.decode("utf-8", errors="replace")
    return result({"exit_code": process.returncode, "output": output[-20000:],
                   "truncated": len(output) > 20000}, error=process.returncode != 0)


def workspace_tools(directory: str | Path, *, manifest: bool | None = None) -> list[FunctionTool]:
    """`manifest` 为 None 时按磁盘上是否已有清单自动决定，模块模式显式要求常驻。"""

    root = Path(directory).resolve()

    def list_files(args, context):
        base = local_path(root, args.get("path", "."))
        pattern = args.get("glob", "*")
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ValueError("glob应相对所选目录")
        entries = list(islice(base.glob(pattern), 201))
        return result({
            "directory": str(root), "platform": sys.platform,
            "python": sys.executable,
            "files": [
                {"path": str(path.relative_to(root)), "is_directory": path.is_dir()}
                for path in sorted(entries[:200])
            ],
            "truncated": len(entries) > 200,
        })

    def read_file(args, context):
        source = SourceFile.load(local_path(root, args["path"]))
        return result({"path": args["path"], "encoding": source.encoding,
                       **line_view(source.text, args)})

    def write_file(args, context):
        path = local_path(root, args["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(args["content"].encode("utf-8"))
        return result({"written": args["path"]})

    def edit_file(args, context):
        path = local_path(root, args["path"])
        source = SourceFile.load(path)
        original = source.text.replace("\r\n", "\n")
        old = args["old_text"].replace("\r\n", "\n")
        new = args["new_text"].replace("\r\n", "\n")
        if original.count(old) != 1:
            raise ValueError("old_text必须唯一匹配当前文件；请读取原文或提供更多上下文")
        updated = original.replace(old, new, 1)
        path.write_bytes(updated.replace("\n", source.newline).encode(source.encoding))
        diff = "".join(difflib.unified_diff(
            original.splitlines(keepends=True), updated.splitlines(keepends=True),
            fromfile=args["path"], tofile=args["path"],
        ))
        return result({"edited": args["path"], "diff": diff})

    async def run_command(args, context):
        return await execute_command(root, args)

    tools = [
        FunctionTool("list_files", "列出本地工程文件和运行环境；glob可用**/*.c等模式。",
                     schema({"path": PATH, "glob": {"type": "string"}}), list_files,
                     execution_mode="parallel"),
        FunctionTool("read_file", "按行读取所选本地工程的文本文件，不限已生成文件。",
                     schema({"path": PATH, **PAGING}, ["path"]), read_file,
                     execution_mode="parallel"),
        FunctionTool("edit_file", "唯一精确匹配old_text再替换，返回差异；保留编码与换行。",
                     schema({"path": PATH, "old_text": PATH, "new_text": {"type": "string"}},
                            ["path", "old_text", "new_text"]), edit_file),
        FunctionTool("write_file", "创建文件或完整重写为UTF-8；局部修改请用edit_file。",
                     schema({"path": PATH, "content": {"type": "string"}}, ["path", "content"]),
                     write_file),
        FunctionTool("run_command", "在本地工程运行argv命令，返回退出码与输出。"
                     "例如[\"python\",\"-c\",\"print(1+1)\"]或编译/验证脚本。"
                     "python指当前环境解释器，不用模型口算。命令有本机用户权限，无OS沙箱。",
                     schema({"command": {"type": "array", "minItems": 1,
                                         "items": {"type": "string"}},
                             "timeout": {"type": "number", "exclusiveMinimum": 0}}, ["command"]),
                     run_command),
    ]
    if manifest or (manifest is None and (root / MANIFEST_STATE_FILE).is_file()):
        tools.append(manifest_tool(root))
    return tools


def manifest_tool(root: Path) -> FunctionTool:
    """清单专用只读核对，不替代工程行为验证。"""

    async def check(args, context):
        report = await asyncio.to_thread(check_demo_manifest, root)
        payload = {
            **report,
            "next_step": (
                "清单检查通过，结合已执行的工程验证交付结果"
                if report["ok"]
                else "先按 detail 修正代码或 demo-project.json（含 managed_files 与 .prj），"
                "再重新调用本工具直到 ok 为 true；不要只说已同步"
            ),
        }
        return result(payload, error=not report["ok"])

    return FunctionTool(
        "check_project_manifest",
        "核对 demo-project.json 与磁盘代码是否自洽：生成文件是否登记、清单实际值是否与"
        "源码寄存器赋值一致、模块是否接进分发层、资源是否重复、项目文件表是否完整。"
        "改完代码或清单一律调用一次；返回 ok=false 时必须修正后重跑。",
        schema({}),
        check,
        execution_mode="parallel",
    )


def example_tools(directory: str | Path) -> list[FunctionTool]:
    """动态读取已下载的官方ZIP，不把某几个型号/外设编进发现逻辑。"""
    root = Path(directory).resolve()

    def list_examples(args, context):
        archive = args.get("archive")
        if archive:
            with ZipFile(local_path(root, archive)) as package:
                paths = [info.filename for info in package.infolist() if not info.is_dir()]
        else:
            paths = [path.relative_to(root).as_posix() for path in root.rglob("*.zip")]
        query = args.get("query", "").casefold()
        paths = sorted(path for path in paths if query in path.casefold())
        return result({
            "scope": "members" if archive else "archives",
            "archive": archive, "paths": paths[:100], "truncated": len(paths) > 100,
            "next_step": (
                "用read_official_example读取需要的member"
                if archive else
                "当前query只筛选ZIP路径；先按型号定位archive，再指定archive查包内外设"
            ),
        })

    def read_example(args, context):
        with ZipFile(local_path(root, args["archive"])) as package:
            raw = package.read(args["member"])
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("gb18030")
        return result({"archive": args["archive"], "member": args["member"],
                       **line_view(text.replace("\r\r\n", "\n"), args)})

    return [
        FunctionTool("list_official_examples", "查已下载的官方ZIP；指定archive后列包内文件。"
                     "query按路径筛选；未适配模板的外设也可查阅。",
                     schema({"archive": PATH, "query": {"type": "string"}}), list_examples,
                     execution_mode="parallel"),
        FunctionTool("read_official_example", "读取指定官方ZIP内源码，不解压或修改原始包。",
                     schema({"archive": PATH, "member": PATH, **PAGING}, ["archive", "member"]),
                     read_example, execution_mode="parallel"),
    ]


WORKSPACE_PROMPT = """## 本地嵌入式开发
你可以读、编辑、创建工程文件并运行命令，由你根据需求规划和调整步骤。
先定位相关工程入口、源文件、现有功能和约束；独立的读取/检索可并行，不必通读所有文件。
已知位置的独立读取、同一文件中可确定的多处编辑可在同一轮提交；有依赖的操作按结果推进。
规格检索用于补充明确缺失的事实。先定位，再用read_context展开选中的证据；
已有资料足以实施时就修改和验证，不反复扩大检索。遇到真实阻塞才停下说明。
例程发现分两层：先找型号的ZIP，再指定archive筛选包内外设和源码。
涉及频率、波特率、误差、位宽上限或方案比较表中的新数值，实际用run_command运行计算，
再依据运行结果作答，不以口算代替验证。公式、字段及适用范围仍须对照资料，
代码能运行不等于公式正确。优先复用已有测试和计算脚本，新增检查只补现有验证的缺口；
需要后续参数复算时保留可复跑脚本。
寄存器配置先对照原文位定义解码实际写入值；计算结果相同不代表字段含义正确。
优先执行工程已有或宿主指定的独立验收；模型自写检查不能代替独立判据。
检查失败时先根据失败项、预期值和实际值定位；诊断不足时再展开完整日志或检查器源码。
不要将生成模板目录当成能力上限。模板覆盖时可复用；覆盖之外可读取例程并编写代码。
只分析时不要改业务文件；用户要求实现时实际修改并验证，不停在方案或伪代码。
改动后检查初始化/中断顺序、共享资源、接口挂接和工程文件条目；执行可用测试和验证命令。
缺少外部SDK或目标编译器时，可继续有原文依据的源码实现和可执行检查，
不必为重查例程已使用的寄存器而持续寻找SDK；不要伪造SDK到交付工程。
保留现有功能与工程选项。源码是当前事实，生成清单是历史记录：新增或修改生成文件后，
把新文件写进 managed_files，并更新受影响的 actual、入口字段和工程文件表；
不能把旧清单直接当当前配置。
交付前必须调用 check_project_manifest（工程存在 demo-project.json 时可用）。
它返回 ok=false 就按 detail 修正代码或清单再重跑，直到 ok=true；
不要用"已同步"之类的说明替代这次检查结果。
直接编辑生成文件后继续按定制代码维护，避免模板重建覆盖。
确认后简洁交付修改位置、实际配置、依据和已执行验证，区分已知事实与待确认方案。
本机命令不是OS沙箱；围绕所选工程工作。没有厂商编译器或板卡时如实区分验证层级。
"""


def manifest_prompt(directory: str | Path) -> str:
    """清单存在时把"有哪些模块"直接放到开场，避免模型靠猜或整份读入。"""

    state_path = Path(directory) / MANIFEST_STATE_FILE
    if not state_path.is_file():
        return ""
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return f"\n工程清单 {MANIFEST_STATE_FILE} 存在但读不出来；先修复它再核对。"
    modules = [str(item.get("id")) for item in state.get("modules") or [] if isinstance(item, dict)]
    managed = [name for name in state.get("managed_files") or [] if isinstance(name, str)]
    return (
        f"\n工程清单 {MANIFEST_STATE_FILE}：产品 {state.get('product')} "
        f"{state.get('spec_version')}；模块 {modules or '无'}；已登记文件 {len(managed)} 个。"
        "新增或修改生成文件后同步 managed_files 和受影响的 actual/入口字段，"
        "并调用 check_project_manifest 直到 ok=true。"
    )


def workspace_prompt(directory: str | Path) -> str:
    return (
        WORKSPACE_PROMPT + manifest_prompt(directory) + f"\n当前目录：{Path(directory).resolve()}\n"
        f"平台：{sys.platform}；Python：{sys.executable}\n"
        "run_command接收argv，不自动经过shell；不要假设grep/sed等命令已安装。"
    )
