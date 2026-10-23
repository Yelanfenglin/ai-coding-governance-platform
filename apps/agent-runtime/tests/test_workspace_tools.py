from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

from app import cli
from app.contracts import AssistantTurn, Message, ToolCall
from app.demos import DemoProject
from app.demos.catalog import ARCHIVE
from app.providers.mock import MockProvider
from app.tools import ToolRegistry
from app.workspace_tools import example_tools, workspace_tools


@pytest.fixture
def workspace(tmp_path):
    registry = ToolRegistry()
    for tool in workspace_tools(tmp_path):
        registry.register(tool)
    return tmp_path, registry


async def call(registry, name, **args):
    return await registry.execute(ToolCall(name, name, args))


@pytest.mark.parametrize("encoding", ["utf-8-sig", "gb18030"])
async def test_read_and_edit_arbitrary_code_preserves_encoding_newlines(workspace, encoding):
    root, registry = workspace
    old = "/* 工程师代码 */\r\nvoid custom_spi_init(void) { divider = 24; }\r\n"
    path = root / "custom_spi.c"
    path.write_bytes(old.encode(encoding))
    read = await call(registry, "read_file", path=path.name, start_line=2, max_lines=1)
    assert not read.is_error and "2: void custom_spi_init" in read.content
    edit = await call(registry, "edit_file", path=path.name,
                      old_text="divider = 24;", new_text="divider = 12;")
    assert not edit.is_error and "-void custom_spi" in json.loads(edit.content)["diff"]
    assert path.read_bytes() == old.replace("divider = 24", "divider = 12").encode(encoding)


async def test_ambiguous_edit_is_an_observation_and_leaves_original_intact(workspace):
    root, registry = workspace
    path = root / "main.c"
    path.write_text("x = 1;\nx = 1;\n", encoding="utf-8")
    before = path.read_bytes()
    edit = await call(registry, "edit_file", path=path.name, old_text="x = 1;", new_text="x = 2;")
    assert edit.is_error and "唯一" in edit.content
    assert path.read_bytes() == before


async def test_write_read_list_and_path_boundary(workspace):
    root, registry = workspace
    write = await call(
        registry, "write_file", path="drivers/uart.c", content="void uart(void) {}\n",
    )
    assert not write.is_error
    listing = await call(registry, "list_files", glob="**/*.c")
    listing = json.loads(listing.content)
    assert listing["files"] == [{"path": str(Path("drivers/uart.c")), "is_directory": False}]
    assert listing["python"] == sys.executable
    escaped = await call(registry, "write_file", path="../outside.c", content="not authorized")
    assert escaped.is_error and not (root.parent / "outside.c").exists()


async def test_command_runs_model_supplied_calculation_and_propagates_exit_code(workspace):
    root, registry = workspace
    # 命令内容是调用方给定的，不调用任何固化的模块求解器。
    code = (
        "from fractions import Fraction; import json, os; "
        "print(json.dumps({'rate': float(Fraction(48000000, 32 * 150)), "
        "'cwd': os.path.basename(os.getcwd())})); "
        "raise SystemExit(4)"
    )
    executed = await call(registry, "run_command", command=["python", "-c", code])
    assert executed.is_error
    data = json.loads(executed.content)
    assert data["exit_code"] == 4
    assert json.loads(data["output"]) == {"rate": 10000.0, "cwd": root.name}


async def test_command_timeout_does_not_leave_a_delayed_write(workspace):
    root, registry = workspace
    executed = await call(registry, "run_command", timeout=0.2, command=[
        "python", "-c",
        "import time; from pathlib import Path; time.sleep(0.8); Path('late').write_text('bad')",
    ])
    assert executed.is_error and "timed out" in executed.content
    await asyncio.sleep(0.85)
    assert not (root / "late").exists()


async def test_command_preserves_empty_and_unicode_arguments(workspace):
    _, registry = workspace
    executed = await call(registry, "run_command", command=[
        "python", "-c", "import sys, json; print(json.dumps(sys.argv[1:], ensure_ascii=False))",
        "", "中文参数 with space",
    ])
    assert not executed.is_error
    assert json.loads(json.loads(executed.content)["output"]) == ["", "中文参数 with space"]


async def test_cancelling_command_cleans_up_its_child_process(workspace):
    root, registry = workspace
    child = (
        "import time; from pathlib import Path; time.sleep(1.2); "
        "Path('child_completed').write_text('bad')"
    )
    parent = (
        "import subprocess, sys, time; from pathlib import Path; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
        "Path('ready').touch(); time.sleep(60)"
    )
    task = asyncio.create_task(call(registry, "run_command", command=["python", "-c", parent]))
    for _ in range(100):
        if (root / "ready").exists():
            break
        await asyncio.sleep(0.02)
    assert (root / "ready").exists()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 3)
    await asyncio.sleep(1.3)
    assert not (root / "child_completed").exists()


async def test_official_examples_discovery_is_not_a_chip_or_module_whitelist(tmp_path):
    root = tmp_path / "official"
    archive = root / "different_chip/new_vendor.zip"
    archive.parent.mkdir(parents=True)
    original = "/* 官方串口例程 */\r\nvoid UART_Init(void) { BAUD = 17; }\r\n"
    with ZipFile(archive, "w") as package:
        package.writestr("Demo/UART_Work.c", original.encode("gb18030"))
    registry = ToolRegistry()
    for tool in example_tools(root):
        registry.register(tool)
    archives = await call(registry, "list_official_examples", query="different_chip")
    assert json.loads(archives.content)["paths"] == ["different_chip/new_vendor.zip"]
    files = await call(registry, "list_official_examples", archive="different_chip/new_vendor.zip",
                       query="uart")
    assert json.loads(files.content)["paths"] == ["Demo/UART_Work.c"]
    read = await call(registry, "read_official_example", archive="different_chip/new_vendor.zip",
                      member="Demo/UART_Work.c", start_line=2)
    assert not read.is_error and "BAUD = 17" in read.content
    assert archive.is_file() and list(root.rglob("*.c")) == [], "只读包，不自动解压"
    missing_archive = await call(registry, "list_official_examples", query="UART")
    missing = json.loads(missing_archive.content)
    assert missing["paths"] == [] and missing["scope"] == "archives"
    assert "ZIP路径" in missing["next_step"], "包外没命中不能误报成没有该外设例程"
    assert json.loads(files.content)["scope"] == "members"


async def test_cli_workspace_can_implement_non_template_code(tmp_path, monkeypatch, capsys):
    turns = [
        [ToolCall("discover", "list_files", {})],
        [ToolCall("create", "write_file", {"path": "spi.py", "content": "divider = 12\n"})],
        [ToolCall("edit", "edit_file", {
            "path": "spi.py", "old_text": "divider = 12", "new_text": "divider = 24",
        })],
        [ToolCall("verify", "run_command", {
            "command": ["python", "-c", "import spi; assert spi.divider == 24; print('verified')"],
        })],
    ]
    provider = MockProvider([
        *[AssistantTurn(Message("assistant", tool_calls=calls), "tool_calls") for calls in turns],
        AssistantTurn(Message("assistant", "已实现并运行验证脚本"), "stop"),
    ])
    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **kwargs: provider)
    args = cli.build_parser().parse_args([
        "编写并验证SPI配置脚本", "--workspace", str(tmp_path), "--product", "CUSTOM_CHIP",
        "--version", "custom-version", "--base-url", "http://offline.invalid",
        "--no-memory", "--no-compaction", "--json",
    ])
    assert await cli.run_once(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["tool_calls"] == ["list_files", "write_file", "edit_file", "run_command"]
    assert (tmp_path / "spi.py").read_text() == "divider = 24\n"
    assert "实际修改并验证" in provider.requests[0].messages[0].content
    assert str(tmp_path) in provider.requests[0].messages[0].content
    assert "verified" in provider.requests[-1].messages[-1].content


def test_old_generated_project_is_still_updateable_with_unchanged_templates(tmp_path):
    fixture = Path(__file__).parent / "fixtures"
    target = tmp_path / "legacy"
    shutil.copytree(fixture / "legacy_module_project", target)
    examples = tmp_path / "examples"
    archive = examples / ARCHIVE
    archive.parent.mkdir(parents=True)
    shutil.copyfile(fixture / "demo_sources/M9P6362_reviewed.zip", archive)
    project = DemoProject(target, examples)
    before = project.file("modules/demo_timer1.c")
    state = project.generate("pwm0", {"duty_percent": 40})
    assert state["modules"][0]["parameters"]["duty_percent"] == 40
    assert project.file("modules/demo_timer1.c") == before
