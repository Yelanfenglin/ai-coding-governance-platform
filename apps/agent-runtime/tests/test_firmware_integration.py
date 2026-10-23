from __future__ import annotations

import configparser
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from app import cli
from app.contracts import AssistantTurn, Message, ToolCall
from app.demos import DemoError
from app.demos.catalog import ARCHIVE
from app.demos.firmware import FirmwareProject, code_mask, functions
from app.demos.tools import demo_tools
from app.providers.mock import MockProvider
from app.tools import ToolRegistry


@pytest.fixture
def project(tmp_path):
    fixture = Path(__file__).parent / "fixtures"
    examples = tmp_path / "examples"
    archive = examples / ARCHIVE
    archive.parent.mkdir(parents=True)
    shutil.copyfile(fixture / "demo_sources/M9P6362_reviewed.zip", archive)
    root = tmp_path / "firmware"
    shutil.copytree(fixture / "firmware_tc1", root)
    return FirmwareProject(root, examples)


def snapshot(root):
    return {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


def test_inspection_reads_real_project_resources_and_ignores_comments(project):
    path = project.directory / "main.C"
    path.write_text(
        path.read_text(encoding="utf-8") + "\n// TC0EN = 1; fake { }\n", encoding="utf-8"
    )
    state = project.read()
    resources = {row["resource"]: row["evidence"] for row in state["detected_resources"]}
    assert set(resources) == {"TC1", "IO:C1"}
    assert resources["TC1"][0]["file"] == "TC1_Work.c"
    assert all(row["symbol"] != "TC1_INIT" for row in resources["TC1"])
    assert state["clock"]["status"] == "matches_reviewed_profile"
    assert state["issues"] == []
    assert {entry["name"] for entry in state["entries"]} == {"main", "int_h", "int_l"}


def test_preview_does_not_write_and_apply_preserves_original_logic(project):
    before = snapshot(project.directory)
    result = project.generate("pwm0", {"duty_percent": 30}, preview=True)
    assert result["preview"] is True
    assert snapshot(project.directory) == before
    assert any(
        row["path"] == "main.C" and "Demo_ModulesInit();" in row["diff"]
        for row in result["changes"]
    )
    result = project.generate("pwm0", {"duty_percent": 30})
    assert (Path(result["backup"]) / "main.C").read_bytes() == before["main.C"]
    code = (project.directory / "main.C").read_bytes().decode("utf-8")
    restored = "".join(
        line
        for line in code.splitlines(keepends=True)
        if "agent_demo/demo_app.h" not in line and "Demo_ModulesInit();" not in line
    )
    assert restored == before["main.C"].decode()
    for name in ("TC1_Work.c", "COMMON.H", "MASSESCHIP_DEFINE.H"):
        assert (project.directory / name).read_bytes() == before[name]
    original_config = configparser.ConfigParser()
    original_config.read_string(before["M9P6362_TC1.prj"].decode())
    updated = configparser.ConfigParser()
    updated.read(project.directory / "M9P6362_TC1.prj", encoding="utf-8")
    assert dict(updated["Settings"]) == dict(original_config["Settings"])
    assert updated.getint("File Count", "Count") == len(updated["Project Files"])
    assert "agent_demo\\modules\\demo_pwm0.c" in updated["Project Files"].values()


def test_parameter_update_preserves_later_business_edits_and_does_not_repeat_hooks(project):
    project.generate("pwm0", {"duty_percent": 30})
    main = project.directory / "main.C"
    main.write_text(
        main.read_text(encoding="utf-8").replace(
            "TC1_Work();",
            "TC1_Work(); /* engineer business update */",
        ),
        encoding="utf-8",
    )
    before_main = main.read_bytes()
    result = project.generate("pwm0", {"duty_percent": 40})
    assert main.read_bytes() == before_main
    assert [row["path"] for row in result["changes"]] == ["agent_demo/modules/demo_pwm0.h"]
    assert main.read_text(encoding="utf-8").count("Demo_ModulesInit();") == 1


@pytest.mark.parametrize("module,parameters", [("pwm0", {"channels": [1]}), ("timer1", {})])
def test_existing_register_and_pin_conflicts_have_locations_and_write_nothing(
    project, module, parameters
):
    before = snapshot(project.directory)
    with pytest.raises(DemoError, match=r"TC1_Work.c:\d+"):
        project.generate(module, parameters)
    assert snapshot(project.directory) == before


def test_timer_integration_keeps_existing_irq_and_adds_poll_once(project):
    project.generate("timer0", {"toggle_pin": "A0"})
    project.generate("timer0", {"interval_us": 500})
    code = (project.directory / "main.C").read_text(encoding="utf-8")
    assert code.count("void main(") == code.count("int_h(void)") == code.count("int_l(void)") == 1
    for call in ("Demo_ModulesInit();", "Demo_Poll();", "Demo_InterruptHigh();"):
        assert code.count(call) == 1
    assert "TC1_Interrupt();" in code and "TC1_Work();" in code


def test_new_handlers_run_before_existing_continue_or_early_return(project):
    main = project.directory / "main.C"
    text = (
        main.read_text(encoding="utf-8")
        .replace(
            "TC1_Work();",
            "if (IOC1) continue;\n        TC1_Work();",
        )
        .replace("TC1_Interrupt();", "if (TC1IF) return;\n    TC1_Interrupt();")
    )
    main.write_text(text, encoding="utf-8")
    project.generate("timer0", {"toggle_pin": "A0"})
    changed = main.read_text(encoding="utf-8")
    assert changed.index("Demo_Poll();") < changed.index("continue;")
    assert changed.index("Demo_InterruptHigh();") < changed.index("return;")


def test_edited_generated_module_is_preserved(project):
    project.generate("pwm0")
    path = project.directory / "agent_demo/modules/demo_pwm0.c"
    path.write_text(path.read_text(encoding="utf-8") + "\n/* engineer change */", encoding="utf-8")
    before = snapshot(project.directory)
    with pytest.raises(DemoError, match="人工修改"):
        project.generate("pwm0", {"duty_percent": 40})
    assert snapshot(project.directory) == before


def test_gb18030_and_crlf_of_existing_files_are_preserved(project):
    main = project.directory / "main.C"
    old = main.read_text(encoding="utf-8").replace("\n", "\r\n")
    main.write_bytes(old.encode("gb18030"))
    project.generate("pwm0")
    updated = main.read_bytes().decode("gb18030")
    lines = updated.splitlines(keepends=True)
    assert "\n" not in updated.replace("\r\n", "")
    restored = "".join(
        line
        for line in lines
        if "agent_demo/demo_app.h" not in line and "Demo_ModulesInit();" not in line
    )
    assert restored == old


def test_unconfirmed_clock_cannot_be_overwritten_by_default_configuration(project):
    path = project.directory / "main.C"
    path.write_text(
        path.read_text(encoding="utf-8").replace("OSCM = 0x00", "OSCM = 0x01"), encoding="utf-8"
    )
    before = snapshot(project.directory)
    assert project.read()["clock"]["status"] == "unconfirmed"
    with pytest.raises(DemoError, match="时钟"):
        project.generate("pwm0")
    assert snapshot(project.directory) == before


def test_lexical_structure_ignores_braces_in_comments_and_string_literals():
    text = 'void main(void) { /* } */ const char *s = "{"; while(1) { } }'
    (main,) = functions(code_mask(text))
    assert main.end == len(text) - 1


async def test_read_search_preview_and_cli_use_real_project_files(project, monkeypatch, capsys):
    registry = ToolRegistry()
    for tool in demo_tools(project):
        registry.register(tool)
    found = await registry.execute(ToolCall("find", "search_project", {"text": "TC1_Interrupt"}))
    assert any(row["path"] == "main.C" for row in json.loads(found.content)["matches"])
    read = await registry.execute(ToolCall("read", "read_project_file", {"path": "main.C"}))
    assert "TC1_Work();" in read.content
    escaped = await registry.execute(ToolCall("escape", "read_project_file", {"path": "../x"}))
    assert escaped.is_error
    provider = MockProvider(
        [
            AssistantTurn(
                Message(
                    "assistant",
                    tool_calls=[
                        ToolCall(
                            "preview",
                            "preview_demo_integration",
                            {"module_id": "pwm0"},
                        )
                    ],
                ),
                "tool_calls",
            ),
            AssistantTurn(
                Message(
                    "assistant",
                    tool_calls=[
                        ToolCall(
                            "apply",
                            "generate_demo_module",
                            {"module_id": "pwm0"},
                        )
                    ],
                ),
                "tool_calls",
            ),
            AssistantTurn(Message("assistant", "接入完成"), "stop"),
        ]
    )
    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **kwargs: provider)
    args = cli.build_parser().parse_args(
        [
            "在已有工程加PWM",
            "--base-url",
            "http://offline.invalid",
            "--json",
            "--no-memory",
            "--firmware-project",
            str(project.directory),
            "--examples-root",
            str(project.examples_root),
        ]
    )
    assert await cli.run_once(args) == 0
    assert json.loads(capsys.readouterr().out)["tool_calls"] == [
        "preview_demo_integration",
        "generate_demo_module",
    ]
    assert (project.directory / "agent_demo/modules/demo_pwm0.c").is_file()


@pytest.mark.parametrize(
    "module,parameters",
    [
        ("pwm0", {"duty_percent": 30}),
        ("timer0", {"toggle_pin": "A0"}),
    ],
)
def test_host_compile_runs_original_main_and_both_existing_and_added_behavior(
    project, module, parameters
):
    gcc = shutil.which("gcc")
    if gcc is None:
        pytest.skip("host GCC unavailable")
    project.generate(module, parameters)
    registers = {"OSCM", "GIEH", "GIEL", "MPPWM0", "T0CR", "T0CR2", "T1CR", "IOA0", "IOC1"}
    for port in "ABCD":
        registers.update(prefix + port for prefix in ("ANS", "PU", "PD", "IO", "OE"))
    for index in range(2):
        registers.update(f"TC{index}{suffix}" for suffix in ("CL", "CH", "EN", "IE", "IF", "IP"))
    registers.update(("TC1PRL", "TC1PRH", "ANSA0", "OEA0"))
    for index in range(4):
        registers.update(f"{prefix}{index}" for prefix in ("ANSC", "OEC"))
        registers.update(f"PWM0{index}{suffix}" for suffix in ("CR", "D", "OE"))
    root = project.directory
    (root / "zc.h").write_text(
        "#define interrupt\n#define high_priority\n#define low_priority\n"
        "void test_watchdog(void);\n#define CLRWDT() test_watchdog()\n"
        + "\n".join(f"extern volatile unsigned char {register};" for register in registers),
        encoding="utf-8",
    )
    (root / "registers.c").write_text(
        "\n".join(f"volatile unsigned char {register};" for register in registers),
        encoding="utf-8",
    )
    expected = (
        "assert(TC0CL == 249 && PWM00D == 75 && PWM00OE == 1);"
        if module == "pwm0"
        else "assert(TC0CL == 249 && IOA0 != 0);"
    )
    (root / "check.c").write_text(
        '#include <assert.h>\n#include <setjmp.h>\n#include "COMMON.H"\n'
        '#include "agent_demo/demo_app.h"\n'
        "void firmware_main(void); void int_h(void);\nstatic jmp_buf end; static int passes;\n"
        "void test_watchdog(void) { if (++passes == 4) longjmp(end, 1);"
        " TC1IF = 1; TC0IF = 1; int_h(); }\n"
        "int main(void) { if (setjmp(end) == 0) firmware_main();\n"
        "assert(passes == 4 && TC1PRL == 0xE7 && TC1PRH == 0x03);\n"
        "assert(IOC1 != 0 && b_TC1_Flag == 0 && GIEH == 1);\n"
        + expected
        + "\nGIEH = 0; Demo_ModulesInit(); assert(GIEH == 0); return 0; }\n",
        encoding="utf-8",
    )
    env = {**os.environ, "PATH": str(Path(gcc).parent) + os.pathsep + os.environ["PATH"]}
    # 官方COMMON.H以tentative definition声明共享flag；host用-fcommon模拟此链接约定。
    common = [gcc, "-std=c11", "-fcommon", "-Wall", "-Werror", "-Wno-main", "-I", str(root)]
    main_obj = root / "firmware.o"
    result = subprocess.run(
        [
            *common,
            "-Dmain=firmware_main",
            "-x",
            "c",
            "-c",
            str(root / "main.C"),
            "-o",
            str(main_obj),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    executable = root / "host_check.exe"
    sources = [
        root / "TC1_Work.c",
        *sorted((root / "agent_demo").rglob("*.c")),
        root / "registers.c",
        root / "check.c",
    ]
    result = subprocess.run(
        [*common, str(main_obj), *map(str, sources), "-o", str(executable)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
