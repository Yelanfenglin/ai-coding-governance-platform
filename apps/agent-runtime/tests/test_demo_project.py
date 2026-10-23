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
from app.demo_cli import main
from app.demos import DemoError, DemoProject
from app.demos.catalog import ARCHIVE
from app.providers.mock import MockProvider


@pytest.fixture
def examples(tmp_path: Path) -> Path:
    root = tmp_path / "examples"
    archive = root / ARCHIVE
    archive.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).parent / "fixtures/demo_sources/M9P6362_reviewed.zip", archive)
    return root


def contents(path: Path) -> dict[str, bytes]:
    return {
        str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*") if file.is_file()
    }


def test_official_source_becomes_parameterized_module_and_vendor_project(tmp_path, examples):
    project = DemoProject(tmp_path / "generated", examples)
    state = project.generate("pwm0", {"frequency_hz": 2000, "duty_percent": 30, "channels": [0, 2]})
    code = project.file("modules/demo_pwm0.c")
    assert "Demo_Pwm0_Init" in code and "TC0CL = DEMO_PWM0_PERIOD" in code
    assert "ANSC2 = 0" in code and "ANSC1 = 0" not in code
    assert "Delay_50ms" not in code and "COPYRIGHT 2025 Masses-Chip" in code
    assert "PWM02OE = 1" in code and "T0CR2 = 0" in code
    (module,) = state["modules"]
    assert module["actual"]["period_register"] == 124
    assert module["actual"]["duty_percent"] == 30.4
    assert state["checks"]["target_compile"] == "not_run"
    assert state["spec_version"] == "V1.16" and state["demo_version"] == "V1.07"
    parser = configparser.ConfigParser()
    parser.read_string(project.file("demo.prj"))
    assert parser.getint("Settings", "ConfigBitValue3") == 64, "保留PWM0工程选项字"
    assert parser.getint("File Count", "Count") == len(parser["Project Files"])


def test_sequential_modules_share_one_main_and_dispatch_interrupts(tmp_path, examples):
    project = DemoProject(tmp_path / "combo", examples)
    project.generate("pwm0")
    state = project.generate("timer1", {"interval_us": 10000})
    assert [module["id"] for module in state["modules"]] == ["pwm0", "timer1"]
    main_code = project.file("main.C")
    assert main_code.count("void main(") == main_code.count("int_h(void)") == 1
    aggregate = project.file("demo_app.c")
    assert "Demo_Pwm0_Init();" in aggregate and "Demo_Timer1_Init();" in aggregate
    assert "Demo_Timer1_Interrupt();" in aggregate and "Demo_Timer1_Poll();" in aggregate
    assert "GIEH = 1" not in project.file("modules/demo_timer1.c")
    assert "TC1PRL = (unsigned char)(DEMO_TIMER1_PERIOD & 0xFFu)" in project.file(
        "modules/demo_timer1.c"
    )
    assert "#define DEMO_TIMER1_PERIOD 9999u" in project.file("modules/demo_timer1.h")
    assert "static volatile unsigned char demo_timer1_pending" in project.file(
        "modules/demo_timer1.c"
    )
    assert "ab8 flag;" not in project.file("modules/demo_timer1.h")
    assert (project.directory / "modules/demo_pwm0.c").is_file()
    assert (project.directory / "modules/demo_timer1.c").is_file()
    assert not (project.directory / "zc.h").exists(), "厂商SDK不伪造到交付目录"


@pytest.mark.parametrize(
    "first,first_parameters,second",
    [
        ("pwm0", {}, "timer0"),
        ("pwm0", {"channels": [1]}, "timer1"),
    ],
)
def test_resource_conflicts_do_not_change_any_file(
    tmp_path, examples, first, first_parameters, second
):
    project = DemoProject(tmp_path / "conflict", examples)
    project.generate(first, first_parameters)
    before = contents(project.directory)
    with pytest.raises(DemoError, match="资源冲突"):
        project.generate(second)
    assert contents(project.directory) == before


def test_reserved_resources_and_parameter_updates_preserve_existing_modules(tmp_path, examples):
    project = DemoProject(tmp_path / "reserved", examples)
    with pytest.raises(DemoError, match="已有工程保留"):
        project.generate("pwm0", reserved_resources=["TC0"])
    assert not project.directory.exists()
    project.generate("timer1", {"toggle_pin": "none"}, reserved_resources=["TC0"])
    with pytest.raises(DemoError, match="资源冲突"):
        project.generate("pwm0")
    state = project.generate("timer1", {"interval_us": 5000, "toggle_pin": "none"})
    assert len(state["modules"]) == 1
    assert state["reserved_resources"] == ["TC0"]
    assert "Demo_Timer1_Poll();" not in project.file("demo_app.c")


def test_hand_edited_source_is_not_overwritten_by_next_module(tmp_path, examples):
    project = DemoProject(tmp_path / "manual", examples)
    project.generate("pwm0")
    edited = project.directory / "modules/demo_pwm0.c"
    edited.write_text(
        edited.read_text(encoding="utf-8") + "\n/* engineer change */\n", encoding="utf-8"
    )
    before = contents(project.directory)
    with pytest.raises(DemoError, match="人工修改"):
        project.generate("timer1")
    assert contents(project.directory) == before


@pytest.mark.parametrize(
    "module,parameters",
    [
        ("pwm0", {"channels": [0, 0]}),
        ("pwm0", {"duty_percent": 0}),
        ("timer1", {"interval_us": True}),
        ("timer0", {"unknown_register": 1}),
    ],
)
def test_invalid_configuration_is_rejected_before_creating_project(
    tmp_path, examples, module, parameters
):
    project = DemoProject(tmp_path / "invalid", examples)
    with pytest.raises(DemoError):
        project.generate(module, parameters)
    assert not project.directory.exists()


def test_direct_cli_generates_and_reads_local_files(tmp_path, examples, capsys):
    path = tmp_path / "cli-project"
    args = ["--examples-root", str(examples), "--project", str(path)]
    assert main(["generate", *args, "--module", "timer1"]) == 0
    assert json.loads(capsys.readouterr().out)["modules"][0]["id"] == "timer1"
    assert main(["show", *args]) == 0
    assert json.loads(capsys.readouterr().out)["exists"] is True
    assert (path / "modules/demo_timer1.c").is_file()


def test_partial_parameter_update_keeps_other_configuration_and_modules(tmp_path, examples):
    project = DemoProject(tmp_path / "update", examples)
    project.generate("pwm0", {"frequency_hz": 2000, "duty_percent": 30, "channels": [2]})
    project.generate("timer1", {"interval_us": 10000})
    timer_before = project.file("modules/demo_timer1.c")
    state = project.generate("pwm0", {"duty_percent": 40})
    assert state["modules"][0]["parameters"] == {
        "frequency_hz": 2000, "duty_percent": 40, "channels": [2],
    }
    assert project.file("modules/demo_timer1.c") == timer_before
    assert "Demo_Timer1_Init();" in project.file("demo_app.c")


async def test_agent_cli_can_generate_two_real_modules(tmp_path, examples, monkeypatch, capsys):
    provider = MockProvider(
        [
            AssistantTurn(
                Message(
                    "assistant",
                    tool_calls=[
                        ToolCall("read", "inspect_demo_project", {}),
                    ],
                ),
                "tool_calls",
            ),
            AssistantTurn(
                Message(
                    "assistant",
                    tool_calls=[
                        ToolCall("pwm", "generate_demo_module", {"module_id": "pwm0"}),
                        ToolCall("timer", "generate_demo_module", {"module_id": "timer1"}),
                    ],
                ),
                "tool_calls",
            ),
            AssistantTurn(Message("assistant", "工程已生成"), "stop"),
        ]
    )
    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **kwargs: provider)
    target = tmp_path / "agent-project"
    args = cli.build_parser().parse_args(
        [
            "生成PWM和定时模块",
            "--base-url",
            "http://offline.invalid",
            "--json",
            "--no-memory",
            "--demo-project",
            str(target),
            "--examples-root",
            str(examples),
        ]
    )
    assert await cli.run_once(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["tool_calls"] == [
        "inspect_demo_project",
        "generate_demo_module",
        "generate_demo_module",
    ]
    assert len(DemoProject(target, examples).read()["modules"]) == 2
    assert "模块化 Demo 生成" in provider.requests[0].messages[0].content
    assert "Demo_Timer1_Init" in (target / "demo_app.c").read_text(encoding="utf-8")


@pytest.mark.parametrize("first_module", ["pwm0", "timer0"])
def test_host_c_link_and_module_interfaces_with_register_stubs(tmp_path, examples, first_module):
    gcc = shutil.which("gcc")
    if gcc is None:
        pytest.skip("host GCC is unavailable; target compiler is not substituted")
    project = DemoProject(tmp_path / "host", examples)
    project.generate(
        first_module,
        {"duty_percent": 30} if first_module == "pwm0" else {"toggle_pin": "A0"},
    )
    project.generate("timer1", {"interval_us": 10000})
    registers = set(
        [
            "ANSA",
            "PUA",
            "PDA",
            "IOA",
            "OEA",
            "ANSB",
            "PUB",
            "PDB",
            "IOB",
            "OEB",
            "ANSC",
            "PUC",
            "PDC",
            "IOC",
            "OEC",
            "ANSD",
            "PUD",
            "PDD",
            "IOD",
            "OED",
            "GIEH",
            "GIEL",
            "MPPWM0",
            "T0CR",
            "T0CR2",
            "TC0CL",
            "TC0CH",
            "TC0EN",
            "T1CR",
            "TC1CH",
            "TC1CL",
            "TC1PRL",
            "TC1PRH",
            "TC1EN",
            "TC1IE",
            "TC1IF",
            "TC1IP",
            "IOC1",
            "OSCM",
            "TC0IE",
            "TC0IF",
            "TC0IP",
            "ANSA0",
            "OEA0",
            "IOA0",
        ]
    )
    for i in range(4):
        registers.update([f"ANSC{i}", f"OEC{i}", f"PWM0{i}CR", f"PWM0{i}D", f"PWM0{i}OE"])
    (project.directory / "zc.h").write_text(
        "#define interrupt\n#define high_priority\n#define low_priority\n"
        "#define CLRWDT() ((void)0)\n"
        + "\n".join(f"extern volatile unsigned char {reg};" for reg in registers),
        encoding="utf-8",
    )
    (project.directory / "register_stubs.c").write_text(
        "\n".join(f"volatile unsigned char {reg};" for reg in registers),
        encoding="utf-8",
    )
    runner = project.directory / "host_check.c"
    first_header = f'#include "modules/demo_{first_module}.h"\n'
    first_assertions = (
        """    assert(TC0CL == 249 && PWM00D == 75 && PWM00OE == 1);
    Demo_Pwm0_Stop(); assert(PWM00OE == 0);
    Demo_Pwm0_Start(); Demo_Pwm0_SetDuty(20); assert(PWM00D == 20);
"""
        if first_module == "pwm0"
        else """    assert(TC0CL == 249 && TC0IE == 1 && OEA0 == 1);
    TC0IF = 1; TC1IF = 1; Demo_InterruptHigh();
    assert(TC0IF == 0 && TC1IF == 0);
    assert(Demo_Timer0_TakeTick() == 1); assert(Demo_Timer0_TakeTick() == 0);
    assert(Demo_Timer1_TakeTick() == 1); assert(Demo_Timer1_TakeTick() == 0);
"""
    )
    runner.write_text(
        first_header + """#include <assert.h>
#include "demo_platform.h"
#include "demo_app.h"
#include "modules/demo_timer1.h"
int main(void) {
    Demo_PlatformInit(); Demo_ModulesInit(); assert(GIEH == 0); Demo_EnableInterrupts();
""" + first_assertions + """
    assert(TC1PRL == 0x0F && TC1PRH == 0x27 && GIEH == 1);
    TC1IF = 1; Demo_InterruptHigh(); assert(TC1IF == 0);
    assert(Demo_Timer1_TakeTick() == 1); assert(Demo_Timer1_TakeTick() == 0);
    TC1IF = 1; Demo_InterruptHigh(); Demo_Poll(); assert(IOC1 != 0);
    return 0;
}
""",
        encoding="utf-8",
    )
    common = [gcc, "-std=c11", "-Wall", "-Werror", "-Wno-main", "-I", str(project.directory)]
    # MSYS2 的编译器子进程需要同一工具链的 DLL，不能优先加载其它软件的旧版 DLL。
    compiler_env = {**os.environ, "PATH": str(Path(gcc).parent) + os.pathsep + os.environ["PATH"]}
    result = subprocess.run(
        [*common, "-x", "c", "-fsyntax-only", str(project.directory / "main.C")],
        capture_output=True,
        text=True,
        env=compiler_env,
    )
    assert result.returncode == 0, result.stderr
    executable = project.directory / "host_check.exe"
    sources = [
        project.directory / "demo_platform.c",
        project.directory / "demo_app.c",
        *sorted((project.directory / "modules").glob("*.c")),
        project.directory / "register_stubs.c",
        runner,
    ]
    result = subprocess.run(
        [*common, *map(str, sources), "-o", str(executable)],
        capture_output=True,
        text=True,
        env=compiler_env,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True, env=compiler_env)
    assert result.returncode == 0, result.stderr
