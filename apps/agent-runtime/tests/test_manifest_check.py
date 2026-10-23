"""清单一致性检查的正反例：既抓得住真实遗漏，也不误伤已经通过的交付。"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from app import cli
from app.contracts import AssistantTurn, Message, ToolCall
from app.demos.catalog import ARCHIVE
from app.demos.firmware import FirmwareProject
from app.demos.manifest_check import check_demo_manifest
from app.demos.project import DemoProject
from app.providers.mock import MockProvider
from app.tools import ToolRegistry
from app.workspace_tools import workspace_prompt, workspace_tools

REAL_DELIVERY = Path(__file__).resolve().parents[1] / "output" / "autonomous_module_20261022"


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


@pytest.fixture
def demo_project(tmp_path):
    """独立 demo 目录：生成物在 `modules/` 下，不是 `agent_demo/`。"""

    fixture = Path(__file__).parent / "fixtures"
    examples = tmp_path / "examples"
    (examples / ARCHIVE).parent.mkdir(parents=True)
    shutil.copyfile(fixture / "demo_sources/M9P6362_reviewed.zip", examples / ARCHIVE)
    root = tmp_path / "standalone"
    root.mkdir()
    return DemoProject(root, examples)


def state_of(root: Path) -> dict:
    return json.loads((root / "demo-project.json").read_text(encoding="utf-8"))


def write_state(root: Path, state: dict) -> None:
    (root / "demo-project.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def detail_of(report: dict, check: str) -> str:
    return next(item for item in report["items"] if item["check"] == check)["detail"]


def evidence_of(report: dict, check: str) -> list[str]:
    return next(item for item in report["items"] if item["check"] == check)["evidence"]


# --- 正面：流水线自己生成的工程必须全过 --------------------------------------


def test_generated_project_passes_every_check(project):
    project.generate("pwm0", {"duty_percent": 30})

    report = check_demo_manifest(project.directory)

    assert report["failed"] == []
    assert report["ok"] is True
    assert report["modules"] == ["pwm0"]


@pytest.mark.parametrize("module", ["pwm0", "timer0", "timer1"])
def test_supported_standalone_generators_pass_manifest_check(demo_project, module):
    demo_project.generate(module, {})

    assert check_demo_manifest(demo_project.directory)["failed"] == []


@pytest.mark.parametrize("suffix", [".o", ".obj", ".exe", ".log"])
def test_host_build_outputs_do_not_require_registration(project, suffix):
    project.generate("pwm0", {"duty_percent": 30})
    output = project.directory / "agent_demo/tools" / f"pwm_host{suffix}"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"host verification output")

    assert check_demo_manifest(project.directory)["failed"] == []


def test_comment_mentioning_a_register_value_is_not_a_mismatch(project):
    project.generate("pwm0", {"duty_percent": 30})
    module = project.directory / "agent_demo" / "modules" / "demo_pwm0.c"
    text = module.read_text(encoding="utf-8")
    module.write_text(
        text.replace(
            "void Demo_Pwm0_Start(void)",
            "/* PWM00D = 0x7F; 换算是这样算的，不是真实赋值 */\nvoid Demo_Pwm0_Start(void)",
        ),
        encoding="utf-8",
    )

    assert check_demo_manifest(project.directory)["failed"] == []


def test_registered_host_tool_does_not_need_project_file_entry(project):
    """`agent_demo/tools/` 用主机编译器跑，不编进器件工程。"""

    project.generate("pwm0", {"duty_percent": 30})
    root = project.directory
    tool = root / "agent_demo" / "tools" / "pwm0_host_test.c"
    tool.parent.mkdir(parents=True)
    tool.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    state = state_of(root)
    state["managed_files"].append("agent_demo/tools/pwm0_host_test.c")
    write_state(root, state)

    assert check_demo_manifest(root)["failed"] == []


def test_standalone_demo_module_directory_is_covered(demo_project):
    """登记范围由清单自身决定；独立 demo 的模块目录同样要查。"""

    demo_project.generate("pwm0", {"duty_percent": 30})
    root = demo_project.directory
    assert check_demo_manifest(root)["failed"] == []

    (root / "modules" / "demo_extra.c").write_text(
        "void Demo_Extra_Init(void)\n{\n}\n", encoding="utf-8"
    )

    report = check_demo_manifest(root)

    assert report["failed"] == ["generated_files_registered"]
    assert "modules/demo_extra.c" in evidence_of(report, "generated_files_registered")


# --- 反面：2026-10-22 真实交付里的那种遗漏 ----------------------------------


def test_hand_added_module_file_must_be_registered(project):
    """复刻真实遗漏：USART 模块已落盘，清单仍只登记 PWM。"""

    project.generate("pwm0", {"duty_percent": 30})
    root = project.directory
    extra = root / "agent_demo" / "modules" / "demo_usart0.c"
    extra.write_text("void Demo_Usart0_Init(void)\n{\n}\n", encoding="utf-8")

    report = check_demo_manifest(root)

    assert report["failed"] == ["generated_files_registered"]
    assert "agent_demo/modules/demo_usart0.c" in evidence_of(report, "generated_files_registered")


def test_registered_module_missing_from_project_file_is_reported(project):
    project.generate("pwm0", {"duty_percent": 30})
    root = project.directory
    extra = root / "agent_demo" / "modules" / "demo_extra.c"
    extra.write_text("void Demo_Extra_Init(void)\n{\n}\n", encoding="utf-8")
    state = state_of(root)
    state["managed_files"].append("agent_demo/modules/demo_extra.c")
    write_state(root, state)

    report = check_demo_manifest(root)

    assert report["failed"] == ["project_file_listing"]
    assert "agent_demo/modules/demo_extra.c" in detail_of(report, "project_file_listing")


def test_code_drift_from_recorded_actual_is_reported(project):
    """代码改了、清单没改：`#define DEMO_PWM0_PERIOD 249u` 改成 250u。"""

    project.generate("pwm0", {"duty_percent": 30})
    header = project.directory / "agent_demo" / "modules" / "demo_pwm0.h"
    text = header.read_text(encoding="utf-8")
    drifted, count = re.subn(r"(#define DEMO_PWM0_PERIOD )\d+u", r"\g<1>250u", text)
    assert count == 1
    header.write_text(drifted, encoding="utf-8")

    report = check_demo_manifest(project.directory)

    assert report["failed"] == ["actual_matches_code"]
    assert "pwm0.period_register=249" in detail_of(report, "actual_matches_code")


def test_dispatch_layer_must_call_declared_entry(project):
    project.generate("pwm0", {"duty_percent": 30})
    app = project.directory / "agent_demo" / "demo_app.c"
    text = app.read_text(encoding="utf-8")
    removed, count = re.subn(r"[ \t]*Demo_Pwm0_Init\(\);\r?\n", "", text)
    assert count == 1
    app.write_text(removed, encoding="utf-8")

    report = check_demo_manifest(project.directory)

    assert report["failed"] == ["module_wired"]
    assert "Demo_Pwm0_Init" in detail_of(report, "module_wired")


def test_declared_entry_missing_from_code_is_reported(project):
    project.generate("pwm0", {"duty_percent": 30})
    root = project.directory
    state = state_of(root)
    state["modules"][0]["poll"] = "Demo_Pwm0_Poll"
    write_state(root, state)

    report = check_demo_manifest(root)

    assert set(report["failed"]) == {"module_symbols_present", "module_wired"}
    assert "pwm0.poll=Demo_Pwm0_Poll" in detail_of(report, "module_symbols_present")


def test_declaration_and_call_cannot_replace_an_entry_definition(project):
    project.generate("pwm0", {"duty_percent": 30})
    source = project.directory / "agent_demo/modules/demo_pwm0.c"
    text = source.read_text(encoding="utf-8")
    assert text.count("void Demo_Pwm0_Init(void)") == 1
    source.write_text(text.replace("void Demo_Pwm0_Init(void)",
                                   "void Renamed_Pwm0_Init(void)"), encoding="utf-8")

    report = check_demo_manifest(project.directory)

    assert report["failed"] == ["module_symbols_present"]
    assert "Demo_Pwm0_Init" in detail_of(report, "module_symbols_present")


def test_dangling_project_file_entry_is_reported(project):
    project.generate("pwm0", {"duty_percent": 30})
    prj = project.directory / "M9P6362_TC1.prj"
    text = prj.read_text(encoding="utf-8")
    text = text.replace("[Project Files]\n", "[Project Files]\nFile98=agent_demo/modules/gone.c\n")
    text = re.sub(
        r"(\[File Count\]\r?\nCount=)(\d+)",
        lambda match: f"{match.group(1)}{int(match.group(2)) + 1}",
        text,
        count=1,
    )
    prj.write_text(text, encoding="utf-8")

    report = check_demo_manifest(project.directory)

    assert report["failed"] == ["project_file_listing"]
    assert "gone.c" in detail_of(report, "project_file_listing")
    assert "File Count" not in detail_of(report, "project_file_listing")


def test_two_modules_claiming_one_peripheral_are_reported(project):
    project.generate("pwm0", {"duty_percent": 30})
    root = project.directory
    state = state_of(root)
    state["modules"].append({"id": "pwm1", "resources": ["TC0", "PWM1"], "init": "Demo_Pwm0_Init"})
    write_state(root, state)

    report = check_demo_manifest(root)

    assert report["failed"] == ["resources_unique"]
    assert "TC0" in detail_of(report, "resources_unique")


def test_missing_or_broken_manifest_is_reported(tmp_path):
    missing = check_demo_manifest(tmp_path / "empty")
    assert missing["failed"] == ["manifest_present"]

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "demo-project.json").write_text("{ not json", encoding="utf-8")
    report = check_demo_manifest(broken)
    assert report["failed"] == ["manifest_present"]
    assert "JSON" in detail_of(report, "manifest_present")


def test_workspace_prompt_names_existing_manifest(project):
    project.generate("pwm0", {"duty_percent": 30})

    prompt = workspace_prompt(project.directory)

    assert "工程清单 demo-project.json" in prompt
    assert "['pwm0']" in prompt
    assert "check_project_manifest" in prompt


def test_workspace_prompt_is_quiet_without_manifest(project):
    plain = project.directory / "plain"
    plain.mkdir()

    assert "工程清单" not in workspace_prompt(plain)


# --- 真实交付目录（不在仓库内则跳过） ----------------------------------------


@pytest.mark.skipif(not REAL_DELIVERY.is_dir(), reason="真实交付目录未随仓库提供")
def test_real_delivery_omission_is_detected():
    report = check_demo_manifest(REAL_DELIVERY / "before_feedback")

    assert report["failed"] == ["generated_files_registered"]
    assert "agent_demo/modules/demo_usart0.c" in evidence_of(report, "generated_files_registered")


@pytest.mark.skipif(not REAL_DELIVERY.is_dir(), reason="真实交付目录未随仓库提供")
def test_real_delivery_fixed_project_passes():
    assert check_demo_manifest(REAL_DELIVERY / "customer-firmware")["failed"] == []


# --- 模型可达性与交付门禁 ----------------------------------------------------


def test_manifest_tool_follows_manifest_presence(project):
    project.generate("pwm0", {"duty_percent": 30})
    names = {tool.name for tool in workspace_tools(project.directory)}
    assert "check_project_manifest" in names

    plain = project.directory / "plain"
    plain.mkdir()
    assert "check_project_manifest" not in {tool.name for tool in workspace_tools(plain)}
    assert "check_project_manifest" in {
        tool.name for tool in workspace_tools(plain, manifest=True)
    }


async def test_manifest_tool_reports_failure_so_the_model_must_fix(project):
    project.generate("pwm0", {"duty_percent": 30})
    extra = project.directory / "agent_demo" / "modules" / "demo_usart0.c"
    extra.write_text("void Demo_Usart0_Init(void)\n{\n}\n", encoding="utf-8")

    registry = ToolRegistry()
    for tool in workspace_tools(project.directory):
        registry.register(tool)
    outcome = await registry.execute(ToolCall("check", "check_project_manifest", {}))

    assert outcome.is_error
    payload = json.loads(outcome.content)
    assert payload["failed"] == ["generated_files_registered"]
    assert "重新调用本工具" in payload["next_step"]


async def test_cli_exit_code_makes_inconsistent_delivery_a_failure(project, monkeypatch, capsys):
    project.generate("pwm0", {"duty_percent": 30})
    # The new file is produced during this run, rather than being an inherited problem.
    monkeypatch.setattr(
        cli, "OpenAiCompatibleProvider",
        lambda **kwargs: MockProvider(
            [AssistantTurn(Message("assistant", "", tool_calls=[ToolCall(
                "write-usart", "write_file", {
                    "path": "agent_demo/modules/demo_usart0.c",
                    "content": "void Demo_Usart0_Init(void)\n{\n}\n",
                },
            )]), "tool_calls"),
             AssistantTurn(Message("assistant", "已同步配置记录"), "stop")]
        ),
    )

    code = await cli.run_once(cli.build_parser().parse_args([
        "加上串口", "--base-url", "http://offline.invalid", "--json", "--no-memory",
        "--workspace", str(project.directory),
    ]))

    assert code == cli.EXIT_MANIFEST
    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest"]["failed"] == ["generated_files_registered"]


async def test_workspace_cli_exposes_manifest_checker(project, monkeypatch, capsys):
    project.generate("pwm0", {"duty_percent": 30})
    provider = MockProvider([
        AssistantTurn(Message("assistant", "", tool_calls=[
            ToolCall("check", "check_project_manifest", {}),
        ]), "tool_calls"),
        AssistantTurn(Message("assistant", "已核对清单"), "stop"),
    ])
    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **kwargs: provider)

    code = await cli.run_once(cli.build_parser().parse_args([
        "只核对清单", "--base-url", "http://offline.invalid", "--json", "--no-memory",
        "--workspace", str(project.directory),
    ]))

    assert "check_project_manifest" in {tool.name for tool in provider.requests[0].tools}
    observation = next(m for m in provider.requests[1].messages if m.tool_call_id == "check")
    assert json.loads(observation.content)["ok"] is True
    assert code == cli.EXIT_OK
    capsys.readouterr()


async def test_read_only_question_is_not_blocked_by_old_manifest_problem(
    project, monkeypatch, capsys,
):
    project.generate("pwm0", {"duty_percent": 30})
    extra = project.directory / "agent_demo/modules/demo_usart0.c"
    extra.write_text("void Demo_Usart0_Init(void)\n{\n}\n", encoding="utf-8")
    before = {p: p.read_bytes() for p in project.directory.rglob("*") if p.is_file()}
    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **kwargs: MockProvider([
        AssistantTurn(Message("assistant", "", tool_calls=[ToolCall(
            "read", "read_file", {"path": "demo-project.json"},
        )]), "tool_calls"),
        AssistantTurn(Message("assistant", "只解释工程，未修改文件。"), "stop"),
    ]))

    code = await cli.run_once(cli.build_parser().parse_args([
        "只解释工程，不修改文件", "--base-url", "http://offline.invalid", "--json",
        "--no-memory", "--workspace", str(project.directory),
        "--session", str(project.directory / "session.jsonl"),
    ]))

    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest"]["failed"] == ["generated_files_registered"]
    assert code == cli.EXIT_OK
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("max_steps", [None, 1])
async def test_command_writes_are_checked_without_masking_budget_status(
    project, monkeypatch, capsys, max_steps,
):
    project.generate("pwm0", {"duty_percent": 30})
    provider = MockProvider([
        AssistantTurn(Message("assistant", "", tool_calls=[ToolCall(
            "write-from-command", "run_command", {"command": ["python", "-c",
                "from pathlib import Path; "
                "Path('agent_demo/modules/extra.c').write_text('void Extra_Init(void) {}')"]},
        )]), "tool_calls"),
        AssistantTurn(Message("assistant", "已完成"), "stop"),
    ])
    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **kwargs: provider)
    argv = ["新增模块", "--base-url", "http://offline.invalid", "--json", "--no-memory",
            "--workspace", str(project.directory)]
    if max_steps is not None:
        argv.extend(["--max-steps", str(max_steps)])

    code = await cli.run_once(cli.build_parser().parse_args(argv))

    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest"]["failed"] == ["generated_files_registered"]
    assert code == (cli.EXIT_MANIFEST if max_steps is None else cli.EXIT_BUDGET)


async def test_cli_exit_code_stays_ok_for_consistent_delivery(project, monkeypatch, capsys):
    project.generate("pwm0", {"duty_percent": 30})
    monkeypatch.setattr(
        cli, "OpenAiCompatibleProvider",
        lambda **kwargs: MockProvider([AssistantTurn(Message("assistant", "已完成"), "stop")]),
    )

    code = await cli.run_once(cli.build_parser().parse_args([
        "加上串口", "--base-url", "http://offline.invalid", "--json", "--no-memory",
        "--workspace", str(project.directory),
    ]))

    assert code == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest"]["ok"] is True
