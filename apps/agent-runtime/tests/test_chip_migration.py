from __future__ import annotations

import configparser
import os
import shutil

import pytest

from evaluation.checks.adc_migration import SEED, check, text


@pytest.fixture
def migrated_project(tmp_path, monkeypatch):
    compiler = os.environ.get("CC") or shutil.which("gcc")
    if not compiler:
        pytest.skip("set CC to a host GCC executable")
    monkeypatch.setenv("CC", compiler)
    project = tmp_path / "project"
    shutil.copytree(SEED, project)
    source = project / "source/agent_demo/modules"
    target = project / "target"
    modules = target / "modules"
    modules.mkdir()
    # A known-good test witness, not a runtime code-generation shortcut.
    code = text(source / "demo_adc0.c").replace("../demo_platform.h", "../COMMON.H")
    (modules / "demo_adc0.c").write_text(code.replace("B7", "B4"), encoding="utf-8")
    header = text(source / "demo_adc0.h").replace("VALUE 0xAFu", "VALUE 0xACu")
    (modules / "demo_adc0.h").write_text(
        header.replace("VALUE 0x70u", "VALUE 0x20u"), encoding="utf-8",
    )
    main = target / "M9F6820_TC1.C"
    code = text(main).replace('#include "COMMON.H"', '#include "COMMON.H"\n'
                             '#include "modules/demo_adc0.h"\n'
                             'volatile unsigned int app_adc_value;')
    code = code.replace("TC1_INIT();", "TC1_INIT();\n    Demo_Adc0_Init();")
    main.write_text(code.replace("TC1_Work();", "TC1_Work();\n    Demo_Adc0_Start();\n"
                                "    Demo_Adc0_Poll();\n    Demo_Adc0_Read(&app_adc_value);"
                                .replace("&app_adc_value", "(unsigned int *)&app_adc_value")),
                    encoding="utf-8")
    prj = target / "M9F6820_TC1.prj"
    config = configparser.ConfigParser()
    config.read_string(text(prj))
    files = config["Project Files"]
    for name in ("modules/demo_adc0.c", "modules/demo_adc0.h"):
        files[f"File{len(files)}"] = name
    config["File Count"]["Count"] = str(len(files))
    with prj.open("w", encoding="utf-8") as stream:
        config.write(stream)
    (target / "MIGRATION.md").write_text("Known-good host test witness.\n", encoding="utf-8")
    return project


def test_same_behavior_contract_accepts_both_chips_and_preserves_target(migrated_project, tmp_path):
    report = check(migrated_project, tmp_path / "checks")
    assert report["passed"], [row for row in report["checks"] if not row["passed"]]
    assert {"source:shared_behavior", "target:shared_behavior",
            "target:old_behavior_and_sampling"} <= {row["id"] for row in report["checks"]}


@pytest.mark.parametrize(("path", "old", "new", "diagnostic"), [
    ("modules/demo_adc0.h", "VALUE 0xACu", "VALUE 0xAFu", "init.channel"),
    ("modules/demo_adc0.h", "VALUE 0x20u", "VALUE 0x70u", "init.divider"),
    ("modules/demo_adc0.c", "g_adc0_state = DEMO_ADC0_IDLE;",
     "g_adc0_state = DEMO_ADC0_READY;", "consumed.read_once"),
    ("M9F6820_TC1.C", "TC1_Interrupt();", "/* missed interrupt forwarding */;", "integration:"),
])
def test_migration_checker_rejects_configuration_and_behavior_regressions(
    migrated_project, tmp_path, path, old, new, diagnostic,
):
    file = migrated_project / "target" / path
    code = text(file)
    assert old in code
    file.write_text(code.replace(old, new), encoding="utf-8")
    report = check(migrated_project, tmp_path / "checks")
    assert not report["passed"]
    assert diagnostic in str([row for row in report["checks"] if not row["passed"]])
