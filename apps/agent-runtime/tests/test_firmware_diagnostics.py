"""Real C checks explain defects without changing the native-manual predicates."""
from __future__ import annotations

import configparser
import json
import os
import shutil
from pathlib import Path

import pytest

from evaluation.checks.adc import build_checker

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def adc_project(tmp_path):
    seed = ROOT / "evaluation/fixtures/pwm_usart"
    project = tmp_path / "project"
    shutil.copytree(seed, project)
    modules = project / "agent_demo/modules"
    for source in (ROOT / "tests/fixtures/adc_module").glob("*"):
        shutil.copy2(source, modules / source.name)
    app = project / "agent_demo/demo_app.c"
    text = app.read_text(encoding="utf-8")
    text = '#include "modules/demo_adc0.h"\n' + text
    text = text.replace("Demo_Usart0_Init();", "Demo_Usart0_Init();\n    Demo_Adc0_Init();")
    text = text.replace("Demo_Usart0_Poll();", "Demo_Usart0_Poll();\n    Demo_Adc0_Poll();")
    app.write_text(text, encoding="utf-8")
    prj = project / "M9P6362_TC1.prj"
    config = configparser.ConfigParser()
    config.read(prj, encoding="utf-8")
    for name in ("demo_adc0.h", "demo_adc0.c"):
        config["Project Files"][f"File{len(config['Project Files'])}"] = (
            f"agent_demo\\modules\\{name}"
        )
    config["File Count"]["Count"] = str(len(config["Project Files"]))
    with prj.open("w", encoding="utf-8") as stream:
        config.write(stream)
    return project, seed


@pytest.mark.parametrize(
    ("value", "diagnostic"),
    [
        ("0xAFu", None),
        ("0xCFu", "ADC.Init.ADFM: expected=1 actual=0"),
        ("0xEFu", "ADC.Init.ADS: expected=0 actual=1"),
    ],
)
def test_adc_register_diagnostics(adc_project, tmp_path, value, diagnostic):
    compiler = os.environ.get("CC") or shutil.which("gcc")
    if not compiler:
        pytest.skip("set CC to a host GCC executable")
    project, seed = adc_project
    header = project / "agent_demo/modules/demo_adc0.h"
    text = header.read_text(encoding="utf-8")
    header.write_text(text.replace("VALUE 0xAFu", f"VALUE {value}"), encoding="utf-8")
    out = tmp_path / "checks"
    status = build_checker()(workspace=project, seed=seed, out=out,
                             duty_register=60, compiler=compiler)
    report = json.loads((out / "independent_checks.json").read_text(encoding="utf-8"))
    if diagnostic is None:
        assert status == 0
        assert report["failures"] == []
        assert all(item["exit"] == 0 for item in report["checks"].values())
    else:
        assert status == 1
        assert report["checks"]["adc_init"]["exit"] == 1
        assert diagnostic in report["checks"]["adc_init"]["stderr"]
        assert any(diagnostic in item["detail"] for item in report["failures"])
