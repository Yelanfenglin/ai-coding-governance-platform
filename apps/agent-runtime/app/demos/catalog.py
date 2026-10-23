"""已审查的例程适配，不把整个下载目录宣称为已支持的生成范围。"""

from __future__ import annotations

import re
from pathlib import Path
from zipfile import ZipFile

PRODUCT = "M9P6362"
SPEC_VERSION = "V1.16"
DEMO_VERSION = "V1.07"
ARCHIVE = "M9P6362/M9P6362_Demo_V1.07.zip"
ARCHIVE_PREFIX = "M9P6362_Demo_V1.07"
SOURCE_URL = "http://www.masses-chip.com/3090.html"
DOWNLOAD_URL = "http://www.masses-chip.com/u/2026/03/M9P6362_Demo_V1.07.zip"
CLOCK_HZ = 32_000_000


class DemoError(ValueError):
    """无法生成的具体原因（参数、来源、资源冲突或已有人工修改）。"""


MODULES = {
    "pwm0": {
        "id": "pwm0",
        "name": "PWM0 输出",
        "peripheral": "PWM0",
        "description": "32MHz、128预分频、8位PWM0，C0–C3。",
        "defaults": {"frequency_hz": 1000, "duty_percent": 50, "channels": [0]},
        "properties": {
            "frequency_hz": {
                "type": "number",
                "minimum": 977,
                "maximum": 125000,
                "description": "目标频率 (Hz)",
            },
            "duty_percent": {
                "type": "number",
                "minimum": 1,
                "maximum": 99,
                "description": "高电平占空比 (%)",
            },
            "channels": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {"type": "integer", "minimum": 0, "maximum": 3},
                "description": "通道0–3，对应C0–C3",
            },
        },
        "sections": [
            {"section": "7.2", "page": 56},
            {"section": "7.6–7.7", "page": 58},
            {"section": "7.8", "page": 59},
            {"section": "6.9.3", "page": 54},
        ],
    },
    "timer0": {
        "id": "timer0",
        "name": "TC0 定时事件",
        "peripheral": "TC0",
        "description": "8位定时事件，可选GPIO指示；与PWM0共用TC0，不能重复占用。",
        "defaults": {"interval_us": 1000, "toggle_pin": "C1"},
        "properties": {
            "interval_us": {
                "type": "integer",
                "minimum": 4,
                "maximum": 1024,
                "description": "事件间隔 (µs)，4µs计数分辨率",
            },
            "toggle_pin": {
                "type": "string",
                "enum": ["none", "C1", "A0"],
                "description": "指示端口，none仅提供事件接口",
            },
        },
        "sections": [{"section": "7.1–7.3", "page": 55}, {"section": "5.3", "page": 40}],
    },
    "timer1": {
        "id": "timer1",
        "name": "TC1 定时事件",
        "peripheral": "TC1",
        "description": "16位定时事件，可选GPIO指示；可与PWM0在不同引脚上组合。",
        "defaults": {"interval_us": 1000, "toggle_pin": "C1"},
        "properties": {
            "interval_us": {
                "type": "integer",
                "minimum": 1,
                "maximum": 65536,
                "description": "事件间隔 (µs)，1µs计数分辨率",
            },
            "toggle_pin": {
                "type": "string",
                "enum": ["none", "C1", "A0"],
                "description": "指示端口，none仅提供事件接口",
            },
        },
        "sections": [
            {"section": "8.1", "page": 60},
            {"section": "8.2", "page": 61},
            {"section": "8.5–8.6", "page": 62},
            {"section": "5.3", "page": 40},
        ],
    },
}

def catalog(examples_root: Path) -> dict:
    return {
        "product": PRODUCT,
        "spec_version": SPEC_VERSION,
        "demo_version": DEMO_VERSION,
        "clock_hz": CLOCK_HZ,
        "instruction_divider": 8,
        "source_url": SOURCE_URL,
        "archive": ARCHIVE,
        "source_available": (examples_root / ARCHIVE).is_file(),
        "coverage": "reviewed_generation_adapters_only",
        "coverage_note": (
            "这里只列已审查的生成适配器，不是芯片完整外设清单。"
            "未列出的外设不能说成芯片没有；其它硬件能力需查规格书。"
        ),
        "modules": list(MODULES.values()),
        "resource_names": ["TC0", "TC1", "PWM0", "IO:C0", "IO:C1", "IO:C2", "IO:C3", "IO:A0"],
    }


def source_text(examples_root: Path, peripheral: str, filename: str) -> str:
    archive = examples_root / ARCHIVE
    if not archive.is_file():
        raise DemoError(f"未找到官方例程包 {ARCHIVE}，请配置 --examples-root / DEMO_EXAMPLES_ROOT")
    member = f"{ARCHIVE_PREFIX}/{PRODUCT}_{peripheral}/{filename}"
    with ZipFile(archive) as package:
        raw = package.read(member)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("gb18030")
    return text.replace("\r\r\n", "\n").replace("\r\n", "\n")


def source_function(source: str, name: str) -> str:
    """切出已审查例程中的函数；不执行包内脚本。"""
    found = re.search(rf"void\s+{re.escape(name)}\s*\(\s*void\s*\)\s*\{{", source)
    if found is None:
        raise DemoError(f"官方源码缺少已审查函数 {name}")
    depth = 1
    for index in range(found.end(), len(source)):
        depth += (source[index] == "{") - (source[index] == "}")
        if depth == 0:
            return source[found.start() : index + 1] + "\n"
    raise DemoError(f"官方源码函数未闭合：{name}")
