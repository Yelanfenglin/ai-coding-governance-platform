"""把官方外设函数适配成独立接口；共享 main/系统初始化由 project 统一生成。"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from ..tools import validate_arguments
from .catalog import (
    ARCHIVE,
    ARCHIVE_PREFIX,
    CLOCK_HZ,
    DOWNLOAD_URL,
    MODULES,
    PRODUCT,
    SOURCE_URL,
    DemoError,
    source_function,
    source_text,
)


@dataclass
class ModuleBuild:
    files: dict[str, str]
    info: dict


def _assign(code: str, register: str, value: str) -> str:
    code, count = re.subn(rf"\b{register}\s*=\s*[^;]+;", f"{register} = {value};", code)
    if count != 1:
        raise DemoError(f"官方例程的 {register} 赋值与已审查版本不同")
    return code


def _declaration(module_id: str, declarations: str, definitions: str) -> str:
    guard = f"DEMO_{module_id.upper()}_H"
    return (
        f'#ifndef {guard}\n#define {guard}\n\n#include "../demo_platform.h"\n\n'
        f"{definitions}\n{declarations}\n#endif\n"
    )


def build_module(root: Path, module_id: str, parameters: dict) -> ModuleBuild:
    if module_id not in MODULES:
        raise DemoError(f"不支持模块 {module_id}，当前支持 {', '.join(MODULES)}")
    definition = MODULES[module_id]
    parameters = {**definition["defaults"], **parameters}
    error = validate_arguments(
        {
            "type": "object",
            "properties": definition["properties"],
            "additionalProperties": False,
        },
        parameters,
    )
    if error:
        raise DemoError(error)
    peripheral = definition["peripheral"]
    source = source_text(root, peripheral, f"{peripheral}_Work.c")
    if module_id == "pwm0":
        build = _pwm(source, parameters)
    else:
        build = _timer(source, module_id, parameters)
    build.info.update(
        {
            "id": module_id,
            "name": definition["name"],
            "parameters": parameters,
            "source": {
                "archive": ARCHIVE,
                "url": SOURCE_URL,
                "download_url": DOWNLOAD_URL,
                "member": f"{ARCHIVE_PREFIX}/{PRODUCT}_{peripheral}/{peripheral}_Work.c",
            },
            "spec_sections": definition["sections"],
        }
    )
    return build


def _pwm(source: str, parameters: dict) -> ModuleBuild:
    frequency = float(parameters["frequency_hz"])
    channels = [int(channel) for channel in parameters["channels"]]
    if not math.isfinite(frequency) or len(channels) != len(set(channels)):
        raise DemoError("PWM频率必须有限，输出通道不能重复")
    counts = math.floor(CLOCK_HZ / 128 / frequency + 0.5)
    duty = math.floor(counts * float(parameters["duty_percent"]) / 100 + 0.5)
    if not 2 <= counts <= 256 or not 0 < duty < counts:
        raise DemoError("目标超出当前官方8位/128分频例程的有效分辨率，请调整频率或占空比")
    code = source_function(source, "PWM0_INIT").replace("PWM0_INIT", "Demo_Pwm0_Init", 1)
    # 原例程初始化四路并在 Work 中交替开关；模块只初始化所选路，连续输出，不带阻塞延时。
    code = (
        "\n".join(
            line
            for line in code.splitlines()
            if not any(
                re.search(rf"\b(?:ANSC{i}|OEC{i}|PWM0{i}CR|PWM0{i}D)\s*=", line)
                for i in range(4)
                if i not in channels
            )
        )
        + "\n"
    )
    code = _assign(code, "TC0CL", "DEMO_PWM0_PERIOD")
    code = code.replace("T0CR = 0x0F;", "T0CR = 0x0F;\n    T0CR2 = 0;")
    for channel in channels:
        code = _assign(code, f"PWM0{channel}D", "DEMO_PWM0_DUTY")
    start = "\n".join(f"    PWM0{i}OE = 1;" for i in channels)
    stop = "\n".join(f"    PWM0{i}OE = 0;" for i in channels)
    set_duty = "\n".join(f"    PWM0{i}D = ticks;" for i in channels)
    code = code.rstrip()[:-1] + "    Demo_Pwm0_Start();\n}\n"
    code += (
        f"\nvoid Demo_Pwm0_Start(void)\n{{\n{start}\n}}\n"
        f"\nvoid Demo_Pwm0_Stop(void)\n{{\n{stop}\n}}\n"
        f"\nvoid Demo_Pwm0_SetDuty(unsigned char ticks)\n{{\n{set_duty}\n}}\n"
    )
    header = _declaration(
        "pwm0",
        (
            "void Demo_Pwm0_Init(void);\nvoid Demo_Pwm0_Start(void);\n"
            "void Demo_Pwm0_Stop(void);\n"
            "/* ticks: high-level timer ticks, not percent; keep within the configured period. */\n"
            "void Demo_Pwm0_SetDuty(unsigned char ticks);"
        ),
        f"#define DEMO_PWM0_PERIOD {counts - 1}u\n#define DEMO_PWM0_DUTY {duty}u",
    )
    return ModuleBuild(
        _files("pwm0", header, code),
        {
            "resources": ["TC0", "PWM0", *[f"IO:C{i}" for i in channels]],
            "init": "Demo_Pwm0_Init",
            "poll": None,
            "interrupt_high": None,
            "actual": {
                "frequency_hz": CLOCK_HZ / 128 / counts,
                "duty_percent": 100 * duty / counts,
                "period_register": counts - 1,
                "frequency_error_percent": (CLOCK_HZ / 128 / counts / frequency - 1) * 100,
                "duty_error_points": 100 * duty / counts - float(parameters["duty_percent"]),
            },
            "adaptations": [
                "仅初始化所选通道",
                "保留官方C0–C3映射",
                "移除50ms开关/延时",
                "显式复位TC0后分频",
                "按固定预分频换算周期与占空比并报告实际值",
            ],
        },
    )


def _timer(source: str, module_id: str, parameters: dict) -> ModuleBuild:
    index = int(module_id[-1])
    peripheral = f"TC{index}"
    prefix = f"Demo_Timer{index}"
    flag = f"demo_timer{index}_pending"
    resolution_us = 4 if index == 0 else 1
    counts = math.floor(int(parameters["interval_us"]) / resolution_us + 0.5)
    period = counts - 1
    pin = parameters["toggle_pin"]
    code = source_function(source, f"{peripheral}_INIT").replace(
        f"{peripheral}_INIT",
        f"{prefix}_Init",
        1,
    )
    code = re.sub(r"\s*GIEH\s*=\s*1;", "", code)  # 共享初始化完成后统一开放全局中断。
    code = code.replace("{", f"{{\n    {flag} = 0;", 1)
    if index == 0:
        code = _assign(code, "TC0CL", "DEMO_TIMER0_PERIOD")
        code = re.sub(r"(T0CR\s*=\s*0x0F;)", r"\1\n    T0CR2 = 0;", code)
    else:
        code = _assign(code, "TC1PRL", "(unsigned char)(DEMO_TIMER1_PERIOD & 0xFFu)")
        code = _assign(code, "TC1PRH", "(unsigned char)(DEMO_TIMER1_PERIOD >> 8)")
    if pin != "none":
        port, bit = pin
        code = code.replace("{", f"{{\n    ANS{port}{bit} = 0;\n    OE{port}{bit} = 1;", 1)
    interrupt = (
        source_function(source, f"{peripheral}_Interrupt")
        .replace(
            f"{peripheral}_Interrupt",
            f"{prefix}_Interrupt",
            1,
        )
        .replace(f"b_{peripheral}_Flag", flag)
    )
    poll = ""
    if pin != "none":
        poll = (
            source_function(source, f"{peripheral}_Work")
            .replace(
                f"{peripheral}_Work",
                f"{prefix}_Poll",
                1,
            )
            .replace(f"b_{peripheral}_Flag", flag)
            .replace("IOC1", f"IO{pin}")
        )
    take = (
        f"unsigned char {prefix}_TakeTick(void)\n{{\n    unsigned char pending;\n"
        f"    pending = {flag};\n    {flag} = 0;\n    return pending;\n}}\n"
    )
    body = f"static volatile unsigned char {flag};\n\n{code}\n{interrupt}\n{poll}\n{take}"
    header = _declaration(
        module_id,
        (
            f"void {prefix}_Init(void);\nvoid {prefix}_Interrupt(void);\n"
            f"unsigned char {prefix}_TakeTick(void);\n"
            "/* TakeTick coalesces pending interrupts, as in the vendor's flag-based example. */\n"
            + (f"void {prefix}_Poll(void);" if pin != "none" else "")
        ),
        f"#define DEMO_TIMER{index}_PERIOD {period}u",
    )
    return ModuleBuild(
        _files(module_id, header, body),
        {
            "resources": [peripheral, *([f"IO:{pin}"] if pin != "none" else [])],
            "init": f"{prefix}_Init",
            "poll": f"{prefix}_Poll" if pin != "none" else None,
            "interrupt_high": f"{prefix}_Interrupt",
            "actual": {
                "interval_us": counts * resolution_us,
                "period_register": period,
                "interval_error_us": counts * resolution_us - int(parameters["interval_us"]),
            },
            "adaptations": [
                "flag改为模块内volatile变量",
                "全局中断由共享入口统一开放",
                "指示端口按参数配置",
                "提供独立中断/轮询/取事件接口",
            ],
        },
    )


def _files(module_id: str, header: str, code: str) -> dict[str, str]:
    notice = (
        "/* Derived from MASSES CHIP M9P6362 Demo V1.07.\n"
        " * Original work: (C) COPYRIGHT 2025 Masses-Chip.\n"
        f" * Official source: {SOURCE_URL}\n"
        " * Modular adaptation and changed parameters: see SOURCES.md.\n */\n"
    )
    return {
        f"modules/demo_{module_id}.h": notice + header,
        f"modules/demo_{module_id}.c": notice + f'#include "demo_{module_id}.h"\n\n' + code,
    }
