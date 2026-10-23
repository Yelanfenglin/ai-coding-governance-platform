"""一个受管理的示例工程：模块追加/更新、共享入口、资源检查与交付。"""

from __future__ import annotations

import configparser
import json
import posixpath
import re
from collections import Counter
from pathlib import Path

from .adapters import ModuleBuild, build_module
from .catalog import (
    ARCHIVE,
    CLOCK_HZ,
    DEMO_VERSION,
    PRODUCT,
    SOURCE_URL,
    SPEC_VERSION,
    DemoError,
    catalog,
    source_function,
    source_text,
)

STATE_FILE = "demo-project.json"


class DemoProject:
    def __init__(self, directory: str | Path, examples_root: str | Path) -> None:
        self.directory = Path(directory)
        self.examples_root = Path(examples_root)

    def catalog(self) -> dict:
        return catalog(self.examples_root)

    def _state(self) -> dict:
        path = self.directory / STATE_FILE
        if not path.exists():
            return {
                "format": 1,
                "product": PRODUCT,
                "spec_version": SPEC_VERSION,
                "demo_version": DEMO_VERSION,
                "clock_hz": CLOCK_HZ,
                "modules": [],
                "reserved_resources": [],
                "managed_files": [],
            }
        state = json.loads(path.read_text(encoding="utf-8"))
        if state["product"] != PRODUCT or state["spec_version"] != SPEC_VERSION:
            raise DemoError("此生成器只适用于M9P6362 V1.16，请使用独立项目目录")
        return state

    def read(self) -> dict:
        state = self._state()
        return {
            **state,
            "directory": str(self.directory.resolve()),
            "exists": (self.directory / STATE_FILE).exists(),
            "files": [
                {"path": name, "exists": (self.directory / name).is_file()}
                for name in state["managed_files"]
            ],
            "checks": {
                "static_structure": state.get("static_structure", "not_run"),
                "target_compile": "not_run",
                "hardware_test": "not_run",
            },
            "integration": {
                "standalone_entry": "main.C",
                "system_init": "demo_platform.c: Demo_PlatformInit",
                "module_init": "demo_app.c: Demo_ModulesInit",
                "main_loop": "demo_app.c: Demo_Poll",
                "interrupt_dispatch": "demo_app.c: Demo_InterruptHigh / Demo_InterruptLow",
                "global_interrupt_enable": "demo_app.c: Demo_EnableInterrupts",
                "existing_project": "沿用原系统初始化和中断向量，挂接上述函数；不复制main.C",
                "details": "README.md",
            },
        }

    def generate(
        self,
        module_id: str,
        parameters: dict | None = None,
        *,
        reserved_resources: list[str] | None = None,
    ) -> dict:
        if parameters is not None and not isinstance(parameters, dict):
            raise DemoError("parameters 必须是JSON对象")
        old = self._state()
        requests = [
            {"id": entry["id"], "parameters": entry["parameters"]} for entry in old["modules"]
        ]
        replacement = {"id": module_id, "parameters": parameters or {}}
        for index, request in enumerate(requests):
            if request["id"] == module_id:
                replacement["parameters"] = {**request["parameters"], **(parameters or {})}
                requests[index] = replacement
                break
        else:
            requests.append(replacement)
        reserved = old["reserved_resources"] if reserved_resources is None else reserved_resources
        if not isinstance(reserved, list) or any(not isinstance(value, str) for value in reserved):
            raise DemoError("reserved_resources 必须是资源名列表，如 TC0、IO:C1")
        reserved = list(dict.fromkeys(self._resource_name(value) for value in reserved))
        builds = [
            build_module(self.examples_root, request["id"], request["parameters"])
            for request in requests
        ]
        self._check_resources(builds, reserved)
        files = self._render(builds, reserved)
        self._check_structure(files, builds)

        # 工程的手工修改不能被后续点击生成覆盖。比较上次配置的生成结果，不加文件哈希体系。
        old_files = {}
        if old["modules"]:
            old_builds = [
                build_module(self.examples_root, entry["id"], entry["parameters"])
                for entry in old["modules"]
            ]
            old_files = self._render(old_builds, old["reserved_resources"])
        for name in files:
            path = self.directory / name
            if path.is_symlink():
                raise DemoError(f"不会覆盖链接文件：{name}")
            if path.exists() and (
                name not in old_files or path.read_text(encoding="utf-8") != old_files[name]
            ):
                raise DemoError(f"{name} 已有人工修改或不是本流程生成的文件，请先保留修改或换目录")

        state = {
            **old,
            "modules": [build.info for build in builds],
            "reserved_resources": reserved,
            "managed_files": sorted(files),
            "static_structure": "passed",
        }
        for name, content in files.items():
            path = self.directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        (self.directory / STATE_FILE).write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.read()

    @staticmethod
    def _resource_name(value: str) -> str:
        value = value.strip().upper()
        return f"IO:{value}" if re.fullmatch(r"[ABCD][0-7]", value) else value

    @staticmethod
    def _check_resources(builds: list[ModuleBuild], reserved: list[str]) -> None:
        owners = {resource: "已有工程保留" for resource in reserved}
        for build in builds:
            for resource in build.info["resources"]:
                if resource in owners:
                    raise DemoError(
                        f"资源冲突：{resource} 已被 {owners[resource]} 使用，"
                        f"{build.info['name']} 无法重复占用；"
                        + (
                            "请调整逻辑引脚"
                            if resource.startswith("IO:")
                            else "这是外设级冲突，换引脚无法解决；请保留原模块或选择未占用外设"
                        )
                    )
                owners[resource] = build.info["name"]

    def _render(self, builds: list[ModuleBuild], reserved: list[str]) -> dict[str, str]:
        files = {name: content for build in builds for name, content in build.files.items()}
        system = source_function(source_text(self.examples_root, "PWM0", "main.C"), "SYSTEM_INIT")
        system = system.replace("SYSTEM_INIT", "Demo_PlatformInit", 1)
        files["demo_platform.h"] = (
            '#ifndef DEMO_PLATFORM_H\n#define DEMO_PLATFORM_H\n#include "zc.h"\n'
            '#include "MASSESCHIP_DEFINE.H"\n\n#define DEMO_CLOCK_HZ 32000000ul\n'
            "void Demo_PlatformInit(void);\n#endif\n"
        )
        files["demo_platform.c"] = (
            f"/* System initialization from {ARCHIVE}, PWM0/main.C; MASSES CHIP. */\n"
            '#include "demo_platform.h"\n\n' + system
        )
        files["MASSESCHIP_DEFINE.H"] = source_text(
            self.examples_root,
            "PWM0",
            "MASSESCHIP_DEFINE.H",
        )
        files["demo_app.h"] = (
            "#ifndef DEMO_APP_H\n#define DEMO_APP_H\n\n"
            "void Demo_ModulesInit(void);\nvoid Demo_EnableInterrupts(void);\n"
            "void Demo_Poll(void);\n"
            "void Demo_InterruptHigh(void);\nvoid Demo_InterruptLow(void);\n#endif\n"
        )
        include = "\n".join(f'#include "modules/demo_{build.info["id"]}.h"' for build in builds)
        init = "\n".join(f"    {build.info['init']}();" for build in builds)
        poll = "\n".join(f"    {build.info['poll']}();" for build in builds if build.info["poll"])
        high = "\n".join(
            f"    {build.info['interrupt_high']}();"
            for build in builds
            if build.info["interrupt_high"]
        )
        files["demo_app.c"] = (
            '#include "demo_app.h"\n' + include + "\n\n"
            f"void Demo_ModulesInit(void)\n{{\n{init}\n}}\n\n"
            "void Demo_EnableInterrupts(void)\n{\n" + ("    GIEH = 1;\n" if high else "") + "}\n\n"
            f"void Demo_Poll(void)\n{{\n{poll}\n}}\n\n"
            f"void Demo_InterruptHigh(void)\n{{\n{high}\n}}\n\n"
            "void Demo_InterruptLow(void)\n{\n}\n"
        )
        files["main.C"] = (
            '#include "demo_platform.h"\n#include "demo_app.h"\n\n'
            "void main(void)\n{\n    Demo_PlatformInit();\n    Demo_ModulesInit();\n"
            "    Demo_EnableInterrupts();\n"
            "    while (1)\n    {\n        CLRWDT();\n        Demo_Poll();\n    }\n}\n\n"
            "void interrupt high_priority int_h(void)\n{\n    Demo_InterruptHigh();\n}\n\n"
            "void interrupt low_priority int_l(void)\n{\n    Demo_InterruptLow();\n}\n"
        )
        sources = sorted(name for name in files if Path(name).suffix.lower() in {".c", ".h"})
        files["demo.prj"] = self._project_file(sources)
        files["SOURCES.md"] = self._sources(builds)
        files["README.md"] = self._readme(builds, reserved)
        return files

    def _project_file(self, files: list[str]) -> str:
        # 统一使用官方 PWM0 工程的选项字；TC1 示例中的 WDT 选项不覆盖全局配置。
        original = configparser.ConfigParser()
        original.optionxform = str
        original.read_string(source_text(self.examples_root, "PWM0", "M9P6362_PWM0.prj"))
        settings = "\n".join(f"{key}={value}" for key, value in original.items("Settings"))
        rows = "\n".join(f"File{i}={name.replace('/', chr(92))}" for i, name in enumerate(files))
        return (
            f"[Project]\nProject=ZCW-ICE\n[File Count]\nCount={len(files)}\n"
            f"[Filter Index]\nIndex=2\n[Project Files]\n{rows}\n[Settings]\n{settings}\n"
        )

    @staticmethod
    def _sources(builds: list[ModuleBuild]) -> str:
        lines = [
            "# 来源与适配",
            "",
            f"官方例程：[{ARCHIVE}]({SOURCE_URL})，版本{DEMO_VERSION}。",
            f"规格书：{PRODUCT} {SPEC_VERSION}。例程版本与规格书版本分别记录。",
            "",
            "系统初始化及工程选项字取自 PWM0 工程（32MHz/8T）；模块不重设时钟。",
            "TC0/TC1 的独立工程选项字不合并，统一沿用 PWM0 的WDT/复位配置。",
            "",
        ]
        for build in builds:
            info = build.info
            lines.extend(
                [
                    f"## {info['name']}",
                    f"包内源码：`{info['source']['member']}`。",
                    "规格位置："
                    + "、".join(
                        f"§{ref['section']} p.{ref['page']}" for ref in info["spec_sections"]
                    )
                    + "。",
                    "适配：" + "；".join(info["adaptations"]) + "。",
                    "",
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def _readme(builds: list[ModuleBuild], reserved: list[str]) -> str:
        lines = [
            f"# {PRODUCT} 模块示例工程",
            "",
            f"规格书 {SPEC_VERSION}，官方例程 {DEMO_VERSION}。",
            "系统配置：HIRC 32MHz / 8T；所有模块共享此配置。",
            "",
            "## 模块与资源",
            "",
            "| 模块 | 参数 | 实际配置 | 独占资源 |",
            "|---|---|---|---|",
        ]
        for build in builds:
            info = build.info
            requested = json.dumps(info["parameters"], ensure_ascii=False)
            lines.append(
                f"| {info['name']} | {requested} | "
                f"{json.dumps(info['actual'], ensure_ascii=False)} | "
                f"{', '.join(info['resources'])} |"
            )
        lines.extend(
            [
                "",
                f"已有工程保留：{', '.join(reserved) or '无'}。",
                "",
                "频率/时间值按官方固定分频换算，实际配置栏显示寄存器量化后的结果。",
                "IO:A0、IO:C0等为逻辑端口名；请按所选封装核对物理引脚。",
                "",
                "## 单独运行",
                "",
                "在厂商 ZCW-ICE 中打开 demo.prj，选择 M9P6362。工程依赖厂商工具链的 zc.h，",
                "该SDK头文件不在官方示例包中，不提供伪造替代文件。源码使用UTF-8编码。",
                "main.C 演示一次系统初始化、模块初始化、主循环与唯一的中断入口。",
                "",
                "## 接入已有工程",
                "",
                "1. 将 modules 下的 .c/.h、demo_app.c/.h 和必要头文件加入工程。",
                "2. 已有系统初始化时保留原初始化；核对32MHz/8T和资源，",
                "   不重复调用 Demo_PlatformInit。",
                "3. 外设初始化处调用 Demo_ModulesInit，主循环调用 Demo_Poll。",
                "   全部中断挂接准备完成后再调用 Demo_EnableInterrupts，或沿用已有全局中断控制。",
                "4. 在已有高/低优先级中断函数中分别调用",
                "   Demo_InterruptHigh / Demo_InterruptLow。",
                "   保留原有中断入口，不把 main.C 中的入口再复制一份。",
                "5. 修改配置请在生成入口更新参数；流程会保留并拒绝覆盖已有人工修改。",
                "",
            ]
        )
        for build in builds:
            info = build.info
            if info["id"] == "pwm0":
                lines.extend(
                    ["PWM接口：Demo_Pwm0_Start / Stop；", "SetDuty接收计数ticks，不是百分比。", ""]
                )
            else:
                index = info["id"][-1]
                lines.extend(
                    [
                        f"定时接口：Demo_Timer{index}_TakeTick 返回是否有待处理事件。",
                        "配置GPIO指示时由Poll消费该标志；none时由业务代码消费。标志会合并多次中断，",
                        "不作为精确的累计计数器。",
                        "",
                    ]
                )
        lines.extend(
            [
                "## 检查结果",
                "",
                "已检查入口/函数唯一性、调用接口、文件依赖及资源占用。",
                "厂商编译：未执行。实机运行：未执行。",
                "",
                "官方原文与改动依据见 SOURCES.md；项目状态保存在 demo-project.json。",
                "",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _check_structure(files: dict[str, str], builds: list[ModuleBuild]) -> None:
        definitions = []
        for name, text in files.items():
            if Path(name).suffix.lower() == ".c":
                definitions.extend(
                    re.findall(
                        r"^(?:void|unsigned char)\s+(?:interrupt\s+\w+\s+)?(\w+)\s*\([^;]*?\)\s*\{",
                        text,
                        flags=re.M,
                    )
                )
            for included in re.findall(r'^#include "([^"]+)"', text, flags=re.M):
                if included == "zc.h":
                    continue
                target = posixpath.normpath(posixpath.join(posixpath.dirname(name), included))
                if target not in files:
                    raise DemoError(f"生成文件缺少依赖：{name} -> {included}")
        repeated = [name for name, count in Counter(definitions).items() if count > 1]
        if repeated:
            raise DemoError(f"生成文件存在重复函数：{', '.join(repeated)}")
        required = {"main", "int_h", "int_l", "Demo_PlatformInit", "Demo_ModulesInit"}
        required.update(build.info["init"] for build in builds)
        if not required.issubset(definitions):
            raise DemoError(f"缺少生成入口：{', '.join(required - set(definitions))}")

    def file(self, name: str) -> str:
        if name not in self._state()["managed_files"] and name != STATE_FILE:
            raise DemoError("文件不在本工程的生成清单中")
        return (self.directory / name).read_text(encoding="utf-8")
