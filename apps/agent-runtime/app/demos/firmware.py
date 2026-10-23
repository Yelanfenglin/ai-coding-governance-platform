"""在现有 ZCW-ICE C 工程中接入模块；有限结构识别，不是通用 C 解析器。"""

from __future__ import annotations

import configparser
import difflib
import json
import posixpath
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .adapters import build_module
from .catalog import PRODUCT, DemoError, source_text
from .project import STATE_FILE, DemoProject


def code_mask(text: str) -> str:
    """注释与字符串保留长度/换行，只用于定位，编辑始终作用于原文。"""
    return re.sub(
        r'/\*.*?\*/|//[^\r\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
        lambda match: re.sub(r"[^\r\n]", " ", match[0]),
        text,
        flags=re.S,
    )


def closing_brace(mask: str, opening: int) -> int:
    depth = 0
    for position in range(opening, len(mask)):
        depth += (mask[position] == "{") - (mask[position] == "}")
        if depth == 0:
            return position
    raise DemoError("C函数或循环的花括号未闭合，未修改工程")


@dataclass
class CFunction:
    name: str
    start: int
    opening: int
    end: int
    priority: str | None


def functions(mask: str) -> list[CFunction]:
    pattern = (
        r"(?m)^[ \t]*(?:static\s+)?(?:void|int)\s+"
        r"(?:interrupt\s+(?P<priority>high_priority|low_priority)\s+)?"
        r"(?P<name>[A-Za-z_]\w*)\s*\(\s*(?:void)?\s*\)\s*\{"
    )
    return [
        CFunction(
            match["name"],
            match.start(),
            match.end() - 1,
            closing_brace(mask, match.end() - 1),
            match["priority"],
        )
        for match in re.finditer(pattern, mask)
    ]


@dataclass
class SourceFile:
    path: Path
    raw: bytes
    text: str
    encoding: str

    @classmethod
    def load(cls, path: Path) -> SourceFile:
        raw = path.read_bytes()
        encoding = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            encoding = "gb18030"
            text = raw.decode(encoding)
        return cls(path, raw, text, encoding)

    @property
    def newline(self) -> str:
        return "\r\n" if "\r\n" in self.text else "\n"


class FirmwareProject(DemoProject):
    def __init__(self, directory: str | Path, examples_root: str | Path) -> None:
        selected = Path(directory).resolve()
        self.project_file = selected if selected.suffix.lower() == ".prj" else None
        super().__init__(selected.parent if self.project_file else selected, examples_root)

    def _path(self, name: str) -> Path:
        target = (self.directory / name.replace("\\", "/")).resolve()
        if not target.is_relative_to(self.directory):
            raise DemoError("工程文件不能指向所选目录之外")
        return target

    def _state(self) -> dict:
        state = super()._state()
        if (self.directory / STATE_FILE).exists() and state.get("mode") != "existing":
            raise DemoError("这是独立Demo目录，请沿用 --demo-project")
        return {**state, "mode": "existing"}

    def _scan(self) -> tuple[dict, dict[str, SourceFile]]:
        candidates = (
            [self.project_file] if self.project_file else sorted(self.directory.glob("*.prj"))
        )
        if len(candidates) != 1:
            raise DemoError("请指定唯一的 .prj 工程文件，或只包含一个 .prj 的目录")
        project_name = candidates[0].relative_to(self.directory).as_posix()
        documents = {project_name: SourceFile.load(candidates[0])}
        config = configparser.ConfigParser(interpolation=None)
        config.read_string(documents[project_name].text)
        if config.get("Settings", "Device", fallback="") != PRODUCT:
            raise DemoError("现有工程必须明确配置 Device=M9P6362")
        state = self._state()
        managed = set(state["managed_files"])
        pending = [value.replace("\\", "/") for _, value in config.items("Project Files")]
        external_includes = set()
        while pending:
            name = posixpath.normpath(pending.pop())
            if name in documents:
                continue
            document = SourceFile.load(self._path(name))
            documents[name] = document
            for include in re.findall(r'^\s*#include\s+"([^"]+)"', document.text, re.M):
                included = posixpath.normpath(posixpath.join(posixpath.dirname(name), include))
                if self._path(included).is_file():
                    pending.append(included)
                else:
                    external_includes.add(include)

        resources: dict[str, list[dict]] = {}
        entries: list[dict] = []
        issues = []
        oscm_values = []
        register = re.compile(
            r"\b(?:TC[01](?:C[HL]|EN|IE|IF|IP|PR[HL])|T[01]CR2?|"
            r"PWM0[0-3](?:CR|D|OE|EN)|MPPWM0|"
            r"(?:IO|OE|ANS)[ABCD][0-7])\b"
        )
        for name, document in documents.items():
            if name in managed or document.path.suffix.lower() != ".c":
                continue
            mask = code_mask(document.text)
            definitions = functions(mask)
            for function in definitions:
                if function.name == "main" or function.priority:
                    entries.append(
                        {
                            "file": name,
                            "name": function.name,
                            "priority": function.priority,
                            "line": document.text.count("\n", 0, function.start) + 1,
                        }
                    )
            if re.search(r"(?m)^\s*#\s*(?:if|ifdef|ifndef)\b", mask):
                issues.append(f"{name} 含条件编译；本轮不推断哪条分支有效")
            for match in re.finditer(r"\bOSCM\s*=\s*([^;]+);", mask):
                try:
                    oscm_values.append(int(match[1].strip(), 0))
                except ValueError:
                    oscm_values.append(None)
            for match in register.finditer(mask):
                symbol = match[0]
                if symbol.startswith(("TC0", "T0")):
                    resource = "TC0"
                elif symbol.startswith(("TC1", "T1")):
                    resource = "TC1"
                elif symbol.startswith(("PWM0", "MPPWM0")):
                    resource = "PWM0"
                else:
                    resource = f"IO:{symbol[-2:]}"
                resources.setdefault(resource, []).append(
                    {
                        "file": name,
                        "line": document.text.count("\n", 0, match.start()) + 1,
                        "symbol": symbol,
                    }
                )
            # 整端口业务读写保留整组引脚；官方系统复位初始化不等于应用占用了每个口。
            for match in re.finditer(r"\b(?:IO|OE|ANS)([ABCD])\b", mask):
                owner = next(
                    (fn.name for fn in definitions if fn.opening < match.start() < fn.end),
                    "",
                )
                if owner in {"SYSTEM_INIT", "Demo_PlatformInit"}:
                    continue
                for bit in range(8):
                    resources.setdefault(f"IO:{match[1]}{bit}", []).append(
                        {
                            "file": name,
                            "line": document.text.count("\n", 0, match.start()) + 1,
                            "symbol": match[0],
                        }
                    )

        reference = configparser.ConfigParser()
        reference.read_string(source_text(self.examples_root, "PWM0", "M9P6362_PWM0.prj"))
        # V1.07 PWM0/TC1工程的已审查选项，仅ConfigBitValue3分别为64/120；不猜位编码。
        options_match = all(
            config.get("Settings", f"ConfigBitValue{index}", fallback="")
            == reference.get("Settings", f"ConfigBitValue{index}")
            for index in range(10)
            if index != 3
        ) and config.get("Settings", "ConfigBitValue3", fallback="") in {"64", "120"}
        clock_matches = options_match and bool(oscm_values) and set(oscm_values) == {0}
        if not clock_matches:
            issues.append("时钟/工程选项未匹配已审查的32MHz/8T配置；本轮不重设原工程时钟")
        main_entries = [entry for entry in entries if entry["name"] == "main"]
        if len(main_entries) != 1:
            issues.append("需要一个可识别的 main(void) 或 main() 入口")
        for priority in ("high_priority", "low_priority"):
            if sum(entry["priority"] == priority for entry in entries) > 1:
                issues.append(f"存在多个 {priority} 中断入口")
        return {
            **state,
            "directory": str(self.directory),
            "project_file": project_name,
            "exists": True,
            "files": [{"path": name, "exists": True} for name in sorted(documents)],
            "entries": entries,
            "detected_resources": [
                {"resource": resource, "evidence": evidence[:5]}
                for resource, evidence in sorted(resources.items())
            ],
            "clock": {
                "status": "matches_reviewed_profile" if clock_matches else "unconfirmed",
                "hz": 32_000_000 if clock_matches else None,
                "instruction_divider": 8,
                "oscm_writes": oscm_values,
            },
            "issues": issues,
            "external_includes": sorted(external_includes),
            "analysis_scope": "工程列出的C文件与本地包含文件；直接寄存器引用，不作完整C语义分析",
            "integration": {
                "existing_entries": entries,
                "module_init": "agent_demo/demo_app.c: Demo_ModulesInit",
                "module_poll": "agent_demo/demo_app.c: Demo_Poll",
                "module_irq": "agent_demo/demo_app.c: Demo_InterruptHigh",
                "system_init": "保留原系统初始化；不生成或调用Demo_PlatformInit",
            },
            "checks": {"target_compile": "not_run", "hardware_test": "not_run"},
        }, documents

    def read(self) -> dict:
        return self._scan()[0]

    def file(self, name: str) -> str:
        state, documents = self._scan()
        if name in documents:
            return documents[name].text
        if name in state["managed_files"] or name == STATE_FILE:
            return SourceFile.load(self._path(name)).text
        raise DemoError("请读取工程清单中的文件")

    def _render_modules(self, builds, reserved, header: str) -> dict[str, str]:
        rendered = super()._render(builds, reserved)
        wanted = {"demo_app.c", "demo_app.h", "demo_platform.h", "SOURCES.md"}
        files = {
            f"agent_demo/{name}": text
            for name, text in rendered.items()
            if name.startswith("modules/") or name in wanted
        }
        files["agent_demo/demo_platform.h"] = (
            '#ifndef DEMO_PLATFORM_H\n#define DEMO_PLATFORM_H\n#include "zc.h"\n'
            f'#include "{posixpath.relpath(header, "agent_demo")}"\n#endif\n'
        )
        files["agent_demo/SOURCES.md"] = files["agent_demo/SOURCES.md"].replace(
            "系统初始化及工程选项字取自 PWM0 工程（32MHz/8T）；模块不重设时钟。\n"
            "TC0/TC1 的独立工程选项字不合并，统一沿用 PWM0 的WDT/复位配置。",
            "保留已有工程的系统初始化及全部工程选项；模块不重设时钟或WDT/复位配置。\n"
            "当前工程选项和OSCM写入已与审查过的32MHz/8T配置对照。\n"
            "新增模块初始化期间暂时关闭GIEH并恢复原值；新定时模块初始化后开放高优先级中断。",
        )
        app = files["agent_demo/demo_app.c"]
        init = next(fn for fn in functions(code_mask(app)) if fn.name == "Demo_ModulesInit")
        app = app[: init.end] + "    GIEH = saved_gieh;\n" + app[init.end :]
        files["agent_demo/demo_app.c"] = (
            app[: init.opening + 1]
            + "\n    unsigned char saved_gieh = GIEH;\n    GIEH = 0;"
            + app[init.opening + 1 :]
        )
        return files

    @staticmethod
    def _hooks(text: str, needs_poll: bool, needs_interrupt: bool) -> str:
        mask = code_mask(text)
        main = next(fn for fn in functions(mask) if fn.name == "main")
        candidates = [
            match
            for match in re.finditer(
                r"\b(?:while\s*\(\s*1\s*\)|for\s*\(\s*;\s*;\s*\))\s*\{",
                mask,
            )
            if main.opening < match.start() < main.end
            and mask[main.opening + 1 : match.start()].count("{")
            == mask[main.opening + 1 : match.start()].count("}")
        ]
        if len(candidates) != 1:
            raise DemoError("需要 main 内唯一的顶层 while(1)/for(;;) 循环，未改动原代码")
        loop = candidates[0]
        loop_end = closing_brace(mask, loop.end() - 1)
        newline = "\r\n" if "\r\n" in text else "\n"
        insertions = []

        def insert_body_call(opening: int, call: str) -> None:
            after_open = opening + 1
            line_end = re.match(r"[ \t]*\r?\n", text[after_open:])
            if line_end:
                position = after_open + line_end.end()
                indent = re.match(r"[ \t]*", text[position:])[0] or "    "
                insertions.append((position, f"{indent}{call}();{newline}"))
            else:
                insertions.append((after_open, f"{newline}    {call}();{newline}"))

        initial = mask[main.opening + 1 : loop.start()]
        calls = ["Demo_ModulesInit"] + (["Demo_EnableInterrupts"] if needs_interrupt else [])
        missing = [name for name in calls if not re.search(rf"\b{name}\s*\(", initial)]
        if missing:
            line_start = text.rfind("\n", 0, loop.start()) + 1
            indent = text[line_start : loop.start()]
            calls_text = "".join(f"{indent}{name}();{newline}" for name in missing)
            insertions.append((line_start, calls_text))
        if needs_poll and not re.search(r"\bDemo_Poll\s*\(", mask[loop.end() : loop_end]):
            insert_body_call(loop.end() - 1, "Demo_Poll")
        if '#include "agent_demo/demo_app.h"' not in text:
            includes = list(re.finditer(r"(?m)^[ \t]*#include[^\r\n]*(?:\r?\n|$)", text))
            position = includes[-1].end() if includes else 0
            insertions.append((position, f'#include "agent_demo/demo_app.h"{newline}'))
        if needs_interrupt:
            high = [fn for fn in functions(mask) if fn.priority == "high_priority"]
            if high:
                entry = high[0]
                if not re.search(r"\bDemo_InterruptHigh\s*\(", mask[entry.opening : entry.end]):
                    insert_body_call(entry.opening, "Demo_InterruptHigh")
            else:
                insertions.append(
                    (
                        len(text),
                        (
                            f"{newline}void interrupt high_priority int_h(void){newline}{{{newline}"
                            f"    Demo_InterruptHigh();{newline}}}{newline}"
                        ),
                    )
                )
        for position, value in sorted(insertions, reverse=True):
            text = text[:position] + value + text[position:]
        return text

    @staticmethod
    def _project_entries(text: str, names: list[str]) -> str:
        config = configparser.ConfigParser(interpolation=None)
        config.read_string(text)
        existing = {value.replace("\\", "/") for _, value in config.items("Project Files")}
        additions = [name for name in names if name not in existing]
        if not additions:
            return text
        newline = "\r\n" if "\r\n" in text else "\n"
        section = re.search(r"(?ms)^\[Project Files\][^\r\n]*\r?\n(.*?)(?=^\[|\Z)", text)
        count = re.search(r"(?m)^\[File Count\]\s*\r?\nCount=(\d+)", text)
        if section is None or count is None:
            raise DemoError("无法定位原工程文件清单，未修改工程")
        indices = [
            int(key[4:]) for key, _ in config.items("Project Files") if key.startswith("file")
        ]
        start = max(indices, default=-1) + 1
        rows = "".join(
            f"File{start + i}={name.replace('/', chr(92))}{newline}"
            for i, name in enumerate(additions)
        )
        text = text[: section.end()] + rows + text[section.end() :]
        text = text[: count.start(1)] + str(len(existing) + len(additions)) + text[count.end(1) :]
        return text

    def generate(
        self,
        module_id: str,
        parameters: dict | None = None,
        *,
        reserved_resources: list[str] | None = None,
        preview: bool = False,
    ) -> dict:
        if parameters is not None and not isinstance(parameters, dict):
            raise DemoError("parameters必须是JSON对象")
        report, documents = self._scan()
        if report["issues"]:
            raise DemoError("；".join(report["issues"]))
        requests = {row["id"]: row["parameters"] for row in report["modules"]}
        requests[module_id] = {**requests.get(module_id, {}), **(parameters or {})}
        builds = [build_module(self.examples_root, key, value) for key, value in requests.items()]
        reserved = (
            report["reserved_resources"] if reserved_resources is None else reserved_resources
        )
        if not isinstance(reserved, list) or any(not isinstance(value, str) for value in reserved):
            raise DemoError("reserved_resources必须是资源名列表")
        reserved = list(dict.fromkeys(self._resource_name(value) for value in reserved))
        detected = {row["resource"]: row["evidence"][0] for row in report["detected_resources"]}
        for build in builds:
            for resource in build.info["resources"]:
                if resource in detected:
                    evidence = detected[resource]
                    raise DemoError(
                        f"资源冲突：{resource} 已在 {evidence['file']}:{evidence['line']} "
                        f"引用 {evidence['symbol']}，保留原功能时不能接入 {build.info['name']}"
                    )
        self._check_resources(builds, reserved)
        headers = [name for name in documents if Path(name).name.upper() == "MASSESCHIP_DEFINE.H"]
        if len(headers) != 1:
            raise DemoError("需要工程中唯一的MASSESCHIP_DEFINE.H")
        files = self._render_modules(builds, reserved, headers[0])
        old_builds = [
            build_module(self.examples_root, row["id"], row["parameters"])
            for row in report["modules"]
        ]
        previous = self._render_modules(old_builds, report["reserved_resources"], headers[0])
        for name in files:
            path = self._path(name)
            if path.exists() and (
                name not in report["managed_files"] or SourceFile.load(path).text != previous[name]
            ):
                raise DemoError(f"{name} 已有人工修改，未覆盖")
        main_entry = next(entry for entry in report["entries"] if entry["name"] == "main")
        main_file = documents[main_entry["file"]]
        high_entries = [
            entry for entry in report["entries"] if entry["priority"] == "high_priority"
        ]
        needs_interrupt = any(build.info["interrupt_high"] for build in builds)
        if needs_interrupt and high_entries and high_entries[0]["file"] != main_entry["file"]:
            raise DemoError("首版只接入与main同文件的中断向量，未改动原工程")
        files[main_entry["file"]] = self._hooks(
            main_file.text,
            any(build.info["poll"] for build in builds),
            needs_interrupt,
        )
        project_file = report["project_file"]
        module_sources = [name for name in files if Path(name).suffix.lower() in {".c", ".h"}]
        files[project_file] = self._project_entries(documents[project_file].text, module_sources)
        changes = []
        for name, text in files.items():
            document = documents.get(name)
            old_text = (
                document.text
                if document
                else (SourceFile.load(self._path(name)).text if self._path(name).exists() else "")
            )
            if old_text != text:
                change = {"path": name, "action": "modify" if old_text else "add"}
                if document:
                    change["diff"] = "".join(
                        difflib.unified_diff(
                            old_text.splitlines(keepends=True),
                            text.splitlines(keepends=True),
                            fromfile=name,
                            tofile=name,
                        )
                    )
                changes.append(change)
        if preview:
            return {
                "preview": True,
                "directory": str(self.directory),
                "changes": changes,
                "modules": [build.info for build in builds],
                "checks": report["checks"],
            }
        # 以刚读取的字节作写入前提；不把磁盘上新产生的用户修改覆盖掉。
        for name in (main_entry["file"], project_file):
            if self._path(name).read_bytes() != documents[name].raw:
                raise DemoError(f"{name} 在分析后有变更，请重新读取")
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        backup = self.directory / ".agent-demo-backups" / stamp
        for change in changes:
            name = change["path"]
            path = self._path(name)
            if name in documents:
                saved = backup / name
                saved.parent.mkdir(parents=True, exist_ok=True)
                saved.write_bytes(documents[name].raw)
                raw = files[name].encode(documents[name].encoding)
            else:
                raw = files[name].encode("utf-8")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        state = {
            **self._state(),
            "modules": [build.info for build in builds],
            "managed_files": sorted(name for name in files if name.startswith("agent_demo/")),
            "reserved_resources": reserved,
            "project_file": project_file,
        }
        (self.directory / STATE_FILE).write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {
            **self.read(),
            "changes": changes,
            "backup": str(backup) if backup.exists() else None,
        }
