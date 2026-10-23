"""工程清单一致性检查：`demo-project.json` 与磁盘上的代码/工程文件是否自洽。

为什么需要它：`workspace_tools` 只提供通用的读写与命令，模型可以直接改 `main.C`、
`.prj`、新增模块文件，而"生成清单"（`demo-project.json`）是另一份可写文件。
2026-10-22 的真实交付里就出现过：USART 模块的 4 个文件已落盘、`main.C`/`.prj` 已改，
清单仍停留在只登记 PWM 的状态（`output/autonomous_module_20261022/before_feedback/`），
而当时的验收脚本只比对行为与保留文件，没有任何一项读清单，所以没人发现。

这里只做**判断**，不写文件、不修数据：清单是模型的交付物，修复动作应由模型完成。
检查项的口径尽量来自清单自身的声明（resources / init / actual / managed_files），
避免把某个型号的寄存器常量写死进通用规则。

两个容易误报的地方，按真实交付（`output/autonomous_module_20261022/`）校准过：

- 寄存器赋值普遍写成宏名（`TC0CL = DEMO_PWM0_PERIOD;`），所以先展开 `#define` 再比较，
  只认能解析成整型字面量的赋值；`PWM00D = ticks;` 这种运行时变量不参与判断。
- `agent_demo/tools/` 是主机侧换算脚本与 host 测试，用主机编译器跑，不进器件工程，
  因此不要求出现在 `.prj` 的 Project Files 里；`main.C`/`.prj` 属于"已有工程被修改"，
  由 `firmware.py` 的哈希比对跟踪，本来就不在 `managed_files` 里。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .firmware import code_mask, functions

STATE_FILE = "demo-project.json"

#: 由厂商提供、不属于"生成清单"的原工程文件。
VENDOR_FILES = frozenset({"COMMON.H", "MASSESCHIP_DEFINE.H", "TC1_Work.c"})

#: 运行时备份目录，不是交付内容。
IGNORED_DIRS = frozenset({".agent-demo-backups", "__pycache__"})

#: 编译与验证输出，不要求加入交付源码清单；显式登记的文件仍检查是否存在。
BUILD_ARTIFACT_SUFFIXES = frozenset({
    ".o", ".obj", ".exe", ".elf", ".bin", ".hex", ".map", ".lst", ".log",
})

#: 主机侧脚本/测试目录；用主机编译器跑，不编进器件工程。
HOST_TOOL_DIRS = frozenset({"tools"})

#: 分发层文件与它负责调用的入口，来自 `firmware.py` 生成的 `demo_app.h`。
_DISPATCH = {
    "init": ("demo_app.c", "Demo_ModulesInit"),
    "poll": ("demo_app.c", "Demo_Poll"),
    "interrupt_high": ("demo_app.c", "Demo_InterruptHigh"),
}

#: 清单字段 → 该模块资源编号下的寄存器候选名（`{n}` 取模块 id 末尾数字）。
_ACTUAL_RULES: dict[str, dict[str, tuple[str, ...]]] = {
    "pwm": {
        "period_register": ("TC{n}CL",),
        "duty_register": ("PWM{n}0D", "PWM{n}1D", "PWM{n}2D", "PWM{n}3D"),
    },
    "timer0": {
        "period_register": ("TC0CL",),
    },
    "timer1": {
        "period_register": ("TC1PRL",),
    },
    "usart": {
        "tx{n}cr": ("TX{n}CR",),
        "rx{n}cr": ("RX{n}CR",),
        "brgd{n}h": ("BRGD{n}H",),
        "brgd{n}l": ("BRGD{n}L",),
    },
}

_ASSIGNMENT = r"\b{register}\s*=\s*([^;]+);"
_DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)[ \t]+([^\r\n]*)", re.M)
_LITERAL = re.compile(r"(0[xX][0-9A-Fa-f]+|\d+)[uUlL]{0,2}$")
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
_MACRO_DEPTH = 4


@dataclass
class Finding:
    """单项检查结果；`detail` 只在失败时给具体位置，方便模型照着改。"""

    name: str
    ok: bool
    detail: str = ""
    evidence: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "check": self.name,
            "ok": self.ok,
            "detail": self.detail,
            "evidence": self.evidence[:20],
        }


def check_demo_manifest(root: str | Path) -> dict[str, Any]:
    """返回结构化报告；`ok` 为 False 表示清单与代码不自洽。"""

    root = Path(root).resolve()
    state_path = root / STATE_FILE
    if not state_path.is_file():
        return {
            "ok": False,
            "root": str(root),
            "items": [
                Finding("manifest_present", False, f"{STATE_FILE} 不存在，无法核对清单").as_dict()
            ],
            "failed": ["manifest_present"],
        }
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except ValueError as error:
        return {
            "ok": False,
            "root": str(root),
            "items": [
                Finding("manifest_present", False, f"{STATE_FILE} 不是合法 JSON：{error}").as_dict()
            ],
            "failed": ["manifest_present"],
        }

    texts = _source_texts(root, state)
    findings = [
        Finding("manifest_present", True, "", [STATE_FILE]),
        _managed_files_exist(root, state),
        _generated_files_registered(root, state),
        _module_symbols_present(state, texts),
        _module_wired(state, texts),
        _actual_matches_code(state, texts),
        _resources_unique(state),
        _project_file_listing(root, state),
    ]
    failed = [item.name for item in findings if not item.ok]
    return {
        "ok": not failed,
        "root": str(root),
        "product": state.get("product"),
        "modules": [module.get("id") for module in state.get("modules") or []],
        "items": [item.as_dict() for item in findings],
        "failed": failed,
    }


# --- 单项检查 ----------------------------------------------------------------


def _managed_files_exist(root: Path, state: dict[str, Any]) -> Finding:
    managed = _strings(state.get("managed_files"))
    missing = [name for name in managed if not (root / name).is_file()]
    return Finding(
        "managed_files_exist",
        not missing and bool(managed),
        _join("清单登记的这些文件不存在", missing)
        if missing
        else _empty(managed, "清单未登记任何文件"),
        missing,
    )


def _generated_files_registered(root: Path, state: dict[str, Any]) -> Finding:
    """盘上属于本次生成的代码，必须出现在清单里；这是本轮遗漏的正面检查。

    范围取清单自己登记过的目录（`agent_demo/`、`modules/` …），不写死某个前缀：
    清单登记了哪个目录，就说明本项目在那个目录下产出文件，该目录里的新文件必须登记。
    根目录下的 `main.C`、`.prj` 属于"已有工程被修改"，由哈希比对跟踪，不在本检查内。
    """

    managed = set(_strings(state.get("managed_files")))
    scopes = {parts[0] for parts in (_parts(name) for name in managed) if len(parts) > 1}
    unregistered: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if (any(part in IGNORED_DIRS for part in relative.parts)
                or path.suffix.lower() in BUILD_ARTIFACT_SUFFIXES):
            continue
        name = relative.as_posix()
        if relative.parts[0] in scopes and name not in managed:
            unregistered.append(name)
    return Finding(
        "generated_files_registered",
        not unregistered,
        _join("这些生成文件没有登记进 managed_files", unregistered) if unregistered else "",
        unregistered,
    )


def _module_symbols_present(state: dict[str, Any], texts: dict[str, str]) -> Finding:
    """清单声明的入口函数必须在代码里真实存在，避免"清单写了、代码没有"。"""

    missing: list[str] = []
    for module in _modules(state):
        for key in ("init", "poll", "interrupt_high"):
            symbol = module.get(key)
            if not isinstance(symbol, str) or not symbol.strip():
                continue
            if not _defines(symbol, texts):
                missing.append(f"{module.get('id')}.{key}={symbol}")
    return Finding(
        "module_symbols_present",
        not missing,
        _join("清单声明但代码里找不到的入口", missing) if missing else "",
        missing,
    )


def _module_wired(state: dict[str, Any], texts: dict[str, str]) -> Finding:
    """清单声明的入口必须被 `demo_app.c` 的分发函数调用，否则模块写了但没接进工程。"""

    dispatch_name = next(
        (name for name in texts if Path(name).name.lower() == "demo_app.c"), None
    )
    if dispatch_name is None:
        return Finding("module_wired", True, "清单未登记 demo_app.c，跳过接入检查", [])
    mask = code_mask(texts[dispatch_name])
    bodies: dict[str, str] = {}
    for function in functions(mask):
        bodies[function.name] = mask[function.opening : function.end]
    missing: list[str] = []
    checked = 0
    for module in _modules(state):
        for key, (_, owner) in _DISPATCH.items():
            symbol = module.get(key)
            if not isinstance(symbol, str) or not symbol.strip():
                continue
            body = bodies.get(owner)
            if body is None:
                missing.append(f"{dispatch_name} 里找不到 {owner}，无法接入 {symbol}")
                continue
            checked += 1
            if not re.search(rf"\b{re.escape(symbol)}\s*\(", body):
                missing.append(f"{module.get('id')}.{key}={symbol} 未被 {owner} 调用")
    return Finding(
        "module_wired",
        not missing,
        _join("清单声明的模块没有接进分发层", missing) if missing else "",
        missing or [f"{dispatch_name} 已核对 {checked} 处入口"],
    )


def _actual_matches_code(state: dict[str, Any], texts: dict[str, str]) -> Finding:
    """清单 `actual` 里的寄存器值与源码赋值是否一致。"""

    macros = _macros(texts)
    mismatches: list[str] = []
    unverified: list[str] = []
    for module in _modules(state):
        actual = module.get("actual")
        if not isinstance(actual, dict):
            continue
        module_id = str(module.get("id") or "")
        prefix = re.match(r"[a-z]+", module_id)
        rules = _ACTUAL_RULES.get(
            module_id, _ACTUAL_RULES.get(prefix.group(0) if prefix else "", {}),
        )
        index = _index_of(module_id)
        for template_field, templates in rules.items():
            field_name = template_field.format(n=index)
            value = _as_int(actual.get(field_name))
            if value is None:
                continue
            registers = [template.format(n=index) for template in templates]
            observed, unresolved = _register_values(registers, texts, macros)
            if value in observed:
                continue
            if observed or not unresolved:
                mismatches.append(
                    f"{module_id}.{field_name}={value} 与代码 {'/'.join(registers)}="
                    f"{'/'.join(str(item) for item in observed) or '未找到'}"
                )
            else:
                unverified.append(
                    f"{module_id}.{field_name}：{'/'.join(registers)} 只有非字面量赋值"
                    f"（{'、'.join(unresolved[:3])}），未参与核对"
                )
        mismatches.extend(_internal_consistency(module_id, actual))
    return Finding(
        "actual_matches_code",
        not mismatches,
        _join("清单记录的实际值与代码不符", mismatches) if mismatches else "",
        mismatches or unverified,
    )


def _resources_unique(state: dict[str, Any]) -> Finding:
    seen: dict[str, str] = {}
    conflicts: list[str] = []
    for module in _modules(state):
        module_id = str(module.get("id") or "?")
        for resource in _strings(module.get("resources")):
            owner = seen.get(resource)
            if owner is not None and owner != module_id:
                conflicts.append(f"{resource} 同时被 {owner} 与 {module_id} 占用")
            else:
                seen[resource] = module_id
    for resource in _strings(state.get("reserved_resources")):
        owner = seen.get(resource)
        if owner is not None:
            conflicts.append(f"{resource} 已在 reserved_resources 中保留，却被 {owner} 使用")
    return Finding(
        "resources_unique",
        not conflicts,
        _join("资源冲突", conflicts) if conflicts else "",
        conflicts,
    )


def _project_file_listing(root: Path, state: dict[str, Any]) -> Finding:
    """`.prj` 的文件表必须与清单登记的器件侧代码一致，否则 IDE 里少文件。"""

    project_file = str(state.get("project_file") or "")
    if not project_file:
        return Finding("project_file_listing", True, "清单未声明 project_file，跳过", [])
    path = root / project_file
    if not path.is_file():
        return Finding("project_file_listing", False, f"{project_file} 不存在", [project_file])
    text = path.read_text(encoding="utf-8", errors="replace")
    listed = _project_entries(text)
    declared = _declared_file_count(text)
    problems: list[str] = []
    if declared is not None and declared != len(listed):
        problems.append(f"File Count={declared} 但 Project Files 有 {len(listed)} 项")
    normalised = {entry.replace("\\", "/") for entry in listed}
    dangling = sorted(entry for entry in normalised if not (root / entry).is_file())
    if dangling:
        problems.append(f"{project_file} 列出的文件在磁盘上不存在：{'、'.join(dangling)}")
    missing = [name for name in _device_sources(state) if name not in normalised]
    if missing:
        problems.append(f"这些器件侧源码没有加入 {project_file}：{'、'.join(missing)}")
    return Finding(
        "project_file_listing",
        not problems,
        _join(f"{project_file} 与清单不一致", problems) if problems else "",
        problems,
    )


# --- 辅助 --------------------------------------------------------------------


def _source_texts(root: Path, state: dict[str, Any]) -> dict[str, str]:
    """只读清单登记的源码；清单没登记的代码不参与符号/数值判断。"""

    texts: dict[str, str] = {}
    for name in _strings(state.get("managed_files")):
        if Path(name).suffix.lower() not in {".c", ".h", ".C"}:
            continue
        path = root / name
        if not path.is_file():
            continue
        try:
            texts[name] = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return texts


def _defines(symbol: str, texts: dict[str, str]) -> bool:
    """复用已有工程的函数识别；声明和调用均不能代替函数实现。"""

    return any(
        function.name == symbol
        for text in texts.values()
        for function in functions(code_mask(text))
    )


def _strip_comments(text: str) -> str:
    """注释里提到 `TX0IF = 1` 不算代码赋值，这里去掉注释与字符串内容。"""

    return re.sub(
        r"/\*.*?\*/|//[^\r\n]*|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
        lambda match: re.sub(r"[^\r\n]", " ", match[0]),
        text,
        flags=re.S,
    )


def _device_sources(state: dict[str, Any]) -> list[str]:
    """需要编进器件工程的源码；`tools/` 下是主机侧脚本与 host 测试，不进 `.prj`。"""

    return [
        name
        for name in _strings(state.get("managed_files"))
        if Path(name).suffix.lower() in {".c", ".h"}
        and not HOST_TOOL_DIRS.intersection(_parts(name))
    ]


def _parts(name: str) -> tuple[str, ...]:
    return PurePosixPath(name).parts


def _macros(texts: dict[str, str]) -> dict[str, str]:
    """收集源码与头文件里的对象宏，供寄存器赋值展开。"""

    macros: dict[str, str] = {}
    for text in texts.values():
        for match in _DEFINE.finditer(_strip_comments(text)):
            macros.setdefault(match.group(1), match.group(2).strip())
    return macros


def _resolve(expression: str, macros: dict[str, str], depth: int = 0) -> int | None:
    """解析整型字面量或可逐层展开的宏名；复合表达式一律判为不可解析。"""

    text = expression.strip()
    while text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    literal = _LITERAL.match(text)
    if literal:
        try:
            return int(literal.group(1), 0)
        except ValueError:
            return None
    if depth >= _MACRO_DEPTH or not _IDENTIFIER.fullmatch(text):
        return None
    definition = macros.get(text)
    return None if definition is None else _resolve(definition, macros, depth + 1)


def _register_values(
    registers: list[str], texts: dict[str, str], macros: dict[str, str]
) -> tuple[list[int], list[str]]:
    """返回（能解析出的赋值、无法解析的赋值原文）。"""

    observed: list[int] = []
    unresolved: list[str] = []
    for text in texts.values():
        cleaned = _strip_comments(text)
        for register in registers:
            pattern = _ASSIGNMENT.format(register=re.escape(register))
            for match in re.finditer(pattern, cleaned):
                expression = " ".join(match.group(1).split())
                value = _resolve(expression, macros)
                if value is None:
                    unresolved.append(f"{register}={expression}")
                else:
                    observed.append(value)
    return sorted(set(observed)), unresolved


def _internal_consistency(module_id: str, actual: dict[str, Any]) -> list[str]:
    """清单内部自洽性：不依赖代码也能查出来的自相矛盾。"""

    problems: list[str] = []
    duty, steps = _as_int(actual.get("duty_register")), _as_int(actual.get("duty_steps"))
    percent = actual.get("duty_percent")
    if duty is not None and steps and isinstance(percent, (int, float)):
        expected = 100 * duty / steps
        if abs(expected - float(percent)) > 1e-6:
            problems.append(
                f"{module_id}.duty_percent={percent} 与 duty_register/duty_steps"
                f"（{duty}/{steps}={expected:g}）不符"
            )
    high, low = _as_int(actual.get("brgd0h")), _as_int(actual.get("brgd0l"))
    brgd = _as_int(actual.get("brgd"))
    if high is not None and low is not None and brgd is not None:
        combined = ((high & 3) << 8) | low
        if combined != brgd:
            problems.append(
                f"{module_id}.brgd={brgd} 与 BRGD0H/BRGD0L 组合值 {combined} 不符"
            )
    return problems


def _project_entries(text: str) -> list[str]:
    entries: list[str] = []
    section = _ini_section(text, "Project Files")
    for line in section.splitlines():
        if "=" not in line:
            continue
        value = line.split("=", 1)[1].strip()
        if value:
            entries.append(value)
    return entries


def _declared_file_count(text: str) -> int | None:
    section = _ini_section(text, "File Count")
    for line in section.splitlines():
        if line.strip().lower().startswith("count"):
            return _as_int(line.split("=", 1)[1].strip()) if "=" in line else None
    return None


def _ini_section(text: str, name: str) -> str:
    match = re.search(rf"\[{re.escape(name)}\](.*?)(?=\n\[|\Z)", text, flags=re.S | re.I)
    return match.group(1) if match else ""


def _modules(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [module for module in state.get("modules") or [] if isinstance(module, dict)]


def _strings(value: Any) -> list[str]:
    return [item for item in value or [] if isinstance(item, str)]


def _index_of(module_id: str) -> str:
    match = re.search(r"(\d+)$", module_id)
    return match.group(1) if match else "0"


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError:
            return None
    return None


def _join(prefix: str, items: list[str]) -> str:
    return f"{prefix}：" + "；".join(items) if items else ""


def _empty(items: list[str], message: str) -> str:
    return "" if items else message
