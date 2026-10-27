"""工具注册表：未知工具、参数不符合 schema、工具异常都转成 observation，loop 不因此中断。"""

from __future__ import annotations

import inspect
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .contracts import Tool, ToolCall, ToolContext, ToolDefinition, ToolResult

#: 派发前执行的 JSON Schema 关键字；不支持的关键字由 _check_supported 拒绝。
VALIDATION_KEYWORDS: frozenset[str] = frozenset(
    {
        "type",
        "required",
        "properties",
        "items",
        "enum",
        "const",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "additionalProperties",
    }
)

#: 纯注解关键字，不参与参数值校验。
ANNOTATION_KEYWORDS: frozenset[str] = frozenset(
    {"title", "description", "default", "examples", "$comment", "$schema", "$id"}
)

SUPPORTED_SCHEMA_KEYWORDS: frozenset[str] = VALIDATION_KEYWORDS | ANNOTATION_KEYWORDS


@dataclass(slots=True)
class FunctionTool:
    """把一个普通函数包装成 Tool，handler 可以是同步或异步。"""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object"})
    handler: Callable[[dict[str, Any], ToolContext], ToolResult | Awaitable[ToolResult]] | None = (
        None
    )
    #: 工具是否可安全重复执行。**当前不重试工具**，此字段为将来的工具级 retry 预留。
    idempotent: bool = False
    #: 并发模式：parallel 表示可与其他 parallel 工具并发；默认 exclusive（屏障）。
    execution_mode: str = "exclusive"

    async def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        if self.handler is None:
            return ToolResult(content=f"tool {self.name} has no handler", is_error=True)
        outcome = self.handler(args, context)
        if inspect.isawaitable(outcome):
            return await outcome
        return outcome


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> ToolRegistry:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name=tool.name, description=tool.description, input_schema=tool.input_schema
            )
            for tool in self._tools.values()
        ]

    def execution_mode(self, name: str) -> str:
        """按工具名查并发模式；未知工具按 exclusive 处理，不会被并行。"""
        tool = self._tools.get(name)
        if tool is None:
            return "exclusive"
        return getattr(tool, "execution_mode", "exclusive")

    async def execute(self, call: ToolCall, context: ToolContext | None = None) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(content=f"unknown tool: {call.name}", is_error=True)

        invalid = validate_arguments(tool.input_schema, call.arguments)
        if invalid is not None:
            return ToolResult(content=invalid, is_error=True)

        try:
            return await tool.execute(call.arguments, context or ToolContext())
        except Exception as exc:
            return ToolResult(content=f"tool {call.name} failed: {exc}", is_error=True)


def validate_arguments(input_schema: Any, args: Any) -> str | None:
    """在派发前检查 schema 与参数；通过时返回 None，否则返回错误描述。

    非法 JSON 原文作为入参错误处理；不支持的关键字或无效约束直接返回错误。
    """
    if not isinstance(args, dict):
        return f"tool arguments are not a JSON object: {args!r}"
    unsupported = _check_supported(input_schema, "arguments")
    if unsupported is not None:
        return unsupported
    return _validate(args, input_schema, "arguments")


_TYPE_NAMES = {
    "object": "object",
    "array": "array",
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "null": "null",
}


def _is_number(value: Any) -> bool:
    """JSON number：`int` / `float`，但**不含** `bool`（Python 里 `bool` 是 `int` 的子类）。"""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_integer(value: Any) -> bool:
    """按数值是否含小数部分判断 JSON integer，接受 1.0 并排除布尔值。"""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and value.is_integer()


def _is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _json_type(value: Any) -> str:
    # bool 是 int 的子类：True 不能算 integer/number。
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "integer":
        return _is_integer(value)
    if expected == "number":
        return _is_number(value)
    return _json_type(value) == expected


def _render(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 60 else f"{text[:57]}..."


def _check_supported(schema: Any, path: str) -> str | None:
    """检查 schema 支持集和约束格式，通过时返回 None。

    执行校验关键字，忽略纯注解；拒绝未知关键字、无效正则和非法约束值。
    """
    if not isinstance(schema, dict):
        return f"invalid tool schema: {path} must be a JSON object (got {_render(schema)})"

    for keyword in schema:
        if keyword not in SUPPORTED_SCHEMA_KEYWORDS:
            return (
                f"invalid tool schema: {path} declares unsupported keyword {keyword!r}"
                f" (supported: {', '.join(sorted(VALIDATION_KEYWORDS))})"
            )

    declared = schema.get("type")
    if isinstance(declared, str):
        if declared not in _TYPE_NAMES:
            return f"invalid tool schema: {path} declares unsupported type {declared!r}"
    elif isinstance(declared, list):
        for name in declared:
            if not isinstance(name, str) or name not in _TYPE_NAMES:
                return f"invalid tool schema: {path} declares unsupported type {name!r}"
    elif "type" in schema:
        return f"invalid tool schema: {path}.type must be a string or an array of strings"

    if "const" in schema and not _is_json_scalar(schema["const"]):
        return f"invalid tool schema: {path}.const must be a JSON scalar"

    if "enum" in schema:
        options = schema["enum"]
        if not isinstance(options, list) or not options:
            return f"invalid tool schema: {path}.enum must be a non-empty array"
        if not all(_is_json_scalar(option) for option in options):
            return f"invalid tool schema: {path}.enum must only contain JSON scalars"

    if "pattern" in schema:
        pattern = schema["pattern"]
        if not isinstance(pattern, str):
            return f"invalid tool schema: {path}.pattern must be a string"
        try:
            re.compile(pattern)
        except re.error as error:
            return (
                f"invalid tool schema: {path}.pattern {pattern!r} is not a valid"
                f" regular expression ({error})"
            )

    for keyword in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if keyword in schema and not _is_number(schema[keyword]):
            return f"invalid tool schema: {path}.{keyword} must be a number"
    for keyword in ("minLength", "maxLength", "minItems", "maxItems"):
        bound = schema.get(keyword)
        if keyword in schema and (not isinstance(bound, int) or isinstance(bound, bool)):
            return f"invalid tool schema: {path}.{keyword} must be an integer"

    properties = schema.get("properties")
    if "properties" in schema and not isinstance(properties, dict):
        return f"invalid tool schema: {path}.properties must be a JSON object"
    declared_names = properties if isinstance(properties, dict) else {}
    for name, child in declared_names.items():
        error = _check_supported(child, _join(path, str(name)))
        if error is not None:
            return error

    if "required" in schema:
        required = schema["required"]
        if not isinstance(required, list) or not all(isinstance(name, str) for name in required):
            return f"invalid tool schema: {path}.required must be an array of strings"
        if declared_names:
            for name in required:
                if name not in declared_names:
                    return (
                        f"invalid tool schema: {path}.required lists {name!r},"
                        " which is not a declared property"
                    )

    if "items" in schema:
        items = schema["items"]
        if isinstance(items, list):
            return f"invalid tool schema: {path}.items must be a single schema, not an array"
        error = _check_supported(items, f"{path}[]")
        if error is not None:
            return error

    if "additionalProperties" in schema:
        additional = schema["additionalProperties"]
        if isinstance(additional, dict):
            error = _check_supported(additional, f"{path}.*")
            if error is not None:
                return error
        elif not isinstance(additional, bool):
            return f"invalid tool schema: {path}.additionalProperties must be a boolean or a schema"

    return None


def _validate(value: Any, schema: Any, path: str) -> str | None:
    if not isinstance(schema, dict):
        return None

    if "const" in schema and not _same_json_value(value, schema["const"]):
        # 按 JSON 值语义比较，区分布尔值与数值，避免 True == 1 被当作相同值。
        return (
            f"invalid argument: {path} must equal {_render(schema['const'])}"
            f" (got {_render(value)})"
        )

    options = schema.get("enum")
    if isinstance(options, list) and not any(
        _same_json_value(value, option) for option in options
    ):
        rendered = ", ".join(_render(option) for option in options)
        return f"invalid argument: {path} must be one of [{rendered}] (got {_render(value)})"

    expected = schema.get("type")
    if isinstance(expected, str):
        if not _type_matches(value, expected):
            return (
                f"invalid argument: {path} must be {_TYPE_NAMES.get(expected, expected)}"
                f" (got {_json_type(value)} {_render(value)})"
            )
    elif isinstance(expected, list) and expected and not any(
        isinstance(name, str) and _type_matches(value, name) for name in expected
    ):
        names = " | ".join(str(name) for name in expected)
        return (
            f"invalid argument: {path} must be {names}"
            f" (got {_json_type(value)} {_render(value)})"
        )

    numeric = _is_number(value)
    if numeric:
        for keyword, failed in (
            ("minimum", lambda bound: value < bound),
            ("maximum", lambda bound: value > bound),
            ("exclusiveMinimum", lambda bound: value <= bound),
            ("exclusiveMaximum", lambda bound: value >= bound),
        ):
            bound = schema.get(keyword)
            if _is_number(bound) and failed(bound):
                return f"invalid argument: {path} violates {keyword}={bound} (got {_render(value)})"

    if isinstance(value, str):
        minimum = schema.get("minLength")
        if isinstance(minimum, int) and len(value) < minimum:
            return f"invalid argument: {path} must be at least {minimum} characters"
        maximum = schema.get("maxLength")
        if isinstance(maximum, int) and len(value) > maximum:
            return f"invalid argument: {path} must be at most {maximum} characters"
        pattern = schema.get("pattern")
        # 编译不过的模式已经由 `_check_supported` 明确拒绝，不会走到这里「假装匹配」。
        if isinstance(pattern, str) and re.search(pattern, value) is None:
            return f"invalid argument: {path} does not match pattern {pattern!r}"

    if isinstance(value, list):
        minimum = schema.get("minItems")
        if isinstance(minimum, int) and len(value) < minimum:
            return f"invalid argument: {path} must have at least {minimum} items"
        maximum = schema.get("maxItems")
        if isinstance(maximum, int) and len(value) > maximum:
            return f"invalid argument: {path} must have at most {maximum} items"
        items = schema.get("items")
        if items is not None:
            for index, element in enumerate(value):
                error = _validate(element, items, f"{path}[{index}]")
                if error is not None:
                    return error

    if isinstance(value, dict):
        required = schema.get("required")
        if isinstance(required, Sequence) and not isinstance(required, str):
            for name in required:
                if name not in value:
                    return f"missing required argument: {_join(path, str(name))}"
        properties = schema.get("properties")
        declared = properties if isinstance(properties, dict) else {}
        additional = schema.get("additionalProperties", True)
        for key, element in value.items():
            if key in declared:
                error = _validate(element, declared[key], _join(path, str(key)))
                if error is not None:
                    return error
            elif additional is False:
                return f"unexpected argument: {_join(path, str(key))}"
            elif isinstance(additional, dict):
                error = _validate(element, additional, _join(path, str(key)))
                if error is not None:
                    return error

    return None


def _join(path: str, name: str) -> str:
    return name if path == "arguments" else f"{path}.{name}"


def _same_json_value(value: Any, option: Any) -> bool:
    """`True == 1` 在 Python 里成立，但 JSON 里 `true` 和 `1` 不是同一个枚举值。"""
    if isinstance(value, bool) != isinstance(option, bool):
        return False
    return value == option
