"""工具参数的 schema 值校验（RC06 / DSH `tools/schema.ts`）。

上游在调用 handler 之前校验 schema；本地之前只检查「参数是 dict + required 键存在」，
所以声明为整数的参数传字符串照样进 handler。这里固定**支持子集**的行为。

两件事都要保证：
1. 声明了的关键字按所声明的语义执行（`integer` 按数值判、`const` 按严格 JSON 值判）；
2. 不支持或本身无效的约束**明确拒绝**，而不是静默忽略 —— 上游 `assertSupportedJsonSchema`
   对不支持/位置不对的关键字就是拒绝。
"""

from __future__ import annotations

from typing import Any

from app.contracts import ToolCall, ToolResult
from app.tools import FunctionTool, ToolRegistry, validate_arguments


def registry_with(schema: dict[str, Any], received: list[Any]) -> ToolRegistry:
    def handler(args: dict[str, Any], context: object) -> ToolResult:
        received.append(args)
        return ToolResult(content="accepted")

    return ToolRegistry().register(
        FunctionTool(name="configure", input_schema=schema, handler=handler)
    )


async def test_integer_parameter_rejects_a_string() -> None:
    """反例：frequency 声明为 integer，传 "invalid" 不能进 handler。"""
    received: list[Any] = []
    registry = registry_with(
        {
            "type": "object",
            "required": ["frequency"],
            "properties": {"frequency": {"type": "integer", "minimum": 1}},
        },
        received,
    )

    result = await registry.execute(ToolCall("bad", "configure", {"frequency": "invalid"}))

    assert result.is_error
    assert received == [], "参数不合格时 handler 不应被调用"
    assert "frequency" in result.content and "integer" in result.content


async def test_minimum_is_enforced() -> None:
    received: list[Any] = []
    registry = registry_with(
        {
            "type": "object",
            "properties": {"frequency": {"type": "integer", "minimum": 1}},
        },
        received,
    )

    result = await registry.execute(ToolCall("bad", "configure", {"frequency": 0}))

    assert result.is_error and received == []
    assert "minimum" in result.content


async def test_valid_arguments_reach_the_handler_unchanged() -> None:
    received: list[Any] = []
    registry = registry_with(
        {
            "type": "object",
            "required": ["frequency"],
            "properties": {"frequency": {"type": "integer", "minimum": 1}},
        },
        received,
    )

    result = await registry.execute(ToolCall("ok", "configure", {"frequency": 20000}))

    assert not result.is_error
    assert received == [{"frequency": 20000}]


async def test_bool_is_not_an_integer() -> None:
    """Python 里 `True` 是 `int` 的子类，但 JSON 的 true 不是 integer。"""
    assert validate_arguments({"properties": {"n": {"type": "integer"}}}, {"n": True})
    assert validate_arguments({"properties": {"n": {"type": "boolean"}}}, {"n": True}) is None


async def test_enum_and_const() -> None:
    schema = {
        "type": "object",
        "properties": {
            "kind": {"enum": ["fast", "safe"]},
            "version": {"const": "V1.0"},
        },
    }

    assert validate_arguments(schema, {"kind": "fast", "version": "V1.0"}) is None
    error = validate_arguments(schema, {"kind": "other"})
    assert error is not None and "one of" in error
    assert validate_arguments(schema, {"version": "V2.0"})


async def test_nested_object_required_and_additional_properties() -> None:
    schema = {
        "type": "object",
        "properties": {
            "config": {
                "type": "object",
                "required": ["channel"],
                "additionalProperties": False,
                "properties": {"channel": {"type": "string", "minLength": 1}},
            }
        },
    }

    assert validate_arguments(schema, {"config": {"channel": "ADCON0"}}) is None
    assert validate_arguments(schema, {"config": {}})
    assert validate_arguments(schema, {"config": {"channel": ""}})
    assert validate_arguments(schema, {"config": {"channel": "x", "extra": 1}})


async def test_array_items_and_bounds() -> None:
    schema = {
        "type": "object",
        "properties": {
            "ids": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        },
    }

    assert validate_arguments(schema, {"ids": ["a"]}) is None
    assert validate_arguments(schema, {"ids": []})
    assert validate_arguments(schema, {"ids": [1]})


async def test_unsupported_keywords_are_rejected_not_ignored() -> None:
    """反例（F03）：`oneOf` 不在支持集里，必须**拒绝派发**而不是静默忽略。

    上游 `assertSupportedJsonSchema` 对不支持的关键字是拒绝。本地之前只在文档里声明
    「不实现 oneOf」，实际却按空 schema 放行 —— 于是 `{"oneOf": [...]}` 下的布尔值
    也能进 handler，调用方会以为自己被约束过。
    """
    received: list[Any] = []
    registry = registry_with(
        {"type": "object", "properties": {"n": {"oneOf": [{"type": "string"}]}}},
        received,
    )

    result = await registry.execute(ToolCall("bad", "configure", {"n": 123}))

    assert result.is_error
    assert received == [], "声明了不支持的约束时不能调用 handler"
    assert "oneOf" in result.content and "unsupported" in result.content


async def test_annotation_keywords_are_still_accepted() -> None:
    """注解关键字不影响校验结果，不能因为「收敛支持集」被一起拒掉。"""
    received: list[Any] = []
    registry = registry_with(
        {
            "title": "configure",
            "description": "tune the radio",
            "type": "object",
            "properties": {
                "frequency": {
                    "type": "integer",
                    "description": "in Hz",
                    "default": 1,
                    "examples": [2, 3],
                }
            },
        },
        received,
    )

    result = await registry.execute(ToolCall("ok", "configure", {"frequency": 2}))

    assert not result.is_error and received == [{"frequency": 2}]


async def test_integral_float_is_an_integer() -> None:
    """反例（F03）：JSON 的 integer 是「数值没有小数部分」（上游 `Number.isInteger`）。

    按 Python 对象类型判（`isinstance(value, int)`）会把模型常发的 `1.0` 拒掉 ——
    那是实现细节泄漏到协议语义里。
    """
    received: list[Any] = []
    registry = registry_with(
        {"type": "object", "properties": {"n": {"type": "integer"}}}, received
    )

    assert not (await registry.execute(ToolCall("ok", "configure", {"n": 1.0}))).is_error
    assert received == [{"n": 1.0}], "合法整数值必须原样到达 handler"

    rejected = await registry.execute(ToolCall("bad", "configure", {"n": 1.5}))
    assert rejected.is_error
    assert len(received) == 1


async def test_const_distinguishes_boolean_from_number() -> None:
    """反例（F03）：`const: 1` 不能接受 `true`（Python 里 `True == 1`）。

    上游用严格比较（`!==`）；本地 `enum` 已经按 JSON 值语义比，`const` 之前漏了。
    """
    schema = {"type": "object", "properties": {"n": {"const": 1}}}

    assert validate_arguments(schema, {"n": True}) is not None
    assert validate_arguments(schema, {"n": 1}) is None
    # JSON 里 1 和 1.0 是同一个数，const 必须一视同仁。
    assert validate_arguments(schema, {"n": 1.0}) is None


async def test_invalid_pattern_is_rejected() -> None:
    """反例（F03）：编译不过的 pattern 不能当成「匹配成功」。

    之前 `except re.error: matched = True`，等于把声明了却无法执行的约束静默取消。
    """
    received: list[Any] = []
    registry = registry_with(
        {"type": "object", "properties": {"s": {"type": "string", "pattern": "["}}},
        received,
    )

    result = await registry.execute(ToolCall("bad", "configure", {"s": "abc"}))

    assert result.is_error
    assert received == []
    assert "pattern" in result.content and "regular expression" in result.content


async def test_invalid_schema_declarations_are_rejected() -> None:
    """schema 自己写错时也要明确报错，不能退化成「什么都不校验」。"""
    assert validate_arguments({"type": "object", "properties": {"n": {"type": "float"}}}, {"n": 1})
    assert validate_arguments(
        {"type": "object", "properties": {"n": {"items": [{"type": "string"}]}}}, {"n": []}
    )
    assert validate_arguments(
        {"type": "object", "required": ["missing"], "properties": {"n": {"type": "string"}}},
        {"n": "x"},
    )
    assert validate_arguments({"type": "object", "properties": {"n": {"enum": []}}}, {"n": 1})


async def test_non_object_arguments_are_rejected() -> None:
    """坏 JSON 会保留成原始字符串（DSH `parseArguments`），这里转成入参错误。"""
    assert validate_arguments({"type": "object"}, "{not json") is not None
