"""提供添加、替换和删除记忆的工具；写入采用 exclusive 调度。"""

from __future__ import annotations

import json

from .contracts import MemoryItem, ToolResult
from .memory import JsonMemoryStore, MemoryFullError, MemoryMetadata
from .session_search import SessionSearch
from .tools import FunctionTool

MEMORY_WRITE_POLICY = (
    "Save only stable user-confirmed settings/preferences or findings with an identifiable "
    "document/chunk source. Preserve exactly what was confirmed: a 16 MHz clock does not "
    "confirm HIRC or Fosc. Never invent dates, clock sources or provenance. Keep hypotheses "
    "and task logs in conversation history. After a correction, update both the value and "
    "any incorrect source or applicability; omit metadata only when it remains valid. "
    "An empty source clears a false attribution. One successful update completes the write."
)


def _schema(properties: dict[str, object], required: list[str]) -> dict[str, object]:
    return {"type": "object", "properties": properties, "required": required}


def _saved(item: MemoryItem) -> ToolResult:
    # 仅回显本次保存的条目。
    return ToolResult(content=json.dumps({
        "saved": True, "key": item.key, "value": item.value, "source": item.source,
        "scope": item.scope, "applies_to": item.applies_to,
    }, ensure_ascii=False))


def remember_fact_tool(
    store: JsonMemoryStore,
    *,
    applies_to: dict[str, str] | None = None,
    default_source: str = "",
) -> FunctionTool:
    async def execute(args: dict[str, object], context: object) -> ToolResult:
        key = str(args["key"])
        value = str(args["value"])
        conditions = args.get("applies_to", applies_to or {})
        metadata = MemoryMetadata(
            source=str(args.get("source", default_source)),
            scope=str(args.get("scope", store.default_scope)),
            applies_to=dict(conditions) if isinstance(conditions, dict) else {},
        )
        try:
            item = await store.remember(key, value, metadata)
        except MemoryFullError as error:
            return ToolResult(
                content=f"memory full, consolidate or remove entries first: {error}",
                is_error=True,
            )
        return _saved(item)

    return FunctionTool(
        name="remember_fact",
        description="Save a durable fact. " + MEMORY_WRITE_POLICY,
        input_schema=_schema(
            {
                "key": {"type": "string"},
                "value": {"type": "string"},
                "scope": {"type": "string"},
                "source": {"type": "string"},
                "applies_to": {"type": "object"},
            },
            ["key", "value"],
        ),
        handler=execute,
    )


def replace_fact_tool(store: JsonMemoryStore) -> FunctionTool:
    async def execute(args: dict[str, object], context: object) -> ToolResult:
        key = str(args["key"])
        item = await store.replace(
            key, str(args["value"]), source=args.get("source"), scope=args.get("scope"),
            applies_to=args.get("applies_to"),
        )
        if item is None:
            return ToolResult(content=f"no such memory entry: {key}", is_error=True)
        return _saved(item)

    return FunctionTool(
        name="replace_fact",
        description="Correct an existing fact, including false provenance. " + MEMORY_WRITE_POLICY,
        input_schema=_schema(
            {
                "key": {"type": "string"}, "value": {"type": "string"},
                "source": {"type": "string"}, "scope": {"type": "string"},
                "applies_to": {"type": "object"},
            }, ["key", "value"]
        ),
        handler=execute,
    )


def forget_fact_tool(store: JsonMemoryStore) -> FunctionTool:
    async def execute(args: dict[str, object], context: object) -> ToolResult:
        removed = await store.remove(str(args["key"]))
        if not removed:
            return ToolResult(content=f"no such memory entry: {args['key']}", is_error=True)
        return ToolResult(content=f"forgot {args['key']}")

    return FunctionTool(
        name="forget_fact",
        description="Remove a memory entry that is no longer relevant.",
        input_schema=_schema({"key": {"type": "string"}}, ["key"]),
        handler=execute,
    )


def search_sessions_tool(search: SessionSearch) -> FunctionTool:
    async def execute(args: dict[str, object], context: object) -> ToolResult:
        query = str(args["query"])
        limit = int(args.get("limit", 5))  # type: ignore[arg-type]
        hits = await search.search(query, limit=limit)
        if not hits:
            return ToolResult(content=f"no past session matched: {query}")
        return ToolResult(content="\n".join(hit.render() for hit in hits))

    return FunctionTool(
        name="search_sessions",
        description=(
            "Search past sessions for how something was done before (full-text, not memory)."
        ),
        input_schema=_schema(
            {"query": {"type": "string"}, "limit": {"type": "integer"}}, ["query"]
        ),
        handler=execute,
        execution_mode="parallel",  # 只读，可与其它只读工具并发
    )


def memory_tools(
    store: JsonMemoryStore,
    *,
    user_profile: JsonMemoryStore | None = None,
    applies_to: dict[str, str] | None = None,
    default_source: str = "",
) -> list[FunctionTool]:
    """通过 target 选择项目事实库或用户画像库。"""
    project = [
        remember_fact_tool(store, applies_to=applies_to, default_source=default_source),
        replace_fact_tool(store), forget_fact_tool(store),
    ]
    if user_profile is None:
        return project
    profile = [
        remember_fact_tool(user_profile, default_source=default_source),
        replace_fact_tool(user_profile), forget_fact_tool(user_profile),
    ]

    def route(project_tool: FunctionTool, profile_tool: FunctionTool) -> FunctionTool:
        async def execute(args: dict[str, object], context: object) -> ToolResult:
            target = profile_tool if args.get("target") == "user" else project_tool
            return await target.execute(args, context)

        schema = project_tool.input_schema
        schema["properties"]["target"] = {"type": "string", "enum": ["project", "user"]}
        return FunctionTool(
            name=project_tool.name,
            description=project_tool.description + (
                " target=project (default) for project/model facts; target=user for user profile."
                " Include product/version in project keys when relevant."
            ),
            input_schema=schema, handler=execute,
        )

    return [route(project_tool, profile_tool) for project_tool, profile_tool in zip(
        project, profile, strict=True
    )]
