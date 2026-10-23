"""会话回溯（Hermes 的第三层：`session_search`）。

历史会话不塞进长期记忆（记忆是有界的），而是单独建 SQLite FTS5 索引，
需要时按关键词把**原始消息**捞回来。不是向量检索。

参照 Hermes `hermes_state_fts.py:40-65` + `hermes_state_search.py:1063` +
`session_search_tool.py:238-251`：

- 进索引的不只是 `content`：`tool_name` / `tool_calls` 同样建索引列。只索引正文会
  让「哪个工具、什么参数、哪次调用」里的领域事实查不回来 —— 工具调用只有参数，
  正文是空的。
- 工具调用消息（assistant 无正文）不能被跳过：上游明确保留 `content` 为空的行。
- 返回结果带上工具关联字段（工具名、参数、call id），否则拿回一条「ok」也无法
  判断它是哪次调用的结果。
- 排序用 FTS5 的 `ORDER BY rank`（上游默认），不是按插入顺序取前 N 条。

FTS5 的 unicode61 对中文按整词切分，「通道」匹配不到「通道选择位」，
所以 FTS 零命中时用 LIKE 兜底；兜底同样覆盖工具字段。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: 索引列。带 `UNINDEXED` 的只随行返回，不参与匹配。
_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages USING fts5(
    session,
    turn UNINDEXED,
    step UNINDEXED,
    role UNINDEXED,
    content,
    tool_name,
    tool_calls,
    tool_call_id UNINDEXED,
    tokenize='unicode61'
)
"""

_COLUMNS = (
    "session",
    "turn",
    "step",
    "role",
    "content",
    "tool_name",
    "tool_calls",
    "tool_call_id",
)

_SELECT = f"SELECT {', '.join(_COLUMNS)} FROM messages"

#: LIKE 兜底要一起搜的列（工具元数据也是历史事实的载体）。
_LIKE_COLUMNS = ("content", "tool_name", "tool_calls")


@dataclass(slots=True, frozen=True)
class SessionHit:
    session: str
    turn: int
    step: int
    role: str
    content: str
    #: 工具名：工具结果消息取 `message.name`，assistant 调用取被调用工具名。
    tool_name: str = ""
    #: 工具调用（名称 / 参数 / call id）的文本形式，和索引里的内容一致。
    tool_calls: str = ""
    #: 工具结果与发起调用之间的关联 id。
    tool_call_id: str = ""

    def render(self) -> str:
        header = f"[{self.session} turn={self.turn} step={self.step}] {self.role}"
        if self.tool_name:
            header += f" tool_name={self.tool_name}"
        if self.tool_call_id:
            header += f" tool_call_id={self.tool_call_id}"
        if self.tool_calls:
            header += f" tool_calls={self.tool_calls}"
        return f"{header}: {self.content}" if self.content else header


@dataclass(slots=True)
class _Row:
    """索引行（结构化后写入，不直接暴露）。"""

    session: str
    turn: int
    step: int
    role: str
    content: str
    tool_name: str
    tool_calls: str
    tool_call_id: str

    def as_tuple(self) -> tuple[str, int, int, str, str, str, str, str]:
        return (
            self.session,
            self.turn,
            self.step,
            self.role,
            self.content,
            self.tool_name,
            self.tool_calls,
            self.tool_call_id,
        )


def render_tool_calls(calls: Any) -> str:
    """把工具调用压成可检索文本：名称、参数、call id 一个都不丢。

    `arguments` 在协议里可能是原始 JSON 字符串（解析失败时 DSH 保留原文），
    两种形态都要按原样写入，而不是重新序列化成 `{}`。
    """
    if not isinstance(calls, list):
        return ""
    parts: list[str] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            rendered = arguments
        else:
            rendered = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        name = str(call.get("name") or "")
        call_id = str(call.get("id") or "")
        parts.append(f"{name}({rendered}) [id={call_id}]")
    return " ".join(parts)


def _call_names(calls: Any) -> str:
    if not isinstance(calls, list):
        return ""
    names = [
        str(call.get("name") or "")
        for call in calls
        if isinstance(call, dict) and call.get("name")
    ]
    return " ".join(names)


def _like_pattern(term: str) -> str:
    """按字面量处理通配字符，避免用户输入被当成 LIKE 模式。"""
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _like_condition() -> str:
    return " OR ".join(f"{column} LIKE ? ESCAPE '\\'" for column in _LIKE_COLUMNS)


class SessionSearch:
    """历史会话的全文索引。sqlite3 是同步的，全部走 `asyncio.to_thread`。"""

    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)

    async def index_file(self, path: str | Path, session: str | None = None) -> int:
        """索引一个 JSONL 会话文件，返回写入条数。重复索引会先清掉该会话。"""
        target = Path(path)
        name = session or target.stem
        return await asyncio.to_thread(self._index_sync, target, name)

    async def index_directory(self, directory: str | Path, pattern: str = "*.jsonl") -> int:
        total = 0
        for path in sorted(Path(directory).glob(pattern)):
            total += await self.index_file(path)
        return total

    async def search(self, query: str, limit: int = 5) -> list[SessionHit]:
        if not query.strip():
            return []
        return await asyncio.to_thread(self._search_sync, query, limit)

    # --- 同步实现 ---------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        self._ensure_schema(connection)
        return connection

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        """建表 / 升级旧表。

        旧表缺少工具列时 FTS5 不支持 `ADD COLUMN`，只能重建；**重建必须把已有行搬过去**。
        只 DROP 会造成「第一次查询把全部历史索引清空，然后返回正常零命中」——
        查询端看起来一切正常，历史却消失了（且没有任何重建提示）。
        这里逐行复制旧列，新列留空：正文检索仍然可用，重新 `index_*` 某个会话时才补上
        它的工具字段（那一会话的旧行会先被 DELETE 再插入）。
        """
        connection.execute(_SCHEMA)
        present = {row[1] for row in connection.execute("PRAGMA table_info(messages)")}
        if set(_COLUMNS).issubset(present):
            return
        salvageable = [column for column in _COLUMNS if column in present]
        legacy_rows = connection.execute(
            f"SELECT {', '.join(salvageable)} FROM messages"
        ).fetchall()
        connection.execute("DROP TABLE messages")
        connection.execute(_SCHEMA)
        placeholders = ", ".join("?" for _ in salvageable)
        connection.executemany(
            f"INSERT INTO messages ({', '.join(salvageable)}) VALUES ({placeholders})",
            legacy_rows,
        )

    def _index_sync(self, target: Path, session: str) -> int:
        try:
            raw = target.read_text(encoding="utf-8")
        except FileNotFoundError:
            return 0
        rows = [row for row in self._rows(raw, session)]
        with self._connect() as connection:
            connection.execute("DELETE FROM messages WHERE session = ?", (session,))
            connection.executemany(
                f"INSERT INTO messages ({', '.join(_COLUMNS)})"
                f" VALUES ({', '.join('?' for _ in _COLUMNS)})",
                [row.as_tuple() for row in rows],
            )
        return len(rows)

    @staticmethod
    def _rows(raw: str, session: str) -> list[_Row]:
        rows: list[_Row] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                node = json.loads(line)
            except ValueError:
                continue
            event = node.get("event", {})
            if event.get("type") != "message":
                continue
            message = event.get("message", {})
            content = str(message.get("content") or "").strip()
            calls = message.get("tool_calls") or []
            tool_calls = render_tool_calls(calls)
            tool_name = str(message.get("name") or "").strip() or _call_names(calls)
            tool_call_id = str(message.get("tool_call_id") or "").strip()
            if not tool_calls and tool_call_id:
                # 工具结果消息本身没有参数，但 call id 是它和发起调用的唯一关联。
                # 也放进被索引列：按 call id 能同时捞回「调用」和「结果」两条。
                tool_calls = f"[id={tool_call_id}]"
            # 工具调用消息正文为空，但它承载了工具名与参数 —— 不能按「无正文」跳过。
            if not content and not tool_calls and not tool_name and not tool_call_id:
                continue
            rows.append(
                _Row(
                    session=session,
                    turn=int(event.get("turn", 0)),
                    step=int(event.get("step", 0)),
                    role=str(message.get("role", "")),
                    content=content,
                    tool_name=tool_name,
                    tool_calls=tool_calls,
                    tool_call_id=tool_call_id,
                )
            )
        return rows

    def _search_sync(self, query: str, limit: int) -> list[SessionHit]:
        with self._connect() as connection:
            hits = self._match(connection, query, limit)
            if not hits:
                hits = self._like(connection, query, limit)
        return hits

    def _match(self, connection: sqlite3.Connection, query: str, limit: int) -> list[SessionHit]:
        tokens = [token for token in query.split() if token]
        if not tokens:
            return []
        match_expression = " AND ".join(f'"{token}"' for token in tokens)
        try:
            rows = connection.execute(
                f"{_SELECT} WHERE messages MATCH ? ORDER BY rank LIMIT ?",
                (match_expression, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [SessionHit(*row) for row in rows]

    def _like(self, connection: sqlite3.Connection, query: str, limit: int) -> list[SessionHit]:
        """兜底：每个关键词各自做子串匹配（与 FTS 的 AND 语义一致），不是整句相邻匹配。"""
        terms = [term for term in query.split() if term]
        if not terms:
            return []
        conditions = " AND ".join(f"({_like_condition()})" for _ in terms)
        params: list[object] = [
            pattern for term in terms for pattern in (_like_pattern(term),) * len(_LIKE_COLUMNS)
        ]
        params.append(limit)
        rows = connection.execute(
            f"{_SELECT} WHERE {conditions} LIMIT ?",
            params,
        ).fetchall()
        return [SessionHit(*row) for row in rows]
