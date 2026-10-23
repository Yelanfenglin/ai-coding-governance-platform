"""长期记忆（Phase F）：有界的事实库 / 用户画像。

参照 Hermes Agent（`NousResearch/hermes-agent`）的记忆设计：

- **有界**：字符预算（MEMORY.md 2200 / USER.md 1375），写不下就**显式失败**，
  绝不静默淘汰 —— 让 Agent/调用方自己 consolidate；
- **两个库**：`JsonMemoryStore`（环境事实、约定、教训）与用户画像（`scope="user"` 的库），
  在回合开始时作为 **frozen snapshot** 注入（Hermes 为保 prefix cache 不在中途刷新）；
- **Agent 自管理**：`memory_tools()` 给出 add / replace / remove 三个工具；
- **软失效 + applies_to**：`invalidate()` 保留记录可审计；`applies_to` 与当前上下文
  （型号 / 版本）冲突的条目不召回。

记忆不替代 citation：注入块里写死「verify before use, citations win」。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .contracts import MemoryItem

#: 默认字符预算。Hermes 是 2200 / 1375；这里默认放大到 6000 / 3000，仍然是有界的
#: —— 预算本身可配置。注意：按 **字符数**（`len(str)`）计算，既不是字节数也不是 token 数。
#: 按路径共享的提交锁（进程内）。
_PATH_LOCKS: dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _path_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _PATH_LOCKS[key] = lock
        return lock


def _lock_path_for(path: Path) -> Path:
    """旁挂锁文件：和 journal 同目录同前缀，便于随数据一起清理。"""
    return path.with_name(path.name + ".lock")


def _open_lock_file(path: Path) -> Any:
    try:
        return path.open("r+b")
    except FileNotFoundError:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("w+b")


def _acquire(handle: Any) -> None:
    """阻塞式获取 OS 文件锁（Hermes 的 `_path_lock`）。"""
    if os.name == "nt":
        import msvcrt  # noqa: PLC0415 - 平台相关模块，只在需要时导入

        while True:
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                # Windows 没有阻塞式区间锁，用短重试模拟（LK_LOCK 自带 10 次 1s 重试，
                # 粒度太粗，会让等待方整秒卡住）。
                time.sleep(0.005)
    else:  # pragma: no cover - POSIX 分支
        import fcntl  # noqa: PLC0415

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _release(handle: Any) -> None:
    if os.name == "nt":
        import msvcrt  # noqa: PLC0415

        with contextlib.suppress(OSError):
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:  # pragma: no cover - POSIX 分支
        import fcntl  # noqa: PLC0415

        with contextlib.suppress(OSError):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """跨进程互斥：同一 journal 的提交与读取在同一把 OS 文件锁下进行。

    进程内的 `threading.Lock` 只能约束同一个解释器；两个 CLI 进程 / Agent 与外部
    API 同时写一份记忆库时，仍会各自依据旧状态通过预算检查。Hermes 用文件锁解决，
    这里用同样的机制（Windows `msvcrt.locking` / POSIX `flock`）。
    """
    lock_path = _lock_path_for(path)
    handle = _open_lock_file(lock_path)
    try:
        if lock_path.stat().st_size == 0:
            handle.write(b"\0")  # msvcrt 锁的是字节区间，文件为空时先占一个字节
            handle.flush()
        handle.seek(0)
        _acquire(handle)
        try:
            yield
        finally:
            _release(handle)
    finally:
        handle.close()


DEFAULT_MEMORY_CHARS = 6000
DEFAULT_USER_CHARS = 3000

MEMORY_HEADER = (
    "## Memory (durable facts)\n"
    "Stable facts from earlier work. Verify before relying on them: "
    "tool evidence and citations always win."
)
USER_HEADER = "## User profile (stable preferences)"


class MemoryFullError(RuntimeError):
    """超过字符预算。Hermes 的做法是显式失败，让调用方自己腾地方。"""


@dataclass(frozen=True)
class MemoryMetadata:
    source: str = ""
    scope: str = "project"
    confidence: float = 1.0
    applies_to: dict[str, str] = field(default_factory=dict)


class MemoryStore(Protocol):
    async def remember(
        self, key: str, value: str, metadata: MemoryMetadata | None = None
    ) -> MemoryItem: ...

    async def recall(
        self,
        query: str = "",
        limit: int = 20,
        scope: str | None = None,
        applies_to: dict[str, str] | None = None,
    ) -> list[MemoryItem]: ...

    async def forget(self, key: str) -> bool: ...


class JsonMemoryStore:
    """单文件 JSON 事实库，按 key 去重（同一 key 再次写入视为更新）。"""

    def __init__(
        self,
        path: str | Path,
        max_chars: int = DEFAULT_MEMORY_CHARS,
        *,
        default_scope: str = "project",
    ) -> None:
        self.path = Path(path)
        self.max_chars = max_chars
        self.default_scope = default_scope
        #: 提交锁按**路径**共享（Hermes 用基于文件的锁）：同一路径的多个 store 实例
        #: 必须受同一把锁约束，否则两个提交者会各自依据旧状态通过容量检查。
        #: 这是进程内的一层；跨进程由提交时的 `_file_lock`（OS 文件锁）兜住。
        self._write_lock = _path_lock(self.path)

    # --- 写入 -------------------------------------------------------------

    async def remember(
        self, key: str, value: str, metadata: MemoryMetadata | None = None
    ) -> MemoryItem:
        meta = metadata or MemoryMetadata(scope=self.default_scope)
        items = await self._load()
        updated = MemoryItem(
            key=key,
            value=value,
            source=meta.source,
            scope=meta.scope,
            updated_at=_now(),
            confidence=meta.confidence,
            applies_to=dict(meta.applies_to),
        )
        self._ensure_fits(items, updated)  # 快速失败
        await self._append_op({"op": "upsert", "item": asdict(updated)}, updated)
        return updated

    async def replace(
        self,
        key: str,
        value: str,
        *,
        source: str | None = None,
        scope: str | None = None,
        applies_to: dict[str, str] | None = None,
    ) -> MemoryItem | None:
        """更新条目及显式提供的元数据；省略的字段保留，source="" 可清除错误来源。

        Hermes patch 修改文本条目；这里的元数据是本地结构化适配，必须能一起纠正。
        """
        items = await self._load()
        for item in items:
            if item.key == key:
                updated = MemoryItem(
                    key=key,
                    value=value,
                    source=item.source if source is None else source,
                    scope=item.scope if scope is None else scope,
                    updated_at=_now(),
                    confidence=item.confidence,
                    applies_to=dict(item.applies_to if applies_to is None else applies_to),
                )
                self._ensure_fits(items, updated)  # 快速失败
                await self._append_op({"op": "upsert", "item": asdict(updated)}, updated)
                return updated
        return None

    async def remove(self, key: str) -> bool:
        """硬删除（Hermes 的 `remove`）。"""
        items = await self._load()
        if not any(item.key == key for item in items):
            return False
        await self._append_op({"op": "remove", "key": key})
        return True

    async def forget(self, key: str) -> bool:
        return await self.remove(key)

    async def invalidate(self, key: str) -> bool:
        """软失效：保留记录并标记时间，便于审计「这条被推翻过」。"""
        items = await self._load()
        if not any(item.key == key and item.invalidated_at is None for item in items):
            return False
        # 失效时间随操作落盘，重放时不再重新生成 —— 否则无法审计实际失效时间。
        await self._append_op({"op": "invalidate", "key": key, "at": _now()})
        return True

    # --- 读取 -------------------------------------------------------------

    async def all(self) -> list[MemoryItem]:
        return await self._load()

    async def recall(
        self,
        query: str = "",
        limit: int = 20,
        scope: str | None = None,
        applies_to: dict[str, str] | None = None,
    ) -> list[MemoryItem]:
        """召回：先过滤（active / scope / applies_to），再按关键词打分。"""
        items = await self._load()
        candidates = [
            item
            for item in items
            if item.active
            and (scope is None or item.scope == scope)
            and _applies(item, applies_to)
        ]
        if not query:
            return candidates[:limit]
        scored = [(score_item(query, item), item) for item in candidates]
        scored = [(score, item) for score, item in scored if score > 0]
        scored.sort(key=lambda pair: (pair[0], pair[1].updated_at), reverse=True)
        return [item for _, item in scored[:limit]]

    def usage(self, items: list[MemoryItem] | None = None) -> tuple[int, int]:
        """`(已用字符, 预算)`，渲染时显示占用率（Hermes 的 `[67% — 1,474/2,200]`）。"""
        active = [item for item in (items or []) if item.active]
        return self._used(active), self.max_chars

    async def render(
        self,
        *,
        header: str | None = None,
        applies_to: dict[str, str] | None = None,
        limit: int = 20,
    ) -> str:
        """渲染成注入块；没有条目时返回空串（不污染上下文）。"""
        title = header or (USER_HEADER if self.default_scope == "user" else MEMORY_HEADER)
        items = await self.recall(applies_to=applies_to, limit=limit)
        if not items:
            return ""
        used, budget = self.usage(await self.all())
        percent = int(used / budget * 100) if budget else 0
        lines = [title, f"[{percent}% — {used}/{budget} chars]"]
        lines.extend(_entry_text(item) for item in items)
        return "\n".join(lines)

    # --- 持久化：append-only journal ---------------------------------------
    #
    # 写入是「追加一条操作」，不是「读整个文件再覆盖」。这样 Agent 工具与外部 API
    # 并发写入时不会互相覆盖（读改写交错会静默丢事实），也不会留下半写的文件。
    #
    # 预算不变量在**提交时**校验：追加操作在锁内重新读取 journal、检查预算、再写，
    # 所以并发写不会把总量顶穿；`remember()` 里的提前检查只是快速失败。

    async def _load(self) -> list[MemoryItem]:
        raw = await asyncio.to_thread(self._read_text_locked)
        return _replay(raw or "", self.default_scope)

    async def _append_op(
        self, operation: dict[str, object], item: MemoryItem | None = None
    ) -> None:
        await asyncio.to_thread(self._append_op_sync, operation, item)

    def _read_text(self) -> str | None:
        try:
            return self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

    def _read_text_locked(self) -> str | None:
        """读也走同一把锁：并发追加期间读到半行会被重放逻辑当成坏行丢掉。"""
        if not self.path.exists():
            return None
        with _file_lock(self.path):
            return self._read_text()

    def _append_op_sync(self, operation: dict[str, object], item: MemoryItem | None) -> None:
        # 进程内按路径共享的锁 + 跨进程文件锁：两层都要，前者避免同进程内空转抢锁。
        with self._write_lock, _file_lock(self.path):
            text = self._read_text()
            if text is not None and _is_legacy(text):
                _migrate(self.path, text)
                text = self._read_text()
            if item is not None:
                state = _replay(text or "", self.default_scope)
                self._ensure_fits(state, item)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(f"{json.dumps(operation, ensure_ascii=False)}\n")

    def _ensure_fits(self, items: list[MemoryItem], updated: MemoryItem) -> None:
        others = [item for item in items if item.key != updated.key]
        needed = self._used(others) + len(_entry_text(updated))
        if needed > self.max_chars:
            raise MemoryFullError(
                f"memory budget exceeded: {needed} > {self.max_chars} chars; "
                "remove or consolidate entries first"
            )

    def _used(self, items: list[MemoryItem]) -> int:
        return sum(len(_entry_text(item)) for item in items if item.active)


def score_item(query: str, item: MemoryItem) -> float:
    """关键词命中率 × confidence（记忆库小，不需要检索层）。"""
    tokens = {token for token in query.lower().split() if len(token) > 1}
    if not tokens:
        return 0.0
    haystack = _entry_text(item).lower()
    hits = sum(1 for token in tokens if token in haystack)
    if hits == 0:
        return 0.0
    return (hits / len(tokens)) * item.confidence


def _entry_text(item: MemoryItem) -> str:
    parts = [f"scope={item.scope}"]
    if item.applies_to:
        rendered = ",".join(f"{key}={value}" for key, value in sorted(item.applies_to.items()))
        parts.append(f"applies_to={rendered}")
    if item.source:
        parts.append(f"source={item.source}")
    parts.append(f"confidence={item.confidence:.2f}")
    return f"{item.key}: {item.value} ({', '.join(parts)})"


def _applies(item: MemoryItem, applies_to: dict[str, str] | None) -> bool:
    """applies_to 冲突的条目不召回：换型号 / 换版本时旧事实自然失效。"""
    if not applies_to:
        return True
    # CLI 与 RAG 用 product；早期调用方用 chip。两者表达同一型号条件。
    current = {("product" if k == "chip" else k): v for k, v in applies_to.items()}
    for key, value in item.applies_to.items():
        key = "product" if key == "chip" else key
        if key in current and current[key] != value:
            return False
    return True


def _is_legacy(text: str) -> bool:
    """旧格式是整文件一个 JSON 对象 `{"items": [...]}`，不是逐行 journal。"""
    stripped = text.strip()
    if not stripped.startswith("{"):
        return False
    try:
        payload = json.loads(stripped)
    except ValueError:
        return False
    return isinstance(payload, dict) and "items" in payload


def _legacy_items(text: str, default_scope: str) -> list[MemoryItem]:
    payload = json.loads(text.strip())
    items: list[MemoryItem] = []
    for entry in payload.get("items", []):
        if not isinstance(entry, dict):
            continue
        items.append(
            MemoryItem(
                key=entry["key"],
                value=entry["value"],
                source=entry.get("source", ""),
                scope=entry.get("scope", default_scope),
                updated_at=entry.get("updated_at", ""),
                confidence=float(entry.get("confidence", 1.0)),
                applies_to=dict(entry.get("applies_to", {})),
                invalidated_at=entry.get("invalidated_at"),
            )
        )
    return items


def _migrate(path: Path, text: str) -> None:
    """把旧格式文件重写成 journal（保留原内容，不做静默丢弃）。"""
    items = _legacy_items(text, "project")
    lines = [
        json.dumps({"op": "upsert", "item": asdict(item)}, ensure_ascii=False) for item in items
    ]
    payload = "".join(f"{line}\n" for line in lines)
    temporary = path.with_suffix(path.suffix + ".migrate")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _replay(text: str, default_scope: str) -> list[MemoryItem]:
    """重放 journal（并处理旧格式）。"""
    if not text.strip():
        return []
    if _is_legacy(text):
        return _legacy_items(text, default_scope)
    items: list[MemoryItem] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            operation = json.loads(line)
        except ValueError:
            continue
        if not isinstance(operation, dict):
            continue
        items = _apply_operation(items, operation, default_scope)
    return items


def _apply_operation(
    items: list[MemoryItem], operation: dict[str, Any], default_scope: str
) -> list[MemoryItem]:
    """重放一条 journal 操作（不修改历史：失效时间取操作里记录的时间）。"""
    kind = operation.get("op")
    if kind == "upsert":
        entry = operation.get("item")
        if not isinstance(entry, dict):
            return items
        updated = MemoryItem(
            key=entry["key"],
            value=entry["value"],
            source=entry.get("source", ""),
            scope=entry.get("scope", default_scope),
            updated_at=entry.get("updated_at", ""),
            confidence=float(entry.get("confidence", 1.0)),
            applies_to=dict(entry.get("applies_to", {})),
            invalidated_at=entry.get("invalidated_at"),
        )
        return [item for item in items if item.key != updated.key] + [updated]
    if kind == "remove":
        key = operation.get("key")
        return [item for item in items if item.key != key]
    if kind == "invalidate":
        key = operation.get("key")
        recorded = operation.get("at")
        moment = recorded if isinstance(recorded, str) and recorded else _now()
        for item in items:
            if item.key == key and item.invalidated_at is None:
                item.invalidated_at = moment
        return items
    return items


def _now() -> str:
    return datetime.now(UTC).isoformat()
