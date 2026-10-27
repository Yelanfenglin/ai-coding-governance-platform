"""将实现说明关联到交付中观察到的文件、行号和检查证据。"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from .tools import FunctionTool
from .workspace_tools import local_path, result, schema


def _text(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("gb18030")


def _sources(tools: list[dict]) -> dict[tuple[str, str], dict]:
    sources = {}
    for tool in tools:
        if tool["name"] not in {"search_spec", "read_context"} or tool.get("is_error"):
            continue
        for match in re.finditer(
            r"^(?:\[\d+\] )?document=(\S+) chunk=(\S+)([^\r\n]*)",
            tool.get("content", ""), re.M,
        ):
            fields = match[3]

            def field(name, fields=fields):
                value = re.search(rf"(?:^|\s){name}=(\S+)", fields)
                return value[1] if value else ""

            section = re.search(r"section=(.*?)(?: kind=| score=|$)", fields)
            sources[(match[1], match[2])] = {
                "document_id": match[1], "chunk_id": match[2],
                "product": field("product"), "version": field("version"),
                "pages": field("pages"), "section": section[1] if section else "",
                "call_id": tool["call_id"],
            }
    return sources


class ImplementationReview:
    def __init__(self, workspace: Path, snapshot: Callable[[], dict[Path, bytes]],
                 tools: list[dict]) -> None:
        self.workspace, self.snapshot, self.tools = workspace, snapshot, tools
        self.decisions: dict[str, dict] = {}

    def tool(self) -> FunctionTool:
        text = {"type": "string", "minLength": 1}
        code_ref = schema({
            "path": text, "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
        }, ["path", "start_line", "end_line"])
        evidence_ref = schema({"document_id": text, "chunk_id": text},
                              ["document_id", "chunk_id"])
        decision = schema({
            "id": text, "title": text, "requirement": text,
            "rationale": {
                **text,
                "description": "分清规格明示、计算前提与推导结果；"
                               "引用存在不表示原文直接给出该结论。",
            },
            "code_refs": {"type": "array", "minItems": 1, "items": code_ref},
            "evidence_refs": {"type": "array", "items": evidence_ref},
            "check_call_ids": {
                "type": "array", "items": text,
                "description": "使用本轮 run_command/verify_project 返回正文的 check_call_id；"
                               "不要去日志里查 ID。同一次检查可关联多个实现，未执行检查时填 []。",
            },
            "limitations": {
                "type": "array", "items": text,
                "description": "按本轮实际输出说明未覆盖的验证范围；"
                               "已做主机编译或模拟时不要写成仅静态检查。",
            },
        }, ["id", "title", "requirement", "rationale", "code_refs", "check_call_ids"])
        return FunctionTool(
            "record_implementation",
            "记录本轮实现依据供工程审阅。最终修改和检查后批量调用一次：将用户需求、"
            "配置选择或计算理由关联到当前源码行、实际返回过的 document/chunk，以及"
            "run_command/verify_project 的 call id。没有证据或检查就留空并说明限制；"
            "不猜 ID，不把定位核对当成技术正确性证明。同 id 更新已有说明。",
            schema({"decisions": {"type": "array", "minItems": 1, "items": decision}},
                   ["decisions"]), self.record,
        )

    def record(self, args, context):
        files, sources = self.snapshot(), _sources(self.tools)
        recorded = []
        try:
            for decision in args["decisions"]:
                refs = []
                for ref in decision["code_refs"]:
                    path = local_path(self.workspace, ref["path"]).relative_to(self.workspace)
                    if path not in files:
                        raise ValueError(f"源码不在本轮交付文件范围内：{path}")
                    lines = _text(files[path]).splitlines()
                    start, end = int(ref["start_line"]), int(ref["end_line"])
                    if not 1 <= start <= end <= len(lines):
                        raise ValueError(
                            f"源码行范围无效：{path}:{start}-{end}（共{len(lines)}行）",
                        )
                    refs.append({"path": path.as_posix(), "start_line": start, "end_line": end,
                                 "excerpt": "\n".join(lines[start - 1:end])})
                evidence = []
                for ref in decision.get("evidence_refs", []):
                    key = (ref["document_id"], ref["chunk_id"])
                    if key not in sources:
                        raise ValueError(f"本轮未返回该规格引用：{key}")
                    evidence.append(sources[key])
                for call_id in decision.get("check_call_ids", []):
                    if not any(row["call_id"] == call_id and row["name"] in {
                        "run_command", "verify_project",
                    } and "content" in row for row in self.tools):
                        raise ValueError(f"找不到已返回的检查调用：{call_id}")
                recorded.append({**decision, "code_refs": refs, "evidence_refs": evidence})
        except ValueError as exc:
            return result({"recorded": False, "error": str(exc)}, error=True)
        for decision in recorded:
            self.decisions[decision["id"]] = decision
        return result({"recorded": True, "ids": [row["id"] for row in recorded],
                       "note": "已记录说明与可定位引用；交付时再次核对源码和检查版本。"})

    def finish(self, files: dict[Path, bytes], final_revision: str) -> list[dict]:
        decisions = []
        for decision in self.decisions.values():
            refs = []
            for ref in decision["code_refs"]:
                raw = files.get(Path(ref["path"]))
                excerpt = ("\n".join(_text(raw).splitlines()[
                    ref["start_line"] - 1:ref["end_line"]
                ]) if raw is not None else None)
                refs.append({**ref, "status": "located" if excerpt == ref["excerpt"]
                             else "missing" if raw is None else "changed"})
            checks = []
            for call_id in decision.get("check_call_ids", []):
                row = next(row for row in self.tools if row["call_id"] == call_id)
                try:
                    payload = json.loads(row["content"])
                except ValueError:
                    payload = {}
                matches = row["input_revision"] == row.get("output_revision") == final_revision
                checks.append({
                    "call_id": call_id, "name": row["name"], "content": row["content"],
                    "matches_delivery": matches,
                    "status": ("failed" if row.get("is_error") else
                               "outdated" if not matches else
                               "passed" if payload.get("exit_code") == 0 else "unverified"),
                })
            decisions.append({**decision, "code_refs": refs, "checks": checks})
        return decisions
