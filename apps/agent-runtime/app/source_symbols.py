"""Read-only C syntax navigation. Names are candidates, not linker-resolved symbols."""
from __future__ import annotations

import asyncio
import os
from itertools import islice
from pathlib import Path

import tree_sitter_c
from tree_sitter import Language, Node, Parser

from .demos.firmware import SourceFile
from .tools import FunctionTool
from .workspace_tools import PATH, local_path, result, schema

C_LANGUAGE = Language(tree_sitter_c.language())
IGNORED = {".git", ".venv", "node_modules", "__pycache__", ".agent-runtime",
           ".agent-demo-backups", "output", "build", "dist"}


def walk(node: Node):
    pending = [node]
    while pending:
        current = pending.pop()
        yield current
        pending.extend(reversed(current.named_children))


def spelling(node: Node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8")


def ancestor(node: Node, kinds: set[str]) -> Node | None:
    current = node.parent
    while current is not None:
        if current.type in kinds:
            return current
        current = current.parent
    return None


def declarator_name(node: Node | None) -> Node | None:
    while node is not None:
        if node.type in {"identifier", "type_identifier"}:
            return node
        child = node.child_by_field_name("declarator")
        if child is None and node.type == "parenthesized_declarator" and node.named_children:
            child = node.named_children[0]
        node = child
    return None


def declarations(tree: Node):
    """Keep function pointers distinct from functions; retain C preprocessor branches."""
    for node in walk(tree):
        if node.type == "function_declarator":
            name = node.child_by_field_name("declarator")
            if name is None or name.type != "identifier":
                continue
            owner = ancestor(node, {"function_definition", "declaration",
                                    "parameter_declaration", "field_declaration"})
            if owner is not None and owner.type in {"function_definition", "declaration"}:
                role = "definition" if owner.type == "function_definition" else "declaration"
                yield name, owner, "function", role
        elif node.type in {"preproc_def", "preproc_function_def"}:
            name = node.child_by_field_name("name")
            if name is not None:
                yield name, node, "macro", "definition"
        elif node.type == "type_definition":
            for child in node.children_by_field_name("declarator"):
                name = declarator_name(child)
                if name is not None:
                    yield name, node, "type", "definition"
        elif node.type in {"struct_specifier", "union_specifier", "enum_specifier"}:
            name = node.child_by_field_name("name")
            if name is not None and node.child_by_field_name("body") is not None:
                yield name, node, "type", "definition"


def source_paths(root: Path, selected: str):
    base = local_path(root, selected)
    if not base.exists():
        raise FileNotFoundError(selected)
    if base.is_file():
        if base.suffix.lower() not in {".c", ".h"}:
            raise ValueError("符号导航当前支持 C 的 .c/.h 文件（含大写扩展名）")
        yield base
        return
    for directory, dirs, files in os.walk(base, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in IGNORED)
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix.lower() in {".c", ".h"}:
                yield path


def inspect_source(root: Path, args: dict, *, references: bool = False) -> dict:
    matches, diagnostics = [], []
    scanned = 0
    parser = Parser(C_LANGUAGE)
    query = args.get("query", "")
    for path in source_paths(root, args.get("path", ".")):
        relative = path.relative_to(root).as_posix()
        try:
            local_path(root, relative)  # A symlink must not expose a file outside this workspace.
            document = SourceFile.load(path)
        except (OSError, ValueError, UnicodeError) as exc:
            diagnostics.append({"path": relative, "error": str(exc)})
            continue
        scanned += 1
        source = document.text.encode("utf-8")
        parsed = parser.parse(source)
        tree = parsed.root_node
        lines = document.text.splitlines()
        if tree.has_error:
            diagnostic = {"path": relative, "has_parse_errors": True}
            if args.get("include_diagnostics", False):
                errors = (n for n in walk(tree) if n.is_error or n.is_missing)
                diagnostic["parse_errors"] = [
                    {"line": n.start_point.row + 1, "node": n.type} for n in islice(errors, 5)
                ]
            diagnostics.append(diagnostic)
        declared = list(declarations(tree))
        if references:
            roles = {name.start_byte: f"{kind}_{role}" for name, _, kind, role in declared}
            for node in walk(tree):
                if node.type not in {"identifier", "type_identifier", "field_identifier"}:
                    continue
                if spelling(node, source) != query:
                    continue
                owner = ancestor(node, {"function_definition"})
                caller = declarator_name(owner.child_by_field_name("declarator")) if owner else None
                kind = roles.get(node.start_byte, "reference")
                if (node.parent is not None and node.parent.type == "call_expression"
                        and node.parent.child_by_field_name("function") == node):
                    kind = "call"
                context = args.get("context_lines", 1)
                start = max(0, node.start_point.row - context)
                end = min(len(lines), node.end_point.row + context + 1)
                matches.append({
                    "path": relative, "line": node.start_point.row + 1, "kind": kind,
                    "enclosing_function": spelling(caller, source) if caller else None,
                    "context": "\n".join(f"{i+1}: {lines[i]}" for i in range(start, end)),
                })
        else:
            for name, owner, kind, role in declared:
                value = spelling(name, source)
                if query.casefold() not in value.casefold():
                    continue
                if args.get("kind", "all") not in {"all", kind}:
                    continue
                body = owner.child_by_field_name("body")
                end = body.start_byte if kind == "function" and body else owner.end_byte
                signature = " ".join(source[owner.start_byte:end].decode("utf-8").split())
                entry = {"name": value, "kind": kind, "role": role, "path": relative,
                         "line": name.start_point.row + 1,
                         "start_line": owner.start_point.row + 1,
                         "end_line": owner.end_point.row + 1, "signature": signature[:400]}
                if args.get("include_body", False):
                    start = owner.start_point.row
                    end_line = min(owner.end_point.row + 1, start + 40)
                    entry["content"] = "\n".join(f"{i+1}: {lines[i]}"
                                                   for i in range(start, end_line))
                    entry["content_has_more"] = end_line < owner.end_point.row + 1
                matches.append(entry)
    omitted = 0
    match_mode = "exact" if references else args.get("match", "auto")
    if not references and match_mode != "contains" and query:
        exact = [entry for entry in matches if entry["name"] == query]
        if exact or match_mode == "exact":
            omitted, matches, match_mode = len(matches) - len(exact), exact, "exact"
        else:
            match_mode = "contains"
    elif match_mode == "auto":
        match_mode = "contains"
    limit = args.get("max_results", 20)
    selected = matches[:limit]
    visible_paths = {entry["path"] for entry in selected}
    base = local_path(root, args.get("path", "."))
    if base.is_file():
        visible_paths.add(base.relative_to(root).as_posix())
    detailed = args.get("include_diagnostics", False)
    visible_diagnostics = diagnostics if detailed else [
        entry for entry in diagnostics if entry["path"] in visible_paths
    ]
    return {"language": "C", "resolution": "syntax_only_no_preprocessing",
            "scanned_files": scanned, "total_matches": len(matches),
            "match_mode": match_mode, "omitted_partial_matches": omitted,
            "matches": selected, "truncated": len(matches) > limit,
            "files_with_parse_issues": sum(bool(d.get("has_parse_errors")) for d in diagnostics),
            "unreadable_files": sum("error" in d for d in diagnostics),
            "diagnostics_scope": "all" if detailed else "matched",
            "diagnostics": visible_diagnostics[:20],
            "diagnostics_truncated": len(visible_diagnostics) > 20}


def symbol_tools(root: Path) -> list[FunctionTool]:
    common = {"path": PATH, "max_results": {"type": "integer", "minimum": 1, "maximum": 100},
              "include_diagnostics": {"type": "boolean"}}

    async def find(args, context):
        return result(await asyncio.to_thread(inspect_source, root, args))

    async def references(args, context):
        return result(await asyncio.to_thread(inspect_source, root, args, references=True))

    return [
        FunctionTool("find_symbols", "定位C工程的函数定义/声明、宏与类型，返回签名和源码行号；"
                     "默认完整名称精确匹配，未命中再包含匹配；match=contains可显式广搜。"
                     "空query列目录，include_body附带最多40行原文，path可缩小范围。"
                     "默认只附命中文件的诊断摘要；include_diagnostics展开扫描诊断。",
                     schema({**common, "query": {"type": "string"},
                             "match": {"type": "string", "enum": ["auto", "exact", "contains"]},
                             "kind": {"type": "string",
                                      "enum": ["all", "function", "macro", "type"]},
                             "include_body": {"type": "boolean"}}), find,
                     execution_mode="parallel"),
        FunctionTool("find_symbol_references", "按C标识符精确名称找调用与引用，返回所属函数和原文。"
                     "这是语法候选：不区分跨文件同名符号、不解析指针目标、不展开宏；"
                     "排除注释与字符串。path限定范围；include_diagnostics按需展开扫描诊断。",
                     schema({**common, "query": PATH,
                             "context_lines": {"type": "integer", "minimum": 0, "maximum": 5}},
                            ["query"]), references, execution_mode="parallel"),
    ]
