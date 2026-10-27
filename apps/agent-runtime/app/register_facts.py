"""Read named register writes in one straight-line C function, without executing it.

This does not resolve headers/addresses, calls or build-time branches. Masked writes
retain unknown bits instead of filling them with reset values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from tree_sitter import Node, Parser

from .demos.firmware import SourceFile
from .source_symbols import C_LANGUAGE, declarations, declarator_name, spelling, walk
from .workspace_tools import local_path


def _constant(node: Node, source: bytes, macros: dict, seen: tuple = ()) -> int | None:
    text = spelling(node, source)
    if node.type == "number_literal":
        literal = re.sub(r"[uUlL]+$", "", text)
        try:
            base = 8 if re.fullmatch(r"0[0-7]+", literal) else 0
            return int(literal, base)
        except ValueError:
            return None
    if node.type == "parenthesized_expression" and len(node.named_children) == 1:
        return _constant(node.named_children[0], source, macros, seen)
    if node.type == "identifier":
        value = macros.get(text)
        if value is None or text in seen or len(seen) >= 16:
            return None
        fragment = f"void _constant(void) {{ _value = {value}; }}".encode()
        root = Parser(C_LANGUAGE).parse(fragment).root_node
        if root.has_error:
            return None
        assignment = next(n for n in walk(root) if n.type == "assignment_expression")
        return _constant(assignment.child_by_field_name("right"), fragment, macros, (*seen, text))
    if node.type == "unary_expression":
        value = _constant(node.child_by_field_name("argument"), source, macros, seen)
        if value is None:
            return None
        op = spelling(node.child_by_field_name("operator"), source)
        return {"~": lambda: ~value, "+": lambda: value, "-": lambda: -value}.get(
            op, lambda: None
        )()
    if node.type == "binary_expression":
        left = _constant(node.child_by_field_name("left"), source, macros, seen)
        right = _constant(node.child_by_field_name("right"), source, macros, seen)
        if left is None or right is None:
            return None
        op = spelling(node.child_by_field_name("operator"), source)
        if op in {"<<", ">>"} and (left < 0 or not 0 <= right < 64):
            return None
        return {
            "|": lambda: left | right,
            "&": lambda: left & right,
            "^": lambda: left ^ right,
            "+": lambda: left + right,
            "-": lambda: left - right,
            "<<": lambda: left << right,
            ">>": lambda: left >> right,
        }.get(op, lambda: None)()
    return None


@dataclass
class _Bits:
    mask: int
    known: int = 0
    value: int = 0
    origins: dict[int, dict] = field(default_factory=dict)

    def write(self, op: str, value: int | None, location: dict) -> None:
        if value is None or op not in {"=", "|=", "&=", "^="}:
            self.known = self.value = 0
            self.origins.clear()
            return
        if op == "=":
            changed = self.known = self.mask
            self.value = value & self.mask
        elif op == "|=":
            changed = value & self.mask
            self.known |= changed
            self.value |= changed
        elif op == "&=":
            changed = ~value & self.mask
            self.known |= changed
            self.value &= value
        else:
            changed = value & self.known
            self.value ^= value & self.mask
        for bit in range(self.mask.bit_length()):
            if changed & (1 << bit):
                self.origins[bit] = location


def read_register_facts(
    workspace: Path,
    path: str,
    function: str,
    fields: dict[str, tuple[str, int, int]],
) -> dict:
    root = workspace.resolve()
    target = local_path(root, path)
    if target.suffix.lower() not in {".c", ".h"}:
        raise ValueError("register inspection supports C .c/.h files")
    document = SourceFile.load(target)
    source = document.text.encode("utf-8")
    tree = Parser(C_LANGUAGE).parse(source).root_node
    matches = [
        owner
        for name, owner, kind, role in declarations(tree)
        if kind == "function" and role == "definition" and spelling(name, source) == function
    ]
    if len(matches) != 1:
        raise ValueError("source function must identify exactly one definition; use find_symbols")
    selected = matches[0]
    body = selected.child_by_field_name("body")
    relative = target.relative_to(root).as_posix()
    states = {}
    for register, offset, width in fields.values():
        states.setdefault(register, _Bits(0)).mask |= ((1 << width) - 1) << offset
    result = {
        "path": relative,
        "function": function,
        "function_line": selected.start_point.row + 1,
        "facts": {},
        "origins": {},
        "writes": [],
        "diagnostics": [],
        "basis": "Named writes in this selected straight-line function only; not execution, "
        "call-path, header/address binding, compiler or hardware verification.",
    }

    def diagnostic(node: Node, reason: str):
        result["diagnostics"].append({"line": node.start_point.row + 1, "reason": reason})

    if tree.has_error:
        diagnostic(selected, "C syntax has parse errors; register facts remain unknown")
        return result
    if any(n.type.startswith("preproc_") for n in walk(body)) or any(
        n.type.startswith("preproc_") for n in _parents(selected)
    ):
        diagnostic(selected, "conditional/preprocessed function body is unsupported")
        return result
    for node in walk(selected):
        if node.type in {"parameter_declaration", "declaration"}:
            for child in node.children_by_field_name("declarator"):
                name = declarator_name(child)
                if name is not None and spelling(name, source) in states:
                    diagnostic(name, "local/parameter name shadows a modeled register")
        if node.type in {
            "if_statement",
            "for_statement",
            "while_statement",
            "do_statement",
            "switch_statement",
            "goto_statement",
            "labeled_statement",
            "call_expression",
            "asm_statement",
        }:
            diagnostic(
                node, "control flow, calls or assembly are unsupported; facts remain unknown"
            )
    if result["diagnostics"]:
        return result

    # Only same-file, unconditional object macros before this function are available.
    macros = {}
    for node in tree.named_children:
        if node.start_byte >= selected.start_byte:
            break
        if node.type == "preproc_include":
            macros.clear()  # Included text may redefine an earlier object macro.
        for definition in walk(node):
            if definition.type == "preproc_def":
                name = spelling(definition.child_by_field_name("name"), source)
                value = definition.child_by_field_name("value")
                macros[name] = (
                    spelling(value, source)
                    if definition == node and name not in macros and value
                    else None
                )
            elif definition.type == "preproc_call":
                match = re.match(r"#\s*undef\s+(\w+)", spelling(definition, source))
                if match:
                    macros[match[1]] = None

    def statement(node: Node) -> bool:
        if node.type == "comment":
            return True
        if node.type == "compound_statement":
            return all(statement(child) for child in node.named_children)
        if node.type == "return_statement":
            if node.named_children and _constant(node.named_children[0], source, macros) is None:
                diagnostic(node, "unsupported return expression")
            return False
        if node.type == "declaration":
            if any(n.type in {"assignment_expression", "update_expression"} for n in walk(node)):
                diagnostic(node, "side effects inside a declaration are unsupported")
            return True
        assignment = (
            node.named_children[0]
            if node.type == "expression_statement" and node.named_children
            else None
        )
        if assignment is None or assignment.type != "assignment_expression":
            diagnostic(node, "unsupported statement; register facts remain unknown")
            return True
        left = assignment.child_by_field_name("left")
        if left.type != "identifier":
            diagnostic(left, "indirect/aliased writes are unsupported")
            return True
        register = spelling(left, source)
        right = assignment.child_by_field_name("right")
        if any(n.type in {"assignment_expression", "update_expression"} for n in walk(right)):
            diagnostic(right, "nested write side effects are unsupported")
            return True
        if register not in states:
            return True
        op = spelling(assignment.child_by_field_name("operator"), source)
        value = _constant(right, source, macros)
        location = {
            "path": relative,
            "line": node.start_point.row + 1,
            "column": node.start_point.column + 1,
            "end_line": node.end_point.row + 1,
            "statement": spelling(node, source),
            "register": register,
            "operator": op,
            "expression": spelling(right, source),
            "constant_value": value,
        }
        result["writes"].append(location)
        states[register].write(op, value, location)
        if value is None or op not in {"=", "|=", "&=", "^="}:
            result["diagnostics"].append(
                {"line": location["line"], "reason": f"unresolved {register} write"}
            )
        return True

    statement(body)
    if result["diagnostics"]:
        return result
    for name, (register, offset, width) in fields.items():
        bits = states[register]
        mask = ((1 << width) - 1) << offset
        if bits.known & mask != mask:
            continue
        result["facts"][name] = (bits.value >> offset) & ((1 << width) - 1)
        locations = {
            (bits.origins[bit]["line"], bits.origins[bit]["column"]): bits.origins[bit]
            for bit in range(offset, offset + width)
        }
        result["origins"][name] = [locations[position] for position in sorted(locations)]
    return result


def _parents(node: Node):
    current = node.parent
    while current is not None:
        yield current
        current = current.parent
