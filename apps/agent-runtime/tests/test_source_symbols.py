from __future__ import annotations

import json

import pytest

from app import cli
from app.contracts import AssistantTurn, Message, ToolCall
from app.providers.mock import stream_from_coroutine
from app.tools import ToolRegistry
from app.workspace_tools import workspace_tools


@pytest.fixture
def project(tmp_path):
    code = (
        '/* shared() in a comment is not a reference. */\n'
        '#define ACTIVE 1\n'
        'typedef struct State { int count; } State;\n'
        'int *shared(int value);\n'
        'void (*callback)(int);\n'
        'static int *shared(int value) { return 0; }\n'
        'void run(void) {\n'
        '  const char *text = "shared()";\n'
        '  callback = shared;\n'
        '  shared(1);\n'
        '}\n'
    )
    (tmp_path / "module.C").write_bytes(code.encode("utf-8"))
    registry = ToolRegistry()
    for tool in workspace_tools(tmp_path):
        registry.register(tool)
    return tmp_path, registry


async def invoke(registry, name, **args):
    row = await registry.execute(ToolCall("test", name, args))
    assert not row.is_error, row.content
    return json.loads(row.content)


async def test_definitions_declarations_pointer_types_and_exact_references(project):
    root, registry = project
    before = (root / "module.C").read_bytes()
    symbols = await invoke(registry, "find_symbols", kind="function", include_body=True)
    assert [(r["name"], r["role"], r["line"]) for r in symbols["matches"]] == [
        ("shared", "declaration", 4), ("shared", "definition", 6), ("run", "definition", 7),
    ]
    assert "6: static int *shared" in symbols["matches"][1]["content"]
    refs = await invoke(registry, "find_symbol_references", query="shared", context_lines=0)
    assert [(r["line"], r["kind"]) for r in refs["matches"]] == [
        (4, "function_declaration"), (6, "function_definition"), (9, "reference"), (10, "call"),
    ]
    assert refs["matches"][-1]["enclosing_function"] == "run"
    assert refs["resolution"] == "syntax_only_no_preprocessing"
    assert (root / "module.C").read_bytes() == before
    macro = await invoke(registry, "find_symbols", kind="macro")
    assert macro["matches"][0]["name"] == "ACTIVE"
    types = await invoke(registry, "find_symbols", kind="type")
    assert {r["name"] for r in types["matches"]} == {"State"}


async def test_live_edits_new_files_deletion_scope_and_result_limit(project):
    root, registry = project
    (root / "other.c").write_text("static void shared(void) {}\n", encoding="utf-8")
    found = await invoke(registry, "find_symbols", query="shared", max_results=1)
    assert found["total_matches"] == 3 and found["truncated"]
    await registry.execute(ToolCall("edit", "edit_file", {
        "path": "other.c", "old_text": "shared", "new_text": "renamed",
    }))
    current = await invoke(registry, "find_symbols", query="renamed", path="other.c")
    assert current["total_matches"] == 1 and current["scanned_files"] == 1
    (root / "other.c").unlink()
    assert (await invoke(registry, "find_symbols", query="renamed"))["total_matches"] == 0


@pytest.mark.parametrize("encoding", ["utf-8-sig", "gb18030"])
async def test_encoding_conditional_branches_vendor_interrupt_and_parse_diagnostics(
    project, encoding,
):
    root, registry = project
    source = ("/* 中文规格 */\r\n#if ACTIVE\r\n"
              "void interrupt high_priority handler(void) { work(); }\r\n"
              "#else\r\nvoid handler(void) { fallback(); }\r\n#endif\r\n")
    (root / "vendor.H").write_bytes(source.encode(encoding))
    row = await invoke(registry, "find_symbols", query="handler", path="vendor.H")
    assert [m["line"] for m in row["matches"]] == [3, 5]
    assert row["diagnostics"]  # Vendor-specific interrupt keywords are reported, not hidden.
    refs = await invoke(registry, "find_symbol_references", query="work", path="vendor.H")
    assert refs["matches"][0]["line"] == 3


async def test_ignored_build_outputs_and_workspace_boundary(project):
    root, registry = project
    (root / "build").mkdir()
    (root / "build/generated.c").write_text("void generated(void) {}")
    assert (await invoke(registry, "find_symbols", query="generated"))["total_matches"] == 0
    explicit = await invoke(registry, "find_symbols", query="generated", path="build/generated.c")
    assert explicit["total_matches"] == 1
    outside = await registry.execute(ToolCall("escape", "find_symbols", {"path": "../"}))
    assert outside.is_error


async def test_large_translation_unit_symbol_lookup(project):
    root, registry = project
    code = "\n".join(f"int task_{i}(int value) {{ return value + {i}; }}" for i in range(500))
    (root / "large.c").write_text(code, encoding="utf-8")
    row = await invoke(registry, "find_symbols", path="large.c", query="task_", max_results=1)
    assert row["total_matches"] == 500 and row["truncated"]
    assert row["matches"][0]["name"] == "task_0"


async def test_symbol_locations_flow_through_cli_into_targeted_read(project, monkeypatch):
    root, _ = project

    class Provider:
        step = 0

        def generate_stream(self, request):
            return stream_from_coroutine(lambda: self.generate(request))

        async def generate(self, request):
            self.step += 1
            if self.step == 1:
                assert {t.name for t in request.tools} >= {"find_symbols", "find_symbol_references"}
                call = ToolCall("locate", "find_symbols", {"query": "run", "kind": "function"})
            elif self.step == 2:
                symbol = json.loads(request.messages[-1].content)["matches"][0]
                call = ToolCall("read", "read_file", {
                    "path": symbol["path"], "start_line": symbol["start_line"],
                    "max_lines": symbol["end_line"] - symbol["start_line"] + 1,
                })
            else:
                assert "10:   shared(1);" in request.messages[-1].content
                return AssistantTurn(Message("assistant", "Located the call in run()."), "stop")
            return AssistantTurn(Message("assistant", tool_calls=[call]), "tool_calls")

    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **_: Provider())
    args = cli.build_parser().parse_args([
        "locate run", "--workspace", str(root), "--base-url", "http://unused.invalid",
        "--no-memory",
    ])
    assert await cli.run_once(args, console=False) == 0
