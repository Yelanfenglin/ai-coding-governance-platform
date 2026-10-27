from __future__ import annotations

import json
from uuid import uuid4

import pytest

from app.contracts import ToolCall
from app.rag_tools import RagClient, RagClientConfig, rag_tools
from app.register_facts import read_register_facts
from app.spec_evidence import SpecRelationGraph
from app.tools import ToolRegistry

FIELDS = {
    "DIR.INPUT": ("DIR", 0, 1),
    "MODE.ANALOG": ("MODE", 0, 1),
    "MODE.OTHER": ("MODE", 1, 1),
    "INTCR.EN": ("INTCR", 4, 1),
}


def inspect(tmp_path, text, fields=FIELDS):
    path = tmp_path / "init.C"
    path.write_text(text, encoding="utf-8")
    return read_register_facts(tmp_path, "init.C", "init", fields)


def test_literal_macros_and_masks_attach_actual_file_locations(tmp_path):
    text = (
        "#define ENABLE (1U << 4)\nvoid init(void) {\n DIR = 0;\n"
        " MODE = 0xFF;\n INTCR = ENABLE;\n}\n"
    )
    report = inspect(tmp_path, text)
    assert report["facts"] == {
        "DIR.INPUT": 0,
        "MODE.ANALOG": 1,
        "MODE.OTHER": 1,
        "INTCR.EN": 1,
    }
    assert report["origins"]["MODE.ANALOG"][0]["path"] == "init.C"
    assert report["origins"]["MODE.ANALOG"][0]["line"] == 4
    assert report["origins"]["MODE.ANALOG"][0]["statement"] == "MODE = 0xFF;"
    assert not report["diagnostics"]
    assert (tmp_path / "init.C").read_text(encoding="utf-8") == text


def test_masked_writes_keep_other_bits_unknown_and_track_multiple_writes_on_one_line(tmp_path):
    report = inspect(tmp_path, "void init(void) { MODE &= ~(1U << 0); INTCR |= (1U << 4); }")
    assert report["facts"] == {"MODE.ANALOG": 0, "INTCR.EN": 1}
    assert "MODE.OTHER" not in report["facts"] and "DIR.INPUT" not in report["facts"]
    report = inspect(
        tmp_path, "void init(void) { MODE |= 1; MODE &= ~2; }", {"MODE.PAIR": ("MODE", 0, 2)}
    )
    assert report["facts"] == {"MODE.PAIR": 1}
    origins = report["origins"]["MODE.PAIR"]
    assert len(origins) == 2 and origins[0]["column"] != origins[1]["column"]


@pytest.mark.parametrize(
    "body",
    [
        "DIR=0; if (ready) MODE=0; else MODE=1;",
        "DIR=0; configure_port(); MODE=0;",
        "int MODE=0; DIR=0;",
        "DIR=0; MODE=HEADER_MACRO;",
        "DIR=0; *alias=0;",
        "DIR=0; return (MODE=1);",
        "\n#if FLAG\n MODE=0;\n#endif\n DIR=0;",
    ],
)
def test_unsupported_analysis_never_turns_into_a_successful_register_snapshot(tmp_path, body):
    report = inspect(tmp_path, "void init(void) { " + body + " }")
    assert report["facts"] == {} and report["diagnostics"]


def test_returns_stop_later_unreachable_writes_and_parameters_cannot_shadow_registers(tmp_path):
    report = inspect(tmp_path, "void init(void) { DIR=0; return; DIR=1; }")
    assert report["facts"] == {"DIR.INPUT": 0}
    report = inspect(tmp_path, "void init(int DIR) { DIR=0; MODE=1; }")
    assert report["facts"] == {} and report["diagnostics"]


def test_path_and_function_selection_remain_explicit(tmp_path):
    with pytest.raises(ValueError, match="所选目录"):
        read_register_facts(tmp_path, "../outside.c", "init", FIELDS)
    (tmp_path / "two.c").write_text("void init(void){} void init(void){}", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        read_register_facts(tmp_path, "two.c", "init", FIELDS)


def fixture_graph():
    hit = {
        "document_id": str(uuid4()),
        "chunk_id": str(uuid4()),
        "product": "CHIP_A",
        "version": "V1.2",
        "text": "EN=1 enables the port.",
        "page_start": 2,
        "page_end": 2,
    }
    graph = SpecRelationGraph.from_mapping(
        {
            "schema_version": 1,
            "review": {"source_review": "synthetic", "independent_review": "not_applicable"},
            "scopes": [
                {
                    "product": "CHIP_A",
                    "version": "V1.2",
                    "nodes": [{"id": "irq", "label": "irq"}, {"id": "CTRL.EN", "label": "EN"}],
                    "fields": {"CTRL.EN": {"register": "CTRL", "lsb": 4, "width": 1}},
                    "edges": [
                        {
                            "id": "enable",
                            "source": "irq",
                            "target": "CTRL.EN",
                            "relation": "requires_enable",
                            "description": "enable the port",
                            "requires": {"CTRL.EN": 1},
                            "evidence": {
                                "document_id": hit["document_id"],
                                "chunk_id": hit["chunk_id"],
                                "quotes": [hit["text"]],
                            },
                        }
                    ],
                }
            ],
        }
    )
    return graph, hit


def registry(graph, hit, workspace):
    async def transport(url, params, headers):
        return 200, hit

    tools = ToolRegistry()
    for tool in rag_tools(
        RagClient(RagClientConfig("http://rag"), transport), relations=graph, workspace=workspace
    ):
        tools.register(tool)
    return tools


async def test_graph_combines_attested_named_writes_with_source_quotes_and_preserves_unknowns(
    tmp_path,
):
    graph, hit = fixture_graph()
    tools = registry(graph, hit, tmp_path)
    args = {
        "product": "CHIP_A",
        "version": "V1.2",
        "entity": "irq",
        "read_missing": True,
        "source": {"path": "init.c", "function": "init"},
    }
    (tmp_path / "init.c").write_text("void init(void) {\n CTRL=4;\n}", encoding="utf-8")
    result = await tools.execute(ToolCall("bad", "inspect_spec_evidence", args))
    assert not result.is_error
    report = json.loads(result.content)
    assert report["ready"] and report["configuration_status"] == "unsatisfied"
    origin = report["relations"][0]["configuration"]["source_locations"]["CTRL.EN"][0]
    assert origin["line"] == 2 and origin["statement"] == "CTRL=4;"
    (tmp_path / "init.c").write_text("void init(void) { CTRL |= 16; }", encoding="utf-8")
    result = await tools.execute(ToolCall("fixed", "inspect_spec_evidence", args))
    report = json.loads(result.content)
    assert report["ready"] and report["configuration_status"] == "satisfied"
    assert report["fetched_sources"] == []
    (tmp_path / "init.c").write_text("void init(void) { if (ready) CTRL |= 16; }", encoding="utf-8")
    result = await tools.execute(ToolCall("unknown", "inspect_spec_evidence", args))
    report = json.loads(result.content)
    assert report["configuration_status"] == "unknown"
    assert report["source_analysis"]["diagnostics"]


async def test_register_facts_cannot_be_overridden_and_source_access_requires_workspace(tmp_path):
    graph, hit = fixture_graph()
    args = {
        "product": "CHIP_A",
        "version": "V1.2",
        "entity": "irq",
        "source": {"path": "missing.c", "function": "init"},
    }
    result = await registry(graph, hit, None).execute(
        ToolCall("none", "inspect_spec_evidence", args)
    )
    assert result.is_error and "selected workspace" in result.content
    result = await registry(graph, hit, tmp_path).execute(
        ToolCall(
            "override",
            "inspect_spec_evidence",
            {**args, "facts": {"CTRL.EN": 1}},
        )
    )
    assert result.is_error and "mode/context" in result.content


async def test_source_discovery_preserves_arguments_and_waits_for_topic_before_reading(tmp_path):
    graph, hit = fixture_graph()
    graph.scopes[("CHIP_A", "V1.2")].relations[0].when["input_mode"] = "digital"
    tools = registry(graph, hit, tmp_path)
    args = {
        "product": "CHIP_A",
        "version": "V1.2",
        "read_missing": True,
        "source": {"path": "missing.c", "function": "init"},
    }
    result = await tools.execute(ToolCall("discover", "inspect_spec_evidence", args))
    assert not result.is_error
    report = json.loads(result.content)
    assert report["source_analysis_pending"] and "source_analysis" not in report
    assert report["next_arguments"]["source"] == args["source"]
    assert report["context_values"] == {"input_mode": ["digital"]}
    result = await tools.execute(
        ToolCall(
            "select",
            "inspect_spec_evidence",
            {**report["next_arguments"], "entity": "irq"},
        )
    )
    assert result.is_error  # The file is actually read only after topic selection.
