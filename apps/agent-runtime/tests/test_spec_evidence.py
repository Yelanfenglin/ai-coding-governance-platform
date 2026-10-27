from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import pytest

from app.contracts import AssistantTurn, Message, ToolCall
from app.loop import RunOptions, run_agent, user_message
from app.providers.mock import MockProvider
from app.rag_tools import RagClient, RagClientConfig, rag_tools
from app.spec_evidence import SpecEvidence, SpecRelationGraph
from app.tools import ToolRegistry

SCOPE = {"product": "CHIP_A", "version": "V1.2", "entity": "irq"}


def graph_data():
    document = str(uuid4())
    hits = [
        {
            "document_id": document,
            "chunk_id": str(uuid4()),
            "product": "CHIP_A",
            "version": "V1.2",
            "text": text,
            "kind": "TEXT",
            "section_path": f"5.{index}",
            "page_start": index + 1,
            "page_end": index + 1,
            "score": 0.8,
        }
        for index, text in enumerate(("BIT_A = 1 enables the pin.", "BIT_B = 1 enables the port."))
    ]
    payload = {
        "schema_version": 1,
        "review": {"source_review": "synthetic_fixture", "independent_review": "not_applicable"},
        "scopes": [
            {
                "product": "CHIP_A",
                "version": "V1.02",
                "nodes": [{"id": node, "label": node} for node in ("irq", "BIT_A", "BIT_B")],
                "edges": [
                    {
                        "id": f"need_{index}",
                        "source": "irq",
                        "target": f"BIT_{letter}",
                        "relation": "requires_enable",
                        "description": f"Enable {letter}",
                        "evidence": {
                            "document_id": document,
                            "chunk_id": hit["chunk_id"],
                            "quotes": [hit["text"]],
                        },
                    }
                    for index, (letter, hit) in enumerate(zip(("A", "B"), hits, strict=True))
                ],
            }
        ],
    }
    return payload, hits


def chain_data():
    payload, hits = graph_data()
    scope = payload["scopes"][0]
    scope["topics"] = ["irq"]
    scope["nodes"][0]["aliases"] = ["pin interrupt"]
    scope["edges"][0]["requires"] = {"BIT_A": 1}
    scope["edges"][1]["source"] = "BIT_A"
    scope["edges"][1]["requires"] = {"BIT_B": 1}
    return payload, hits


def test_multihop_sources_and_configuration_gaps_are_reported_separately():
    payload, hits = chain_data()
    tracker = SpecEvidence(SpecRelationGraph.from_mapping(payload))
    for hit in hits:
        tracker.record(hit, hit["text"])
    report = tracker.inspect("CHIP_A", "V1.2", "PIN INTERRUPT", facts={"BIT_A": 1, "BIT_B": 0})
    assert report["ready"] and report["missing"] == []
    assert [r["depth"] for r in report["relations"]] == [1, 2]
    assert report["relations"][1]["path"] == ["irq", "BIT_A", "BIT_B"]
    assert report["configuration_status"] == "unsatisfied"
    assert report["next_action"] == "report_configuration_gaps"
    assert report["relations"][1]["configuration"]["different"] == {
        "BIT_B": {"expected": 1, "provided": 0},
    }
    incomplete = tracker.inspect("CHIP_A", "V1.2", "irq", facts={"BIT_A": 1})
    assert incomplete["ready"] and incomplete["configuration_status"] == "unknown"
    complete = tracker.inspect("CHIP_A", "V1.2", "irq", facts={"BIT_A": 1, "BIT_B": 1})
    assert complete["configuration_status"] == "satisfied"


async def test_conditional_branches_fetch_only_applicable_sources_and_keep_unknowns_explicit():
    payload, hits = chain_data()
    payload["scopes"][0]["edges"][1]["when"] = {"mode": "digital"}
    calls = []

    async def transport(url, params, headers):
        calls.append(url)
        return 200, next(hit for hit in hits if hit["chunk_id"] in url)

    registry = registry_for(payload, transport)
    common = {**SCOPE, "read_missing": True}
    unknown = await registry.execute(ToolCall("unknown", "inspect_spec_evidence", common))
    report = json.loads(unknown.content)
    assert len(calls) == 1 and not report["ready"]
    assert report["unresolved_conditions"] == [
        {"id": "need_1", "missing_facts": {"mode": "digital"}}
    ]
    disabled = await registry.execute(
        ToolCall("disabled", "inspect_spec_evidence", {**common, "facts": {"mode": "analog"}})
    )
    report = json.loads(disabled.content)
    assert report["ready"] and len(calls) == 1
    assert [r["id"] for r in report["excluded_relations"]] == ["need_1"]
    enabled = await registry.execute(
        ToolCall("enabled", "inspect_spec_evidence", {**common, "facts": {"mode": "digital"}})
    )
    report = json.loads(enabled.content)
    assert report["ready"] and len(calls) == 2
    assert report["configuration_status"] == "unknown"


def test_depth_boundary_does_not_claim_complete_evidence():
    payload, hits = chain_data()
    tracker = SpecEvidence(SpecRelationGraph.from_mapping(payload))
    for hit in hits:
        tracker.record(hit, hit["text"])
    report = tracker.inspect("CHIP_A", "V1.2", "irq", max_depth=1)
    assert not report["ready"] and report["missing"] == []
    assert report["traversal_boundaries"] == [
        {"kind": "depth_limit", "node": "BIT_A", "relation": "need_1"},
    ]
    assert tracker.inspect("CHIP_A", "V1.2", "irq", max_depth=2)["ready"]


def test_cycles_across_sibling_paths_cannot_claim_complete_evidence():
    payload, hits = graph_data()
    scope = payload["scopes"][0]
    scope["topics"] = ["irq"]
    scope["edges"].extend(
        [
            {**scope["edges"][0], "id": "a_to_b", "source": "BIT_A", "target": "BIT_B"},
            {**scope["edges"][1], "id": "b_to_a", "source": "BIT_B", "target": "BIT_A"},
        ]
    )
    tracker = SpecEvidence(SpecRelationGraph.from_mapping(payload))
    for hit in hits:
        tracker.record(hit, hit["text"])
    report = tracker.inspect("CHIP_A", "V1.2", "irq")
    assert not report["ready"] and report["missing"] == []
    assert report["traversal_boundaries"] == [{"kind": "cycle", "nodes": ["BIT_A", "BIT_B"]}]


def test_inapplicable_graph_has_no_vacuous_ready_and_values_do_not_assume_defaults():
    payload, hits = graph_data()
    for edge in payload["scopes"][0]["edges"]:
        edge["when"] = {"mode": "digital"}
        edge["requires"] = {"EN": 1}
    tracker = SpecEvidence(SpecRelationGraph.from_mapping(payload))
    for hit in hits:
        tracker.record(hit, hit["text"])
    disabled = tracker.inspect("CHIP_A", "V1.2", "irq", facts={"mode": "analog"})
    assert not disabled["ready"] and disabled["next_action"] == "use_ordinary_retrieval"
    wrong_type = tracker.inspect("CHIP_A", "V1.2", "irq", facts={"mode": "digital", "EN": True})
    assert wrong_type["configuration_status"] == "unsatisfied"


def test_graph_growth_does_not_change_tool_definition():
    first, _ = graph_data()
    second, _ = chain_data()
    second["scopes"][0]["nodes"][0]["label"] = "a different chip feature"
    a = SpecEvidence(SpecRelationGraph.from_mapping(first)).tool()
    b = SpecEvidence(SpecRelationGraph.from_mapping(second)).tool()
    assert a.description == b.description and a.input_schema == b.input_schema


async def test_unknown_topic_returns_scoped_choices_without_fetching_or_guessing():
    payload, _ = chain_data()

    async def transport(url, params, headers):
        raise AssertionError("unknown topics must not trigger source reads")

    registry = registry_for(payload, transport)
    result = await registry.execute(
        ToolCall(
            "unknown",
            "inspect_spec_evidence",
            {**SCOPE, "entity": "change interrupt", "read_missing": True},
        )
    )
    report = json.loads(result.content)
    assert result.is_error and not report["ready"]
    assert report["available_entities"] == [{"id": "irq", "label": "irq"}]
    assert report["next_action"] == "select_entity"


def test_raw_registers_use_scoped_field_positions_and_do_not_invent_reset_values():
    payload, hits = graph_data()
    scope = payload["scopes"][0]
    scope["fields"] = {"BIT_A": {"register": "CTRL", "lsb": 2, "width": 1}}
    scope["edges"][0]["requires"] = {"BIT_A": 1}
    tracker = SpecEvidence(SpecRelationGraph.from_mapping(payload))
    for hit in hits:
        tracker.record(hit, hit["text"])
    assert (
        tracker.inspect("CHIP_A", "V1.2", "irq", facts={"CTRL": 4})["configuration_status"]
        == "satisfied"
    )
    assert tracker.inspect("CHIP_A", "V1.2", "irq", facts={})["configuration_status"] == "unknown"
    scope["fields"]["BIT_A"]["lsb"] = 4
    other = SpecEvidence(SpecRelationGraph.from_mapping(payload))
    assert (
        other.inspect("CHIP_A", "V1.2", "irq", facts={"CTRL": 4})["configuration_status"]
        == "unsatisfied"
    )
    with pytest.raises(ValueError, match="conflicting raw"):
        tracker.inspect("CHIP_A", "V1.2", "irq", facts={"CTRL": 4, "BIT_A": 0})
    with pytest.raises(ValueError, match="nonnegative integer"):
        tracker.inspect("CHIP_A", "V1.2", "irq", facts={"CTRL": "0x04"})


def test_mode_selected_status_reports_supplied_field_value_including_zero():
    payload, hits = graph_data()
    scope = payload["scopes"][0]
    scope["fields"] = {"BIT_A": {"register": "CTRL", "lsb": 1, "width": 1}}
    scope["edges"][0].update(relation="reports_status", when={"mode": "led"})
    scope["edges"][1].update(relation="reports_mode", when={"mode": "pwm"}, target="BIT_A")
    tracker = SpecEvidence(SpecRelationGraph.from_mapping(payload))
    for hit in hits:
        tracker.record(hit, hit["text"])
    led = tracker.inspect("CHIP_A", "V1.2", "irq", facts={"mode": "led", "CTRL": 0})
    assert led["ready"] and led["relations"][0]["field_value"] == 0
    assert [row["id"] for row in led["relations"]] == ["need_0"]
    pwm = tracker.inspect("CHIP_A", "V1.2", "irq", facts={"mode": "pwm", "CTRL": 2})
    assert pwm["ready"] and pwm["relations"][0]["field_value"] == 1
    assert [row["id"] for row in pwm["relations"]] == ["need_1"]
    unknown = tracker.inspect("CHIP_A", "V1.2", "irq", facts={"mode": "led"})
    assert unknown["relations"][0]["field_value"] is None
    assert unknown["configuration_status"] == "not_checked"


@pytest.mark.parametrize("invalid", [{"when": []}, {"requires": {"EN": None}}])
def test_invalid_condition_models_are_rejected(invalid):
    payload, _ = graph_data()
    payload["scopes"][0]["edges"][0].update(invalid)
    with pytest.raises(ValueError, match="facts and conditions"):
        SpecRelationGraph.from_mapping(payload)


def registry_for(payload, transport):
    registry = ToolRegistry()
    for tool in rag_tools(
        RagClient(RagClientConfig("http://rag"), transport),
        relations=SpecRelationGraph.from_mapping(payload),
    ):
        registry.register(tool)
    return registry


async def inspect(registry):
    result = await registry.execute(ToolCall("inspect", "inspect_spec_evidence", SCOPE))
    assert not result.is_error, result.content
    return json.loads(result.content)


async def test_graph_hints_do_not_count_as_observed_evidence_and_only_missing_source_is_read():
    payload, hits = graph_data()
    calls = []

    async def transport(url, params, headers):
        calls.append(url)
        return (200, hits[:1]) if url.endswith("/search") else (200, hits[1])

    registry = registry_for(payload, transport)
    initial = await inspect(registry)
    assert initial["missing"] == ["need_0", "need_1"] and not initial["ready"]
    await registry.execute(ToolCall("search", "search_spec", {"question": "interrupt"}))
    partial = await inspect(registry)
    assert partial["missing"] == ["need_1"]
    missing = partial["relations"][1]["source_reference"]
    await registry.execute(ToolCall("read", "read_context", missing))
    complete = await inspect(registry)
    assert complete["ready"] and complete["missing"] == []
    assert complete["next_action"] == "continue_task_using_evidence"
    assert complete["relations"][1]["observed_source"]["page_start"] == 2
    assert calls == [
        "http://rag/search",
        f"http://rag/documents/{hits[1]['document_id']}/context/{hits[1]['chunk_id']}",
    ]


async def test_excerpt_hidden_text_and_failed_reads_cannot_satisfy_a_requirement():
    payload, hits = graph_data()

    async def transport(url, params, headers):
        return (200, hits) if url.endswith("/search") else (404, {"detail": hits[0]["text"]})

    registry = registry_for(payload, transport)
    await registry.execute(ToolCall("prefix", "search_spec", {"question": "irq", "max_chars": 3}))
    assert (await inspect(registry))["missing"] == ["need_0", "need_1"]
    failed = await registry.execute(
        ToolCall(
            "read",
            "read_context",
            {
                "document_id": hits[0]["document_id"],
                "chunk_id": hits[0]["chunk_id"],
            },
        )
    )
    assert failed.is_error and not (await inspect(registry))["ready"]
    await registry.execute(ToolCall("full", "search_spec", {"question": "irq"}))
    assert (await inspect(registry))["ready"]
    # An explicit later prefix must not erase an already returned complete observation.
    await registry.execute(ToolCall("prefix2", "search_spec", {"question": "irq", "max_chars": 3}))
    assert (await inspect(registry))["ready"]


@pytest.mark.parametrize("change", [{"product": "OTHER"}, {"version": "V2.0"}])
async def test_matching_quotes_in_wrong_scope_are_rejected(change):
    payload, hits = graph_data()

    async def transport(url, params, headers):
        return 200, [{**hit, **change} for hit in hits]

    registry = registry_for(payload, transport)
    await registry.execute(ToolCall("read", "search_spec", {"question": "irq"}))
    report = await inspect(registry)
    assert not report["ready"]
    assert {edge["status"] for edge in report["relations"]} == {"wrong_scope"}


async def test_parent_and_child_registries_do_not_share_observations():
    payload, hits = graph_data()

    async def transport(url, params, headers):
        return 200, hits

    parent, child = registry_for(payload, transport), registry_for(payload, transport)
    await parent.execute(ToolCall("read", "search_spec", {"question": "irq"}))
    assert (await inspect(parent))["ready"]
    assert not (await inspect(child))["ready"]


async def test_directed_reads_are_bounded_and_stop_when_sources_are_present():
    payload, hits = graph_data()
    calls = []

    async def transport(url, params, headers):
        calls.append(url)
        return 200, next(hit for hit in hits if hit["chunk_id"] in url)

    registry = registry_for(payload, transport)
    arguments = {**SCOPE, "read_missing": True, "source_limit": 1}
    first = await registry.execute(ToolCall("first", "inspect_spec_evidence", arguments))
    assert json.loads(first.content)["missing"] == ["need_1"] and len(calls) == 1
    second = await registry.execute(ToolCall("second", "inspect_spec_evidence", arguments))
    assert json.loads(second.content)["ready"] and len(calls) == 2
    third = await registry.execute(ToolCall("third", "inspect_spec_evidence", arguments))
    assert json.loads(third.content)["fetched_sources"] == [] and len(calls) == 2


async def test_directed_reads_reject_source_identity_mismatch_and_transport_failure():
    payload, hits = graph_data()

    async def transport(url, params, headers):
        if hits[1]["chunk_id"] in url:
            raise RuntimeError("offline")
        return 200, {**hits[0], "chunk_id": str(uuid4())}

    registry = registry_for(payload, transport)
    result = await registry.execute(
        ToolCall(
            "read",
            "inspect_spec_evidence",
            {
                **SCOPE,
                "read_missing": True,
            },
        )
    )
    report = json.loads(result.content)
    assert report["missing"] == ["need_0", "need_1"] and not report["ready"]
    assert {row["error"] for row in report["fetched_sources"]} == {
        "source_identity_mismatch",
        "RuntimeError",
    }


async def test_unreturned_body_text_does_not_satisfy_another_graph_topic():
    payload, hits = graph_data()
    scope = payload["scopes"][0]
    scope["nodes"].append({"id": "other", "label": "other"})
    scope["edges"][1]["source"] = "other"
    scope["edges"][1]["evidence"]["chunk_id"] = hits[0]["chunk_id"]

    async def transport(url, params, headers):
        return 200, {**hits[0], "text": hits[0]["text"] + "\n" + hits[1]["text"]}

    registry = registry_for(payload, transport)
    result = await registry.execute(
        ToolCall(
            "read",
            "inspect_spec_evidence",
            {
                **SCOPE,
                "read_missing": True,
            },
        )
    )
    assert json.loads(result.content)["ready"]
    other = await registry.execute(
        ToolCall(
            "other",
            "inspect_spec_evidence",
            {
                **SCOPE,
                "entity": "other",
            },
        )
    )
    assert json.loads(other.content)["missing"] == ["need_1"]


async def test_directed_read_cancellation_propagates_without_marking_evidence_ready():
    payload, _ = graph_data()
    started = asyncio.Event()

    async def transport(url, params, headers):
        started.set()
        await asyncio.Event().wait()

    registry = registry_for(payload, transport)
    task = asyncio.create_task(
        registry.execute(
            ToolCall(
                "read",
                "inspect_spec_evidence",
                {
                    **SCOPE,
                    "read_missing": True,
                    "source_limit": 1,
                },
            )
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not (await inspect(registry))["ready"]


def test_graph_scope_and_structure_are_explicit(tmp_path):
    payload, _ = graph_data()
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    tracker = SpecEvidence(SpecRelationGraph.load(path))
    assert tracker.inspect("CHIP_A", "V01.002")["available_entities"] == [
        {"id": "irq", "label": "irq"},
    ]
    with pytest.raises(ValueError, match="product/version"):
        tracker.inspect("CHIP_A", "V2.0", "irq")
    with pytest.raises(ValueError, match="unknown or ambiguous"):
        tracker.inspect("CHIP_A", "V1.2", "missing")
    payload["scopes"][0]["edges"][0]["target"] = "missing-node"
    with pytest.raises(ValueError, match="unknown relation node"):
        SpecRelationGraph.from_mapping(payload)


def test_graph_requires_source_quotes_and_review_metadata():
    payload, _ = graph_data()
    payload["scopes"][0]["edges"][0]["evidence"]["quotes"] = []
    with pytest.raises(ValueError, match="source quotes"):
        SpecRelationGraph.from_mapping(payload)
    payload, _ = graph_data()
    del payload["review"]
    with pytest.raises(ValueError, match="structure"):
        SpecRelationGraph.from_mapping(payload)


async def test_existing_loop_closes_evidence_gaps_without_tool_schema_changes():
    payload, hits = graph_data()
    requests = []

    async def transport(url, params, headers):
        requests.append(url)
        return (200, hits[:1]) if url.endswith("/search") else (200, hits[1])

    registry = registry_for(payload, transport)

    sequence = iter(range(10))

    def call(name, arguments):
        return AssistantTurn(
            Message(
                "assistant",
                tool_calls=[
                    ToolCall(f"call-{next(sequence)}", name, arguments),
                ],
            ),
            stop_reason="tool_calls",
        )

    provider = MockProvider(
        [
            call("inspect_spec_evidence", SCOPE),
            call("search_spec", {"question": "irq"}),
            call("inspect_spec_evidence", SCOPE),
            call(
                "read_context",
                {"document_id": hits[1]["document_id"], "chunk_id": hits[1]["chunk_id"]},
            ),
            call("inspect_spec_evidence", SCOPE),
            AssistantTurn(Message("assistant", "Both sources read."), stop_reason="stop"),
        ]
    )
    result = await run_agent(
        RunOptions(
            provider=provider,
            tools=registry,
            messages=[user_message("Check both interrupt gates.")],
        )
    )
    reports = [
        json.loads(message.content)
        for message in result.messages
        if message.role == "tool" and message.name == "inspect_spec_evidence"
    ]
    assert [report["missing"] for report in reports] == [["need_0", "need_1"], ["need_1"], []]
    assert result.status == "completed" and len(requests) == 2
    assert len(provider.requests) == 6
    assert all(request.tools == provider.requests[0].tools for request in provider.requests)
