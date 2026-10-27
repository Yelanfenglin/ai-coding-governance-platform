"""Optional source-linked relation graph and per-run evidence checklist.

The graph suggests what to read. Only content actually returned by RAG tools can
satisfy its source quotes. This is not semantic entailment or a hardware checker.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

from .contracts import ToolResult
from .tools import FunctionTool

SPEC_EVIDENCE_GUIDANCE = (
    "When inspect_spec_evidence is available, use it before broad searches for scoped "
    "specification configuration/dependency questions, with read_missing=true. Use a feature/pin "
    "topic label/alias; omit entity to list topics. Pass only mode/configuration "
    "facts explicitly supplied by the user or observed in source. For a supplied C function, "
    "use source={path,function} instead of hand-transcribing register values. Report readiness and "
    "configuration_status separately. Use ordinary retrieval for unsupported topics or remaining "
    "requirements; do not repeat retrieval for already covered relations."
)


class UnknownTopic(ValueError):
    """The scope exists, so the tool can return its actual topic choices."""


def _text(value: dict, key: str) -> str:
    result = value[key]
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"{key} must be nonempty text")
    return result.strip()


def _scope(product: str, version: str) -> tuple[str, str]:
    match = re.fullmatch(r"[Vv]?(\d+)\.(\d+)", version.strip())
    if not product.strip() or match is None:
        raise ValueError("product and major.minor version are required")
    return product.strip().upper(), f"V{int(match[1])}.{int(match[2])}"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _settings(value: object) -> dict[str, str | int | bool]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not key.strip() or type(item) not in (str, int, bool)
        for key, item in value.items()
    ):
        raise ValueError("facts and conditions must map names to text, integers or booleans")
    return dict(value)


def _compare(expected: dict, facts: dict) -> tuple[dict, dict]:
    missing = {key: value for key, value in expected.items() if key not in facts}
    different = {
        key: {"expected": value, "provided": facts[key]}
        for key, value in expected.items()
        if key in facts and (type(value) is not type(facts[key]) or value != facts[key])
    }
    return missing, different


@dataclass(frozen=True)
class Relation:
    identifier: str
    source: str
    target: str
    predicate: str
    description: str
    document_id: str
    chunk_id: str
    quotes: tuple[str, ...]
    when: dict[str, str | int | bool] = field(default_factory=dict)
    requires: dict[str, str | int | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class ScopeGraph:
    nodes: dict[str, str]
    relations: tuple[Relation, ...]
    topics: tuple[str, ...]
    aliases: dict[str, tuple[str, ...]]
    fields: dict[str, tuple[str, int, int]]

    def expand_facts(self, facts: dict) -> dict:
        expanded = dict(facts)
        for name, (register, offset, width) in self.fields.items():
            if register not in facts:
                continue
            raw = facts[register]
            if type(raw) is not int or raw < 0:
                raise ValueError(f"raw register {register} must be a nonnegative integer")
            value = (raw >> offset) & ((1 << width) - 1)
            if name in facts and (type(facts[name]) is not int or facts[name] != value):
                raise ValueError(f"conflicting raw register and field facts: {name}")
            expanded[name] = value
        return expanded


@dataclass(frozen=True)
class SpecRelationGraph:
    scopes: dict[tuple[str, str], ScopeGraph]
    review: dict[str, str]

    @classmethod
    def load(cls, path: Path) -> SpecRelationGraph:
        return cls.from_mapping(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_mapping(cls, payload: dict) -> SpecRelationGraph:
        try:
            if payload["schema_version"] != 1:
                raise ValueError("unsupported relation graph schema_version")
            review = {
                key: _text(payload["review"], key)
                for key in ("source_review", "independent_review")
            }
            scopes = {}
            for entry in payload["scopes"]:
                scope = _scope(_text(entry, "product"), _text(entry, "version"))
                nodes = {_text(node, "id"): _text(node, "label") for node in entry["nodes"]}
                if not nodes or len(nodes) != len(entry["nodes"]) or scope in scopes:
                    raise ValueError("empty or duplicate scope/nodes")
                aliases = {}
                for node in entry["nodes"]:
                    names = node.get("aliases", [])
                    if not isinstance(names, list) or any(
                        not isinstance(name, str) or not name.strip() for name in names
                    ):
                        raise ValueError("node aliases must be nonempty text")
                    aliases[node["id"]] = tuple(names)
                relations = []
                identifiers = set()
                for edge in entry["edges"]:
                    source = edge["evidence"]
                    quotes = source["quotes"]
                    if (
                        not isinstance(quotes, list)
                        or not quotes
                        or any(not isinstance(q, str) or not q.strip() for q in quotes)
                    ):
                        raise ValueError("each relation needs nonempty source quotes")
                    relation = Relation(
                        _text(edge, "id"),
                        _text(edge, "source"),
                        _text(edge, "target"),
                        _text(edge, "relation"),
                        _text(edge, "description"),
                        str(UUID(_text(source, "document_id"))),
                        str(UUID(_text(source, "chunk_id"))),
                        tuple(quotes),
                        _settings(edge.get("when", {})),
                        _settings(edge.get("requires", {})),
                    )
                    if (
                        relation.source not in nodes
                        or relation.target not in nodes
                        or relation.identifier in identifiers
                    ):
                        raise ValueError("unknown relation node or duplicate relation id")
                    identifiers.add(relation.identifier)
                    relations.append(relation)
                if not relations:
                    raise ValueError("a scope must contain source-linked relations")
                fields = {}
                for name, definition in entry.get("fields", {}).items():
                    register = _text(definition, "register")
                    offset, width = definition["lsb"], definition["width"]
                    if (
                        not isinstance(name, str)
                        or not name.strip()
                        or type(offset) is not int
                        or offset < 0
                        or type(width) is not int
                        or width < 1
                        or offset + width > 64
                    ):
                        raise ValueError("invalid register field name or bit range")
                    fields[name] = (register, offset, width)
                sources = {edge.source for edge in relations}
                topics = entry.get("topics", sorted(sources))
                if (
                    not isinstance(topics, list)
                    or not topics
                    or any(not isinstance(node, str) or node not in sources for node in topics)
                    or len(set(topics)) != len(topics)
                ):
                    raise ValueError("topics must name distinct nodes with outgoing relations")
                scopes[scope] = ScopeGraph(nodes, tuple(relations), tuple(topics), aliases, fields)
            if not scopes:
                raise ValueError("relation graph has no scopes")
            return cls(scopes, review)
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("invalid relation graph structure") from error


class SpecEvidence:
    def __init__(self, graph: SpecRelationGraph):
        self.graph = graph
        self.observed: dict[tuple[str, str], dict[str, Any]] = {}

    def record(self, fragment: dict[str, Any], visible_text: str) -> None:
        """Record visible source content, never hidden text behind an excerpt."""
        try:
            key = (str(UUID(str(fragment["document_id"]))), str(UUID(str(fragment["chunk_id"]))))
            product, version = _scope(fragment["product"], fragment["version"])
        except (KeyError, TypeError, ValueError, AttributeError):
            return
        previous = self.observed.get(key)
        texts = []
        if previous and previous["product"] == product and previous["version"] == version:
            texts = previous["texts"]
            if any(visible_text in text for text in texts):
                return
            texts = [text for text in texts if text not in visible_text]
        self.observed[key] = {
            "document_id": key[0],
            "chunk_id": key[1],
            "product": product,
            "version": version,
            "page_start": fragment.get("page_start"),
            "page_end": fragment.get("page_end"),
            "section_path": fragment.get("section_path"),
            "texts": [*texts, visible_text],
        }

    def inspect(
        self,
        product: str,
        version: str,
        entity: str = "",
        *,
        facts: dict | None = None,
        max_depth: int = 3,
    ) -> dict:
        supplied = _settings(facts) if facts is not None else {}
        if type(max_depth) is not int or not 1 <= max_depth <= 6:
            raise ValueError("max_depth must be between 1 and 6")
        scope = _scope(product, version)
        graph = self.graph.scopes.get(scope)
        if graph is None:
            raise ValueError("no relation graph for this product/version; use ordinary retrieval")
        supplied = graph.expand_facts(supplied)
        topics = [{"id": node, "label": graph.nodes[node]} for node in graph.topics]
        matches = [
            node
            for node in graph.topics
            if entity.casefold()
            in (name.casefold() for name in (node, graph.nodes[node], *graph.aliases[node]))
        ]
        if not entity:
            context_values = {}
            register_keys = set(graph.fields) | {r for r, _, _ in graph.fields.values()}
            for edge in graph.relations:
                for key, value in {**edge.when, **edge.requires}.items():
                    if key not in register_keys:
                        options = context_values.setdefault(key, [])
                        if not any(type(item) is type(value) and item == value for item in options):
                            options.append(value)
            return {
                "scope": scope,
                "ready": False,
                "next_action": "select_entity",
                "available_entities": topics,
                "context_values": context_values,
                "graph_review": self.graph.review,
            }
        if len(matches) != 1:
            raise UnknownTopic("unknown or ambiguous graph entity; select an available topic")
        selected = []
        excluded = []
        unresolved = []
        boundaries = []
        outgoing = {}
        for edge in graph.relations:
            outgoing.setdefault(edge.source, []).append(edge)
        queue = deque([(matches[0], (matches[0],))])
        visited = set()
        while queue:
            node, path = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            for edge in outgoing.get(node, []):
                unknown, different = _compare(edge.when, supplied)
                if different:
                    excluded.append(
                        {"id": edge.identifier, "when": edge.when, "different": different}
                    )
                    continue
                if len(path) > max_depth:
                    boundaries.append(
                        {"kind": "depth_limit", "node": node, "relation": edge.identifier}
                    )
                    continue
                selected.append((edge, path, unknown))
                if unknown:
                    unresolved.append({"id": edge.identifier, "missing_facts": unknown})
                else:
                    queue.append((edge.target, (*path, edge.target)))

        # Detect cycles across converging paths as well as parent/child back edges.
        degree = {}
        active = {}
        for edge, _, unknown in selected:
            if not unknown:
                degree.setdefault(edge.source, 0)
                degree[edge.target] = degree.get(edge.target, 0) + 1
                active.setdefault(edge.source, []).append(edge.target)
        pending = deque(node for node, count in degree.items() if count == 0)
        while pending:
            for target in active.get(pending.popleft(), []):
                degree[target] -= 1
                if degree[target] == 0:
                    pending.append(target)
        if any(degree.values()):
            boundaries.append({"kind": "cycle", "nodes": sorted(n for n in degree if degree[n])})

        relations = []
        for edge, path, unknown in selected:
            observed = self.observed.get((edge.document_id, edge.chunk_id))
            state = "source_not_read"
            matched_quotes = []
            if observed:
                if (observed["product"], observed["version"]) != scope:
                    state = "wrong_scope"
                else:
                    matched_quotes = [
                        q
                        for q in edge.quotes
                        if any(_normalize(q) in _normalize(text) for text in observed["texts"])
                    ]
                    state = (
                        "source_quotes_present"
                        if len(matched_quotes) == len(edge.quotes)
                        else "quotes_not_in_returned_text"
                    )
            relations.append(
                {
                    "id": edge.identifier,
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.predicate,
                    "requirement": edge.description,
                    "path": [*path, edge.target],
                    "depth": len(path),
                    "applicability": "unknown" if unknown else "applies",
                    "when": edge.when,
                    "requires": edge.requires,
                    "field_value": supplied.get(edge.target),
                    "field_layout": {
                        name: {"register": register, "lsb": offset, "width": width}
                        for name, (register, offset, width) in graph.fields.items()
                        if name in edge.requires or name == edge.target
                    },
                    "configuration": self._configuration(
                        edge, supplied, facts is not None, unknown
                    ),
                    "status": state,
                    "source_reference": {
                        "document_id": edge.document_id,
                        "chunk_id": edge.chunk_id,
                    },
                    "expected_source_quotes": list(edge.quotes),
                    "observed_quotes": matched_quotes,
                    "observed_source": {
                        key: value for key, value in observed.items() if key != "texts"
                    }
                    if observed
                    else None,
                }
            )
        missing = [edge["id"] for edge in relations if edge["status"] != "source_quotes_present"]
        states = {row["configuration"]["status"] for row in relations}
        configuration = (
            "unsatisfied"
            if "unsatisfied" in states
            else "unknown"
            if "unknown" in states
            else "satisfied"
            if "satisfied" in states
            else "not_checked"
        )
        ready = bool(relations) and not (missing or unresolved or boundaries)
        action = (
            "resolve_graph_boundary"
            if boundaries
            else "provide_condition_facts"
            if unresolved
            else "use_ordinary_retrieval"
            if not relations
            else "retrieve_missing_only"
            if missing
            else "report_configuration_gaps"
            if configuration == "unsatisfied"
            else "request_missing_configuration"
            if configuration == "unknown"
            else "continue_task_using_evidence"
        )
        return {
            "scope": {"product": scope[0], "version": scope[1]},
            "entity": matches[0],
            "graph_review": self.graph.review,
            "relations": relations,
            "missing": missing,
            "excluded_relations": excluded,
            "unresolved_conditions": unresolved,
            "traversal_boundaries": boundaries,
            "ready": ready,
            "configuration_status": configuration,
            "configuration_basis": "Caller facts only; code paths and hardware unverified.",
            "next_action": action,
            "check_scope": "Source quote presence and scope only. Not answer correctness, hardware "
            "verification, or proof that the graph covers every task requirement. "
            "A fact missing from returned evidence does not establish its absence from the source.",
        }

    @staticmethod
    def _configuration(edge: Relation, facts: dict, requested: bool, unknown: dict) -> dict:
        if not edge.requires or not requested or unknown:
            return {"status": "unknown" if unknown and edge.requires else "not_checked"}
        missing, different = _compare(edge.requires, facts)
        return {
            "status": "unsatisfied" if different else "unknown" if missing else "satisfied",
            "missing_facts": missing,
            "different": different,
        }

    def tool(
        self,
        reader: Callable[[str, str], Awaitable[tuple[dict[str, Any], str]]] | None = None,
        source_reader: Callable[[str, str, ScopeGraph], Awaitable[dict]] | None = None,
    ) -> FunctionTool:
        async def execute(args, context):
            facts = args.get("facts")
            source = None

            def discovery(report):
                report["fetched_sources"] = []
                if args.get("source"):
                    report["next_arguments"] = {
                        key: args[key]
                        for key in (
                            "product",
                            "version",
                            "source",
                            "facts",
                            "read_missing",
                            "source_limit",
                            "max_depth",
                        )
                        if key in args
                    }
                    report["source_analysis_pending"] = True
                    report["next_instruction"] = (
                        "Select an available entity, reuse next_arguments including source, and "
                        "supply only context facts known from the task (see context_values). "
                        "Do not hand-transcribe register values. Source has not been read yet."
                    )
                return report

            def inspect():
                return self.inspect(
                    args["product"],
                    args["version"],
                    args.get("entity", ""),
                    facts=facts,
                    max_depth=args.get("max_depth", 3),
                )

            try:
                report = inspect()
                if args.get("source"):
                    if source_reader is None:
                        raise ValueError("source inspection requires a selected workspace")
                    if not report.get("relations"):
                        return ToolResult(content=json.dumps(discovery(report), ensure_ascii=False))
                    graph = self.graph.scopes.get(_scope(args["product"], args["version"]))
                    if graph is None:
                        raise ValueError("no relation graph for this product/version")
                    facts = _settings(facts) if facts is not None else {}
                    register_keys = set(graph.fields) | {r for r, _, _ in graph.fields.values()}
                    if register_keys.intersection(facts):
                        raise ValueError("with source, facts may contain only mode/context values")
                    source = await source_reader(
                        args["source"]["path"],
                        args["source"]["function"],
                        graph,
                    )
                    facts = {**facts, **source["facts"]}
                    report = inspect()
            except UnknownTopic as error:
                report = discovery(self.inspect(args["product"], args["version"]))
                report["error"] = str(error)
                return ToolResult(content=json.dumps(report, ensure_ascii=False), is_error=True)
            except (OSError, UnicodeError, ValueError) as error:
                return ToolResult(content=str(error), is_error=True)
            fetched = []
            if args.get("read_missing") and report.get("missing"):
                if reader is None:
                    return ToolResult(
                        content="canonical source reader is not configured", is_error=True
                    )
                references = {}
                for relation in report["relations"]:
                    if (
                        relation["id"] in report["missing"]
                        and relation["applicability"] == "applies"
                    ):
                        ref = relation["source_reference"]
                        references[(ref["document_id"], ref["chunk_id"])] = ref

                async def fetch(ref):
                    try:
                        return ref, await reader(ref["document_id"], ref["chunk_id"])
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        return ref, error

                limit = int(args.get("source_limit", 4))
                results = await asyncio.gather(
                    *(fetch(ref) for ref in list(references.values())[:limit])
                )
                for ref, result in results:
                    if isinstance(result, Exception):
                        fetched.append({**ref, "error": type(result).__name__})
                        continue
                    fragment, body = result
                    if (
                        str(fragment.get("document_id")) != ref["document_id"]
                        or str(fragment.get("chunk_id")) != ref["chunk_id"]
                    ):
                        fetched.append({**ref, "error": "source_identity_mismatch"})
                        continue
                    quotes = list(
                        dict.fromkeys(
                            quote
                            for relation in report["relations"]
                            if relation["source_reference"] == ref
                            for quote in relation["expected_source_quotes"]
                            if _normalize(quote) in _normalize(body)
                        )
                    )
                    # Return every quote that is recorded; unseen full bodies never satisfy needs.
                    self.record(fragment, "\n".join(quotes))
                    fetched.append({**ref, "returned_quotes": quotes})
                report = inspect()
            report["fetched_sources"] = fetched
            if source is not None:
                for relation in report.get("relations", []):
                    relation["configuration"]["source_locations"] = {
                        key: source["origins"][key]
                        for key in relation["requires"]
                        if key in source["origins"]
                    }
                    relation["field_source_locations"] = source["origins"].get(
                        relation["target"],
                        [],
                    )
                report["source_analysis"] = {
                    key: value for key, value in source.items() if key != "origins"
                }
                report["configuration_basis"] = source["basis"]
            return ToolResult(content=json.dumps(report, ensure_ascii=False))

        return FunctionTool(
            name="inspect_spec_evidence",
            description=(
                "Traverse a source-linked specification relation graph (default depth 3) against "
                "RAG evidence returned in this run. Omit entity to discover scoped topics; use a "
                "topic ID, label or alias. Optional facts are explicit known configuration/mode "
                "values, never guessed defaults; raw registers must be nonnegative integers and "
                "are decoded using scoped field layouts. Report branches, source gaps and "
                "required settings separately. For a C function, source={path,function} reads "
                "named straight-line register writes from the selected workspace and returns "
                "source locations; then facts may contain only mode/context values. Unsupported "
                "source analysis stays unknown, without running code. "
                "Unknown conditions and depth/cycle limits stay "
                "unresolved. Use read_missing=true to "
                "fetch only missing canonical sources and return supported quotes in this call "
                "(at most source_limit, default 4). Otherwise use search_spec/read_context. "
                "When ready, continue the task without "
                "repeating retrieval for these facts. ready means source coverage within this "
                "graph, not configuration success or overall answer correctness."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "product": {"type": "string", "minLength": 1},
                    "version": {"type": "string", "minLength": 1},
                    "entity": {"type": "string", "description": "Topic ID/label; omit to list"},
                    "facts": {
                        "type": "object",
                        "description": "Known field/mode values; omitted keys remain unknown",
                        "additionalProperties": {"type": ["string", "integer", "boolean"]},
                    },
                    "source": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "minLength": 1},
                            "function": {"type": "string", "minLength": 1},
                        },
                        "required": ["path", "function"],
                        "additionalProperties": False,
                    },
                    "max_depth": {"type": "integer", "minimum": 1, "maximum": 6},
                    "read_missing": {"type": "boolean"},
                    "source_limit": {"type": "integer", "minimum": 1, "maximum": 8},
                },
                "required": ["product", "version"],
                "additionalProperties": False,
            },
            handler=execute,
            execution_mode="exclusive",  # Observe completed parallel reads at a tool barrier.
        )
