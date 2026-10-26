"""HTTP adapter acceptance against the actual runtime, tools, log and delivery path."""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import httpx
import pytest

pytest.importorskip("fastapi")

from app import cli  # noqa: E402
from app.contracts import AssistantTurn, Message, ToolCall  # noqa: E402
from app.providers.mock import stream_from_coroutine  # noqa: E402
from app.session import JsonlSessionStore  # noqa: E402
from app.web import API_PREFIX, create_app  # noqa: E402


class Provider:
    def __init__(self, turns):
        self.turns = iter(turns)
        self.requests = []

    def generate_stream(self, request):
        self.requests.append(request)
        return stream_from_coroutine(lambda: self.generate(request))

    async def generate(self, request):
        return next(self.turns)


def final(text="done"):
    return AssistantTurn(Message("assistant", text), "stop")


def tools(*calls):
    return AssistantTurn(Message("assistant", tool_calls=list(calls)), "tool_calls")


def options(workspace, *extra):
    return cli.build_parser().parse_args([
        "--workspace", str(workspace), "--state-dir", str(workspace / ".agent-runtime"),
        "--base-url", "http://unused.invalid/v1", *extra,
    ])


async def conversation(client):
    response = await client.post(f"{API_PREFIX}/conversations", json={
        "title": "firmware", "project_id": "local",
    })
    assert response.status_code == 201
    return response.json()["id"]


async def start(client, identifier, prompt="implement", key=None):
    return await client.post(f"{API_PREFIX}/tasks", json={
        "conversation_id": identifier, "prompt": prompt, "idempotency_key": key or str(uuid4()),
    })


async def complete(app, identifier):
    await asyncio.wait_for(app.state.agent.jobs[identifier].runner, 5)
    return app.state.agent.jobs[identifier].record


@pytest.fixture
async def connected(tmp_path):
    app = create_app(options(tmp_path))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                         base_url="http://127.0.0.1") as client,
    ):
        yield app, client, tmp_path


async def test_health_exposes_activity_without_invoking_the_model(connected, monkeypatch):
    app, client, _ = connected

    def unexpected(**_):
        raise AssertionError("health must not call a model")

    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", unexpected)
    response = await client.get(f"{API_PREFIX}/health")
    assert response.status_code == 200
    assert response.json() == {"ready": True, "service": "embedded-agent", "active_task": None}


async def test_http_task_edits_real_files_resumes_memory_and_exposes_delivery(
    connected, monkeypatch, capsys,
):
    app, client, workspace = connected
    first = Provider([
        tools(ToolCall("write", "write_file", {"path": "demo.c", "content": "int x = 1;\n"})),
        tools(ToolCall("remember", "remember_fact", {
            "key": "confirmed.clock", "value": "CLOCK_CONFIRMED_BY_USER", "source": "user",
        })),
        tools(ToolCall("check", "run_command", {"command": [
            "python", "-c",
            "from pathlib import Path; assert 'x = 1' in Path('demo.c').read_text()",
        ]})),
        tools(ToolCall("explain", "record_implementation", {"decisions": [{
            "id": "module", "title": "Module configuration", "requirement": "Create demo.c",
            "rationale": "Create the requested source and run the actual assertion.",
            "code_refs": [{"path": "demo.c", "start_line": 1, "end_line": 1}],
            "check_call_ids": ["check"],
        }]})),
        final("implemented"),
    ])
    second = Provider([final("continued")])
    providers = iter([first, second])
    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **_: next(providers))
    identifier = await conversation(client)
    created = await start(client, identifier)
    assert created.status_code == 201
    job_id = created.json()["id"]
    record = await complete(app, job_id)
    assert record["status"] == "completed"
    assert record["response"]["answer"] == "implemented"
    assert record["response"]["verification"]["status"] == "unverified"
    assert (workspace / "demo.c").read_text() == "int x = 1;\n"
    assert capsys.readouterr().out == "", "web must not print CLI JSON or model deltas"
    delivery = (await client.get(f"{API_PREFIX}/tasks/{job_id}/delivery")).json()
    assert delivery["changed_files"] == ["demo.c"]
    assert "+int x = 1;" in delivery["diff"]
    assert delivery["commands"][0]["is_error"] is False
    assert delivery["decisions"][0]["code_refs"][0]["status"] == "located"
    assert delivery["decisions"][0]["checks"][0]["status"] == "passed"
    assert delivery["decisions"][0]["evidence_refs"] == []
    stream = await client.get(f"{API_PREFIX}/tasks/{job_id}/events")
    events = [json.loads(line[6:]) for line in stream.text.splitlines()
              if line.startswith("data: ")]
    assert events[-1]["type"] == "TaskFinished"
    assert [e["type"] for e in events].index("TurnFinished") < len(events) - 1
    assert any(e["type"] == "ToolFinished" for e in events)
    replay = await client.get(f"{API_PREFIX}/tasks/{job_id}/events", headers={
        "Last-Event-ID": str(events[-2]["sequence"]),
    })
    assert replay.text.count("data: ") == 1

    created = await start(client, identifier, "continue")
    continued = await complete(app, created.json()["id"])
    assert continued["response"]["resumed_messages"] > 2
    assert "[实现依据]" in next(
        m.content for m in reversed(second.requests[0].messages) if m.role == "user"
    )
    assert "CLOCK_CONFIRMED_BY_USER" in str(second.requests[0].messages)
    assert any(m.role == "assistant" and m.content == "implemented"
               for m in second.requests[0].messages)
    assert len((await client.get(f"{API_PREFIX}/conversations/{identifier}")).json()["tasks"]) == 2


@pytest.mark.parametrize("check_name", ["run_command", "verify_project"])
async def test_check_reference_is_available_in_observation_without_reading_logs(
    connected, monkeypatch, check_name,
):
    app, client, workspace = connected
    (workspace / "module.c").write_text("int value = 2;\n", encoding="utf-8")
    command = ["python", "-c", "from pathlib import Path; assert Path('module.c').exists()"]
    if check_name == "verify_project":
        config = workspace.parent / f"{workspace.name}-verification.json"
        config.write_text(json.dumps({"command": command}), encoding="utf-8")
        app.state.agent.options.verification = config

    class LinkingProvider(Provider):
        def __init__(self):
            super().__init__([])
            self.step = 0

        async def generate(self, request):
            self.step += 1
            if self.step == 1:
                return tools(ToolCall(str(uuid4()), check_name,
                                      {"command": command} if check_name == "run_command" else {}))
            if self.step == 2:
                observed = json.loads(request.messages[-1].content)
                return tools(ToolCall(str(uuid4()), "record_implementation", {"decisions": [{
                    "id": "module", "title": "Module", "requirement": "Keep module.c",
                    "rationale": "The configured check confirms that the source exists.",
                    "code_refs": [{"path": "module.c", "start_line": 1, "end_line": 1}],
                    "check_call_ids": [observed["check_call_id"]],
                }]}))
            return final("linked")

    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **_: LinkingProvider())
    identifier = await conversation(client)
    created = await start(client, identifier, "Review the existing source")
    record = await complete(app, created.json()["id"])
    assert record["status"] == "completed"
    delivery = (await client.get(f"{API_PREFIX}/tasks/{record['id']}/delivery")).json()
    linked = delivery["decisions"][0]["checks"]
    assert len(linked) == 1 and linked[0]["status"] == "passed"
    assert linked[0]["name"] == check_name


async def test_cancellation_waits_for_runtime_cleanup_and_can_continue(connected, monkeypatch):
    app, client, workspace = connected
    entered, closed = asyncio.Event(), asyncio.Event()

    class Hanging:
        async def generate_stream(self, request):
            entered.set()
            try:
                await asyncio.Event().wait()
                yield  # pragma: no cover
            finally:
                closed.set()

    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **_: Hanging())
    identifier = await conversation(client)
    key = str(uuid4())
    created = await start(client, identifier, key=key)
    job_id = created.json()["id"]
    await asyncio.wait_for(entered.wait(), 3)
    assert (await start(client, identifier, key=key)).json()["id"] == job_id
    assert (await start(client, identifier, prompt="another")).status_code == 409
    assert (await start(client, identifier, prompt="different", key=key)).status_code == 409
    cancelled = await client.post(f"{API_PREFIX}/tasks/{job_id}/cancel")
    assert cancelled.status_code == 202
    record = await complete(app, job_id)
    assert record["status"] == "cancelled"
    assert closed.is_set()
    session = JsonlSessionStore(workspace / ".agent-runtime" / "web" / "conversations"
                               / identifier / "session.jsonl")
    assert (await session.load_timeline()).status == "aborted"
    resumed = Provider([final("resumed")])
    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **_: resumed)
    created = await start(client, identifier, "resume")
    assert (await complete(app, created.json()["id"]))["status"] == "completed"
    inputs = [m.content for m in resumed.requests[0].messages if m.role == "user"]
    assert [text.split("\n\n[实现依据]", 1)[0] for text in inputs] == [
        "implement", "resume",
    ]
    assert all("record_implementation" in text for text in inputs)


async def test_model_completed_does_not_override_failed_host_verification(tmp_path, monkeypatch):
    config = tmp_path / "verification.json"
    config.write_text(json.dumps({"command": ["{python}", "-c", "raise SystemExit(7)"]}))
    workspace = tmp_path / "project"
    workspace.mkdir()
    app = create_app(options(workspace, "--verification", str(config)))
    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **_: Provider([final("done")]))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://127.0.0.1") as client:
        identifier = await conversation(client)
        created = await start(client, identifier)
        record = await complete(app, created.json()["id"])
    assert record["response"]["status"] == "completed"
    assert record["status"] == "failed"
    assert record["exit_code"] == cli.EXIT_VERIFICATION
    assert app.state.agent.jobs[record["id"]].events[-1]["payload"]["status"] == "failed"


async def test_history_and_events_survive_service_restart(connected, monkeypatch):
    app, client, workspace = connected
    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **_: Provider([final()]))
    identifier = await conversation(client)
    created = await start(client, identifier)
    job_id = created.json()["id"]
    await complete(app, job_id)
    restarted = create_app(options(workspace))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=restarted),
                                 base_url="http://127.0.0.1") as after:
        assert (await after.get(f"{API_PREFIX}/conversations")).json()[0]["id"] == identifier
        record = (await after.get(f"{API_PREFIX}/tasks/{job_id}")).json()
        assert record["status"] == "completed"
        assert "TaskFinished" in (await after.get(f"{API_PREFIX}/tasks/{job_id}/events")).text
    # Metadata committed immediately before the final SSE append must also replay a terminal.
    event_path = app.state.agent.jobs[job_id].directory / "events.jsonl"
    lines = event_path.read_text(encoding="utf-8").splitlines()
    event_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    repaired = create_app(options(workspace))
    job = repaired.state.agent.jobs[job_id]
    assert job.events[-1]["type"] == "TaskFinished"
    assert job.events[-1]["payload"]["status"] == "completed"
    # Interrupted process state is not mislabeled as running or automatically rerun.
    path = app.state.agent.jobs[job_id].directory / "task.json"
    record["status"] = "running"
    path.write_text(json.dumps(record))
    recovered = create_app(options(workspace))
    assert recovered.state.agent.jobs[job_id].record["status"] == "failed"
    assert "重启" in recovered.state.agent.jobs[job_id].record["response"]["error"]


async def test_local_service_rejects_foreign_browser_origins_and_arbitrary_paths(connected):
    _app, client, _workspace = connected
    assert (await client.get(f"{API_PREFIX}/projects")).status_code == 200
    assert (await client.get(f"{API_PREFIX}/projects", headers={"Host": "attacker.example"}
                             )).status_code == 403
    assert (await client.post(f"{API_PREFIX}/conversations", headers={
        "Origin": "https://attacker.example",
    }, json={"title": "bad", "project_id": "local"})).status_code == 403
    assert (await client.post(f"{API_PREFIX}/conversations", json={
        "title": "bad", "project_id": "local", "workspace": "C:/",
    })).status_code == 422
    assert (await client.post(f"{API_PREFIX}/tasks", json={
        "conversation_id": str(uuid4()), "prompt": "bad", "idempotency_key": str(uuid4()),
    })).status_code == 404
    assert (await client.get(f"{API_PREFIX}/tasks/{uuid4()}/delivery")).status_code == 404


async def test_skill_catalog_activation_history_and_per_task_selection(connected, monkeypatch):
    app, client, workspace = connected
    catalog = (await client.get(f"{API_PREFIX}/skills")).json()
    assert {item["name"] for item in catalog} >= {"chip-migration", "module-development"}
    assert all(set(item) == {"name", "title", "description"} for item in catalog)
    provider = Provider([final("selected"), final("ordinary")])
    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **_: provider)
    identifier = await conversation(client)
    payload = {"conversation_id": identifier, "prompt": "本轮需求",
               "idempotency_key": str(uuid4()), "skill": "module-development"}
    created = await client.post(f"{API_PREFIX}/tasks", json=payload)
    assert created.status_code == 201
    record = await complete(app, created.json()["id"])
    assert record["skill"] == "module-development" and record["skill_title"] == "模块开发"
    first = provider.requests[0].messages[-1]
    assert first.role == "user" and '<skill name="module-development"' in first.content
    assert "# 工程内的可复用模块开发" in first.content
    assert not any(m.tool_calls for m in provider.requests[0].messages)
    assert (await client.post(f"{API_PREFIX}/tasks", json=payload)).json()["id"] == record["id"]
    assert (await client.post(f"{API_PREFIX}/tasks", json={
        **payload, "skill": "chip-migration",
    })).status_code == 409

    invalid = await client.post(f"{API_PREFIX}/tasks", json={
        **payload, "idempotency_key": str(uuid4()), "skill": "../unknown",
    })
    assert invalid.status_code == 400 and len(app.state.agent.jobs) == 1
    continued = await start(client, identifier, "普通追问")
    after = await complete(app, continued.json()["id"])
    assert after["skill"] is None
    assert provider.requests[-1].messages[-1].content.startswith("普通追问\n")
    assert '<skill name=' not in provider.requests[-1].messages[-1].content
    assert sum('<skill name="module-development" location=' in m.content
               for m in provider.requests[-1].messages) == 1
    restarted = create_app(options(workspace))
    assert restarted.state.agent.jobs[record["id"]].record["skill_title"] == "模块开发"


async def test_http_recheck_is_exclusive_persistent_and_does_not_rerun_model(tmp_path, monkeypatch):
    from app.project_verification import ProjectVerification

    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "module.c").write_text("int value = 1;\n")
    health = tmp_path / "health.txt"
    health.write_text("down")
    config = tmp_path / "verification.json"
    config.write_text(json.dumps({"command": ["{python}", "-c",
        f"from pathlib import Path; raise SystemExit(0 if Path({str(health)!r}).read_text()"
        " == 'ready' else 7)"]}))
    opts = options(workspace, "--verification", str(config))
    app = create_app(opts)
    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **_: Provider([final()]))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://127.0.0.1") as client:
        identifier = await conversation(client)
        created = await start(client, identifier)
        record = await complete(app, created.json()["id"])
        job = app.state.agent.jobs[record["id"]]
        originals = {p: p.read_bytes() for p in (
            job.directory / "task.json", job.directory / "events.jsonl",
            job.directory / "delivery/delivery.json",
        )}
        assert record["status"] == "failed"

        def no_model(**_):
            raise AssertionError("recheck must not call a model")

        monkeypatch.setattr(cli, "OpenAiCompatibleProvider", no_model)
        health.write_text("ready")
        entered, release = asyncio.Event(), asyncio.Event()
        original_run = ProjectVerification.run

        async def paused_run(self, *, phase):
            entered.set()
            await release.wait()
            return await original_run(self, phase=phase)

        monkeypatch.setattr(ProjectVerification, "run", paused_run)
        endpoint = f"{API_PREFIX}/tasks/{record['id']}/recheck"
        recheck = asyncio.create_task(client.post(endpoint))
        await asyncio.wait_for(entered.wait(), 3)
        assert (await client.post(endpoint)).status_code == 409
        assert (await start(client, identifier, "new task")).status_code == 409
        release.set()
        response = await recheck
        assert response.status_code == 200 and response.json()["passed"]
        assert all(p.read_bytes() == value for p, value in originals.items())

    restarted = create_app(opts)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=restarted),
                                 base_url="http://127.0.0.1") as client:
        delivery = (await client.get(f"{API_PREFIX}/tasks/{record['id']}/delivery")).json()
        assert delivery["task_acceptance"] == "failed"
        assert delivery["recheck"]["runs"][0]["passed"]
        assert delivery["recheck"]["runs"][0]["matches_current"]


async def test_http_undo_keeps_original_records_survives_restart_and_checks_conflicts(
    connected, monkeypatch,
):
    app, client, root = connected
    (root / "module.c").write_bytes(b"int old = 1;\r\n")
    provider = Provider([
        tools(ToolCall("change", "write_file", {"path": "module.c", "content": "int new = 2;\n"}),
              ToolCall("add", "write_file", {"path": "new.c", "content": "int added = 3;\n"})),
        final(),
    ])
    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", lambda **_: provider)
    identifier = await conversation(client)
    created = await start(client, identifier)
    record = await complete(app, created.json()["id"])
    job = app.state.agent.jobs[record["id"]]
    artifacts = {p: p.read_bytes() for p in (
        job.directory / "task.json", job.directory / "events.jsonl",
        job.directory / "delivery/delivery.json", job.directory / "delivery/changes.diff",
    )}

    def no_model(**_):
        raise AssertionError("undo must not call a model")

    monkeypatch.setattr(cli, "OpenAiCompatibleProvider", no_model)
    endpoint = f"{API_PREFIX}/tasks/{record['id']}/undo"
    # A file edited after delivery blocks the complete undo, including other files.
    (root / "new.c").write_bytes(b"user changed new.c")
    response = await client.post(endpoint)
    assert response.status_code == 409 and "new.c" in response.json()["detail"]
    assert (root / "module.c").read_bytes() == b"int new = 2;\n"
    (root / "new.c").write_bytes(b"int added = 3;\n")
    app.state.agent.maintaining = True
    assert (await client.post(endpoint)).status_code == 409
    assert (await client.post(f"{API_PREFIX}/tasks/{record['id']}/recheck")).status_code == 409
    assert (await start(client, identifier, "new task")).status_code == 409
    app.state.agent.maintaining = False
    response = await client.post(endpoint)
    assert response.status_code == 200 and response.json()["status"] == "completed"
    assert (root / "module.c").read_bytes() == b"int old = 1;\r\n"
    assert not (root / "new.c").exists()
    assert all(p.read_bytes() == raw for p, raw in artifacts.items())
    assert (await client.post(endpoint)).json() == response.json()

    restarted = create_app(options(root))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=restarted),
                                 base_url="http://127.0.0.1") as reopened:
        delivery = (await reopened.get(f"{API_PREFIX}/tasks/{record['id']}/delivery")).json()
        assert not delivery["undo"]["available"]
        assert delivery["undo"]["runs"][0]["status"] == "completed"
        assert (await reopened.get(f"{API_PREFIX}/tasks/{record['id']}")).json() == record
