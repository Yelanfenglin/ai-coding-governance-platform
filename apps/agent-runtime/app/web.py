"""Local HTTP/SSE adapter for the existing runtime and Vue workspace.

One local process owns the selected project. The browser never supplies an
arbitrary filesystem path, provider configuration, or verification command.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import REGISTRY, CollectorRegistry
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field

from . import cli, metrics
from .circuit import from_env as circuit_from_env
from .observability import configure
from .reverification import recheck_delivery, recheck_summary
from .skills import load_skills
from .undo import undo_delivery, undo_summary

API_PREFIX = "/api/agent"
ACTIVE = {"pending", "running", "cancelling"}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class ConversationInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    title: str = Field(min_length=1)
    project_id: str


class TaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    conversation_id: UUID
    prompt: str = Field(min_length=1)
    skill: str | None = Field(default=None, min_length=1)
    idempotency_key: UUID


@dataclass
class WebTask:
    record: dict
    directory: Path
    events: list[dict] = field(default_factory=list)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    runner: asyncio.Task | None = None

    def save(self) -> None:
        _write_json(self.directory / "task.json", self.record)

    def publish(self, kind: str, payload: dict) -> None:
        event = {"sequence": len(self.events) + 1, "type": kind, "payload": payload,
                 "observed_at": datetime.now(UTC).isoformat()}
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.events.append(event)
        self.changed.set()


class LocalAgent:
    def __init__(self, options: argparse.Namespace) -> None:
        self.options = options
        # 单进程单事件循环：这一个注册表就是 Web 侧的“跨任务共享”。
        # CLI 每次 run_once 自己建一个（一个任务一个进程），所以只有 Web 有跨任务保护。
        self.breakers = circuit_from_env()
        workspace = options.workspace or options.demo_project or options.firmware_project
        if not workspace:
            raise ValueError("agent-web requires --workspace, --demo-project or --firmware-project")
        self.workspace = cli._workspace_target(workspace)
        if not self.workspace.is_dir():
            raise ValueError("the selected project directory does not exist")
        if not options.base_url:
            raise ValueError("OPENAI_BASE_URL (or --base-url) is required")
        self.state = (Path(options.state_dir) if options.state_dir else
                      self.workspace / ".agent-runtime").resolve()
        self.root = self.state / "web"
        self.jobs: dict[str, WebTask] = {}
        self.active: WebTask | None = None
        self.maintaining = False
        for path in (self.root / "tasks").glob("*/task.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            events_path = path.parent / "events.jsonl"
            events = ([json.loads(line) for line in events_path.read_text(
                encoding="utf-8").splitlines() if line] if events_path.exists() else [])
            job = WebTask(record, path.parent, events)
            self.jobs[record["id"]] = job
            if record["status"] in ACTIVE:
                self.finish(job, "failed", {"error": "本地 Agent 服务已重启；可在原会话继续。"}, 1)
            elif not events or events[-1]["type"] != "TaskFinished":
                # Metadata can be replaced just before a process stops, before the
                # terminal SSE frame is appended. Reconstruct that frame, never the tools.
                job.publish("TaskFinished", {"status": record["status"],
                            "response": record["response"], "exit_code": record["exit_code"]})

    def conversations(self) -> list[dict]:
        return sorted((json.loads(path.read_text(encoding="utf-8")) for path in
                       (self.root / "conversations").glob("*/conversation.json")),
                      key=lambda row: row["created_at"], reverse=True)

    def conversation(self, identifier: UUID) -> dict:
        path = self.root / "conversations" / str(identifier) / "conversation.json"
        if not path.is_file():
            raise HTTPException(404, "会话不存在")
        return json.loads(path.read_text(encoding="utf-8"))

    def job(self, identifier: UUID) -> WebTask:
        if str(identifier) not in self.jobs:
            raise HTTPException(404, "任务不存在")
        return self.jobs[str(identifier)]

    def finish(self, job: WebTask, status: str, response: dict, code: int) -> None:
        job.record.update(status=status, response=response, exit_code=code,
                          finished_at=datetime.now(UTC).isoformat())
        job.save()
        job.publish("TaskFinished", {"status": status, "response": response, "exit_code": code})

    async def execute(self, job: WebTask) -> None:
        job.record["status"] = "running"
        job.record["started_at"] = datetime.now(UTC).isoformat()
        job.save()
        job.publish("TaskStarted", {"prompt": job.record["prompt"],
                                    "skill": job.record.get("skill")})
        options = copy.copy(self.options)
        options.question = job.record["prompt"]
        options.skill = job.record.get("skill")
        options.state_dir = str(self.state)
        options.session = str(self.root / "conversations" / job.record["conversation_id"]
                              / "session.jsonl")
        options.delivery_dir = job.directory / "delivery"
        options.json = True
        response: dict = {}

        def receive(result: dict, _code: int) -> None:
            response.update(result)

        try:
            code = await cli.run_once(
                options, event_sink=lambda event: job.publish(type(event).__name__, asdict(event)),
                result_sink=receive, cancel_event=job.cancelled, console=False,
                task_id=job.record["id"], conversation_id=job.record["conversation_id"],
                breakers=self.breakers,
            )
            status = ("cancelled" if response.get("status") == "aborted" else
                      "budget_exceeded" if response.get("status") == "budget_exceeded" else
                      "completed" if code == 0 else "failed")
            if not response:
                response["error"] = "Agent 未能启动，请检查本机服务配置和日志。"
            self.finish(job, status, response, code)
        except asyncio.CancelledError:
            self.finish(job, "cancelled", {"error": "本地服务已停止"}, 1)
            raise
        except Exception as exc:
            self.finish(job, "failed", {"error": str(exc)}, 1)
        finally:
            self.active = None


def create_app(options: argparse.Namespace, *, ui_directory: Path | None = None) -> FastAPI:
    agent = LocalAgent(options)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        metrics.configure_logging(os.getenv("MONITORING_LOG_PATH"))
        telemetry = configure()
        try:
            yield
        finally:
            if agent.active is not None:
                agent.active.cancelled.set()
                await agent.active.runner
            if telemetry is not None:
                await asyncio.to_thread(telemetry.shutdown)

    # Native FastAPI tracing otherwise exports every UI poll and parents a
    # background task under its already-finished POST request.
    app = FastAPI(title="Local Embedded Agent", lifespan=lifespan, telemetry={"tracing": False})
    http_registry = CollectorRegistry()
    http_registry.register(REGISTRY)
    Instrumentator(excluded_handlers=["/metrics", ".*/health"],
                   registry=http_registry).instrument(app).expose(
                       app, include_in_schema=False,
                   )
    app.state.agent = agent

    @app.middleware("http")
    async def local_browser_only(request: Request, call_next):
        # A drive-by website must not command a privileged localhost Agent.
        host = request.headers.get("host", "")
        origin = request.headers.get("origin")
        hostname = urlsplit(f"http://{host}").hostname
        # Docker's Prometheus uses this host name. The exception exposes only
        # aggregate GET metrics, never the task/file APIs or browser cross-origin access.
        scrape = (request.method == "GET" and request.url.path == "/metrics"
                  and hostname == "host.docker.internal" and origin is None)
        if ((hostname not in {"127.0.0.1", "localhost", "::1"} and not scrape)
                or (origin and urlsplit(origin).netloc != host)):
            return JSONResponse({"detail": "只接受本机同源访问"}, status_code=403)
        return await call_next(request)

    api = APIRouter(prefix=API_PREFIX)

    @api.get("/health")
    async def health():
        job = agent.active
        return {"ready": True, "service": "embedded-agent", "active_task": (
            {"id": job.record["id"], "status": job.record["status"],
             "started_at": job.record.get("started_at"),
             "last_event_at": job.events[-1].get("observed_at") if job.events else None,
             "last_event_type": job.events[-1]["type"] if job.events else None}
            if job is not None else None
        )}

    @api.get("/projects")
    async def projects():
        return [{"id": "local", "display_name": agent.workspace.name,
                 "path": str(agent.workspace), "model": options.model,
                 "product": options.product, "version": options.version}]

    @api.get("/conversations")
    async def conversations():
        return agent.conversations()

    @api.get("/skills")
    async def skills():
        return [{"name": name, "title": skill["title"], "description": skill["description"]}
                for name, skill in load_skills().items()]

    @api.post("/conversations", status_code=201)
    async def create_conversation(body: ConversationInput):
        if body.project_id != "local":
            raise HTTPException(404, "工程不存在")
        record = {"id": str(uuid4()), "project_id": body.project_id, "title": body.title,
                  "created_at": datetime.now(UTC).isoformat()}
        _write_json(agent.root / "conversations" / record["id"] / "conversation.json", record)
        return record

    @api.get("/conversations/{identifier}")
    async def get_conversation(identifier: UUID):
        conversation = agent.conversation(identifier)
        jobs = [job.record for job in agent.jobs.values()
                if job.record["conversation_id"] == str(identifier)]
        return {**conversation, "tasks": sorted(jobs, key=lambda row: row["created_at"])}

    @api.post("/tasks", status_code=201)
    async def create_task(body: TaskInput):
        agent.conversation(body.conversation_id)
        for job in agent.jobs.values():
            if job.record["idempotency_key"] == str(body.idempotency_key):
                if (job.record["conversation_id"] != str(body.conversation_id)
                        or job.record["prompt"] != body.prompt
                        or job.record.get("skill") != body.skill):
                    raise HTTPException(409, "重复请求的内容不同")
                return job.record
        if agent.active is not None or agent.maintaining:
            raise HTTPException(409, "当前工程已有任务运行，请等待完成或先取消")
        catalog = load_skills()
        if body.skill is not None and body.skill not in catalog:
            raise HTTPException(400, "技能不存在，请从开发技能中重新选择")
        identifier = str(uuid4())
        record = {"id": identifier, "conversation_id": str(body.conversation_id),
                  "prompt": body.prompt, "idempotency_key": str(body.idempotency_key),
                  "skill": body.skill,
                  "skill_title": catalog[body.skill]["title"] if body.skill else None,
                  "status": "pending", "created_at": datetime.now(UTC).isoformat(),
                  "response": None, "exit_code": None}
        job = WebTask(record, agent.root / "tasks" / identifier)
        job.save()
        agent.jobs[identifier] = job
        agent.active = job
        job.runner = asyncio.create_task(agent.execute(job))
        return record

    @api.get("/tasks/{identifier}")
    async def get_task(identifier: UUID):
        return agent.job(identifier).record

    @api.post("/tasks/{identifier}/cancel", status_code=202)
    async def cancel_task(identifier: UUID):
        job = agent.job(identifier)
        if job.record["status"] in ACTIVE:
            job.cancelled.set()
            job.record["status"] = "cancelling"
            job.save()
        return job.record

    @api.get("/tasks/{identifier}/events")
    async def events(identifier: UUID, request: Request, after: int = 0):
        job = agent.job(identifier)
        try:
            sequence = max(after, int(request.headers.get("last-event-id", "0")))
        except ValueError as exc:
            raise HTTPException(400, "无效的事件序号") from exc

        async def stream():
            nonlocal sequence
            while True:
                job.changed.clear()
                for event in job.events[sequence:]:
                    sequence = event["sequence"]
                    yield (f"id: {sequence}\ndata: "
                           + json.dumps(event, ensure_ascii=False) + "\n\n")
                if job.record["status"] not in ACTIVE:
                    return
                try:
                    await asyncio.wait_for(job.changed.wait(), timeout=15)
                except TimeoutError:
                    yield ": keepalive\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @api.get("/tasks/{identifier}/delivery")
    async def delivery(identifier: UUID):
        job = agent.job(identifier)
        path = job.directory / "delivery" / "delivery.json"
        if not path.is_file():
            raise HTTPException(404, "本次任务尚无交付记录")
        report = json.loads(path.read_text(encoding="utf-8"))
        return {key: report.get(key) for key in (
            "changed_files", "commands", "evidence", "verification", "task_acceptance", "decisions",
        )} | {"diff": (path.parent / "changes.diff").read_text(encoding="utf-8", errors="replace"),
              "recheck": recheck_summary(path, options.verification, agent.workspace),
              "undo": undo_summary(path, agent.workspace)}

    @api.post("/tasks/{identifier}/recheck")
    async def recheck(identifier: UUID):
        job = agent.job(identifier)
        if agent.active is not None or agent.maintaining:
            raise HTTPException(409, "当前工程正在执行任务、复验或撤回，请等待完成")
        path = job.directory / "delivery" / "delivery.json"
        if not path.is_file():
            raise HTTPException(404, "本次任务尚无交付记录")
        agent.maintaining = True
        try:
            return await recheck_delivery(path, options.verification, agent.workspace)
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(409, str(exc)) from exc
        finally:
            agent.maintaining = False

    @api.post("/tasks/{identifier}/undo")
    async def undo(identifier: UUID):
        job = agent.job(identifier)
        if agent.active is not None or agent.maintaining:
            raise HTTPException(409, "当前工程正在执行任务、复验或撤回，请等待完成")
        path = job.directory / "delivery" / "delivery.json"
        if not path.is_file():
            raise HTTPException(404, "本次任务尚无交付记录")
        agent.maintaining = True
        try:
            # No await between preflight and restoration: another Web task cannot
            # enter this workspace while files are being restored.
            return undo_delivery(path, agent.workspace)
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(409, str(exc)) from exc
        finally:
            agent.maintaining = False

    app.include_router(api)
    if ui_directory is not None:
        app.mount("/", StaticFiles(directory=ui_directory, html=True), name="web")
    return app


def main(argv: list[str] | None = None) -> int:
    import uvicorn
    from dotenv import load_dotenv

    # Environment defaults must be loaded before constructing the shared CLI parser.
    env_parser = argparse.ArgumentParser(add_help=False)
    env_parser.add_argument("--env-file", type=Path)
    env_options, _ = env_parser.parse_known_args(argv)
    if env_options.env_file:
        load_dotenv(env_options.env_file, override=False)
    parser = cli.build_parser()
    parser.prog = "agent-web"
    parser.description = "Serve the local Agent and existing Vue workspace (no Codex process)."
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--port", type=int, default=8081)
    options = parser.parse_args(argv)
    if not (options.workspace or options.demo_project or options.firmware_project):
        options.workspace = os.environ.get("AGENT_WORKSPACE")
    directory = Path(__file__).resolve().parents[2] / "web-next" / "dist"
    try:
        app = create_app(options, ui_directory=directory if directory.is_dir() else None)
    except ValueError as exc:
        parser.error(str(exc))
    uvicorn.run(app, host="127.0.0.1", port=options.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
