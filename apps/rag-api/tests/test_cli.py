"""Offline tests for the ingest CLI and P0 task runner."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from app.ingest.cli import CliRuntime, RegistrationRequest, build_parser, main, run_command
from app.ingest.tasks import (
    InProcessTaskRunner,
    P2TaskRunnerDisabledError,
    RecoverablePostgresTaskRunner,
)

DOCUMENT_A = UUID("11111111-1111-4111-8111-111111111111")
DOCUMENT_B = UUID("22222222-2222-4222-8222-222222222222")


@dataclass(frozen=True)
class FakeCommand:
    document_id: UUID
    pdf_path: Path
    layout_path: Path
    product: str
    version: str
    language: str
    title: str


@dataclass(frozen=True)
class FakeResult:
    document_id: UUID
    chunk_count: int
    vector_count: int
    status: str


class FakePipeline:
    def __init__(self, *, result_status: str = "parsed") -> None:
        self.result_status = result_status
        self.registrations: list[RegistrationRequest] = []
        self.commands: list[FakeCommand] = []

    async def register(self, request: RegistrationRequest) -> object:
        self.registrations.append(request)
        return SimpleNamespace(document_id=DOCUMENT_A, status="pending")

    async def run(self, command: FakeCommand) -> FakeResult:
        self.commands.append(command)
        return FakeResult(command.document_id, 3, 3, self.result_status)


class FakeSubmitRegistrar:
    def __init__(self) -> None:
        self.commands: list[object] = []

    async def submit(self, command: object) -> object:
        self.commands.append(command)
        return SimpleNamespace(document_id=DOCUMENT_A, status="pending")


class FakeRepository:
    def __init__(self) -> None:
        self.documents: dict[UUID, object] = {
            DOCUMENT_A: SimpleNamespace(
                document_id=DOCUMENT_A,
                status="parsed",
                progress=100,
                error=None,
                current=True,
                product="M8P6272",
                version="V1.10",
            ),
            DOCUMENT_B: SimpleNamespace(
                document_id=DOCUMENT_B,
                status="failed",
                progress=30,
                error="parse failed",
                current=False,
                product="M8P6272",
                version="V1.11",
            ),
        }
        self.chunk_counts = {DOCUMENT_A: 2, DOCUMENT_B: 0}
        self.get_calls: list[tuple[UUID, bool]] = []
        self.count_calls: list[UUID] = []

    async def get_document(self, document_id: UUID, *, for_update: bool = False) -> object:
        self.get_calls.append((document_id, for_update))
        return self.documents.get(document_id)

    async def list_documents_for_reconcile(self) -> list[object]:
        return list(self.documents.values())

    async def count_chunks(self, document_id: UUID) -> int:
        self.count_calls.append(document_id)
        return self.chunk_counts[document_id]


class FakeProvider:
    model_id = "text-embedding-v4"

    def __init__(self) -> None:
        self.embed_calls: list[list[str]] = []
        self.closed = False

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(texts)
        return [[1.0, 0.0] for _ in texts]

    async def aclose(self) -> None:
        self.closed = True


class FakeProviderFactory:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.provider = FakeProvider()

    def __call__(self, provider_name: str) -> FakeProvider:
        self.calls.append(provider_name)
        return self.provider


class FakeStore:
    def __init__(self) -> None:
        self.counts = {DOCUMENT_A: 2, DOCUMENT_B: 1}
        self.count_calls: list[UUID] = []
        self.reindex_calls: list[tuple[object, FakeProvider]] = []
        self.deleted: list[UUID] = []
        self.reindex_report: object = {"success": True, "documents": 2, "vectors": 2}

    async def count_document(self, document_id: UUID) -> int:
        self.count_calls.append(document_id)
        return self.counts[document_id]

    async def reindex_embeddings(self, *, repository: object, provider: FakeProvider) -> object:
        self.reindex_calls.append((repository, provider))
        await provider.embed(["first", "second"])
        return self.reindex_report

    async def delete_document(self, document_id: UUID) -> None:
        self.deleted.append(document_id)


class FakeStorage:
    def __init__(self) -> None:
        self.staging_cleanups: list[UUID] = []
        self.asset_cleanups: list[UUID] = []

    def cleanup_staging(self, document_id: UUID) -> None:
        self.staging_cleanups.append(document_id)

    def delete_document_assets(self, document_id: UUID) -> None:
        self.asset_cleanups.append(document_id)


def _runtime(
    *,
    pipeline: FakePipeline | None = None,
    repository: FakeRepository | None = None,
    store: FakeStore | None = None,
    provider_factory: FakeProviderFactory | None = None,
    storage: FakeStorage | None = None,
) -> CliRuntime:
    selected_pipeline = pipeline or FakePipeline()
    return CliRuntime(
        pipeline=selected_pipeline,
        registrar=selected_pipeline,
        repository=repository or FakeRepository(),
        store=store or FakeStore(),
        provider_factory=provider_factory or FakeProviderFactory(),
        storage=storage or FakeStorage(),
        command_factory=FakeCommand,
    )


def _files(tmp_path: Path) -> tuple[Path, Path]:
    pdf = tmp_path / "spec.pdf"
    layout = tmp_path / "layout.json"
    pdf.write_bytes(b"%PDF-1.7\n")
    layout.write_text("{}", encoding="utf-8")
    return pdf, layout


def test_parser_accepts_contract_commands_and_language_default(tmp_path: Path) -> None:
    pdf, layout = _files(tmp_path)
    parser = build_parser()

    args = parser.parse_args(
        [
            "run",
            "--pdf",
            str(pdf),
            "--layout",
            str(layout),
            "--product",
            "M8P6272",
            "--version",
            "1.10",
        ]
    )

    assert args.command == "run"
    assert args.language == "zh-CN"
    assert parser.parse_args(["status", "--document-id", str(DOCUMENT_A)]).document_id == (
        DOCUMENT_A
    )
    assert parser.parse_args(["reconcile"]).command == "reconcile"
    assert parser.parse_args(
        ["reindex-embeddings", "--provider", "dashscope"]
    ).provider == "dashscope"
    assert parser.parse_args(["cleanup", "--document-id", str(DOCUMENT_A)]).command == (
        "cleanup"
    )


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["run", "--pdf", "spec.pdf"],
        [
            "run",
            "--pdf",
            "spec.pdf",
            "--layout",
            "layout.json",
            "--product",
            "M8P6272",
            "--version",
            "1.10",
            "--language",
            "fr",
        ],
        ["status"],
        ["reindex-embeddings"],
        ["reindex-embeddings", "--provider", "hf"],
        ["cleanup", "--document-id", "not-a-uuid"],
    ],
)
def test_parser_reports_missing_or_invalid_arguments(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(arguments)

    assert exc_info.value.code == 2


def test_run_uses_registered_pg_uuid_and_shared_pipeline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pdf, layout = _files(tmp_path)
    pipeline = FakePipeline()
    runtime = _runtime(pipeline=pipeline)
    args = build_parser().parse_args(
        [
            "run",
            "--pdf",
            str(pdf),
            "--layout",
            str(layout),
            "--product",
            " M8P6272 ",
            "--version",
            " 1.10 ",
            "--language",
            "zh-CN",
        ]
    )

    exit_code = asyncio.run(run_command(args, runtime))

    assert exit_code == 0
    assert len(pipeline.registrations) == 1
    assert pipeline.registrations[0].product == "M8P6272"
    assert len(pipeline.commands) == 1
    command = pipeline.commands[0]
    assert command.document_id == DOCUMENT_A
    assert command.pdf_path == pdf
    assert command.layout_path == layout
    assert command.title == "spec"
    output = json.loads(capsys.readouterr().out)
    assert output["document_id"] == str(DOCUMENT_A)
    assert output["result"]["vector_count"] == 3


def test_run_returns_failure_for_failed_pipeline_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pdf, layout = _files(tmp_path)
    args = build_parser().parse_args(
        [
            "run",
            "--pdf",
            str(pdf),
            "--layout",
            str(layout),
            "--product",
            "M8P6272",
            "--version",
            "1.10",
        ]
    )

    runtime = _runtime(pipeline=FakePipeline(result_status="failed"))
    exit_code = asyncio.run(run_command(args, runtime))

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out)["result"]["status"] == "failed"


def test_run_document_service_fallback_streams_sha_before_pipeline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pdf, layout = _files(tmp_path)
    pipeline = FakePipeline()
    registrar = FakeSubmitRegistrar()
    runtime = _runtime(pipeline=pipeline)
    runtime.registrar = registrar
    args = build_parser().parse_args(
        [
            "run",
            "--pdf",
            str(pdf),
            "--layout",
            str(layout),
            "--product",
            "M8P6272",
            "--version",
            "1.10",
        ]
    )

    exit_code = asyncio.run(run_command(args, runtime))

    assert exit_code == 0
    assert len(registrar.commands) == 1
    submission = registrar.commands[0]
    assert submission.sha256 == hashlib.sha256(pdf.read_bytes()).hexdigest()
    assert submission.size_bytes == pdf.stat().st_size
    assert submission.version == "V1.10"
    assert pipeline.commands[0].document_id == DOCUMENT_A
    assert json.loads(capsys.readouterr().out)["command"] == "run"


def test_status_calls_repository_and_prints_safe_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = FakeRepository()
    args = build_parser().parse_args(["status", "--document-id", str(DOCUMENT_A)])

    exit_code = asyncio.run(run_command(args, _runtime(repository=repository)))

    assert exit_code == 0
    assert repository.get_calls == [(DOCUMENT_A, False)]
    output = json.loads(capsys.readouterr().out)
    assert output["document"]["status"] == "parsed"
    assert output["document"]["progress"] == 100


def test_status_not_found_returns_failure(capsys: pytest.CaptureFixture[str]) -> None:
    missing_id = UUID("33333333-3333-4333-8333-333333333333")
    args = build_parser().parse_args(["status", "--document-id", str(missing_id)])

    exit_code = asyncio.run(run_command(args, _runtime()))

    assert exit_code == 1
    assert "document not found" in capsys.readouterr().err


def test_reconcile_reports_pg_milvus_mismatch_and_orphan_vectors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = FakeRepository()
    store = FakeStore()
    args = build_parser().parse_args(["reconcile"])

    exit_code = asyncio.run(run_command(args, _runtime(repository=repository, store=store)))

    assert exit_code == 1
    assert repository.count_calls == [DOCUMENT_A, DOCUMENT_B]
    assert store.count_calls == [DOCUMENT_A, DOCUMENT_B]
    output = json.loads(capsys.readouterr().out)
    assert output["checked"] == 2
    assert output["mismatch_count"] == 1
    assert output["orphan_vector_documents"] == 1
    assert output["documents"][1]["document_id"] == str(DOCUMENT_B)


def test_reconcile_returns_success_when_counts_match(
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = FakeStore()
    store.counts[DOCUMENT_B] = 0
    args = build_parser().parse_args(["reconcile"])

    exit_code = asyncio.run(run_command(args, _runtime(store=store)))

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["mismatch_count"] == 0


def test_reindex_calls_selected_provider_and_store_and_closes_provider(
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = FakeRepository()
    store = FakeStore()
    factory = FakeProviderFactory()
    args = build_parser().parse_args(["reindex-embeddings", "--provider", "dashscope"])

    exit_code = asyncio.run(
        run_command(
            args,
            _runtime(repository=repository, store=store, provider_factory=factory),
        )
    )

    assert exit_code == 0
    assert factory.calls == ["dashscope"]
    assert store.reindex_calls == [(repository, factory.provider)]
    assert factory.provider.embed_calls == [["first", "second"]]
    assert factory.provider.closed
    output = json.loads(capsys.readouterr().out)
    assert output["provider"] == "dashscope"
    assert output["model_id"] == "text-embedding-v4"


def test_reindex_report_mismatch_returns_failure(capsys: pytest.CaptureFixture[str]) -> None:
    store = FakeStore()
    store.reindex_report = {"success": True, "mismatch_count": 1}
    args = build_parser().parse_args(["reindex-embeddings", "--provider", "dashscope"])

    exit_code = asyncio.run(run_command(args, _runtime(store=store)))

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out)["report"]["mismatch_count"] == 1


def test_cleanup_deletes_only_failed_document_residue(
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = FakeRepository()
    store = FakeStore()
    storage = FakeStorage()
    args = build_parser().parse_args(["cleanup", "--document-id", str(DOCUMENT_B)])

    exit_code = asyncio.run(
        run_command(
            args,
            _runtime(repository=repository, store=store, storage=storage),
        )
    )

    assert exit_code == 0
    assert repository.get_calls == [(DOCUMENT_B, True)]
    assert store.deleted == [DOCUMENT_B]
    assert storage.staging_cleanups == [DOCUMENT_B]
    assert storage.asset_cleanups == [DOCUMENT_B]
    assert json.loads(capsys.readouterr().out)["status_retained"] == "failed"


def test_cleanup_refuses_current_document(capsys: pytest.CaptureFixture[str]) -> None:
    store = FakeStore()
    args = build_parser().parse_args(["cleanup", "--document-id", str(DOCUMENT_A)])

    exit_code = asyncio.run(run_command(args, _runtime(store=store)))

    assert exit_code == 1
    assert store.deleted == []
    assert "restricted" in capsys.readouterr().err


def test_main_returns_integer_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pdf, layout = _files(tmp_path)

    exit_code = main(
        [
            "run",
            "--pdf",
            str(pdf),
            "--layout",
            str(layout),
            "--product",
            "M8P6272",
            "--version",
            "1.10",
        ],
        runtime=_runtime(),
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["command"] == "run"


class FakeTaskRepository:
    def __init__(self) -> None:
        self.transitions: list[tuple[UUID, str, str | None]] = []

    async def set_status(
        self,
        document_id: UUID,
        status: object,
        *,
        error: str | None = None,
    ) -> None:
        status_text = str(getattr(status, "value", status))
        self.transitions.append((document_id, status_text, error))


def test_in_process_runner_is_serial_and_persists_status_progression() -> None:
    async def scenario() -> tuple[FakeTaskRepository, list[UUID], int]:
        repository = FakeTaskRepository()
        processed: list[UUID] = []
        active = 0
        max_active = 0

        async def processor(document_id: UUID) -> object:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0)
            processed.append(document_id)
            active -= 1
            return SimpleNamespace(status="parsed")

        runner = InProcessTaskRunner(repository=repository, processor=processor)
        await runner.start()
        assert await runner.submit(DOCUMENT_A) == DOCUMENT_A
        assert await runner.submit(DOCUMENT_A) == DOCUMENT_A
        assert await runner.submit(DOCUMENT_B) == DOCUMENT_B
        await runner.stop()
        return repository, processed, max_active

    repository, processed, max_active = asyncio.run(scenario())

    assert processed == [DOCUMENT_A, DOCUMENT_B]
    assert max_active == 1
    for document_id in (DOCUMENT_A, DOCUMENT_B):
        statuses = [
            status
            for recorded_id, status, _error in repository.transitions
            if recorded_id == document_id
        ]
        assert statuses == ["pending", "parsing", "parsed"]


def test_in_process_runner_persists_redacted_failure_and_keeps_running() -> None:
    async def scenario() -> FakeTaskRepository:
        repository = FakeTaskRepository()

        async def processor(document_id: UUID) -> None:
            if document_id == DOCUMENT_A:
                raise RuntimeError("sensitive-provider-detail")

        runner = InProcessTaskRunner(repository=repository, processor=processor)
        await runner.start()
        await runner.submit(DOCUMENT_A)
        await runner.submit(DOCUMENT_B)
        await runner.stop()
        return repository

    repository = asyncio.run(scenario())

    first_failure = [row for row in repository.transitions if row[:2] == (DOCUMENT_A, "failed")]
    assert len(first_failure) == 1
    assert first_failure[0][2] == "RuntimeError: ingest task failed"
    assert all(
        "sensitive-provider-detail" not in (error or "")
        for _, _, error in repository.transitions
    )
    second_statuses = [
        status for doc_id, status, _ in repository.transitions if doc_id == DOCUMENT_B
    ]
    assert second_statuses == ["pending", "parsing", "parsed"]


def test_recoverable_postgres_runner_is_explicitly_disabled() -> None:
    async def scenario() -> None:
        runner = RecoverablePostgresTaskRunner()
        assert runner.enabled is False
        with pytest.raises(P2TaskRunnerDisabledError, match="P2"):
            await runner.start()
        with pytest.raises(P2TaskRunnerDisabledError, match="P2"):
            await runner.submit(DOCUMENT_A)
        await runner.stop()

    asyncio.run(scenario())
