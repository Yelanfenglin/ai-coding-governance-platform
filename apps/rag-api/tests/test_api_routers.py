from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest

from app.dependencies import get_document_service, get_rag_service, get_storage_manager
from app.document_service import (
    DocumentConflictError,
    DocumentDeleteError,
    DocumentNotFoundError,
    DocumentService,
    DocumentSubmitCommand,
)
from app.main import create_app
from app.planning import QueryPlan
from app.rag_service import RagNotFoundError, RagQueryService
from app.schemas import JobAccepted, QueryRequest


def _settings(**updates: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "api_prefix": "/api/rag",
        "api_key": "",
        "embedding_dimension": 2,
        "query_top_k_max": 50,
        "context_top_n": 4,
        "reranker_enabled": False,
        "llm_enabled": False,
        "generation_failure_mode": "retrieval_only",
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _request(app: object, method: str, path: str, **kwargs: object) -> httpx.Response:
    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(run())


class _EmptySearchService:
    async def search(self, _params: object) -> list[object]:
        return []


def test_optional_api_key_authentication() -> None:
    app = create_app(_settings(api_key="shared-secret"))
    app.dependency_overrides[get_rag_service] = lambda: _EmptySearchService()

    missing = _request(app, "GET", "/api/rag/search", params={"question": "q"})
    wrong = _request(
        app,
        "GET",
        "/api/rag/search",
        params={"question": "q"},
        headers={"X-API-Key": "wrong"},
    )
    allowed = _request(
        app,
        "GET",
        "/api/rag/search",
        params={"question": "q"},
        headers={"X-API-Key": "shared-secret"},
    )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert allowed.status_code == 200
    assert allowed.json() == []


def test_unconfigured_api_key_does_not_require_header_and_invalid_query_is_422() -> None:
    app = create_app(_settings())
    app.dependency_overrides[get_rag_service] = lambda: _EmptySearchService()

    assert _request(app, "GET", "/api/rag/search", params={"question": "okay"}).status_code == 200
    assert _request(app, "GET", "/api/rag/search", params={"question": "   "}).status_code == 422
    assert (
        _request(
            app,
            "POST",
            "/api/rag/query",
            json={"question": "q", "version": "1.2.3"},
        ).status_code
        == 422
    )

    configured_service = RagQueryService(
        settings=_settings(query_top_k_max=5),
        repository=_Repository(with_documents=False),
    )
    configured_app = create_app(_settings())
    configured_app.dependency_overrides[get_rag_service] = lambda: configured_service
    assert (
        _request(
            configured_app,
            "GET",
            "/api/rag/search",
            params={"question": "q", "top_k": 6},
        ).status_code
        == 422
    )


class _Repository:
    def __init__(
        self,
        *,
        with_documents: bool = True,
        chunk_fields: dict[str, object] | None = None,
    ) -> None:
        self.document_id = UUID("11111111-1111-1111-1111-111111111111")
        self.chunk_id = UUID("22222222-2222-2222-2222-222222222222")
        self.scope: object | None = None
        self.with_documents = with_documents
        self.chunk_fields = chunk_fields or {}
        self.chunks_for_sparse: list[dict[str, object]] = []
        self.neighbor_pairs: list[tuple[dict[str, object], dict[str, object]]] = []

    async def resolve_search_documents(self, scope: object) -> list[dict[str, object]]:
        self.scope = scope
        if not self.with_documents:
            return []
        return [
            {
                "id": self.document_id,
                "product": "M8P6272",
                "version": "V1.10",
                "status": "parsed",
                "current": True,
            }
        ]

    async def get_chunks_by_embedding_ids(
        self, embedding_ids: list[str]
    ) -> dict[str, dict[str, object]]:
        assert embedding_ids == ["embedding-1"]
        return {
            "embedding-1": {
                "id": self.chunk_id,
                "document_id": self.document_id,
                "embedding_id": "embedding-1",
                "section_path": "2 > 2.6",
                "page_start": 15,
                "page_end": 15,
                "kind": "TABLE",
                "text": "ADCON0 bit 1:0 select the clock source.",
                **self.chunk_fields,
            }
        }

    async def list_chunks_for_documents(self, document_ids: list[UUID]) -> list[dict[str, object]]:
        assert tuple(document_ids) == (self.document_id,)
        return list(self.chunks_for_sparse)

    async def get_chunk_neighbors(
        self, chunk_id: UUID, radius: int = 1
    ) -> list[tuple[dict[str, object], dict[str, object]]]:
        assert chunk_id == self.chunk_id
        assert radius == 1
        return list(self.neighbor_pairs)


class _EmbeddingProvider:
    dimension = 2
    model_id = "embedding-v1"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        assert texts == ["ADCON0 的低两位是什么？"]
        return [[0.6, 0.8]]


class _VectorStore:
    dimension = 2
    embedding_fingerprint = "embedding-v1"

    def __init__(self) -> None:
        self.scope: object | None = None

    async def search(
        self, vector: list[float], scope: object, limit: int
    ) -> list[dict[str, object]]:
        assert vector == [0.6, 0.8]
        assert limit == 8
        self.scope = scope
        return [{"embedding_id": "embedding-1", "score": 0.84}]


def test_query_retrieval_only_is_explicit_and_citations_are_server_generated() -> None:
    repository = _Repository()
    store = _VectorStore()
    service = RagQueryService(
        store=store,
        embedding_provider=_EmbeddingProvider(),
        settings=_settings(),
        repository=repository,
    )
    app = create_app(_settings())
    app.dependency_overrides[get_rag_service] = lambda: service

    response = _request(
        app,
        "POST",
        "/api/rag/query",
        json={
            "question": "  ADCON0 的低两位是什么？ ",
            "product": "M8P6272",
            "version": "1.10",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["answer"] == ""
    assert payload["model"] == "retrieval-only"
    assert payload["degraded"] is True
    assert payload["degraded_reason"] == "generation_disabled"
    assert payload["citations"] == [
        {
            "product": "M8P6272",
            "version": "V1.10",
            "section_path": "2 > 2.6",
            "page": 15,
            "kind": "TABLE",
            "text": "ADCON0 bit 1:0 select the clock source.",
            "score": 0.84,
            "chunk_id": "22222222-2222-2222-2222-222222222222",
            "vector_score": 0.84,
            "rerank_score": None,
        }
    ]
    assert repository.scope.product == "M8P6272"
    assert repository.scope.version == "V1.10"
    assert store.scope.document_ids == (repository.document_id,)
    assert not hasattr(store.scope, "expression")


def test_search_kind_uses_typed_store_scope_not_a_user_expression() -> None:
    repository = _Repository()
    store = _VectorStore()
    service = RagQueryService(
        store=store,
        embedding_provider=_EmbeddingProvider(),
        settings=_settings(),
        repository=repository,
    )
    app = create_app(_settings())
    app.dependency_overrides[get_rag_service] = lambda: service

    response = _request(
        app,
        "GET",
        "/api/rag/search",
        params={
            "question": "ADCON0 的低两位是什么？",
            "product": "M8P6272' or current == true",
            "version": "1.10",
            "kind": "TABLE",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()[0]["vector_score"] == 0.84
    assert [kind.value for kind in store.scope.kinds] == ["TABLE"]
    assert store.scope.document_ids == (repository.document_id,)
    assert not hasattr(store.scope, "expression")


def test_query_generation_error_mode_returns_503_and_empty_scope_returns_404() -> None:
    error_service = RagQueryService(
        store=_VectorStore(),
        embedding_provider=_EmbeddingProvider(),
        settings=_settings(generation_failure_mode="error"),
        repository=_Repository(),
    )
    error_app = create_app(_settings())
    error_app.dependency_overrides[get_rag_service] = lambda: error_service
    unavailable = _request(
        error_app,
        "POST",
        "/api/rag/query",
        json={"question": "ADCON0 的低两位是什么？"},
    )
    assert unavailable.status_code == 503

    empty_service = RagQueryService(
        store=None,
        embedding_provider=None,
        settings=_settings(),
        repository=_Repository(with_documents=False),
    )
    empty_app = create_app(_settings())
    empty_app.dependency_overrides[get_rag_service] = lambda: empty_service
    not_found = _request(
        empty_app,
        "POST",
        "/api/rag/query",
        json={"question": "ADCON0 的低两位是什么？"},
    )
    assert not_found.status_code == 404


class _LLM:
    model_id = "llm-v1"

    def __init__(self) -> None:
        self.contexts: list[object] = []

    async def generate(self, _question: str, contexts: list[object]) -> SimpleNamespace:
        self.contexts = list(contexts)
        return SimpleNamespace(
            answer="The low bits select the clock source.",
            model_id=self.model_id,
            citations=[{"chunk_id": "attacker-controlled"}],
        )


class _Reranker:
    def __init__(self) -> None:
        self.candidates: list[object] = []

    async def rerank(
        self, _question: str, candidates: list[object], top_n: int
    ) -> list[SimpleNamespace]:
        self.candidates = list(candidates)
        return [
            SimpleNamespace(id=candidate.id, rerank_score=0.95) for candidate in candidates[:top_n]
        ]


@pytest.mark.parametrize(
    ("kind", "indexed_text", "uncertainties"),
    [
        (
            "VISUAL",
            "IRCCAL 的 3V/5V 百分比曲线，横轴 0–7D，纵轴 0–140%。",
            ["百分比的物理含义未在图中解释。"],
        ),
        ("TEXT", "正文 A\n\n[图：地址由 0 与 FSR0 组成]\n\n正文 B", []),
        (
            "TEXT",
            "EEPROM 读取示例\nMOVIA 00H\nMOVAR EEADRL\n;// EEPROM地址区间为0x0800-0x087F",
            [],
        ),
    ],
    ids=["figure-description", "inline-figure-order", "inline-code"],
)
def test_indexed_text_reaches_search_rerank_and_generation(
    kind: str, indexed_text: str, uncertainties: list[str]
) -> None:
    repository = _Repository(
        chunk_fields={
            "kind": kind,
            "text": "![图](snapshots/source.png)",
            "embedding_text": indexed_text,
            "chunk_metadata": {
                "figure_uncertainties": uncertainties,
                "review_log": "internal review details",
            },
        }
    )
    reranker = _Reranker()
    llm = _LLM()
    service = RagQueryService(
        store=_VectorStore(),
        embedding_provider=_EmbeddingProvider(),
        reranker=reranker,
        llm_provider=llm,
        settings=_settings(reranker_enabled=True, llm_enabled=True),
        repository=repository,
    )
    app = create_app(_settings())
    app.dependency_overrides[get_rag_service] = lambda: service
    question = {"question": "ADCON0 的低两位是什么？"}
    expected_text = indexed_text
    if uncertainties:
        expected_text += "\n不确定项：" + "；".join(uncertainties)

    search = _request(app, "GET", "/api/rag/search", params=question)
    assert search.status_code == 200, search.text
    assert search.json()[0]["text"] == expected_text

    query = _request(app, "POST", "/api/rag/query", json=question)
    assert query.status_code == 200, query.text
    assert reranker.candidates[0].text == expected_text
    assert llm.contexts[0].text == expected_text
    assert query.json()["citations"][0]["text"] == expected_text
    assert "snapshots/" not in llm.contexts[0].text
    assert "internal review details" not in llm.contexts[0].text


@pytest.mark.parametrize("indexed_text", [None, "", " \n"])
def test_legacy_chunk_without_indexed_text_keeps_its_original_text(indexed_text: object) -> None:
    service = RagQueryService(
        store=_VectorStore(),
        embedding_provider=_EmbeddingProvider(),
        settings=_settings(),
        repository=_Repository(chunk_fields={"embedding_text": indexed_text}),
    )
    app = create_app(_settings())
    app.dependency_overrides[get_rag_service] = lambda: service
    response = _request(
        app, "GET", "/api/rag/search", params={"question": "ADCON0 的低两位是什么？"}
    )
    assert response.status_code == 200, response.text
    assert response.json()[0]["text"] == "ADCON0 bit 1:0 select the clock source."


def test_generated_answer_cannot_supply_its_own_citations() -> None:
    llm = _LLM()
    service = RagQueryService(
        store=_VectorStore(),
        embedding_provider=_EmbeddingProvider(),
        llm_provider=llm,
        settings=_settings(llm_enabled=True),
        repository=_Repository(),
    )
    app = create_app(_settings())
    app.dependency_overrides[get_rag_service] = lambda: service

    response = _request(
        app,
        "POST",
        "/api/rag/query",
        json={"question": "ADCON0 的低两位是什么？"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["answer"] == "The low bits select the clock source."
    assert payload["model"] == "llm-v1"
    assert payload["citations"][0]["chunk_id"] == ("22222222-2222-2222-2222-222222222222")
    assert "attacker-controlled" not in response.text
    assert llm.contexts[0].source_tag.startswith(
        "[source:22222222-2222-2222-2222-222222222222 M8P6272/V1.10/"
    )


class _DocumentErrorService:
    async def delete(self, _document_id: UUID) -> None:
        raise DocumentConflictError("the current document cannot be deleted")

    async def status(self, _document_id: UUID) -> object:
        raise DocumentNotFoundError("document was not found")


def test_document_routes_map_not_found_and_current_conflict() -> None:
    app = create_app(_settings())
    app.dependency_overrides[get_document_service] = lambda: _DocumentErrorService()
    document_id = "33333333-3333-3333-3333-333333333333"

    status_response = _request(app, "GET", f"/api/rag/documents/{document_id}/status")
    delete_response = _request(app, "DELETE", f"/api/rag/documents/{document_id}")
    invalid_id = _request(app, "GET", "/api/rag/documents/not-a-uuid/status")

    assert status_response.status_code == 404
    assert delete_response.status_code == 409
    assert invalid_id.status_code == 422


class _UploadStorage:
    def __init__(self) -> None:
        self.upload: object | None = None

    async def save_upload(self, upload: object, staging_id: UUID) -> SimpleNamespace:
        self.upload = upload
        body = await upload.read()
        assert body.startswith(b"%PDF-")
        return SimpleNamespace(
            path=Path(f"C:/rag-staging/{staging_id}.pdf"),
            original_filename=upload.filename,
            sha256="a" * 64,
            size_bytes=len(body),
        )

    def cleanup_staging(self, _staging_id: UUID) -> None:
        return None


class _UploadService:
    def __init__(self, storage: _UploadStorage) -> None:
        self.storage = storage
        self.command: object | None = None

    async def submit(self, command: object) -> JobAccepted:
        assert self.storage.upload.file.closed is True
        assert isinstance(command.path, Path)
        assert not hasattr(command, "upload_file")
        self.command = command
        document_id = UUID("44444444-4444-4444-4444-444444444444")
        return JobAccepted(
            job_id=document_id,
            document_id=document_id,
            status="pending",
        )


def test_upload_closes_uploadfile_before_submitting_path_only_task() -> None:
    storage = _UploadStorage()
    service = _UploadService(storage)
    app = create_app(_settings())
    app.dependency_overrides[get_storage_manager] = lambda: storage
    app.dependency_overrides[get_document_service] = lambda: service

    response = _request(
        app,
        "POST",
        "/api/rag/documents",
        data={"product": " M8P6272 ", "version": "1.10", "language": "zh-CN"},
        files={"file": ("../../evil.pdf", b"%PDF-1.7\ncontent", "application/pdf")},
    )

    assert response.status_code == 202, response.text
    assert response.json()["job_id"] == response.json()["document_id"]
    assert service.command.original_filename == "evil.pdf"
    assert service.command.version == "V1.10"


def test_upload_rejects_non_pdf_extension_before_storage() -> None:
    storage = _UploadStorage()
    service = _UploadService(storage)
    app = create_app(_settings())
    app.dependency_overrides[get_storage_manager] = lambda: storage
    app.dependency_overrides[get_document_service] = lambda: service

    response = _request(
        app,
        "POST",
        "/api/rag/documents",
        data={"product": "M8P6272", "version": "1.10"},
        files={"file": ("manual.txt", b"%PDF-1.7", "application/pdf")},
    )

    assert response.status_code == 422
    assert storage.upload is None


class _CandidateRepository:
    def __init__(self) -> None:
        self.document_id = UUID("55555555-5555-5555-5555-555555555555")
        self.candidate: dict[str, object] | None = None
        self.created = 0

    async def find_identical(
        self, product: str, version: str, file_sha256: str
    ) -> dict[str, object] | None:
        if self.candidate is None:
            return None
        if (
            self.candidate["product"],
            self.candidate["version"],
            self.candidate["file_sha256"],
        ) == (product, version, file_sha256):
            return self.candidate
        return None

    async def create_candidate(self, data: object) -> dict[str, object]:
        self.created += 1
        self.candidate = {
            "id": self.document_id,
            "product": data.product,
            "version": data.version,
            "file_sha256": data.file_sha256,
            "status": "pending",
        }
        return self.candidate


class _PathRunner:
    def __init__(self) -> None:
        self.submissions: list[tuple[UUID, Path]] = []

    async def submit_path(self, document_id: UUID, path: Path) -> UUID:
        self.submissions.append((document_id, path))
        return document_id


class _CleanupStorage:
    def __init__(self, calls: list[object] | None = None) -> None:
        self.cleaned: list[object] = []
        self.calls = calls

    def cleanup_staging(self, staging_id: object) -> None:
        self.cleaned.append(staging_id)

    def delete_document_assets(self, document_id: UUID) -> str:
        if self.calls is not None:
            self.calls.append(("file_delete", document_id))
        return "trash-token"

    def restore_document_assets(self, document_id: UUID, token: object) -> None:
        if self.calls is not None:
            self.calls.append(("file_restore", document_id, token))


def test_document_submission_is_idempotent_and_background_receives_only_path() -> None:
    repository = _CandidateRepository()
    runner = _PathRunner()
    storage = _CleanupStorage()
    service = DocumentService(
        repository=repository,
        storage=storage,
        task_runner=runner,
    )
    staging_a = UUID("66666666-6666-6666-6666-666666666666")
    staging_b = UUID("77777777-7777-7777-7777-777777777777")

    def command(staging_id: UUID) -> DocumentSubmitCommand:
        return DocumentSubmitCommand(
            path=Path(f"C:/staging/{staging_id}/source.pdf"),
            original_filename="manual.pdf",
            sha256="b" * 64,
            size_bytes=10,
            product="M8P6272",
            version="1.10",
            staging_id=staging_id,
        )

    first = asyncio.run(service.submit(command(staging_a)))
    duplicate = asyncio.run(service.submit(command(staging_b)))

    assert first.document_id == duplicate.document_id == repository.document_id
    assert repository.created == 1
    assert runner.submissions == [
        (repository.document_id, Path(f"C:/staging/{staging_a}/source.pdf"))
    ]
    assert storage.cleaned == [staging_b]


class _DeleteRepository:
    def __init__(self, calls: list[object], *, current: bool = False, fail_pg: bool = False):
        self.calls = calls
        self.current = current
        self.fail_pg = fail_pg

    async def get_document(self, document_id: UUID, *, for_update: bool) -> dict[str, object]:
        assert for_update is True
        return {"id": document_id, "status": "superseded", "current": self.current}

    async def get_vector_records_for_document(self, document_id: UUID) -> list[str]:
        return [f"snapshot:{document_id}"]

    async def delete_document_row(self, document_id: UUID) -> None:
        self.calls.append(("pg_delete", document_id))
        if self.fail_pg:
            raise RuntimeError("injected PG failure")


class _DeleteStore:
    def __init__(self, calls: list[object]) -> None:
        self.calls = calls

    async def delete_document(self, document_id: UUID) -> None:
        self.calls.append(("vector_delete", document_id))

    async def upsert(self, snapshot: object) -> None:
        self.calls.append(("vector_restore", snapshot))


def test_document_delete_order_conflict_and_reverse_compensation() -> None:
    document_id = UUID("88888888-8888-8888-8888-888888888888")
    calls: list[object] = []
    service = DocumentService(
        repository=_DeleteRepository(calls),
        store=_DeleteStore(calls),
        storage=_CleanupStorage(calls),
    )
    asyncio.run(service.delete(document_id))
    assert calls == [
        ("vector_delete", document_id),
        ("file_delete", document_id),
        ("pg_delete", document_id),
    ]

    conflict_calls: list[object] = []
    conflict = DocumentService(
        repository=_DeleteRepository(conflict_calls, current=True),
        store=_DeleteStore(conflict_calls),
        storage=_CleanupStorage(conflict_calls),
    )
    try:
        asyncio.run(conflict.delete(document_id))
    except DocumentConflictError:
        pass
    else:
        raise AssertionError("current document deletion must conflict")
    assert conflict_calls == []

    compensated_calls: list[object] = []
    compensated = DocumentService(
        repository=_DeleteRepository(compensated_calls, fail_pg=True),
        store=_DeleteStore(compensated_calls),
        storage=_CleanupStorage(compensated_calls),
    )
    try:
        asyncio.run(compensated.delete(document_id))
    except DocumentDeleteError as exc:
        assert "compensated" in str(exc)
    else:
        raise AssertionError("injected PG failure must be surfaced")
    assert compensated_calls == [
        ("vector_delete", document_id),
        ("file_delete", document_id),
        ("pg_delete", document_id),
        ("file_restore", document_id, "trash-token"),
        ("vector_restore", [f"snapshot:{document_id}"]),
    ]


class _Planner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object, object]] = []

    async def plan(
        self, question: str, product: str | None = None, version: str | None = None
    ) -> QueryPlan:
        self.calls.append((question, product, version))
        return QueryPlan(
            intent="lookup",
            product="M8P6272",
            version="V1.10",
            strategy="dense",
            needs_clarification=False,
            rewritten_query=question,
            confidence=0.9,
        )


def test_planner_completes_missing_scope_before_retrieval() -> None:
    planner = _Planner()
    repository = _Repository()
    service = RagQueryService(
        store=_VectorStore(),
        embedding_provider=_EmbeddingProvider(),
        settings=_settings(planner_enabled=True),
        repository=repository,
        planner=planner,
    )
    app = create_app(_settings())
    app.dependency_overrides[get_rag_service] = lambda: service

    response = _request(
        app,
        "POST",
        "/api/rag/query",
        json={"question": "ADCON0 的低两位是什么？"},
    )

    assert response.status_code == 200, response.text
    assert planner.calls == [("ADCON0 的低两位是什么？", None, None)]
    assert repository.scope.product == "M8P6272"
    assert repository.scope.version == "V1.10"
    assert "planner_scope_completed" in response.json()["degraded_reason"]


def test_bm25_route_adds_lexical_candidate_before_selection() -> None:
    repository = _Repository()
    sparse_chunk_id = UUID("00000000-0000-0000-0000-000000000001")
    repository.chunks_for_sparse = [
        {
            "id": repository.chunk_id,
            "document_id": repository.document_id,
            "embedding_id": "embedding-1",
            "section_path": "2 > 2.6",
            "page_start": 15,
            "page_end": 15,
            "kind": "TABLE",
            "embedding_text": "ADCON0 bit 1:0 select the clock source.",
            "text": "ADCON0 bit 1:0 select the clock source.",
            "chunk_metadata": {},
        },
        {
            "id": sparse_chunk_id,
            "document_id": repository.document_id,
            "embedding_id": "embedding-2",
            "section_path": "2 > 2.6.1",
            "page_start": 16,
            "page_end": 16,
            "kind": "TEXT",
            "embedding_text": "ADCON0 的低两位 ADON 和 GO/DONE 决定转换启停。",
            "text": "ADCON0 的低两位 ADON 和 GO/DONE 决定转换启停。",
            "chunk_metadata": {},
        },
    ]
    service = RagQueryService(
        store=_VectorStore(),
        embedding_provider=_EmbeddingProvider(),
        settings=_settings(bm25_enabled=True),
        repository=repository,
    )
    app = create_app(_settings())
    app.dependency_overrides[get_rag_service] = lambda: service

    response = _request(
        app,
        "GET",
        "/api/rag/search",
        params={"question": "ADCON0 的低两位是什么？", "top_k": 8},
    )

    assert response.status_code == 200, response.text
    ids = [hit["chunk_id"] for hit in response.json()]
    assert str(sparse_chunk_id) in ids
    assert str(sparse_chunk_id) in ids[:2]


def test_neighbor_window_adds_adjacent_chunk_to_context() -> None:
    repository = _Repository()
    neighbor_id = UUID("33333333-3333-3333-3333-333333333333")
    repository.neighbor_pairs = [
        (
            {
                "id": neighbor_id,
                "document_id": repository.document_id,
                "section_path": "2 > 2.6.1",
                "page_start": 16,
                "page_end": 16,
                "kind": "TEXT",
                "embedding_text": "相邻块：ADCON0 的低两位在下一节继续说明。",
                "text": "相邻块：ADCON0 的低两位在下一节继续说明。",
                "chunk_metadata": {},
            },
            {
                "id": repository.document_id,
                "product": "M8P6272",
                "version": "V1.10",
            },
        )
    ]
    service = RagQueryService(
        store=_VectorStore(),
        embedding_provider=_EmbeddingProvider(),
        settings=_settings(context_neighbor_window=1),
        repository=repository,
    )
    app = create_app(_settings())
    app.dependency_overrides[get_rag_service] = lambda: service

    response = _request(
        app,
        "POST",
        "/api/rag/query",
        json={"question": "ADCON0 的低两位是什么？"},
    )

    assert response.status_code == 200, response.text
    citation_ids = [citation["chunk_id"] for citation in response.json()["citations"]]
    assert str(neighbor_id) in citation_ids
    assert citation_ids[0] == str(repository.chunk_id)


def test_query_trace_records_success_failure_and_disabled_mode(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    settings = _settings(
        llm_enabled=True,
        reranker_enabled=False,
        query_trace_enabled=True,
        query_trace_path=trace_path,
    )
    service = RagQueryService(
        store=_VectorStore(),
        embedding_provider=_EmbeddingProvider(),
        llm_provider=_LLM(),
        settings=settings,
        repository=_Repository(),
    )
    response = asyncio.run(
        service.query(
            QueryRequest(
                question="ADCON0 的低两位是什么？",
                product="M8P6272",
                version="1.10",
            )
        )
    )
    assert response.answer
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    stages = [stage["name"] for stage in rows[-1]["stages"]]
    assert stages == ["plan", "retrieve", "select", "generate", "checks"]
    generate = next(stage for stage in rows[-1]["stages"] if stage["name"] == "generate")
    assert generate["attempted"] is True
    assert generate["success"] is True

    failing = RagQueryService(
        store=_VectorStore(),
        embedding_provider=_EmbeddingProvider(),
        settings=settings,
        repository=_Repository(with_documents=False),
    )
    with pytest.raises(RagNotFoundError):
        asyncio.run(
            failing.query(
                QueryRequest(
                    question="ADCON0 的低两位是什么？",
                    product="M8P6272",
                    version="1.10",
                )
            )
        )
    failed_rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    failed_stages = [stage["name"] for stage in failed_rows[-1]["stages"]]
    assert "retrieve" in failed_stages
    assert "error" in failed_stages

    disabled_path = tmp_path / "disabled.jsonl"
    disabled = RagQueryService(
        store=_VectorStore(),
        embedding_provider=_EmbeddingProvider(),
        settings=_settings(query_trace_path=disabled_path),
        repository=_Repository(with_documents=False),
    )
    with pytest.raises(RagNotFoundError):
        asyncio.run(
            disabled.query(
                QueryRequest(
                    question="ADCON0 的低两位是什么？",
                    product="M8P6272",
                    version="1.10",
                )
            )
        )
    assert not disabled_path.exists()
