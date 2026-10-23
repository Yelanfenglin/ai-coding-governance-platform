"""Bounded ingest task runners.

P0 uses one in-process consumer for resource isolation, while every externally
observable state transition is persisted through the ``documents`` repository.
The queue and local de-duplication set are scheduling aids only; they are never
reported as durable job state.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Protocol
from uuid import UUID


class TaskRunner(Protocol):
    """Lifecycle contract shared by the application lifespan and submit service."""

    async def submit(self, document_id: UUID) -> UUID: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class TaskRepository(Protocol):
    """The durable subset of ``RagRepository`` needed by the P0 runner."""

    async def set_status(
        self,
        document_id: UUID,
        status: object,
        *,
        error: str | None = None,
    ) -> None: ...


TaskProcessor = Callable[[UUID], Awaitable[object | None]]


class InProcessTaskRunner:
    """Run registered ingest jobs with exactly one in-process consumer.

    ``submit`` is intended for a document row that the shared registration stage
    has already created.  ``job_id`` is therefore the same UUID as
    ``document_id``.  The processor should execute the shared ``IngestPipeline``
    (normally via a service that reconstructs its command from the registered
    row); this class contains no parsing, version, transaction, or compensation
    logic.

    P0 cannot recover its in-memory queue after process death.  The persisted
    pending/parsing/failed/parsed values remain authoritative and make such jobs
    observable; automatic claiming/recovery belongs to the disabled P2 runner.
    """

    _STOP = object()

    def __init__(self, *, repository: TaskRepository, processor: TaskProcessor) -> None:
        self._repository = repository
        self._processor = processor
        self._queue: asyncio.Queue[UUID | object] = asyncio.Queue()
        self._scheduled: set[UUID] = set()
        self._schedule_lock = asyncio.Lock()
        self._worker: asyncio.Task[None] | None = None
        self._closed = False

    async def submit(self, document_id: UUID) -> UUID:
        """Persist pending and enqueue once; return ``job_id=document_id``."""

        if not isinstance(document_id, UUID):
            raise TypeError("document_id must be a UUID")

        async with self._schedule_lock:
            if self._closed:
                raise RuntimeError("task runner is stopped")
            if document_id in self._scheduled:
                return document_id
            await self._set_status(document_id, "pending")
            self._scheduled.add(document_id)
            await self._queue.put(document_id)
        return document_id

    async def start(self) -> None:
        """Start the sole consumer; repeated calls are idempotent."""

        if self._closed:
            raise RuntimeError("task runner is stopped")
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run(), name="rag-ingest-worker-0")

    async def stop(self) -> None:
        """Stop accepting submissions and drain already accepted work."""

        async with self._schedule_lock:
            if self._closed:
                worker = self._worker
            else:
                self._closed = True
                if self._worker is None and self._scheduled:
                    self._worker = asyncio.create_task(
                        self._run(),
                        name="rag-ingest-worker-0",
                    )
                worker = self._worker
                if worker is not None and not worker.done():
                    await self._queue.put(self._STOP)

        if worker is not None:
            await worker

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is self._STOP:
                    return
                document_id = item
                if not isinstance(document_id, UUID):
                    continue
                await self._execute(document_id)
            finally:
                self._queue.task_done()

    async def _execute(self, document_id: UUID) -> None:
        try:
            await self._set_status(document_id, "parsing")
            result = await self._processor(document_id)
            final_status = _status_text(getattr(result, "status", None)) or "parsed"
            if final_status not in {"parsed", "failed", "superseded"}:
                final_status = "parsed"
            error = "ingest task reported failure" if final_status == "failed" else None
            await self._set_status(document_id, final_status, error=error)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - persist failure and keep the worker alive
            await self._set_status(
                document_id,
                "failed",
                error=f"{type(exc).__name__}: ingest task failed",
            )
        finally:
            self._scheduled.discard(document_id)

    async def _set_status(
        self,
        document_id: UUID,
        status: str,
        *,
        error: str | None = None,
    ) -> None:
        persisted_status = _document_status(status)
        if error is None:
            await self._repository.set_status(document_id, persisted_status)
        else:
            await self._repository.set_status(document_id, persisted_status, error=error)


class P2TaskRunnerDisabledError(RuntimeError):
    """Raised when the intentionally disabled P2 runner is selected."""


class RecoverablePostgresTaskRunner:
    """P2 target interface -- intentionally disabled in this phase.

    The enabled implementation must claim a ``pending`` document inside a short
    PostgreSQL transaction using ``SELECT ... FOR UPDATE SKIP LOCKED`` (or an
    equivalent atomic update), change it to ``parsing``, and then release the
    transaction before expensive work.  Startup must recover ``parsing`` rows
    whose lease/updated timestamp is older than the configured timeout.  The
    claim and database constraints must guarantee one executor per document.

    No process-local dictionary may represent durable status.  If attempt,
    lease, or history records become requirements, P2 must add the reviewed
    ``ingest_jobs`` migration rather than hide them here.
    """

    enabled = False
    disabled_reason = "P2 recoverable PostgreSQL task runner is not enabled"

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    async def submit(self, document_id: UUID) -> UUID:
        del document_id
        raise P2TaskRunnerDisabledError(self.disabled_reason)

    async def start(self) -> None:
        raise P2TaskRunnerDisabledError(self.disabled_reason)

    async def stop(self) -> None:
        # A no-op keeps application shutdown safe if construction succeeded but
        # startup was rejected because P2 is disabled.
        return None


def _document_status(value: str) -> object:
    """Return the persisted wire value without importing the ORM model.

    ``DocumentStatus`` is a string enum, and the repository is the conversion
    boundary.  Avoiding an ORM import keeps this scheduler usable in isolated
    unit tests and prevents task startup from constructing unrelated DB models.
    """

    return value


def _status_text(value: object) -> str | None:
    if value is None:
        return None
    raw = value.value if isinstance(value, Enum) else value
    return str(raw).lower()


__all__ = [
    "InProcessTaskRunner",
    "P2TaskRunnerDisabledError",
    "RecoverablePostgresTaskRunner",
    "TaskRunner",
]
