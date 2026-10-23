"""Safe PostgreSQL target, health, and migration checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine


class DatabaseTargetError(RuntimeError):
    """Raised when the connected database is not the isolated RAG database."""


class MigrationStateError(RuntimeError):
    """Raised when the database schema is not at the Alembic head."""


@dataclass(frozen=True, slots=True)
class DependencyHealth:
    name: str
    ok: bool
    detail: str | None = None
    latency_ms: float | None = None

    @property
    def healthy(self) -> bool:
        return self.ok


async def _current_database(connection: AsyncConnection) -> str | None:
    return await connection.scalar(text("SELECT current_database()"))


async def assert_database_target(engine: AsyncEngine, expected: str = "rag_kb") -> None:
    """Verify the actual PostgreSQL database before any application writes."""

    try:
        async with engine.connect() as connection:
            actual = await _current_database(connection)
    except Exception:
        raise DatabaseTargetError("unable to verify the PostgreSQL target database") from None

    if actual != expected:
        raise DatabaseTargetError(
            f"refusing database target {actual!r}; expected isolated database {expected!r}"
        )


async def database_health(engine: AsyncEngine) -> DependencyHealth:
    """Return a sanitized, non-raising database health result."""

    started = perf_counter()
    try:
        async with engine.connect() as connection:
            actual = await _current_database(connection)
        elapsed_ms = (perf_counter() - started) * 1000
        if actual != "rag_kb":
            return DependencyHealth(
                name="postgresql",
                ok=False,
                detail="connected to an unexpected database",
                latency_ms=elapsed_ms,
            )
        return DependencyHealth(
            name="postgresql",
            ok=True,
            detail="rag_kb is reachable",
            latency_ms=elapsed_ms,
        )
    except Exception as exc:
        return DependencyHealth(
            name="postgresql",
            ok=False,
            detail=f"database unavailable ({type(exc).__name__})",
            latency_ms=(perf_counter() - started) * 1000,
        )


def _migration_heads(sync_connection: object) -> tuple[str, ...]:
    context = MigrationContext.configure(sync_connection)  # type: ignore[arg-type]
    return tuple(context.get_current_heads())


async def assert_migration_head(engine: AsyncEngine) -> None:
    """Require all configured Alembic heads to be applied."""

    config_path = Path(__file__).resolve().parents[2] / "alembic.ini"
    try:
        config = Config(str(config_path))
        expected_heads = set(ScriptDirectory.from_config(config).get_heads())
        async with engine.connect() as connection:
            current_heads = set(await connection.run_sync(_migration_heads))
    except Exception:
        raise MigrationStateError("unable to determine the database migration state") from None

    if current_heads != expected_heads:
        current = ", ".join(sorted(current_heads)) or "<none>"
        expected = ", ".join(sorted(expected_heads)) or "<none>"
        raise MigrationStateError(
            f"database migration is not at head (current: {current}; expected: {expected})"
        )
