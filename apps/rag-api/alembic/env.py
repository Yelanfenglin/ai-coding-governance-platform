"""Alembic environment using SQLAlchemy's psycopg async dialect."""

from __future__ import annotations

from logging.config import fileConfig

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import pool
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from alembic import context
from app.db.models import Base


class _MigrationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    database_url: SecretStr

    @model_validator(mode="after")
    def validate_target(self) -> _MigrationSettings:
        try:
            url = make_url(self.database_url.get_secret_value())
        except (ArgumentError, TypeError, ValueError):
            raise ValueError("RAG_DATABASE_URL is not a valid SQLAlchemy URL") from None
        if url.drivername != "postgresql+psycopg" or url.database != "rag_kb":
            raise ValueError(
                "Alembic requires postgresql+psycopg and the isolated rag_kb database"
            )
        return self


# Alembic executes this file outside the normal module registry.
_MigrationSettings.model_rebuild(_types_namespace=globals())

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    settings = _MigrationSettings()  # type: ignore[call-arg]
    return settings.database_url.get_secret_value()


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_sync_migrations(connection: object) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable: AsyncEngine = create_async_engine(
        _database_url(),
        poolclass=pool.NullPool,
        pool_pre_ping=True,
    )
    try:
        async with connectable.connect() as connection:
            await connection.run_sync(_run_sync_migrations)
    finally:
        await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    import asyncio

    asyncio.run(run_migrations_online())
