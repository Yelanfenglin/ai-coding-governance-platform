"""Async SQLAlchemy engine and session lifecycle helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.settings import Settings


def create_async_engine_from(settings: Settings) -> AsyncEngine:
    """Build an async engine without opening a connection."""

    return create_async_engine(
        settings.database_url.get_secret_value(),
        pool_pre_ping=True,
    )


def create_session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Yield a session; callers explicitly choose their transaction boundary."""

    async with factory() as session:
        try:
            yield session
        except BaseException:
            await session.rollback()
            raise
