"""Async PostgreSQL helpers."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.sql import text

from src.config import settings
from src.models.database import Base


def get_engine() -> AsyncEngine:
    """Return a new async engine instance.

    Using a fresh engine per call avoids reusing asyncpg connections across
    multiple pytest event loops, which can lead to "event loop is closed" and
    "operation is in progress" errors on Windows.
    """

    return create_async_engine(settings.DATABASE_URI, echo=False, future=True)


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return a new session factory bound to a fresh engine."""

    return async_sessionmaker(get_engine(), expire_on_commit=False)


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield an AsyncSession bound to a short-lived engine.

    Each context manager call creates its own engine + session factory,
    ensuring isolation between tests and event loops.
    """

    session_factory = get_session_factory()
    session = session_factory()
    try:
        yield session
    finally:
        await session.close()


async def init_database() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def check_postgres_health() -> bool:
    try:
        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
