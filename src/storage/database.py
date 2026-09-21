from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.storage.models import Base
from src.storage.schema import add_missing_columns


def database_kind(url: str) -> str:
    lowered = url.lower()
    if "postgres" in lowered:
        return "PostgreSQL"
    if "sqlite" in lowered:
        return "SQLite"
    return "unknown"


def async_database_url(url: str) -> str:
    if url.startswith("sqlite+aiosqlite://"):
        return url
    if url.startswith("sqlite:///"):
        return "sqlite+aiosqlite:///" + url.removeprefix("sqlite:///")
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + url.removeprefix("postgresql://")
    if url.startswith("postgres://"):
        return "postgresql+asyncpg://" + url.removeprefix("postgres://")
    return url


def _sqlite_on_connect(dbapi_connection: object, _connection_record: object) -> None:
    cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class Database:
    def __init__(self, url: str) -> None:
        self.url = async_database_url(url)
        connect_args: dict[str, object] = {}
        if self.url.startswith("sqlite+"):
            connect_args["timeout"] = 30
        self.engine: AsyncEngine = create_async_engine(
            self.url,
            echo=False,
            connect_args=connect_args,
        )
        if self.url.startswith("sqlite+"):
            event.listen(self.engine.sync_engine, "connect", _sqlite_on_connect)
        self.session_factory = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )

    async def create_all(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.run_sync(add_missing_columns)

    async def drop_all(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def aclose(self) -> None:
        await self.engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as session:
            yield session
