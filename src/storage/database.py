from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.storage.models import Base
from src.storage.schema import add_missing_columns

_ASYNCPG_UNSUPPORTED_QUERY_KEYS = frozenset(
    {
        "sslmode",
        "ssl",
        "channel_binding",
        "gssencmode",
        "target_session_attrs",
    }
)
_SSL_REQUIRED_MODES = frozenset({"require", "verify-ca", "verify-full"})


def database_kind(url: str) -> str:
    lowered = url.lower()
    if "postgres" in lowered:
        return "PostgreSQL"
    if "sqlite" in lowered:
        return "SQLite"
    return "unknown"


def postgres_requires_ssl(url: str) -> bool:
    query = dict(parse_qsl(urlsplit(url).query, keep_blank_values=True))
    sslmode = query.get("sslmode", "").lower()
    if sslmode in _SSL_REQUIRED_MODES:
        return True
    if sslmode in {"disable", "allow", "prefer"}:
        return False
    host = (urlsplit(url).hostname or "").lower()
    return host.endswith(".rlwy.net") or host.endswith(".railway.app")


def async_database_url(url: str) -> str:
    if url.startswith("sqlite+aiosqlite://"):
        return url
    if url.startswith("sqlite:///"):
        return "sqlite+aiosqlite:///" + url.removeprefix("sqlite:///")
    if url.startswith("postgresql+asyncpg://"):
        normalized = url
    elif url.startswith("postgresql://"):
        normalized = "postgresql+asyncpg://" + url.removeprefix("postgresql://")
    elif url.startswith("postgres://"):
        normalized = "postgresql+asyncpg://" + url.removeprefix("postgres://")
    else:
        return url
    parts = urlsplit(normalized)
    kept = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in _ASYNCPG_UNSUPPORTED_QUERY_KEYS
    ]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(kept), parts.fragment))


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
        elif self.url.startswith("postgresql+") and postgres_requires_ssl(url):
            connect_args["ssl"] = True
        engine_kwargs: dict[str, object] = {
            "echo": False,
            "connect_args": connect_args,
        }
        if self.url.startswith("postgresql+"):
            engine_kwargs.update(
                {
                    "pool_size": 10,
                    "max_overflow": 20,
                    "pool_pre_ping": True,
                    "pool_timeout": 10,
                    "pool_recycle": 300,
                }
            )
        self.engine: AsyncEngine = create_async_engine(self.url, **engine_kwargs)
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
