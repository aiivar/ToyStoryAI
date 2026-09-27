"""Асинхронный движок и сессии SQLAlchemy.

`Database` инкапсулирует создание async-движка, фабрику сессий и создание
схемы при старте приложения.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.models import Base


def _ensure_sqlite_parent_dir(url: str) -> None:
    """Создать каталог файла SQLite, если его ещё нет."""

    prefix = "sqlite+aiosqlite:///"
    if not url.startswith(prefix):
        return

    raw_path = url[len(prefix) :]
    if not raw_path or raw_path == ":memory:":
        return

    parent = Path(raw_path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


class Database:
    """Владелец async-движка и фабрики сессий."""

    def __init__(self, url: str, **engine_kwargs: Any) -> None:
        _ensure_sqlite_parent_dir(url)
        self.url = url
        self.engine: AsyncEngine = create_async_engine(
            url,
            future=True,
            **engine_kwargs,
        )
        self._session_factory = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )

    async def create_schema(self) -> None:
        """Создать отсутствующие таблицы по метаданным моделей."""

        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    def session(self) -> AsyncSession:
        """Открыть новую сессию (вызывающий закрывает её сам)."""

        return self._session_factory()

    @asynccontextmanager
    async def session_scope(self) -> AsyncIterator[AsyncSession]:
        """Сессия с автоматическим закрытием."""

        session = self._session_factory()
        try:
            yield session
        finally:
            await session.close()

    async def dispose(self) -> None:
        """Закрыть пул соединений движка."""

        await self.engine.dispose()


__all__ = ["Database"]
