"""Асинхронный движок, сессии и лёгкая миграция схемы.

`Database` инкапсулирует создание async-движка, фабрику сессий и подготовку
схемы при старте приложения.

Подготовка схемы состоит из двух шагов: `create_all` создаёт отсутствующие
таблицы, а `migrate_schema` добавляет в существующие таблицы колонки, которых
в них ещё нет. Второй шаг нужен потому, что `create_all` колонки не
добавляет: без него бот не стартует на непустой базе, созданной прошлой
версией кода (`no such column: generations.pages_count`).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.models import Base, Generation
from app.utils.logger import get_logger

logger = get_logger()


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


def _existing_columns(connection: Any, table_name: str) -> set[str]:
    """Имена колонок, которые уже есть в таблице."""

    inspector = sa.inspect(connection)
    if table_name not in inspector.get_table_names():
        return set()
    return {column["name"] for column in inspector.get_columns(table_name)}


def _add_missing_columns(sync_connection: Any) -> list[str]:
    """Добавить в `generations` колонки модели, которых в таблице нет.

    Список берётся из метаданных модели, а не из вручную заданного
    перечня: новая колонка в модели обязана доезжать до старой базы сама.
    Первичный ключ пропускается — `ALTER TABLE` его добавить не умеет.
    Ошибка по отдельной колонке не должна ронять остальные и не должна
    мешать боту стартовать: причина попадает в лог, а несостоявшаяся
    колонка проявится позже понятной ошибкой чтения.
    """

    table = Generation.__table__
    existing = _existing_columns(sync_connection, table.name)
    if not existing:
        # Таблицы ещё нет: `create_all` создаст её сразу со всеми
        # колонками, миграция не нужна.
        return []

    added: list[str] = []
    for column in table.columns:
        if column.name in existing or column.primary_key:
            continue
        ddl_type = column.type.compile(sync_connection.dialect)
        statement = (
            f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {ddl_type}'
        )
        try:
            sync_connection.execute(sa.text(statement))
        except SQLAlchemyError as exc:
            logger.warning(
                "Не удалось добавить колонку {}.{}: {}",
                table.name,
                column.name,
                exc,
            )
            continue
        added.append(column.name)
        logger.info(
            "Миграция: в {} добавлена колонка {}", table.name, column.name
        )
    return added


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
        """Подготовить схему: создать таблицы и добавить недостающие колонки.

        Миграция идёт вторым шагом, потому что `create_all` на существующей
        таблице молча ничего не делает, и без неё старая база осталась бы
        без новых колонок.
        """

        await self._create_all()
        await self.migrate_schema()

    async def _create_all(self) -> None:
        """Создать отсутствующие таблицы по метаданным моделей."""

        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def migrate_schema(self) -> tuple[str, ...]:
        """Добавить в существующие таблицы колонки, которых в них нет.

        Возвращает имена добавленных колонок: повторный запуск возвращает
        пустой кортеж, поэтому вызывающий может отличить миграцию от
        бездействия, а тест — проверить идемпотентность.

        Сбой самой миграции (заблокированная база, битая страница файла)
        не поднимается наружу: лучше старт с понятной ошибкой в логе, чем
        бот, который не поднялся из-за недобавленной диагностической
        колонки. Причина попадает в лог уровнем ERROR.
        """

        try:
            async with self.engine.begin() as connection:
                added = await connection.run_sync(_add_missing_columns)
        except SQLAlchemyError as exc:
            logger.error(
                "Миграция схемы не выполнена, база осталась как есть: {}",
                exc,
            )
            return ()
        return tuple(added)

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
