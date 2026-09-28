"""Тесты лёгкой миграции схемы `generations` (задача 6.1).

`Base.metadata.create_all` создаёт отсутствующие таблицы, но не добавляет
колонки в уже существующие. Тесты строят файл базы со старой схемой, копируют
его и проверяют, что миграция добавляет новые колонки, не теряя старые строки
и оставаясь идемпотентной.
"""

from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import sqlalchemy as sa

from app.db.engine import Database
from app.db.models import Generation
from app.db.repository import GenerationRepository

TMP_ROOT = Path(".tmp/tests")

# DDL старой схемы: ровно те колонки, что были в `bot.db` до постраничной
# книжки (`pages_count` и `output_dir` в ней ещё не было).
OLD_SCHEMA = """
CREATE TABLE generations (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    user_id BIGINT NOT NULL,
    created_at DATETIME NOT NULL,
    type VARCHAR(16) NOT NULL,
    status VARCHAR(16) NOT NULL,
    tale_text TEXT,
    video_url VARCHAR(2048),
    local_path VARCHAR(1024),
    kling_request_id VARCHAR(255),
    error TEXT
)
"""

OLD_ROW = {
    "user_id": 4242,
    "created_at": datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    "type": "tale",
    "status": "success",
    "tale_text": "Жила-была рыжая игрушка. " * 8,
    "local_path": "uploads/old.jpg",
}

# Колонки, которых в старой схеме не было.
NEW_COLUMNS = ("output_dir", "pages_count")


def make_old_database() -> Path:
    """Создать файл базы со старой схемой и одной старой успешной сказкой."""

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TMP_ROOT / f"old-schema-{uuid4().hex}.db"
    connection = sqlite3.connect(path)
    try:
        connection.execute(OLD_SCHEMA)
        connection.execute(
            "INSERT INTO generations "
            "(user_id, created_at, type, status, tale_text, local_path) "
            "VALUES (:user_id, :created_at, :type, :status, :tale_text, "
            ":local_path)",
            OLD_ROW,
        )
        connection.commit()
    finally:
        connection.close()
    return path


def copy_old_database(name: str) -> tuple[Path, Path]:
    """Вернуть оригинал старой базы и её копию под заданным именем."""

    original = make_old_database()
    copy_path = TMP_ROOT / f"{name}-{uuid4().hex}.db"
    shutil.copy2(original, copy_path)
    return original, copy_path


def column_names(db_path: Path) -> set[str]:
    """Имена колонок таблицы `generations` в файле базы."""

    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute("PRAGMA table_info(generations)").fetchall()
    finally:
        connection.close()
    return {str(row[1]) for row in rows}


async def test_migration_adds_new_columns_to_old_database() -> None:
    """На копии старой базы миграция добавляет `pages_count` и `output_dir`."""

    original, copy_path = copy_old_database("old-copy")
    db = Database(f"sqlite+aiosqlite:///{copy_path}")
    try:
        added = await db.migrate_schema()
        names = column_names(copy_path)
    finally:
        await db.dispose()
        original.unlink(missing_ok=True)
        copy_path.unlink(missing_ok=True)

    assert sorted(added) == list(NEW_COLUMNS)
    assert set(NEW_COLUMNS) <= names
    # Старые колонки на месте: миграция ничего не пересоздаёт.
    assert {"id", "video_url", "kling_request_id", "error"} <= names


async def test_old_rows_stay_readable_after_migration() -> None:
    """Старые строки читаются после миграции и не теряют содержимое."""

    original, copy_path = copy_old_database("old-rows")
    db = Database(f"sqlite+aiosqlite:///{copy_path}")
    try:
        await db.create_schema()
        repository = GenerationRepository(db)
        history = await repository.get_history(OLD_ROW["user_id"])
    finally:
        await db.dispose()
        original.unlink(missing_ok=True)
        copy_path.unlink(missing_ok=True)

    assert len(history) == 1
    record = history[0]
    assert record.tale_text == OLD_ROW["tale_text"]
    assert record.local_path == "uploads/old.jpg"
    # Новые колонки у старой строки пустые, а не потерянные.
    assert record.pages_count is None
    assert record.output_dir is None


async def test_migration_is_idempotent() -> None:
    """Повторный запуск миграции ничего не меняет и не падает."""

    original, copy_path = copy_old_database("idempotent")
    db = Database(f"sqlite+aiosqlite:///{copy_path}")
    try:
        first = await db.migrate_schema()
        names_after_first = column_names(copy_path)
        second = await db.migrate_schema()
        third = await db.migrate_schema()
        names_after_third = column_names(copy_path)
    finally:
        await db.dispose()
        original.unlink(missing_ok=True)
        copy_path.unlink(missing_ok=True)

    assert sorted(first) == list(NEW_COLUMNS)
    assert second == ()
    assert third == ()
    assert names_after_third == names_after_first


async def test_migration_does_nothing_on_new_database() -> None:
    """На новой базе `create_all` уже создал все колонки — миграция пуста."""

    db_path = TMP_ROOT / f"fresh-{uuid4().hex}.db"
    db = Database(f"sqlite+aiosqlite:///{db_path}")
    try:
        await db.create_schema()
        added = await db.migrate_schema()
    finally:
        await db.dispose()
        db_path.unlink(missing_ok=True)

    assert added == ()
    model_columns = {column.name for column in Generation.__table__.columns}
    assert model_columns >= set(NEW_COLUMNS)


async def test_failed_alter_does_not_break_startup() -> None:
    """Неудачный `ALTER TABLE` не поднимается наружу: старт бота не ломается."""

    original, copy_path = copy_old_database("guarded")
    copy_path.chmod(0o444)
    db = Database(f"sqlite+aiosqlite:///{copy_path}")
    try:
        added = await db.migrate_schema()
    finally:
        await db.dispose()
        copy_path.chmod(0o644)
        original.unlink(missing_ok=True)
        copy_path.unlink(missing_ok=True)

    assert added == ()


async def test_unopenable_database_does_not_break_startup() -> None:
    """Совсем непроходимая база не превращается в падение на старте."""

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    # Каталог вместо файла базы: SQLite такой путь не открывает.
    as_directory = TMP_ROOT / f"broken-{uuid4().hex}.db"
    as_directory.mkdir()
    db = Database(f"sqlite+aiosqlite:///{as_directory}")
    try:
        added = await db.migrate_schema()
    finally:
        await db.dispose()
        as_directory.rmdir()

    assert added == ()


async def test_create_schema_runs_migration_itself() -> None:
    """`create_schema` на старой базе сходу исправляет схему и читает строки."""

    original, copy_path = copy_old_database("create-schema")
    db = Database(f"sqlite+aiosqlite:///{copy_path}")
    try:
        await db.create_schema()
        async with db.engine.connect() as connection:
            names = await connection.run_sync(_columns_of)
    finally:
        await db.dispose()
        original.unlink(missing_ok=True)
        copy_path.unlink(missing_ok=True)

    model_columns = {column.name for column in Generation.__table__.columns}
    assert names == model_columns


def _columns_of(sync_connection: object) -> set[str]:
    """Имена колонок `generations` через инспектор SQLAlchemy."""

    inspector = sa.inspect(sync_connection)
    return {
        column["name"] for column in inspector.get_columns("generations")
    }
