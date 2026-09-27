"""Общие фикстуры тестов."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import pytest

from app.config import Settings
from app.db.engine import Database
from app.db.repository import GenerationRepository

TMP_ROOT = Path(".tmp/tests")


@pytest.fixture(autouse=True)
def _block_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Запретить живые сетевые вызовы в unit-тестах (кроме loopback)."""

    import socket

    real_connect = socket.socket.connect
    allowed = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}

    def guarded_connect(self, address):  # type: ignore[no-untyped-def]
        host = address[0] if isinstance(address, tuple) else str(address)
        if host in allowed:
            return real_connect(self, address)
        raise RuntimeError(
            f"Живой сетевой вызов запрещён в тестах: {address!r}"
        )

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture
def settings() -> Settings:
    """Настройки с тестовым токеном и изоляцией от реального `.env`."""

    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        max_bot_token="test-token",
    )


@pytest.fixture
async def database() -> AsyncIterator[Database]:
    """Изолированная файловая SQLite-база для одного теста."""

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    db_path = TMP_ROOT / f"test-{uuid4().hex}.db"
    db = Database(f"sqlite+aiosqlite:///{db_path}")
    await db.create_schema()
    try:
        yield db
    finally:
        await db.dispose()
        db_path.unlink(missing_ok=True)


@pytest.fixture
async def repository(database: Database) -> GenerationRepository:
    """Репозиторий поверх изолированной базы."""

    return GenerationRepository(database)
