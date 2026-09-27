"""Тесты точки входа (`app/main.py`)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from maxapi import Dispatcher
from maxapi.exceptions import InvalidToken

from app import main as main_module

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TMP_ROOT = PROJECT_ROOT / ".tmp" / "tests"


class _FailingDispatcher(Dispatcher):
    """Диспетчер, эмулирующий неверный токен MAX."""

    async def start_polling(self, bot, *, skip_updates: bool = False) -> None:
        raise InvalidToken("invalid token")


async def test_main_exits_gracefully_on_invalid_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """При неверном токене `main` завершается без исключения и с кодом 1."""

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    db_path = TMP_ROOT / f"cli-{uuid4().hex}.db"

    monkeypatch.setenv("MAX_BOT_TOKEN", "invalid-token")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("UPLOADS_DIR", str(TMP_ROOT / "cli-uploads"))
    monkeypatch.setattr(main_module, "Dispatcher", _FailingDispatcher)

    code = await main_module.main()

    assert code == 1
    db_path.unlink(missing_ok=True)


async def test_main_disables_auto_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`main` отключает авто-запросы chat/from_user (иначе chat.not.found)."""

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    db_path = TMP_ROOT / f"cli-{uuid4().hex}.db"
    captured: dict = {}

    class _RecordingBot:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        async def download_bytes(self, url: str) -> bytes:  # pragma: no cover
            return b""

    monkeypatch.setenv("MAX_BOT_TOKEN", "invalid-token")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("UPLOADS_DIR", str(TMP_ROOT / "cli-uploads"))
    monkeypatch.setattr(main_module, "Bot", _RecordingBot)
    monkeypatch.setattr(main_module, "Dispatcher", _FailingDispatcher)

    code = await main_module.main()

    assert code == 1
    assert captured["auto_requests"] is False
    assert captured["auto_check_subscriptions"] is False
    db_path.unlink(missing_ok=True)


@pytest.mark.integration
def test_module_entrypoint_starts_and_stops() -> None:
    """`python -m app.main` стартует и корректно выходит при неверном токене."""

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    db_path = TMP_ROOT / f"entrypoint-{uuid4().hex}.db"

    env = os.environ.copy()
    env.update(
        {
            "MAX_BOT_TOKEN": "definitely-invalid-token",
            "DATABASE_URL": f"sqlite+aiosqlite:///{db_path}",
            "UPLOADS_DIR": str(TMP_ROOT / "entrypoint-uploads"),
            "KLING_API_KEY": "",
            "LOG_LEVEL": "INFO",
        }
    )

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "app.main"],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - среда без сети
        pytest.skip(f"Процесс не завершился за 60 с: {exc}")
        return

    output = completed.stdout + completed.stderr
    assert completed.returncode == 1
    assert "Неверный MAX_BOT_TOKEN" in output
    db_path.unlink(missing_ok=True)
