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


class _RecordingBot:
    """Заглушка `Bot`, запоминающая аргументы конструктора."""

    # Аргументы последнего собранного бота: тесты проверяют флаги
    # auto_requests/auto_check_subscriptions, не поднимая сеть.
    last_kwargs: dict = {}

    def __init__(self, **kwargs: object) -> None:
        _RecordingBot.last_kwargs = dict(kwargs)

    async def download_bytes(self, url: str) -> bytes:  # pragma: no cover
        return b""

    async def send_message(self, **kwargs: object) -> None:  # pragma: no cover
        return None


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


async def test_main_warns_about_mock_images_without_routerai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Без `ROUTERAI_API_KEY` при старте есть предупреждение про mock 32×32."""

    from loguru import logger as loguru_logger

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    db_path = TMP_ROOT / f"mock-{uuid4().hex}.db"
    messages: list[str] = []

    monkeypatch.setenv("MAX_BOT_TOKEN", "invalid-token")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("UPLOADS_DIR", str(TMP_ROOT / "mock-uploads"))
    monkeypatch.setenv("ROUTERAI_API_KEY", "")
    monkeypatch.setattr(main_module, "Dispatcher", _FailingDispatcher)
    monkeypatch.setattr(main_module, "Bot", _RecordingBot)
    # `setup_logging` пересоздаёт приёмники loguru, поэтому подменяем его:
    # предупреждение всё равно уходит в тот же логгер.
    monkeypatch.setattr(main_module, "setup_logging", lambda settings: None)
    sink_id = loguru_logger.add(
        lambda message: messages.append(str(message)),
        level="WARNING",
    )
    try:
        code = await main_module.main()
    finally:
        loguru_logger.remove(sink_id)

    assert code == 1
    warnings = [text for text in messages if "ROUTERAI_API_KEY" in text]
    assert warnings, "Ожидалось предупреждение о mock-картинках"
    assert "32×32" in warnings[0]
    db_path.unlink(missing_ok=True)


async def test_main_does_not_warn_when_routerai_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """С заданным ключом RouterAI предупреждения о mock-картинках нет."""

    from loguru import logger as loguru_logger

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    db_path = TMP_ROOT / f"real-{uuid4().hex}.db"
    messages: list[str] = []

    monkeypatch.setenv("MAX_BOT_TOKEN", "invalid-token")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("UPLOADS_DIR", str(TMP_ROOT / "real-uploads"))
    monkeypatch.setenv("ROUTERAI_API_KEY", "test-key")
    monkeypatch.setattr(main_module, "Dispatcher", _FailingDispatcher)
    monkeypatch.setattr(main_module, "Bot", _RecordingBot)
    monkeypatch.setattr(main_module, "setup_logging", lambda settings: None)
    sink_id = loguru_logger.add(
        lambda message: messages.append(str(message)),
        level="WARNING",
    )
    try:
        code = await main_module.main()
    finally:
        loguru_logger.remove(sink_id)

    assert code == 1
    assert not [text for text in messages if "ROUTERAI_API_KEY" in text]
    db_path.unlink(missing_ok=True)


async def test_main_disables_auto_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`main` отключает авто-запросы chat/from_user (иначе chat.not.found)."""

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    db_path = TMP_ROOT / f"cli-{uuid4().hex}.db"
    monkeypatch.setenv("MAX_BOT_TOKEN", "invalid-token")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("UPLOADS_DIR", str(TMP_ROOT / "cli-uploads"))
    monkeypatch.setattr(main_module, "Bot", _RecordingBot)
    monkeypatch.setattr(main_module, "Dispatcher", _FailingDispatcher)

    code = await main_module.main()

    assert code == 1
    assert _RecordingBot.last_kwargs["auto_requests"] is False
    assert _RecordingBot.last_kwargs["auto_check_subscriptions"] is False
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


async def test_main_passes_upload_retry_budget_to_bot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Бюджет ожидания готовности медиа передаётся в `Bot`.

    MAX обрабатывает mp3 асинхронно и отвечает `attachment.not.ready`;
    штатных пяти попыток по 2 с не хватает свежему файлу на несколько
    мегабайт, поэтому бюджет задаётся настройками и обязан доехать до бота.
    """

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    db_path = TMP_ROOT / f"cli-{uuid4().hex}.db"
    monkeypatch.setenv("MAX_BOT_TOKEN", "invalid-token")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("UPLOADS_DIR", str(TMP_ROOT / "cli-uploads"))
    monkeypatch.setenv("MAX_UPLOAD_ATTEMPTS", "30")
    monkeypatch.setenv("MAX_UPLOAD_RETRY_DELAY", "3")
    monkeypatch.setattr(main_module, "Bot", _RecordingBot)
    monkeypatch.setattr(main_module, "Dispatcher", _FailingDispatcher)

    code = await main_module.main()

    assert code == 1
    assert _RecordingBot.last_kwargs["after_upload_attempts"] == 30
    assert _RecordingBot.last_kwargs["after_upload_retry_delay"] == 3.0
    db_path.unlink(missing_ok=True)
