"""Тесты конфигурации (`app/config.py`)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_loads_default_values_with_required_token(monkeypatch) -> None:
    """Значения по умолчанию загружаются, если задан обязательный токен."""

    monkeypatch.delenv("MAX_BOT_TOKEN", raising=False)

    config = Settings(_env_file=None, max_bot_token="token-123")  # type: ignore[call-arg]

    assert config.max_bot_token == "token-123"
    assert config.deepseek_base_url == "https://api.deepseek.com"
    assert config.deepseek_model == "deepseek-flash"
    assert config.kling_model == "kling-v3-image-to-video"
    assert config.kling_poll_interval == 10.0
    assert config.kling_max_wait == 600.0
    assert config.database_url == "sqlite+aiosqlite:///./bot.db"
    assert config.log_level == "INFO"
    # Kling-ключ может быть пустым — это включает mock-провайдер.
    assert config.kling_api_key == ""


def test_fails_without_required_token(monkeypatch) -> None:
    """Отсутствие обязательного токена MAX приводит к ошибке валидации."""

    monkeypatch.delenv("MAX_BOT_TOKEN", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_environment_overrides_defaults(monkeypatch) -> None:
    """Значения из окружения переопределяют значения по умолчанию."""

    monkeypatch.setenv("MAX_BOT_TOKEN", "env-token")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("KLING_POLL_INTERVAL", "3")

    config = Settings(_env_file=None)  # type: ignore[call-arg]

    assert config.max_bot_token == "env-token"
    assert config.deepseek_model == "deepseek-v4-pro"
    assert config.kling_poll_interval == 3.0
