"""Тесты конфигурации (`app/config.py`)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.story_pipeline.settings import StorySettings

# Переменные, влияющие на новые умолчания: чистим окружение, иначе тест
# зависит от того, что разработчик прописал у себя в `.env`.
_ROUTERAI_ENV_VARS = (
    "ROUTERAI_API_KEY",
    "ROUTERAI_BASE_URL",
    "ROUTERAI_IMAGE_MODEL",
    "ROUTERAI_TIMEOUT",
    "STORY_IMAGE_ASPECT_RATIO",
)

_VIDEO_AND_GEMINI_ENV_VARS = (
    "KLING_API_KEY",
    "KLING_BASE_URL",
    "KLING_MODEL",
    "GEMINI_API_KEY",
    "GEMINI_BASE_URL",
    "GEMINI_MODEL",
)


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


def _bot_settings() -> Settings:
    """Настройки бота с обязательным токеном, без чтения `.env`."""

    config = Settings(  # type: ignore[call-arg]
        _env_file=None, max_bot_token="token-123"
    )
    return config


def test_routerai_image_defaults(monkeypatch) -> None:
    """Настройки RouterAI для иллюстраций имеют ожидаемые умолчания."""

    for name in _ROUTERAI_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    config = _bot_settings()

    assert config.routerai_api_key == ""
    assert config.routerai_base_url == "https://routerai.ru/api/v1"
    assert config.routerai_image_model == "google/gemini-3.1-flash-lite-image"
    assert config.routerai_timeout == 120.0


def test_routerai_image_environment_overrides_defaults(monkeypatch) -> None:
    """ROUTERAI_* переопределяются из окружения.

    STORY_IMAGE_ASPECT_RATIO — настройка пайплайна, у настроек бота такого
    поля нет.
    """

    for name in _ROUTERAI_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ROUTERAI_API_KEY", "routerai-key")
    monkeypatch.setenv("ROUTERAI_BASE_URL", "https://routerai.test/api/v1")
    monkeypatch.setenv("ROUTERAI_IMAGE_MODEL", "google/gemini-3.1-flash-image")
    monkeypatch.setenv("STORY_IMAGE_ASPECT_RATIO", "3:2")

    config = _bot_settings()

    assert config.routerai_api_key == "routerai-key"
    assert config.routerai_base_url == "https://routerai.test/api/v1"
    assert config.routerai_image_model == "google/gemini-3.1-flash-image"
    # Пропорции страниц — настройка пайплайна, у бота своего поля нет.
    assert not hasattr(config, "story_image_aspect_ratio")


def test_both_settings_classes_load_without_kling_and_gemini_keys(
    monkeypatch,
) -> None:
    """Оба класса настроек грузятся без ключей Kling и Gemini.

    Ключи Kling и Gemini не обязательны: их отсутствие включает mock-провайдеры
    и не должно ронять загрузку конфигурации.
    """

    for name in _VIDEO_AND_GEMINI_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name in _ROUTERAI_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MAX_BOT_TOKEN", "token-123")

    bot_settings = Settings(_env_file=None)  # type: ignore[call-arg]
    story_settings = StorySettings(_env_file=None)  # type: ignore[call-arg]

    assert bot_settings.kling_api_key == ""
    assert story_settings.gemini_api_key == ""
    assert story_settings.routerai_api_key == ""
    assert story_settings.story_image_aspect_ratio == "4:3"
