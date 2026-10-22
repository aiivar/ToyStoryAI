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

_GEMINI_ENV_VARS = (
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
    assert config.database_url == "sqlite+aiosqlite:///./bot.db"
    assert config.log_level == "INFO"


def test_no_kling_variable_is_required(monkeypatch) -> None:
    """Загрузка настроек не зависит ни от одной Kling-переменной.

    Переменные прежнего флоу остались бы в `.env` разработчика; если бы
   Settings их требовала, бот не поднялся бы с типичным локальным окружением.
    Ставятся заведомо неверные значения: подставленная настройка проявилась бы
    как поле с этим значением.
    """

    for name in (
        "KLING_API_KEY",
        "KLING_BASE_URL",
        "KLING_MODEL",
        "KLING_POLL_INTERVAL",
        "KLING_MAX_WAIT",
        "MOCK_VIDEO_URL",
    ):
        monkeypatch.setenv(name, "sentinel-from-legacy-env")

    config = Settings(_env_file=None, max_bot_token="token-123")  # type: ignore[call-arg]

    assert not [name for name in Settings.model_fields if "kling" in name]
    assert "mock_video_url" not in Settings.model_fields
    assert not hasattr(config, "kling_api_key")


def test_fails_without_required_token(monkeypatch) -> None:
    """Отсутствие обязательного токена MAX приводит к ошибке валидации."""

    monkeypatch.delenv("MAX_BOT_TOKEN", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_environment_overrides_defaults(monkeypatch) -> None:
    """Значения из окружения переопределяют значения по умолчанию."""

    monkeypatch.setenv("MAX_BOT_TOKEN", "env-token")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("DEEPSEEK_MAX_TOKENS", "8192")

    config = Settings(_env_file=None)  # type: ignore[call-arg]

    assert config.max_bot_token == "env-token"
    assert config.deepseek_model == "deepseek-v4-pro"
    assert config.deepseek_max_tokens == 8192


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


def test_both_settings_classes_load_without_api_keys(
    monkeypatch,
) -> None:
    """Оба класса настроек грузятся без единого ключа внешнего API.

    Ключи не обязательны: их отсутствие включает mock-провайдеры и не должно
    ронять загрузку конфигурации.
    """

    for name in _GEMINI_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name in _ROUTERAI_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MAX_BOT_TOKEN", "token-123")

    bot_settings = Settings(_env_file=None)  # type: ignore[call-arg]
    story_settings = StorySettings(_env_file=None)  # type: ignore[call-arg]

    assert bot_settings.routerai_api_key == ""
    assert bot_settings.deepseek_api_key == ""
    assert story_settings.routerai_api_key == ""
    assert story_settings.story_image_aspect_ratio == "4:3"


# --- Бюджет ожидания готовности медиа в MAX -----------------------------


def test_max_upload_retry_budget_is_long_enough_for_fresh_mp3(
    monkeypatch,
) -> None:
    """Бюджет ожидания готовности медиа перекрывает обработку mp3 в MAX.

    Живой прогон: свежий mp3 на 2,5 МБ / 5 минут получил токен загрузки, но
    MAX не успел его обработать за штатные 5 попыток по 2 с, и озвучка не
    дошла до пользователя. Обработка заняла около 12 с, поэтому суммарное
    ожидание должно быть заметно больше с запасом на тяжёлые файлы.
    """

    for name in ("MAX_UPLOAD_ATTEMPTS", "MAX_UPLOAD_RETRY_DELAY"):
        monkeypatch.delenv(name, raising=False)

    config = _bot_settings()

    assert config.max_upload_attempts >= 15
    assert config.max_upload_retry_delay >= 2.0
    assert (
        config.max_upload_attempts * config.max_upload_retry_delay
        >= 60.0
    )


def test_max_upload_retry_values_are_validated(monkeypatch) -> None:
    """Ноль попыток или нулевая задержка отклоняются при загрузке."""

    monkeypatch.setenv("MAX_BOT_TOKEN", "token-123")
    monkeypatch.setenv("MAX_UPLOAD_ATTEMPTS", "0")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]

    monkeypatch.setenv("MAX_UPLOAD_ATTEMPTS", "20")
    monkeypatch.setenv("MAX_UPLOAD_RETRY_DELAY", "0")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_max_upload_retry_env_overrides_defaults(monkeypatch) -> None:
    """Значения бюджета ожидания читаются из окружения."""

    monkeypatch.setenv("MAX_BOT_TOKEN", "token-123")
    monkeypatch.setenv("MAX_UPLOAD_ATTEMPTS", "45")
    monkeypatch.setenv("MAX_UPLOAD_RETRY_DELAY", "1.5")

    config = Settings(_env_file=None)  # type: ignore[call-arg]

    assert config.max_upload_attempts == 45
    assert config.max_upload_retry_delay == 1.5
