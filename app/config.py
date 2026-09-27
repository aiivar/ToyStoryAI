"""Конфигурация приложения на pydantic-settings.

Все endpoint'ы вендоров, названия моделей, интервал опроса, максимальное
ожидание и URL базы берутся из окружения или `.env`. В коде нет ничего
провайдер-специфичного, кроме значений по умолчанию из брифа.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Типизированные настройки, читаемые из окружения и `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- MAX Bot API ---------------------------------------------------
    max_bot_token: str = Field(
        ...,
        description="Токен бота MAX. Единственное обязательное значение.",
    )

    # --- DeepSeek (генерация сказки) -----------------------------------
    deepseek_api_key: str = Field(
        default="",
        description="API-ключ DeepSeek; пустой ключ вызовет ошибку шага.",
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        description="Базовый URL OpenAI-совместимого API DeepSeek.",
    )
    deepseek_model: str = Field(
        default="deepseek-flash",
        description="Модель сказки; допускается deepseek-v4-pro.",
    )
    deepseek_timeout: float = Field(
        default=60.0,
        gt=0,
        description="Таймаут HTTP-запроса к DeepSeek в секундах.",
    )
    deepseek_max_tokens: int = Field(
        default=4096,
        gt=0,
        description=(
            "Лимит токенов ответа. У reasoning-моделей сюда входят и "
            "токены размышлений, поэтому значение должно быть с запасом."
        ),
    )

    # --- Видеопровайдер (Kling AI / GMI Cloud) -------------------------
    kling_api_key: str = Field(
        default="",
        description=(
            "API-ключ видеопровайдера. Пустое значение включает "
            "MockVideoProvider."
        ),
    )
    kling_base_url: str = Field(
        default="https://api.gmi-serving.com/v1",
        description="Базовый URL совместимого с Kling видеопровайдера.",
    )
    kling_model: str = Field(
        default="kling-v3-image-to-video",
        description="Модель image-to-video видеопровайдера.",
    )
    kling_create_path: str = Field(
        default="/videos/image2video",
        description="Путь создания запроса генерации видео.",
    )
    kling_status_path: str = Field(
        default="/videos/image2video/{request_id}",
        description="Путь опроса статуса запроса; содержит {request_id}.",
    )
    kling_poll_interval: float = Field(
        default=10.0,
        ge=0,
        description="Интервал опроса статуса видео в секундах.",
    )
    kling_max_wait: float = Field(
        default=600.0,
        gt=0,
        description="Максимальное ожидание генерации видео в секундах.",
    )
    kling_duration: str = Field(
        default="5",
        description="Длительность видео в секундах (параметр запроса).",
    )
    kling_mode: str = Field(
        default="std",
        description="Режим качества видеопровайдера (standard).",
    )
    kling_resolution: str = Field(
        default="720p",
        description="Разрешение видео.",
    )
    kling_sound: bool = Field(
        default=False,
        description="Генерация звука; в MVP выключена.",
    )
    kling_timeout: float = Field(
        default=120.0,
        gt=0,
        description="Таймаут HTTP-запроса к видеопровайдеру в секундах.",
    )
    mock_video_url: str = Field(
        default="https://example.com/mock/toy-story-preview.mp4",
        description="Ссылка на тестовое видео, выдаваемая заглушкой.",
    )

    # --- Хранение ------------------------------------------------------
    database_url: str = Field(
        default="sqlite+aiosqlite:///./bot.db",
        description="URL асинхронной базы данных SQLAlchemy.",
    )
    uploads_dir: str = Field(
        default="uploads",
        description="Каталог для сохранения принятых фото игрушек.",
    )

    # --- Валидация фото -------------------------------------------------
    photo_max_size_mb: float = Field(
        default=10.0,
        gt=0,
        description="Максимальный размер фото в мегабайтах.",
    )
    photo_min_side_px: int = Field(
        default=300,
        gt=0,
        description="Минимальная сторона фото в пикселях.",
    )
    photo_aspect_min: float = Field(
        default=1 / 2.5,
        gt=0,
        description="Минимальное соотношение сторон (ширина/высота).",
    )
    photo_aspect_max: float = Field(
        default=2.5,
        gt=0,
        description="Максимальное соотношение сторон (ширина/высота).",
    )

    # --- Прочее ---------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Уровень логирования (DEBUG/INFO/WARNING/ERROR).",
    )


def get_settings() -> Settings:
    """Вернуть настройки из окружения и `.env`.

    Вынесено в функцию, чтобы тесты могли подменять источник значений.
    """

    return Settings()
