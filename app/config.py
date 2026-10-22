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
    max_upload_attempts: int = Field(
        default=30,
        gt=0,
        description=(
            "Сколько раз пытаться отправить сообщение с вложением, пока MAX "
            "не закончит его обработку (ответ attachment.not.ready)."
        ),
    )
    max_upload_retry_delay: float = Field(
        default=3.0,
        gt=0,
        description=(
            "Пауза между попытками отправки вложения в секундах. Вместе с "
            "числом попыток задаёт бюджет ожидания готовности медиа."
        ),
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

    # --- RouterAI (генерация иллюстраций) ------------------------------
    routerai_api_key: str = Field(
        default="",
        description=(
            "API-ключ RouterAI. Пустое значение включает MockImageProvider "
            "(картинки 32×32) и исключает платные вызовы."
        ),
    )
    routerai_base_url: str = Field(
        default="https://routerai.ru/api/v1",
        description="Базовый URL OpenAI-совместимого API RouterAI.",
    )
    routerai_image_model: str = Field(
        default="google/gemini-3.1-flash-lite-image",
        description="Модель генерации изображений в RouterAI.",
    )
    routerai_timeout: float = Field(
        default=120.0,
        gt=0,
        description=(
            "Таймаут HTTP-запроса к RouterAI в секундах. Живой вызов одной "
            "картинки занимает около 5 с, запас нужен на медленный ответ и "
            "повтор после временного сбоя."
        ),
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
