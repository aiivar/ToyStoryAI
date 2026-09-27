"""Настройки пайплайна сказок на pydantic-settings.

Отдельный класс `StorySettings`, независимый от `app.config.Settings`:
последний требует обязательный `MAX_BOT_TOKEN`, который пайплайну не нужен.
Имена `DEEPSEEK_*` переиспользуются, новые настройки — `GEMINI_*` и `STORY_*`.
Обязательных полей нет: пустой `.env` не должен ронять импорт.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorySettings(BaseSettings):
    """Типизированные настройки пайплайна, читаемые из окружения и `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- DeepSeek (генерация текста) -----------------------------------
    deepseek_api_key: str = Field(
        default="",
        description="API-ключ DeepSeek; пустой ключ вызывает ошибку до генерации.",
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        description="Базовый URL OpenAI-совместимого API DeepSeek.",
    )
    deepseek_model: str = Field(
        default="deepseek-flash",
        description="Модель генерации текста сказки.",
    )
    deepseek_timeout: float = Field(
        default=60.0,
        description="Таймаут HTTP-запроса к DeepSeek в секундах.",
    )
    deepseek_max_tokens: int = Field(
        default=4096,
        description=(
            "Лимит токенов ответа. У reasoning-моделей сюда входят и "
            "токены размышлений, поэтому значение должно быть с запасом."
        ),
    )

    # --- Gemini (генерация изображений) --------------------------------
    gemini_api_key: str = Field(
        default="",
        description=(
            "API-ключ Gemini. Пустое значение включает MockImageProvider "
            "и исключает платные вызовы."
        ),
    )
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com",
        description="Базовый URL Gemini API.",
    )
    gemini_model: str = Field(
        default="gemini-3.1-flash-image",
        description=(
            "Модель генерации изображений Gemini — Nano Banana 2 "
            "(семейство Gemini 3.1 Flash Image)."
        ),
    )
    gemini_timeout: float = Field(
        default=120.0,
        description="Таймаут HTTP-запроса к Gemini в секундах.",
    )

    # --- Параметры пайплайна -------------------------------------------
    story_default_pages: int = Field(
        default=6,
        description="Число страниц сказки по умолчанию.",
    )
    story_pages_min: int = Field(
        default=1,
        description="Минимально допустимое число страниц.",
    )
    story_pages_max: int = Field(
        default=12,
        description="Максимально допустимое число страниц.",
    )
    story_max_revisions: int = Field(
        default=2,
        description="Максимум циклов ревизии «судья → ревизор → судья».",
    )
    story_age_min: int = Field(
        default=3,
        description="Минимальный целевой возраст читателя.",
    )
    story_age_max: int = Field(
        default=6,
        description="Максимальный целевой возраст читателя.",
    )
    story_output_dir: str = Field(
        default="output",
        description="Каталог результатов прогона.",
    )


def get_story_settings() -> StorySettings:
    """Вернуть настройки пайплайна из окружения и `.env`.

    Вынесено в функцию, чтобы тесты могли подменять источник значений.
    """

    return StorySettings()
