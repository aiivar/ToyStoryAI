"""Настройки пайплайна сказок на pydantic-settings.

Отдельный класс `StorySettings`, независимый от `app.config.Settings`:
последний требует обязательный `MAX_BOT_TOKEN`, который пайплайну не нужен.
Имена `DEEPSEEK_*` переиспользуются, новые настройки — `ROUTERAI_*` и
`STORY_*`. Обязательных полей нет: пустой `.env` не должен ронять импорт.
"""

from __future__ import annotations

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.story_pipeline.page_words import (
    DEFAULT_PAGE_WORDS_MAX,
    DEFAULT_PAGE_WORDS_MIN,
)

# Соотношения сторон, которые принимает генератор изображений RouterAI.
# Список зафиксирован документацией вендора: значение вне списка означало бы
# молчаливый отказ провайдера уже во время платного вызова.
SUPPORTED_ASPECT_RATIOS: tuple[str, ...] = (
    "1:1",
    "1:4",
    "1:8",
    "2:3",
    "3:2",
    "3:4",
    "4:1",
    "4:3",
    "4:5",
    "5:4",
    "8:1",
    "9:16",
    "16:9",
    "21:9",
)


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
        default=8192,
        description=(
            "Лимит токенов ответа. У reasoning-моделей сюда входят и "
            "токены размышлений, поэтому значение должно быть с запасом: "
            "при втрое более длинном тексте страниц и STORY_PAGES_MAX = 12 "
            "прежних 4096 хватало впритык, и обрыв выглядел бы как "
            "«модель не выдала схему»."
        ),
    )

    # --- RouterAI (генерация изображений) ------------------------------
    routerai_api_key: str = Field(
        default="",
        description=(
            "API-ключ RouterAI. Пустое значение включает MockImageProvider "
            "и исключает платные вызовы."
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
            "Таймаут HTTP-запроса к RouterAI в секундах. Живой вызов "
            "одной картинки занимает около 5 с, запас нужен на медленный "
            "ответ и повтор после временного сбоя."
        ),
    )

    # --- Параметры пайплайна -------------------------------------------
    story_image_aspect_ratio: str = Field(
        default="4:3",
        description=(
            "Соотношение сторон страниц-иллюстраций из списка "
            "SUPPORTED_ASPECT_RATIOS. Пустое значение не отправляет "
            "соотношение провайдеру и оставляет его умолчание."
        ),
    )
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
    story_illustrated_pages: int = Field(
        default=3,
        ge=0,
        description=(
            "Бюджет иллюстраций на прогон: сколько страниц получают "
            "картинку. Ноль отключает ограничение — иллюстрируются все "
            "страницы. Значение не меньше числа страниц означает то же "
            "самое. Ограничение нужно для разработки и приёмки: с нулём "
            "каталог прогона получает файл на каждую страницу."
        ),
    )
    story_page_words_min: int = Field(
        default=DEFAULT_PAGE_WORDS_MIN,
        description=(
            "Минимальный объём страницы сказки в словах. Нижняя граница "
            "измерима и проверяется кодом, поэтому «объём страницы» — не "
            "пожелание, а величина: страница короче уходит в корректирующий "
            "перезапрос, а если и после него — в `story.json` как деградация."
        ),
    )
    story_page_words_max: int = Field(
        default=DEFAULT_PAGE_WORDS_MAX,
        description=(
            "Верхняя граница объёма страницы сказки в словах. В промпте "
            "нужна как ориентир для модели: без верхней границы страница "
            "может вырасти в полосу текста без сцены."
        ),
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

    @field_validator("story_image_aspect_ratio")
    @classmethod
    def _check_aspect_ratio(cls, value: str) -> str:
        """Отклонить соотношение сторон, которого нет у провайдера.

        Проверка выполняется при загрузке настроек, то есть до первого
        платного вызова: опечатка в `.env` не должна обнаруживаться уже на
        середине прогона. Пустое значение допустимо — это «не отправлять
        соотношение и оставить умолчание провайдера».
        """

        stripped = value.strip()
        if stripped and stripped not in SUPPORTED_ASPECT_RATIOS:
            supported = ", ".join(SUPPORTED_ASPECT_RATIOS)
            raise ValueError(
                "STORY_IMAGE_ASPECT_RATIO: неподдерживаемое соотношение "
                f"сторон {stripped!r}. Допустимые значения: {supported} "
                "(или пусто — не отправлять соотношение провайдеру)."
            )
        return stripped

    @field_validator("story_page_words_min", "story_page_words_max")
    @classmethod
    def _check_page_words_positive(
        cls, value: int, info: ValidationInfo
    ) -> int:
        """Отклонить неположительный объём страницы, назвав переменную.

        Имя переменной окружения, а не поля: то, что человек правит в
        `.env`, и есть причина отказа. Ноль и отрицательное значение не
        имеют смысла как «объём страницы», а молчаливый откат к дефолту
        скрыл бы опечатку до первого живого прогона.
        """

        if value < 1:
            raise ValueError(
                f"{info.field_name.upper()}: объём страницы должен быть "
                f"положительным числом слов, получено {value}."
            )
        return value

    @model_validator(mode="after")
    def _check_page_words_range(self) -> StorySettings:
        """Проверить, что диапазон объёма не пуст.

        `max < min` проходит проверку положительности обоих полей, но
        означал бы, что ни одна страница не может попасть в диапазон:
        модель получила бы невыполнимое требование и гарантированно
        деградировала на каждом прогоне.
        """

        if self.story_page_words_max < self.story_page_words_min:
            raise ValueError(
                "STORY_PAGE_WORDS_MAX меньше STORY_PAGE_WORDS_MIN "
                f"({self.story_page_words_max} < {self.story_page_words_min}): "
                "диапазон объёма страницы пуст."
            )
        return self


def get_story_settings() -> StorySettings:
    """Вернуть настройки пайплайна из окружения и `.env`.

    Вынесено в функцию, чтобы тесты могли подменять источник значений.
    """

    return StorySettings()
