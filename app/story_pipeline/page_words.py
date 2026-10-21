"""Границы объёма страницы сказки: счёт слов, дефолты и проверка пары.

Объём страницы задан диапазоном слов, а не числом предложений: предложения
языковая модель считает неточно («2–3 предложения» на живой проверке дали
27–41 слово с разбросом больше чем втрое), а слова можно посчитать в коде
тем же способом, каким их считает валидатор, и получить исполняемую
проверку вместо пожелания.

Модуль держит три вещи, нужные сразу трём потребителям — сценаристу,
ревизору и оркестратору, — поэтому они не дублируются по модулям этапов:
счёт слов, дефолты прогона и проверку вырожденной пары границ.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.story_pipeline.errors import PipelineValidationError

if TYPE_CHECKING:  # pragma: no cover - импорт только для типов
    from app.story_pipeline.models import StoryState

# Дефолты совпадают со значениями `StorySettings.story_page_words_min/max`:
# разрешённые границы кладутся в состояние прогона, откуда их читают
# сценарист и ревизор. Само значение живёт в настройках, а не в промптах,
# чтобы подбор цифр по итогам живых прогонов не был правкой кода.
DEFAULT_PAGE_WORDS_MIN = 60
DEFAULT_PAGE_WORDS_MAX = 120


def count_words(text: str) -> int:
    """Число слов в тексте страницы.

    Считаем `len(text.split())`: счёт обязан совпадать с тем, чем измеряется
    недобор в метрике и в артефакте прогона, поэтому он вынесен в одну
    функцию, а не расходится по вызывающим местам.

    Расхождение с моделью ожидаемо (она считает слова своими средствами) и
    поэтому не страшно: порог задаётся снизу, и запас дефолта съедает его.
    """

    return len(text.split())


def resolve_page_words(state: StoryState) -> tuple[int, int]:
    """Границы объёма страницы для этапа: из состояния, иначе дефолты.

    Этапы читают границы из состояния, а не из настроек: их разрешает
    оркестратор один раз на прогон. Дефолтный запас нужен прогонам, которые
    собирают состояние сами и не проходят через оркестратор, — иначе в
    промпт уехало бы `None`.
    """

    words_min = (
        DEFAULT_PAGE_WORDS_MIN
        if state.page_words_min is None
        else state.page_words_min
    )
    words_max = (
        DEFAULT_PAGE_WORDS_MAX
        if state.page_words_max is None
        else state.page_words_max
    )
    return words_min, words_max


def validate_page_words(words_min: int, words_max: int) -> None:
    """Проверить пару границ, назвав ошибочные переменные окружения.

    Проверка выполняется до первого платного вызова: опечатка в `.env` или
    в per-run переопределении обязана обнаруживаться на разборе конфигурации,
    а не на середине прогона, где её уже нечем лечить. Вырожденная пара
    отклоняется, а не молча приводится к дефолту: молчаливый откат к
    дефолту выглядел бы в `story.json` как заданный объём.
    """

    if words_min < 1:
        raise PipelineValidationError(
            "STORY_PAGE_WORDS_MIN должен быть положительным, получено "
            f"{words_min}: объём страницы в словах не задан."
        )
    if words_max < 1:
        raise PipelineValidationError(
            "STORY_PAGE_WORDS_MAX должен быть положительным, получено "
            f"{words_max}: объём страницы в словах не задан."
        )
    if words_max < words_min:
        raise PipelineValidationError(
            "STORY_PAGE_WORDS_MAX меньше STORY_PAGE_WORDS_MIN "
            f"({words_max} < {words_min}): диапазон объёма страницы пуст."
        )


__all__ = [
    "DEFAULT_PAGE_WORDS_MAX",
    "DEFAULT_PAGE_WORDS_MIN",
    "count_words",
    "resolve_page_words",
    "validate_page_words",
]
