"""Модели состояния и контракт узла пайплайна сказок.

Единый мутируемый объект `StoryState` протаскивается через все этапы и
сериализуется в JSON без бинарных данных: изображения представлены путями
и seed. Протокол `Stage` описывает узел графа.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class NarrativeArc(BaseModel):
    """Нарративная структура сказки по канонам детского повествования."""

    exposition: str = Field(description="Завязка: знакомство с героем и миром.")
    rising_action: str = Field(description="Развитие действия: появление задачи.")
    climax: str = Field(description="Кульминация: главное испытание героя.")
    resolution: str = Field(description="Развязка: разрешение и мораль.")


class StoryBrief(BaseModel):
    """План сказки, созданный этапом планирования."""

    logline: str = Field(description="Одна строка, о чём сказка.")
    controlling_idea: str = Field(description="Главная мысль, которую несёт сказка.")
    character_arc: str = Field(description="Как меняется герой по ходу сказки.")
    narrative_arc: NarrativeArc = Field(description="Структура повествования.")


class PageText(BaseModel):
    """Текст одной страницы сказки."""

    page_num: int = Field(ge=1, description="Номер страницы, начиная с 1.")
    text: str = Field(description="Текст страницы на русском языке.")


class PagesPayload(BaseModel):
    """Обёртка ответа сценариста: список страниц сказки.

    Языковая модель возвращает JSON-объект с полем `pages`; отдельная
    обёртка нужна, потому что `complete_json` разбирает ответ в корневую
    pydantic-модель, а список страниц — не корневой объект. Модель живёт
    рядом с `PageText`, чтобы схема ответа и её рендер были в одном месте.
    """

    pages: list[PageText] = Field(
        description="Страницы сказки в порядке нумерации 1..N."
    )


class Character(BaseModel):
    """Персонаж и его описание-якорь для согласованности иллюстраций."""

    name: str = Field(description="Имя персонажа.")
    appearance: str = Field(description="Внешность: телосложение, цвет, черты.")
    clothing: str = Field(description="Одежда и аксессуары.")
    distinguishing_features: str = Field(
        description="Отличительные особенности персонажа."
    )

    @property
    def full_desc(self) -> str:
        """Дословное описание-якорь: appearance, clothing, features через пробел.

        Именно эта строка подставляется в промпт изображения на каждой
        странице, чтобы облик героя не «плыл» между кадрами. Отдельно не
        хранится: единственный источник — три поля выше.
        """

        return " ".join(
            (self.appearance, self.clothing, self.distinguishing_features)
        )


class Location(BaseModel):
    """Локация и её описание для согласованности фона."""

    name: str = Field(description="Название локации.")
    description: str = Field(description="Описание локации для промпта изображения.")


class KeyObject(BaseModel):
    """Ключевой объект и его описание для согласованности реквизита."""

    name: str = Field(description="Название объекта.")
    description: str = Field(description="Описание объекта для промпта изображения.")


class ExtractedEntities(BaseModel):
    """Реестр сущностей, извлечённый из текста сказки.

    Служит якорем согласованности: сцены ссылаются на имена отсюда, а
    `Character.full_desc` повторяется в каждом промпте изображения.
    """

    characters: list[Character] = Field(description="Персонажи.")
    locations: list[Location] = Field(description="Локации.")
    key_objects: list[KeyObject] = Field(description="Ключевые объекты.")
    art_style: str = Field(description="Единый художественный стиль иллюстраций.")


class PageScene(BaseModel):
    """Раскадровка одной страницы: что происходит и кто/что в кадре."""

    page_num: int = Field(ge=1, description="Номер страницы, начиная с 1.")
    scene_description: str = Field(description="Описание сцены для иллюстрации.")
    characters_in_scene: list[str] = Field(description="Имена персонажей в кадре.")
    location: str = Field(description="Название локации сцены.")
    objects_in_scene: list[str] = Field(description="Имена ключевых объектов в кадре.")


class ScenesPayload(BaseModel):
    """Обёртка ответа раскадровщика: список сцен.

    Как и `PagesPayload`, это корневой объект ответа: `complete_json`
    разбирает ответ в pydantic-модель, а список сцен — не корневой объект.
    Модель живёт рядом с `PageScene`, чтобы схема ответа и её рендер были
    в одном месте.
    """

    scenes: list[PageScene] = Field(
        description="Сцены по одной на страницу, с номерами страниц."
    )


class SceneRefViolation(BaseModel):
    """Ссылка сцены на имя, отсутствующее в реестре сущностей.

    Фиксирует страницу, проблемное поле сцены и исходное значение — чтобы
    отчёт и корректирующий перезапрос были адресными, а имя не терялось.
    """

    page_num: int = Field(ge=1, description="Номер проблемной страницы.")
    field: str = Field(
        description=(
            "Поле сцены с нарушением: characters_in_scene, location "
            "или objects_in_scene."
        )
    )
    value: str = Field(description="Исходное значение ссылки из ответа модели.")


class JudgeVerdict(BaseModel):
    """Вердикт LLM-судьи о качестве текста и сцен."""

    passed: bool = Field(description="Пройдена ли проверка.")
    issues: list[str] = Field(description="Найденные проблемы, если есть.")
    problem_pages: list[int] = Field(
        default_factory=list, description="Номера проблемных страниц (1-based)."
    )


class RevisionPayload(BaseModel):
    """Обёртка ответа ревизора: обновлённые страницы и их сцены.

    Как и `PagesPayload`/`ScenesPayload`, это корневой объект ответа:
    `complete_json` разбирает ответ в pydantic-модель, а пара списков — не
    корневой объект. Модель живёт рядом с `PageText`/`PageScene`, чтобы
    схема ответа ревизора и её рендер были в одном месте; текст и сцена
    идут вместе, потому что правка текста требует пересборки сцены.
    """

    pages: list[PageText] = Field(
        description="Обновлённые тексты только проблемных страниц."
    )
    scenes: list[PageScene] = Field(
        description="Обновлённые сцены тех же проблемных страниц."
    )


class PageArt(BaseModel):
    """Готовое изображение страницы: путь и seed, без бинарных данных."""

    page_num: int = Field(ge=1, description="Номер страницы, начиная с 1.")
    path: str = Field(
        description="Путь к файлу страницы; расширение по типу файла."
    )
    seed: int | None = Field(
        default=None, description="Seed генерации, если провайдер его вернул."
    )


class StageMetrics(BaseModel):
    """Метрики одного этапа для наблюдаемости прогона."""

    stage: str = Field(description="Имя этапа.")
    duration_ms: float = Field(description="Длительность этапа в миллисекундах.")
    attempts: int = Field(default=1, description="Число попыток выполнения.")
    prompt_tokens: int | None = Field(
        default=None, description="Токены промпта для LLM-этапа."
    )
    completion_tokens: int | None = Field(
        default=None, description="Токены ответа для LLM-этапа."
    )
    error: str | None = Field(
        default=None, description="Текст ошибки, если этап завершился сбоем."
    )


class StoryState(BaseModel):
    """Единое состояние прогона, протаскиваемое через все этапы."""

    # --- Вход ----------------------------------------------------------
    theme: str = Field(description="Тема сказки.")
    age: int = Field(description="Целевой возраст читателя.")
    moral: str = Field(description="Мораль, которую несёт сказка.")
    num_pages: int = Field(default=6, description="Число страниц сказки.")
    illustrated_pages: list[int] = Field(
        default_factory=list,
        description=(
            "Номера страниц, для которых заказываются иллюстрации: "
            "подмножество 1..num_pages. Пустой список означает «набор не "
            "задан» — иллюстрируются все страницы. Набор лежит в состоянии "
            "рядом со списком изображений, поэтому по артефакту прогона "
            "видно, какие страницы не проиллюстрированы по замыслу, а "
            "какие — из-за сбоя генерации."
        ),
    )
    toy_description: str | None = Field(
        default=None,
        description=(
            "Описание внешности игрушки-героя по фотографии родителя. "
            "Единый источник правды об облике: его читают планировщик, "
            "экстрактор и судья, поэтому разойтись с ним текст и иллюстрации "
            "не могут. Пусто в CLI-сценарии, где фото нет."
        ),
    )

    # --- Артефакты этапов ----------------------------------------------
    brief: StoryBrief | None = Field(default=None, description="План сказки.")
    pages: list[PageText] = Field(default_factory=list, description="Тексты страниц.")
    entities: ExtractedEntities | None = Field(
        default=None, description="Реестр извлечённых сущностей."
    )
    scenes: list[PageScene] = Field(default_factory=list, description="Раскадровка.")
    art_prompts: list[tuple[int, str]] = Field(
        default_factory=list,
        description=(
            "Промпты изображений парами «номер страницы — промпт», причём "
            "только для страниц из набора иллюстрируемых. Номер в паре "
            "обязателен: неявная связь через порядок элементов списка "
            "развалилась бы при первом же рефакторинге, а при разборе "
            "прогона по такому полю видно, к какой странице относится "
            "промпт. Промптов страниц вне набора здесь нет намеренно: "
            "заказа под них не было, и их наличие читалось бы как сбой."
        ),
    )
    character_sheet_path: str | None = Field(
        default=None, description="Путь к reference-изображению персонажей."
    )
    character_sheet_seed: int | None = Field(
        default=None, description="Seed генерации character sheet."
    )
    images: list[PageArt] = Field(
        default_factory=list, description="Готовые изображения страниц."
    )

    # --- Контроль качества ---------------------------------------------
    judge_history: list[JudgeVerdict] = Field(
        default_factory=list, description="Все вердикты судьи по порядку."
    )
    revision_count: int = Field(default=0, description="Число выполненных ревизий.")
    unresolved_pages: list[int] = Field(
        default_factory=list,
        description=(
            "Номера страниц с ссылками на неизвестные имена после "
            "корректирующего перезапроса раскадровки. Передаются в петлю "
            "судья/ревизор для адресной правки; имена не удаляются молча."
        ),
    )
    status: Literal["ok", "needs_review", "failed"] = Field(
        default="ok", description="Итоговый статус прогона."
    )

    # --- Инфраструктура ------------------------------------------------
    metrics: list[StageMetrics] = Field(
        default_factory=list, description="Метрики этапов по порядку."
    )
    output_dir: str | None = Field(
        default=None, description="Каталог результатов прогона."
    )
    started_at: datetime = Field(description="Момент начала прогона.")
    finished_at: datetime | None = Field(
        default=None, description="Момент завершения прогона."
    )


@runtime_checkable
class Stage(Protocol):
    """Узел графа пайплайна.

    Реализация получает текущее состояние, обогащает его артефактами своего
    этапа и возвращает (тот же) объект состояния.
    """

    name: str

    async def run(self, state: StoryState) -> StoryState:
        """Выполнить этап и вернуть обновлённое состояние."""
