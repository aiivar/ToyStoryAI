"""Централизованные промпты пайплайна сказок.

Здесь живут все обращения к языковой модели. Промпты не размещаются в
модулях этапов или ином коде: правка формулировки локализуется в этом файле.
Остальные константы по этапам добавят последующие задачи.
"""

from __future__ import annotations

from app.story_pipeline.models import (
    ExtractedEntities,
    NarrativeArc,
    PageText,
    SceneRefViolation,
)

# Единый художественный стиль иллюстраций. Подставляется в каждый промпт
# изображения как якорь визуальной согласованности между страницами.
ART_STYLE_DEFAULT = "soft watercolor storybook illustration"

# --- Этап планирования -------------------------------------------------

PLANNER_SYSTEM = (
    "Ты — опытный детский сказочник и редактор. Ты планируешь добрую "
    "сказку для ребёнка, строго учитываешь целевой возраст и отвечаешь "
    "только валидным JSON без markdown-обёрток и пояснений."
)

# Плейсхолдеры: theme, age, moral, schema (рендер схемы StoryBrief).
PLANNER_USER = (
    "Составь план детской сказки.\n"
    "Тема: {theme}.\n"
    "Целевой возраст ребёнка: {age} лет.\n"
    "Мораль: {moral}.\n\n"
    "Требования к плану:\n"
    "- logline — одно предложение о том, о чём сказка;\n"
    "- controlling_idea — мораль как тема повествования, а не прямая "
    "проповедь;\n"
    "- character_arc — что герой понимает и как меняется к концу;\n"
    "- narrative_arc — четыре бита (завязка, нарастание, кульминация, "
    "развязка), пригодные для детской сказки для ребёнка {age} лет.\n\n"
    "Верни ответ строго по JSON-схеме:\n{schema}"
)

# --- Этап сценариста ---------------------------------------------------

SCREENWRITER_SYSTEM = (
    "Ты — детский писатель. Ты пишешь тёплую понятную сказку для ребёнка "
    "и отвечаешь только валидным JSON без markdown-обёрток и пояснений."
)

# Плейсхолдеры: logline, controlling_idea, character_arc, arc_plan,
# num_pages, age, schema (рендер схемы PagesPayload).
SCREENWRITER_USER = (
    "Бриф сказки:\n"
    "Логлайн: {logline}\n"
    "Главная мысль: {controlling_idea}\n"
    "Изменение героя: {character_arc}\n\n"
    "Нарративная арка и привязка страниц к её битам:\n"
    "{arc_plan}\n\n"
    "Напиши сказку ровно из {num_pages} страниц для ребёнка {age} лет. "
    "Пронумеруй страницы с 1 до {num_pages} без пропусков и повторов. "
    "На каждой странице 2–3 предложения с сенсорными деталями — что герой "
    "видит, слышит и чувствует. Используй лексику, доступную ребёнку "
    "{age} лет. Каждая страница должна быть иллюстрируема.\n\n"
    "Верни ответ строго по JSON-схеме:\n{schema}"
)

# Плейсхолдеры: problems, base_user, num_pages.
SCREENWRITER_REPAIR_USER = (
    "Предыдущий ответ не соответствует требованиям: {problems}\n\n"
    "{base_user}\n\n"
    "Верни исправленный ответ заново: ровно {num_pages} страниц с "
    "нумерацией 1..{num_pages}, без пропусков, повторов и лишних страниц."
)


# --- Этап экстрактора сущностей ----------------------------------------

EXTRACTOR_SYSTEM = (
    "Ты — художник-постановщик детской иллюстрированной книги. Ты разбираешь "
    "готовый текст сказки на устойчивый реестр сущностей для иллюстратора и "
    "отвечаешь только валидным JSON без markdown-обёрток и пояснений."
)

# Плейсхолдеры: pages (текст страниц), schema (рендер схемы ExtractedEntities).
EXTRACTOR_USER = (
    "Текст сказки по страницам:\n{pages}\n\n"
    "Собери реестр сущностей для иллюстраций.\n"
    "Требования:\n"
    "- art_style — единый художественный стиль для ВСЕЙ сказки: выбери его "
    "один раз и не меняй между страницами;\n"
    "- описания персонажей — конкретные и визуальные: цвет, форма, материал, "
    "одежда, отличительные черты. Их будут дословно повторять на каждой "
    "странице как якорь внешности, поэтому избегай общих слов вроде "
    "«красивый» или «милый»;\n"
    "- в реестр попадают только сущности, значимые для иллюстраций;\n"
    "- имена сущностей — короткие и однозначные: на них будут ссылаться "
    "сцены.\n\n"
    "Верни ответ строго по JSON-схеме:\n{schema}"
)

# --- Этап раскадровки --------------------------------------------------

SCENE_SYSTEM = (
    "Ты — режиссёр-раскадровщик детской иллюстрированной книги. Ты "
    "раскладываешь текст на кадры и отвечаешь только валидным JSON без "
    "markdown-обёрток и пояснений."
)

# Плейсхолдеры: entities (реестр), pages (текст страниц),
# schema (рендер схемы ScenesPayload).
SCENE_USER = (
    "Реестр сущностей сказки:\n{entities}\n\n"
    "Текст сказки по страницам:\n{pages}\n\n"
    "Сделай раскадровку: ровно одна сцена на каждую страницу, при этом "
    "page_num сцены совпадает с номером страницы.\n"
    "Требования:\n"
    "- scene_description описывает кадр (действие и композицию: кто где "
    "стоит, что происходит) с явными именами из реестра;\n"
    "- characters_in_scene, location и objects_in_scene используют ТОЧНЫЕ "
    "имена из реестра сущностей, без синонимов и пересказов;\n"
    "- если персонажа на странице нет, верни пустой список "
    "characters_in_scene;\n"
    "- если ключевых объектов в кадре нет, верни пустой список "
    "objects_in_scene.\n\n"
    "Верни ответ строго по JSON-схеме:\n{schema}"
)

# Плейсхолдеры: entities (реестр допустимых имён),
# problem_pages (страницы с нарушениями), schema (рендер схемы ScenesPayload).
SCENE_REPAIR_USER = (
    "В раскадровке есть ссылки на имена, которых нет в реестре сущностей.\n\n"
    "Допустимые имена из реестра:\n{entities}\n\n"
    "Проблемные страницы:\n{problem_pages}\n\n"
    "Перепиши сцены ТОЛЬКО для этих страниц, заменив неверные имена на "
    "точные имена из реестра. Сохрани page_num, смысл и композицию кадра.\n\n"
    "Верни ответ строго по JSON-схеме:\n{schema}"
)


def format_pages(pages: list[PageText]) -> str:
    """Собрать текст страниц в нумерованный блок для промпта."""

    return "\n".join(
        f"Страница {page.page_num}: {page.text}" for page in pages
    )


def format_entity_registry(entities: ExtractedEntities) -> str:
    """Собрать реестр сущностей с описаниями в текстовый блок промпта.

    Персонажи даны вместе с дословным `full_desc`: этот же текст является
    якорем внешности в промптах изображений.
    """

    characters = "\n".join(
        f"- {character.name}: {character.full_desc}"
        for character in entities.characters
    )
    locations = "\n".join(
        f"- {location.name}: {location.description}"
        for location in entities.locations
    )
    objects = "\n".join(
        f"- {item.name}: {item.description}" for item in entities.key_objects
    )
    return (
        f"Единый стиль иллюстраций: {entities.art_style}\n"
        f"Персонажи:\n{characters}\n"
        f"Локации:\n{locations}\n"
        f"Ключевые объекты:\n{objects}"
    )


def format_scene_violations(violations: list[SceneRefViolation]) -> str:
    """Собрать список нарушений ссылок сцен в текстовый блок промпта.

    Группирует нарушения по проблемным страницам, чтобы перезапрос был
    адресным: страница → поле → неверное значение.
    """

    by_page: dict[int, list[SceneRefViolation]] = {}
    for violation in violations:
        by_page.setdefault(violation.page_num, []).append(violation)
    lines: list[str] = []
    for page_num in sorted(by_page):
        details = "; ".join(
            f"{violation.field} = «{violation.value}»"
            for violation in by_page[page_num]
        )
        lines.append(f"Страница {page_num}: {details}")
    return "\n".join(lines)


def _beat_page_ranges(num_pages: int) -> list[str]:
    """Разбить страницы 1..N на четыре бита арки как можно ровнее.

    Остаток от деления распределяется по первым битам, поэтому при N=6
    получаются диапазоны «1–2», «3–4», «5», «6». Для пустого бита
    возвращается «нет».
    """

    quotient, remainder = divmod(max(num_pages, 0), 4)
    sizes = [quotient + (1 if i < remainder else 0) for i in range(4)]
    ranges: list[str] = []
    start = 1
    for size in sizes:
        if size <= 0:
            ranges.append("нет")
            continue
        end = start + size - 1
        ranges.append(str(start) if start == end else f"{start}–{end}")
        start = end + 1
    return ranges


def format_arc_plan(arc: NarrativeArc, num_pages: int) -> str:
    """Собрать инструкцию о привязке диапазонов страниц к битам арки."""

    exposition, rising, climax, resolution = _beat_page_ranges(num_pages)
    return "\n".join(
        (
            f"Завязка (страницы {exposition}): {arc.exposition}",
            f"Нарастание (страницы {rising}): {arc.rising_action}",
            f"Кульминация (страницы {climax}): {arc.climax}",
            f"Развязка (страницы {resolution}): {arc.resolution}",
        )
    )
