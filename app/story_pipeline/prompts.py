"""Централизованные промпты пайплайна сказок.

Здесь живут все обращения к языковой модели. Промпты не размещаются в
модулях этапов или ином коде: правка формулировки локализуется в этом файле.
Остальные константы по этапам добавят последующие задачи.
"""

from __future__ import annotations

from app.story_pipeline.models import NarrativeArc

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
