"""Этап написания страниц сказки: бриф + границы объёма → `state.pages`.

Тонкий узел графа: собирает промпт из `prompts.py`, делает вызов
`LLMClient.complete_json`, при необходимости выполняет один корректирующий
перезапрос по числу/нумерации страниц и недобору объёма и накапливает
`StageMetrics`.

Проверки объёма (`validate_page_length`) и числа страниц
(`validate_page_count`) отделены от LLM и тестируются напрямую, но лечат они
неуспех по-разному: неверное число или нумерация после перезапроса роняют
прогон, а недобор объёма — нет (см. `run`).
"""

from __future__ import annotations

import time

from loguru import logger

from app.story_pipeline.errors import (
    PipelineValidationError,
    StoryPipelineError,
)
from app.story_pipeline.models import (
    PagesPayload,
    PageText,
    StageMetrics,
    StoryState,
)
from app.story_pipeline.illustrations import resolve_illustrated_pages
from app.story_pipeline.page_words import (
    count_words,
    resolve_page_words,
)
from app.story_pipeline.prompts import (
    DUPLICATE_PAGE_NUMBERS,
    EXPECTED_PAGE_COUNT,
    MISSING_PAGE_NUMBERS,
    PAGE_NUMBERS_ONE_BASED,
    PAGE_TOO_SHORT,
    SCREENWRITER_REPAIR_USER,
    SCREENWRITER_SYSTEM,
    SCREENWRITER_USER,
    duplicate_page_numbers,
    format_arc_plan,
    format_illustration_plan,
    format_page_numbers,
    render_answer_format,
)
from app.story_pipeline.providers.llm import LLMClient, render_schema
from app.story_pipeline.stages._usage import add_tokens

# Температура этапа написания сказки (см. spec story-text-generation).
SCREENWRITER_TEMPERATURE = 0.8


def validate_page_count(pages: list[PageText], num_pages: int) -> list[str]:
    """Проверить число и нумерацию страниц, вернув список проблем.

    Чистая функция без LLM и I/O: список пуст, если страниц ровно
    `num_pages` и они пронумерованы 1..N без разрывов и дублей.
    """

    problems: list[str] = []
    numbers = [page.page_num for page in pages]
    if any(page.page_num < 1 for page in pages):
        problems.append(PAGE_NUMBERS_ONE_BASED)
    if len(pages) != num_pages:
        problems.append(
            EXPECTED_PAGE_COUNT.format(expected=num_pages, actual=len(pages))
        )
    duplicates = duplicate_page_numbers(numbers)
    if duplicates:
        problems.append(
            DUPLICATE_PAGE_NUMBERS.format(
                numbers=format_page_numbers(duplicates)
            )
        )
    missing = sorted(set(range(1, num_pages + 1)) - set(numbers))
    if missing:
        problems.append(
            MISSING_PAGE_NUMBERS.format(numbers=format_page_numbers(missing))
        )
    return problems


def validate_page_length(
    pages: list[PageText],
    words_min: int,
) -> list[str]:
    """Проверить объём каждой страницы по нижней границе.

    Проверяется только минимум, и это не недосмотр: превышение максимума
    читателю не мешает, а проверка потолка заставила бы ревизора сжимать
    текст сверху — то есть работать против цели, ради которой объём поднят
    (см. D1 и D4 в design изменения `expand-story-page-text`). Верхняя граница
    нужна модели как ориентир в промпте и на этом её роль заканчивается,
    поэтому в сигнатуре её нет вовсе.

    Чистая функция без LLM и I/O, по образцу `validate_page_count`: список
    текстовых проблем уходит в тот же корректирующий перезапрос.
    """

    problems: list[str] = []
    for page in sorted(pages, key=lambda item: item.page_num):
        words = count_words(page.text)
        if words < words_min:
            problems.append(
                PAGE_TOO_SHORT.format(
                    page_num=page.page_num, words=words, expected=words_min
                )
            )
    return problems


class ScreenwriterStage:
    """Узел графа «Сказка»: пишет страницы по брифу через языковую модель."""

    name = "script"

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def run(self, state: StoryState) -> StoryState:
        """Запросить страницы по схеме `PagesPayload` и записать в состояние.

        Корректирующий перезапрос общий и ровно один — и на неверное число или
        нумерацию страниц, и на недобор объёма. Перезапрос собирается из того
        же `base_user`, поэтому набор иллюстрируемых страниц и границы объёма
        попадают в оба запроса: иначе исправленный текст потерял бы требование
        к сцене или вернулся бы к краткой форме.

        Неуспех после перезапроса лечится в зависимости от вида: неверное
        число страниц означает, что читать книжку нечем, и поднимает
        `PipelineValidationError`; недобор объёма — косметика, поэтому номера
        страниц уходят в `state.short_pages` и в `StageMetrics.error`, а прогон
        продолжается и книжка доставляется.
        """

        started = time.perf_counter()
        brief = state.brief
        if brief is None:
            raise StoryPipelineError(
                "Сценарист запущен без брифа: сначала должен выполниться "
                "планировщик."
            )
        # Границы объёма приезжают в состоянии: их разрешил оркестратор один
        # раз на прогон, и тот же диапазон потом доедет до ревизора.
        words_min, words_max = resolve_page_words(state)
        base_user = SCREENWRITER_USER.format(
            logline=brief.logline,
            controlling_idea=brief.controlling_idea,
            character_arc=brief.character_arc,
            arc_plan=format_arc_plan(brief.narrative_arc, state.num_pages),
            num_pages=state.num_pages,
            age=state.age,
            words_min=words_min,
            words_max=words_max,
            # Набор иллюстрируемых страниц переносит требование визуальной
            # сцены с «каждой страницы» на конкретные номера: визуальные
            # вершины должны оказаться там, где их действительно нарисуют.
            illustration_plan=format_illustration_plan(
                list(resolve_illustrated_pages(state)), state.num_pages
            ),
            schema=render_schema(PagesPayload),
            answer_format=render_answer_format(PagesPayload),
        )
        payload, usage = await self._llm.complete_json(
            system=SCREENWRITER_SYSTEM,
            user=base_user,
            schema=PagesPayload,
            temperature=SCREENWRITER_TEMPERATURE,
        )
        attempts = 1
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens

        # Проблемы объёма считаются отдельно от проблем структуры: последствия
        # у них разные, и сливать их в один список нельзя — иначе неуспех по
        # объёму уронил бы прогон вместе с неверным числом страниц.
        short_problems = validate_page_length(payload.pages, words_min)
        # Общий список для корректирующего перезапроса: модель должна увидеть
        # все недостатки разом, а не выяснять их по одному за платный вызов.
        problems = validate_page_count(
            payload.pages, state.num_pages
        ) + short_problems
        if short_problems:
            # Первый прогретый: по логу видно не только итог деградации, но и
            # то, что недобор был и до перезапроса.
            logger.warning(
                "screenwriter: страницы короче минимума {} слов: {}",
                words_min,
                "; ".join(short_problems),
            )
        if problems:
            repair_user = SCREENWRITER_REPAIR_USER.format(
                problems="; ".join(problems),
                base_user=base_user,
                num_pages=state.num_pages,
                words_min=words_min,
                words_max=words_max,
            )
            payload, usage = await self._llm.complete_json(
                system=SCREENWRITER_SYSTEM,
                user=repair_user,
                schema=PagesPayload,
                temperature=SCREENWRITER_TEMPERATURE,
            )
            attempts += 1
            prompt_tokens = add_tokens(prompt_tokens, usage.prompt_tokens)
            completion_tokens = add_tokens(
                completion_tokens, usage.completion_tokens
            )
            # Структурная поломка и недобор объёма разведены по последствиям.
            # Число страниц или нумерация после перезапроса — читать книжку
            # нечем, прогон падает. Недобор — косметика: номера страниц
            # уезжают в артефакт, прогон продолжается, книжка доставляется.
            remaining_count = validate_page_count(
                payload.pages, state.num_pages
            )
            if remaining_count:
                raise PipelineValidationError(
                    "Сценарист вернул неверное число или нумерацию страниц "
                    "даже после корректирующего перезапроса: "
                    + "; ".join(remaining_count)
                )
            short_problems = validate_page_length(payload.pages, words_min)

        short_pages = _short_page_nums(payload.pages, words_min)
        state.short_pages = short_pages
        if short_problems:
            # Деградация не прячется: она видна в `story.json`, в метрике
            # этапа и в логе, поэтому разбор прогона её не пропустит.
            logger.warning(
                "screenwriter: страницы {} остались короче минимума {} слов "
                "даже после корректирующего перезапроса: книжка доставляется "
                "как есть",
                short_pages,
                words_min,
            )

        state.pages = sorted(payload.pages, key=lambda page: page.page_num)
        state.metrics.append(
            StageMetrics(
                stage=self.name,
                duration_ms=(time.perf_counter() - started) * 1000,
                attempts=attempts,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                error=(
                    "страницы короче минимума "
                    f"{words_min} слов даже после корректирующего перезапроса: "
                    + "; ".join(short_problems)
                    if short_problems
                    else None
                ),
            )
        )
        return state


def _short_page_nums(pages: list[PageText], words_min: int) -> list[int]:
    """Номера страниц, не дотянувших до минимума, по возрастанию.

    В артефакт прогона уходят именно номера: по `story.json` должно быть видно,
    какие страницы читатель получит короткими, без чтения их текста.
    """

    return sorted(
        page.page_num
        for page in pages
        if count_words(page.text) < words_min
    )


__all__ = [
    "SCREENWRITER_TEMPERATURE",
    "ScreenwriterStage",
    "validate_page_count",
    "validate_page_length",
]
