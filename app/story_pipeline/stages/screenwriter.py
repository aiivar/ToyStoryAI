"""Этап написания страниц сказки: бриф + число страниц → `state.pages`.

Тонкий узел графа: собирает промпт из `prompts.py`, делает вызов
`LLMClient.complete_json`, при необходимости выполняет один корректирующий
перезапрос по числу/нумерации страниц и накапливает `StageMetrics`.
Чистая проверка `validate_page_count` отделена от LLM и тестируется напрямую.
"""

from __future__ import annotations

import time

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
from app.story_pipeline.prompts import (
    DUPLICATE_PAGE_NUMBERS,
    EXPECTED_PAGE_COUNT,
    MISSING_PAGE_NUMBERS,
    PAGE_NUMBERS_ONE_BASED,
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


class ScreenwriterStage:
    """Узел графа «Сказка»: пишет страницы по брифу через языковую модель."""

    name = "script"

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def run(self, state: StoryState) -> StoryState:
        """Запросить страницы по схеме `PagesPayload` и записать в состояние.

        При неверном числе или нумерации страниц выполняется ровно один
        корректирующий перезапрос с описанием проблем. Перезапрос собирается
        из того же `base_user`, поэтому набор иллюстрируемых страниц попадает
        в оба запроса: иначе исправленный текст потерял бы требование к сцене.
        """

        started = time.perf_counter()
        brief = state.brief
        if brief is None:
            raise StoryPipelineError(
                "Сценарист запущен без брифа: сначала должен выполниться "
                "планировщик."
            )
        base_user = SCREENWRITER_USER.format(
            logline=brief.logline,
            controlling_idea=brief.controlling_idea,
            character_arc=brief.character_arc,
            arc_plan=format_arc_plan(brief.narrative_arc, state.num_pages),
            num_pages=state.num_pages,
            age=state.age,
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

        problems = validate_page_count(payload.pages, state.num_pages)
        if problems:
            repair_user = SCREENWRITER_REPAIR_USER.format(
                problems="; ".join(problems),
                base_user=base_user,
                num_pages=state.num_pages,
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
            remaining = validate_page_count(payload.pages, state.num_pages)
            if remaining:
                raise PipelineValidationError(
                    "Сценарист вернул неверное число или нумерацию страниц "
                    "даже после корректирующего перезапроса: "
                    + "; ".join(remaining)
                )

        state.pages = sorted(payload.pages, key=lambda page: page.page_num)
        state.metrics.append(
            StageMetrics(
                stage=self.name,
                duration_ms=(time.perf_counter() - started) * 1000,
                attempts=attempts,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )
        return state


__all__ = ["SCREENWRITER_TEMPERATURE", "ScreenwriterStage"]
