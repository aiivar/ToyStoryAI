"""Этап судьи: текст и сцены страниц → вердикт `JudgeVerdict`.

Тонкий узел графа с нулевой температурой: собирает из `prompts.py` единый
взгляд на сказку (замысел брифа, целевой возраст, текст и сцены страниц) и
делает один вызов `LLMClient.complete_json`. Вердикт копится в истории
проверок прогона, метрика — в общем списке метрик. Решение о ревизии
принимает петля (`quality_loop.py`), а не судья.
"""

from __future__ import annotations

import time

from app.story_pipeline.errors import StoryPipelineError
from app.story_pipeline.models import (
    JudgeVerdict,
    PageText,
    StageMetrics,
    StoryState,
)
from app.story_pipeline.prompts import (
    JUDGE_SYSTEM,
    JUDGE_USER,
    format_arc_plan,
    format_scene_pages,
)
from app.story_pipeline.providers.llm import LLMClient, render_schema

# Температура этапа судейства: вердикт должен быть воспроизводимым.
JUDGE_TEMPERATURE = 0.0


def validate_judge_verdict(
    verdict: JudgeVerdict,
    pages: list[PageText],
) -> JudgeVerdict:
    """Нормализовать вердикт судьи до набора реально существующих страниц.

    Чистая функция без LLM и I/O. Отбрасывает номера `problem_pages` вне
    диапазона страниц прогона и схлопывает дубликаты. Если проверка не
    пройдена, а проблемных страниц не осталось, консервативно считает
    проблемными все страницы: вердикт без адреса не должен приводить к
    пустой правке.
    """

    existing = {page.page_num for page in pages}
    problem_pages = sorted(
        {number for number in verdict.problem_pages if number in existing}
    )
    if not verdict.passed and not problem_pages:
        problem_pages = sorted(existing)
    return verdict.model_copy(update={"problem_pages": problem_pages})


class JudgeStage:
    """Узел графа «Судья»: проверяет сказку по трём критериям."""

    name = "judge"

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def run(self, state: StoryState) -> StoryState:
        """Запросить вердикт по схеме `JudgeVerdict` и записать его в историю.

        Изображения судье не передаются: судейство идёт до генерации
        иллюстраций, чтобы не оплачивать отменённые кадры.
        """

        started = time.perf_counter()
        brief = state.brief
        if brief is None:
            raise StoryPipelineError(
                "Судья запущен без брифа: критерий нарративной целостности "
                "проверяется против замысла, сначала должен выполниться "
                "планировщик."
            )
        user = JUDGE_USER.format(
            logline=brief.logline,
            controlling_idea=brief.controlling_idea,
            arc_plan=format_arc_plan(brief.narrative_arc, state.num_pages),
            age=state.age,
            story=format_scene_pages(state.scenes, state.pages),
            schema=render_schema(JudgeVerdict),
        )
        verdict, usage = await self._llm.complete_json(
            system=JUDGE_SYSTEM,
            user=user,
            schema=JudgeVerdict,
            temperature=JUDGE_TEMPERATURE,
        )
        state.judge_history.append(verdict)
        state.metrics.append(
            StageMetrics(
                stage=self.name,
                duration_ms=(time.perf_counter() - started) * 1000,
                attempts=1,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
        )
        return state


__all__ = ["JUDGE_TEMPERATURE", "JudgeStage", "validate_judge_verdict"]
