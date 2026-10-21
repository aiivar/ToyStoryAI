"""Этап планирования сказки: вход theme/age/moral → бриф `StoryBrief`.

Тонкий узел графа: собирает промпт из `prompts.py`, делает один вызов
`LLMClient.complete_json` и кладёт результат в `state.brief`, накапливая
`StageMetrics`. Бизнес-логики и HTTP здесь нет.
"""

from __future__ import annotations

import time

from app.story_pipeline.models import StageMetrics, StoryBrief, StoryState
from app.story_pipeline.prompts import (
    PLANNER_SYSTEM,
    PLANNER_USER,
    format_style_brief,
    render_answer_format,
)
from app.story_pipeline.providers.llm import LLMClient, render_schema
from app.story_pipeline.style import get_style_profile

# Температура этапа планирования (см. spec story-text-generation).
PLANNER_TEMPERATURE = 0.7


class PlannerStage:
    """Узел графа «План»: строит бриф сказки через языковую модель."""

    name = "plan"

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def run(self, state: StoryState) -> StoryState:
        """Запросить бриф по схеме `StoryBrief` и записать его в состояние."""

        started = time.perf_counter()
        profile = get_style_profile(state.style_profile_id, state.age)
        user = PLANNER_USER.format(
            theme=state.theme,
            age=state.age,
            moral=state.moral,
            style_brief=format_style_brief(profile, state.narrator),
            schema=render_schema(StoryBrief),
            answer_format=render_answer_format(StoryBrief),
        )
        brief, usage = await self._llm.complete_json(
            system=PLANNER_SYSTEM,
            user=user,
            schema=StoryBrief,
            temperature=PLANNER_TEMPERATURE,
        )
        state.brief = brief
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


__all__ = ["PLANNER_TEMPERATURE", "PlannerStage"]
