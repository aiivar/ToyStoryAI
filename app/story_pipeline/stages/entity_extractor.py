"""Этап извлечения реестра сущностей: страницы сказки → `ExtractedEntities`.

Тонкий узел графа: собирает промпт из `prompts.py`, делает один вызов
`LLMClient.complete_json` и кладёт результат в `state.entities`, накапливая
`StageMetrics`. Реестр — якорь согласованности иллюстраций, поэтому промпт
передаёт модели описание игрушки-героя и прямо запрещает выдумывать внешность:
иначе описания разойдутся с фотографией, по которой родитель узнаёт игрушку.
"""

from __future__ import annotations

import time

from app.story_pipeline.models import ExtractedEntities, StageMetrics, StoryState
from app.story_pipeline.prompts import (
    EXTRACTOR_SYSTEM,
    EXTRACTOR_USER,
    format_pages,
    format_toy_description,
)
from app.story_pipeline.providers.llm import LLMClient, render_schema

# Температура этапа извлечения сущностей (см. spec story-text-generation).
EXTRACTOR_TEMPERATURE = 0.3


class EntityExtractorStage:
    """Узел графа «Извлечение»: строит реестр сущностей через языковую модель."""

    name = "extract"

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def run(self, state: StoryState) -> StoryState:
        """Запросить реестр по схеме `ExtractedEntities` и записать в состояние."""

        started = time.perf_counter()
        user = EXTRACTOR_USER.format(
            pages=format_pages(state.pages),
            toy_description=format_toy_description(state.toy_description),
            schema=render_schema(ExtractedEntities),
        )
        entities, usage = await self._llm.complete_json(
            system=EXTRACTOR_SYSTEM,
            user=user,
            schema=ExtractedEntities,
            temperature=EXTRACTOR_TEMPERATURE,
        )
        state.entities = entities
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


__all__ = ["EXTRACTOR_TEMPERATURE", "EntityExtractorStage"]
