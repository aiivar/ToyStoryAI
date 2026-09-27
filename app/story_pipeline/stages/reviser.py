"""Этап ревизора: замечания судьи → правки только проблемных страниц.

Тонкий узел графа: берёт из состояния последний вердикт судьи и известные
проблемные страницы (`unresolved_pages` после раскадровки), собирает
адресный промпт из `prompts.py` и делает один вызов
`LLMClient.complete_json`. Правки применяются строго к проблемным страницам;
после этого ссылки сцен на реестр проверяются заново
(`validate_scene_refs`), поэтому остаток неизвестных имён снова попадает в
`state.unresolved_pages`. Решение о числе ревизий принимает петля
(`quality_loop.py`).
"""

from __future__ import annotations

import time

from app.story_pipeline.errors import (
    PipelineValidationError,
    StoryPipelineError,
)
from app.story_pipeline.models import (
    PageScene,
    PageText,
    RevisionPayload,
    StageMetrics,
    StoryState,
)
from app.story_pipeline.prompts import (
    REVISER_SYSTEM,
    REVISER_USER,
    format_issues,
    format_scene_pages,
)
from app.story_pipeline.providers.llm import LLMClient, render_schema
from app.story_pipeline.stages.judge import validate_judge_verdict
from app.story_pipeline.stages.scene_breakdown import validate_scene_refs

# Температура этапа ревизии: нужны вариативные, но связные правки.
REVISER_TEMPERATURE = 0.7


def _problem_pages(state: StoryState) -> list[int]:
    """Собрать адресный набор страниц для правки.

    Берётся последний вердикт судьи (нормализованный) плюс известные
    проблемные страницы из `unresolved_pages`, полученные раскадровкой.
    Судья мог не увидеть проблему ссылок, поэтому объединение обязательно.
    """

    if not state.judge_history:
        raise StoryPipelineError(
            "Ревизор запущен без вердикта судьи: сначала должна выполниться "
            "проверка."
        )
    verdict = validate_judge_verdict(state.judge_history[-1], state.pages)
    return sorted(set(verdict.problem_pages) | set(state.unresolved_pages))


class ReviserStage:
    """Узел графа «Ревизор»: правит только проблемные страницы."""

    name = "reviser"

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def run(self, state: StoryState) -> StoryState:
        """Запросить правки по схеме `RevisionPayload` и применить их."""

        started = time.perf_counter()
        brief = state.brief
        if brief is None:
            raise StoryPipelineError(
                "Ревизор запущен без брифа: правки адресуются против замысла, "
                "сначала должен выполниться планировщик."
            )
        entities = state.entities
        if entities is None:
            raise StoryPipelineError(
                "Ревизор запущен без реестра сущностей: после правки ссылки "
                "сцен проверяются заново, сначала должен выполниться "
                "экстрактор."
            )
        problem_pages = _problem_pages(state)
        if not problem_pages:
            raise StoryPipelineError(
                "Ревизор вызван без проблемных страниц: правка адресуется "
                "только по вердикту судьи или unresolved_pages."
            )

        problem_set = set(problem_pages)
        verdict = state.judge_history[-1]
        user = REVISER_USER.format(
            logline=brief.logline,
            controlling_idea=brief.controlling_idea,
            character_arc=brief.character_arc,
            issues=format_issues(verdict.issues),
            problem_context=format_scene_pages(
                [
                    scene
                    for scene in state.scenes
                    if scene.page_num in problem_set
                ],
                [page for page in state.pages if page.page_num in problem_set],
            ),
            schema=render_schema(RevisionPayload),
        )
        payload, usage = await self._llm.complete_json(
            system=REVISER_SYSTEM,
            user=user,
            schema=RevisionPayload,
            temperature=REVISER_TEMPERATURE,
        )

        state.pages = _apply_page_revisions(
            state.pages, payload, problem_set
        )
        state.scenes = _apply_scene_revisions(
            state.scenes, payload, problem_set
        )
        normalized_scenes, violations = validate_scene_refs(
            state.scenes, entities
        )
        state.scenes = sorted(
            normalized_scenes, key=lambda scene: scene.page_num
        )
        state.unresolved_pages = sorted(
            {violation.page_num for violation in violations}
        )
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


def _apply_page_revisions(
    pages: list[PageText],
    payload: RevisionPayload,
    problem_set: set[int],
) -> list[PageText]:
    """Заменить текст только проблемных страниц, проверив пересборку сцены.

    Если модель изменила текст страницы, но не прислала её сцену, контракт
    ревизии нарушен: сцена осталась бы от прежнего действия. Это не лечится
    разбором JSON, поэтому поднимается `PipelineValidationError`.
    """

    revised_pages = {
        page.page_num: page
        for page in payload.pages
        if page.page_num in problem_set
    }
    revised_scene_numbers = {
        scene.page_num
        for scene in payload.scenes
        if scene.page_num in problem_set
    }
    updated: list[PageText] = []
    for page in pages:
        replacement = revised_pages.get(page.page_num)
        if replacement is None:
            updated.append(page)
            continue
        if (
            replacement.text != page.text
            and page.page_num not in revised_scene_numbers
        ):
            raise PipelineValidationError(
                "Ревизор изменил текст страницы "
                f"{page.page_num}, но не пересобрал её сцену."
            )
        updated.append(replacement)
    return sorted(updated, key=lambda page: page.page_num)


def _apply_scene_revisions(
    scenes: list[PageScene],
    payload: RevisionPayload,
    problem_set: set[int],
) -> list[PageScene]:
    """Заменить сцены только проблемных страниц, сохранив остальные."""

    revised_scenes = {
        scene.page_num: scene
        for scene in payload.scenes
        if scene.page_num in problem_set
    }
    return [
        revised_scenes.get(scene.page_num, scene) for scene in scenes
    ]


__all__ = ["REVISER_TEMPERATURE", "ReviserStage"]
