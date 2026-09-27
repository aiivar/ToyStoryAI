"""Этап раскадровки: страницы + реестр сущностей → `state.scenes`.

Тонкий узел графа: собирает промпт из `prompts.py`, делает вызов
`LLMClient.complete_json` по схеме `ScenesPayload`, упорядочивает сцены по
`page_num` и накапливает `StageMetrics`. Чистая проверка ссылок на реестр
(`validate_scene_refs`) и корректирующий перезапрос проблемных страниц живут
в этом же модуле: они относятся к контракту раскадровки.
"""

from __future__ import annotations

import difflib
import time

from app.story_pipeline.errors import StoryPipelineError
from app.story_pipeline.models import (
    ExtractedEntities,
    PageScene,
    SceneRefViolation,
    ScenesPayload,
    StageMetrics,
    StoryState,
)
from app.story_pipeline.prompts import (
    SCENE_REPAIR_USER,
    SCENE_SYSTEM,
    SCENE_USER,
    format_entity_registry,
    format_pages,
    format_problem_context,
    format_scene_violations,
)
from app.story_pipeline.providers.llm import LLMClient, render_schema

# Температура этапа раскадровки (см. spec story-text-generation).
SCENE_TEMPERATURE = 0.5

# Порог нечёткого сопоставления имён через `difflib`. Ниже порога ссылка
# считается нарушением и не подменяется наугад.
SCENE_REF_FUZZY_THRESHOLD = 0.8


def _add_tokens(total: int | None, value: int | None) -> int | None:
    """Сложить токены двух попыток, сохраняя `None` как «неизвестно»."""

    if value is None:
        return total
    return (total or 0) + value


def _apply_repair(
    scene: PageScene,
    repaired: dict[int, PageScene],
) -> PageScene:
    """Заменить ссылки сцены из ответа ремонта, сохранив описание кадра.

    Ремонт адресно правит только имена из реестра, поэтому `page_num` и
    `scene_description` берутся из исходной сцены: композиция кадра не
    переписывается вслепую.
    """

    replacement = repaired.get(scene.page_num)
    if replacement is None:
        return scene
    return scene.model_copy(
        update={
            "characters_in_scene": replacement.characters_in_scene,
            "location": replacement.location,
            "objects_in_scene": replacement.objects_in_scene,
        }
    )


def _match_name(value: str, candidates: list[str]) -> str | None:
    """Сопоставить ссылку с именем реестра, вернув каноничное имя или `None`.

    Порядок: точное совпадение → совпадение без учёта регистра → нечёткое
    совпадение не ниже `SCENE_REF_FUZZY_THRESHOLD` (ближайшее имя).
    """

    stripped = value.strip()
    if stripped in candidates:
        return stripped
    folded = stripped.casefold()
    for candidate in candidates:
        if candidate.casefold() == folded:
            return candidate
    best_ratio = SCENE_REF_FUZZY_THRESHOLD
    best_candidate: str | None = None
    for candidate in candidates:
        ratio = difflib.SequenceMatcher(
            None, folded, candidate.casefold()
        ).ratio()
        if ratio >= best_ratio:
            best_ratio = ratio
            best_candidate = candidate
    return best_candidate


def _normalize_names(
    values: list[str],
    candidates: list[str],
    *,
    page_num: int,
    field: str,
    violations: list[SceneRefViolation],
) -> list[str]:
    """Нормализовать список ссылок, дополнив `violations` несопоставленными."""

    normalized: list[str] = []
    for value in values:
        matched = _match_name(value, candidates)
        if matched is None:
            violations.append(
                SceneRefViolation(page_num=page_num, field=field, value=value)
            )
            normalized.append(value)
        else:
            normalized.append(matched)
    return normalized


def validate_scene_refs(
    scenes: list[PageScene],
    entities: ExtractedEntities,
) -> tuple[list[PageScene], list[SceneRefViolation]]:
    """Проверить ссылки сцен на реестр и нормализовать их.

    Чистая функция без LLM и I/O. Возвращает новые сцены (вход не
    мутируется) и список нарушений для ссылок, которые не удалось
    сопоставить. Ссылки, которые удалось нормализовать, заменяются
    каноничными именами реестра; несопоставленные значения сохраняются в
    сценах, чтобы имя не терялось молча.
    """

    character_names = [character.name for character in entities.characters]
    location_names = [location.name for location in entities.locations]
    object_names = [item.name for item in entities.key_objects]

    normalized: list[PageScene] = []
    violations: list[SceneRefViolation] = []
    for scene in scenes:
        location = _match_name(scene.location, location_names)
        if location is None:
            violations.append(
                SceneRefViolation(
                    page_num=scene.page_num,
                    field="location",
                    value=scene.location,
                )
            )
            location = scene.location
        normalized.append(
            scene.model_copy(
                update={
                    "characters_in_scene": _normalize_names(
                        scene.characters_in_scene,
                        character_names,
                        page_num=scene.page_num,
                        field="characters_in_scene",
                        violations=violations,
                    ),
                    "location": location,
                    "objects_in_scene": _normalize_names(
                        scene.objects_in_scene,
                        object_names,
                        page_num=scene.page_num,
                        field="objects_in_scene",
                        violations=violations,
                    ),
                },
                deep=True,
            )
        )
    return normalized, violations


class SceneBreakdownStage:
    """Узел графа «Раскадровка»: раскладывает страницы на кадры."""

    name = "scene"

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def run(self, state: StoryState) -> StoryState:
        """Запросить сцены по схеме `ScenesPayload` и записать в состояние.

        После первичной раскадровки ссылки проверяются на реестр. При
        нарушениях выполняется ровно один корректирующий перезапрос только
        проблемных страниц, после чего проверка повторяется.
        """

        started = time.perf_counter()
        entities = state.entities
        if entities is None:
            raise StoryPipelineError(
                "Раскадровщик запущен без реестра сущностей: сначала должен "
                "выполниться экстрактор."
            )
        user = SCENE_USER.format(
            entities=format_entity_registry(entities),
            pages=format_pages(state.pages),
            schema=render_schema(ScenesPayload),
        )
        payload, usage = await self._llm.complete_json(
            system=SCENE_SYSTEM,
            user=user,
            schema=ScenesPayload,
            temperature=SCENE_TEMPERATURE,
        )
        attempts = 1
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens

        scenes = sorted(payload.scenes, key=lambda scene: scene.page_num)
        scenes, violations = validate_scene_refs(scenes, entities)

        if violations:
            problem_set = {violation.page_num for violation in violations}
            repair_user = SCENE_REPAIR_USER.format(
                entities=format_entity_registry(entities),
                problem_pages=format_scene_violations(violations),
                problem_context=format_problem_context(
                    [
                        scene
                        for scene in scenes
                        if scene.page_num in problem_set
                    ],
                    [
                        page
                        for page in state.pages
                        if page.page_num in problem_set
                    ],
                ),
                schema=render_schema(ScenesPayload),
            )
            repair_payload, usage = await self._llm.complete_json(
                system=SCENE_SYSTEM,
                user=repair_user,
                schema=ScenesPayload,
                temperature=SCENE_TEMPERATURE,
            )
            attempts += 1
            prompt_tokens = _add_tokens(prompt_tokens, usage.prompt_tokens)
            completion_tokens = _add_tokens(
                completion_tokens, usage.completion_tokens
            )
            repaired = {
                scene.page_num: scene
                for scene in repair_payload.scenes
                if scene.page_num in problem_set
            }
            # Берём из ответа только ссылки: описание кадра сохраняется
            # исходным, чтобы ремонт имён не подменял композицию.
            scenes = [_apply_repair(scene, repaired) for scene in scenes]
            scenes, violations = validate_scene_refs(scenes, entities)

        state.scenes = scenes
        state.unresolved_pages = sorted(
            {violation.page_num for violation in violations}
        )
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


__all__ = [
    "SCENE_REF_FUZZY_THRESHOLD",
    "SCENE_TEMPERATURE",
    "SceneBreakdownStage",
    "validate_scene_refs",
]
