"""Этап раскадровки: страницы + реестр сущностей → `state.scenes`.

Тонкий узел графа: собирает промпт из `prompts.py`, делает вызов
`LLMClient.complete_json` по схеме `ScenesPayload`, упорядочивает сцены по
`page_num` и накапливает `StageMetrics`. Две чистые проверки контракта
раскадровки живут в этом же модуле и тестируются напрямую:
`validate_scene_coverage` (на каждую страницу сказки — ровно одна сцена) и
`validate_scene_refs` (ссылки сцен на реестр). Каждая из них даёт не более
одного корректирующего перезапроса проблемных страниц; непокрытые страницы
после перезапроса поднимают `PipelineValidationError`, а не доезжают до
иллюстрирования, где файл страницы просто не появился бы.
"""

from __future__ import annotations

import difflib
import time

from app.story_pipeline.errors import (
    PipelineValidationError,
    StoryPipelineError,
)
from app.story_pipeline.models import (
    ExtractedEntities,
    PageScene,
    SceneRefViolation,
    ScenesPayload,
    StageMetrics,
    StoryState,
)
from app.story_pipeline.prompts import (
    DUPLICATE_PAGE_NUMBERS,
    MISSING_PAGE_NUMBERS,
    NO_PAGE_TEXT_PLACEHOLDER,
    NO_SCENES_PLACEHOLDER,
    PAGE_NUMBERS_ONE_BASED,
    SCENE_COUNT_MISMATCH,
    SCENE_COVERAGE_REPAIR_USER,
    SCENE_PAGE_NUMBERS_OUT_OF_RANGE,
    SCENE_REPAIR_USER,
    SCENE_SYSTEM,
    SCENE_USER,
    duplicate_page_numbers,
    format_entity_registry,
    format_page_numbers,
    format_pages,
    format_scene_pages,
    format_scene_violations,
)
from app.story_pipeline.providers.llm import LLMClient, Usage, render_schema
from app.story_pipeline.stages._usage import add_tokens

# Температура этапа раскадровки (см. spec story-text-generation).
SCENE_TEMPERATURE = 0.5

# Порог нечёткого сопоставления имён через `difflib`. Ниже порога ссылка
# считается нарушением и не подменяется наугад.
SCENE_REF_FUZZY_THRESHOLD = 0.8


def validate_scene_coverage(
    scenes: list[PageScene],
    num_pages: int,
) -> list[str]:
    """Проверить, что на каждую страницу сказки приходится ровно одна сцена.

    Чистая функция без LLM и I/O, по образцу `validate_page_count`. Список
    пуст, если сцен ровно `num_pages` и они пронумерованы 1..N без разрывов,
    дублей и номеров вне диапазона.

    Проверка обязательна по двум причинам. Неполное покрытие означает, что
    часть страниц не получит иллюстрации, а прогон при этом отчитается
    успешным. Дубль `page_num` означает, что на одну страницу пришло два
    разных кадра: файл `page_N.png` был бы записан дважды, и один кадр тихо
    исчез бы из результата.
    """

    numbers = [scene.page_num for scene in scenes]
    problems: list[str] = []
    if any(number < 1 for number in numbers):
        problems.append(PAGE_NUMBERS_ONE_BASED)
    if len(scenes) != num_pages:
        problems.append(
            SCENE_COUNT_MISMATCH.format(expected=num_pages, actual=len(scenes))
        )
    out_of_range = sorted(
        {number for number in numbers if number > num_pages}
    )
    if out_of_range:
        problems.append(
            SCENE_PAGE_NUMBERS_OUT_OF_RANGE.format(
                expected=num_pages, numbers=format_page_numbers(out_of_range)
            )
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


def _problem_page_numbers(
    scenes: list[PageScene],
    num_pages: int,
) -> list[int]:
    """Собрать номера страниц, по которым нужен адресный перезапрос.

    Это ровно те номера, на которые указывают проблемы
    `validate_scene_coverage`: недостающие страницы, номера вне диапазона и
    дубли. Для перезапроса важны и те, и другие: лишний номер надо назвать,
    чтобы модель его не вернула.
    """

    numbers = [scene.page_num for scene in scenes]
    problem = set(range(1, num_pages + 1)) - set(numbers)
    problem |= {number for number in numbers if number > num_pages}
    problem |= set(duplicate_page_numbers(numbers))
    return sorted(problem)


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


def _apply_coverage_repair(
    scenes: list[PageScene],
    repaired: list[PageScene],
    problem_pages: list[int],
    num_pages: int,
) -> list[PageScene]:
    """Слить сцены проблемных страниц из ответа перезапроса с исходными.

    Перезапрос адресный: сцены вне `problem_pages` игнорируются, иначе один
    лишний кадр тихо заменил бы готовую сцену. Ответ может как достроить
    недостающую страницу, так и прислать вторую сцену для уже занятого номера:
    и то и другое попадает в результат, а окончательный вердикт даёт
    `validate_scene_coverage` — молча схлопывать дубль нельзя, иначе на
    страницу попадёт кадр, выбранный без объяснения.

    Номера вне 1..N отбрасываются с обеих сторон: они не соответствуют ни
    одной странице сказки, и `page_N.png` для них не создаётся.
    """

    allowed = set(problem_pages)
    replacement: dict[int, list[PageScene]] = {}
    for scene in repaired:
        if scene.page_num not in allowed or scene.page_num > num_pages:
            continue
        replacement.setdefault(scene.page_num, []).append(scene)

    # Исходные сцены группируются по номеру страницы целиком: при дубле
    # номера перезапрос заменяет ВСЕ сцены этой страницы, иначе одна из
    # конкурирующих композиций уцелела бы и дубль не исчез бы.
    original: dict[int, list[PageScene]] = {}
    for scene in scenes:
        if scene.page_num > num_pages:
            continue
        original.setdefault(scene.page_num, []).append(scene)

    merged: list[PageScene] = []
    for page_num in sorted(set(original) | set(replacement)):
        merged.extend(replacement.get(page_num) or original.get(page_num, []))
    return sorted(merged, key=lambda scene: scene.page_num)


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

        Сначала проверяется покрытие страниц: на каждую страницу сказки должна
        приходиться ровно одна сцена. При нарушении выполняется ровно один
        корректирующий перезапрос проблемных страниц; если пробел не закрылся,
        поднимается `PipelineValidationError` — иначе часть страниц осталась
        бы без иллюстрации при статусе прогона `ok`.

        Затем ссылки сцен проверяются на реестр. При нарушениях выполняется
        ещё один корректирующий перезапрос только проблемных страниц, после
        чего проверка повторяется; остаток помечается, а не удаляется молча.
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

        coverage_problems = validate_scene_coverage(scenes, state.num_pages)
        if coverage_problems:
            scenes, usage = await self._repair_coverage(
                state, entities, scenes, coverage_problems
            )
            attempts += 1
            prompt_tokens = add_tokens(prompt_tokens, usage.prompt_tokens)
            completion_tokens = add_tokens(
                completion_tokens, usage.completion_tokens
            )

        scenes, violations = validate_scene_refs(scenes, entities)

        if violations:
            problem_set = {violation.page_num for violation in violations}
            repair_user = SCENE_REPAIR_USER.format(
                entities=format_entity_registry(entities),
                problem_pages=format_scene_violations(violations),
                problem_context=format_scene_pages(
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
            prompt_tokens = add_tokens(prompt_tokens, usage.prompt_tokens)
            completion_tokens = add_tokens(
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

    async def _repair_coverage(
        self,
        state: StoryState,
        entities: ExtractedEntities,
        scenes: list[PageScene],
        problems: list[str],
    ) -> tuple[list[PageScene], Usage]:
        """Одним перезапросом закрыть пробелы покрытия страниц.

        Возвращает слитые сцены и расход перезапроса. Если после адресного
        запроса покрытие не восстановилось, поднимается
        `PipelineValidationError` ровно так же, как это делает сценарист для
        числа страниц: сцена без иллюстрации не должна молча доезжать до
        результата со статусом `ok`.
        """

        problem_pages = _problem_page_numbers(scenes, state.num_pages)
        problem_set = set(problem_pages)
        page_texts = format_pages(
            [page for page in state.pages if page.page_num in problem_set]
        )
        scene_context = format_scene_pages(
            [scene for scene in scenes if scene.page_num in problem_set],
            [page for page in state.pages if page.page_num in problem_set],
        )
        repair_user = SCENE_COVERAGE_REPAIR_USER.format(
            problem_pages=page_texts or NO_PAGE_TEXT_PLACEHOLDER,
            problem_context=scene_context or NO_SCENES_PLACEHOLDER,
            problems="; ".join(problems),
            schema=render_schema(ScenesPayload),
        )
        payload, usage = await self._llm.complete_json(
            system=SCENE_SYSTEM,
            user=repair_user,
            schema=ScenesPayload,
            temperature=SCENE_TEMPERATURE,
        )
        merged = _apply_coverage_repair(
            scenes, payload.scenes, problem_pages, state.num_pages
        )
        remaining = validate_scene_coverage(merged, state.num_pages)
        if remaining:
            raise PipelineValidationError(
                "Раскадровщик вернул неполное покрытие страниц даже после "
                "корректирующего перезапроса: " + "; ".join(remaining)
            )
        return merged, usage


__all__ = [
    "SCENE_REF_FUZZY_THRESHOLD",
    "SCENE_TEMPERATURE",
    "SceneBreakdownStage",
    "validate_scene_coverage",
    "validate_scene_refs",
]
