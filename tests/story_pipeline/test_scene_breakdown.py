"""Тесты этапа раскадровки и чистой проверки ссылок на реестр.

Этап проверяется через публичный интерфейс `async def run(state)` с фейковым
`LLMClient`; `validate_scene_refs` — чистая функция, тестируется напрямую.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.story_pipeline.errors import StoryPipelineError
from app.story_pipeline.models import (
    Character,
    ExtractedEntities,
    KeyObject,
    Location,
    PageScene,
    PageText,
    SceneRefViolation,
    ScenesPayload,
    StoryState,
)
from app.story_pipeline.stages.scene_breakdown import (
    SceneBreakdownStage,
    validate_scene_refs,
)


def _entities() -> ExtractedEntities:
    """Небольшой реестр сущностей для проверки ссылок сцен."""

    return ExtractedEntities(
        characters=[
            Character(
                name="Мишка",
                appearance="коричневый плюшевый медведь",
                clothing="красный вязаный шарф",
                distinguishing_features="заплатка на левом ухе",
            ),
            Character(
                name="Заяц",
                appearance="серый пушистый заяц",
                clothing="синие штанишки",
                distinguishing_features="длинные уши с белыми кончиками",
            ),
        ],
        locations=[
            Location(name="Детская", description="светлая комната"),
            Location(name="Лес", description="сосновый лес утром"),
        ],
        key_objects=[
            KeyObject(name="Корзинка", description="плетёная корзинка с грибами"),
        ],
        art_style="soft watercolor storybook illustration",
    )


def _state() -> StoryState:
    """Состояние с готовыми страницами и реестром сущностей."""

    return StoryState(
        theme="Приключение плюшевого мишки",
        age=6,
        moral="Смелость помогает найти друзей",
        num_pages=2,
        pages=[
            PageText(page_num=1, text="Мишка вышел из шкафа в детской."),
            PageText(page_num=2, text="В лесу мишка нашёл зайца."),
        ],
        entities=_entities(),
        started_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )


def _scene(page_num: int) -> PageScene:
    """Корректная сцена страницы со ссылками строго на реестр."""

    return PageScene(
        page_num=page_num,
        scene_description=f"Мишка в кадре на странице {page_num}.",
        characters_in_scene=["Мишка"],
        location="Лес",
        objects_in_scene=[],
    )


def _scenes(nums: list[int]) -> ScenesPayload:
    """Ответ раскадровщика: сцены с заданными номерами страниц."""

    return ScenesPayload(scenes=[_scene(n) for n in nums])


async def test_scene_breakdown_stores_scenes_sorted_by_page(fake_llm) -> None:
    """Сцены из ответа кладутся в `state.scenes`, упорядоченные по `page_num`."""

    llm = fake_llm([_scenes([2, 1])])
    stage = SceneBreakdownStage(llm)
    state = _state()

    result = await stage.run(state)

    assert result is state
    assert [scene.page_num for scene in state.scenes] == [1, 2]
    assert state.scenes[0].scene_description == "Мишка в кадре на странице 1."
    assert llm.call_count == 1
    assert llm.calls[0]["schema"] is ScenesPayload
    assert llm.calls[0]["temperature"] == 0.5
    assert state.metrics[-1].stage == "scene"
    assert state.unresolved_pages == []


async def test_scene_breakdown_prompt_demands_exact_registry_names(
    fake_llm,
) -> None:
    """Промпт требует по сцене на страницу, точных имён реестра и пустых кадров."""

    llm = fake_llm([_scenes([1, 2])])
    stage = SceneBreakdownStage(llm)
    state = _state()

    await stage.run(state)

    user = llm.calls[0]["user"]
    # Ровно одна сцена на страницу, номер совпадает со страницей.
    assert "ровно одна сцена на каждую страницу" in user
    assert "page_num" in user
    # Описание кадра — действие и композиция.
    assert "композиц" in user
    # Ссылки — точные имена реестра, пустой список допустим.
    assert "ТОЧНЫЕ имена" in user
    assert "пустой список" in user
    # Реестр и страницы переданы в промпт.
    assert "Мишка" in user
    assert "Заяц" in user
    assert "Лес" in user
    assert "Страница 1:" in user
    # Схема берётся из модели.
    assert "$defs" in user


async def test_scene_breakdown_requires_entities(fake_llm) -> None:
    """Без реестра сущностей этап падает типизированно, не обращаясь к LLM."""

    llm = fake_llm([])
    stage = SceneBreakdownStage(llm)
    state = _state()
    state.entities = None

    with pytest.raises(StoryPipelineError):
        await stage.run(state)

    assert llm.call_count == 0


# --- Чистая функция validate_scene_refs ---------------------------------


def test_validate_scene_refs_accepts_exact_registry_names() -> None:
    """Точные имена реестра проходят проверку без нарушений."""

    scenes = [_scene(1), _scene(2)]

    normalized, violations = validate_scene_refs(scenes, _entities())

    assert violations == []
    assert [scene.page_num for scene in normalized] == [1, 2]
    assert normalized == scenes


def test_validate_scene_refs_normalizes_case_insensitively() -> None:
    """Ссылки в другом регистре нормализуются к именам реестра."""

    scene = PageScene(
        page_num=1,
        scene_description="Мишка и заяц у леса.",
        characters_in_scene=["мишка", "ЗАЯЦ"],
        location="лес",
        objects_in_scene=["корзинка"],
    )

    normalized, violations = validate_scene_refs([scene], _entities())

    assert violations == []
    assert normalized[0].characters_in_scene == ["Мишка", "Заяц"]
    assert normalized[0].location == "Лес"
    assert normalized[0].objects_in_scene == ["Корзинка"]


def test_validate_scene_refs_normalizes_close_typos() -> None:
    """Близкие по написанию ссылки нормализуются к ближайшему имени реестра."""

    scene = PageScene(
        page_num=1,
        scene_description="Мишк у леса с корзинкой.",
        characters_in_scene=["Мишк"],
        location="Лесс",
        objects_in_scene=["Корзинко"],
    )

    normalized, violations = validate_scene_refs([scene], _entities())

    assert violations == []
    assert normalized[0].characters_in_scene == ["Мишка"]
    assert normalized[0].location == "Лес"
    assert normalized[0].objects_in_scene == ["Корзинка"]


def test_validate_scene_refs_reports_unknown_names_without_dropping() -> None:
    """Неизвестные имена фиксируются нарушением и остаются в сцене."""

    scene = PageScene(
        page_num=2,
        scene_description="Дракон прилетел к лесу.",
        characters_in_scene=["Мишка", "Дракон"],
        location="Лес",
        objects_in_scene=["Волшебная палочка"],
    )

    normalized, violations = validate_scene_refs([scene], _entities())

    # Имя не удаляется молча — оно остаётся в сцене.
    assert normalized[0].characters_in_scene == ["Мишка", "Дракон"]
    assert normalized[0].objects_in_scene == ["Волшебная палочка"]
    assert normalized[0].location == "Лес"
    found = {
        (violation.page_num, violation.field, violation.value)
        for violation in violations
    }
    assert found == {
        (2, "characters_in_scene", "Дракон"),
        (2, "objects_in_scene", "Волшебная палочка"),
    }


# --- Корректирующий перезапрос раскадровки ------------------------------


async def test_scene_breakdown_repairs_problem_pages_once(fake_llm) -> None:
    """Неизвестные имена вызывают ровно один перезапрос проблемных страниц."""

    bad_page = PageScene(
        page_num=1,
        scene_description="Дракон в детской.",
        characters_in_scene=["Дракон"],
        location="Детская",
        objects_in_scene=[],
    )
    fixed_page = PageScene(
        page_num=1,
        scene_description="Мишка в детской.",
        characters_in_scene=["Мишка"],
        location="Детская",
        objects_in_scene=[],
    )
    llm = fake_llm(
        [
            ScenesPayload(scenes=[bad_page, _scene(2)]),
            ScenesPayload(scenes=[fixed_page]),
        ]
    )
    stage = SceneBreakdownStage(llm)
    state = _state()

    await stage.run(state)

    assert llm.call_count == 2
    repair_user = llm.calls[1]["user"]
    # Перезапросу передан список допустимых имён и проблемная страница.
    assert "Мишка" in repair_user
    assert "Заяц" in repair_user
    assert "Дракон" in repair_user
    assert "Страница 1:" in repair_user
    # Исправленная сцена записана, непроблемная страница не изменилась.
    assert [scene.page_num for scene in state.scenes] == [1, 2]
    assert state.scenes[0].characters_in_scene == ["Мишка"]
    assert state.scenes[1].scene_description == "Мишка в кадре на странице 2."
    assert state.unresolved_pages == []
    metric = state.metrics[-1]
    assert metric.stage == "scene"
    assert metric.attempts == 2
    assert metric.prompt_tokens == 22
    assert metric.completion_tokens == 44


async def test_scene_breakdown_repair_prompt_includes_current_page_context(
    fake_llm,
) -> None:
    """Перезапросу передан контекст проблемной страницы: текущая сцена и текст."""

    bad_page = PageScene(
        page_num=1,
        scene_description="Мишка стоит у окна детской.",
        characters_in_scene=["Дракон"],
        location="Детская",
        objects_in_scene=[],
    )
    fixed_page = PageScene(
        page_num=1,
        scene_description="Мишка у окна.",
        characters_in_scene=["Мишка"],
        location="Детская",
        objects_in_scene=[],
    )
    llm = fake_llm(
        [
            ScenesPayload(scenes=[bad_page]),
            ScenesPayload(scenes=[fixed_page]),
        ]
    )
    stage = SceneBreakdownStage(llm)
    state = _state()

    await stage.run(state)

    repair_user = llm.calls[1]["user"]
    # Текущее описание кадра — чтобы модель видела, что именно правит.
    assert "Мишка стоит у окна детской." in repair_user
    # Текст проблемной страницы — контекст действия.
    assert "Мишка вышел из шкафа в детской." in repair_user


async def test_scene_breakdown_repair_keeps_original_scene_description(
    fake_llm,
) -> None:
    """Ремонт правит ссылки, но не подменяет композицию кадра."""

    bad_page = PageScene(
        page_num=1,
        scene_description="Мишка стоит у окна детской.",
        characters_in_scene=["Дракон"],
        location="Детская",
        objects_in_scene=[],
    )
    rewritten_page = PageScene(
        page_num=1,
        scene_description="Совсем другой кадр про зайца в лесу.",
        characters_in_scene=["Мишка"],
        location="Детская",
        objects_in_scene=[],
    )
    llm = fake_llm(
        [
            ScenesPayload(scenes=[bad_page]),
            ScenesPayload(scenes=[rewritten_page]),
        ]
    )
    stage = SceneBreakdownStage(llm)
    state = _state()

    await stage.run(state)

    # Ссылка исправлена, а исходное описание кадра сохранено.
    assert state.scenes[0].characters_in_scene == ["Мишка"]
    assert state.scenes[0].scene_description == "Мишка стоит у окна детской."


async def test_scene_breakdown_marks_unresolved_pages_without_failing(
    fake_llm,
) -> None:
    """Остаток неизвестных имён не роняет этап, а помечает страницы в состоянии."""

    bad_page = PageScene(
        page_num=1,
        scene_description="Дракон в детской.",
        characters_in_scene=["Дракон"],
        location="Детская",
        objects_in_scene=[],
    )
    llm = fake_llm(
        [
            ScenesPayload(scenes=[bad_page, _scene(2)]),
            ScenesPayload(scenes=[bad_page]),
        ]
    )
    stage = SceneBreakdownStage(llm)
    state = _state()

    # Этап завершается без исключения.
    result = await stage.run(state)

    assert result is state
    assert llm.call_count == 2
    assert state.unresolved_pages == [1]
    # Проблемное имя сохранено в сцене, а не удалено молча.
    assert state.scenes[0].characters_in_scene == ["Дракон"]
