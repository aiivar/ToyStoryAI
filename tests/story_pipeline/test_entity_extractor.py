"""Тесты этапа извлечения сущностей.

Этап проверяется через публичный интерфейс `async def run(state)` с фейковым
`LLMClient`: смотрим, что этап запросил у модели и что записал в состояние.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.story_pipeline.models import (
    Character,
    ExtractedEntities,
    KeyObject,
    Location,
    PageText,
    StoryState,
)
from app.story_pipeline.stages.entity_extractor import EntityExtractorStage


def _state() -> StoryState:
    """Состояние с готовыми страницами сказки."""

    return StoryState(
        theme="Приключение плюшевого мишки",
        age=6,
        moral="Смелость помогает найти друзей",
        num_pages=2,
        pages=[
            PageText(page_num=1, text="Мишка вышел из шкафа в детской."),
            PageText(page_num=2, text="В лесу мишка нашёл зайца."),
        ],
        started_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    )


# Описание игрушки-героя: в этом прогоне оно едет в `theme` (D2), отдельное
# поле состояния появится в задаче 4.3.
TOY_DESCRIPTION = (
    "Плюшевый медвежонок: коричневый, мягкий плюш, заплатка на левом ухе."
)


def _registry() -> ExtractedEntities:
    """Реестр сущностей, который «вернёт» модель."""

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
            Location(name="Детская", description="светлая комната с деревянной кроватью"),
            Location(name="Лес", description="сосновый лес ранним утром"),
        ],
        key_objects=[
            KeyObject(name="Корзинка", description="плетёная корзинка с грибами"),
        ],
        art_style="soft watercolor storybook illustration",
    )


async def test_entity_extractor_stores_registry_and_metric(fake_llm) -> None:
    """Реестр из ответа модели кладётся в `state.entities`, метрика добавлена."""

    registry = _registry()
    llm = fake_llm([registry])
    stage = EntityExtractorStage(llm)
    state = _state()

    result = await stage.run(state)

    assert result is state
    assert state.entities == registry
    assert llm.call_count == 1
    assert llm.calls[0]["schema"] is ExtractedEntities
    assert llm.calls[0]["temperature"] == 0.3
    metric = state.metrics[-1]
    assert metric.stage == "extract"
    assert metric.attempts == 1


async def test_entity_extractor_prompt_requests_verifiable_anchors(
    fake_llm,
) -> None:
    """Промпт несёт все страницы, требование единого стиля и описаний-якорей."""

    llm = fake_llm([_registry()])
    stage = EntityExtractorStage(llm)
    state = _state()

    await stage.run(state)

    user = llm.calls[0]["user"]
    # Обе страницы переданы дословно.
    assert "Страница 1: Мишка вышел из шкафа в детской." in user
    assert "Страница 2: В лесу мишка нашёл зайца." in user
    # Стиль выбирается один раз на всю сказку.
    assert "ВСЕЙ сказки" in user
    assert "один раз" in user
    # Описания конкретные и визуальные, пригодные для дословного повтора.
    assert "дословно" in user
    assert "цвет" in user
    assert "форма" in user
    assert "материал" in user
    # Схема берётся из модели, а не задаётся руками.
    assert "$defs" in user


async def test_entity_extractor_prompt_uses_toy_description_field(
    fake_llm,
) -> None:
    """Экстрактор берёт описание из `state.toy_description`, а не из `theme`.

    `theme` — формулировка задачи для планировщика (D18); описание едет
    отдельным полем, иначе экстрактор получил бы воду вместо облика героя.
    """

    llm = fake_llm([_registry()])
    stage = EntityExtractorStage(llm)
    state = _state()
    state.toy_description = TOY_DESCRIPTION

    await stage.run(state)

    user = llm.calls[0]["user"]
    assert TOY_DESCRIPTION in user
    # Короткая сводка планировщика в промпт экстрактора не подставляется.
    assert state.theme not in user


async def test_entity_extractor_prompt_without_toy_description_is_clean(
    fake_llm,
) -> None:
    """Пустое описание не роняет промпт: блок остаётся внятным заглушкой."""

    llm = fake_llm([_registry()])
    stage = EntityExtractorStage(llm)
    state = _state()
    state.toy_description = None

    await stage.run(state)

    user = llm.calls[0]["user"]
    assert "Описание игрушки-героя" in user
    # Ни «None», ни дырок от пустого блока в промпте нет.
    assert "None" not in user
    assert ". ." not in user
    assert "\n\n\n" not in user


async def test_entity_extractor_prompt_forbids_inventing_appearance(
    fake_llm,
) -> None:
    """Промпт экстрактора получает описание героя и запрет выдумывать вид.

    Описание игрушки — единственный источник правды об облике: если
    экстрактор додумает цвет или наряд, иллюстрации разойдутся с фото, по
    которому родитель их узнаёт.
    """

    llm = fake_llm([_registry()])
    stage = EntityExtractorStage(llm)
    state = _state()
    state.toy_description = TOY_DESCRIPTION

    await stage.run(state)

    user = llm.calls[0]["user"]
    # Описание доходит до экстрактора дословно.
    assert TOY_DESCRIPTION in user
    # Запрет выдумывать внешность назван явно, с полями описания.
    assert "не выдумывай" in user
    for field in ("appearance", "clothing", "distinguishing_features"):
        assert field in user
    # Наряд и цвет берутся из описания, а не из воли модели.
    assert "наряд" in user
    assert "цвет" in user
