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
